// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "scene.h"
#include "scene_util.h"
#include "app.h"

#define HANDMADE_MATH_IMPLEMENTATION
#include "HandmadeMath.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

// Erosion scene - hydraulic erosion simulation with water flow
typedef struct {
	scene_t       base;

	skr_mesh_t     terrain_mesh;
	skr_shader_t   terrain_shader;
	skr_material_t terrain_material;

	// Compute shaders for water simulation
	skr_shader_t   flow_shader;
	skr_shader_t   water_shader;
	skr_compute_t  compute_flow;
	skr_compute_t  compute_water;

	// Terrain textures (ping-pong for height modification)
	skr_tex_t      height_tex_ping;
	skr_tex_t      height_tex_pong;
	skr_tex_t      water_tex_ping;
	skr_tex_t      water_tex_pong;
	skr_tex_t      sediment_tex_ping;
	skr_tex_t      sediment_tex_pong;
	skr_buffer_t   flow_buffer;
	skr_tex_t      debug_tex;

	// Compute parameters
	skr_buffer_t   compute_params_buffer;

	int32_t terrain_size;
	float   terrain_scale;
	float   terrain_height_scale;
	int32_t compute_iteration;
	float   rotation;
	float   rainfall_timer;
} scene_erosion_t;

// Simple Perlin noise implementation
static float _fade(float t) {
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static float _lerp(float t, float a, float b) {
	return a + t * (b - a);
}

static float _grad(int hash, float x, float y) {
	int h = hash & 15;
	float u = h < 8 ? x : y;
	float v = h < 4 ? y : x;
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// Permutation table for Perlin noise
static int _perm[512];
static void _init_perlin() {
	static int p[256] = {
		151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
		8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
		35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
		134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
		55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
		18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
		250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
		189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
		172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
		228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
		107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
	};
	for (int i = 0; i < 256; i++) {
		_perm[i] = p[i];
		_perm[256 + i] = p[i];
	}
}

static float _perlin(float x, float y) {
	int X = (int)floorf(x) & 255;
	int Y = (int)floorf(y) & 255;

	x -= floorf(x);
	y -= floorf(y);

	float u = _fade(x);
	float v = _fade(y);

	int aa = _perm[_perm[X] + Y];
	int ab = _perm[_perm[X] + Y + 1];
	int ba = _perm[_perm[X + 1] + Y];
	int bb = _perm[_perm[X + 1] + Y + 1];

	return _lerp(v,
		_lerp(u, _grad(aa, x, y), _grad(ba, x - 1, y)),
		_lerp(u, _grad(ab, x, y - 1), _grad(bb, x - 1, y - 1))
	);
}

static float _octave_perlin(float x, float y, int octaves, float persistence) {
	float total = 0.0f;
	float frequency = 1.0f;
	float amplitude = 1.0f;
	float max_value = 0.0f;

	for (int i = 0; i < octaves; i++) {
		total += _perlin(x * frequency, y * frequency) * amplitude;
		max_value += amplitude;
		amplitude *= persistence;
		frequency *= 2.0f;
	}

	return total / max_value;
}

static scene_t* _scene_erosion_create() {
	scene_erosion_t* scene = calloc(1, sizeof(scene_erosion_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_erosion_t);
	scene->terrain_size = 256;
	scene->terrain_scale = 64.0f;
	scene->terrain_height_scale = 16.0f;
	scene->compute_iteration = 0;
	scene->rotation = 0.0f;
	scene->rainfall_timer = 0.0f;

	_init_perlin();

	// Create terrain mesh - 1x0x1 grid (flat on XZ plane) with 256x256 resolution
	const int   grid_size    = scene->terrain_size;
	const int   vertex_count = (grid_size + 1) * (grid_size + 1);
	const int   index_count  = grid_size * grid_size * 6;

	su_vertex_pnuc_t* terrain_vertices = malloc(vertex_count * sizeof(su_vertex_pnuc_t));
	uint32_t*         terrain_indices  = malloc(index_count  * sizeof(uint32_t));

	// Generate flat terrain vertices (height will come from shader sampling the texture)
	for (int z = 0; z <= grid_size; z++) {
		for (int x = 0; x <= grid_size; x++) {
			int idx = x + z * (grid_size + 1);
			float u = x / (float)grid_size;
			float v = z / (float)grid_size;

			// Position in normalized coordinates (0-1 range)
			terrain_vertices[idx].position = (skr_vec3_t){u, 0.0f, v};
			terrain_vertices[idx].normal   = (skr_vec3_t){0.0f, 1.0f, 0.0f};
			terrain_vertices[idx].uv       = (skr_vec2_t){u, v};
			terrain_vertices[idx].color    = 0xFFFFFFFF;
		}
	}

	// Generate terrain indices
	int tri_idx = 0;
	for (int z = 0; z < grid_size; z++) {
		for (int x = 0; x < grid_size; x++) {
			int v0 = x + z * (grid_size + 1);
			int v1 = v0 + 1;
			int v2 = v0 + (grid_size + 1);
			int v3 = v2 + 1;

			terrain_indices[tri_idx++] = v0;
			terrain_indices[tri_idx++] = v2;
			terrain_indices[tri_idx++] = v1;

			terrain_indices[tri_idx++] = v1;
			terrain_indices[tri_idx++] = v2;
			terrain_indices[tri_idx++] = v3;
		}
	}

	skr_mesh_create(&su_vertex_type_pnuc, skr_index_fmt_u32, terrain_vertices, vertex_count, terrain_indices, index_count, &scene->terrain_mesh);
	skr_mesh_set_name(&scene->terrain_mesh, "erosion_terrain");
	free(terrain_vertices);
	free(terrain_indices);

	// Generate height texture with Perlin noise
	float* height_data = malloc(scene->terrain_size * scene->terrain_size * sizeof(float));
	for (int y = 0; y < scene->terrain_size; y++) {
		for (int x = 0; x < scene->terrain_size; x++) {
			float nx = x / (float)scene->terrain_size;
			float ny = y / (float)scene->terrain_size;

			// Multi-octave Perlin noise
			float height = _octave_perlin(nx * 4.0f, ny * 4.0f, 6, 0.5f);
			height = (height + 1.0f) * 0.5f; // Normalize to 0-1

			height_data[x + y * scene->terrain_size] = height;
		}
	}

	// Create height textures (R32 format, ping-pong for erosion)
	skr_tex_sampler_t height_sampler = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };

	skr_tex_create(skr_tex_fmt_r32,
		skr_tex_flags_readable | skr_tex_flags_compute,
		height_sampler,
		(skr_vec3i_t){scene->terrain_size, scene->terrain_size, 1}, 1, 1, height_data, &scene->height_tex_ping);
	skr_tex_set_name(&scene->height_tex_ping, "height_ping");

	skr_tex_create(skr_tex_fmt_r32,
		skr_tex_flags_readable | skr_tex_flags_compute,
		height_sampler,
		(skr_vec3i_t){scene->terrain_size, scene->terrain_size, 1}, 1, 1, height_data, &scene->height_tex_pong);
	skr_tex_set_name(&scene->height_tex_pong, "height_pong");
	free(height_data);

	// Create water textures (ping-pong buffers for compute)
	// Initialize with 0.05 water everywhere
	float* water_data = malloc(scene->terrain_size * scene->terrain_size * sizeof(float));
	for (int i = 0; i < scene->terrain_size * scene->terrain_size; i++) {
		water_data[i] = 0.00f;
	}
	skr_tex_sampler_t water_sampler = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };

	skr_tex_create(skr_tex_fmt_r32,
		skr_tex_flags_readable | skr_tex_flags_compute,
		water_sampler,
		(skr_vec3i_t){scene->terrain_size, scene->terrain_size, 1}, 1, 1, water_data, &scene->water_tex_ping);
	skr_tex_set_name(&scene->water_tex_ping, "water_ping");

	skr_tex_create(skr_tex_fmt_r32,
		skr_tex_flags_readable | skr_tex_flags_compute,
		water_sampler,
		(skr_vec3i_t){scene->terrain_size, scene->terrain_size, 1}, 1, 1, water_data, &scene->water_tex_pong);
	skr_tex_set_name(&scene->water_tex_pong, "water_pong");
	free(water_data);

	// Create sediment textures (ping-pong buffers for compute)
	// Initialize with zero sediment
	float* sediment_data = calloc(scene->terrain_size * scene->terrain_size, sizeof(float));
	skr_tex_sampler_t sediment_sampler = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };

	skr_tex_create(skr_tex_fmt_r32,
		skr_tex_flags_readable | skr_tex_flags_compute,
		sediment_sampler,
		(skr_vec3i_t){scene->terrain_size, scene->terrain_size, 1}, 1, 1, sediment_data, &scene->sediment_tex_ping);
	skr_tex_set_name(&scene->sediment_tex_ping, "sediment_ping");

	skr_tex_create(skr_tex_fmt_r32,
		skr_tex_flags_readable | skr_tex_flags_compute,
		sediment_sampler,
		(skr_vec3i_t){scene->terrain_size, scene->terrain_size, 1}, 1, 1, sediment_data, &scene->sediment_tex_pong);
	skr_tex_set_name(&scene->sediment_tex_pong, "sediment_pong");
	free(sediment_data);

	// Create flow direction buffer (StructuredBuffer for flow vectors)
	typedef struct {
		float outflow_right;
		float outflow_up;
		float outflow_left;
		float outflow_down;
	} flow_data_t;

	int32_t flow_count = scene->terrain_size * scene->terrain_size;
	flow_data_t* flow_data = calloc(flow_count, sizeof(flow_data_t));
	skr_buffer_create(flow_data, flow_count, sizeof(flow_data_t), skr_buffer_type_storage, skr_use_compute_readwrite, &scene->flow_buffer);
	skr_buffer_set_name(&scene->flow_buffer, "flow_buffer");
	free(flow_data);

	// Create debug texture (RGBA32F for general-purpose debug visualization)
	float* debug_data = calloc(scene->terrain_size * scene->terrain_size * 4, sizeof(float));
	skr_tex_sampler_t debug_sampler = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };

	skr_tex_create(skr_tex_fmt_rgba128,
		skr_tex_flags_readable | skr_tex_flags_compute,
		debug_sampler,
		(skr_vec3i_t){scene->terrain_size, scene->terrain_size, 1}, 1, 1, debug_data, &scene->debug_tex);
	skr_tex_set_name(&scene->debug_tex, "debug_tex");
	free(debug_data);

	// Load terrain shader
	scene->terrain_shader = su_shader_load("shaders/terrain_erosion.hlsl.sks", "terrain_shader");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->terrain_shader,
		.cull       = skr_cull_back,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->terrain_material);

	// Bind textures and buffers to terrain material
	skr_material_set_tex   (&scene->terrain_material, "height_map", &scene->height_tex_ping);
	skr_material_set_tex   (&scene->terrain_material, "water_map",  &scene->water_tex_ping);
	skr_material_set_buffer(&scene->terrain_material, "flow_map",   &scene->flow_buffer);
	skr_material_set_tex   (&scene->terrain_material, "debug_tex",  &scene->debug_tex);

	// Load compute shaders
	scene->flow_shader = su_shader_load("shaders/compute_flow.hlsl.sks", "flow_compute");
	scene->water_shader = su_shader_load("shaders/compute_water.hlsl.sks", "water_compute");

	skr_compute_create(&scene->flow_shader, &scene->compute_flow);
	skr_compute_create(&scene->water_shader, &scene->compute_water);

	// Create compute parameters buffer
	typedef struct {
		float    terrain_size;
		float    timestep;
		float    water_threshold;
		float    rainfall_rate;
		float    evaporation_rate;
		float    padding[3];
	} compute_params_t;

	compute_params_t compute_params = {
		.terrain_size      = (float)scene->terrain_size,
		.timestep          = 0.03f,
		.water_threshold   = 0.1f,
		.rainfall_rate     = 0.000f,  // Start with rainfall on
		.evaporation_rate  = 0.0001f,
		.padding           = {0, 0, 0}
	};
	skr_buffer_create(&compute_params, 1, sizeof(compute_params_t), skr_buffer_type_constant, skr_use_dynamic, &scene->compute_params_buffer);

	// Set up compute bindings for flow calculation
	skr_compute_set_tex   (&scene->compute_flow, "height_map", &scene->height_tex_ping);
	skr_compute_set_tex   (&scene->compute_flow, "water_map",  &scene->water_tex_ping);
	skr_compute_set_buffer(&scene->compute_flow, "out_flow",   &scene->flow_buffer);
	skr_compute_set_buffer(&scene->compute_flow, "$Global",    &scene->compute_params_buffer);

	// Set up compute bindings for water simulation (ping-pong)
	skr_compute_set_tex   (&scene->compute_water, "height_in",    &scene->height_tex_ping);
	skr_compute_set_buffer(&scene->compute_water, "flow_map",     &scene->flow_buffer);
	skr_compute_set_tex   (&scene->compute_water, "water_in",     &scene->water_tex_ping);
	skr_compute_set_tex   (&scene->compute_water, "sediment_in",  &scene->sediment_tex_ping);
	skr_compute_set_tex   (&scene->compute_water, "height_out",   &scene->height_tex_pong);
	skr_compute_set_tex   (&scene->compute_water, "water_out",    &scene->water_tex_pong);
	skr_compute_set_tex   (&scene->compute_water, "sediment_out", &scene->sediment_tex_pong);
	skr_compute_set_tex   (&scene->compute_water, "debug_out",    &scene->debug_tex);
	skr_compute_set_buffer(&scene->compute_water, "$Global",      &scene->compute_params_buffer);

	return (scene_t*)scene;
}

static void _scene_erosion_destroy(scene_t* base) {
	scene_erosion_t* scene = (scene_erosion_t*)base;

	skr_mesh_destroy(&scene->terrain_mesh);
	skr_material_destroy(&scene->terrain_material);
	skr_shader_destroy(&scene->terrain_shader);
	skr_shader_destroy(&scene->flow_shader);
	skr_shader_destroy(&scene->water_shader);
	skr_compute_destroy(&scene->compute_flow);
	skr_compute_destroy(&scene->compute_water);
	skr_tex_destroy(&scene->height_tex_ping);
	skr_tex_destroy(&scene->height_tex_pong);
	skr_tex_destroy(&scene->water_tex_ping);
	skr_tex_destroy(&scene->water_tex_pong);
	skr_tex_destroy(&scene->sediment_tex_ping);
	skr_tex_destroy(&scene->sediment_tex_pong);
	skr_buffer_destroy(&scene->flow_buffer);
	skr_tex_destroy(&scene->debug_tex);
	skr_buffer_destroy(&scene->compute_params_buffer);

	free(scene);
}

static void _scene_erosion_update(scene_t* base, float delta_time) {
	scene_erosion_t* scene = (scene_erosion_t*)base;
	scene->rotation += delta_time * 0.2f;
	scene->rainfall_timer += delta_time;

	// Toggle rainfall every 3 seconds
	typedef struct {
		float    terrain_size;
		float    timestep;
		float    water_threshold;
		float    rainfall_rate;
		float    evaporation_rate;
		float    padding[3];
	} compute_params_t;

	// Check if we should toggle (every 3 seconds)
	int32_t current_phase = (int32_t)(scene->rainfall_timer / 3.0f);
	bool    rainfall_on   = (current_phase % 2) == 0;

	compute_params_t params = {
		.terrain_size      = (float)scene->terrain_size,
		.timestep          = 0.03f,
		.water_threshold   = 0.1f,
		.rainfall_rate     = rainfall_on ? 0.01f  : 0.01f,
		.evaporation_rate  = rainfall_on ? 0.005f : 0.005f,  // Keep evaporation constant
		.padding           = {0, 0, 0}
	};
	skr_buffer_set(&scene->compute_params_buffer, &params, sizeof(compute_params_t));

	// Execute compute shaders (ping-pong between textures for height, water, and sediment)
	skr_compute_t* water_compute = &scene->compute_water;

	// Update bindings based on iteration
	if (scene->compute_iteration % 2 == 0) {
		// Flow compute: read from ping
		skr_compute_set_tex(&scene->compute_flow, "height_map", &scene->height_tex_ping);
		skr_compute_set_tex(&scene->compute_flow, "water_map",  &scene->water_tex_ping);

		// Water compute: ping -> pong
		skr_compute_set_tex(water_compute, "height_in",    &scene->height_tex_ping);
		skr_compute_set_tex(water_compute, "water_in",     &scene->water_tex_ping);
		skr_compute_set_tex(water_compute, "sediment_in",  &scene->sediment_tex_ping);
		skr_compute_set_tex(water_compute, "height_out",   &scene->height_tex_pong);
		skr_compute_set_tex(water_compute, "water_out",    &scene->water_tex_pong);
		skr_compute_set_tex(water_compute, "sediment_out", &scene->sediment_tex_pong);

		// Update render material to use pong outputs
		skr_material_set_tex(&scene->terrain_material, "height_map", &scene->height_tex_pong);
		skr_material_set_tex(&scene->terrain_material, "water_map",  &scene->water_tex_pong);
	} else {
		// Flow compute: read from pong
		skr_compute_set_tex(&scene->compute_flow, "height_map", &scene->height_tex_pong);
		skr_compute_set_tex(&scene->compute_flow, "water_map",  &scene->water_tex_pong);

		// Water compute: pong -> ping
		skr_compute_set_tex(water_compute, "height_in",    &scene->height_tex_pong);
		skr_compute_set_tex(water_compute, "water_in",     &scene->water_tex_pong);
		skr_compute_set_tex(water_compute, "sediment_in",  &scene->sediment_tex_pong);
		skr_compute_set_tex(water_compute, "height_out",   &scene->height_tex_ping);
		skr_compute_set_tex(water_compute, "water_out",    &scene->water_tex_ping);
		skr_compute_set_tex(water_compute, "sediment_out", &scene->sediment_tex_ping);

		// Update render material to use ping outputs
		skr_material_set_tex(&scene->terrain_material, "height_map", &scene->height_tex_ping);
		skr_material_set_tex(&scene->terrain_material, "water_map",  &scene->water_tex_ping);
	}

	// First pass: calculate flow directions based on water surface height
	skr_compute_execute(&scene->compute_flow, scene->terrain_size / 8, scene->terrain_size / 8, 1);

	// Second pass: simulate water flow, erosion, and deposition
	skr_compute_execute(water_compute, scene->terrain_size / 8, scene->terrain_size / 8, 1);
	scene->compute_iteration++;
}

static void _scene_erosion_render(scene_t* base, int32_t width, int32_t height, HMM_Mat4 viewproj, skr_render_list_t* ref_render_list, app_system_buffer_t* ref_system_buffer) {
	scene_erosion_t* scene = (scene_erosion_t*)base;

	// Build instance data for terrain
	HMM_Mat4 terrain_instance = su_matrix_trs(
		HMM_V3(-scene->terrain_scale * 0.5f, 0.0f, -scene->terrain_scale * 0.5f),
		HMM_V3(0.0f, 0.0f, 0.0f),
		HMM_V3(scene->terrain_scale, scene->terrain_height_scale, scene->terrain_scale)
	);

	// Add to render list
	skr_render_list_add(ref_render_list, &scene->terrain_mesh, &scene->terrain_material, &terrain_instance, sizeof(HMM_Mat4), 1);
}

static bool _scene_erosion_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_erosion_t* scene = (scene_erosion_t*)base;

	// Orbit camera around the terrain
	float radius = scene->terrain_scale * 0.8f;
	float height = scene->terrain_height_scale * 2;
	float angle = 0.5f;// scene->rotation;

	out_camera->position = HMM_V3(cosf(angle) * radius, height, sinf(angle) * radius);
	out_camera->target   = HMM_V3(0.0f, scene->terrain_height_scale * 0.3f, 0.0f);
	out_camera->up       = HMM_V3(0.0f, 1.0f, 0.0f);

	return true;
}

const scene_vtable_t scene_erosion_vtable = {
	.name       = "Hydraulic Erosion Simulation",
	.create     = _scene_erosion_create,
	.destroy    = _scene_erosion_destroy,
	.update     = _scene_erosion_update,
	.render     = _scene_erosion_render,
	.get_camera = _scene_erosion_get_camera,
};
