// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>

// Reaction-diffusion scene - displays compute shader simulation on a quad
typedef struct {
	scene_t       base;

	skr_mesh_t     quad_mesh;
	skr_shader_t   shader;
	skr_shader_t   compute_sh;
	skr_material_t quad_material;
	skr_compute_t  compute_ping;
	skr_compute_t  compute_pong;
	skr_buffer_t   compute_buffer_a;
	skr_buffer_t   compute_buffer_b;
	skr_tex_t      compute_output;

	int32_t sim_size;
	int32_t compute_iteration;
	float   rotation;

	// Flipped once per second and pushed via skr_compute_set_pipeline (see update).
	bool  invert_color;
	float spec_timer;
} scene_reaction_diffusion_t;

// Rebuild both compute pipelines with a new INVERT_COLOR spec value; the
// buffers/textures/params bound at create time persist across the rebuild.
static void _reaction_diffusion_set_invert(scene_reaction_diffusion_t* scene, bool invert) {
	skr_spec_constant_t spec = { .name = "INVERT_COLOR", .value = invert ? 1.0 : 0.0 };
	skr_compute_info_t  info = {
		.spec_constants      = &spec,
		.spec_constant_count = 1,
	};
	skr_compute_set_pipeline(&scene->compute_ping, info);
	skr_compute_set_pipeline(&scene->compute_pong, info);
}

// Helper function for random hash
static float _hash_f(int32_t aPosition, uint32_t aSeed) {
	const uint32_t BIT_NOISE1 = 0x68E31DA4;
	const uint32_t BIT_NOISE2 = 0xB5297A4D;
	const uint32_t BIT_NOISE3 = 0x1B56C4E9;

	uint32_t mangled = (uint32_t)aPosition;
	mangled *= BIT_NOISE1;
	mangled += aSeed;
	mangled ^= (mangled >> 8);
	mangled += BIT_NOISE2;
	mangled ^= (mangled << 8);
	mangled *= BIT_NOISE3;
	mangled ^= (mangled >> 8);
	return (float)mangled / (float)4294967295;
}

static scene_t* _scene_reaction_diffusion_create(void) {
	scene_reaction_diffusion_t* scene = calloc(1, sizeof(scene_reaction_diffusion_t));
	if (!scene) return NULL;

	scene->base.size         = sizeof(scene_reaction_diffusion_t);
	scene->sim_size          = 512;
	scene->compute_iteration = 0;
	scene->rotation          = 0.0f;

	// Create double-sided quad mesh (front face + back face with flipped normals)
	su_vertex_t quad_vertices[] = {
		// Front face (Z+)
		{ .position = {-0.7f, -0.7f, 0.0f}, .normal = { 0.0f,  0.0f,  1.0f}, .uv = {0.0f, 0.0f}, .color = 0xFFFFFFFF },
		{ .position = { 0.7f, -0.7f, 0.0f}, .normal = { 0.0f,  0.0f,  1.0f}, .uv = {1.0f, 0.0f}, .color = 0xFFFFFFFF },
		{ .position = { 0.7f,  0.7f, 0.0f}, .normal = { 0.0f,  0.0f,  1.0f}, .uv = {1.0f, 1.0f}, .color = 0xFFFFFFFF },
		{ .position = {-0.7f,  0.7f, 0.0f}, .normal = { 0.0f,  0.0f,  1.0f}, .uv = {0.0f, 1.0f}, .color = 0xFFFFFFFF },
		// Back face (Z-) - same positions, flipped normals and winding
		{ .position = {-0.7f, -0.7f, 0.0f}, .normal = { 0.0f,  0.0f, -1.0f}, .uv = {0.0f, 0.0f}, .color = 0xFFFFFFFF },
		{ .position = { 0.7f, -0.7f, 0.0f}, .normal = { 0.0f,  0.0f, -1.0f}, .uv = {1.0f, 0.0f}, .color = 0xFFFFFFFF },
		{ .position = { 0.7f,  0.7f, 0.0f}, .normal = { 0.0f,  0.0f, -1.0f}, .uv = {1.0f, 1.0f}, .color = 0xFFFFFFFF },
		{ .position = {-0.7f,  0.7f, 0.0f}, .normal = { 0.0f,  0.0f, -1.0f}, .uv = {0.0f, 1.0f}, .color = 0xFFFFFFFF },
	};
	uint16_t quad_indices[] = {
		0, 1, 2,  2, 3, 0,  // Front face
		5, 4, 7,  7, 6, 5,  // Back face (flipped winding)
	};
	skr_mesh_create(&su_vertex_type, skr_index_fmt_u16, quad_vertices, 8, quad_indices, 12, &scene->quad_mesh);
	skr_mesh_set_name(&scene->quad_mesh, "quad");

	// Load shaders
	scene->shader = su_shader_load("shaders/test.hlsl.sks", "main_shader");
	skr_material_create((skr_material_info_t){
		.shader       = &scene->shader,
		.cull         = skr_cull_back,
		.write_mask   = skr_write_default,
		.depth_test   = skr_compare_less,
	}, &scene->quad_material);

	// Load compute shader
	scene->compute_sh = su_shader_load("shaders/compute_test.hlsl.sks", NULL);
	skr_compute_create(&scene->compute_sh, (skr_compute_info_t){0}, &scene->compute_ping);
	skr_compute_create(&scene->compute_sh, (skr_compute_info_t){0}, &scene->compute_pong);

	// Create compute resources
	float2* initial_data = malloc(scene->sim_size * scene->sim_size * sizeof(float2));
	for (int y = 0; y < scene->sim_size; y++) {
		for (int x = 0; x < scene->sim_size; x++) {
			float r = _hash_f(1, (uint32_t)((x/16)*13+(y/16)*127));
			initial_data[x + y * scene->sim_size].x = r;
			initial_data[x + y * scene->sim_size].y = 1.0f - r;
		}
	}

	skr_buffer_create(initial_data, scene->sim_size * scene->sim_size, sizeof(float2), skr_buffer_type_storage, skr_use_compute_readwrite, &scene->compute_buffer_a);
	skr_buffer_create(initial_data, scene->sim_size * scene->sim_size, sizeof(float2), skr_buffer_type_storage, skr_use_compute_readwrite, &scene->compute_buffer_b);
	free(initial_data);

	skr_tex_sampler_t default_sampler = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };
	skr_tex_create(skr_tex_fmt_rgba32_linear,
		skr_tex_flags_readable | skr_tex_flags_compute,
		default_sampler,
		(skr_vec3i_t){scene->sim_size, scene->sim_size, 1}, 1, 1, NULL, &scene->compute_output);

	// Set up compute bindings
	skr_compute_set_buffer(&scene->compute_ping, "input",   &scene->compute_buffer_a);
	skr_compute_set_buffer(&scene->compute_ping, "output",  &scene->compute_buffer_b);
	skr_compute_set_tex   (&scene->compute_ping, "out_tex", &scene->compute_output);

	skr_compute_set_buffer(&scene->compute_pong, "input",   &scene->compute_buffer_b);
	skr_compute_set_buffer(&scene->compute_pong, "output",  &scene->compute_buffer_a);
	skr_compute_set_tex   (&scene->compute_pong, "out_tex", &scene->compute_output);

	// Set compute parameters using reflection API
	skr_compute_set_param(&scene->compute_ping, "feed",     sksc_shader_var_float, 1, &(float){0.042f});
	skr_compute_set_param(&scene->compute_ping, "kill",     sksc_shader_var_float, 1, &(float){0.059f});
	skr_compute_set_param(&scene->compute_ping, "diffuseA", sksc_shader_var_float, 1, &(float){0.2097f});
	skr_compute_set_param(&scene->compute_ping, "diffuseB", sksc_shader_var_float, 1, &(float){0.105f});
	skr_compute_set_param(&scene->compute_ping, "timestep", sksc_shader_var_float, 1, &(float){0.8f});
	skr_compute_set_param(&scene->compute_ping, "size",     sksc_shader_var_uint,  1, &(uint32_t){scene->sim_size});

	skr_compute_set_param(&scene->compute_pong, "feed",     sksc_shader_var_float, 1, &(float){0.042f});
	skr_compute_set_param(&scene->compute_pong, "kill",     sksc_shader_var_float, 1, &(float){0.059f});
	skr_compute_set_param(&scene->compute_pong, "diffuseA", sksc_shader_var_float, 1, &(float){0.2097f});
	skr_compute_set_param(&scene->compute_pong, "diffuseB", sksc_shader_var_float, 1, &(float){0.105f});
	skr_compute_set_param(&scene->compute_pong, "timestep", sksc_shader_var_float, 1, &(float){0.8f});
	skr_compute_set_param(&scene->compute_pong, "size",     sksc_shader_var_uint,  1, &(uint32_t){scene->sim_size});

	// Bind texture to material
	skr_material_set_tex(&scene->quad_material, "tex", &scene->compute_output);

	return (scene_t*)scene;
}

static void _scene_reaction_diffusion_destroy(scene_t* base) {
	scene_reaction_diffusion_t* scene = (scene_reaction_diffusion_t*)base;

	skr_mesh_destroy(&scene->quad_mesh);
	skr_material_destroy(&scene->quad_material);
	skr_compute_destroy(&scene->compute_ping);
	skr_compute_destroy(&scene->compute_pong);
	skr_shader_destroy(&scene->compute_sh);
	skr_shader_destroy(&scene->shader);
	skr_tex_destroy(&scene->compute_output);
	skr_buffer_destroy(&scene->compute_buffer_a);
	skr_buffer_destroy(&scene->compute_buffer_b);

	free(scene);
}

static void _scene_reaction_diffusion_update(scene_t* base, float delta_time) {
	scene_reaction_diffusion_t* scene = (scene_reaction_diffusion_t*)base;
	scene->rotation += delta_time;

	// Flip the palette each second to exercise live pipeline rebuilds.
	scene->spec_timer += delta_time;
	if (scene->spec_timer >= 1.0f) {
		scene->spec_timer  -= 1.0f;
		scene->invert_color = !scene->invert_color;
		_reaction_diffusion_set_invert(scene, scene->invert_color);
	}

	// Execute compute shader
	for (int c = 0; c < 2; c++) {
		skr_compute_t* current = (scene->compute_iteration % 2 == 0) ? &scene->compute_ping : &scene->compute_pong;
		skr_compute_execute(current, scene->sim_size / 8, scene->sim_size / 8, 1);
		scene->compute_iteration++;
	}
}

static void _scene_reaction_diffusion_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_reaction_diffusion_t* scene = (scene_reaction_diffusion_t*)base;

	// Build instance data for quad
	float4x4 quad_instance = float4x4_trs(
		(float3){0.0f, 0.0f, 0.0f},
		float4_quat_from_euler((float3){0.0f, -scene->rotation, 0.0f}),
		(float3){6.0f, 6.0f, 6.0f} );

	// Add to render list
	skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->quad_material, &quad_instance, sizeof(float4x4), 1);
}

const scene_vtable_t scene_reaction_diffusion_vtable = {
	.name       = "Reaction-Diffusion Simulation",
	.create     = _scene_reaction_diffusion_create,
	.destroy    = _scene_reaction_diffusion_destroy,
	.update     = _scene_reaction_diffusion_update,
	.render     = _scene_reaction_diffusion_render,
	.get_camera = NULL,
};
