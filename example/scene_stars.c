// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// Star field scene - displays randomly distributed stars as single-pixel triangles
// Stars are uniformly distributed on a sphere using proper spherical distribution

#define STAR_COUNT     8000
#define STAR_DISTANCE  40.0f

typedef struct {
	scene_t        base;
	skr_mesh_t     star_mesh;
	skr_shader_t   shader;
	skr_material_t material;
	float          time;
	float          delta_time;

	// Arc-ball camera state
	float  cam_yaw;
	float  cam_pitch;
	float  cam_distance;
	float  cam_yaw_vel;
	float  cam_pitch_vel;
	float  cam_distance_vel;
	float3 cam_target;
	float3 cam_target_vel;
} scene_stars_t;

// Vertex format for stars: position + vertex index (in UV.x) + brightness (in color)
// We use UV.x to store 0, 1, or 2 to identify which vertex of the triangle this is
typedef struct {
	skr_vec3_t position;
	skr_vec3_t normal;   // unused but needed for standard vertex format
	skr_vec2_t uv;       // uv.x = vertex index (0, 1, 2), uv.y = unused
	uint32_t   color;    // brightness encoded in all channels
} star_vertex_t;

// Simple LCG random number generator for reproducible results
static uint32_t _star_rand_state = 12345;

static uint32_t _star_rand(void) {
	_star_rand_state = _star_rand_state * 1103515245 + 12345;
	return (_star_rand_state >> 16) & 0x7FFF;
}

static float _star_randf(void) {
	return (float)_star_rand() / 32767.0f;
}

static void _star_rand_seed(uint32_t seed) {
	_star_rand_state = seed;
}

static scene_t* _scene_stars_create(void) {
	scene_stars_t* scene = calloc(1, sizeof(scene_stars_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_stars_t);
	scene->time      = 0.0f;

	// Initialize camera
	scene->cam_yaw          = 0.0f;
	scene->cam_pitch        = 0.3f;  // Slight upward tilt
	scene->cam_distance     = 5.0f;
	scene->cam_target       = (float3){ 0.0f, 0.0f, 0.0f };
	scene->cam_yaw_vel      = 0.0f;
	scene->cam_pitch_vel    = 0.0f;
	scene->cam_distance_vel = 0.0f;
	scene->cam_target_vel   = (float3){ 0.0f, 0.0f, 0.0f };

	// Seed the random number generator for reproducible star positions
	_star_rand_seed(42);

	// Allocate vertices and indices
	// Each star = 3 vertices + 3 indices
	int32_t        vertex_count = STAR_COUNT * 3;
	int32_t        index_count  = STAR_COUNT * 3;
	star_vertex_t* vertices     = malloc(vertex_count * sizeof(star_vertex_t));
	uint32_t*      indices      = malloc(index_count  * sizeof(uint32_t));

	if (!vertices || !indices) {
		free(vertices);
		free(indices);
		free(scene);
		return NULL;
	}

	// Generate stars with uniform distribution on sphere
	for (int32_t i = 0; i < STAR_COUNT; i++) {
		// Uniform distribution on sphere:
		// z = random(-1, 1)
		// theta = random(0, 2*pi)
		// x = sqrt(1 - z^2) * cos(theta)
		// y = sqrt(1 - z^2) * sin(theta)
		float z       = _star_randf() * 2.0f - 1.0f;
		float theta   = _star_randf() * 2.0f * 3.14159265359f;
		float r       = sqrtf(1.0f - z * z);
		float x       = r * cosf(theta);
		float y       = r * sinf(theta);

		// Scale to star distance
		skr_vec3_t pos = {
			x * STAR_DISTANCE,
			y * STAR_DISTANCE,
			z * STAR_DISTANCE
		};

		// Random brightness (0 to 1) with sRGB to linear conversion
		float    brightness_srgb   = powf(_star_randf(), 2.5f); // exponent of 2.5 to bias towards dimmer stars, more stars are further away IRL
		float    brightness_linear = powf(brightness_srgb, 2.2f);
		uint8_t  bright_u8         = (uint8_t)(brightness_linear * 255.0f);
		uint32_t color             = (0xFF << 24) | (bright_u8 << 16) | (bright_u8 << 8) | bright_u8;

		// Create 3 vertices at the same position, with vertex index in UV.x
		int32_t base_vertex = i * 3;
		for (int32_t v = 0; v < 3; v++) {
			vertices[base_vertex + v] = (star_vertex_t){
				.position = pos,
				.normal   = {0, 1, 0},
				.uv       = {(float)v, brightness_linear}, // vertex index in x, brightness in y for shader
				.color    = color,
			};
		}

		// Create indices for the triangle
		int32_t base_index = i * 3;
		indices[base_index + 0] = base_vertex + 0;
		indices[base_index + 1] = base_vertex + 1;
		indices[base_index + 2] = base_vertex + 2;
	}

	// Create mesh
	skr_mesh_create(&su_vertex_type, skr_index_fmt_u32, vertices, vertex_count, indices, index_count, &scene->star_mesh);
	skr_mesh_set_name(&scene->star_mesh, "star_mesh");

	free(vertices);
	free(indices);

	// Load shader
	scene->shader = su_shader_load("shaders/stars.hlsl.sks", "stars_shader");

	// Create opaque material - stars are too small to overlap
	skr_material_create((skr_material_info_t){
		.shader       = &scene->shader,
		.cull         = skr_cull_none,  // Stars visible from all directions
		.depth_test   = skr_compare_less,
		.write_mask   = skr_write_default,
	}, &scene->material);

	return (scene_t*)scene;
}

static void _scene_stars_destroy(scene_t* base) {
	scene_stars_t* scene = (scene_stars_t*)base;

	skr_mesh_destroy    (&scene->star_mesh);
	skr_material_destroy(&scene->material);
	skr_shader_destroy  (&scene->shader);

	free(scene);
}

static void _scene_stars_update(scene_t* base, float delta_time) {
	scene_stars_t* scene = (scene_stars_t*)base;
	scene->time       += delta_time;
	scene->delta_time  = delta_time;
}

static void _scene_stars_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_stars_t* scene = (scene_stars_t*)base;
	(void)ref_system_buffer;
	(void)width;
	(void)height;

	// Identity transform - stars are already in world space at the correct distance
	float4x4 transform = float4x4_identity();
	skr_render_list_add(ref_render_list, &scene->star_mesh, &scene->material, &transform, sizeof(float4x4), 1);
}

static bool _scene_stars_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_stars_t* scene = (scene_stars_t*)base;
	float delta_time     = scene->delta_time;

	const float rotate_sensitivity = 0.0002f;
	const float pan_sensitivity    = 0.0001f;
	const float zoom_sensitivity   = 0.2f;
	const float velocity_damping   = 0.0001f;
	const float pitch_limit        = 1.5f;
	const float min_distance       = 1.0f;
	const float max_distance       = 40.0f;

	ImGuiIO* io = igGetIO();

	if (!io->WantCaptureMouse) {
		if (io->MouseDown[0]) {
			scene->cam_yaw_vel   -= io->MouseDelta.x * rotate_sensitivity;
			scene->cam_pitch_vel += io->MouseDelta.y * rotate_sensitivity;
		}
		if (io->MouseDown[1]) {
			float cos_yaw = cosf(scene->cam_yaw);
			float sin_yaw = sinf(scene->cam_yaw);
			float3 right  = { cos_yaw, 0.0f, -sin_yaw };
			float  pan_scale = scene->cam_distance * pan_sensitivity;
			scene->cam_target_vel.x -= right.x * io->MouseDelta.x * pan_scale;
			scene->cam_target_vel.z -= right.z * io->MouseDelta.x * pan_scale;
			scene->cam_target_vel.y += io->MouseDelta.y * pan_scale;
		}
		if (io->MouseWheel != 0.0f) {
			scene->cam_distance_vel -= io->MouseWheel * zoom_sensitivity;
		}
	}

	scene->cam_yaw      += scene->cam_yaw_vel;
	scene->cam_pitch    += scene->cam_pitch_vel;
	scene->cam_distance += scene->cam_distance_vel;
	scene->cam_target.x += scene->cam_target_vel.x;
	scene->cam_target.y += scene->cam_target_vel.y;
	scene->cam_target.z += scene->cam_target_vel.z;

	if (scene->cam_pitch >  pitch_limit) scene->cam_pitch =  pitch_limit;
	if (scene->cam_pitch < -pitch_limit) scene->cam_pitch = -pitch_limit;
	if (scene->cam_distance < min_distance) scene->cam_distance = min_distance;
	if (scene->cam_distance > max_distance) scene->cam_distance = max_distance;

	float damping = powf(velocity_damping, delta_time);
	scene->cam_yaw_vel      *= damping;
	scene->cam_pitch_vel    *= damping;
	scene->cam_distance_vel *= damping;
	scene->cam_target_vel.x *= damping;
	scene->cam_target_vel.y *= damping;
	scene->cam_target_vel.z *= damping;

	float cos_pitch = cosf(scene->cam_pitch);
	float sin_pitch = sinf(scene->cam_pitch);
	float cos_yaw   = cosf(scene->cam_yaw);
	float sin_yaw   = sinf(scene->cam_yaw);

	out_camera->position = (float3){
		scene->cam_target.x + scene->cam_distance * cos_pitch * sin_yaw,
		scene->cam_target.y + scene->cam_distance * sin_pitch,
		scene->cam_target.z + scene->cam_distance * cos_pitch * cos_yaw
	};
	out_camera->target = scene->cam_target;
	out_camera->up     = (float3){0, 1, 0};

	return true;
}

static void _scene_stars_render_ui(scene_t* base) {
	scene_stars_t* scene = (scene_stars_t*)base;

	igText("Stars: %d", STAR_COUNT);
	igText("Distance: %.1f", scene->cam_distance);

	if (igButton("Reset Camera", (ImVec2){0, 0})) {
		scene->cam_yaw          = 0.0f;
		scene->cam_pitch        = 0.3f;
		scene->cam_distance     = 5.0f;
		scene->cam_target       = (float3){ 0.0f, 0.0f, 0.0f };
		scene->cam_yaw_vel      = 0.0f;
		scene->cam_pitch_vel    = 0.0f;
		scene->cam_distance_vel = 0.0f;
		scene->cam_target_vel   = (float3){ 0.0f, 0.0f, 0.0f };
	}

	igSeparator();
	igTextWrapped("Left drag: rotate, Right drag: pan, Scroll: zoom");
}

const scene_vtable_t scene_stars_vtable = {
	.name       = "Stars",
	.create     = _scene_stars_create,
	.destroy    = _scene_stars_destroy,
	.update     = _scene_stars_update,
	.render     = _scene_stars_render,
	.get_camera = _scene_stars_get_camera,
	.render_ui  = _scene_stars_render_ui,
};
