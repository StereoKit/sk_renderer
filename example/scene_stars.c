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

#define STAR_COUNT     20000
#define STAR_DISTANCE  40.0f

typedef struct {
	scene_t        base;
	skr_mesh_t     star_mesh;
	skr_buffer_t   star_buffer;
	skr_shader_t   shader;
	skr_material_t material;
	float          time;

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

// Per-star data stored in a StructuredBuffer, indexed by SV_VertexID / 3
typedef struct {
	skr_vec3_t position;
	float      brightness;
} star_data_t;

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

	// Generate star data: position + brightness per star
	star_data_t* star_data = malloc(STAR_COUNT * sizeof(star_data_t));
	if (!star_data) { free(scene); return NULL; }

	for (int32_t i = 0; i < STAR_COUNT; i++) {
		float z     = _star_randf() * 2.0f - 1.0f;
		float theta = _star_randf() * 2.0f * 3.14159265359f;
		float rad   = sqrtf(1.0f - z * z);

		star_data[i] = (star_data_t){
			.position   = { rad * cosf(theta) * STAR_DISTANCE, rad * sinf(theta) * STAR_DISTANCE, z * STAR_DISTANCE },
			.brightness = powf(powf(_star_randf(), 2.5f), 2.2f),
		};
	}

	// Storage buffer for star data, indexed by SV_VertexID / 3 in the shader
	skr_buffer_create(star_data, STAR_COUNT, sizeof(star_data_t), skr_buffer_type_storage, skr_use_static, &scene->star_buffer);
	skr_buffer_set_name(&scene->star_buffer, "star_data");
	free(star_data);

	// Null vertex buffer mesh: vert_count drives vkCmdDraw, all data comes from the StructuredBuffer
	skr_mesh_create(NULL, skr_index_fmt_u32, NULL, STAR_COUNT * 3, NULL, 0, &scene->star_mesh);

	// Load shader
	scene->shader = su_shader_load("shaders/stars.hlsl.sks", "stars_shader");

	// Create opaque material - stars are too small to overlap
	skr_material_create((skr_material_info_t){
		.shader       = &scene->shader,
		.cull         = skr_cull_none,
		.depth_test   = skr_compare_less,
		.write_mask   = skr_write_default,
	}, &scene->material);
	skr_material_set_buffer(&scene->material, "stars", &scene->star_buffer);

	return (scene_t*)scene;
}

static void _scene_stars_destroy(scene_t* base) {
	scene_stars_t* scene = (scene_stars_t*)base;

	skr_mesh_destroy    (&scene->star_mesh);
	skr_buffer_destroy  (&scene->star_buffer);
	skr_material_destroy(&scene->material);
	skr_shader_destroy  (&scene->shader);

	free(scene);
}

static void _scene_stars_update(scene_t* base, float delta_time) {
	scene_stars_t* scene = (scene_stars_t*)base;
	scene->time += delta_time;

	// Camera control constants
	const float rotate_sensitivity = 0.0002f;
	const float pan_sensitivity    = 0.0001f;
	const float zoom_sensitivity   = 0.2f;
	const float velocity_damping   = 0.0001f;  // Per-second retention (lower = more damping)
	const float pitch_limit        = 1.5f;
	const float min_distance       = 1.0f;
	const float max_distance       = 40.0f;

	// Get ImGui IO for mouse input
	ImGuiIO* io = igGetIO();

	if (!io->WantCaptureMouse) {
		// Left mouse drag: arc rotate
		if (io->MouseDown[0]) {
			scene->cam_yaw_vel   -= io->MouseDelta.x * rotate_sensitivity;
			scene->cam_pitch_vel += io->MouseDelta.y * rotate_sensitivity;
		}

		// Right mouse drag: pan
		if (io->MouseDown[1]) {
			float cos_yaw = cosf(scene->cam_yaw);
			float sin_yaw = sinf(scene->cam_yaw);

			float3 right = { cos_yaw, 0.0f, -sin_yaw };

			float pan_scale = scene->cam_distance * pan_sensitivity;
			scene->cam_target_vel.x -= right.x * io->MouseDelta.x * pan_scale;
			scene->cam_target_vel.z -= right.z * io->MouseDelta.x * pan_scale;
			scene->cam_target_vel.y += io->MouseDelta.y * pan_scale;
		}

		// Mouse wheel: zoom
		if (io->MouseWheel != 0.0f) {
			scene->cam_distance_vel -= io->MouseWheel * zoom_sensitivity;
		}
	}

	// Apply velocities
	scene->cam_yaw      += scene->cam_yaw_vel;
	scene->cam_pitch    += scene->cam_pitch_vel;
	scene->cam_distance += scene->cam_distance_vel;
	scene->cam_target.x += scene->cam_target_vel.x;
	scene->cam_target.y += scene->cam_target_vel.y;
	scene->cam_target.z += scene->cam_target_vel.z;

	// Clamp pitch to avoid gimbal issues
	if (scene->cam_pitch >  pitch_limit) scene->cam_pitch =  pitch_limit;
	if (scene->cam_pitch < -pitch_limit) scene->cam_pitch = -pitch_limit;

	// Clamp distance
	if (scene->cam_distance < min_distance) scene->cam_distance = min_distance;
	if (scene->cam_distance > max_distance) scene->cam_distance = max_distance;

	// Apply damping (exponential decay, frame-rate independent)
	float damping = powf(velocity_damping, delta_time);
	scene->cam_yaw_vel      *= damping;
	scene->cam_pitch_vel    *= damping;
	scene->cam_distance_vel *= damping;
	scene->cam_target_vel.x *= damping;
	scene->cam_target_vel.y *= damping;
	scene->cam_target_vel.z *= damping;
}

static void _scene_stars_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_stars_t* scene = (scene_stars_t*)base;
	(void)ref_system_buffer;
	(void)width;
	(void)height;

	// No instance data - stars are already in world space, SV_InstanceID is just view_idx
	skr_render_list_add(ref_render_list, &scene->star_mesh, &scene->material, NULL, 0, 1);
}

static bool _scene_stars_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_stars_t* scene = (scene_stars_t*)base;

	// Compute camera position from spherical coordinates around target
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
