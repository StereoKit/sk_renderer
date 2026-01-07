// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// Lifetime Stress Test Scene
// Tests various resource creation/destruction patterns to validate thread safety
// and deferred destruction systems.

#define MAX_EPHEMERAL_MATERIALS 32
#define MAX_THREAD_MATERIALS    8
#define STRESS_CUBE_COUNT       25

typedef struct {
	skr_material_t material;
	skr_tex_t      texture;
	int32_t        frames_alive;
	int32_t        destroy_after_frames;
	bool           in_use;
} ephemeral_resource_t;

typedef struct {
	skr_material_t material;
	skr_tex_t      texture;
	bool           ready;
	bool           used;
} thread_resource_t;

typedef struct {
	scene_t base;

	// Permanent resources (for comparison/baseline rendering)
	skr_mesh_t     cube_mesh;
	skr_shader_t   shader;
	skr_material_t base_material;
	skr_tex_t      base_texture;

	// Test 1: Create-use-destroy in same frame
	uint32_t test1_count;

	// Test 2: Create one frame, destroy next
	ephemeral_resource_t ephemeral[MAX_EPHEMERAL_MATERIALS];
	uint32_t             ephemeral_next;

	// Test 3: Thread-created resources
	pthread_t          thread;
	thread_resource_t  thread_resources[MAX_THREAD_MATERIALS];
	pthread_mutex_t    thread_mutex;
	volatile bool      thread_running;
	volatile bool      thread_should_stop;
	uint32_t           thread_create_count;

	// Test 4: Rapid create/destroy cycles
	uint32_t rapid_cycle_count;
	uint32_t rapid_cycles_per_frame;

	// Test 5: Texture replacement stress
	skr_tex_t      replaceable_texture;
	skr_material_t replaceable_material;
	uint32_t       texture_replace_count;

	// Test 6: Sampler stress (different sampler settings)
	skr_tex_t      sampler_test_textures[8];
	uint32_t       sampler_test_count;

	// Test 7: True destroy-before-draw (validates crash behavior)
	uint32_t       test7_count;

	// Statistics
	uint32_t frame_count;
	uint32_t total_creates;
	uint32_t total_destroys;
	uint32_t total_draws;
	float    rotation;
} scene_lifetime_stress_t;

// Thread function that creates resources
static void* _thread_create_resources(void* arg) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)arg;

	// Register this thread with sk_renderer command system
	skr_thread_init();

	while (!scene->thread_should_stop) {
		pthread_mutex_lock(&scene->thread_mutex);

		// Find an unused slot
		int32_t slot = -1;
		for (int32_t i = 0; i < MAX_THREAD_MATERIALS; i++) {
			if (!scene->thread_resources[i].ready && !scene->thread_resources[i].used) {
				slot = i;
				break;
			}
		}

		pthread_mutex_unlock(&scene->thread_mutex);

		if (slot >= 0) {
			// Create resources outside the lock
			thread_resource_t* res = &scene->thread_resources[slot];

			// Cull modes for pipeline variety
			skr_cull_ cull_modes[] = { skr_cull_back, skr_cull_front, skr_cull_none };

			// Create a unique colored texture
			uint32_t color = 0xFF000000 | ((slot * 37) << 16) | ((slot * 73) << 8) | (slot * 113);
			res->texture = su_tex_create_solid_color(color);
			skr_tex_set_name(&res->texture, "thread_tex");

			// Create material with varied pipeline settings
			skr_material_create((skr_material_info_t){
				.shader     = &scene->shader,
				.depth_test = skr_compare_less,
				.cull       = cull_modes[slot % 3],  // Vary cull mode based on slot
			}, &res->material);
			skr_material_set_tex(&res->material, "tex", &res->texture);

			// Mark as ready
			pthread_mutex_lock(&scene->thread_mutex);
			res->ready = true;
			scene->thread_create_count++;
			pthread_mutex_unlock(&scene->thread_mutex);
		}

		// Sleep a bit to avoid spinning
		struct timespec ts = { .tv_sec = 0, .tv_nsec = 10000000 }; // 10ms
		nanosleep(&ts, NULL);
	}

	// Unregister this thread from sk_renderer command system
	skr_thread_shutdown();

	return NULL;
}

static scene_t* _scene_lifetime_stress_create(void) {
	scene_lifetime_stress_t* scene = calloc(1, sizeof(scene_lifetime_stress_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_lifetime_stress_t);
	scene->rotation = 0.0f;
	scene->rapid_cycles_per_frame = 5;

	// Create base resources
	scene->cube_mesh = su_mesh_create_cube(0.8f, NULL);
	skr_mesh_set_name(&scene->cube_mesh, "stress_cube");

	scene->base_texture = su_tex_create_checkerboard(64, 8, 0xFFFFFFFF, 0xFF4444FF, true);
	skr_tex_set_name(&scene->base_texture, "stress_base_tex");

	scene->shader = su_shader_load("shaders/test.hlsl.sks", "stress_shader");

	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.depth_test = skr_compare_less,
	}, &scene->base_material);
	skr_material_set_tex(&scene->base_material, "tex", &scene->base_texture);

	// Create replaceable texture/material for Test 5
	scene->replaceable_texture = su_tex_create_solid_color(0xFF00FF00);
	skr_tex_set_name(&scene->replaceable_texture, "replaceable_tex");

	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.depth_test = skr_compare_less,
	}, &scene->replaceable_material);
	skr_material_set_tex(&scene->replaceable_material, "tex", &scene->replaceable_texture);

	// Create sampler test textures with different sampler settings
	for (int32_t i = 0; i < 8; i++) {
		uint32_t color = 0xFF000000 | (i * 32);
		skr_tex_sampler_t sampler = {
			.sample     = (i % 2 == 0) ? skr_tex_sample_linear : skr_tex_sample_point,
			.address    = (i % 3 == 0) ? skr_tex_address_wrap : skr_tex_address_clamp,
			.anisotropy = (i % 4) + 1,
		};

		skr_tex_create(skr_tex_fmt_rgba32_linear, skr_tex_flags_dynamic, sampler, (skr_vec3i_t){4, 4, 1}, 1, 1, NULL, &scene->sampler_test_textures[i]);

		// Fill with color
		uint32_t pixels[16];
		for (int32_t j = 0; j < 16; j++) pixels[j] = color | ((j * 16) << 8);
		skr_tex_set_data(&scene->sampler_test_textures[i], &(skr_tex_data_t){.data = pixels, .mip_count = 1, .layer_count = 1});
	}

	// Initialize thread resources
	pthread_mutex_init(&scene->thread_mutex, NULL);
	scene->thread_running = true;
	scene->thread_should_stop = false;
	pthread_create(&scene->thread, NULL, _thread_create_resources, scene);

	su_log(su_log_info, "Lifetime stress test scene created");

	return (scene_t*)scene;
}

static void _scene_lifetime_stress_destroy(scene_t* base) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;

	// Stop thread
	scene->thread_should_stop = true;
	pthread_join(scene->thread, NULL);
	pthread_mutex_destroy(&scene->thread_mutex);

	// Destroy thread resources
	for (int32_t i = 0; i < MAX_THREAD_MATERIALS; i++) {
		if (scene->thread_resources[i].ready || scene->thread_resources[i].used) {
			skr_material_destroy(&scene->thread_resources[i].material);
			skr_tex_destroy(&scene->thread_resources[i].texture);
		}
	}

	// Destroy ephemeral resources
	for (int32_t i = 0; i < MAX_EPHEMERAL_MATERIALS; i++) {
		if (scene->ephemeral[i].in_use) {
			skr_material_destroy(&scene->ephemeral[i].material);
			skr_tex_destroy(&scene->ephemeral[i].texture);
		}
	}

	// Destroy sampler test textures
	for (int32_t i = 0; i < 8; i++) {
		skr_tex_destroy(&scene->sampler_test_textures[i]);
	}

	// Destroy replaceable resources
	skr_material_destroy(&scene->replaceable_material);
	skr_tex_destroy(&scene->replaceable_texture);

	// Destroy base resources
	skr_material_destroy(&scene->base_material);
	skr_tex_destroy(&scene->base_texture);
	skr_shader_destroy(&scene->shader);
	skr_mesh_destroy(&scene->cube_mesh);

	su_log(su_log_info, "Lifetime stress test: %u creates, %u destroys, %u draws over %u frames",
		scene->total_creates, scene->total_destroys, scene->total_draws, scene->frame_count);

	free(scene);
}

static void _scene_lifetime_stress_update(scene_t* base, float dt) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;
	scene->rotation += dt * 0.5f;
	scene->frame_count++;

	// Test 2: Age ephemeral resources and destroy when ready
	for (int32_t i = 0; i < MAX_EPHEMERAL_MATERIALS; i++) {
		ephemeral_resource_t* eph = &scene->ephemeral[i];
		if (eph->in_use) {
			eph->frames_alive++;
			if (eph->frames_alive >= eph->destroy_after_frames) {
				skr_material_destroy(&eph->material);
				skr_tex_destroy(&eph->texture);
				eph->in_use = false;
				scene->total_destroys++;
			}
		}
	}

	// Test 3: Check for used thread resources and destroy them
	pthread_mutex_lock(&scene->thread_mutex);
	for (int32_t i = 0; i < MAX_THREAD_MATERIALS; i++) {
		thread_resource_t* res = &scene->thread_resources[i];
		if (res->used && !res->ready) {
			// Was used and is no longer marked ready, destroy it
			skr_material_destroy(&res->material);
			skr_tex_destroy(&res->texture);
			res->used = false;
			scene->total_destroys++;
		}
	}
	pthread_mutex_unlock(&scene->thread_mutex);
}

static void _scene_lifetime_stress_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	(void)width; (void)height; (void)ref_system_buffer;
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;

	float4x4 transforms[STRESS_CUBE_COUNT];
	int32_t  draw_idx = 0;
	float3   unit_scale = {1, 1, 1};

	// Cull modes to cycle through for pipeline variety
	skr_cull_ cull_modes[] = { skr_cull_back, skr_cull_front, skr_cull_none };

	// === TEST 1: Create-use-destroy in same frame ===
	// Note: Resources are added to the ephemeral pool with 1-frame lifetime.
	// This ensures destruction happens after skr_renderer_draw processes the list.
	for (int32_t i = 0; i < 3; i++) {
		// Find an empty ephemeral slot
		int32_t slot = -1;
		for (int32_t j = 0; j < MAX_EPHEMERAL_MATERIALS; j++) {
			if (!scene->ephemeral[j].in_use) {
				slot = j;
				break;
			}
		}
		if (slot < 0) continue; // No slots available

		ephemeral_resource_t* eph = &scene->ephemeral[slot];

		// Create temporary texture and material with varied pipeline settings
		uint32_t color = 0xFFFF0000 | ((scene->frame_count * 17 + i * 73) & 0xFFFF);
		eph->texture = su_tex_create_solid_color(color);

		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.depth_test = skr_compare_less,
			.cull       = cull_modes[i % 3],  // Vary cull mode for different pipelines
		}, &eph->material);
		skr_material_set_tex(&eph->material, "tex", &eph->texture);

		eph->frames_alive = 0;
		eph->destroy_after_frames = 1; // Destroy next frame's update (after this frame's render)
		eph->in_use = true;

		scene->total_creates++;

		// Add to render list
		float x = -4.0f + i * 1.5f;
		float y = 2.0f;
		transforms[draw_idx] = float4x4_trs(
			(float3){x, y, 0},
			float4_quat_from_euler((float3){0, scene->rotation + i * 0.5f, 0}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &eph->material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;
		scene->total_draws++;
		scene->test1_count++;
	}

	// === TEST 2: Create resources to destroy next frame ===
	if (scene->frame_count % 3 == 0) {
		int32_t slot = scene->ephemeral_next % MAX_EPHEMERAL_MATERIALS;
		ephemeral_resource_t* eph = &scene->ephemeral[slot];

		// Destroy old if exists
		if (eph->in_use) {
			skr_material_destroy(&eph->material);
			skr_tex_destroy(&eph->texture);
			scene->total_destroys++;
		}

		// Create new with varied pipeline settings
		uint32_t color = 0xFF00FF00 | ((scene->frame_count * 31) & 0xFF00);
		eph->texture = su_tex_create_solid_color(color);

		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.depth_test = skr_compare_less,
			.cull       = cull_modes[slot % 3],  // Vary cull mode based on slot
		}, &eph->material);
		skr_material_set_tex(&eph->material, "tex", &eph->texture);

		eph->frames_alive = 0;
		eph->destroy_after_frames = 2 + (scene->frame_count % 5); // Destroy after 2-6 frames
		eph->in_use = true;
		scene->ephemeral_next++;
		scene->total_creates++;
	}

	// Draw all active ephemeral resources
	for (int32_t i = 0; i < MAX_EPHEMERAL_MATERIALS && draw_idx < STRESS_CUBE_COUNT; i++) {
		ephemeral_resource_t* eph = &scene->ephemeral[i];
		if (eph->in_use) {
			float x = -3.0f + (i % 8) * 1.0f;
			float y = 0.0f;
			float z = (float)(i / 8) * 1.5f;
			transforms[draw_idx] = float4x4_trs(
				(float3){x, y, z},
				float4_quat_from_euler((float3){0, scene->rotation * 0.5f + i * 0.3f, 0}),
				unit_scale);
			skr_render_list_add(ref_render_list, &scene->cube_mesh, &eph->material, &transforms[draw_idx], sizeof(float4x4), 1);
			draw_idx++;
			scene->total_draws++;
		}
	}

	// === TEST 3: Use thread-created resources ===
	pthread_mutex_lock(&scene->thread_mutex);
	for (int32_t i = 0; i < MAX_THREAD_MATERIALS && draw_idx < STRESS_CUBE_COUNT; i++) {
		thread_resource_t* res = &scene->thread_resources[i];
		if (res->ready) {
			float x = 3.0f + (i % 4) * 1.0f;
			float y = -1.5f;
			float z = (float)(i / 4) * 1.5f;
			transforms[draw_idx] = float4x4_trs(
				(float3){x, y, z},
				float4_quat_from_euler((float3){scene->rotation + i * 0.4f, 0, 0}),
				unit_scale);
			skr_render_list_add(ref_render_list, &scene->cube_mesh, &res->material, &transforms[draw_idx], sizeof(float4x4), 1);
			draw_idx++;
			scene->total_draws++;

			// Mark as used, will be destroyed in update and recreated by thread
			if (scene->frame_count % 10 == (uint32_t)i) {
				res->ready = false;
				res->used = true;
			}
		}
	}
	pthread_mutex_unlock(&scene->thread_mutex);

	// === TEST 4: Rapid create/destroy cycles ===
	for (uint32_t cycle = 0; cycle < scene->rapid_cycles_per_frame; cycle++) {
		// Create with varied pipeline settings
		skr_tex_t rapid_tex = su_tex_create_solid_color(0xFFFF00FF);
		skr_material_t rapid_mat;
		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.depth_test = skr_compare_less,
			.cull       = cull_modes[cycle % 3],  // Vary cull mode
		}, &rapid_mat);
		skr_material_set_tex(&rapid_mat, "tex", &rapid_tex);
		scene->total_creates++;

		// Immediately destroy without using
		skr_material_destroy(&rapid_mat);
		skr_tex_destroy(&rapid_tex);
		scene->total_destroys++;
		scene->rapid_cycle_count++;
	}

	// === TEST 5: Texture replacement ===
	if (scene->frame_count % 5 == 0) {
		// Replace the texture with a new one
		skr_tex_t old_tex = scene->replaceable_texture;

		uint32_t new_color = 0xFF000000 | (scene->frame_count * 12345);
		scene->replaceable_texture = su_tex_create_solid_color(new_color);
		skr_tex_set_name(&scene->replaceable_texture, "replaceable_tex_new");

		// Update material to use new texture
		skr_material_set_tex(&scene->replaceable_material, "tex", &scene->replaceable_texture);

		// Destroy old texture
		skr_tex_destroy(&old_tex);
		scene->texture_replace_count++;
		scene->total_creates++;
		scene->total_destroys++;
	}

	// Draw replaceable material cube
	if (draw_idx < STRESS_CUBE_COUNT) {
		transforms[draw_idx] = float4x4_trs(
			(float3){0, -2.5f, 0},
			float4_quat_from_euler((float3){0, 0, scene->rotation * 2.0f}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &scene->replaceable_material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;
		scene->total_draws++;
	}

	// === TEST 6: Sampler cache stress ===
	// Create materials with different sampler settings each frame to stress the cache
	if (scene->frame_count % 2 == 0) {
		int32_t sampler_idx = scene->frame_count % 8;

		// Modify sampler settings on existing texture
		skr_tex_sampler_t new_sampler = {
			.sample     = (scene->frame_count % 2 == 0) ? skr_tex_sample_linear : skr_tex_sample_point,
			.address    = (scene->frame_count % 3 == 0) ? skr_tex_address_wrap : skr_tex_address_clamp,
			.anisotropy = (scene->frame_count % 4) + 1,
		};
		skr_tex_set_sampler(&scene->sampler_test_textures[sampler_idx], new_sampler);
		scene->sampler_test_count++;
	}

	// === TEST 7: True destroy-before-draw ===
	// This test validates that materials destroyed before render list processing
	// are handled correctly. Currently this crashes - we'll fix this next.
	if (draw_idx < STRESS_CUBE_COUNT) {
		// Create a material
		skr_material_t doomed_material;
		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.depth_test = skr_compare_less,
			.cull       = skr_cull_back,
		}, &doomed_material);
		skr_material_set_tex(&doomed_material, "tex", &scene->base_texture);

		// Add it to the render list - stores a POINTER to doomed_material
		transforms[draw_idx] = float4x4_trs(
			(float3){0, 3.0f, 0},
			float4_quat_from_euler((float3){0, scene->rotation * 3.0f, 0}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &doomed_material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;

		// IMMEDIATELY destroy the material - render list still has pointer to freed memory!
		skr_material_destroy(&doomed_material);

		scene->test7_count++;
		scene->total_creates++;
		scene->total_destroys++;
		scene->total_draws++;
		// When skr_renderer_draw processes this render list, it will try to
		// access doomed_material.param_buffer which is now freed -> CRASH
	}

	// Draw base cubes in a grid
	for (int32_t i = 0; i < 5 && draw_idx < STRESS_CUBE_COUNT; i++) {
		float x = -2.0f + i * 1.0f;
		float y = 1.0f;
		transforms[draw_idx] = float4x4_trs(
			(float3){x, y, -3.0f},
			float4_quat_from_euler((float3){0, scene->rotation + i * 0.2f, 0}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &scene->base_material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;
		scene->total_draws++;
	}
}

static void _scene_lifetime_stress_render_ui(scene_t* base) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;

	igText("Test 1 - Same-frame create/destroy: %u", scene->test1_count);
	igText("Test 2 - Multi-frame ephemeral: %u", scene->ephemeral_next);

	int32_t active_ephemeral = 0;
	for (int32_t i = 0; i < MAX_EPHEMERAL_MATERIALS; i++) {
		if (scene->ephemeral[i].in_use) active_ephemeral++;
	}
	igText("  Active ephemeral: %d", active_ephemeral);

	pthread_mutex_lock(&scene->thread_mutex);
	igText("Test 3 - Thread-created: %u", scene->thread_create_count);
	int32_t ready_count = 0;
	for (int32_t i = 0; i < MAX_THREAD_MATERIALS; i++) {
		if (scene->thread_resources[i].ready) ready_count++;
	}
	igText("  Ready to use: %d", ready_count);
	pthread_mutex_unlock(&scene->thread_mutex);

	igText("Test 4 - Rapid cycles: %u", scene->rapid_cycle_count);

	int cycles = (int)scene->rapid_cycles_per_frame;
	if (igSliderInt("Cycles/frame", &cycles, 0, 50, "%d", 0)) {
		scene->rapid_cycles_per_frame = (uint32_t)cycles;
	}

	igText("Test 5 - Texture replacements: %u", scene->texture_replace_count);
	igText("Test 6 - Sampler changes: %u", scene->sampler_test_count);

	igText("Test 7 - Destroy before draw: %u", scene->test7_count);

	igSeparator();
	igText("Totals:");
	igText("  Creates:  %u", scene->total_creates);
	igText("  Destroys: %u", scene->total_destroys);
	igText("  Draws:    %u", scene->total_draws);

	float creates_per_frame = scene->frame_count > 0 ? (float)scene->total_creates / scene->frame_count : 0;
	float destroys_per_frame = scene->frame_count > 0 ? (float)scene->total_destroys / scene->frame_count : 0;
	igText("  Creates/frame:  %.1f", creates_per_frame);
	igText("  Destroys/frame: %.1f", destroys_per_frame);
}

const scene_vtable_t scene_lifetime_stress_vtable = {
	.name      = "Lifetime Stress",
	.create    = _scene_lifetime_stress_create,
	.destroy   = _scene_lifetime_stress_destroy,
	.update    = _scene_lifetime_stress_update,
	.render    = _scene_lifetime_stress_render,
	.render_ui = _scene_lifetime_stress_render_ui,
};
