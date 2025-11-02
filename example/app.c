// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "app.h"
#include "scene.h"
#include "scene_util.h"
#include "bloom.h"

#define HANDMADE_MATH_IMPLEMENTATION
#include "HandmadeMath.h"

#include <stdlib.h>
#include <string.h>

const bool enable_offscreen = false;
const bool enable_bloom     = false;
const bool enable_stereo    = true;

// Application state
struct app_t {
	// Rendering configuration
	int32_t       msaa;
	skr_tex_fmt_  offscreen_format;
	skr_tex_fmt_  depth_format;

	// Scene management
	const scene_vtable_t* scene_types[16];  // Array of available scenes
	int32_t               scene_count;
	int32_t               scene_index;
	scene_t*              scene_current;

	// Render targets (recreated on resize)
	skr_tex_t color_msaa;
	skr_tex_t depth_buffer;
	skr_tex_t scene_color;
	int32_t   current_width;
	int32_t   current_height;

	// Shared render list (reused each frame)
	skr_render_list_t render_list;
};

static void _create_render_targets(app_t* app, int32_t width, int32_t height, skr_tex_t* render_target) {
	skr_tex_sampler_t no_sampler   = {0};
	skr_tex_sampler_t linear_clamp = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };

	// MSAA buffer must match the format of its resolve target
	skr_tex_fmt_ msaa_format = enable_offscreen ? app->offscreen_format : render_target->format;
	skr_tex_create(msaa_format,       skr_tex_flags_writeable, no_sampler, (skr_vec3i_t){width, height, 1}, app->msaa, 1, NULL, &app->color_msaa);
	skr_tex_create(app->depth_format, skr_tex_flags_writeable, no_sampler, (skr_vec3i_t){width, height, 1}, app->msaa, 1, NULL, &app->depth_buffer);

	if (enable_offscreen) {
		skr_tex_create(app->offscreen_format,
			skr_tex_flags_readable | skr_tex_flags_compute,
			linear_clamp,
			(skr_vec3i_t){width, height, 1}, 1, 1, NULL, &app->scene_color);
	}

	app->current_width  = width;
	app->current_height = height;

	skr_logf(skr_log_info, "Render targets created: %dx%d, color=%d, depth=%d, scene=%d",
		width, height, skr_tex_is_valid(&app->color_msaa), skr_tex_is_valid(&app->depth_buffer), skr_tex_is_valid(&app->scene_color));
}

static void _destroy_render_targets(app_t* app) {
	skr_tex_destroy(&app->color_msaa);
	skr_tex_destroy(&app->depth_buffer);
	if (enable_offscreen) {
		skr_tex_destroy(&app->scene_color);
	}
}

static void _switch_scene(app_t* app, int32_t new_index) {
	if (new_index < 0 || new_index >= app->scene_count) return;
	if (new_index == app->scene_index) return;

	// Destroy current scene
	if (app->scene_current) {
		scene_destroy(app->scene_types[app->scene_index], app->scene_current);
	}

	// Create new scene
	app->scene_index   = new_index;
	app->scene_current = scene_create(app->scene_types[new_index]);

	skr_logf(skr_log_info, "Switched to scene: %s", scene_get_name(app->scene_types[new_index]));
}

app_t* app_create() {
	app_t* app = calloc(1, sizeof(app_t));
	if (!app) return NULL;

	app->msaa             = 4;
	app->offscreen_format = skr_tex_fmt_rgba32;//skr_tex_fmt_bgra32;
	#if ANDROID
	app->depth_format     = skr_tex_fmt_depth16;  // Use depth+stencil for stencil masking demo
	#else
	app->depth_format     = skr_tex_fmt_depth16s8;  // Use depth+stencil for stencil masking demo
	#endif

	// Initialize standard vertex types
	su_vertex_types_init();

	// Create shared render list
	skr_render_list_create(&app->render_list);

	// Register available scenes
	app->scene_types[0] = &scene_meshes_vtable;
	app->scene_types[1] = &scene_reaction_diffusion_vtable;
	app->scene_types[2] = &scene_orbital_particles_vtable;
	app->scene_types[3] = &scene_impostor_vtable;
	app->scene_types[4] = &scene_array_texture_vtable;
	app->scene_types[5] = &scene_3d_texture_vtable;
	app->scene_types[6] = &scene_cubemap_vtable;
	app->scene_types[7] = &scene_gltf_vtable;
	app->scene_types[8] = &scene_shadows_vtable;
	app->scene_types[9] = &scene_erosion_vtable;
	app->scene_count = 10;

	// Start with the first scene
	app->scene_index = -1;
	_switch_scene(app, 9);

	skr_log (skr_log_info, "Application created successfully!");
	skr_logf(skr_log_info, "Available scenes: %d (use arrow keys to switch)", app->scene_count);

	return app;
}

void app_destroy(app_t* app) {
	if (!app) return;

	// Destroy current scene
	if (app->scene_current) {
		scene_destroy(app->scene_types[app->scene_index], app->scene_current);
	}

	// Destroy render targets
	_destroy_render_targets(app);

	// Destroy render list
	skr_render_list_destroy(&app->render_list);

	// Destroy bloom
	if (enable_bloom) {
		bloom_destroy();
	}

	free(app);

	skr_log(skr_log_info, "Application destroyed");
}

void app_key_press(app_t* app, app_key_ key) {
	if (!app) return;

	switch (key) {
		case app_key_left:
			_switch_scene(app, (app->scene_index - 1 + app->scene_count) % app->scene_count);
			break;
		case app_key_right:
			_switch_scene(app, (app->scene_index + 1) % app->scene_count);
			break;
	}
}

void app_resize(app_t* app, int32_t width, int32_t height, skr_tex_t* render_target) {
	if (!app) return;

	// Destroy old render targets
	_destroy_render_targets(app);

	// Create new render targets
	_create_render_targets(app, width, height, render_target);

	// Recreate bloom textures
	if (enable_bloom) {
		bloom_resize(width, height);
	}
}

void app_update(app_t* app, float delta_time) {
	if (!app || !app->scene_current) return;

	scene_update(app->scene_types[app->scene_index], app->scene_current, delta_time);
}

void app_render(app_t* app, skr_tex_t* render_target, int32_t width, int32_t height) {
	if (!app || !app->scene_current) return;

	// Check if we need to create or resize render targets
	if (app->current_width != width || app->current_height != height) {
		if (app->current_width == 0) {
			// First time - create render targets
			_create_render_targets(app, width, height, render_target);
			if (enable_bloom) {
				bloom_create(width, height, 7);
			}
		} else {
			// Resize
			app_resize(app, width, height, render_target);
		}
	}

	// Calculate view-projection matrix
	float aspect = (float)width / (float)height;
	HMM_Mat4 projection = HMM_Perspective_RH_NO(HMM_AngleDeg(60.0f), aspect, 0.1f, 100.0f);
	projection.Elements[1][1] *= -1.0f;  // Flip Y for Vulkan

	// Use scene camera if provided, otherwise use default
	scene_camera_t camera;
	const scene_vtable_t* vtable = app->scene_types[app->scene_index];
	bool has_custom_camera = vtable->get_camera && vtable->get_camera(app->scene_current, &camera);

	HMM_Mat4 view = has_custom_camera
		? HMM_LookAt_RH(camera.position, camera.target, camera.up)
		: HMM_LookAt_RH(HMM_V3(0.0f, 3.0f, 8.0f), HMM_V3(0.0f, 0.0f, 0.0f), HMM_V3(0.0f, 1.0f, 0.0f));

	HMM_Mat4 vp       = HMM_MulM4(projection, view);
	HMM_Mat4 viewproj = HMM_Transpose(vp);

	// Calculate inverse matrices
	HMM_Mat4 view_inv       = HMM_Transpose(HMM_InvGeneralM4(view));
	HMM_Mat4 projection_inv = HMM_Transpose(HMM_InvGeneralM4(projection));
	view       = HMM_Transpose(view);
	projection = HMM_Transpose(projection);

	// Calculate camera position and direction
	HMM_Vec3 cam_position = has_custom_camera ? camera.position : HMM_V3(0.0f, 3.0f, 8.0f);
	HMM_Vec3 cam_target   = has_custom_camera ? camera.target   : HMM_V3(0.0f, 0.0f, 0.0f);
	HMM_Vec3 cam_forward  = HMM_NormV3(HMM_SubV3(cam_target, cam_position));

	// Setup application system buffer
	app_system_buffer_t sys_buffer = {0};
	sys_buffer.view_count = 1;
	memcpy(sys_buffer.viewproj       [0], &viewproj,       sizeof(float) * 16);
	memcpy(sys_buffer.view           [0], &view,           sizeof(float) * 16);
	memcpy(sys_buffer.view_inv       [0], &view_inv,       sizeof(float) * 16);
	memcpy(sys_buffer.projection     [0], &projection,     sizeof(float) * 16);
	memcpy(sys_buffer.projection_inv [0], &projection_inv, sizeof(float) * 16);
	sys_buffer.cam_pos[0][0] = cam_position.X;
	sys_buffer.cam_pos[0][1] = cam_position.Y;
	sys_buffer.cam_pos[0][2] = cam_position.Z;
	sys_buffer.cam_pos[0][3] = 0.0f;
	sys_buffer.cam_dir[0][0] = cam_forward.X;
	sys_buffer.cam_dir[0][1] = cam_forward.Y;
	sys_buffer.cam_dir[0][2] = cam_forward.Z;
	sys_buffer.cam_dir[0][3] = 0.0f;

	// Let the scene populate the render list (and optionally do its own render passes)
	scene_render(vtable, app->scene_current, width, height, viewproj, &app->render_list, &sys_buffer);

	// Begin main render pass
	skr_vec4_t clear_color = {0, 0, 0, 0};
	skr_tex_t* color_target   = (app->msaa > 1) ? &app->color_msaa : (enable_offscreen ? &app->scene_color : render_target);
	skr_tex_t* resolve_target = (app->msaa > 1) ? (enable_offscreen ? &app->scene_color : render_target) : NULL;
	skr_renderer_begin_pass(color_target, &app->depth_buffer, resolve_target, skr_clear_all, clear_color, 1.0f, 0);

	// Set viewport and scissor
	skr_renderer_set_viewport((skr_rect_t ){0, 0, (float)width, (float)height});
	skr_renderer_set_scissor ((skr_recti_t){0, 0, width, height});

	// Draw the render list that the scene populated
	skr_renderer_draw    (&app->render_list, &sys_buffer, sizeof(app_system_buffer_t), sys_buffer.view_count);
	skr_render_list_clear(&app->render_list);

	skr_renderer_end_pass();

	// Post-processing
	if (enable_offscreen && enable_bloom) {
		bloom_apply(&app->scene_color, render_target, 1.0f, 4.0f, 0.75f);
	}
}
