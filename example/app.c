// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "app.h"
#include "scene.h"
#include "tools/scene_util.h"
#include "bloom.h"
#include "imgui_backend/imgui_impl_sk_renderer.h"

#include "tools/float_math.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <sk_app.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

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

	// Performance tracking
	float   frame_time_ms;
	float   gpu_time_total_ms;
	float   gpu_time_min_ms;
	float   gpu_time_max_ms;
	int32_t gpu_time_samples;

	// Frame time history for graphs (circular buffer)
	#define FRAME_HISTORY_SIZE 512
	float   frame_time_history[512];
	float   gpu_time_history[512];
	int32_t history_index;
};

static const char* _tex_fmt_name(skr_tex_fmt_ fmt) {
	switch (fmt) {
	case skr_tex_fmt_none:          return "none";
	case skr_tex_fmt_rgba32_srgb:   return "rgba32_srgb";
	case skr_tex_fmt_rgba32_linear: return "rgba32_linear";
	case skr_tex_fmt_bgra32_srgb:   return "bgra32_srgb";
	case skr_tex_fmt_bgra32_linear: return "bgra32_linear";
	case skr_tex_fmt_rg11b10:       return "rg11b10";
	case skr_tex_fmt_rgb10a2:       return "rgb10a2";
	case skr_tex_fmt_rgba64u:       return "rgba64u";
	case skr_tex_fmt_rgba64s:       return "rgba64s";
	case skr_tex_fmt_rgba64f:       return "rgba64f";
	case skr_tex_fmt_rgba128:       return "rgba128";
	case skr_tex_fmt_r8:            return "r8";
	case skr_tex_fmt_r16u:          return "r16u";
	case skr_tex_fmt_r16s:          return "r16s";
	case skr_tex_fmt_r16f:          return "r16f";
	case skr_tex_fmt_r32f:          return "r32f";
	case skr_tex_fmt_r8g8:          return "r8g8";
	case skr_tex_fmt_rgb9e5:        return "rgb9e5";
	case skr_tex_fmt_depth16:       return "depth16";
	case skr_tex_fmt_depth32:       return "depth32";
	case skr_tex_fmt_depth24s8:     return "depth24s8";
	case skr_tex_fmt_depth32s8:     return "depth32s8";
	case skr_tex_fmt_depth16s8:     return "depth16s8";
	default:                        return "unknown";
	}
}

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

	su_log(su_log_info, "Render target: %dx%d @ %dx, %s / %s",
		width, height, app->msaa, _tex_fmt_name(skr_tex_get_format(&app->color_msaa)), _tex_fmt_name(app->depth_format));
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

	su_log(su_log_info, "Switched to scene: %s", scene_get_name(app->scene_types[new_index]));
}

app_t* app_create(int32_t start_scene) {
	app_t* app = calloc(1, sizeof(app_t));
	if (!app) return NULL;

	app->msaa            = 4;
	app->gpu_time_min_ms = 1e10f;
	app->offscreen_format = skr_tex_fmt_rgba32_srgb;//skr_tex_fmt_bgra32_srgb;

	// Choose depth format (prefer smaller/faster formats with stencil for stencil masking demo)
	if (skr_tex_fmt_is_supported(skr_tex_fmt_depth16s8, skr_tex_flags_writeable, app->msaa)) {
		app->depth_format = skr_tex_fmt_depth16s8;
	} else if (skr_tex_fmt_is_supported(skr_tex_fmt_depth24s8, skr_tex_flags_writeable, app->msaa)) {
		app->depth_format = skr_tex_fmt_depth24s8;
	} else if (skr_tex_fmt_is_supported(skr_tex_fmt_depth32s8, skr_tex_flags_writeable, app->msaa)) {
		app->depth_format = skr_tex_fmt_depth32s8;
	} else if (skr_tex_fmt_is_supported(skr_tex_fmt_depth16, skr_tex_flags_writeable, app->msaa)) {
		app->depth_format = skr_tex_fmt_depth16;
	} else if (skr_tex_fmt_is_supported(skr_tex_fmt_depth32, skr_tex_flags_writeable, app->msaa)) {
		app->depth_format = skr_tex_fmt_depth32;
	} else {
		su_log(su_log_critical, "No supported depth format found!");
		free(app);
		return NULL;
	}


	// Create shared render list
	skr_render_list_create(&app->render_list);

	// Register available scenes
	app->scene_types[0]  = &scene_meshes_vtable;
	app->scene_types[1]  = &scene_reaction_diffusion_vtable;
	app->scene_types[2]  = &scene_orbital_particles_vtable;
	app->scene_types[3]  = &scene_impostor_vtable;
	app->scene_types[4]  = &scene_array_texture_vtable;
	app->scene_types[5]  = &scene_3d_texture_vtable;
	app->scene_types[6]  = &scene_cubemap_vtable;
	app->scene_types[7]  = &scene_gltf_vtable;
	app->scene_types[8]  = &scene_shadows_vtable;
	app->scene_types[9]  = &scene_cloth_vtable;
	app->scene_types[10] = &scene_text_vtable;
	app->scene_types[11] = &scene_tex_copy_vtable;
	app->scene_types[12] = &scene_lifetime_stress_vtable;
	app->scene_types[13] = &scene_gaussian_splat_vtable;
	app->scene_count = 14;
#ifdef SKR_HAS_VIDEO
	app->scene_types[app->scene_count++] = &scene_video_vtable;
#endif

	su_log(su_log_info, "Application created successfully!");
	su_log(su_log_info, "Available scenes: %d (use arrow keys to switch)", app->scene_count);

	// Start with the requested scene (default to 0 if out of range or -1)
	app->scene_index = -1;
	int32_t initial_scene = (start_scene >= 0 && start_scene < app->scene_count) ? start_scene : 13;
	_switch_scene(app, initial_scene);

	return app;
}

void app_destroy(app_t* app) {
	if (!app) return;

	// Log GPU performance summary
	if (app->gpu_time_samples > 0) {
		float avg_ms = app->gpu_time_total_ms / app->gpu_time_samples;
		su_log(su_log_info, "GPU Time: avg %.2f ms (%.1f FPS), min %.2f ms, max %.2f ms, %d samples",
			avg_ms, 1000.0f / avg_ms,
			app->gpu_time_min_ms, app->gpu_time_max_ms,
			app->gpu_time_samples);
	}

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

	// Shutdown scene utilities (stops asset loading thread)
	su_shutdown();

	free(app);

	su_log(su_log_info, "Application destroyed");
}

void app_set_scene(app_t* app, int32_t scene_index) {
	if (!app) return;
	if (scene_index < 0 || scene_index >= app->scene_count) return;
	_switch_scene(app, scene_index);
}

int32_t app_scene_count(app_t* app) {
	return app ? app->scene_count : 0;
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

void app_set_frame_time(app_t* app, float frame_time_ms) {
	if (app) app->frame_time_ms = frame_time_ms;
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

	// Calculate view-projection matrix (float_math handles Y flip and row-major layout internally)
	float    aspect     = (float)width / (float)height;
	float4x4 projection = float4x4_perspective(60.0f * (3.14159265359f / 180.0f), aspect, 0.1f, 100.0f);

	// Use scene camera if provided, otherwise use default
	scene_camera_t camera;
	const scene_vtable_t* vtable = app->scene_types[app->scene_index];
	bool has_custom_camera = vtable->get_camera && vtable->get_camera(app->scene_current, &camera);

	float3 cam_position = has_custom_camera ? camera.position : (float3){0.0f, 3.0f, 8.0f};
	float3 cam_target   = has_custom_camera ? camera.target   : (float3){0.0f, 0.0f, 0.0f};
	float3 cam_up       = has_custom_camera ? camera.up       : (float3){0.0f, 1.0f, 0.0f};

	float4x4 view = float4x4_lookat(cam_position, cam_target, cam_up);

	// Calculate camera direction
	float3 cam_forward = float3_norm(float3_sub(cam_target, cam_position));

	// Setup application system buffer
	su_system_buffer_t sys_buffer = {0};
	sys_buffer.view_count = 1;
	sys_buffer.view          [0] = view;
	sys_buffer.projection    [0] = projection;
	sys_buffer.viewproj      [0] = float4x4_mul   (projection, view);
	sys_buffer.view_inv      [0] = float4x4_invert(view);
	sys_buffer.projection_inv[0] = float4x4_invert(projection);
	sys_buffer.cam_pos       [0] = (float4){cam_position.x, cam_position.y, cam_position.z, 0.0f};
	sys_buffer.cam_dir       [0] = (float4){cam_forward.x,  cam_forward.y,  cam_forward.z,  0.0f};

	// Let the scene populate the render list (and optionally do its own render passes)
	scene_render(vtable, app->scene_current, width, height, &app->render_list, &sys_buffer);

	// Prepare ImGui mesh data OUTSIDE render pass (uploads via vkCmdCopyBuffer)
	ImGui_ImplSkRenderer_PrepareDrawData();

	// Begin main render pass
	skr_vec4_t clear_color = {0, 0, 0, 0};
	skr_tex_t* color_target   = (app->msaa > 1) ? &app->color_msaa : (enable_offscreen ? &app->scene_color : render_target);
	skr_tex_t* resolve_target = (app->msaa > 1) ? (enable_offscreen ? &app->scene_color : render_target) : NULL;
	skr_renderer_begin_pass(color_target, &app->depth_buffer, resolve_target, skr_clear_all, clear_color, 1.0f, 0);

	// Set viewport and scissor
	skr_renderer_set_viewport((skr_rect_t ){0, 0, (float)width, (float)height});
	skr_renderer_set_scissor ((skr_recti_t){0, 0, width, height});

	// Draw the render list that the scene populated
	skr_renderer_draw    (&app->render_list, &sys_buffer, sizeof(su_system_buffer_t), sys_buffer.view_count);
	skr_render_list_clear(&app->render_list);

	// Draw ImGui INSIDE the same render pass
	ImGui_ImplSkRenderer_RenderDrawData(width, height);

	// End render pass
	skr_renderer_end_pass();

	// Post-processing
	if (enable_offscreen && enable_bloom) {
		bloom_apply(&app->scene_color, render_target, 1.0f, 4.0f, 0.75f);
	}
}

void app_render_imgui(app_t* app, skr_tex_t* render_target, int32_t width, int32_t height) {
	if (!app) return;

	// Position window on the right side of the screen (locked)
	#if defined(ANDROID)
	float size = 600;
	#else
	float size = 300;
	#endif
	igSetNextWindowPos ((ImVec2){(float)width - size, 0}, ImGuiCond_Always, (ImVec2){0, 0});
	igSetNextWindowSize((ImVec2){size, (float)height}, ImGuiCond_Always);

	// Build a simple info window with no move/resize
	igBegin("sk_renderer", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

	// Show scene info with navigation buttons
	igText("%s", scene_get_name(app->scene_types[app->scene_index]));
	if (igArrowButton("##left",  ImGuiDir_Left )) { _switch_scene(app, (app->scene_index - 1 + app->scene_count) % app->scene_count);}
	igSameLine(0.0f, 5.0f);
	if (igArrowButton("##right", ImGuiDir_Right)) { _switch_scene(app, (app->scene_index + 1) % app->scene_count); }

	igSeparator();

	// Scene-specific UI controls (re-fetch vtable in case scene changed above)
	scene_render_ui(app->scene_types[app->scene_index], app->scene_current);

	igSeparator();

	// Show render info
	igText("Resolution: %d x %d", width, height);
	igText("MSAA: %dx", app->msaa);

	float gpu_ms   = skr_renderer_get_gpu_time_ms();
	float frame_ms = app->frame_time_ms;

	// Track GPU performance stats
	if (gpu_ms > 0.0f) {
		app->gpu_time_total_ms += gpu_ms;
		app->gpu_time_samples++;
		if (gpu_ms < app->gpu_time_min_ms) app->gpu_time_min_ms = gpu_ms;
		if (gpu_ms > app->gpu_time_max_ms) app->gpu_time_max_ms = gpu_ms;
	}

	// Store history in circular buffer
	app->frame_time_history[app->history_index] = frame_ms;
	app->gpu_time_history  [app->history_index] = gpu_ms > 0.0f ? gpu_ms : app->gpu_time_history[(app->history_index + FRAME_HISTORY_SIZE - 1) % FRAME_HISTORY_SIZE];
	app->history_index = (app->history_index + 1) % FRAME_HISTORY_SIZE;

	igText("Frame Time: %.2f ms (%.1f FPS)", frame_ms, 1000.0f / frame_ms);
	igText("GPU Time: %.2f ms (%.1f FPS)", gpu_ms, 1000.0f / gpu_ms);

	// Frame time graph (4ms to 18ms range)
	const float cpu_graph_min = 6.0f;
	const float cpu_graph_max = 10.0f;

	const float gpu_graph_min = 0.0f;
	const float gpu_graph_max = 3.0f;

	// Get available width for full-width plots
	ImVec2 content_region;
	igGetContentRegionAvail(&content_region);
	float plot_width = content_region.x;

	char frame_overlay[32], gpu_overlay[32];
	snprintf(frame_overlay, sizeof(frame_overlay), "Frame: %.1f ms", frame_ms);
	snprintf(gpu_overlay,   sizeof(gpu_overlay),   "GPU: %.1f ms",   gpu_ms > 0.0f ? gpu_ms : 0.0f);

	// Plot frame time - using values_offset for circular buffer
	igPlotLines_FloatPtr("##frame_graph", app->frame_time_history, FRAME_HISTORY_SIZE,
		app->history_index, frame_overlay, cpu_graph_min, cpu_graph_max, (ImVec2){plot_width, 60}, sizeof(float));

	igPlotLines_FloatPtr("##gpu_graph", app->gpu_time_history, FRAME_HISTORY_SIZE,
		app->history_index, gpu_overlay, gpu_graph_min, gpu_graph_max, (ImVec2){plot_width, 60}, sizeof(float));

	igEnd();
}
