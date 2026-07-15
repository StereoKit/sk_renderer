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
#include <stddef.h>

#include <sk_app.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

const bool enable_offscreen       = false;
const bool enable_bloom           = false;
const bool enable_stereo          = true;
enum resolve_mode_ {
	resolve_mode_normal,         // No postfx, auto resolve
	resolve_mode_auto_postfx,    // Auto resolve + postfx invert
	resolve_mode_manual_postfx,  // Manual resolve + integral invert
	resolve_mode_wide_kernel,    // Wide-kernel resolve (Texture2DMS, separate pass)
	resolve_mode_oled_subpixel,  // OLED subpixel-aware resolve (Texture2DMS, separate pass)
	resolve_mode_max,
};
const char* resolve_mode_names[] = { "Normal", "Auto Resolve + Invert", "Manual Resolve + Invert", "Wide Kernel Resolve", "OLED Subpixel Resolve" };

// Application state
struct app_t {
	// Rendering configuration
	int32_t       msaa;
	int32_t       resolve_mode;           // enum resolve_mode_
	int32_t       resolve_mode_previous;
	skr_tex_fmt_  offscreen_format;
	skr_tex_fmt_  depth_format;

	// Resolution scaling
	float render_scale;           // 0.25–1.0, determines render target allocation size
	float render_scale_previous;  // For detecting changes that need reallocation
	float viewport_scale;         // 0.25–1.0, determines active viewport (no realloc)

	// Scene management
	const scene_vtable_t* scene_types[32];  // Array of available scenes
	int32_t               scene_count;
	int32_t               scene_index;
	scene_t*              scene_current;

	// Render targets (recreated on resize)
	skr_tex_t color_msaa;
	skr_tex_t depth_buffer;
	skr_tex_t scene_color;
	skr_tex_t upscale_target;  // Readable postfx output when upscale + postfx both active
	int32_t   current_width;
	int32_t   current_height;

	// Shared render list (reused each frame)
	skr_render_list_t render_list;

	// PostFX
	skr_shader_t   postfx_shader;
	skr_material_t postfx_mat;

	// Manual MSAA resolve
	skr_shader_t   resolve_shader;
	skr_material_t resolve_mat;

	// Wide-kernel resolve (Texture2DMS, separate pass)
	skr_shader_t      wide_resolve_shader;
	skr_material_t    wide_resolve_mat;
	// OLED subpixel resolve (Texture2DMS, separate pass)
	skr_shader_t      oled_resolve_shader;
	skr_material_t    oled_resolve_mat;
	// Upscale (resolution scaling blit)
	skr_shader_t   upscale_shader;
	skr_material_t upscale_mat;

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
	float   cpu_time_history[512];
	int32_t history_index;
	float   frame_ema;
	float   gpu_ema;
	float   cpu_ema;
};

static const char* _tex_fmt_name(skr_tex_fmt_ fmt) {
	switch (fmt) {
	case skr_tex_fmt_none:          return "none";
	case skr_tex_fmt_rgba32_srgb:   return "rgba32_srgb";
	case skr_tex_fmt_rgba32:        return "rgba32";
	case skr_tex_fmt_bgra32_srgb:   return "bgra32_srgb";
	case skr_tex_fmt_bgra32:        return "bgra32";
	case skr_tex_fmt_rg11b10uf:     return "rg11b10uf";
	case skr_tex_fmt_rgb10a2:       return "rgb10a2";
	case skr_tex_fmt_rgba64un:      return "rgba64un";
	case skr_tex_fmt_rgba64sn:      return "rgba64sn";
	case skr_tex_fmt_rgba64ui:      return "rgba64ui";
	case skr_tex_fmt_rgba64si:      return "rgba64si";
	case skr_tex_fmt_rgba64f:       return "rgba64f";
	case skr_tex_fmt_rgba128f:      return "rgba128f";
	case skr_tex_fmt_r8:            return "r8";
	case skr_tex_fmt_r8sn:          return "r8sn";
	case skr_tex_fmt_r8ui:          return "r8ui";
	case skr_tex_fmt_r8si:          return "r8si";
	case skr_tex_fmt_r8_srgb:       return "r8_srgb";
	case skr_tex_fmt_r16:           return "r16";
	case skr_tex_fmt_r16sn:         return "r16sn";
	case skr_tex_fmt_r16ui:         return "r16ui";
	case skr_tex_fmt_r16si:         return "r16si";
	case skr_tex_fmt_r16f:          return "r16f";
	case skr_tex_fmt_r32ui:         return "r32ui";
	case skr_tex_fmt_r32si:         return "r32si";
	case skr_tex_fmt_r32f:          return "r32f";
	case skr_tex_fmt_r8g8:          return "r8g8";
	case skr_tex_fmt_rgb9e5uf:      return "rgb9e5uf";
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

	// Apply render scale. Even dimensions required for MSAA.
	int32_t render_w = (int32_t)(width  * app->render_scale) & ~1;
	int32_t render_h = (int32_t)(height * app->render_scale) & ~1;
	if (render_w < 2) render_w = 2;
	if (render_h < 2) render_h = 2;

	bool needs_upscale = (app->render_scale < 1.0f) || (app->viewport_scale < 1.0f);

	// MSAA buffer must match the format of its resolve target.
	// input_attachment flag needed for manual MSAA resolve (SubpassInputMS reads samples directly).
	skr_tex_fmt_ msaa_format = (enable_offscreen || needs_upscale) ? app->offscreen_format : render_target->format;
	// input_attachment only when manual resolve reads MSAA samples as SubpassInputMS.
	// Wide-kernel resolve needs readable (SAMPLED_BIT) for Texture2DMS access in a separate pass.
	bool wide_kernel     = (app->resolve_mode == resolve_mode_wide_kernel);
	bool oled_subpixel   = (app->resolve_mode == resolve_mode_oled_subpixel);
	bool manual_resolve  = (app->resolve_mode == resolve_mode_manual_postfx);
	bool separate_pass   = wide_kernel || oled_subpixel;
	skr_tex_flags_ msaa_flags = skr_tex_flags_writeable
		| (manual_resolve ? skr_tex_flags_input_attachment : 0)
		| (separate_pass  ? skr_tex_flags_readable         : 0);
	skr_tex_create(msaa_format,       msaa_flags, separate_pass ? linear_clamp : no_sampler, (skr_vec3i_t){render_w, render_h, 1}, app->msaa, 1, NULL, &app->color_msaa);
	skr_tex_flags_ depth_flags = skr_tex_flags_writeable | ((app->msaa > 1) ? skr_tex_flags_input_attachment : 0);
	skr_tex_create(app->depth_format, depth_flags, no_sampler, (skr_vec3i_t){render_w, render_h, 1}, app->msaa, 1, NULL, &app->depth_buffer);

	if (enable_offscreen || needs_upscale) {
		// Offscreen target at render scale. Readable so the upscale blit can sample it.
		// Also writeable + input_attachment for use as resolve intermediate in multi-subpass.
		skr_tex_create(app->offscreen_format,
			skr_tex_flags_readable | skr_tex_flags_writeable | skr_tex_flags_input_attachment
				| (enable_offscreen ? skr_tex_flags_compute : 0),
			linear_clamp,
			(skr_vec3i_t){render_w, render_h, 1}, 1, 1, NULL, &app->scene_color);
		// When upscaling with postfx + MSAA, scene_color is the resolve intermediate
		// and we need a separate readable target for the postfx output → upscale source.
		if (needs_upscale) {
			skr_tex_create(app->offscreen_format,
				skr_tex_flags_readable | skr_tex_flags_writeable,
				linear_clamp,
				(skr_vec3i_t){render_w, render_h, 1}, 1, 1, NULL, &app->upscale_target);
		}
	} else if (app->msaa > 1) {
		// Intermediate color target for auto-resolve + postfx mode.
		// Data stays in tile memory (transient) — postfx reads as input, writes to render_target.
		skr_tex_create(render_target->format,
			skr_tex_flags_writeable | skr_tex_flags_input_attachment | skr_tex_flags_in_tile_msaa,
			no_sampler,
			(skr_vec3i_t){render_w, render_h, 1}, 1, 1, NULL, &app->scene_color);
	}

	app->current_width  = width;
	app->current_height = height;

	su_log(su_log_info, "Render target: %dx%d (render %dx%d @ %.0f%%) @ %dx MSAA, %s / %s",
		width, height, render_w, render_h, app->render_scale * 100.0f,
		app->msaa, _tex_fmt_name(skr_tex_get_format(&app->color_msaa)), _tex_fmt_name(app->depth_format));
}

static void _destroy_render_targets(app_t* app) {
	skr_tex_destroy(&app->color_msaa);
	skr_tex_destroy(&app->depth_buffer);
	skr_tex_destroy(&app->upscale_target);
	bool needs_upscale = (app->render_scale < 1.0f) || (app->viewport_scale < 1.0f);
	if (enable_offscreen || needs_upscale || app->msaa > 1) {
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
	uint64_t start_ns  = ska_time_get_elapsed_ns();
	app->scene_current = scene_create(app->scene_types[new_index]);
	uint64_t elapsed_ns = ska_time_get_elapsed_ns() - start_ns;

	su_log(su_log_info, "Switched to scene: %s (%.2f ms)", scene_get_name(app->scene_types[new_index]), (double)elapsed_ns / 1000000.0);
}

app_t* app_create(int32_t start_scene) {
	app_t* app = calloc(1, sizeof(app_t));
	if (!app) return NULL;

	int32_t max_msaa     = skr_get_max_msaa_samples();
	app->msaa            = 4 < max_msaa ? 4 : max_msaa;
	app->gpu_time_min_ms = 1e10f;
	app->offscreen_format = skr_tex_fmt_rgba32_srgb;//skr_tex_fmt_bgra32_srgb;

	app->render_scale          = 1.0f;
	app->render_scale_previous = 1.0f;
	app->viewport_scale        = 1.0f;

	skr_log(skr_log_info, "Max MSAA samples: %d (using %d)", max_msaa, app->msaa);

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
	app->depth_format = skr_tex_fmt_depth16;


	// Create shared render list
	skr_render_list_create(&app->render_list);

	// Load all PostFX / resolve shaders upfront so dropdown can switch at runtime
	app->postfx_shader = su_shader_load("shaders/postfx_invert.hlsl.sks", "postfx_invert");
	if (skr_shader_is_valid(&app->postfx_shader)) {
		skr_material_create((skr_material_info_t){
			.shader     = &app->postfx_shader,
			.cull       = skr_cull_none,
			.depth_test = skr_compare_always,
		}, &app->postfx_mat);
	}
	app->resolve_shader = su_shader_load("shaders/msaa_resolve.hlsl.sks", "msaa_resolve");
	if (skr_shader_is_valid(&app->resolve_shader)) {
		skr_material_create((skr_material_info_t){
			.shader     = &app->resolve_shader,
			.cull       = skr_cull_none,
			.depth_test = skr_compare_always,
		}, &app->resolve_mat);
	}
	app->wide_resolve_shader = su_shader_load("shaders/msaa_resolve_wide.hlsl.sks", "msaa_resolve_wide");
	if (skr_shader_is_valid(&app->wide_resolve_shader)) {
		skr_material_create((skr_material_info_t){
			.shader     = &app->wide_resolve_shader,
			.cull       = skr_cull_none,
			.depth_test = skr_compare_always,
		}, &app->wide_resolve_mat);
	}
	app->oled_resolve_shader = su_shader_load("shaders/msaa_resolve_oled.hlsl.sks", "msaa_resolve_oled");
	if (skr_shader_is_valid(&app->oled_resolve_shader)) {
		skr_material_create((skr_material_info_t){
			.shader     = &app->oled_resolve_shader,
			.cull       = skr_cull_none,
			.depth_test = skr_compare_always,
		}, &app->oled_resolve_mat);
	}
	app->upscale_shader = su_shader_load("shaders/upscale.hlsl.sks", "upscale");
	if (skr_shader_is_valid(&app->upscale_shader)) {
		skr_material_create((skr_material_info_t){
			.shader     = &app->upscale_shader,
			.cull       = skr_cull_none,
			.depth_test = skr_compare_always,
		}, &app->upscale_mat);
	}

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
	app->scene_types[14] = &scene_tex_compress_vtable;
	app->scene_types[15] = &scene_stars_vtable;
	app->scene_types[16] = &scene_yuv_test_vtable;
	app->scene_types[17] = &scene_gi_vtable;
	app->scene_types[18] = &scene_pbr_vtable;
	app->scene_count = 19;
#ifdef SKR_HAS_VIDEO
	app->scene_types[app->scene_count++] = &scene_video_vtable;
#endif

	su_log(su_log_info, "Application created successfully!");
	su_log(su_log_info, "Available scenes: %d (use arrow keys to switch)", app->scene_count);

	// Start with the requested scene (default to 0 if out of range or -1)
	app->scene_index = -1;
	int32_t initial_scene = (start_scene >= 0 && start_scene < app->scene_count) ? start_scene : 0;
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

	// Destroy postfx (always destroy if created, regardless of runtime toggle)
	if (skr_shader_is_valid(&app->postfx_shader)) {
		skr_material_destroy(&app->postfx_mat);
		skr_shader_destroy(&app->postfx_shader);
	}
	if (skr_shader_is_valid(&app->resolve_shader)) {
		skr_material_destroy(&app->resolve_mat);
		skr_shader_destroy(&app->resolve_shader);
	}
	if (skr_shader_is_valid(&app->wide_resolve_shader)) {
		skr_material_destroy(&app->wide_resolve_mat);
		skr_shader_destroy(&app->wide_resolve_shader);
	}
	if (skr_shader_is_valid(&app->oled_resolve_shader)) {
		skr_material_destroy(&app->oled_resolve_mat);
		skr_shader_destroy(&app->oled_resolve_shader);
	}
	if (skr_shader_is_valid(&app->upscale_shader)) {
		skr_material_destroy(&app->upscale_mat);
		skr_shader_destroy(&app->upscale_shader);
	}

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

int32_t app_scene_index(app_t* app) {
	return app ? app->scene_index : 0;
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

	// Force render target recreation when settings change that affect texture flags/sizes
	bool needs_upscale   = (app->render_scale < 1.0f) || (app->viewport_scale < 1.0f);
	bool mode_changed    = (app->resolve_mode != app->resolve_mode_previous);
	bool scale_changed   = (app->render_scale != app->render_scale_previous);
	bool upscale_changed = needs_upscale != skr_tex_is_valid(&app->upscale_target);
	if ((mode_changed || scale_changed || upscale_changed) && app->current_width != 0) {
		_destroy_render_targets(app);
		_create_render_targets(app, width, height, render_target);
	}
	app->resolve_mode_previous = app->resolve_mode;
	app->render_scale_previous = app->render_scale;

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

	// Compute effective viewport from render target size × viewport_scale
	int32_t render_w      = app->color_msaa.size.x;
	int32_t render_h      = app->color_msaa.size.y;
	int32_t view_w        = (int32_t)(render_w * app->viewport_scale) & ~1;
	int32_t view_h        = (int32_t)(render_h * app->viewport_scale) & ~1;
	if (view_w < 2) view_w = 2;
	if (view_h < 2) view_h = 2;

	// Calculate view-projection matrix using viewport dimensions for correct aspect ratio
	float    aspect     = (float)view_w / (float)view_h;
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

	// Setup application system buffer — screen_size reflects viewport dimensions
	su_system_buffer_t sys_buffer = {0};
	sys_buffer.view_count = 1;
	sys_buffer.view          [0] = view;
	sys_buffer.projection    [0] = projection;
	sys_buffer.viewproj      [0] = float4x4_mul   (projection, view);
	sys_buffer.view_inv      [0] = float4x4_invert(view);
	sys_buffer.projection_inv[0] = float4x4_invert(projection);
	sys_buffer.cam_pos       [0] = (float4){cam_position.x, cam_position.y, cam_position.z, 0.0f};
	sys_buffer.cam_dir       [0] = (float4){cam_forward.x,  cam_forward.y,  cam_forward.z,  0.0f};
	sys_buffer.screen_size       = (float4){(float)view_w, (float)view_h, 1.0f / view_w, 1.0f / view_h};

	// Let the scene populate the render list (and optionally do its own render passes)
	scene_render(vtable, app->scene_current, view_w, view_h, &app->render_list, &sys_buffer);

	// Prepare ImGui mesh data OUTSIDE render pass (uploads via vkCmdCopyBuffer)
	ImGui_ImplSkRenderer_PrepareDrawData();

	// Determine render targets based on resolve mode
	//
	// Mode              | color_target    | resolve_target  | postfx_output
	// ------------------|-----------------|-----------------|---------------
	// normal, no MSAA   | render_target*  | NULL            | NULL
	// normal, MSAA      | color_msaa      | render_target*  | NULL
	// auto_postfx, !MSAA| scene_color     | NULL            | final_output
	// auto_postfx, MSAA | color_msaa      | scene_color     | final_output
	// manual_resolve    | color_msaa      | final_output    | NULL
	// wide_kernel       | color_msaa      | NULL            | NULL (separate pass)
	// oled_subpixel     | color_msaa      | NULL            | NULL (separate pass)
	//
	// *render_target = scene_color when enable_offscreen or needs_upscale
	// final_output   = render_target when !needs_upscale, else upscale_src
	bool use_postfx         = (app->resolve_mode == resolve_mode_auto_postfx)   && skr_material_is_valid(&app->postfx_mat);
	bool use_manual_resolve = (app->resolve_mode == resolve_mode_manual_postfx) && app->msaa > 1 && skr_material_is_valid(&app->resolve_mat);
	bool use_wide_kernel    = (app->resolve_mode == resolve_mode_wide_kernel)   && app->msaa > 1 && skr_material_is_valid(&app->wide_resolve_mat);
	bool use_oled_subpixel  = (app->resolve_mode == resolve_mode_oled_subpixel) && app->msaa > 1 && skr_material_is_valid(&app->oled_resolve_mat);

	// When upscaling, the scene writes to an offscreen target, then a blit upscales to swapchain.
	// When postfx + MSAA + upscale: scene_color is the resolve intermediate, so postfx output
	// goes to upscale_target (a separate readable texture) to avoid aliasing in the framebuffer.
	skr_tex_t* upscale_src = NULL;  // What the upscale blit reads from (NULL = no upscale)
	if (needs_upscale)
		upscale_src = (use_postfx && app->msaa > 1) ? &app->upscale_target : &app->scene_color;

	skr_tex_t* final_output = needs_upscale ? upscale_src : render_target;

	skr_tex_t* color_target;
	skr_tex_t* resolve_target;
	skr_tex_t* postfx_output = NULL;

	if (use_wide_kernel || use_oled_subpixel) {
		// Wide-kernel / OLED subpixel: geometry writes MSAA color (no auto-resolve).
		// Separate pass reads it as Texture2DMS and writes to final_output.
		color_target   = &app->color_msaa;
		resolve_target = NULL;
	} else if (use_manual_resolve) {
		// Manual resolve: geometry writes MSAA color, resolve subpass reads it as
		// SubpassInputMS and writes directly to final_output (no intermediate).
		color_target   = &app->color_msaa;
		resolve_target = final_output;
	} else if (use_postfx) {
		// Auto postfx: geometry writes to scene_color (or resolves there for MSAA),
		// postfx reads as input attachment (tile-local), writes to final_output.
		color_target   = (app->msaa > 1) ? &app->color_msaa : &app->scene_color;
		resolve_target = (app->msaa > 1) ? &app->scene_color : NULL;
		postfx_output  = final_output;
	} else {
		color_target   = (app->msaa > 1) ? &app->color_msaa : ((enable_offscreen || needs_upscale) ? &app->scene_color : render_target);
		resolve_target = (app->msaa > 1) ? ((enable_offscreen || needs_upscale) ? &app->scene_color : render_target) : NULL;
	}

	// Scene geometry via deferred pass (handles multi-view transparently)
	skr_pass_t pass = {
		.color         = color_target,
		.depth         = &app->depth_buffer,
		.resolve       = resolve_target,
		.postfx_output = postfx_output,
		.clear         = skr_clear_all,
		.clear_color   = {0, 0, 0, 0},
		.clear_depth   = 1.0f,
		.viewport      = {0, 0, (float)view_w, (float)view_h},
		.scissor       = {0, 0, view_w, view_h},
		.view_count       = sys_buffer.view_count,
		.views_correlated = true,
	};
	skr_pass_add_draw(&pass, &app->render_list, &sys_buffer, sizeof(su_system_buffer_t));
	if (use_manual_resolve)
		skr_pass_add_resolve(&pass, &app->resolve_mat);
	if (use_postfx)
		skr_pass_add_postfx(&pass, &app->postfx_mat);
	skr_pass_submit(&pass);
	skr_render_list_clear(&app->render_list);

	// Wide-kernel resolve: separate pass reads MSAA color as Texture2DMS
	if (use_wide_kernel) {
		skr_material_set_tex(&app->wide_resolve_mat, "msaa_color", &app->color_msaa);
		skr_renderer_blit(&app->wide_resolve_mat, final_output, (skr_recti_t){0, 0, view_w, view_h});
	}
	// OLED subpixel resolve: per-channel weights based on physical subpixel layout
	if (use_oled_subpixel) {
		skr_material_set_tex(&app->oled_resolve_mat, "msaa_color", &app->color_msaa);
		skr_renderer_blit(&app->oled_resolve_mat, final_output, (skr_recti_t){0, 0, view_w, view_h});
	}

	// Post-processing (operates at render resolution, before upscale)
	if (enable_offscreen && enable_bloom) {
		bloom_apply(&app->scene_color, final_output, 1.0f, 4.0f, 0.75f);
	}

	// Upscale: offscreen → swapchain at full window resolution
	if (upscale_src && skr_material_is_valid(&app->upscale_mat)) {
		float uv_scale[4] = { (float)view_w / (float)render_w, (float)view_h / (float)render_h, 0, 0 };
		skr_material_set_params(&app->upscale_mat, uv_scale, sizeof(uv_scale));
		skr_material_set_tex   (&app->upscale_mat, "src_tex", upscale_src);
		skr_renderer_blit(&app->upscale_mat, render_target, (skr_recti_t){0, 0, width, height});
	}

	// ImGui: always renders to swapchain at native resolution (sharp UI)
	skr_tex_t* imgui_target = upscale_src ? render_target
		: (use_postfx || use_manual_resolve || use_wide_kernel || use_oled_subpixel) ? render_target
		: (resolve_target ? resolve_target : color_target);
	skr_renderer_begin_pass(imgui_target, NULL, NULL, skr_clear_none, (skr_vec4_t){0}, 1.0f, 0, 0x1, 0x1);
	skr_renderer_set_viewport((skr_rect_t ){0, 0, (float)width, (float)height});
	skr_renderer_set_scissor ((skr_recti_t){0, 0, width, height});
	ImGui_ImplSkRenderer_RenderDrawData(width, height);
	skr_renderer_end_pass();
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
	float arrow_size = igGetFrameHeight() * 3.0f;
	if (igArrowButtonEx("##left",  ImGuiDir_Left,  (ImVec2){arrow_size, arrow_size}, 0)) { _switch_scene(app, (app->scene_index - 1 + app->scene_count) % app->scene_count);}
	igSameLine(0.0f, 5.0f);
	if (igArrowButtonEx("##right", ImGuiDir_Right, (ImVec2){arrow_size, arrow_size}, 0)) { _switch_scene(app, (app->scene_index + 1) % app->scene_count); }

	igSeparator();

	// Scene-specific UI controls (re-fetch vtable in case scene changed above)
	scene_render_ui(app->scene_types[app->scene_index], app->scene_current);

	igSeparator();

	// Show render info and controls
	igText("Window: %d x %d", width, height);
	{
		float rs = app->render_scale   * 100.0f;
		float vs = app->viewport_scale * 100.0f;
		if (igSliderFloat("Render Scale",   &rs, 25.0f, 100.0f, "%.0f%%", 0)) app->render_scale   = rs / 100.0f;
		if (igSliderFloat("Viewport Scale", &vs, 25.0f, 100.0f, "%.0f%%", 0)) app->viewport_scale = vs / 100.0f;
	}
	{
		int32_t rw = (int32_t)(width  * app->render_scale) & ~1;
		int32_t rh = (int32_t)(height * app->render_scale) & ~1;
		int32_t vw = (int32_t)(rw * app->viewport_scale) & ~1;
		int32_t vh = (int32_t)(rh * app->viewport_scale) & ~1;
		igText("Render: %d x %d, Viewport: %d x %d", rw, rh, vw, vh);
	}
	igText("MSAA: %dx", app->msaa);
	if (app->msaa > 1)
		igCombo_Str_arr("Resolve Mode", &app->resolve_mode, resolve_mode_names, resolve_mode_max, 0);

	float gpu_ms   = skr_renderer_get_gpu_time_us() / 1000.0f;
	float cpu_ms   = skr_renderer_get_cpu_time_us() / 1000.0f;
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
	app->cpu_time_history  [app->history_index] = cpu_ms > 0.0f ? cpu_ms : app->cpu_time_history[(app->history_index + FRAME_HISTORY_SIZE - 1) % FRAME_HISTORY_SIZE];
	app->history_index = (app->history_index + 1) % FRAME_HISTORY_SIZE;

	// Exponential moving average for readable display (smoothing factor ~0.1)
	const float ema_a = 0.02f;
	app->frame_ema = app->frame_ema > 0.0f ? app->frame_ema + ema_a * (frame_ms - app->frame_ema) : frame_ms;
	app->gpu_ema   = app->gpu_ema   > 0.0f ? app->gpu_ema   + ema_a * (app->gpu_time_history[(app->history_index + FRAME_HISTORY_SIZE - 1) % FRAME_HISTORY_SIZE] - app->gpu_ema) : gpu_ms;
	app->cpu_ema   = app->cpu_ema   > 0.0f ? app->cpu_ema   + ema_a * (app->cpu_time_history[(app->history_index + FRAME_HISTORY_SIZE - 1) % FRAME_HISTORY_SIZE] - app->cpu_ema) : cpu_ms;

	igText("Frame Time: %.3f ms (%.1f FPS)", app->frame_ema, 1000.0f / app->frame_ema);
	igText("CPU Time: %.3f ms", app->cpu_ema);
	igText("GPU Time: %.3f ms", app->gpu_ema);

	// Graph display ranges
	const float frame_graph_min = 6.0f;
	const float frame_graph_max = 10.0f;
	const float cpu_graph_min   = 0.0f;
	const float cpu_graph_max   = 3.0f;
	const float gpu_graph_min   = 0.0f;
	const float gpu_graph_max   = 3.0f;

	// Get available width for full-width plots
	ImVec2 content_region;
	igGetContentRegionAvail(&content_region);
	float plot_width = content_region.x;

	char frame_overlay[32], cpu_overlay[32], gpu_overlay[32];
	snprintf(frame_overlay, sizeof(frame_overlay), "Frame: %.1f ms", app->frame_ema);
	snprintf(cpu_overlay,   sizeof(cpu_overlay),   "CPU: %.1f ms",   app->cpu_ema);
	snprintf(gpu_overlay,   sizeof(gpu_overlay),   "GPU: %.1f ms",   app->gpu_ema);

	// Plot frame time - using values_offset for circular buffer
	igPlotLines_FloatPtr("##frame_graph", app->frame_time_history, FRAME_HISTORY_SIZE,
		app->history_index, frame_overlay, frame_graph_min, frame_graph_max, (ImVec2){plot_width, 60}, sizeof(float));

	igPlotLines_FloatPtr("##cpu_graph", app->cpu_time_history, FRAME_HISTORY_SIZE,
		app->history_index, cpu_overlay, cpu_graph_min, cpu_graph_max, (ImVec2){plot_width, 60}, sizeof(float));

	igPlotLines_FloatPtr("##gpu_graph", app->gpu_time_history, FRAME_HISTORY_SIZE,
		app->history_index, gpu_overlay, gpu_graph_min, gpu_graph_max, (ImVec2){plot_width, 60}, sizeof(float));

	igEnd();
}

