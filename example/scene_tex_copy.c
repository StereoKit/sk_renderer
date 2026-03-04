// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

// Test scene for texture copy and readback functionality
// - Renders spheres with varying vertex resolutions to individual icon textures
// - Uses skr_tex_create_copy to resolve MSAA and create mipped icon textures
// - Displays icons as quads moving towards/away from camera to observe mip-mapping
// - Uses skr_tex_readback to read back texture data and save to file

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

#define ICON_SIZE 256
#define ICON_MSAA 4
#define SPHERE_COUNT 5

typedef struct {
	scene_t        base;

	// Meshes - varying resolutions
	skr_mesh_t     spheres[SPHERE_COUNT];

	// Shaders
	skr_shader_t   shader;       // Lit shader for rendering spheres
	skr_shader_t   icon_shader;  // Unlit shader for displaying icons
	skr_material_t sphere_materials[SPHERE_COUNT];

	// Per-sphere icon render targets (MSAA) - shared across all spheres
	skr_tex_t      icon_msaa;
	skr_tex_t      icon_depth;

	// Resolved icons with mips (one per sphere)
	skr_tex_t      icons[SPHERE_COUNT];
	skr_material_t icon_materials[SPHERE_COUNT];
	bool           icons_created;

	// Quad mesh for displaying icons
	skr_mesh_t     quad_mesh;

	// Readback state (for first icon)
	bool                 readback_pending;
	bool                 readback_saved;
	skr_tex_readback_t   readback;

	float time;
} scene_tex_copy_t;

// Save RGBA data to PPM file (simple format, no external deps)
static bool _save_ppm(const char* filename, const uint8_t* data, int32_t width, int32_t height) {
	FILE* f = fopen(filename, "wb");
	if (!f) return false;

	// PPM header (P6 = binary RGB)
	fprintf(f, "P6\n%d %d\n255\n", width, height);

	// Write RGB data (skip alpha)
	for (int32_t i = 0; i < width * height; i++) {
		fputc(data[i * 4 + 0], f);  // R
		fputc(data[i * 4 + 1], f);  // G
		fputc(data[i * 4 + 2], f);  // B
	}

	fclose(f);
	return true;
}

static scene_t* _scene_tex_copy_create(void) {
	scene_tex_copy_t* scene = calloc(1, sizeof(scene_tex_copy_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_tex_copy_t);
	scene->time      = 0.0f;

	// Create spheres with varying vertex resolutions and colors
	int32_t    segments[SPHERE_COUNT] = {6, 10, 16, 24, 32};
	int32_t    rings   [SPHERE_COUNT] = {4, 6,  10, 16, 24};
	skr_vec4_t colors  [SPHERE_COUNT] = {
		{1.0f, 0.2f, 0.2f, 1.0f},  // Red - low poly
		{0.2f, 1.0f, 0.2f, 1.0f},  // Green
		{0.2f, 0.2f, 1.0f, 1.0f},  // Blue
		{1.0f, 1.0f, 0.2f, 1.0f},  // Yellow
		{1.0f, 0.2f, 1.0f, 1.0f},  // Magenta - high poly
	};

	for (int32_t i = 0; i < SPHERE_COUNT; i++) {
		scene->spheres[i] = su_mesh_create_sphere(segments[i], rings[i], 0.8f, colors[i]);
	}

	// Load shaders
	scene->shader      = su_shader_load("shaders/test.hlsl.sks",  "sphere_shader");
	scene->icon_shader = su_shader_load("shaders/unlit.hlsl.sks", "icon_shader"  );

	// Create materials for each sphere
	for (int32_t i = 0; i < SPHERE_COUNT; i++) {
		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.write_mask = skr_write_default,
			.depth_test = skr_compare_less,
		}, &scene->sphere_materials[i]);
	}

	// Create shared MSAA render target for icon rendering (readable for copy source)
	skr_tex_create(
		skr_tex_fmt_rgba32_srgb,
		skr_tex_flags_writeable | skr_tex_flags_readable,
		su_sampler_linear_clamp,
		(skr_vec3i_t){ICON_SIZE, ICON_SIZE, 1},
		ICON_MSAA, 1, NULL, &scene->icon_msaa);
	skr_tex_set_name(&scene->icon_msaa, "icon_msaa");

	// Create depth buffer for icon rendering
	skr_tex_create(
		skr_tex_fmt_depth16,
		skr_tex_flags_writeable,
		(skr_tex_sampler_t){0},
		(skr_vec3i_t){ICON_SIZE, ICON_SIZE, 1},
		ICON_MSAA, 1, NULL, &scene->icon_depth);
	skr_tex_set_name(&scene->icon_depth, "icon_depth");

	// Create quad mesh for displaying icons (normal towards +Z to face camera at +Z)
	scene->quad_mesh = su_mesh_create_quad(1.5f, 1.5f, (skr_vec3_t){0, 0, 1}, false, (skr_vec4_t){1, 1, 1, 1});
	skr_mesh_set_name(&scene->quad_mesh, "icon_quad");

	su_log(su_log_info, "scene_tex_copy: Created with %d spheres", SPHERE_COUNT);

	return &scene->base;
}

static void _scene_tex_copy_destroy(scene_t* s) {
	scene_tex_copy_t* scene = (scene_tex_copy_t*)s;

	// Clean up pending readback
	if (scene->readback_pending) {
		skr_future_wait(&scene->readback.future);
		skr_tex_readback_destroy(&scene->readback);
	}

	// Destroy icon textures and materials
	if (scene->icons_created) {
		for (int32_t i = 0; i < SPHERE_COUNT; i++) {
			skr_tex_destroy(&scene->icons[i]);
			skr_material_destroy(&scene->icon_materials[i]);
		}
	}

	// Destroy render targets
	skr_tex_destroy(&scene->icon_msaa);
	skr_tex_destroy(&scene->icon_depth);

	// Destroy sphere materials
	for (int32_t i = 0; i < SPHERE_COUNT; i++) {
		skr_material_destroy(&scene->sphere_materials[i]);
	}

	// Destroy meshes
	for (int32_t i = 0; i < SPHERE_COUNT; i++) {
		skr_mesh_destroy(&scene->spheres[i]);
	}
	skr_mesh_destroy(&scene->quad_mesh);

	// Destroy shaders
	skr_shader_destroy(&scene->shader);
	skr_shader_destroy(&scene->icon_shader);

	free(scene);
}

static void _scene_tex_copy_update(scene_t* s, float delta_time) {
	scene_tex_copy_t* scene = (scene_tex_copy_t*)s;
	scene->time += delta_time;

	// Check if readback is complete
	if (scene->readback_pending && skr_future_check(&scene->readback.future)) {
		// Save to file
		if (_save_ppm("icon_readback.ppm", (const uint8_t*)scene->readback.data, ICON_SIZE, ICON_SIZE)) {
			su_log(su_log_info, "scene_tex_copy: Saved icon to icon_readback.ppm (%u bytes)", scene->readback.size);
			scene->readback_saved = true;
		} else {
			su_log(su_log_warning, "scene_tex_copy: Failed to save icon_readback.ppm");
		}

		// Clean up
		skr_tex_readback_destroy(&scene->readback);
		scene->readback_pending = false;
	}
}

static void _render_sphere_to_icon(scene_tex_copy_t* scene, int32_t sphere_idx, su_system_buffer_t* ref_system_buffer) {
	// Create temporary render list for icon rendering
	skr_render_list_t icon_list;
	skr_render_list_create(&icon_list);

	// Set up camera for icon (looking at origin from Z axis)
	su_system_buffer_t icon_system = *ref_system_buffer;
	icon_system.view_count = 1;  // Single view, not stereo
	float4x4 icon_view = float4x4_lookat(
		(float3){0, 0, 2.5f},   // Camera position
		(float3){0, 0, 0},      // Look at origin
		(float3){0, 1, 0});     // Up vector
	float4x4 icon_proj = float4x4_perspective(0.8f, 1.0f, 0.1f, 100.0f);
	// Note: float4x4_perspective already includes Vulkan Y-flip

	float4x4 icon_viewproj = float4x4_mul(icon_proj, icon_view);  // projection * view
	icon_system.viewproj[0]   = icon_viewproj;
	icon_system.view[0]       = icon_view;
	icon_system.projection[0] = icon_proj;

	// Render sphere at origin, slowly rotating
	float4x4 world = float4x4_trs(
		(float3){0, 0, 0},
		float4_quat_from_euler((float3){scene->time * 0.5f, scene->time * 0.3f, 0}),
		(float3){1, 1, 1});

	skr_render_list_add(&icon_list, &scene->spheres[sphere_idx], &scene->sphere_materials[sphere_idx],
		&world, sizeof(float4x4), 1);

	// Render to MSAA target
	skr_pass_t icon_pass = {
		.color       = &scene->icon_msaa,
		.depth       = &scene->icon_depth,
		.clear       = skr_clear_all,
		.clear_color = {0.1f, 0.1f, 0.15f, 1.0f},
		.clear_depth = 1.0f,
		.viewport    = {0, 0, (float)ICON_SIZE, (float)ICON_SIZE},
		.scissor     = {0, 0, ICON_SIZE, ICON_SIZE},
	};
	skr_pass_add_draw(&icon_pass, &icon_list, &icon_system, sizeof(su_system_buffer_t), NULL);
	skr_pass_submit  (&icon_pass);

	skr_render_list_destroy(&icon_list);
}

static void _scene_tex_copy_render(scene_t* s, int32_t width, int32_t height,
                                    skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_tex_copy_t* scene = (scene_tex_copy_t*)s;

	// First pass: Create icon textures for each sphere (once only)
	if (!scene->icons_created) {
		for (int32_t i = 0; i < SPHERE_COUNT; i++) {
			// Render sphere to MSAA target
			_render_sphere_to_icon(scene, i, ref_system_buffer);

			// Resolve MSAA and create mipped texture
			skr_err_ err = skr_tex_create_copy(
				&scene->icon_msaa,
				skr_tex_fmt_none,
				skr_tex_flags_readable | skr_tex_flags_gen_mips,
				1,  // Resolve to 1 sample
				&scene->icons[i]);

			if (err == skr_err_success) {
				char name[64];
				snprintf(name, sizeof(name), "icon_%d", i);
				skr_tex_set_name(&scene->icons[i], name);

				// Generate mips for the resolved texture
				skr_tex_generate_mips(&scene->icons[i], NULL);

				// Create material for displaying this icon (unlit, with V-flip)
				skr_material_create((skr_material_info_t){
					.shader     = &scene->icon_shader,
					.write_mask = skr_write_default,
					.depth_test = skr_compare_less,
				}, &scene->icon_materials[i]);
				skr_material_set_tex(&scene->icon_materials[i], "tex", &scene->icons[i]);
			} else {
				su_log(su_log_warning, "scene_tex_copy: skr_tex_create_copy failed for icon %d: %d", i, err);
			}
		}

		scene->icons_created = true;
		su_log(su_log_info, "scene_tex_copy: Created %d icon textures with mips", SPHERE_COUNT);

		// Start async readback of first icon
		skr_err_ err = skr_tex_readback(&scene->icons[0], 0, 0, &scene->readback);
		if (err == skr_err_success) {
			scene->readback_pending = true;
			su_log(su_log_info, "scene_tex_copy: Started async readback of first icon");
		} else {
			su_log(su_log_warning, "scene_tex_copy: skr_tex_readback failed: %d", err);
		}
	}

	// Main pass: Display icons as quads moving towards/away from camera
	if (scene->icons_created) {
		for (int32_t i = 0; i < SPHERE_COUNT; i++) {
			// Arrange icons in a horizontal row
			float x = (i - (SPHERE_COUNT - 1) * 0.5f) * 2.0f;

			// Each icon oscillates in Z at different phases to observe mip-mapping
			// Range from z=0 (close, high detail) to z=-8 (far, lower mips)
			float phase = scene->time * 0.8f + i * 1.2f;
			float z = -16.0f + sinf(phase) * 20.0f;

			float4x4 world = float4x4_trs(
				(float3){x, 0, z},
				float4_quat_from_euler((float3){0, 0, 0}),
				(float3){1, 1, 1});

			skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->icon_materials[i],
				&world, sizeof(float4x4), 1);
		}
	}
}

static void _scene_tex_copy_render_ui(scene_t* s) {
	scene_tex_copy_t* scene = (scene_tex_copy_t*)s;

	igText("Texture Copy & Mip-Mapping Test");
	igSeparator();
	igText("Each icon is a sphere rendered to texture");
	igText("Icons move towards/away to show mip levels");
	igSeparator();

	if (scene->icons_created) {
		igTextColored((ImVec4){0.4f, 1.0f, 0.4f, 1.0f}, "%d icons created!", SPHERE_COUNT);
		igText("  Size: %dx%d", ICON_SIZE, ICON_SIZE);
		igText("  MSAA: %dx (resolved to 1x)", ICON_MSAA);
		igText("  Mips: auto-generated");
	} else {
		igTextColored((ImVec4){1.0f, 1.0f, 0.4f, 1.0f}, "Creating icons...");
	}

	igSeparator();

	if (scene->readback_pending) {
		igTextColored((ImVec4){1.0f, 1.0f, 0.4f, 1.0f}, "Readback pending...");
	} else if (scene->readback_saved) {
		igTextColored((ImVec4){0.4f, 1.0f, 0.4f, 1.0f}, "Saved: icon_readback.ppm");
	}

	igSeparator();
	igText("Sphere resolutions: 6, 10, 16, 24, 32 segments");
	igText("Watch the texture detail change as icons");
	igText("move closer (sharp) and farther (blurry)");
}

const scene_vtable_t scene_tex_copy_vtable = {
	.name       = "Texture Copy Test",
	.create     = _scene_tex_copy_create,
	.destroy    = _scene_tex_copy_destroy,
	.update     = _scene_tex_copy_update,
	.render     = _scene_tex_copy_render,
	.get_camera = NULL,
	.render_ui  = _scene_tex_copy_render_ui,
};
