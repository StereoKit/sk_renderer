// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "tools/tex_compress.h"
#include "tools/tex_compress_gpu.h"
#include "tools/compress/astc_io.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>

#include <sk_app.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// GPU Texture Compression Demo
// Demonstrates runtime texture compression with BC1 (desktop) or ETC2 (mobile) fallback

typedef enum {
	compress_fmt_none,
	compress_fmt_bc1,
	compress_fmt_etc2,
	compress_fmt_astc4x4,
	compress_fmt_astc6x6,
} compress_fmt_;

typedef struct {
	scene_t        base;
	skr_mesh_t     quad_mesh;
	skr_shader_t   shader;
	skr_material_t material_original;
	skr_material_t material_compressed;
	skr_tex_t      texture_original;
	skr_tex_t      texture_compressed;
	float          time;

	// Image info
	int32_t        img_width;
	int32_t        img_height;
	int32_t        compressed_size;
	double         compress_time_ms;
	double         gpu_compress_time_ms;

	// Format info
	compress_fmt_  current_format;
	bool           bc1_supported;
	bool           etc2_supported;
	bool           astc6x6_supported;       // Texture sampling supported (drives display)
	bool           astc6x6_validate_only;   // Compute works but display format unsupported (AMD desktop)

	// GPU compression
	bool           use_gpu;
	skr_tex_t      texture_source;

	// File loading UI
	char           file_path[512];
	bool           load_requested;

	// Camera
	float          cam_distance;
} scene_bc1_t;

///////////////////////////////////////////////////////////////////////////////
// Image Loading
///////////////////////////////////////////////////////////////////////////////

static void _load_image(scene_bc1_t* scene, const char* path) {
	// Destroy existing textures if any
	if (skr_tex_is_valid(&scene->texture_original))   { skr_tex_destroy(&scene->texture_original);   scene->texture_original   = (skr_tex_t){0}; }
	if (skr_tex_is_valid(&scene->texture_compressed))  { skr_tex_destroy(&scene->texture_compressed);  scene->texture_compressed  = (skr_tex_t){0}; }
	if (skr_tex_is_valid(&scene->texture_source))      { skr_tex_destroy(&scene->texture_source);      scene->texture_source      = (skr_tex_t){0}; }

	// Load source image
	int32_t      width, height;
	skr_tex_fmt_ format;
	void*        pixels = su_image_load(path, &width, &height, &format, 4);

	if (!pixels) {
		su_log(su_log_warning, "TexCompress: Failed to load image: %s", path);
		scene->img_width  = 0;
		scene->img_height = 0;
		return;
	}

	scene->img_width  = width;
	scene->img_height = height;

	// Create original texture for display
	skr_tex_create(skr_tex_fmt_rgba32_srgb,
		skr_tex_flags_readable,
		su_sampler_linear_clamp,
		(skr_vec3i_t){width, height, 1}, 1, 0,
		&(skr_tex_data_t){.data = pixels, .mip_count = 1, .layer_count = 1},
		&scene->texture_original);
	skr_tex_set_name(&scene->texture_original, "original");

	// Use the format selected in UI (current_format is set by the radio buttons)
	skr_tex_fmt_ tex_fmt  = skr_tex_fmt_none;
	const char*  fmt_name = "none";

	if (scene->current_format == compress_fmt_bc1 && scene->bc1_supported) {
		tex_fmt  = skr_tex_fmt_bc1_rgba_srgb;
		fmt_name = "BC1";
	} else if (scene->current_format == compress_fmt_etc2 && scene->etc2_supported) {
		tex_fmt  = skr_tex_fmt_etc1_rgb_srgb;
		fmt_name = "ETC2";
	} else if (scene->current_format == compress_fmt_astc4x4 && (scene->astc6x6_supported || scene->astc6x6_validate_only)) {
		tex_fmt  = skr_tex_fmt_astc4x4_rgba_srgb;
		fmt_name = "ASTC4x4";
	} else if (scene->current_format == compress_fmt_astc6x6 && (scene->astc6x6_supported || scene->astc6x6_validate_only)) {
		tex_fmt  = skr_tex_fmt_astc6x6_rgba_srgb;
		fmt_name = "ASTC6x6";
	} else {
		scene->compressed_size = 0;
		su_log(su_log_warning, "TexCompress: Selected format not supported!");
		su_image_free(pixels);
		return;
	}

	if (scene->use_gpu) {
		// GPU path: create source texture with mips, compress on GPU
		skr_tex_create(skr_tex_fmt_rgba32_linear,
			skr_tex_flags_readable | skr_tex_flags_gen_mips,
			su_sampler_linear_clamp,
			(skr_vec3i_t){width, height, 1}, 1, 0,
			&(skr_tex_data_t){.data = pixels, .mip_count = 1, .layer_count = 1},
			&scene->texture_source);
		skr_tex_set_name(&scene->texture_source, "source_for_gpu");
		skr_tex_generate_mips(&scene->texture_source, NULL);

		uint64_t start_ns = ska_time_get_elapsed_ns();
		switch (scene->current_format) {
			case compress_fmt_bc1:     scene->texture_compressed = tex_compress_gpu_bc1    (&scene->texture_source, true); break;
			case compress_fmt_etc2:    scene->texture_compressed = tex_compress_gpu_etc2   (&scene->texture_source); break;
			// Skip the displayable-texture path on GPUs that can't sample
			// ASTC (AMD desktop) — it'd spam the validation layer. The
			// auto-save below exercises the encoder via buffer readback.
			case compress_fmt_astc4x4:
				if (scene->astc6x6_supported)
					scene->texture_compressed = tex_compress_gpu_astc4x4(&scene->texture_source);
				break;
			case compress_fmt_astc6x6:
				if (scene->astc6x6_supported)
					scene->texture_compressed = tex_compress_gpu_astc6x6(&scene->texture_source);
				break;
			default: break;
		}
		uint64_t end_ns = ska_time_get_elapsed_ns();

		scene->gpu_compress_time_ms = (end_ns - start_ns) / 1000000.0;
		switch (scene->current_format) {
			case compress_fmt_bc1:     scene->compressed_size = bc1_calc_size      (width, height); break;
			case compress_fmt_etc2:    scene->compressed_size = etc2_rgb8_calc_size(width, height); break;
			case compress_fmt_astc4x4: scene->compressed_size = astc4x4_calc_size  (width, height); break;
			case compress_fmt_astc6x6: scene->compressed_size = astc6x6_calc_size  (width, height); break;
			default:                   scene->compressed_size = 0; break;
		}

		su_log(su_log_info, "GPU %s: Compression took %.3f ms (%.1f MP/s)",
			fmt_name, scene->gpu_compress_time_ms,
			(width * height) / (scene->gpu_compress_time_ms * 1000.0));

		// Auto-save ASTC output for validation against astcenc reference.
		// Uses a dedicated readback-only dispatch so this works even on GPUs
		// that can't sample ASTC (AMD desktop), since no destination texture
		// is involved — just a host-visible compute output buffer.
		// Desktop only; no writable CWD on Android, and the validator is
		// desktop-only anyway.
#if !defined(__ANDROID__)
		int32_t  astc_size = 0;
		uint8_t* astc_data = NULL;
		int32_t  astc_block_w = 0, astc_block_h = 0;
		switch (scene->current_format) {
			case compress_fmt_astc4x4:
				astc_data = tex_compress_gpu_astc4x4_readback(&scene->texture_source, &astc_size);
				astc_block_w = 4; astc_block_h = 4;
				break;
			case compress_fmt_astc6x6:
				astc_data = tex_compress_gpu_astc6x6_readback(&scene->texture_source, &astc_size);
				astc_block_w = 6; astc_block_h = 6;
				break;
			default: break;
		}
		if (astc_data) {
			const char* out_path = "astc_output.astc";
			if (astc_write_file(out_path, width, height, astc_block_w, astc_block_h, astc_data, (size_t)astc_size))
				su_log(su_log_info, "%s: saved %s (%d bytes) from %s", fmt_name, out_path, astc_size, path);
			else
				su_log(su_log_warning, "%s: failed to save %s", fmt_name, out_path);
			free(astc_data);
		} else if (scene->current_format == compress_fmt_astc4x4 ||
		           scene->current_format == compress_fmt_astc6x6) {
			su_log(su_log_warning, "%s: readback dispatch failed", fmt_name);
		}
#endif
	} else {
		// CPU path
		uint8_t* compressed_data = NULL;

		if (scene->current_format == compress_fmt_astc4x4 ||
		    scene->current_format == compress_fmt_astc6x6) {
			// No CPU ASTC encoder yet — force GPU path and bail out of CPU flow.
			su_log(su_log_warning, "TexCompress: ASTC has no CPU encoder, switching to GPU path");
			scene->use_gpu = true;
			su_image_free(pixels);
			_load_image(scene, path);
			return;
		}

		uint64_t start_ns = ska_time_get_elapsed_ns();
		if (scene->current_format == compress_fmt_bc1) {
			compressed_data = bc1_compress(pixels, width, height);
			scene->compressed_size = bc1_calc_size(width, height);
		} else {
			compressed_data = etc2_rgb8_compress(pixels, width, height);
			scene->compressed_size = etc2_rgb8_calc_size(width, height);
		}
		uint64_t end_ns = ska_time_get_elapsed_ns();

		scene->compress_time_ms = (end_ns - start_ns) / 1000000.0;
		su_log(su_log_info, "CPU %s: Compression took %.3f ms (%.1f MP/s)",
			fmt_name, scene->compress_time_ms,
			(width * height) / (scene->compress_time_ms * 1000.0));

		skr_tex_create(tex_fmt,
			skr_tex_flags_readable,
			su_sampler_linear_clamp,
			(skr_vec3i_t){width, height, 1}, 1, 0,
			&(skr_tex_data_t){.data = compressed_data, .mip_count = 1, .layer_count = 1},
			&scene->texture_compressed);
		skr_tex_set_name(&scene->texture_compressed, "compressed");
		free(compressed_data);
	}

	// Update materials. If the compressed texture couldn't be created (AMD
	// ASTC validate-only path), bind the original to both sides so the scene
	// still renders — the user gets clean visual feedback that no GPU preview
	// is available, while the .astc file has already been saved for offline
	// validation.
	skr_material_set_tex(&scene->material_original, "tex", &scene->texture_original);
	skr_material_set_tex(&scene->material_compressed, "tex",
		skr_tex_is_valid(&scene->texture_compressed) ? &scene->texture_compressed : &scene->texture_original);

	su_image_free(pixels);

	su_log(su_log_info, "%s: Compressed %dx%d image (%.1f KB -> %.1f KB, %.1f:1 ratio)",
		fmt_name, width, height,
		(width * height * 4) / 1024.0f,
		scene->compressed_size / 1024.0f,
		(float)(width * height * 4) / scene->compressed_size);
}

///////////////////////////////////////////////////////////////////////////////
// Scene Implementation
///////////////////////////////////////////////////////////////////////////////

static scene_t* _scene_bc1_create(void) {
	scene_bc1_t* scene = calloc(1, sizeof(scene_bc1_t));
	if (!scene) return NULL;

	scene->base.size    = sizeof(scene_bc1_t);
	scene->time         = 0.0f;
	scene->cam_distance = 5.0f;
	scene->use_gpu      = true;

	// Initialize GPU compression
	tex_compress_gpu_init();

	// Check format support
	scene->bc1_supported     = skr_tex_fmt_is_supported(skr_tex_fmt_bc1_rgba_srgb,     skr_tex_flags_readable, 1);
	scene->etc2_supported    = skr_tex_fmt_is_supported(skr_tex_fmt_etc1_rgb_srgb,     skr_tex_flags_readable, 1);
	scene->astc6x6_supported = skr_tex_fmt_is_supported(skr_tex_fmt_astc6x6_rgba_srgb, skr_tex_flags_readable, 1);
	// Even without texture sampling we can still run the compute shader and
	// read the blocks back into a .astc file for offline validation against
	// astcenc — desktop AMD GPUs hit this path.
	scene->astc6x6_validate_only = !scene->astc6x6_supported;

	su_log(su_log_info, "TexCompress: BC1 %s, ETC2 %s, ASTC6x6 %s",
		scene->bc1_supported     ? "supported" : "not supported",
		scene->etc2_supported    ? "supported" : "not supported",
		scene->astc6x6_supported ? "supported" :
		scene->astc6x6_validate_only ? "validate-only (no display)" : "not supported");

	// Default format: prefer BC1, fallback to ETC2, then ASTC6x6
	scene->current_format = scene->astc6x6_validate_only ? compress_fmt_astc6x6 : scene->bc1_supported     ? compress_fmt_bc1     :
	                        scene->etc2_supported    ? compress_fmt_etc2    :
	                        (scene->astc6x6_supported || scene->astc6x6_validate_only) ? compress_fmt_astc6x6 :
	                        compress_fmt_none;

	// Default file path
	strncpy(scene->file_path, "tree.png", sizeof(scene->file_path) - 1);
	//strncpy(scene->file_path, "/home/koujaku/Dropbox/Photos/Wallpapers/zm4nfgq29yi91.jpg", sizeof(scene->file_path) - 1);

	// Create quad mesh for displaying textures (facing +Z)
	scene->quad_mesh = su_mesh_create_quad(2.0f, 2.0f, (skr_vec3_t){0, 0, 1}, false, (skr_vec4_t){1, 1, 1, 1});
	skr_mesh_set_name(&scene->quad_mesh, "tex_compress_quad");

	// Load unlit shader
	scene->shader = su_shader_load("shaders/unlit.hlsl.sks", "tex_compress_shader");

	// Create materials (with alpha blending for transparency support)
	skr_material_create((skr_material_info_t){
		.shader      = &scene->shader,
		.cull        = skr_cull_back,
		.depth_test  = skr_compare_less,
		.blend_state = skr_blend_alpha,
	}, &scene->material_original);

	skr_material_create((skr_material_info_t){
		.shader      = &scene->shader,
		.cull        = skr_cull_back,
		.depth_test  = skr_compare_less,
		.blend_state = skr_blend_alpha,
	}, &scene->material_compressed);

	// Load default image
	_load_image(scene, scene->file_path);

	return (scene_t*)scene;
}

static void _scene_bc1_destroy(scene_t* base) {
	scene_bc1_t* scene = (scene_bc1_t*)base;

	skr_mesh_destroy    (&scene->quad_mesh);
	skr_material_destroy(&scene->material_original);
	skr_material_destroy(&scene->material_compressed);
	skr_shader_destroy  (&scene->shader);
	if (skr_tex_is_valid(&scene->texture_original))   skr_tex_destroy(&scene->texture_original);
	if (skr_tex_is_valid(&scene->texture_compressed)) skr_tex_destroy(&scene->texture_compressed);
	if (skr_tex_is_valid(&scene->texture_source))     skr_tex_destroy(&scene->texture_source);

	tex_compress_gpu_shutdown();
	free(scene);
}

static void _scene_bc1_update(scene_t* base, float delta_time) {
	scene_bc1_t* scene = (scene_bc1_t*)base;
	scene->time += delta_time;

	// Handle load request from UI
	if (scene->load_requested) {
		scene->load_requested = false;
		_load_image(scene, scene->file_path);
	}

	// TEMPORARY profiling: dispatch the currently-selected encoder every
	// frame via the profile-only path. All three variants share a cached
	// output buffer and do just the compute dispatch at mip 0 — no texture
	// creation, no buffer copy, no per-frame allocation. This gives an
	// apples-to-apples shader cost comparison on the perf graph.
	if (scene->use_gpu && skr_tex_is_valid(&scene->texture_source)) {
		switch (scene->current_format) {
			case compress_fmt_bc1:     tex_compress_gpu_bc1_profile    (&scene->texture_source, true); break;
			case compress_fmt_etc2:    tex_compress_gpu_etc2_profile   (&scene->texture_source);       break;
			case compress_fmt_astc4x4: tex_compress_gpu_astc4x4_profile(&scene->texture_source);       break;
			case compress_fmt_astc6x6: tex_compress_gpu_astc6x6_profile(&scene->texture_source);       break;
			default: break;
		}
	}
}

static void _scene_bc1_render(scene_t* base, int32_t width, int32_t height,
                              skr_render_list_t* ref_render_list,
                              su_system_buffer_t* ref_system_buffer) {
	scene_bc1_t* scene = (scene_bc1_t*)base;
	(void)width;
	(void)height;
	(void)ref_system_buffer;

	if (!skr_tex_is_valid(&scene->texture_original)) return;

	// Calculate aspect ratio for proper quad sizing
	float aspect = (float)scene->img_width / (float)scene->img_height;
	float quad_height = 2.0f;
	float quad_width  = quad_height * aspect;

	// Left quad: original texture at x=-1.5
	float4x4 left_world = float4x4_trs(
		(float3){-quad_width * 0.5f - 0.2f, 0.0f, 0.0f},
		(float4){0, 0, 0, 1},
		(float3){quad_width * 0.5f, quad_height * 0.5f, 1.0f});

	// Right quad: compressed at x=+1.5
	float4x4 right_world = float4x4_trs(
		(float3){quad_width * 0.5f + 0.2f, 0.0f, 0.0f},
		(float4){0, 0, 0, 1},
		(float3){quad_width * 0.5f, quad_height * 0.5f, 1.0f});

	skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->material_original,   &left_world,  sizeof(float4x4), 1);
	skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->material_compressed, &right_world, sizeof(float4x4), 1);
}

static bool _scene_bc1_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_bc1_t* scene = (scene_bc1_t*)base;

	ImGuiIO* io = igGetIO();
	if (!io->WantCaptureMouse) {
		if (io->MouseWheel != 0.0f) {
			scene->cam_distance -= io->MouseWheel * 0.5f;
		}
		if (io->MouseDown[0]) {
			scene->cam_distance += io->MouseDelta.y * 0.02f;
		}
		if (scene->cam_distance < 1.0f)  scene->cam_distance = 1.0f;
		if (scene->cam_distance > 20.0f) scene->cam_distance = 20.0f;
	}

	out_camera->position = (float3){0, 0, scene->cam_distance};
	out_camera->target   = (float3){0, 0, 0};
	out_camera->up       = (float3){0, 1, 0};
	return true;
}

// Helper to get just the filename from a path
static const char* _get_filename(const char* path) {
	if (!path) return "(none)";
	const char* last_slash  = strrchr(path, '/');
	const char* last_bslash = strrchr(path, '\\');
	const char* name = path;
	if (last_slash  && last_slash  > name) name = last_slash  + 1;
	if (last_bslash && last_bslash > name) name = last_bslash + 1;
	return name;
}

static void _scene_bc1_render_ui(scene_t* base) {
	scene_bc1_t* scene = (scene_bc1_t*)base;

	igText("Texture Compression");
	igSeparator();

	// Format support status
	igText("Format Support:");
	igTextColored(scene->bc1_supported     ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  BC1:     %s", scene->bc1_supported     ? "Yes" : "No");
	igTextColored(scene->etc2_supported    ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  ETC2:    %s", scene->etc2_supported    ? "Yes" : "No");
	igTextColored(
		scene->astc6x6_supported     ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} :
		scene->astc6x6_validate_only ? (ImVec4){1.0f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  ASTC6x6: %s",
		scene->astc6x6_supported     ? "Yes" :
		scene->astc6x6_validate_only ? "Validate only (no display)" : "No");

	igSeparator();

	// GPU toggle
	if (igCheckbox("GPU Compression", &scene->use_gpu)) {
		scene->load_requested = true;
	}

	// Format selection — dropdown of supported formats only. We build a
	// parallel array of (label, enum_value) so skipped formats don't push
	// later entries onto the wrong index.
	{
		const char* labels [16] = {0};
		int         values [16] = {0};
		int         count       = 0;

		if (scene->bc1_supported) {
			labels[count] = "BC1 (DXT1)";                      values[count++] = compress_fmt_bc1;
		}
		if (scene->etc2_supported) {
			labels[count] = "ETC2 RGB8";                       values[count++] = compress_fmt_etc2;
		}
		if (scene->astc6x6_supported || scene->astc6x6_validate_only) {
			labels[count] = "ASTC 4x4";  values[count++] = compress_fmt_astc4x4;
			labels[count] = "ASTC 6x6";  values[count++] = compress_fmt_astc6x6;
		}

		int selected = 0;
		for (int i = 0; i < count; i++) {
			if (values[i] == (int)scene->current_format) { selected = i; break; }
		}
		if (igCombo_Str_arr("Format", &selected, labels, count, -1)) {
			scene->current_format = (compress_fmt_)values[selected];
			scene->load_requested = true;
		}
	}

	igSeparator();

	// File loading
	igText("File: %s", _get_filename(scene->file_path));

	if (su_file_dialog_supported()) {
		if (igButton("Load Image...", (ImVec2){-1, 0})) {
			char* path = su_file_dialog_open("Select Image", "Image Files", "png;jpg;jpeg;bmp;tga");
			if (path) {
				strncpy(scene->file_path, path, sizeof(scene->file_path) - 1);
				scene->load_requested = true;
				free(path);
			}
		}
	} else {
		// Fallback: text input for platforms without file dialog
		igInputText("##path", scene->file_path, sizeof(scene->file_path), 0, NULL, NULL);
		igSameLine(0, 10);
		if (igButton("Load", (ImVec2){60, 0})) {
			scene->load_requested = true;
		}
	}

	igSeparator();

	// Image info
	if (scene->img_width > 0) {
		const char* fmt_name = "None";
		switch (scene->current_format) {
			case compress_fmt_bc1:     fmt_name = "BC1 (DXT1)"; break;
			case compress_fmt_etc2:    fmt_name = "ETC2 RGB8";  break;
			case compress_fmt_astc4x4: fmt_name = "ASTC 4x4";   break;
			case compress_fmt_astc6x6: fmt_name = "ASTC 6x6";   break;
			default: break;
		}

		igText("Image: %d x %d", scene->img_width, scene->img_height);
		igText("Format: %s", fmt_name);
		int32_t original_size = scene->img_width * scene->img_height * 4;
		igText("Original:   %.1f KB (RGBA8)", original_size / 1024.0f);
		igText("Compressed: %.1f KB", scene->compressed_size / 1024.0f);
		igText("Ratio:      %.1f:1", (float)original_size / scene->compressed_size);

		igSeparator();
		if (scene->use_gpu) {
			double megapix  = (scene->img_width * scene->img_height) / 1000000.0;
			double mp_per_s = megapix / (scene->gpu_compress_time_ms / 1000.0);
			igText("GPU:  %.2f ms (%.1f MP/s)", scene->gpu_compress_time_ms, mp_per_s);
		} else {
			double megapix  = (scene->img_width * scene->img_height) / 1000000.0;
			double mp_per_s = megapix / (scene->compress_time_ms / 1000.0);
			igText("CPU:  %.2f ms (%.1f MP/s)", scene->compress_time_ms, mp_per_s);
		}

		igSeparator();
		igTextColored((ImVec4){0.7f, 0.7f, 0.7f, 1.0f}, "Left: Original  |  Right: %s%s",
			scene->use_gpu ? "GPU " : "", fmt_name);
	} else {
		igTextColored((ImVec4){1.0f, 0.5f, 0.5f, 1.0f}, "No image loaded");
	}
}

const scene_vtable_t scene_tex_compress_vtable = {
	.name       = "Texture Compression",
	.create     = _scene_bc1_create,
	.destroy    = _scene_bc1_destroy,
	.update     = _scene_bc1_update,
	.render     = _scene_bc1_render,
	.get_camera = _scene_bc1_get_camera,
	.render_ui  = _scene_bc1_render_ui,
};
