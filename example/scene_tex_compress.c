// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "tools/tex_compress.h"
#include "tools/tex_compress_gpu.h"
#include "tools/tex_psnr.h"
#include "tools/compress/astc_io.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>

#include <sk_app.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// GPU Texture Compression Demo
// Demonstrates runtime texture compression with BC1 (desktop) or ASTC (mobile),
// plus ASTC 8x8 HDR for float content.

typedef enum {
	compress_fmt_none,
	compress_fmt_bc1,
	compress_fmt_bc7,
	compress_fmt_bc6h,
	compress_fmt_astc4x4,
	compress_fmt_astc6x6,
	compress_fmt_astc8x8hdr,
} compress_fmt_;

typedef struct {
	scene_t        base;
	skr_mesh_t     quad_mesh;
	skr_shader_t   shader;
	skr_material_t material_compare;     // single quad, swipe between left/right textures
	skr_tex_t      texture_original;
	skr_tex_t      texture_compressed;
	float          swipe;                // 0..1, fraction of quad showing original
	float          brightness;           // multiplier on sampled color, default 1
	float          time;

	// Image info
	int32_t        img_width;
	int32_t        img_height;
	int32_t        original_size;    // Uncompressed bytes; full mip chain on the GPU path
	int32_t        compressed_size;  // Compressed bytes over the same mip range
	double         compress_time_ms;
	double         gpu_compress_time_ms;
	double         psnr_db;          // Mip-0 PSNR vs original; <0 = not measured

	// Format info
	compress_fmt_  current_format;
	bool           bc1_supported;
	bool           bc7_supported;           // BC7 sampling — desktop yes, mobile no
	bool           bc6h_supported;          // BC6H UF16 sampling — desktop yes, mobile no
	bool           astc6x6_supported;       // Texture sampling supported (drives display)
	bool           astc6x6_validate_only;   // Compute works but display format unsupported (AMD desktop)
	bool           astc8x8hdr_supported;    // ASTC 8x8 HDR sampling — Adreno/Mali yes, AMD desktop no

	// GPU compression
	bool           use_gpu;
	skr_tex_t      texture_source;

	// File loading UI
	char           file_path[512];
	bool           load_requested;

	// Camera
	float          cam_distance;
} scene_texcomp_t;

///////////////////////////////////////////////////////////////////////////////
// DDS output (BC6H validation)
///////////////////////////////////////////////////////////////////////////////

// Minimal DDS writer for BC6H/BC7 blocks, mip 0 only — the BC analog of
// the .astc auto-save. DDS with a DX10 extension header is the standard
// container for these formats, so external tools (texconv, GIMP, RenderDoc)
// and the CPU reference decoders can all read it.
// dxgi_format: 95 = BC6H_UF16, 99 = BC7_UNORM_SRGB.
static bool _dds_write_bc(const char* path, int32_t width, int32_t height, uint32_t dxgi_format, const void* data, size_t size) {
	FILE* f = fopen(path, "wb");
	if (!f) return false;

	uint32_t hdr[37] = {0};
	hdr[ 0] = 0x20534444;             // 'DDS '
	hdr[ 1] = 124;                    // header size
	hdr[ 2] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000; // CAPS|HEIGHT|WIDTH|PIXELFORMAT|LINEARSIZE
	hdr[ 3] = (uint32_t)height;
	hdr[ 4] = (uint32_t)width;
	hdr[ 5] = (uint32_t)size;         // linear size of mip 0
	hdr[ 7] = 1;                      // mip count
	hdr[19] = 32;                     // pixelformat size
	hdr[20] = 0x4;                    // DDPF_FOURCC
	hdr[21] = 0x30315844;             // 'DX10'
	hdr[27] = 0x1000;                 // DDSCAPS_TEXTURE
	hdr[32] = dxgi_format;
	hdr[33] = 3;                      // D3D10_RESOURCE_DIMENSION_TEXTURE2D
	hdr[35] = 1;                      // array size

	bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1 &&
	          fwrite(data, size, 1, f) == 1;
	fclose(f);
	return ok;
}

///////////////////////////////////////////////////////////////////////////////
// Image Loading
///////////////////////////////////////////////////////////////////////////////

static void _load_image(scene_texcomp_t* scene, const char* path) {
	// Destroy existing textures if any
	if (skr_tex_is_valid(&scene->texture_original))   { skr_tex_destroy(&scene->texture_original);   scene->texture_original   = (skr_tex_t){0}; }
	if (skr_tex_is_valid(&scene->texture_compressed))  { skr_tex_destroy(&scene->texture_compressed);  scene->texture_compressed  = (skr_tex_t){0}; }
	if (skr_tex_is_valid(&scene->texture_source))      { skr_tex_destroy(&scene->texture_source);      scene->texture_source      = (skr_tex_t){0}; }

	// Load source image. su_image_load detects .hdr (Radiance) and decodes
	// to skr_tex_fmt_rg11b10; everything else comes back as RGBA8 sRGB.
	int32_t      width, height;
	skr_tex_fmt_ src_fmt = skr_tex_fmt_none;
	void*        pixels  = su_image_load(path, &width, &height, &src_fmt, 4);

	if (!pixels) {
		su_log(su_log_warning, "TexCompress: Failed to load image: %s", path);
		scene->img_width  = 0;
		scene->img_height = 0;
		return;
	}

	scene->img_width  = width;
	scene->img_height = height;

	bool is_hdr = (src_fmt == skr_tex_fmt_rg11b10uf);

	// Create original texture for display. HDR shows up tone-mapped via the
	// shader; for now we just bind it raw and let the unlit shader sample it.
	skr_tex_create(is_hdr ? skr_tex_fmt_rg11b10uf : skr_tex_fmt_rgba32_srgb,
		skr_tex_flags_readable,
		su_sampler_linear_clamp,
		(skr_vec3i_t){width, height, 1}, 1, 0,
		&(skr_tex_data_t){.data = pixels, .mip_count = 1, .layer_count = 1},
		&scene->texture_original);
	skr_tex_set_name(&scene->texture_original, "original");

	// Auto-switch format when image type changes — HDR file forces an HDR
	// format (BC6H on desktop, ASTC 8x8 HDR otherwise); non-HDR file falls
	// back from HDR-only selection to a sensible LDR default.
	bool is_hdr_fmt = scene->current_format == compress_fmt_bc6h ||
	                  scene->current_format == compress_fmt_astc8x8hdr;
	if (is_hdr && !is_hdr_fmt) {
		scene->current_format = scene->bc6h_supported ? compress_fmt_bc6h : compress_fmt_astc8x8hdr;
	} else if (!is_hdr && is_hdr_fmt) {
		scene->current_format = scene->bc1_supported ? compress_fmt_bc1 : compress_fmt_astc6x6;
	}

	// Use the format selected in UI (current_format is set by the radio buttons)
	skr_tex_fmt_ tex_fmt  = skr_tex_fmt_none;
	const char*  fmt_name = "none";

	if (scene->current_format == compress_fmt_bc1 && scene->bc1_supported) {
		tex_fmt  = skr_tex_fmt_bc1_rgba_srgb;
		fmt_name = "BC1";
	} else if (scene->current_format == compress_fmt_bc7 && scene->bc7_supported) {
		tex_fmt  = skr_tex_fmt_bc7_rgba_srgb;
		fmt_name = "BC7";
	} else if (scene->current_format == compress_fmt_bc6h && scene->bc6h_supported) {
		tex_fmt  = skr_tex_fmt_bc6h_rgbuf;
		fmt_name = "BC6H";
	} else if (scene->current_format == compress_fmt_astc4x4 && (scene->astc6x6_supported || scene->astc6x6_validate_only)) {
		tex_fmt  = skr_tex_fmt_astc4x4_rgba_srgb;
		fmt_name = "ASTC4x4";
	} else if (scene->current_format == compress_fmt_astc6x6 && (scene->astc6x6_supported || scene->astc6x6_validate_only)) {
		tex_fmt  = skr_tex_fmt_astc6x6_rgba_srgb;
		fmt_name = "ASTC6x6";
	} else if (scene->current_format == compress_fmt_astc8x8hdr) {
		tex_fmt  = skr_tex_fmt_astc8x8_rgba_hdr;
		fmt_name = "ASTC8x8 HDR";
	} else {
		scene->compressed_size = 0;
		su_log(su_log_warning, "TexCompress: Selected format not supported!");
		// The compare material may still point at the compressed texture we
		// destroyed above — rebind both sides to something valid before bailing.
		skr_material_set_tex(&scene->material_compare, "tex_left",  &scene->texture_original);
		skr_material_set_tex(&scene->material_compare, "tex_right", &scene->texture_original);
		su_image_free(pixels);
		return;
	}

	if (scene->use_gpu) {
		// GPU path: create source texture with mips, compress on GPU. HDR
		// source uses rg11b10 (the format su_image_load decodes Radiance
		// .hdr into); LDR source uses rgba32_linear so mip filtering stays
		// in linear space rather than sRGB.
		skr_tex_create(is_hdr ? skr_tex_fmt_rg11b10uf : skr_tex_fmt_rgba32_linear,
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
			case compress_fmt_bc7:     scene->texture_compressed = tex_compress_gpu_bc7    (&scene->texture_source);       break;
			case compress_fmt_bc6h:    scene->texture_compressed = tex_compress_gpu_bc6h   (&scene->texture_source);       break;
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
			case compress_fmt_astc8x8hdr:
				if (scene->astc8x8hdr_supported)
					scene->texture_compressed = tex_compress_gpu_astc8x8hdr(&scene->texture_source);
				break;
			default: break;
		}
		uint64_t end_ns = ska_time_get_elapsed_ns();

		scene->gpu_compress_time_ms = (end_ns - start_ns) / 1000000.0;

		// The GPU path compresses the full mip chain, so size both sides of
		// the ratio over the whole chain.
		skr_vec3i_t base = {width, height, 1};
		uint32_t    mips = skr_tex_calc_mip_count(base);
		uint64_t    orig_bytes = 0, comp_bytes = 0;
		for (uint32_t m = 0; m < mips; m++) {
			orig_bytes += skr_tex_calc_mip_size(skr_tex_fmt_rgba32_linear, base, m);
			comp_bytes += skr_tex_calc_mip_size(tex_fmt, base, m);
		}
		scene->original_size   = (int32_t)orig_bytes;
		scene->compressed_size = (int32_t)comp_bytes;

		// Timed on the CPU around command recording — the actual GPU cost
		// shows up on the perf graph via the per-frame profile dispatch.
		su_log(su_log_info, "GPU %s: command recording took %.3f ms",
			fmt_name, scene->gpu_compress_time_ms);

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
			case compress_fmt_astc8x8hdr:
				astc_data = tex_compress_gpu_astc8x8hdr_readback(&scene->texture_source, &astc_size);
				astc_block_w = 8; astc_block_h = 8;
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
		           scene->current_format == compress_fmt_astc6x6 ||
		           scene->current_format == compress_fmt_astc8x8hdr) {
			su_log(su_log_warning, "%s: readback dispatch failed", fmt_name);
		}

		// The BC formats use a DDS container instead of .astc.
		if (scene->current_format == compress_fmt_bc6h || scene->current_format == compress_fmt_bc7) {
			bool     is_bc7   = scene->current_format == compress_fmt_bc7;
			int32_t  bc_size  = 0;
			uint8_t* bc_data  = is_bc7 ? tex_compress_gpu_bc7_readback (&scene->texture_source, &bc_size)
			                           : tex_compress_gpu_bc6h_readback(&scene->texture_source, &bc_size);
			if (bc_data) {
				const char* out_path = is_bc7 ? "bc7_output.dds" : "bc6h_output.dds";
				if (_dds_write_bc(out_path, width, height, is_bc7 ? 99u : 95u, bc_data, (size_t)bc_size))
					su_log(su_log_info, "%s: saved %s (%d bytes) from %s", fmt_name, out_path, bc_size, path);
				else
					su_log(su_log_warning, "%s: failed to save %s", fmt_name, out_path);
				free(bc_data);
			} else {
				su_log(su_log_warning, "%s: readback dispatch failed", fmt_name);
			}
		}
#endif
	} else {
		// CPU path — BC1 is the only CPU encoder.
		if (scene->current_format != compress_fmt_bc1) {
			su_log(su_log_warning, "TexCompress: %s has no CPU encoder, switching to GPU path", fmt_name);
			scene->use_gpu = true;
			su_image_free(pixels);
			_load_image(scene, path);
			return;
		}

		uint64_t start_ns = ska_time_get_elapsed_ns();
		uint8_t* compressed_data = bc1_compress(pixels, width, height);
		scene->original_size   = width * height * 4;
		scene->compressed_size = bc1_calc_size(width, height);
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

	// Update compare material: tex_left = original, tex_right = compressed.
	// If the compressed texture couldn't be created (AMD ASTC validate-only
	// path), bind original to both — swipe still works, both halves just
	// show the same content.
	skr_material_set_tex(&scene->material_compare, "tex_left",  &scene->texture_original);
	skr_material_set_tex(&scene->material_compare, "tex_right",
		skr_tex_is_valid(&scene->texture_compressed) ? &scene->texture_compressed : &scene->texture_original);

	// Measure mip-0 quality through the hardware decoder. LDR only — the
	// 8-bit sRGB PSNR convention doesn't apply to HDR content.
	scene->psnr_db = -1.0;
	if (!is_hdr && skr_tex_is_valid(&scene->texture_compressed)) {
		scene->psnr_db = tex_psnr(&scene->texture_original, &scene->texture_compressed);
		if (scene->psnr_db >= 0)
			su_log(su_log_info, "%s: PSNR %.2f dB", fmt_name, scene->psnr_db);
	}

	su_image_free(pixels);

	su_log(su_log_info, "%s: Compressed %dx%d image (%.1f KB -> %.1f KB, %.1f:1 ratio)",
		fmt_name, width, height,
		scene->original_size / 1024.0f,
		scene->compressed_size / 1024.0f,
		(float)scene->original_size / scene->compressed_size);
}

///////////////////////////////////////////////////////////////////////////////
// Scene Implementation
///////////////////////////////////////////////////////////////////////////////

static scene_t* _scene_texcomp_create(void) {
	scene_texcomp_t* scene = calloc(1, sizeof(scene_texcomp_t));
	if (!scene) return NULL;

	scene->base.size    = sizeof(scene_texcomp_t);
	scene->time         = 0.0f;
	scene->cam_distance = 5.0f;
	scene->use_gpu      = true;
	scene->swipe        = 0.5f;
	scene->brightness   = 1.0f;

	// Initialize GPU compression and quality measurement
	tex_compress_gpu_init();
	tex_psnr_init();

	// Check format support
	scene->bc1_supported        = skr_tex_fmt_is_supported(skr_tex_fmt_bc1_rgba_srgb,     skr_tex_flags_readable, 1);
	scene->bc7_supported        = skr_tex_fmt_is_supported(skr_tex_fmt_bc7_rgba_srgb,     skr_tex_flags_readable, 1);
	scene->bc6h_supported       = skr_tex_fmt_is_supported(skr_tex_fmt_bc6h_rgbuf,        skr_tex_flags_readable, 1);
	scene->astc6x6_supported    = skr_tex_fmt_is_supported(skr_tex_fmt_astc6x6_rgba_srgb, skr_tex_flags_readable, 1);
	// HDR ASTC reuses the LDR 8x8 Vulkan format (HDR signalled per-block via
	// CEM). Sampling support is the LDR 8x8 sample feature plus the optional
	// astcHdr device feature; here we just probe sampleability of the LDR
	// format and accept that HDR blocks may decode to garbage on some HW.
	scene->astc8x8hdr_supported = skr_tex_fmt_is_supported(skr_tex_fmt_astc8x8_rgba_hdr,  skr_tex_flags_readable, 1);
	// Even without texture sampling we can still run the compute shader and
	// read the blocks back into a .astc file for offline validation against
	// astcenc — desktop AMD GPUs hit this path.
	scene->astc6x6_validate_only = !scene->astc6x6_supported;

	su_log(su_log_info, "TexCompress: BC1 %s, BC7 %s, BC6H %s, ASTC6x6 %s, ASTC8x8HDR %s",
		scene->bc1_supported     ? "supported" : "not supported",
		scene->bc7_supported     ? "supported" : "not supported",
		scene->bc6h_supported    ? "supported" : "not supported",
		scene->astc6x6_supported ? "supported" :
		scene->astc6x6_validate_only ? "validate-only (no display)" : "not supported",
		scene->astc8x8hdr_supported ? "supported" : "validate-only (no display)");

	// Default format: prefer BC1 (displayable on desktop); ASTC6x6 otherwise —
	// it always at least runs the encoder via the validate-only path.
	scene->current_format = scene->bc1_supported ? compress_fmt_bc1 : compress_fmt_astc6x6;

	// Default file path
	strncpy(scene->file_path, "tree.png", sizeof(scene->file_path) - 1);
	//strncpy(scene->file_path, "monks_forest_4k.hdr", sizeof(scene->file_path) - 1);
	//strncpy(scene->file_path, "kloppenheim_06_puresky_4k.hdr", sizeof(scene->file_path) - 1);

	// Create quad mesh for displaying textures (facing +Z)
	scene->quad_mesh = su_mesh_create_quad(2.0f, 2.0f, (skr_vec3_t){0, 0, 1}, false, (skr_vec4_t){1, 1, 1, 1});
	skr_mesh_set_name(&scene->quad_mesh, "tex_compress_quad");

	// Compare shader does the swipe + line in one quad — replaces what was
	// previously two unlit quads side-by-side.
	scene->shader = su_shader_load("shaders/tex_compare.hlsl.sks", "tex_compare_shader");

	skr_material_create((skr_material_info_t){
		.shader      = &scene->shader,
		.cull        = skr_cull_back,
		.depth_test  = skr_compare_less,
		.blend_state = skr_blend_alpha,
	}, &scene->material_compare);

	// Load default image
	_load_image(scene, scene->file_path);

	return (scene_t*)scene;
}

static void _scene_texcomp_destroy(scene_t* base) {
	scene_texcomp_t* scene = (scene_texcomp_t*)base;

	skr_mesh_destroy    (&scene->quad_mesh);
	skr_material_destroy(&scene->material_compare);
	skr_shader_destroy  (&scene->shader);
	if (skr_tex_is_valid(&scene->texture_original))   skr_tex_destroy(&scene->texture_original);
	if (skr_tex_is_valid(&scene->texture_compressed)) skr_tex_destroy(&scene->texture_compressed);
	if (skr_tex_is_valid(&scene->texture_source))     skr_tex_destroy(&scene->texture_source);

	tex_psnr_shutdown();
	tex_compress_gpu_shutdown();
	free(scene);
}

static void _scene_texcomp_update(scene_t* base, float delta_time) {
	scene_texcomp_t* scene = (scene_texcomp_t*)base;
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
			case compress_fmt_bc1:        tex_compress_gpu_bc1_profile       (&scene->texture_source, true); break;
			case compress_fmt_bc7:        tex_compress_gpu_bc7_profile       (&scene->texture_source);       break;
			case compress_fmt_bc6h:       tex_compress_gpu_bc6h_profile      (&scene->texture_source);       break;
			case compress_fmt_astc4x4:    tex_compress_gpu_astc4x4_profile   (&scene->texture_source);       break;
			case compress_fmt_astc6x6:    tex_compress_gpu_astc6x6_profile   (&scene->texture_source);       break;
			case compress_fmt_astc8x8hdr: tex_compress_gpu_astc8x8hdr_profile(&scene->texture_source);       break;
			default: break;
		}
	}
}

static void _scene_texcomp_render(scene_t* base, int32_t width, int32_t height,
                              skr_render_list_t* ref_render_list,
                              su_system_buffer_t* ref_system_buffer) {
	scene_texcomp_t* scene = (scene_texcomp_t*)base;
	(void)width;
	(void)height;
	(void)ref_system_buffer;

	if (!skr_tex_is_valid(&scene->texture_original)) return;

	// Push the swipe + brightness values to the shader each frame (cheap —
	// material rebuilds the cbuffer only when params actually change).
	skr_material_set_param(&scene->material_compare, "swipe",      sksc_shader_var_float, 1, &scene->swipe);
	skr_material_set_param(&scene->material_compare, "brightness", sksc_shader_var_float, 1, &scene->brightness);

	// Single aspect-fitted quad showing the swipe-blended view.
	float aspect      = (float)scene->img_width / (float)scene->img_height;
	float quad_height = 2.0f;
	float quad_width  = quad_height * aspect;
	float4x4 world = float4x4_trs(
		(float3){0.0f, 0.0f, 0.0f},
		(float4){0, 0, 0, 1},
		(float3){quad_width * 0.5f, quad_height * 0.5f, 1.0f});

	skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->material_compare, &world, sizeof(float4x4), 1);
}

static bool _scene_texcomp_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_texcomp_t* scene = (scene_texcomp_t*)base;

	ImGuiIO* io = igGetIO();
	if (!io->WantCaptureMouse) {
		if (io->MouseWheel != 0.0f) {
			scene->cam_distance -= io->MouseWheel * 0.5f;
		}
		if (io->MouseDown[0]) {
			scene->cam_distance += io->MouseDelta.y * 0.02f;
		}
		if (scene->cam_distance < 0.2f)  scene->cam_distance = 0.2f;
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

static void _scene_texcomp_render_ui(scene_t* base) {
	scene_texcomp_t* scene = (scene_texcomp_t*)base;

	igText("Texture Compression");
	igSeparator();

	// Format support status
	igText("Format Support:");
	igTextColored(scene->bc1_supported ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  BC1:          %s", scene->bc1_supported ? "Yes" : "No");
	igTextColored(scene->bc7_supported ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  BC7:          %s", scene->bc7_supported ? "Yes" : "No");
	igTextColored(scene->bc6h_supported ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  BC6H:         %s", scene->bc6h_supported ? "Yes" : "No");
	// ASTC 4x4 and 6x6 share one sampling probe — Vulkan's LDR ASTC feature
	// is all-or-nothing. The encoder always runs; validate-only means the
	// result can't be displayed, only read back.
	igTextColored(
		scene->astc6x6_supported ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 1.0f, 0.5f, 1.0f},
		"  ASTC 4x4/6x6: %s",
		scene->astc6x6_supported ? "Yes" : "Validate only (no display)");
	igTextColored(
		scene->astc8x8hdr_supported ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 1.0f, 0.5f, 1.0f},
		"  ASTC 8x8 HDR: %s",
		scene->astc8x8hdr_supported ? "Yes" : "Validate only (no display)");

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
		if (scene->bc7_supported) {
			labels[count] = "BC7";                             values[count++] = compress_fmt_bc7;
		}
		if (scene->bc6h_supported) {
			labels[count] = "BC6H (HDR)";                      values[count++] = compress_fmt_bc6h;
		}
		if (scene->astc6x6_supported || scene->astc6x6_validate_only) {
			labels[count] = "ASTC 4x4";       values[count++] = compress_fmt_astc4x4;
			labels[count] = "ASTC 6x6";       values[count++] = compress_fmt_astc6x6;
		}
		// Always offer ASTC HDR: encoder runs on any GPU even when the
		// output format isn't sampleable (validate-only path saves the .astc
		// file, magenta on screen).
		labels[count] = "ASTC 8x8 HDR";   values[count++] = compress_fmt_astc8x8hdr;

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
			char* path = su_file_dialog_open("Select Image", "Image Files", "png;jpg;jpeg;bmp;tga;hdr");
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
			case compress_fmt_bc1:        fmt_name = "BC1 (DXT1)";   break;
			case compress_fmt_bc7:        fmt_name = "BC7";          break;
			case compress_fmt_bc6h:       fmt_name = "BC6H";         break;
			case compress_fmt_astc4x4:    fmt_name = "ASTC 4x4";     break;
			case compress_fmt_astc6x6:    fmt_name = "ASTC 6x6";     break;
			case compress_fmt_astc8x8hdr: fmt_name = "ASTC 8x8 HDR"; break;
			default: break;
		}

		igText("Image: %d x %d", scene->img_width, scene->img_height);
		igText("Format: %s", fmt_name);
		igText("Original:   %.1f KB (RGBA8%s)", scene->original_size / 1024.0f,
			scene->use_gpu ? ", all mips" : "");
		igText("Compressed: %.1f KB", scene->compressed_size / 1024.0f);
		if (scene->compressed_size > 0)
			igText("Ratio:      %.1f:1", (float)scene->original_size / scene->compressed_size);
		if (scene->psnr_db >= 0)
			igText("PSNR:       %.2f dB", scene->psnr_db);

		igSeparator();
		if (scene->use_gpu) {
			// CPU-side command recording cost; actual GPU encoder cost is on
			// the perf graph via the per-frame profile dispatch.
			igText("Record: %.2f ms (CPU side)", scene->gpu_compress_time_ms);
		} else {
			double megapix  = (scene->img_width * scene->img_height) / 1000000.0;
			double mp_per_s = megapix / (scene->compress_time_ms / 1000.0);
			igText("CPU:  %.2f ms (%.1f MP/s)", scene->compress_time_ms, mp_per_s);
		}

		igSeparator();
		igSliderFloat("Swipe",      &scene->swipe,      0.0f, 1.0f,  "%.2f",  0);
		// Logarithmic slider — HDR content may want brightness ≈ 0.01 to see
		// detail, LDR content may want brightness = 1 to 4 to brighten dark
		// areas. Log scale gives both fine control at the low end and reach.
		igSliderFloat("Brightness", &scene->brightness, 0.01f, 8.0f, "×%.2f", ImGuiSliderFlags_Logarithmic);
		igTextColored((ImVec4){0.7f, 0.7f, 0.7f, 1.0f}, "Left: Original  |  Right: %s%s",
			scene->use_gpu ? "GPU " : "", fmt_name);
	} else {
		igTextColored((ImVec4){1.0f, 0.5f, 0.5f, 1.0f}, "No image loaded");
	}
}

const scene_vtable_t scene_tex_compress_vtable = {
	.name       = "Texture Compression",
	.create     = _scene_texcomp_create,
	.destroy    = _scene_texcomp_destroy,
	.update     = _scene_texcomp_update,
	.render     = _scene_texcomp_render,
	.get_camera = _scene_texcomp_get_camera,
	.render_ui  = _scene_texcomp_render_ui,
};
