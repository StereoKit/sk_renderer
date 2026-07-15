// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#include "tex_compress_gpu.h"
#include "scene_util.h"

#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// State
///////////////////////////////////////////////////////////////////////////////

typedef struct {
	skr_shader_t  bc1_shader;
	skr_shader_t  bc6h_shader;
	skr_shader_t  astc4x4_shader;
	skr_shader_t  astc6x6_shader;
	skr_shader_t  astc8x8hdr_shader;
	// BC1 alpha handling and sRGB encoding are spec constants, so each use
	// gets its own pipeline with the branches compiled out. The srgb variant
	// gamma-encodes linear-light sources (float or sRGB-view textures, which
	// Load as linear) so the 5:6:5 endpoints quantize in perceptual space.
	skr_compute_t bc1_compute_opaque;
	skr_compute_t bc1_compute_alpha;
	skr_compute_t bc1_compute_srgb;
	skr_compute_t bc6h_compute;
	skr_compute_t astc4x4_compute;
	skr_compute_t astc6x6_compute;
	skr_compute_t astc8x8hdr_compute;
	bool          initialized;

	// Cached profiling buffer so per-frame dispatches don't pay the
	// allocate-then-destroy cost every frame. Sized to the last source
	// texture; grown as needed.
	skr_buffer_t  profile_buffer;
	uint32_t      profile_buffer_bytes;
} tex_compress_gpu_state_t;

static tex_compress_gpu_state_t g_tc = {0};

///////////////////////////////////////////////////////////////////////////////
// Init / Shutdown
///////////////////////////////////////////////////////////////////////////////

void tex_compress_gpu_init(void) {
	if (g_tc.initialized) return;

	g_tc.bc1_shader        = su_shader_load("shaders/bc1_compress.hlsl.sks",        "bc1_compress");
	g_tc.bc6h_shader       = su_shader_load("shaders/bc6h_compress.hlsl.sks",       "bc6h_compress");
	g_tc.astc4x4_shader    = su_shader_load("shaders/astc4x4_compress.hlsl.sks",    "astc4x4_compress");
	g_tc.astc6x6_shader    = su_shader_load("shaders/astc6x6_compress.hlsl.sks",    "astc6x6_compress");
	g_tc.astc8x8hdr_shader = su_shader_load("shaders/astc8x8hdr_compress.hlsl.sks", "astc8x8hdr_compress");

	if (skr_shader_is_valid(&g_tc.bc1_shader)) {
		skr_spec_constant_t alpha_on = { .name = "ENABLE_ALPHA", .value = 1.0 };
		skr_spec_constant_t srgb_on  = { .name = "SRGB_ENCODE",  .value = 1.0 };
		skr_compute_create(&g_tc.bc1_shader, (skr_compute_info_t){0}, &g_tc.bc1_compute_opaque);
		skr_compute_create(&g_tc.bc1_shader, (skr_compute_info_t){ .spec_constants = &alpha_on, .spec_constant_count = 1 }, &g_tc.bc1_compute_alpha);
		skr_compute_create(&g_tc.bc1_shader, (skr_compute_info_t){ .spec_constants = &srgb_on,  .spec_constant_count = 1 }, &g_tc.bc1_compute_srgb);
	}
	if (skr_shader_is_valid(&g_tc.bc6h_shader))       skr_compute_create(&g_tc.bc6h_shader,       (skr_compute_info_t){0}, &g_tc.bc6h_compute);
	if (skr_shader_is_valid(&g_tc.astc4x4_shader))    skr_compute_create(&g_tc.astc4x4_shader,    (skr_compute_info_t){0}, &g_tc.astc4x4_compute);
	if (skr_shader_is_valid(&g_tc.astc6x6_shader))    skr_compute_create(&g_tc.astc6x6_shader,    (skr_compute_info_t){0}, &g_tc.astc6x6_compute);
	if (skr_shader_is_valid(&g_tc.astc8x8hdr_shader)) skr_compute_create(&g_tc.astc8x8hdr_shader, (skr_compute_info_t){0}, &g_tc.astc8x8hdr_compute);

	g_tc.initialized = true;
}

void tex_compress_gpu_shutdown(void) {
	if (skr_buffer_is_valid (&g_tc.profile_buffer))     skr_buffer_destroy (&g_tc.profile_buffer);
	if (skr_compute_is_valid(&g_tc.bc1_compute_opaque)) skr_compute_destroy(&g_tc.bc1_compute_opaque);
	if (skr_compute_is_valid(&g_tc.bc1_compute_alpha))  skr_compute_destroy(&g_tc.bc1_compute_alpha);
	if (skr_compute_is_valid(&g_tc.bc1_compute_srgb))   skr_compute_destroy(&g_tc.bc1_compute_srgb);
	if (skr_compute_is_valid(&g_tc.bc6h_compute))       skr_compute_destroy(&g_tc.bc6h_compute);
	if (skr_compute_is_valid(&g_tc.astc4x4_compute))    skr_compute_destroy(&g_tc.astc4x4_compute);
	if (skr_compute_is_valid(&g_tc.astc6x6_compute))    skr_compute_destroy(&g_tc.astc6x6_compute);
	if (skr_compute_is_valid(&g_tc.astc8x8hdr_compute)) skr_compute_destroy(&g_tc.astc8x8hdr_compute);
	if (skr_shader_is_valid (&g_tc.bc1_shader))         skr_shader_destroy (&g_tc.bc1_shader);
	if (skr_shader_is_valid (&g_tc.bc6h_shader))        skr_shader_destroy (&g_tc.bc6h_shader);
	if (skr_shader_is_valid (&g_tc.astc4x4_shader))     skr_shader_destroy (&g_tc.astc4x4_shader);
	if (skr_shader_is_valid (&g_tc.astc6x6_shader))     skr_shader_destroy (&g_tc.astc6x6_shader);
	if (skr_shader_is_valid (&g_tc.astc8x8hdr_shader))  skr_shader_destroy (&g_tc.astc8x8hdr_shader);
	g_tc = (tex_compress_gpu_state_t){0};
}

///////////////////////////////////////////////////////////////////////////////
// Internal: GPU compression core
///////////////////////////////////////////////////////////////////////////////

static skr_tex_t _compress_gpu(skr_compute_t* compute, skr_tex_t* source, skr_tex_fmt_ compressed_fmt, uint32_t block_w, uint32_t block_h, uint32_t block_bytes) {
	skr_tex_t result = {0};

	if (!g_tc.initialized || !skr_compute_is_valid(compute)) {
		su_log(su_log_warning, "tex_compress_gpu: not initialized or compute invalid");
		return result;
	}
	if (!skr_tex_is_valid(source)) {
		su_log(su_log_warning, "tex_compress_gpu: source texture invalid");
		return result;
	}

	skr_vec3i_t base_size = skr_tex_get_size(source);
	// Honor the source's actual mip chain (it may have only mip 0).
	uint32_t    mip_count = source->mip_levels ? source->mip_levels : skr_tex_calc_mip_count(base_size);

	// Calculate total buffer size across all mips
	uint32_t total_blocks = 0;
	for (uint32_t m = 0; m < mip_count; m++) {
		skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(base_size, m);
		uint32_t    blocks_x = (mip_size.x + block_w - 1) / block_w;
		uint32_t    blocks_y = (mip_size.y + block_h - 1) / block_h;
		total_blocks += blocks_x * blocks_y;
	}

	// Create device-local storage buffer for all compressed mip data.
	// Each block occupies block_bytes; packed tightly, mip-major.
	skr_buffer_t output_buffer;
	if (skr_buffer_create(NULL, total_blocks, block_bytes,
			skr_buffer_type_storage, skr_use_compute_readwrite,
			&output_buffer) != skr_err_success) {
		su_log(su_log_warning, "tex_compress_gpu: failed to allocate output buffer (%u blocks)", total_blocks);
		return result;
	}
	skr_buffer_set_name(&output_buffer, "tc_gpu_output");

	// Create the compressed output texture (empty, we'll copy into it).
	// skr_tex_flags_dynamic adds TRANSFER_DST needed for skr_tex_set_buffer.
	skr_tex_create(compressed_fmt, skr_tex_flags_readable | skr_tex_flags_dynamic,
		su_sampler_linear_clamp, base_size, 1, mip_count, NULL, &result);
	skr_tex_set_name(&result, "tc_gpu_compressed");

	bool has_output_tex = skr_tex_is_valid(&result);

	// Bind source and output once — buffer_offset changes per mip via params
	skr_compute_set_tex   (compute, "source_tex",    source);
	skr_compute_set_buffer(compute, "output_blocks", &output_buffer);

	// Dispatch compute for each mip level
	uint32_t buffer_offset = 0;
	for (uint32_t m = 0; m < mip_count; m++) {
		skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(base_size, m);
		uint32_t    blocks_x = (mip_size.x + block_w - 1) / block_w;
		uint32_t    blocks_y = (mip_size.y + block_h - 1) / block_h;

		uint32_t val;
		val = m;             skr_compute_set_param(compute, "mip_level",     sksc_shader_var_uint, 1, &val);
		val = mip_size.x;    skr_compute_set_param(compute, "image_width",   sksc_shader_var_uint, 1, &val);
		val = mip_size.y;    skr_compute_set_param(compute, "image_height",  sksc_shader_var_uint, 1, &val);
		val = blocks_x;      skr_compute_set_param(compute, "blocks_x",      sksc_shader_var_uint, 1, &val);
		val = buffer_offset; skr_compute_set_param(compute, "buffer_offset", sksc_shader_var_uint, 1, &val);

		skr_compute_execute(compute, (blocks_x + 7) / 8, (blocks_y + 7) / 8, 1);

		buffer_offset += blocks_x * blocks_y;
	}

	// Copy compressed data from buffer to texture (GPU-side, no readback).
	// Skip if output format isn't supported (still ran compute for profiling).
	if (has_output_tex)
		skr_tex_set_buffer(&result, &output_buffer, 0, mip_count);

	// Destroy is deferred — actual Vulkan destruction happens after the GPU
	// finishes this command buffer, so it's safe to call immediately.
	skr_buffer_destroy(&output_buffer);

	return result;
}

///////////////////////////////////////////////////////////////////////////////
// Public API
///////////////////////////////////////////////////////////////////////////////

skr_tex_t tex_compress_gpu_bc1(skr_tex_t* source, bool enable_alpha) {
	return _compress_gpu(enable_alpha ? &g_tc.bc1_compute_alpha : &g_tc.bc1_compute_opaque,
		source, skr_tex_fmt_bc1_rgba_srgb, 4, 4, 8);
}

skr_tex_t tex_compress_gpu_bc6h(skr_tex_t* source) {
	return _compress_gpu(&g_tc.bc6h_compute, source, skr_tex_fmt_bc6h_rgbuf, 4, 4, 16);
}

skr_tex_t tex_compress_gpu_astc4x4(skr_tex_t* source) {
	return _compress_gpu(&g_tc.astc4x4_compute, source, skr_tex_fmt_astc4x4_rgba_srgb, 4, 4, 16);
}

skr_tex_t tex_compress_gpu_astc6x6(skr_tex_t* source) {
	return _compress_gpu(&g_tc.astc6x6_compute, source, skr_tex_fmt_astc6x6_rgba_srgb, 6, 6, 16);
}

skr_tex_t tex_compress_gpu_astc8x8hdr(skr_tex_t* source) {
	return _compress_gpu(&g_tc.astc8x8hdr_compute, source, skr_tex_fmt_astc8x8_rgba_hdr, 8, 8, 16);
}

///////////////////////////////////////////////////////////////////////////////
// Cubemap compression
//
// The 2D encoder samples a Texture2D, so we can't feed it a cube view directly.
// Instead we copy each face (all mips) into a temporary 2D texture, compress
// that with the existing 2D path, and copy the compressed result into the
// matching layer of a compressed cubemap. All copies are GPU-side image copies
// (skr_tex_copy); no readback.
///////////////////////////////////////////////////////////////////////////////

static skr_tex_t _compress_gpu_cube(skr_compute_t* compute, skr_tex_t* cube_source, skr_tex_fmt_ compressed_fmt, uint32_t block_w, uint32_t block_h, uint32_t block_bytes) {
	skr_tex_t result = {0};

	if (!g_tc.initialized || !skr_compute_is_valid(compute)) {
		su_log(su_log_warning, "tex_compress_gpu_cube: not initialized or compute invalid");
		return result;
	}
	if (!skr_tex_is_valid(cube_source)) {
		su_log(su_log_warning, "tex_compress_gpu_cube: source texture invalid");
		return result;
	}

	skr_vec3i_t  size      = skr_tex_get_size(cube_source);
	skr_tex_fmt_ src_fmt   = skr_tex_get_format(cube_source);
	uint32_t     mip_count = cube_source->mip_levels ? cube_source->mip_levels : skr_tex_calc_mip_count((skr_vec3i_t){size.x, size.y, 1});

	// Compressed output cubemap (empty; we copy faces into it).
	skr_tex_create(compressed_fmt, skr_tex_flags_readable | skr_tex_flags_dynamic | skr_tex_flags_cubemap,
		su_sampler_linear_clamp, (skr_vec3i_t){size.x, size.y, 6}, 1, mip_count, NULL, &result);
	if (!skr_tex_is_valid(&result)) {
		su_log(su_log_warning, "tex_compress_gpu_cube: failed to create compressed cubemap");
		return result;
	}
	skr_tex_set_name(&result, "tc_gpu_compressed_cube");

	// Scratch 2D face matching the source format, full mip chain.
	skr_tex_t face = {0};
	skr_tex_create(src_fmt, skr_tex_flags_readable | skr_tex_flags_dynamic,
		su_sampler_linear_clamp, (skr_vec3i_t){size.x, size.y, 1}, 1, mip_count, NULL, &face);
	if (!skr_tex_is_valid(&face)) {
		su_log(su_log_warning, "tex_compress_gpu_cube: failed to create scratch face");
		skr_tex_destroy(&result);
		return (skr_tex_t){0};
	}

	for (uint32_t layer = 0; layer < 6; layer++) {
		// Copy this face's mips out of the cube into the 2D scratch texture.
		for (uint32_t m = 0; m < mip_count; m++)
			skr_tex_copy(cube_source, &face, m, layer, m, 0, 1);

		// Compress the scratch face (2D path, all mips).
		skr_tex_t comp = _compress_gpu(compute, &face, compressed_fmt, block_w, block_h, block_bytes);
		if (skr_tex_is_valid(&comp)) {
			for (uint32_t m = 0; m < mip_count; m++)
				skr_tex_copy(&comp, &result, m, 0, m, layer, 1);
			skr_tex_destroy(&comp);
		}
	}

	skr_tex_destroy(&face);
	return result;
}

skr_tex_t tex_compress_gpu_cube_bc1(skr_tex_t* cube_source) {
	// Cube sources arrive as linear light — float formats and sRGB-view
	// textures both Load as linear — so gamma-encode into an sRGB BC1 for
	// usable precision in the darks.
	return _compress_gpu_cube(&g_tc.bc1_compute_srgb, cube_source, skr_tex_fmt_bc1_rgb_srgb, 4, 4, 8);
}

skr_tex_t tex_compress_gpu_cube_bc6h(skr_tex_t* cube_source) {
	return _compress_gpu_cube(&g_tc.bc6h_compute, cube_source, skr_tex_fmt_bc6h_rgbuf, 4, 4, 16);
}

skr_tex_t tex_compress_gpu_cube_astc4x4(skr_tex_t* cube_source) {
	return _compress_gpu_cube(&g_tc.astc4x4_compute, cube_source, skr_tex_fmt_astc4x4_rgba, 4, 4, 16);
}

skr_tex_t tex_compress_gpu_cube_astc6x6(skr_tex_t* cube_source) {
	return _compress_gpu_cube(&g_tc.astc6x6_compute, cube_source, skr_tex_fmt_astc6x6_rgba, 6, 6, 16);
}

skr_tex_t tex_compress_gpu_cube_astc8x8hdr(skr_tex_t* cube_source) {
	return _compress_gpu_cube(&g_tc.astc8x8hdr_compute, cube_source, skr_tex_fmt_astc8x8_rgba_hdr, 8, 8, 16);
}

///////////////////////////////////////////////////////////////////////////////
// Readback (desktop only — stalls GPU)
///////////////////////////////////////////////////////////////////////////////

// Allocates a host-visible storage buffer, runs the compute shader at mip 0,
// waits for GPU completion, returns a malloc'd byte copy.
//
// The begin/flush/end sandwich handles both call sites:
//   * Outside a frame (ref_count=0): without begin, the inner acquire/release
//     inside skr_compute_execute auto-submits and flush finds nothing — we'd
//     read the buffer before the GPU finished. Begin keeps the cmd open.
//   * Inside a frame (ref_count=1): begin/end alone can't reach 0, so the
//     fence never signals and we hang. Flush forces submission regardless.
static uint8_t* _compress_readback(skr_compute_t* compute, skr_tex_t* source, uint32_t block_w, uint32_t block_h, uint32_t block_bytes, int32_t* out_size) {
	if (out_size) *out_size = 0;
	if (!g_tc.initialized || !skr_compute_is_valid(compute) || !skr_tex_is_valid(source))
		return NULL;

	skr_vec3i_t base_size = skr_tex_get_size(source);
	uint32_t    blocks_x  = (base_size.x + block_w - 1) / block_w;
	uint32_t    blocks_y  = (base_size.y + block_h - 1) / block_h;
	uint32_t    total     = blocks_x * blocks_y;
	uint32_t    bytes     = total * block_bytes;

	skr_buffer_t out_buf;
	if (skr_buffer_create(NULL, total, block_bytes,
			skr_buffer_type_storage,
			(skr_use_)(skr_use_dynamic | skr_use_compute_readwrite),
			&out_buf) != skr_err_success) {
		return NULL;
	}
	skr_buffer_set_name(&out_buf, "tc_gpu_readback");

	skr_cmd_begin();

	skr_compute_set_tex   (compute, "source_tex",    source);
	skr_compute_set_buffer(compute, "output_blocks", &out_buf);

	uint32_t val;
	val = 0;           skr_compute_set_param(compute, "mip_level",     sksc_shader_var_uint, 1, &val);
	val = base_size.x; skr_compute_set_param(compute, "image_width",   sksc_shader_var_uint, 1, &val);
	val = base_size.y; skr_compute_set_param(compute, "image_height",  sksc_shader_var_uint, 1, &val);
	val = blocks_x;    skr_compute_set_param(compute, "blocks_x",      sksc_shader_var_uint, 1, &val);
	val = 0;           skr_compute_set_param(compute, "buffer_offset", sksc_shader_var_uint, 1, &val);

	skr_compute_execute(compute, (blocks_x + 7) / 8, (blocks_y + 7) / 8, 1);

	skr_future_t f = skr_cmd_flush();
	skr_future_wait(&f);
	skr_cmd_end();

	uint8_t* bytes_out = (uint8_t*)malloc(bytes);
	if (!bytes_out) {
		skr_buffer_destroy(&out_buf);
		return NULL;
	}
	skr_buffer_get(&out_buf, bytes_out, bytes);
	skr_buffer_destroy(&out_buf);

	if (out_size) *out_size = (int32_t)bytes;
	return bytes_out;
}

uint8_t* tex_compress_gpu_bc6h_readback(skr_tex_t* source, int32_t* out_size) {
	return _compress_readback(&g_tc.bc6h_compute, source, 4, 4, 16, out_size);
}

uint8_t* tex_compress_gpu_astc4x4_readback(skr_tex_t* source, int32_t* out_size) {
	return _compress_readback(&g_tc.astc4x4_compute, source, 4, 4, 16, out_size);
}

uint8_t* tex_compress_gpu_astc6x6_readback(skr_tex_t* source, int32_t* out_size) {
	return _compress_readback(&g_tc.astc6x6_compute, source, 6, 6, 16, out_size);
}

uint8_t* tex_compress_gpu_astc8x8hdr_readback(skr_tex_t* source, int32_t* out_size) {
	return _compress_readback(&g_tc.astc8x8hdr_compute, source, 8, 8, 16, out_size);
}

///////////////////////////////////////////////////////////////////////////////
// Profile-only dispatch (single mip, cached output buffer)
///////////////////////////////////////////////////////////////////////////////

static void _profile_dispatch(skr_compute_t* compute, skr_tex_t* source, uint32_t block_w, uint32_t block_h, uint32_t block_bytes) {
	if (!g_tc.initialized || !skr_compute_is_valid(compute) || !skr_tex_is_valid(source))
		return;

	skr_vec3i_t base_size = skr_tex_get_size(source);
	uint32_t    blocks_x  = (base_size.x + block_w - 1) / block_w;
	uint32_t    blocks_y  = (base_size.y + block_h - 1) / block_h;
	uint32_t    total     = blocks_x * blocks_y;
	uint32_t    bytes     = total * block_bytes;

	// Grow the cached buffer on demand. Shared across all formats — the
	// buffer is just a byte bag, the shader decides its view of it. Skipping
	// the per-frame vkCreateBuffer/vkAllocateMemory keeps the profile path
	// measuring the shader itself, not the allocator.
	if (!skr_buffer_is_valid(&g_tc.profile_buffer) || g_tc.profile_buffer_bytes < bytes) {
		if (skr_buffer_is_valid(&g_tc.profile_buffer))
			skr_buffer_destroy(&g_tc.profile_buffer);
		if (skr_buffer_create(NULL, bytes, 1,
				skr_buffer_type_storage, skr_use_compute_readwrite,
				&g_tc.profile_buffer) != skr_err_success) {
			g_tc.profile_buffer_bytes = 0;
			return;
		}
		skr_buffer_set_name(&g_tc.profile_buffer, "tc_gpu_profile");
		g_tc.profile_buffer_bytes = bytes;
	}

	skr_compute_set_tex   (compute, "source_tex",    source);
	skr_compute_set_buffer(compute, "output_blocks", &g_tc.profile_buffer);

	uint32_t val;
	val = 0;           skr_compute_set_param(compute, "mip_level",     sksc_shader_var_uint, 1, &val);
	val = base_size.x; skr_compute_set_param(compute, "image_width",   sksc_shader_var_uint, 1, &val);
	val = base_size.y; skr_compute_set_param(compute, "image_height",  sksc_shader_var_uint, 1, &val);
	val = blocks_x;    skr_compute_set_param(compute, "blocks_x",      sksc_shader_var_uint, 1, &val);
	val = 0;           skr_compute_set_param(compute, "buffer_offset", sksc_shader_var_uint, 1, &val);

	skr_compute_execute(compute, (blocks_x + 7) / 8, (blocks_y + 7) / 8, 1);
}

void tex_compress_gpu_bc1_profile(skr_tex_t* source, bool enable_alpha) {
	_profile_dispatch(enable_alpha ? &g_tc.bc1_compute_alpha : &g_tc.bc1_compute_opaque,
		source, 4, 4, 8);
}

void tex_compress_gpu_bc6h_profile(skr_tex_t* source) {
	_profile_dispatch(&g_tc.bc6h_compute, source, 4, 4, 16);
}

void tex_compress_gpu_astc4x4_profile(skr_tex_t* source) {
	_profile_dispatch(&g_tc.astc4x4_compute, source, 4, 4, 16);
}

void tex_compress_gpu_astc6x6_profile(skr_tex_t* source) {
	_profile_dispatch(&g_tc.astc6x6_compute, source, 6, 6, 16);
}

void tex_compress_gpu_astc8x8hdr_profile(skr_tex_t* source) {
	_profile_dispatch(&g_tc.astc8x8hdr_compute, source, 8, 8, 16);
}
