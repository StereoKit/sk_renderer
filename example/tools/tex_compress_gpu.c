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
	skr_shader_t  etc2_shader;
	skr_shader_t  astc4x4_shader;
	skr_shader_t  astc6x6_shader;
	skr_compute_t bc1_compute;
	skr_compute_t etc2_compute;
	skr_compute_t astc4x4_compute;
	skr_compute_t astc6x6_compute;
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
	g_tc.bc1_shader     = su_shader_load("shaders/bc1_compress.hlsl.sks",     "bc1_compress");
	g_tc.etc2_shader    = su_shader_load("shaders/etc2_compress.hlsl.sks",    "etc2_compress");
	g_tc.astc4x4_shader = su_shader_load("shaders/astc4x4_compress.hlsl.sks", "astc4x4_compress");
	g_tc.astc6x6_shader = su_shader_load("shaders/astc6x6_compress.hlsl.sks", "astc6x6_compress");

	if (skr_shader_is_valid(&g_tc.bc1_shader))     skr_compute_create(&g_tc.bc1_shader,     &g_tc.bc1_compute);
	if (skr_shader_is_valid(&g_tc.etc2_shader))    skr_compute_create(&g_tc.etc2_shader,    &g_tc.etc2_compute);
	if (skr_shader_is_valid(&g_tc.astc4x4_shader)) skr_compute_create(&g_tc.astc4x4_shader, &g_tc.astc4x4_compute);
	if (skr_shader_is_valid(&g_tc.astc6x6_shader)) skr_compute_create(&g_tc.astc6x6_shader, &g_tc.astc6x6_compute);

	g_tc.initialized = true;
}

void tex_compress_gpu_shutdown(void) {
	if (skr_buffer_is_valid (&g_tc.profile_buffer)) skr_buffer_destroy (&g_tc.profile_buffer);
	if (skr_compute_is_valid(&g_tc.bc1_compute))     skr_compute_destroy(&g_tc.bc1_compute);
	if (skr_compute_is_valid(&g_tc.etc2_compute))    skr_compute_destroy(&g_tc.etc2_compute);
	if (skr_compute_is_valid(&g_tc.astc4x4_compute)) skr_compute_destroy(&g_tc.astc4x4_compute);
	if (skr_compute_is_valid(&g_tc.astc6x6_compute)) skr_compute_destroy(&g_tc.astc6x6_compute);
	if (skr_shader_is_valid (&g_tc.bc1_shader))      skr_shader_destroy (&g_tc.bc1_shader);
	if (skr_shader_is_valid (&g_tc.etc2_shader))     skr_shader_destroy (&g_tc.etc2_shader);
	if (skr_shader_is_valid (&g_tc.astc4x4_shader))  skr_shader_destroy (&g_tc.astc4x4_shader);
	if (skr_shader_is_valid (&g_tc.astc6x6_shader))  skr_shader_destroy (&g_tc.astc6x6_shader);
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
	uint32_t    mip_count = skr_tex_calc_mip_count(base_size);

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
	skr_buffer_create(NULL, total_blocks, block_bytes,
		skr_buffer_type_storage, skr_use_compute_readwrite,
		&output_buffer);
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
	uint32_t alpha = enable_alpha ? 1 : 0;
	skr_compute_set_param(&g_tc.bc1_compute, "enable_alpha", sksc_shader_var_uint, 1, &alpha);
	return _compress_gpu(&g_tc.bc1_compute, source, skr_tex_fmt_bc1_rgba_srgb, 4, 4, 8);
}

skr_tex_t tex_compress_gpu_etc2(skr_tex_t* source) {
	return _compress_gpu(&g_tc.etc2_compute, source, skr_tex_fmt_etc1_rgb_srgb, 4, 4, 8);
}

skr_tex_t tex_compress_gpu_astc4x4(skr_tex_t* source) {
	return _compress_gpu(&g_tc.astc4x4_compute, source, skr_tex_fmt_astc4x4_rgba_srgb, 4, 4, 16);
}

skr_tex_t tex_compress_gpu_astc6x6(skr_tex_t* source) {
	return _compress_gpu(&g_tc.astc6x6_compute, source, skr_tex_fmt_astc6x6_rgba_srgb, 6, 6, 16);
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

uint8_t* tex_compress_gpu_astc4x4_readback(skr_tex_t* source, int32_t* out_size) {
	return _compress_readback(&g_tc.astc4x4_compute, source, 4, 4, 16, out_size);
}

uint8_t* tex_compress_gpu_astc6x6_readback(skr_tex_t* source, int32_t* out_size) {
	return _compress_readback(&g_tc.astc6x6_compute, source, 6, 6, 16, out_size);
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
	uint32_t alpha = enable_alpha ? 1 : 0;
	skr_compute_set_param(&g_tc.bc1_compute, "enable_alpha", sksc_shader_var_uint, 1, &alpha);
	_profile_dispatch(&g_tc.bc1_compute, source, 4, 4, 8);
}

void tex_compress_gpu_etc2_profile(skr_tex_t* source) {
	_profile_dispatch(&g_tc.etc2_compute, source, 4, 4, 8);
}

void tex_compress_gpu_astc4x4_profile(skr_tex_t* source) {
	_profile_dispatch(&g_tc.astc4x4_compute, source, 4, 4, 16);
}

void tex_compress_gpu_astc6x6_profile(skr_tex_t* source) {
	_profile_dispatch(&g_tc.astc6x6_compute, source, 6, 6, 16);
}
