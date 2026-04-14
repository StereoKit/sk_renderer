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
	skr_compute_t bc1_compute;
	skr_compute_t etc2_compute;
	bool          initialized;
} tex_compress_gpu_state_t;

static tex_compress_gpu_state_t g_tc = {0};

///////////////////////////////////////////////////////////////////////////////
// Init / Shutdown
///////////////////////////////////////////////////////////////////////////////

void tex_compress_gpu_init(void) {
	g_tc.bc1_shader  = su_shader_load("shaders/bc1_compress.hlsl.sks",  "bc1_compress");
	g_tc.etc2_shader = su_shader_load("shaders/etc2_compress.hlsl.sks", "etc2_compress");

	if (skr_shader_is_valid(&g_tc.bc1_shader))  skr_compute_create(&g_tc.bc1_shader,  &g_tc.bc1_compute);
	if (skr_shader_is_valid(&g_tc.etc2_shader)) skr_compute_create(&g_tc.etc2_shader, &g_tc.etc2_compute);

	g_tc.initialized = true;
}

void tex_compress_gpu_shutdown(void) {
	if (skr_compute_is_valid(&g_tc.bc1_compute))  skr_compute_destroy(&g_tc.bc1_compute);
	if (skr_compute_is_valid(&g_tc.etc2_compute)) skr_compute_destroy(&g_tc.etc2_compute);
	if (skr_shader_is_valid (&g_tc.bc1_shader))   skr_shader_destroy (&g_tc.bc1_shader);
	if (skr_shader_is_valid (&g_tc.etc2_shader))  skr_shader_destroy (&g_tc.etc2_shader);
	g_tc = (tex_compress_gpu_state_t){0};
}

///////////////////////////////////////////////////////////////////////////////
// Internal: GPU compression core
///////////////////////////////////////////////////////////////////////////////

static skr_tex_t _compress_gpu(skr_compute_t* compute, skr_tex_t* source, skr_tex_fmt_ compressed_fmt) {
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
		uint32_t    blocks_x = (mip_size.x + 3) / 4;
		uint32_t    blocks_y = (mip_size.y + 3) / 4;
		total_blocks += blocks_x * blocks_y;
	}

	// Create device-local storage buffer for all compressed mip data.
	// Each block is 8 bytes (uint2), tightly packed mip-major.
	skr_buffer_t output_buffer;
	skr_buffer_create(NULL, total_blocks, 8,
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
		uint32_t    blocks_x = (mip_size.x + 3) / 4;
		uint32_t    blocks_y = (mip_size.y + 3) / 4;

		uint32_t val;
		val = m;             skr_compute_set_param(compute, "mip_level",     sksc_shader_var_uint, 1, &val);
		val = mip_size.x;    skr_compute_set_param(compute, "image_width",   sksc_shader_var_uint, 1, &val);
		val = mip_size.y;    skr_compute_set_param(compute, "image_height",  sksc_shader_var_uint, 1, &val);
		val = blocks_x;      skr_compute_set_param(compute, "blocks_x",      sksc_shader_var_uint, 1, &val);
		val = buffer_offset;  skr_compute_set_param(compute, "buffer_offset", sksc_shader_var_uint, 1, &val);

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
	return _compress_gpu(&g_tc.bc1_compute, source, skr_tex_fmt_bc1_rgba_srgb);
}

skr_tex_t tex_compress_gpu_etc2(skr_tex_t* source) {
	return _compress_gpu(&g_tc.etc2_compute, source, skr_tex_fmt_etc1_rgb_srgb);
}
