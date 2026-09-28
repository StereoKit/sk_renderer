// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#include "tex_compress.h"
#include <sk_texenc.h>
#include "../scene_util.h"
#include "stb_image.h"

#include <stdlib.h>

///////////////////////////////////////////////////////////////////////////////

#ifdef SK_TEXENC_DEBUG
// Cached so per-frame profile dispatches measure the shader, not the allocator
static skr_buffer_t g_profile_buffer;
#endif

// sk_texenc's shaders are zlib-compressed, and stb_image already carries an inflate
static size_t _inflate(void* context, const void* src, size_t src_bytes, void* out_dst, size_t dst_bytes) {
	(void)context;
	int32_t written = stbi_zlib_decode_buffer((char*)out_dst, (int32_t)dst_bytes, (const char*)src, (int32_t)src_bytes);
	return written < 0 ? 0 : (size_t)written;
}

void tex_compress_init(void) {
	sk_texenc_init(_inflate, NULL);
}

void tex_compress_shutdown(void) {
#ifdef SK_TEXENC_DEBUG
	if (skr_buffer_is_valid(&g_profile_buffer)) skr_buffer_destroy(&g_profile_buffer);
	g_profile_buffer = (skr_buffer_t){0};
#endif
	sk_texenc_shutdown();
}

///////////////////////////////////////////////////////////////////////////////

static sk_texenc_fmt_ _texenc_fmt(tex_compress_fmt_ format) {
	switch (format) {
	case tex_compress_fmt_bc1:        return sk_texenc_fmt_bc1;
	case tex_compress_fmt_bc1_alpha:  return sk_texenc_fmt_bc1_alpha;
	case tex_compress_fmt_bc7:        return sk_texenc_fmt_bc7;
	case tex_compress_fmt_bc6h:       return sk_texenc_fmt_bc6h;
	case tex_compress_fmt_astc4x4:    return sk_texenc_fmt_astc4x4;
	case tex_compress_fmt_astc6x6:    return sk_texenc_fmt_astc6x6;
	case tex_compress_fmt_astc8x8hdr: return sk_texenc_fmt_astc8x8hdr;
	}
	return sk_texenc_fmt_none;
}

skr_tex_t tex_compress(skr_tex_t* source, tex_compress_fmt_ format) {
	return sk_texenc_2d(source, _texenc_fmt(format), sk_texenc_flags_none, su_sampler_linear_clamp);
}

skr_tex_t tex_compress_cube(skr_tex_t* cube_source, tex_compress_fmt_ format) {
	return sk_texenc_cube(cube_source, _texenc_fmt(format), sk_texenc_flags_none, su_sampler_linear_clamp);
}

bool tex_compress_available(tex_compress_fmt_ format) {
	return sk_texenc_available(_texenc_fmt(format), sk_texenc_flags_none);
}

///////////////////////////////////////////////////////////////////////////////

#ifdef SK_TEXENC_DEBUG

// The begin/flush/end sandwich works both outside a frame, where begin keeps
// the command buffer open, and inside one, where only flush can submit it.
uint8_t* tex_compress_readback(skr_tex_t* source, tex_compress_fmt_ format, int32_t* out_size) {
	*out_size = 0;
	uint32_t bytes = sk_texenc_debug_size(source, _texenc_fmt(format), 0);
	if (bytes == 0) return NULL;

	skr_buffer_t blocks;
	if (skr_buffer_create(NULL, bytes, 1, skr_buffer_type_storage, (skr_use_)(skr_use_dynamic | skr_use_compute_readwrite), &blocks) != skr_err_success)
		return NULL;
	skr_buffer_set_name(&blocks, "tc_gpu_readback");

	skr_cmd_begin();
	bool         queued = sk_texenc_debug_encode(source, _texenc_fmt(format), sk_texenc_flags_none, 0, &blocks);
	skr_future_t future = skr_cmd_flush();
	skr_future_wait(&future);
	skr_cmd_end();

	uint8_t* result = queued ? (uint8_t*)malloc(bytes) : NULL;
	if (result) {
		skr_buffer_get(&blocks, result, bytes);
		*out_size = (int32_t)bytes;
	}
	skr_buffer_destroy(&blocks);
	return result;
}

void tex_compress_profile(skr_tex_t* source, tex_compress_fmt_ format) {
	uint32_t bytes = sk_texenc_debug_size(source, _texenc_fmt(format), 0);
	if (bytes == 0) return;

	if (skr_buffer_get_size(&g_profile_buffer) < bytes) {
		if (skr_buffer_is_valid(&g_profile_buffer)) skr_buffer_destroy(&g_profile_buffer);
		g_profile_buffer = (skr_buffer_t){0};
		if (skr_buffer_create(NULL, bytes, 1, skr_buffer_type_storage, skr_use_compute_readwrite, &g_profile_buffer) != skr_err_success)
			return;
		skr_buffer_set_name(&g_profile_buffer, "tc_gpu_profile");
	}
	sk_texenc_debug_encode(source, _texenc_fmt(format), sk_texenc_flags_none, 0, &g_profile_buffer);
}

#endif
