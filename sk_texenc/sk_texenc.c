// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L
#endif

#include "sk_texenc.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if SK_TEXENC_BC
	#include "bc1_compress.hlsl.h"
	#include "bc6h_compress.hlsl.h"
	#include "bc7_compress.hlsl.h"
#endif
#if SK_TEXENC_ASTC
	#include "astc4x4_compress.hlsl.h"
	#include "astc6x6_compress.hlsl.h"
#endif
#if SK_TEXENC_ASTC_HDR
	#include "astc8x8hdr_compress.hlsl.h"
#endif

// Lazily created objects are claimed with a compare-exchange before they're
// built, so racing first uses wait on one build instead of each compiling one.
#define _TE_BUILDING ((void*)1)
#if defined(_MSC_VER)
	#include <intrin.h>
	static void* _te_load(void* volatile* slot) {
		return _InterlockedCompareExchangePointer(slot, NULL, NULL);
	}
	static bool _te_claim(void* volatile* slot) {
		return _InterlockedCompareExchangePointer(slot, _TE_BUILDING, NULL) == NULL;
	}
	static void _te_publish(void* volatile* slot, void* value) {
		_InterlockedExchangePointer(slot, value);
	}
#else
	static void* _te_load(void* volatile* slot) {
		return __atomic_load_n(slot, __ATOMIC_ACQUIRE);
	}
	static bool _te_claim(void* volatile* slot) {
		void* expected = NULL;
		return __atomic_compare_exchange_n(slot, &expected, _TE_BUILDING, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
	}
	static void _te_publish(void* volatile* slot, void* value) {
		__atomic_store_n(slot, value, __ATOMIC_RELEASE);
	}
#endif

static void _te_sleep_1ms(void) {
#ifdef _WIN32
	Sleep(1);
#else
	struct timespec ts = { 0, 1000000 };
	nanosleep(&ts, NULL);
#endif
}

// The slot's object, or NULL when the caller now owns the build and must
// _te_publish the result (NULL on failure, which lets the next caller retry).
static void* _te_acquire(void* volatile* slot) {
	for (;;) {
		void* obj = _te_load(slot);
		if (obj == NULL && _te_claim(slot)) return NULL;
		if (obj != NULL && obj != _TE_BUILDING) return obj;
		_te_sleep_1ms();
	}
}

///////////////////////////////////////////////////////////////////////////////
// Tables
///////////////////////////////////////////////////////////////////////////////

typedef enum {
	_te_shader_bc1,
	_te_shader_bc6h,
	_te_shader_bc7,
	_te_shader_astc4x4,
	_te_shader_astc6x6,
	_te_shader_astc8x8hdr,
	_te_shader_count,
} _te_shader_;

// Which SRGB_ENCODE/LINEAR_DATA specialization a pipeline gets
typedef enum {
	_te_mode_plain,  // UNORM source of gamma bytes, or HDR
	_te_mode_encode, // source Loads as linear light, gamma-encode it
	_te_mode_linear, // non-color data
	_te_mode_count,
} _te_mode_;

#define _TE_FMT_COUNT (sk_texenc_fmt_astc8x8hdr + 1)

typedef struct {
	_te_shader_  shader;
	bool         alpha;      // bc1's ENABLE_ALPHA
	bool         hdr;
	skr_tex_fmt_ out_srgb;
	skr_tex_fmt_ out_linear;
	uint32_t     block_w, block_h, block_bytes;
} _te_fmt_info_t;

static const _te_fmt_info_t _te_formats[_TE_FMT_COUNT] = {
	[sk_texenc_fmt_bc1]        = { _te_shader_bc1,        false, false, skr_tex_fmt_bc1_rgb_srgb,      skr_tex_fmt_bc1_rgb,          4, 4,  8 },
	[sk_texenc_fmt_bc1_alpha]  = { _te_shader_bc1,        true,  false, skr_tex_fmt_bc1_rgba_srgb,     skr_tex_fmt_bc1_rgba,         4, 4,  8 },
	[sk_texenc_fmt_bc7]        = { _te_shader_bc7,        false, false, skr_tex_fmt_bc7_rgba_srgb,     skr_tex_fmt_bc7_rgba,         4, 4, 16 },
	[sk_texenc_fmt_bc6h]       = { _te_shader_bc6h,       false, true,  skr_tex_fmt_bc6h_rgbuf,        skr_tex_fmt_bc6h_rgbuf,       4, 4, 16 },
	[sk_texenc_fmt_astc4x4]    = { _te_shader_astc4x4,    false, false, skr_tex_fmt_astc4x4_rgba_srgb, skr_tex_fmt_astc4x4_rgba,     4, 4, 16 },
	[sk_texenc_fmt_astc6x6]    = { _te_shader_astc6x6,    false, false, skr_tex_fmt_astc6x6_rgba_srgb, skr_tex_fmt_astc6x6_rgba,     6, 6, 16 },
	[sk_texenc_fmt_astc8x8hdr] = { _te_shader_astc8x8hdr, false, true,  skr_tex_fmt_astc8x8_rgba_hdr,  skr_tex_fmt_astc8x8_rgba_hdr, 8, 8, 16 },
};

typedef struct {
	const void* data;          // zlib-compressed .sks
	uint32_t    size;
	uint32_t    unzipped_size;
	const char* name;
} _te_shader_src_t;

static const _te_shader_src_t _te_shader_srcs[_te_shader_count] = {
#if SK_TEXENC_BC
	[_te_shader_bc1]        = { sks_bc1_compress_hlsl_zip,        sizeof(sks_bc1_compress_hlsl_zip),        sks_bc1_compress_hlsl_unzipped_size,        "sk_texenc_bc1"        },
	[_te_shader_bc6h]       = { sks_bc6h_compress_hlsl_zip,       sizeof(sks_bc6h_compress_hlsl_zip),       sks_bc6h_compress_hlsl_unzipped_size,       "sk_texenc_bc6h"       },
	[_te_shader_bc7]        = { sks_bc7_compress_hlsl_zip,        sizeof(sks_bc7_compress_hlsl_zip),        sks_bc7_compress_hlsl_unzipped_size,        "sk_texenc_bc7"        },
#endif
#if SK_TEXENC_ASTC
	[_te_shader_astc4x4]    = { sks_astc4x4_compress_hlsl_zip,    sizeof(sks_astc4x4_compress_hlsl_zip),    sks_astc4x4_compress_hlsl_unzipped_size,    "sk_texenc_astc4x4"    },
	[_te_shader_astc6x6]    = { sks_astc6x6_compress_hlsl_zip,    sizeof(sks_astc6x6_compress_hlsl_zip),    sks_astc6x6_compress_hlsl_unzipped_size,    "sk_texenc_astc6x6"    },
#endif
#if SK_TEXENC_ASTC_HDR
	[_te_shader_astc8x8hdr] = { sks_astc8x8hdr_compress_hlsl_zip, sizeof(sks_astc8x8hdr_compress_hlsl_zip), sks_astc8x8hdr_compress_hlsl_unzipped_size, "sk_texenc_astc8x8hdr" },
#endif
};

// Every encoder declares these first in $Global, in this order; _te_params_match checks
typedef struct {
	uint32_t mip_level;
	uint32_t image_width;
	uint32_t image_height;
	uint32_t blocks_x;
	uint32_t buffer_offset;
} _te_params_t;

///////////////////////////////////////////////////////////////////////////////
// State
///////////////////////////////////////////////////////////////////////////////

static struct {
	bool                 initialized;
	sk_texenc_inflate_fn inflate;
	void*                inflate_context;
	bool                 has_astc; // compiled in and sampleable
	skr_shader_t*        shaders [_te_shader_count];
	skr_compute_t*       computes[_TE_FMT_COUNT][_te_mode_count];
} _te;

///////////////////////////////////////////////////////////////////////////////

void sk_texenc_init(sk_texenc_inflate_fn inflate, void* opt_inflate_context) {
	if (_te.initialized) return;
	_te.inflate         = inflate;
	_te.inflate_context = opt_inflate_context;
	// Vulkan's ASTC LDR support is one feature bit, so one probe covers every block size
	_te.has_astc        = SK_TEXENC_ASTC && skr_tex_fmt_is_supported(skr_tex_fmt_astc4x4_rgba_srgb, skr_tex_flags_readable, 1);
	_te.initialized     = true;
}

void sk_texenc_shutdown(void) {
	for (int32_t f = 0; f < _TE_FMT_COUNT; f++) {
		for (int32_t m = 0; m < _te_mode_count; m++) {
			if (_te.computes[f][m] == NULL) continue;
			skr_compute_destroy(_te.computes[f][m]);
			free(_te.computes[f][m]);
		}
	}
	for (int32_t s = 0; s < _te_shader_count; s++) {
		if (_te.shaders[s] == NULL) continue;
		skr_shader_destroy(_te.shaders[s]);
		free(_te.shaders[s]);
	}
	memset(&_te, 0, sizeof(_te));
}

///////////////////////////////////////////////////////////////////////////////

static bool _te_fmt_valid(sk_texenc_fmt_ format) {
	return format > sk_texenc_fmt_none && format < _TE_FMT_COUNT;
}

bool sk_texenc_prefers_astc(void) {
	return _te.has_astc;
}

skr_tex_fmt_ sk_texenc_output_format(sk_texenc_fmt_ format, sk_texenc_flags_ flags) {
	if (!_te_fmt_valid(format)) return skr_tex_fmt_none;
	const _te_fmt_info_t* info = &_te_formats[format];
	return (flags & sk_texenc_flags_linear) ? info->out_linear : info->out_srgb;
}

bool sk_texenc_available(sk_texenc_fmt_ format, sk_texenc_flags_ flags) {
	if (!_te.initialized || !_te_fmt_valid(format)) return false;
	if (_te_shader_srcs[_te_formats[format].shader].data == NULL) return false;
	return skr_tex_fmt_is_supported(sk_texenc_output_format(format, flags), skr_tex_flags_readable, 1);
}

///////////////////////////////////////////////////////////////////////////////

static bool _te_params_match(const sksc_shader_meta_t* meta) {
	static const struct { const char* name; uint32_t offset; } fields[] = {
		{ "mip_level",     offsetof(_te_params_t, mip_level)     },
		{ "image_width",   offsetof(_te_params_t, image_width)   },
		{ "image_height",  offsetof(_te_params_t, image_height)  },
		{ "blocks_x",      offsetof(_te_params_t, blocks_x)      },
		{ "buffer_offset", offsetof(_te_params_t, buffer_offset) },
	};
	for (uint32_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
		int32_t                  idx = sksc_shader_meta_get_var_index(meta, fields[i].name);
		const sksc_shader_var_t* var = idx >= 0 ? sksc_shader_meta_get_var_info(meta, idx) : NULL;
		if (var == NULL || var->offset != fields[i].offset) return false;
	}
	return true;
}

static const skr_shader_t* _te_get_shader(_te_shader_ id) {
	const _te_shader_src_t* src = &_te_shader_srcs[id];
	if (src->data == NULL) return NULL;

	void* volatile* slot   = (void* volatile*)&_te.shaders[id];
	skr_shader_t*   shader = (skr_shader_t*)_te_acquire(slot);
	if (shader) return shader;

	// skr_shader_create copies what it keeps, so the inflated container is temporary
	void* sks = malloc(src->unzipped_size);
	bool  ok  = _te.inflate(_te.inflate_context, src->data, src->size, sks, src->unzipped_size) == src->unzipped_size;
	if (!ok) skr_log(skr_log_critical, "sk_texenc: failed to inflate %s", src->name);

	shader = (skr_shader_t*)malloc(sizeof(skr_shader_t));
	ok     = ok && skr_shader_create(sks, src->unzipped_size, shader) == skr_err_success;
	free(sks);
	if (ok && !_te_params_match(&shader->meta)) {
		skr_log(skr_log_critical, "sk_texenc: %s's $Global doesn't start with _te_params_t's fields", src->name);
		skr_shader_destroy(shader);
		ok = false;
	}
	if (!ok) {
		free(shader);
		_te_publish(slot, NULL);
		return NULL;
	}
	skr_shader_set_name(shader, src->name);
	_te_publish(slot, shader);
	return shader;
}

static const skr_compute_t* _te_get_compute(sk_texenc_fmt_ format, _te_mode_ mode) {
	void* volatile* slot    = (void* volatile*)&_te.computes[format][mode];
	skr_compute_t*  compute = (skr_compute_t*)_te_acquire(slot);
	if (compute) return compute;

	const _te_fmt_info_t* info   = &_te_formats[format];
	const skr_shader_t*   shader = _te_get_shader(info->shader);
	if (shader == NULL) {
		_te_publish(slot, NULL);
		return NULL;
	}

	skr_spec_constant_t spec[2];
	uint32_t            spec_count = 0;
	if (info->alpha)              spec[spec_count++] = (skr_spec_constant_t){ .name = "ENABLE_ALPHA", .value = 1.0 };
	if (mode == _te_mode_encode)  spec[spec_count++] = (skr_spec_constant_t){ .name = "SRGB_ENCODE",  .value = 1.0 };
	if (mode == _te_mode_linear)  spec[spec_count++] = (skr_spec_constant_t){ .name = "LINEAR_DATA",  .value = 1.0 };

	uint64_t start = skr_time_now_ns();
	compute = (skr_compute_t*)malloc(sizeof(skr_compute_t));
	if (skr_compute_create(shader, (skr_compute_info_t){ .spec_constants = spec, .spec_constant_count = spec_count }, compute) != skr_err_success) {
		free(compute);
		_te_publish(slot, NULL);
		return NULL;
	}
	skr_log(skr_log_info, "sk_texenc: created %s pipeline (mode %d) in %.1fms", _te_shader_srcs[info->shader].name, (int)mode, (skr_time_now_ns() - start) / 1000000.0);
	_te_publish(slot, compute);
	return compute;
}

///////////////////////////////////////////////////////////////////////////////

// Formats whose Load returns linear light: sRGB views decode, floats never
// held gamma to begin with.
static bool _te_loads_linear(skr_tex_fmt_ format) {
	switch (format) {
	case skr_tex_fmt_rgba32_srgb:
	case skr_tex_fmt_bgra32_srgb:
	case skr_tex_fmt_r8_srgb:
	case skr_tex_fmt_rgba64f:
	case skr_tex_fmt_rgba128f:
	case skr_tex_fmt_rg11b10uf:
	case skr_tex_fmt_rgb9e5uf:
	case skr_tex_fmt_r16f:
	case skr_tex_fmt_r32f:
		return true;
	default:
		return false;
	}
}

static _te_mode_ _te_pick_mode(const _te_fmt_info_t* info, sk_texenc_flags_ flags, skr_tex_fmt_ source_format) {
	if (info->hdr)                      return _te_mode_plain;
	if (flags & sk_texenc_flags_linear) return _te_mode_linear;
	return _te_loads_linear(source_format) ? _te_mode_encode : _te_mode_plain;
}

static uint32_t _te_mip_blocks(skr_vec3i_t base, uint32_t mip, const _te_fmt_info_t* info, uint32_t* opt_out_blocks_x) {
	skr_vec3i_t size     = skr_tex_calc_mip_dimensions(base, mip);
	uint32_t    blocks_x = ((uint32_t)size.x + info->block_w - 1) / info->block_w;
	uint32_t    blocks_y = ((uint32_t)size.y + info->block_h - 1) / info->block_h;
	if (opt_out_blocks_x) *opt_out_blocks_x = blocks_x;
	return blocks_x * blocks_y;
}

static void _te_dispatch_mip(const skr_compute_t* compute, const skr_compute_bind_t* binds, skr_vec3i_t base, uint32_t mip, const _te_fmt_info_t* info, uint32_t buffer_offset) {
	uint32_t     blocks_x;
	uint32_t     blocks_y = _te_mip_blocks(base, mip, info, &blocks_x) / blocks_x;
	skr_vec3i_t  mip_size = skr_tex_calc_mip_dimensions(base, mip);
	_te_params_t params   = {
		.mip_level     = mip,
		.image_width   = (uint32_t)mip_size.x,
		.image_height  = (uint32_t)mip_size.y,
		.blocks_x      = blocks_x,
		.buffer_offset = buffer_offset,
	};
	skr_compute_dispatch(compute, binds, 2, &params, sizeof(params), (blocks_x + 7) / 8, (blocks_y + 7) / 8, 1);
}

static skr_tex_t _te_encode(skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, skr_tex_sampler_t sampler, bool cube) {
	skr_tex_t result = {0};
	if (!_te.initialized || !_te_fmt_valid(format) || !skr_tex_is_valid(source)) return result;
	if (cube && source->layer_count < 6) return result;

	const _te_fmt_info_t* info       = &_te_formats[format];
	skr_tex_fmt_          source_fmt = skr_tex_get_format(source);
	skr_tex_fmt_          out_fmt    = sk_texenc_output_format(format, flags);
	if (!skr_tex_fmt_is_supported(out_fmt, skr_tex_flags_readable, 1)) return result;

	const skr_compute_t* compute = _te_get_compute(format, _te_pick_mode(info, flags, source_fmt));
	if (compute == NULL) return result;

	skr_vec3i_t size      = skr_tex_get_size(source);
	skr_vec3i_t base      = { size.x, size.y, 1 };
	uint32_t    layers    = cube ? 6 : 1;
	uint32_t    mip_count = source->mip_levels > 0 ? source->mip_levels : 1;

	// skr_tex_set_buffer's layout: mip-major, with each mip's layers back to back
	uint32_t total_blocks = 0;
	for (uint32_t m = 0; m < mip_count; m++)
		total_blocks += _te_mip_blocks(base, m, info, NULL) * layers;

	skr_buffer_t blocks;
	if (skr_buffer_create(NULL, total_blocks, info->block_bytes, skr_buffer_type_storage, skr_use_compute_readwrite, &blocks) != skr_err_success)
		return result;
	skr_buffer_set_name(&blocks, "sk_texenc_blocks");

	// dynamic is what grants the transfer destination skr_tex_set_buffer copies into
	skr_tex_flags_ out_flags = skr_tex_flags_readable | skr_tex_flags_dynamic | (cube ? skr_tex_flags_cubemap : 0);
	if (skr_tex_create(out_fmt, out_flags, sampler, (skr_vec3i_t){ size.x, size.y, (int32_t)layers }, 1, mip_count, NULL, &result) != skr_err_success) {
		skr_buffer_destroy(&blocks);
		return (skr_tex_t){0};
	}

	// The encoders read a Texture2D, so cube faces are copied out one at a time
	skr_tex_t face = {0};
	if (cube && skr_tex_create(source_fmt, skr_tex_flags_readable | skr_tex_flags_dynamic, sampler, base, 1, mip_count, NULL, &face) != skr_err_success) {
		skr_buffer_destroy(&blocks);
		skr_tex_destroy(&result);
		return (skr_tex_t){0};
	}

	skr_compute_bind_t binds[2] = {
		{ .bind = skr_compute_get_bind(compute, "source_tex")                        },
		{ .bind = skr_compute_get_bind(compute, "output_blocks"), .buffer = &blocks  },
	};

	skr_cmd_begin();
	for (uint32_t layer = 0; layer < layers; layer++) {
		binds[0].tex = cube ? &face : source;
		if (cube) {
			for (uint32_t m = 0; m < mip_count; m++)
				skr_tex_copy(source, &face, m, layer, m, 0, 1);
		}

		uint32_t mip_start = 0;
		for (uint32_t m = 0; m < mip_count; m++) {
			uint32_t mip_blocks = _te_mip_blocks(base, m, info, NULL);
			_te_dispatch_mip(compute, binds, base, m, info, mip_start + layer * mip_blocks);
			mip_start += mip_blocks * layers;
		}
	}
	skr_tex_set_buffer(&result, &blocks, 0, mip_count);
	skr_cmd_end();

	// Destruction is deferred until the GPU is done, so these are safe with the work still queued
	skr_buffer_destroy(&blocks);
	if (cube) skr_tex_destroy(&face);
	return result;
}

skr_tex_t sk_texenc_2d(skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, skr_tex_sampler_t sampler) {
	return _te_encode(source, format, flags, sampler, false);
}

skr_tex_t sk_texenc_cube(skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, skr_tex_sampler_t sampler) {
	return _te_encode(source, format, flags, sampler, true);
}

///////////////////////////////////////////////////////////////////////////////

#ifdef SK_TEXENC_DEBUG

uint32_t sk_texenc_debug_size(const skr_tex_t* source, sk_texenc_fmt_ format, uint32_t mip) {
	if (!_te_fmt_valid(format) || !skr_tex_is_valid(source)) return 0;
	const _te_fmt_info_t* info = &_te_formats[format];
	skr_vec3i_t           size = skr_tex_get_size(source);
	return _te_mip_blocks((skr_vec3i_t){ size.x, size.y, 1 }, mip, info, NULL) * info->block_bytes;
}

bool sk_texenc_debug_encode(skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, uint32_t mip, skr_buffer_t* ref_blocks) {
	if (!_te.initialized || !_te_fmt_valid(format) || !skr_tex_is_valid(source)) return false;
	if (mip >= (source->mip_levels > 0 ? source->mip_levels : 1))                 return false;
	if (skr_buffer_get_size(ref_blocks) < sk_texenc_debug_size(source, format, mip)) return false;

	const _te_fmt_info_t* info    = &_te_formats[format];
	const skr_compute_t*  compute = _te_get_compute(format, _te_pick_mode(info, flags, skr_tex_get_format(source)));
	if (compute == NULL) return false;

	skr_vec3i_t        size     = skr_tex_get_size(source);
	skr_compute_bind_t binds[2] = {
		{ .bind = skr_compute_get_bind(compute, "source_tex"),    .tex    = source     },
		{ .bind = skr_compute_get_bind(compute, "output_blocks"), .buffer = ref_blocks },
	};
	_te_dispatch_mip(compute, binds, (skr_vec3i_t){ size.x, size.y, 1 }, mip, info, 0);
	return true;
}

#endif
