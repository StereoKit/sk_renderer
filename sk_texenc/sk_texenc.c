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

#define _TE_MAX_MIPS 16

// texenc_common.hlsli declares these first in every encoder's $Global; _te_params_match checks
typedef struct {
	uint32_t base_width;
	uint32_t base_height;
	uint32_t mip_count;
	uint32_t groups_x;
	uint32_t mip_group_start[_TE_MAX_MIPS];
	uint32_t mip_block_start[_TE_MAX_MIPS];
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
		{ "base_width",      offsetof(_te_params_t, base_width)      },
		{ "base_height",     offsetof(_te_params_t, base_height)     },
		{ "mip_count",       offsetof(_te_params_t, mip_count)       },
		{ "groups_x",        offsetof(_te_params_t, groups_x)        },
		{ "mip_group_start", offsetof(_te_params_t, mip_group_start) },
		{ "mip_block_start", offsetof(_te_params_t, mip_block_start) },
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

// One dispatch for mips [mip_first, mip_end) of every layer. Earlier mips get
// no groups or blocks, so the shader's lookup skips past them.
static void _te_dispatch(const skr_compute_t* compute, const skr_compute_bind_t* binds, skr_vec3i_t base, uint32_t mip_first, uint32_t mip_end, uint32_t layers, const _te_fmt_info_t* info) {
	_te_params_t params = {
		.base_width  = (uint32_t)base.x,
		.base_height = (uint32_t)base.y,
		.mip_count   = mip_end,
	};
	uint32_t groups = 0;
	uint32_t blocks = 0;
	for (uint32_t m = mip_first; m < mip_end; m++) {
		uint32_t blocks_x;
		uint32_t mip_blocks = _te_mip_blocks(base, m, info, &blocks_x);
		uint32_t blocks_y   = mip_blocks / blocks_x;
		params.mip_group_start[m] = groups;
		params.mip_block_start[m] = blocks;
		groups += ((blocks_x + 7) / 8) * ((blocks_y + 7) / 8);
		blocks += mip_blocks * layers; // skr_tex_set_buffer's layout: mip-major, each mip's layers back to back
	}
	// Each dispatch axis guarantees 65535 groups, so the flat index wraps
	params.groups_x   = groups < 65535 ? groups : 65535;
	uint32_t groups_y = (groups + params.groups_x - 1) / params.groups_x;
	skr_compute_dispatch(compute, binds, 2, &params, sizeof(params), params.groups_x, groups_y, layers);
}

static uint64_t _te_total_blocks(skr_vec3i_t base, uint32_t mip_count, uint32_t layers, const _te_fmt_info_t* info) {
	uint64_t total = 0;
	for (uint32_t m = 0; m < mip_count; m++)
		total += (uint64_t)_te_mip_blocks(base, m, info, NULL) * layers;
	return total;
}

skr_tex_t sk_texenc_encode(skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, skr_tex_sampler_t sampler) {
	skr_tex_t result = {0};
	if (!_te.initialized || !_te_fmt_valid(format) || !skr_tex_is_valid(source)) return result;
	// The shaders read a Texture2DArray, which 3D and multisampled textures can't fill
	if ((source->flags & skr_tex_flags_3d) || skr_tex_get_multisample(source) > 1) return result;
	bool cube = (source->flags & skr_tex_flags_cubemap) != 0;
	if (cube && source->layer_count != 6) return result;

	const _te_fmt_info_t* info       = &_te_formats[format];
	skr_tex_fmt_          source_fmt = skr_tex_get_format(source);
	skr_tex_fmt_          out_fmt    = sk_texenc_output_format(format, flags);
	if (!skr_tex_fmt_is_supported(out_fmt, skr_tex_flags_readable, 1)) return result;

	const skr_compute_t* compute = _te_get_compute(format, _te_pick_mode(info, flags, source_fmt));
	if (compute == NULL) return result;

	skr_vec3i_t size      = skr_tex_get_size(source);
	skr_vec3i_t base      = { size.x, size.y, 1 };
	uint32_t    layers    = source->layer_count > 0 ? source->layer_count : 1;
	uint32_t    mip_count = source->mip_levels  > 0 ? source->mip_levels  : 1;
	if (mip_count > _TE_MAX_MIPS) return result;

	uint64_t total_blocks = _te_total_blocks(base, mip_count, layers, info);
	if (total_blocks * info->block_bytes > skr_get_max_storage_size()) {
		skr_log(skr_log_warning, "sk_texenc: a %dx%d texture with %u layers needs more storage buffer than this device can bind", size.x, size.y, layers);
		return result;
	}

	skr_buffer_t blocks;
	if (skr_buffer_create(NULL, (uint32_t)total_blocks, info->block_bytes, skr_buffer_type_storage, skr_use_compute_readwrite, &blocks) != skr_err_success)
		return result;
	skr_buffer_set_name(&blocks, "sk_texenc_blocks");

	// dynamic is what grants the transfer destination skr_tex_set_buffer copies into
	skr_tex_flags_ out_flags = skr_tex_flags_readable | skr_tex_flags_dynamic;
	if      (cube)                                 out_flags |= skr_tex_flags_cubemap;
	else if (source->flags & skr_tex_flags_array)  out_flags |= skr_tex_flags_array;
	if (skr_tex_create(out_fmt, out_flags, sampler, (skr_vec3i_t){ size.x, size.y, (int32_t)layers }, 1, mip_count, NULL, &result) != skr_err_success) {
		skr_buffer_destroy(&blocks);
		return (skr_tex_t){0};
	}

	// The shaders read a Texture2DArray, which sk_renderer binds through an array view
	skr_compute_bind_t binds[2] = {
		{ .bind = skr_compute_get_bind(compute, "source_tex"),    .tex    = source  },
		{ .bind = skr_compute_get_bind(compute, "output_blocks"), .buffer = &blocks },
	};

	skr_cmd_begin();
	_te_dispatch(compute, binds, base, 0, mip_count, layers, info);
	skr_tex_set_buffer(&result, &blocks, 0, mip_count);
	skr_cmd_end();

	// Destruction is deferred until the GPU is done, so this is safe with the work still queued
	skr_buffer_destroy(&blocks);
	return result;
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
	if (mip >= (source->mip_levels > 0 ? source->mip_levels : 1) || mip >= _TE_MAX_MIPS) return false;
	if (skr_buffer_get_size(ref_blocks) < sk_texenc_debug_size(source, format, mip)) return false;

	const _te_fmt_info_t* info    = &_te_formats[format];
	const skr_compute_t*  compute = _te_get_compute(format, _te_pick_mode(info, flags, skr_tex_get_format(source)));
	if (compute == NULL) return false;

	skr_vec3i_t        size     = skr_tex_get_size(source);
	skr_compute_bind_t binds[2] = {
		{ .bind = skr_compute_get_bind(compute, "source_tex"),    .tex    = source     },
		{ .bind = skr_compute_get_bind(compute, "output_blocks"), .buffer = ref_blocks },
	};
	_te_dispatch(compute, binds, (skr_vec3i_t){ size.x, size.y, 1 }, mip, mip + 1, 1, info);
	return true;
}

bool sk_texenc_debug_shader(sk_texenc_fmt_ format, const void* sks, size_t sks_size) {
	if (!_te.initialized || !_te_fmt_valid(format)) return false;
	_te_shader_   id     = _te_formats[format].shader;
	skr_shader_t* shader = (skr_shader_t*)calloc(1, sizeof(skr_shader_t));
	if (skr_shader_create(sks, (uint32_t)sks_size, shader) != skr_err_success || !_te_params_match(&shader->meta)) {
		skr_log(skr_log_critical, "sk_texenc: the replacement for %s isn't a usable encoder", _te_shader_srcs[id].name);
		if (skr_shader_is_valid(shader)) skr_shader_destroy(shader);
		free(shader);
		return false;
	}
	skr_shader_set_name(shader, _te_shader_srcs[id].name);

	// Pipelines built from the old shader go with it, to rebuild from the new one
	for (int32_t f = 0; f < _TE_FMT_COUNT; f++) {
		if (_te_formats[f].shader != id) continue;
		for (int32_t m = 0; m < _te_mode_count; m++) {
			if (_te.computes[f][m] == NULL) continue;
			skr_compute_destroy(_te.computes[f][m]);
			free(_te.computes[f][m]);
			_te.computes[f][m] = NULL;
		}
	}
	if (_te.shaders[id]) {
		skr_shader_destroy(_te.shaders[id]);
		free(_te.shaders[id]);
	}
	_te.shaders[id] = shader;
	return true;
}

#endif
