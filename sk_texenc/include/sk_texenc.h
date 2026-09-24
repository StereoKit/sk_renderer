// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#pragma once

// sk_texenc: GPU block compression for sk_renderer textures.
//
// Encoding runs as compute dispatches that write blocks into a storage buffer,
// which is then copied into the compressed texture. Nothing is read back, so
// the whole encode is queued GPU work. Encoders compiled in depend on the
// build: BC on x64, ASTC on ARM, both for WebGPU (see CMakeLists.txt).
//
// Thread safety: after sk_texenc_init, every function except
// sk_texenc_shutdown may be called from any thread sk_renderer allows
// recording on. Pipelines are created lazily on first use and shared.

#include <sk_renderer.h>
#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
	#ifdef SK_TEXENC_BUILD_SHARED
		#define SK_TEXENC_API __declspec(dllexport)
	#else
		#define SK_TEXENC_API
	#endif
#elif defined(__GNUC__) || defined(__clang__)
	#define SK_TEXENC_API __attribute__((visibility("default")))
#else
	#define SK_TEXENC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum sk_texenc_fmt_ {
	sk_texenc_fmt_none = 0,
	sk_texenc_fmt_bc1,        // 4 bpp RGB, opaque
	sk_texenc_fmt_bc1_alpha,  // 4 bpp RGBA, 1-bit punch-through alpha
	sk_texenc_fmt_bc7,        // 8 bpp RGBA
	sk_texenc_fmt_bc6h,       // 8 bpp HDR RGB, needs a float source
	sk_texenc_fmt_astc4x4,    // 8 bpp RGBA
	sk_texenc_fmt_astc6x6,    // 3.56 bpp RGBA
	sk_texenc_fmt_astc8x8hdr, // 2 bpp HDR RGB, needs a float source
} sk_texenc_fmt_;

typedef enum sk_texenc_flags_ {
	sk_texenc_flags_none   = 0,
	// Non-color data such as roughness or normals. Channels are weighted
	// evenly and the output is UNORM rather than sRGB. HDR formats ignore it.
	// Use a UNORM source: an sRGB source is decoded before it's encoded.
	sk_texenc_flags_linear = 1 << 0,
} sk_texenc_flags_;

// Shaders are embedded zlib-compressed (RFC 1950), and sk_texenc vendors no
// decompressor, so the host supplies one. Inflate src into out_dst, which is
// exactly dst_bytes long, and return the bytes written, or 0 on failure. Same
// shape as sk_ktx2's ktx2_inflate_fn. Called from whichever thread first
// needs an encoder.
typedef size_t (*sk_texenc_inflate_fn)(void* context, const void* src, size_t src_bytes, void* out_dst, size_t dst_bytes);

// Call after skr_init. Probes format support and creates no pipelines.
SK_TEXENC_API void         sk_texenc_init         (sk_texenc_inflate_fn inflate, void* opt_inflate_context);
// Call before skr_shutdown, with no encodes in flight on other threads.
SK_TEXENC_API void         sk_texenc_shutdown     (void);

// True when this encoder was compiled in and its output format is sampleable.
SK_TEXENC_API bool         sk_texenc_available    (sk_texenc_fmt_ format, sk_texenc_flags_ flags);
// True when ASTC is the family to use: it's compiled in and sampleable. A
// GPU that samples both families should get ASTC.
SK_TEXENC_API bool         sk_texenc_prefers_astc (void);
// The texture format an encode produces. LDR output is sRGB unless flags
// has sk_texenc_flags_linear.
SK_TEXENC_API skr_tex_fmt_ sk_texenc_output_format(sk_texenc_fmt_ format, sk_texenc_flags_ flags);

// Encodes every mip the source has, and adds none. For sRGB output the
// source's format decides the conversion: sRGB and float sources Load as
// linear light and get gamma-encoded, while UNORM sources are taken as
// already gamma-encoded bytes. Returns an invalid texture on failure.
SK_TEXENC_API skr_tex_t    sk_texenc_2d           (skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, skr_tex_sampler_t sampler);
// As sk_texenc_2d, for a 6 layer cubemap source. Output is a cubemap.
SK_TEXENC_API skr_tex_t    sk_texenc_cube         (skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, skr_tex_sampler_t sampler);

#ifdef SK_TEXENC_DEBUG
// Test builds only, from the SK_TEXENC_DEBUG cmake option. Queues one mip's
// encode into ref_blocks with no output texture, so it works for formats the
// GPU can't sample. ref_blocks needs sk_texenc_debug_size bytes of storage.
SK_TEXENC_API bool         sk_texenc_debug_encode (skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, uint32_t mip, skr_buffer_t* ref_blocks);
SK_TEXENC_API uint32_t     sk_texenc_debug_size   (const skr_tex_t* source, sk_texenc_fmt_ format, uint32_t mip);
#endif

#ifdef __cplusplus
}
#endif
