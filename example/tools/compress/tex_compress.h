// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#pragma once

#include <sk_renderer.h>

// GPU Texture Compression, the example's wrapper around sk_texenc.
//
// Usage:
//   tex_compress_init();
//
//   skr_tex_t source     = ...; // texture with mips generated
//   skr_tex_t compressed = tex_compress(&source, tex_compress_fmt_bc7);
//
//   tex_compress_shutdown();
//
// LDR formats take an rgba32 source: a UNORM view of gamma bytes, or an sRGB
// view, which sk_texenc gamma-encodes after the linear Load. The HDR formats
// (bc6h, astc8x8hdr) expect a float source (rg11b10/rgba16f/...).

typedef enum tex_compress_fmt_ {
	tex_compress_fmt_bc1,          // 4 bpp LDR RGB, opaque 4-color mode only
	tex_compress_fmt_bc1_alpha,    // 4 bpp LDR RGBA, punch-through (1-bit) alpha
	tex_compress_fmt_bc7,          // 8 bpp LDR RGBA, high quality (mode 6 + mode 5 trial)
	tex_compress_fmt_bc6h,         // 8 bpp HDR RGB (UF16, mode 11) — float source
	tex_compress_fmt_astc4x4,      // 8 bpp LDR RGB, high quality
	tex_compress_fmt_astc6x6,      // 3.6 bpp LDR RGBA, multi-mode selector
	tex_compress_fmt_astc8x8hdr,   // 2 bpp HDR RGB (CEM 11) — float source
} tex_compress_fmt_;

void      tex_compress_init    (void);
void      tex_compress_shutdown(void);

// True when sk_texenc embeds this encoder and the GPU can sample its output.
bool      tex_compress_available(tex_compress_fmt_ format);

// Compress a 2D source texture, full mip chain. Returns an invalid texture
// on failure (unsupported format, invalid source, allocation failure).
skr_tex_t tex_compress         (skr_tex_t* source, tex_compress_fmt_ format);

// Compress a 6-layer cubemap, keeping the source's mip count. The result has
// skr_tex_flags_cubemap.
skr_tex_t tex_compress_cube    (skr_tex_t* cube_source, tex_compress_fmt_ format);

///////////////////////////////////////////////////////////////////////////////
// Validation & profiling, only when sk_texenc is built with SK_TEXENC_DEBUG
///////////////////////////////////////////////////////////////////////////////

#ifdef SK_TEXENC_DEBUG
// Compress mip 0, wait for the GPU, and return the raw block bytes (malloc'd;
// caller frees). Stalls the GPU. Works even when the output format isn't
// sampleable, which is how ASTC gets validated on desktop GPUs.
uint8_t*  tex_compress_readback(skr_tex_t* source, tex_compress_fmt_ format, int32_t* out_size);

// Dispatch just the encoder at mip 0 into a cached throwaway buffer, so the
// perf graph shows the shader's cost without the texture upload around it.
void      tex_compress_profile (skr_tex_t* source, tex_compress_fmt_ format);
#endif
