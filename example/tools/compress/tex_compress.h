// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#pragma once

#include <sk_renderer.h>

// GPU Texture Compression
//
// Compresses RGBA textures on the GPU with compute shaders — zero CPU
// readback, compressed data stays on the GPU via buffer-to-image copy
// (skr_tex_set_buffer). Encoder shaders live next to this file in
// shaders/; see tools/compress/README.md for the toolkit layout.
//
// Usage:
//   tex_compress_init();
//
//   skr_tex_t source = ...; // texture with mips generated
//   skr_tex_t compressed = tex_compress(&source, tex_compress_fmt_bc7);
//
//   tex_compress_shutdown();
//
// LDR formats expect an rgba32/rgba32_linear source; the HDR formats
// (bc6h, astc8x8hdr) expect a float-format source (rg11b10/rgba16f/...).
// Callers should probe output-format sampleability themselves via
// skr_tex_fmt_is_supported — the desktop/mobile split is BC vs ASTC.

typedef enum tex_compress_fmt_ {
	tex_compress_fmt_bc1,          // 4 bpp LDR RGB, opaque 4-color mode only
	tex_compress_fmt_bc1_alpha,    // 4 bpp LDR RGBA, punch-through (1-bit) alpha
	tex_compress_fmt_bc7,          // 8 bpp LDR RGBA, high quality (mode 6 + mode 5 trial)
	tex_compress_fmt_bc6h,         // 8 bpp HDR RGB (UF16, mode 11) — float source
	tex_compress_fmt_astc4x4,      // 8 bpp LDR RGB, high quality
	tex_compress_fmt_astc6x6,      // 3.6 bpp LDR RGBA, multi-mode selector
	tex_compress_fmt_astc8x8hdr,   // 2 bpp HDR RGB (CEM 11) — float source
} tex_compress_fmt_;

typedef enum tex_compress_load_ {
	// Load one encoder family: ASTC if this GPU can sample it, BC otherwise.
	// One family is all a runtime needs — ASTC covers mobile, BC covers
	// desktop — and the other family's shaders are never even loaded.
	tex_compress_load_auto,
	// Load every encoder regardless of sampling support. For validation:
	// tex_compress_readback works without a sampleable output format, which
	// is how the demo scene exercises ASTC encoders on desktop GPUs.
	tex_compress_load_all,
} tex_compress_load_;

// The first init after a shutdown decides what's loaded; later calls no-op.
void      tex_compress_init    (tex_compress_load_ load);
void      tex_compress_shutdown(void);

// True when this format's encoder is loaded AND its output format is
// sampleable on this GPU — i.e. tex_compress/_cube will produce a texture
// that can actually be displayed. Use this to pick formats at runtime
// instead of probing skr_tex_fmt_is_supported directly, which knows nothing
// about which encoder family got loaded.
bool      tex_compress_available(tex_compress_fmt_ format);

// Compress a 2D source texture, full mip chain. Returns an invalid texture
// on failure (unsupported format, invalid source, allocation failure).
skr_tex_t tex_compress         (skr_tex_t* source, tex_compress_fmt_ format);

// Compress a 6-layer cubemap by running the 2D encoder on each face and
// assembling the results (GPU-side copies, no readback). The returned
// texture has skr_tex_flags_cubemap and the source's mip count. Cube
// sources arrive as linear light (float and sRGB-view textures both Load
// as linear), so the LDR formats gamma-encode into their sRGB output
// format — quantizing in perceptual space keeps precision in the darks.
skr_tex_t tex_compress_cube    (skr_tex_t* cube_source, tex_compress_fmt_ format);

///////////////////////////////////////////////////////////////////////////////
// Validation & profiling utilities (used by the tex-compress demo scene)
///////////////////////////////////////////////////////////////////////////////

// Compress mip 0 into a host-visible buffer, wait for the GPU, and return
// the raw block bytes (malloc'd; caller frees). Desktop-only — stalls the
// GPU. This is what feeds the auto-saved .astc/.dds regression artifacts;
// it works even when the output format isn't sampleable on this hardware,
// since no destination texture is involved.
uint8_t*  tex_compress_readback(skr_tex_t* source, tex_compress_fmt_ format, int32_t* out_size);

// Dispatch just the encoder at mip 0 into a cached throwaway buffer — no
// texture, no readback, no per-call allocation. For per-frame profiling
// where you want the encoder shader cost on the perf graph, not the
// surrounding texture-upload plumbing.
void      tex_compress_profile (skr_tex_t* source, tex_compress_fmt_ format);
