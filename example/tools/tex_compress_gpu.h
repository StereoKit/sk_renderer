// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#pragma once

#include <sk_renderer.h>

// GPU Texture Compression
//
// Uses compute shaders to compress RGBA textures to BC1, ETC2, ASTC 4x4,
// or ASTC 6x6 on the GPU. Zero CPU readback — compressed data stays on
// the GPU via buffer-to-image copy (skr_tex_set_buffer).
//
// Usage:
//   tex_compress_gpu_init();
//
//   skr_tex_t source = ...; // RGBA32 texture with mips generated
//   skr_tex_t compressed = tex_compress_gpu_astc6x6(&source);
//
//   tex_compress_gpu_shutdown();

void      tex_compress_gpu_init    (void);
void      tex_compress_gpu_shutdown(void);

// Compress a source RGBA texture to BC1 with full mip chain.
// Source must be skr_tex_fmt_rgba32 or rgba32_linear with mips generated.
// enable_alpha: false = opaque 4-color mode, true = punch-through alpha
skr_tex_t tex_compress_gpu_bc1     (skr_tex_t* source, bool enable_alpha);

// Compress a source RGBA texture to ETC2 RGB8 with full mip chain.
// Source must be skr_tex_fmt_rgba32 or rgba32_linear with mips generated.
skr_tex_t tex_compress_gpu_etc2    (skr_tex_t* source);

// Compress to ASTC 4x4 (RGB only) with full mip chain. Per-pixel weights,
// 3-bit weights, 8-bit endpoints.
skr_tex_t tex_compress_gpu_astc4x4 (skr_tex_t* source);

// Compress to ASTC 6x6 with full mip chain. Multi-mode per-block selector
// that picks among:
//   CEM 8 (RGB-only) modes — 4x4 grid 3-bit, 6x6pp 2-bit
//   CEM 12 (RGBA) modes    — 3x3 grid 3-bit, 5x5 grid 2-bit (BISE trit),
//                            3x3 dual-plane 2-bit
// per-block based on alpha pattern + reconstruction SSE. Opaque blocks
// pay only ~2 mode evaluations (matches RGB-only encoder cost); blocks
// with varying alpha pay 3.
skr_tex_t tex_compress_gpu_astc6x6 (skr_tex_t* source);

// Compress to ASTC 8x8 HDR (CEM 11 RGB direct), single-mode v1: 4x4 weight
// grid, 2-bit weights, 8-bit endpoints. Source must be a float-format
// texture (rgba16/rgba32 float). Alpha is ignored. Output format is
// skr_tex_fmt_astc8x8_rgba_hdr (decoder produces FP16). Hardware support
// required for sampling — AMD desktop will display magenta.
skr_tex_t tex_compress_gpu_astc8x8hdr(skr_tex_t* source);

// Readback-only variants: allocate a host-visible buffer, dispatch the
// compute shader at mip 0, wait for GPU, and return malloc'd bytes.
// Desktop-only; stalls the GPU. Caller frees the returned pointer.
uint8_t*  tex_compress_gpu_astc4x4_readback   (skr_tex_t* source, int32_t* out_size);
uint8_t*  tex_compress_gpu_astc6x6_readback   (skr_tex_t* source, int32_t* out_size);
uint8_t*  tex_compress_gpu_astc8x8hdr_readback(skr_tex_t* source, int32_t* out_size);

// Profile-only variants: dispatch just the compute shader at mip 0 into a
// cached device-local throwaway buffer. No texture, no readback, no buffer
// reallocation per call — for per-frame profiling where you want to measure
// the encoder shader cost, not the surrounding texture-upload plumbing.
void      tex_compress_gpu_bc1_profile       (skr_tex_t* source, bool enable_alpha);
void      tex_compress_gpu_etc2_profile      (skr_tex_t* source);
void      tex_compress_gpu_astc4x4_profile   (skr_tex_t* source);
void      tex_compress_gpu_astc6x6_profile   (skr_tex_t* source);
void      tex_compress_gpu_astc8x8hdr_profile(skr_tex_t* source);
