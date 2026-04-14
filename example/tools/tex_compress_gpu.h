// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#pragma once

#include <sk_renderer.h>

// GPU Texture Compression
//
// Uses compute shaders to compress RGBA textures to BC1 or ETC2 on the GPU.
// Zero CPU readback - compressed data stays on the GPU via buffer-to-image
// copy (skr_tex_set_buffer).
//
// Usage:
//   tex_compress_gpu_init();
//
//   skr_tex_t source = ...; // RGBA32 texture with mips generated
//   skr_tex_t compressed = tex_compress_gpu_bc1(&source);
//
//   tex_compress_gpu_shutdown();

// Initialize the GPU compression system. Loads shaders and creates compute
// pipeline state. Call once after sk_renderer initialization.
void      tex_compress_gpu_init    (void);
void      tex_compress_gpu_shutdown(void);

// Compress a source RGBA texture to BC1 with full mip chain.
// Source must be skr_tex_fmt_rgba32 or rgba32_linear with mips generated.
// enable_alpha: false = opaque 4-color mode, true = punch-through alpha
skr_tex_t tex_compress_gpu_bc1     (skr_tex_t* source, bool enable_alpha);

// Compress a source RGBA texture to ETC2 RGB8 with full mip chain.
// Source must be skr_tex_fmt_rgba32 or rgba32_linear with mips generated.
// Blocking - dispatches compute, copies buffer to texture, waits for GPU.
skr_tex_t tex_compress_gpu_etc2    (skr_tex_t* source);
