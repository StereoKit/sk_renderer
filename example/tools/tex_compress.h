// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#pragma once

#include <stdint.h>
#include <stdbool.h>

// BC1 (DXT1) Compression
//
// A simple, fast BC1 encoder using min/max endpoint selection with
// perceptual weighting. Supports punch-through alpha.
//
// Features:
// - Perceptually weighted color matching (prioritizes green)
// - Punch-through alpha support (binary transparency)
// - Handles non-power-of-two dimensions
//
// Output format: 8 bytes per 4x4 block
// - 2 bytes: c0 (RGB565)
// - 2 bytes: c1 (RGB565)
// - 4 bytes: 16x 2-bit indices
//
// Alpha mode (c0 <= c1): indices 0,1,2 = colors, index 3 = transparent
// Opaque mode (c0 > c1): indices 0,1,2,3 = 4 interpolated colors

// Alpha threshold for punch-through transparency (0-255)
// Pixels with alpha < this value become transparent
#define BC1_ALPHA_THRESHOLD 128

// Endpoint selection method:
//   0 = Bounding box with inset (fast, decent quality)
//   1 = PCA principal axis (slower, better quality for gradients)
#define BC1_USE_PCA 0

// SIMD acceleration (x64 only):
//   0 = Scalar code (portable)
//   1 = SSE4.1 intrinsics (faster on x64)
// MSVC accepts SSE4.1 intrinsics on any x64 target without a flag and doesn't
// set __SSE4_1__. GCC/Clang only define __SSE4_1__ when -msse4.1 (or higher
// -march) is passed; the example CMakeLists adds it for x64 MinGW builds.
#if (defined(_MSC_VER) && defined(_M_X64)) || defined(__SSE4_1__)
	#define BC1_USE_SIMD 1
#else
	#define BC1_USE_SIMD 0
#endif

// Compress RGBA8 image to BC1
//
// Parameters:
//   rgba   - Source image data (RGBA8, 4 bytes per pixel)
//   width  - Image width in pixels (any size, doesn't need to be multiple of 4)
//   height - Image height in pixels (any size, doesn't need to be multiple of 4)
//
// Returns:
//   Newly allocated BC1 data, caller must free()
//   Size is ((width+3)/4) * ((height+3)/4) * 8 bytes
//   Returns NULL on allocation failure
//
uint8_t* bc1_compress(const uint8_t* rgba, int32_t width, int32_t height);

// Calculate BC1 data size for given dimensions
// Returns size in bytes
static inline int32_t bc1_calc_size(int32_t width, int32_t height) {
	int32_t blocks_x = (width  + 3) / 4;
	int32_t blocks_y = (height + 3) / 4;
	return blocks_x * blocks_y * 8;
}

// Calculate ASTC 6x6 data size for given dimensions
// Returns size in bytes (16 bytes per 6x6 block)
static inline int32_t astc6x6_calc_size(int32_t width, int32_t height) {
	int32_t blocks_x = (width  + 5) / 6;
	int32_t blocks_y = (height + 5) / 6;
	return blocks_x * blocks_y * 16;
}

// Calculate ASTC 4x4 data size for given dimensions
// Returns size in bytes (16 bytes per 4x4 block)
static inline int32_t astc4x4_calc_size(int32_t width, int32_t height) {
	int32_t blocks_x = (width  + 3) / 4;
	int32_t blocks_y = (height + 3) / 4;
	return blocks_x * blocks_y * 16;
}

// Calculate ASTC 8x8 data size for given dimensions (16 bytes per 8x8 block).
// Same byte budget as 4x4/6x6 — 8x8 is just larger spatial coverage.
static inline int32_t astc8x8_calc_size(int32_t width, int32_t height) {
	int32_t blocks_x = (width  + 7) / 8;
	int32_t blocks_y = (height + 7) / 8;
	return blocks_x * blocks_y * 16;
}
