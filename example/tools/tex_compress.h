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
#if defined(__x86_64__) || defined(_M_X64)
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

// ETC2 RGB8 Compression
//
// ETC2 (Ericsson Texture Compression 2) is mandatory on OpenGL ES 3.0+ and
// Vulkan mobile devices. Backwards compatible with ETC1.
//
// Features:
// - Sub-block based compression (two 2x4 or 4x2 sub-blocks per 4x4 block)
// - Intensity modulation (base color + luminance offsets)
// - ETC2 extended modes: T-mode, H-mode, Planar for better gradients/edges
// - Handles non-power-of-two dimensions
//
// Output format: 8 bytes per 4x4 block
// - Bytes 0-3: Base colors, flip bit, diff bit, table indices
// - Bytes 4-7: 16x 2-bit pixel indices (MSB and LSB planes)
//
// Note: This encoder implements ETC1 modes (individual/differential) which
// are valid ETC2. For better quality, T/H/Planar modes can be added later.

// Compress RGBA8 image to ETC2 RGB8
//
// Parameters:
//   rgba   - Source image data (RGBA8, 4 bytes per pixel, alpha ignored)
//   width  - Image width in pixels (any size, doesn't need to be multiple of 4)
//   height - Image height in pixels (any size, doesn't need to be multiple of 4)
//
// Returns:
//   Newly allocated ETC2 data, caller must free()
//   Size is ((width+3)/4) * ((height+3)/4) * 8 bytes
//   Returns NULL on allocation failure
//
uint8_t* etc2_rgb8_compress(const uint8_t* rgba, int32_t width, int32_t height);

// Calculate ETC2 RGB8 data size for given dimensions
// Returns size in bytes (same as BC1 - 8 bytes per 4x4 block)
static inline int32_t etc2_rgb8_calc_size(int32_t width, int32_t height) {
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
