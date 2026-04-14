// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#include "tex_compress.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if BC1_USE_SIMD
#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3
#include <smmintrin.h>  // SSE4.1
#endif

// ETC SIMD support - auto-detect SSE4.1 or NEON
#ifndef ETC_USE_SSE
	#if defined(__SSE4_1__) || (defined(_MSC_VER) && defined(__AVX__))
		#define ETC_USE_SSE 1
	#else
		#define ETC_USE_SSE 0
	#endif
#endif

#ifndef ETC_USE_NEON
	#if defined(__ARM_NEON) || defined(__ARM_NEON__)
		#define ETC_USE_NEON 1
	#else
		#define ETC_USE_NEON 0
	#endif
#endif

// Combined SIMD flag
#ifndef ETC_USE_SIMD
	#define ETC_USE_SIMD (ETC_USE_SSE || ETC_USE_NEON)
#endif

#if ETC_USE_SSE
	#if !BC1_USE_SIMD
		#include <emmintrin.h>  // SSE2
		#include <tmmintrin.h>  // SSSE3
		#include <smmintrin.h>  // SSE4.1
	#endif
#endif

#if ETC_USE_NEON
	#include <arm_neon.h>
#endif

///////////////////////////////////////////////////////////////////////////////
// Internal Helpers
///////////////////////////////////////////////////////////////////////////////

// Convert RGB888 to RGB565
static uint16_t _rgb888_to_565(uint8_t r, uint8_t g, uint8_t b) {
	return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

// Expand RGB565 back to RGB888 for comparison
static void _rgb565_to_888(uint16_t c, uint8_t* r, uint8_t* g, uint8_t* b) {
	*r = ((c >> 11) & 0x1F) * 255 / 31;
	*g = ((c >> 5)  & 0x3F) * 255 / 63;
	*b = (c         & 0x1F) * 255 / 31;
}

// Index mapping tables for projection-based index selection
// 4-color mode: comparison result 0,1,2,3 -> BC1 index 0,2,3,1
static const uint8_t _idx_map_4color[4] = {0, 2, 3, 1};
// 3-color mode: comparison result 0,1,2 -> BC1 index 0,2,1
static const uint8_t _idx_map_3color[4] = {0, 2, 1, 0};  // [3] unused

// Perceptually weighted squared distance between two RGB colors
// Human eyes are most sensitive to green, then red, then blue
// Weights approximate luminance contribution: R=0.299, G=0.587, B=0.114
// Simplified to integer weights: R=2, G=4, B=1
static int32_t _color_dist_sq(uint8_t r0, uint8_t g0, uint8_t b0,
                              uint8_t r1, uint8_t g1, uint8_t b1) {
	int32_t dr = (int32_t)r0 - (int32_t)r1;
	int32_t dg = (int32_t)g0 - (int32_t)g1;
	int32_t db = (int32_t)b0 - (int32_t)b1;
	return dr*dr*2 + dg*dg*4 + db*db;
}

#if BC1_USE_PCA
// Find endpoints using PCA (Principal Component Analysis)
// Projects colors onto their principal axis and takes extremes
static void _find_endpoints_pca(const uint8_t* rgba, int32_t stride, bool has_transparent,
                                uint8_t* out_min_r, uint8_t* out_min_g, uint8_t* out_min_b,
                                uint8_t* out_max_r, uint8_t* out_max_g, uint8_t* out_max_b) {
	// Collect opaque pixels and compute mean
	float pixels[16][3];
	int32_t count = 0;
	float mean_r = 0, mean_g = 0, mean_b = 0;

	for (int32_t y = 0; y < 4; y++) {
		for (int32_t x = 0; x < 4; x++) {
			const uint8_t* p = rgba + y * stride + x * 4;
			if (has_transparent && p[3] < BC1_ALPHA_THRESHOLD) continue;

			pixels[count][0] = p[0];
			pixels[count][1] = p[1];
			pixels[count][2] = p[2];
			mean_r += p[0];
			mean_g += p[1];
			mean_b += p[2];
			count++;
		}
	}

	if (count == 0) {
		*out_min_r = *out_min_g = *out_min_b = 0;
		*out_max_r = *out_max_g = *out_max_b = 0;
		return;
	}

	float inv_count = 1.0f / count;
	mean_r *= inv_count;
	mean_g *= inv_count;
	mean_b *= inv_count;

	// Build covariance matrix (symmetric, so only need 6 values)
	float cov_rr = 0, cov_rg = 0, cov_rb = 0;
	float cov_gg = 0, cov_gb = 0, cov_bb = 0;

	for (int32_t i = 0; i < count; i++) {
		float dr = pixels[i][0] - mean_r;
		float dg = pixels[i][1] - mean_g;
		float db = pixels[i][2] - mean_b;

		cov_rr += dr * dr;
		cov_rg += dr * dg;
		cov_rb += dr * db;
		cov_gg += dg * dg;
		cov_gb += dg * db;
		cov_bb += db * db;
	}

	// Power iteration to find dominant eigenvector
	// Start with luminance direction as initial guess
	float axis_r = 0.299f, axis_g = 0.587f, axis_b = 0.114f;

	for (int32_t iter = 0; iter < 4; iter++) {
		// Multiply by covariance matrix
		float new_r = cov_rr * axis_r + cov_rg * axis_g + cov_rb * axis_b;
		float new_g = cov_rg * axis_r + cov_gg * axis_g + cov_gb * axis_b;
		float new_b = cov_rb * axis_r + cov_gb * axis_g + cov_bb * axis_b;

		// Normalize
		float len = new_r * new_r + new_g * new_g + new_b * new_b;
		if (len < 1e-10f) break;

		len = 1.0f / sqrtf(len);
		axis_r = new_r * len;
		axis_g = new_g * len;
		axis_b = new_b * len;
	}

	// Project all pixels onto axis and find extremes
	float min_t =  1e30f;
	float max_t = -1e30f;

	for (int32_t i = 0; i < count; i++) {
		float t = (pixels[i][0] - mean_r) * axis_r +
		          (pixels[i][1] - mean_g) * axis_g +
		          (pixels[i][2] - mean_b) * axis_b;
		if (t < min_t) min_t = t;
		if (t > max_t) max_t = t;
	}

	// Extend endpoints slightly past the extremes
	// This gives the interpolated colors c2/c3 more room to hit interior pixels
	float range  = max_t - min_t;
	float extend = range / 16.0f;
	min_t -= extend;
	max_t += extend;

	// Compute endpoint colors by projecting along axis from mean
	// Clamp to valid RGB range
	float min_rf = mean_r + min_t * axis_r;
	float min_gf = mean_g + min_t * axis_g;
	float min_bf = mean_b + min_t * axis_b;
	float max_rf = mean_r + max_t * axis_r;
	float max_gf = mean_g + max_t * axis_g;
	float max_bf = mean_b + max_t * axis_b;

	*out_min_r = (uint8_t)(min_rf < 0 ? 0 : (min_rf > 255 ? 255 : min_rf));
	*out_min_g = (uint8_t)(min_gf < 0 ? 0 : (min_gf > 255 ? 255 : min_gf));
	*out_min_b = (uint8_t)(min_bf < 0 ? 0 : (min_bf > 255 ? 255 : min_bf));
	*out_max_r = (uint8_t)(max_rf < 0 ? 0 : (max_rf > 255 ? 255 : max_rf));
	*out_max_g = (uint8_t)(max_gf < 0 ? 0 : (max_gf > 255 ? 255 : max_gf));
	*out_max_b = (uint8_t)(max_bf < 0 ? 0 : (max_bf > 255 ? 255 : max_bf));
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Block Encoder
///////////////////////////////////////////////////////////////////////////////

#if BC1_USE_SIMD && !BC1_USE_PCA
// SSE2/SSSE3 optimized block encoder for bounding box method
// Processes 4 pixels at a time for min/max finding
static void _encode_bc1_block_simd(const uint8_t* rgba, int32_t stride, uint8_t* out) {
	// Load all 16 pixels (4 rows of 4 pixels each)
	// Each pixel is RGBA, so 16 bytes per row
	__m128i row0 = _mm_loadu_si128((const __m128i*)(rgba + stride * 0));
	__m128i row1 = _mm_loadu_si128((const __m128i*)(rgba + stride * 1));
	__m128i row2 = _mm_loadu_si128((const __m128i*)(rgba + stride * 2));
	__m128i row3 = _mm_loadu_si128((const __m128i*)(rgba + stride * 3));

	// Find min/max across all 16 pixels using SSE2 min/max
	__m128i min_rgba = _mm_min_epu8(_mm_min_epu8(row0, row1), _mm_min_epu8(row2, row3));
	__m128i max_rgba = _mm_max_epu8(_mm_max_epu8(row0, row1), _mm_max_epu8(row2, row3));

	// Reduce across the 4 pixels in each register
	// Shuffle to compare pixels 0,1 with 2,3
	__m128i min_shuf = _mm_shuffle_epi32(min_rgba, _MM_SHUFFLE(2, 3, 0, 1));
	__m128i max_shuf = _mm_shuffle_epi32(max_rgba, _MM_SHUFFLE(2, 3, 0, 1));
	min_rgba = _mm_min_epu8(min_rgba, min_shuf);
	max_rgba = _mm_max_epu8(max_rgba, max_shuf);

	// Shuffle again to compare final two
	min_shuf = _mm_shuffle_epi32(min_rgba, _MM_SHUFFLE(1, 0, 3, 2));
	max_shuf = _mm_shuffle_epi32(max_rgba, _MM_SHUFFLE(1, 0, 3, 2));
	min_rgba = _mm_min_epu8(min_rgba, min_shuf);
	max_rgba = _mm_max_epu8(max_rgba, max_shuf);

	// Extract min/max RGB values
	uint32_t min_val = _mm_cvtsi128_si32(min_rgba);
	uint32_t max_val = _mm_cvtsi128_si32(max_rgba);

	uint8_t min_r = (min_val >>  0) & 0xFF;
	uint8_t min_g = (min_val >>  8) & 0xFF;
	uint8_t min_b = (min_val >> 16) & 0xFF;
	uint8_t max_r = (max_val >>  0) & 0xFF;
	uint8_t max_g = (max_val >>  8) & 0xFF;
	uint8_t max_b = (max_val >> 16) & 0xFF;

	// Inset bounding box by 1/16 of range
	int32_t inset_r = (max_r - min_r) >> 4;
	int32_t inset_g = (max_g - min_g) >> 4;
	int32_t inset_b = (max_b - min_b) >> 4;

	min_r += inset_r;  max_r -= inset_r;
	min_g += inset_g;  max_g -= inset_g;
	min_b += inset_b;  max_b -= inset_b;

	// Convert to RGB565
	uint16_t c0 = _rgb888_to_565(max_r, max_g, max_b);
	uint16_t c1 = _rgb888_to_565(min_r, min_g, min_b);

	// 4-color mode: ensure c0 > c1
	if (c0 < c1) {
		uint16_t tmp = c0; c0 = c1; c1 = tmp;
	}
	if (c0 == c1 && c0 < 0xFFFF) {
		c0++;
	}

	// Expand endpoints back to RGB888
	uint8_t colors[4][3];
	_rgb565_to_888(c0, &colors[0][0], &colors[0][1], &colors[0][2]);
	_rgb565_to_888(c1, &colors[1][0], &colors[1][1], &colors[1][2]);

	// c2 = 2/3 * c0 + 1/3 * c1
	colors[2][0] = (2 * colors[0][0] + colors[1][0] + 1) / 3;
	colors[2][1] = (2 * colors[0][1] + colors[1][1] + 1) / 3;
	colors[2][2] = (2 * colors[0][2] + colors[1][2] + 1) / 3;

	// c3 = 1/3 * c0 + 2/3 * c1
	colors[3][0] = (colors[0][0] + 2 * colors[1][0] + 1) / 3;
	colors[3][1] = (colors[0][1] + 2 * colors[1][1] + 1) / 3;
	colors[3][2] = (colors[0][2] + 2 * colors[1][2] + 1) / 3;

	// Find best index for each pixel using projection onto c0→c1 axis
	// Colors lie on a line: c0 at t=0, c2 at t=1/3, c3 at t=2/3, c1 at t=1
	// Thresholds at midpoints: t < 1/6 → 0, t < 1/2 → 2, t < 5/6 → 3, else → 1

	// Axis direction with perceptual weights baked in
	int32_t axis_r = ((int32_t)colors[1][0] - colors[0][0]) * 2;  // weight 2
	int32_t axis_g = ((int32_t)colors[1][1] - colors[0][1]) * 4;  // weight 4
	int32_t axis_b = ((int32_t)colors[1][2] - colors[0][2]);      // weight 1

	int32_t axis_len_sq = axis_r * axis_r / 2 + axis_g * axis_g / 4 + axis_b * axis_b;

	uint32_t indices = 0;
	if (axis_len_sq == 0) {
		// Degenerate case: all same color, all indices = 0
		indices = 0;
	} else {
		// Precompute thresholds (scaled by 6 to avoid fractions)
		// Compare proj*6 against axis_len_sq * {1, 3, 5}
		int32_t thresh_1 = axis_len_sq;
		int32_t thresh_3 = axis_len_sq * 3;
		int32_t thresh_5 = axis_len_sq * 5;

		// Precompute c0 projection to factor out of inner loop
		int32_t c0_proj = colors[0][0] * axis_r + colors[0][1] * axis_g + colors[0][2] * axis_b;

		// Phase 1: Compute all 16 projections (scaled by 6)
		int32_t projs[16];
		for (int32_t y = 0; y < 4; y++) {
			const uint8_t* row = rgba + y * stride;
			for (int32_t x = 0; x < 4; x++) {
				const uint8_t* p = row + x * 4;
				int32_t proj = p[0] * axis_r + p[1] * axis_g + p[2] * axis_b - c0_proj;
				projs[y * 4 + x] = proj * 6;
			}
		}

		// Phase 2: Compute indices from projections using SSE
		// _mm_cmpgt_epi32 returns -1 for true, 0 for false
		// Sum of 3 comparisons gives -3 to 0, negate to get 0 to 3
		__m128i t1 = _mm_set1_epi32(thresh_1 - 1);  // for >= comparison via >
		__m128i t3 = _mm_set1_epi32(thresh_3 - 1);
		__m128i t5 = _mm_set1_epi32(thresh_5 - 1);

		for (int32_t i = 0; i < 16; i += 4) {
			__m128i p = _mm_loadu_si128((const __m128i*)&projs[i]);

			// Count how many thresholds each projection exceeds
			__m128i cmp1 = _mm_cmpgt_epi32(p, t1);  // -1 if proj > thresh_1-1, i.e., proj >= thresh_1
			__m128i cmp3 = _mm_cmpgt_epi32(p, t3);
			__m128i cmp5 = _mm_cmpgt_epi32(p, t5);

			// Sum: each element is -3, -2, -1, or 0
			__m128i sum = _mm_add_epi32(_mm_add_epi32(cmp1, cmp3), cmp5);

			// Negate to get 0, 1, 2, 3
			__m128i idx = _mm_sub_epi32(_mm_setzero_si128(), sum);

			// Extract and map using lookup table
			indices |= ((uint32_t)_idx_map_4color[_mm_extract_epi32(idx, 0)] << ((i + 0) * 2));
			indices |= ((uint32_t)_idx_map_4color[_mm_extract_epi32(idx, 1)] << ((i + 1) * 2));
			indices |= ((uint32_t)_idx_map_4color[_mm_extract_epi32(idx, 2)] << ((i + 2) * 2));
			indices |= ((uint32_t)_idx_map_4color[_mm_extract_epi32(idx, 3)] << ((i + 3) * 2));
		}
	}

	// Write output
	out[0] = c0 & 0xFF;
	out[1] = c0 >> 8;
	out[2] = c1 & 0xFF;
	out[3] = c1 >> 8;
	out[4] = (indices >>  0) & 0xFF;
	out[5] = (indices >>  8) & 0xFF;
	out[6] = (indices >> 16) & 0xFF;
	out[7] = (indices >> 24) & 0xFF;
}
#endif

// Encode a single 4x4 block to BC1 (8 bytes output)
// Input: 16 pixels of RGBA8 (64 bytes)
// Output: 8 bytes BC1 data
// Supports punch-through alpha: pixels with alpha < BC1_ALPHA_THRESHOLD become transparent
static void _encode_bc1_block(const uint8_t* rgba, int32_t stride, uint8_t* out) {
	// Step 1: Check for transparency
	bool has_transparent = false;
	bool has_opaque      = false;

	for (int32_t y = 0; y < 4; y++) {
		for (int32_t x = 0; x < 4; x++) {
			const uint8_t* p = rgba + y * stride + x * 4;
			if (p[3] < BC1_ALPHA_THRESHOLD) has_transparent = true;
			else                            has_opaque      = true;
		}
	}

	// Handle fully transparent block
	if (!has_opaque) {
		out[0] = out[1] = 0;
		out[2] = out[3] = 0;
		out[4] = out[5] = out[6] = out[7] = 0xFF;
		return;
	}

	// Step 2: Find endpoint colors
	uint8_t min_r, min_g, min_b;
	uint8_t max_r, max_g, max_b;

#if BC1_USE_PCA
	_find_endpoints_pca(rgba, stride, has_transparent, &min_r, &min_g, &min_b, &max_r, &max_g, &max_b);
#else
	// Bounding box method: find min/max per channel
	min_r = min_g = min_b = 255;
	max_r = max_g = max_b = 0;

	for (int32_t y = 0; y < 4; y++) {
		for (int32_t x = 0; x < 4; x++) {
			const uint8_t* p = rgba + y * stride + x * 4;
			if (has_transparent && p[3] < BC1_ALPHA_THRESHOLD) continue;

			if (p[0] < min_r) min_r = p[0];
			if (p[1] < min_g) min_g = p[1];
			if (p[2] < min_b) min_b = p[2];
			if (p[0] > max_r) max_r = p[0];
			if (p[1] > max_g) max_g = p[1];
			if (p[2] > max_b) max_b = p[2];
		}
	}

	// Inset bounding box by 1/16 of range
	int32_t inset_r = (max_r - min_r) / 16;
	int32_t inset_g = (max_g - min_g) / 16;
	int32_t inset_b = (max_b - min_b) / 16;

	min_r += inset_r;  max_r -= inset_r;
	min_g += inset_g;  max_g -= inset_g;
	min_b += inset_b;  max_b -= inset_b;
#endif

	// Step 3: Convert to RGB565
	uint16_t c0 = _rgb888_to_565(max_r, max_g, max_b);
	uint16_t c1 = _rgb888_to_565(min_r, min_g, min_b);

	// Step 4: Set up color mode based on transparency
	uint8_t colors[4][3];

	if (has_transparent) {
		// 3-color + alpha mode: c0 <= c1
		// Ensure c0 <= c1 by swapping if needed
		if (c0 > c1) {
			uint16_t tmp = c0; c0 = c1; c1 = tmp;
		}
		// Handle case where colors are equal (need c0 <= c1, so decrement c0)
		if (c0 == c1 && c0 > 0) {
			c0--;
		}

		_rgb565_to_888(c0, &colors[0][0], &colors[0][1], &colors[0][2]);
		_rgb565_to_888(c1, &colors[1][0], &colors[1][1], &colors[1][2]);

		// c2 = (c0 + c1) / 2 (only midpoint in alpha mode)
		colors[2][0] = (colors[0][0] + colors[1][0] + 1) / 2;
		colors[2][1] = (colors[0][1] + colors[1][1] + 1) / 2;
		colors[2][2] = (colors[0][2] + colors[1][2] + 1) / 2;

		// c3 = transparent (not used for color matching)
	} else {
		// 4-color mode: c0 > c1
		if (c0 < c1) {
			uint16_t tmp = c0; c0 = c1; c1 = tmp;
		}
		if (c0 == c1 && c0 < 0xFFFF) {
			c0++;
		}

		_rgb565_to_888(c0, &colors[0][0], &colors[0][1], &colors[0][2]);
		_rgb565_to_888(c1, &colors[1][0], &colors[1][1], &colors[1][2]);

		// c2 = 2/3 * c0 + 1/3 * c1
		colors[2][0] = (2 * colors[0][0] + colors[1][0] + 1) / 3;
		colors[2][1] = (2 * colors[0][1] + colors[1][1] + 1) / 3;
		colors[2][2] = (2 * colors[0][2] + colors[1][2] + 1) / 3;

		// c3 = 1/3 * c0 + 2/3 * c1
		colors[3][0] = (colors[0][0] + 2 * colors[1][0] + 1) / 3;
		colors[3][1] = (colors[0][1] + 2 * colors[1][1] + 1) / 3;
		colors[3][2] = (colors[0][2] + 2 * colors[1][2] + 1) / 3;
	}

	// Step 5: For each pixel, find best matching color index using projection
	// Colors lie on a line from c0 to c1, project each pixel onto this axis

	// Axis direction with perceptual weights
	int32_t axis_r = ((int32_t)colors[1][0] - colors[0][0]) * 2;  // weight 2
	int32_t axis_g = ((int32_t)colors[1][1] - colors[0][1]) * 4;  // weight 4
	int32_t axis_b = ((int32_t)colors[1][2] - colors[0][2]);      // weight 1

	int32_t axis_len_sq = axis_r * axis_r / 2 + axis_g * axis_g / 4 + axis_b * axis_b;

	uint32_t indices = 0;
	if (axis_len_sq == 0) {
		// Degenerate case: all same color
		// For transparent blocks, transparent pixels still need index 3
		if (has_transparent) {
			for (int32_t y = 0; y < 4; y++) {
				for (int32_t x = 0; x < 4; x++) {
					const uint8_t* p = rgba + y * stride + x * 4;
					if (p[3] < BC1_ALPHA_THRESHOLD) {
						indices |= (3 << ((y * 4 + x) * 2));
					}
				}
			}
		}
	} else if (has_transparent) {
		// 3-color + alpha mode: c0 at t=0, c2 at t=1/2, c1 at t=1
		// Thresholds at midpoints: t < 1/4 → 0, t < 3/4 → 2, else → 1
		int32_t thresh_1 = axis_len_sq;      // 1/4 scaled by 4
		int32_t thresh_3 = axis_len_sq * 3;  // 3/4 scaled by 4

		// Precompute c0 projection
		int32_t c0_proj = colors[0][0] * axis_r + colors[0][1] * axis_g + colors[0][2] * axis_b;

		for (int32_t y = 0; y < 4; y++) {
			for (int32_t x = 0; x < 4; x++) {
				const uint8_t* p = rgba + y * stride + x * 4;
				int32_t bit_pos = (y * 4 + x) * 2;

				// Transparent pixel -> index 3
				if (p[3] < BC1_ALPHA_THRESHOLD) {
					indices |= ((uint32_t)3 << bit_pos);
					continue;
				}

				// Project onto axis
				int32_t proj = p[0] * axis_r + p[1] * axis_g + p[2] * axis_b - c0_proj;
				int32_t proj_4 = proj * 4;

				// Determine index: 0 if proj_4 < thresh_1, 2 if < thresh_3, else 1
				int32_t idx = (proj_4 >= thresh_1) + (proj_4 >= thresh_3);

				indices |= ((uint32_t)_idx_map_3color[idx] << bit_pos);
			}
		}
	} else {
		// 4-color mode: c0 at t=0, c2 at t=1/3, c3 at t=2/3, c1 at t=1
		// Thresholds at midpoints: t < 1/6 → 0, t < 1/2 → 2, t < 5/6 → 3, else → 1
		int32_t thresh_1 = axis_len_sq;
		int32_t thresh_3 = axis_len_sq * 3;
		int32_t thresh_5 = axis_len_sq * 5;

		// Precompute c0 projection
		int32_t c0_proj = colors[0][0] * axis_r + colors[0][1] * axis_g + colors[0][2] * axis_b;

		for (int32_t y = 0; y < 4; y++) {
			for (int32_t x = 0; x < 4; x++) {
				const uint8_t* p = rgba + y * stride + x * 4;

				// Project onto axis
				int32_t proj = p[0] * axis_r + p[1] * axis_g + p[2] * axis_b - c0_proj;
				int32_t proj_6 = proj * 6;

				// Determine index
				int32_t idx = (proj_6 >= thresh_1) + (proj_6 >= thresh_3) + (proj_6 >= thresh_5);

				indices |= ((uint32_t)_idx_map_4color[idx] << ((y * 4 + x) * 2));
			}
		}
	}

	// Step 6: Write output (little-endian)
	out[0] = c0 & 0xFF;
	out[1] = c0 >> 8;
	out[2] = c1 & 0xFF;
	out[3] = c1 >> 8;
	out[4] = (indices >>  0) & 0xFF;
	out[5] = (indices >>  8) & 0xFF;
	out[6] = (indices >> 16) & 0xFF;
	out[7] = (indices >> 24) & 0xFF;
}

///////////////////////////////////////////////////////////////////////////////
// Public API
///////////////////////////////////////////////////////////////////////////////

uint8_t* bc1_compress(const uint8_t* rgba, int32_t width, int32_t height) {
	int32_t blocks_x  = (width  + 3) / 4;
	int32_t blocks_y  = (height + 3) / 4;
	int32_t bc1_size  = blocks_x * blocks_y * 8;
	uint8_t* bc1_data = malloc(bc1_size);
	if (!bc1_data) return NULL;

	int32_t stride = width * 4;

	// Temporary buffer for edge blocks that extend past image bounds
	uint8_t block_rgba[4 * 4 * 4];

	for (int32_t by = 0; by < blocks_y; by++) {
		for (int32_t bx = 0; bx < blocks_x; bx++) {
			int32_t px = bx * 4;
			int32_t py = by * 4;

			const uint8_t* block_ptr;
			int32_t        block_stride;

			// Handle edge blocks by copying with clamping
			if (px + 4 > width || py + 4 > height) {
				for (int32_t y = 0; y < 4; y++) {
					for (int32_t x = 0; x < 4; x++) {
						int32_t sx = px + x < width  ? px + x : width  - 1;
						int32_t sy = py + y < height ? py + y : height - 1;
						const uint8_t* src = rgba + sy * stride + sx * 4;
						uint8_t*       dst = block_rgba + y * 16 + x * 4;
						dst[0] = src[0];
						dst[1] = src[1];
						dst[2] = src[2];
						dst[3] = src[3];
					}
				}
				block_ptr    = block_rgba;
				block_stride = 16;
			} else {
				block_ptr    = rgba + py * stride + px * 4;
				block_stride = stride;
			}

			uint8_t* out = bc1_data + (by * blocks_x + bx) * 8;

#if BC1_USE_SIMD && !BC1_USE_PCA
			// SIMD fast path: interior blocks with no transparency
			if (block_stride == stride) {
				// Quick transparency scan
				bool has_alpha = false;
				for (int32_t i = 0; i < 4 && !has_alpha; i++) {
					const uint8_t* row = block_ptr + i * block_stride;
					if (row[3] < BC1_ALPHA_THRESHOLD || row[7]  < BC1_ALPHA_THRESHOLD ||
					    row[11] < BC1_ALPHA_THRESHOLD || row[15] < BC1_ALPHA_THRESHOLD) {
						has_alpha = true;
					}
				}

				if (!has_alpha) {
					_encode_bc1_block_simd(block_ptr, block_stride, out);
					continue;
				}
			}
#endif
			_encode_bc1_block(block_ptr, block_stride, out);
		}
	}

	return bc1_data;
}

///////////////////////////////////////////////////////////////////////////////
// ETC2 RGB8 Compression
///////////////////////////////////////////////////////////////////////////////

// ETC modifier table - pairs of (small, large) intensity offsets
// Index 0: base + small, Index 1: base + large
// Index 2: base - small, Index 3: base - large
static const int32_t _etc_modifier_table[8][2] = {
	{  2,   8},
	{  5,  17},
	{  9,  29},
	{ 13,  42},
	{ 18,  60},
	{ 24,  80},
	{ 33, 106},
	{ 47, 183},
};

// Clamp value to [0, 255]
static inline int32_t _etc_clamp(int32_t v) {
	return v < 0 ? 0 : (v > 255 ? 255 : v);
}

// Expand 4-bit color to 8-bit by replicating high bits into low bits
static inline uint8_t _etc_expand4(int32_t c) {
	return (c << 4) | c;
}

// Expand 5-bit color to 8-bit
static inline uint8_t _etc_expand5(int32_t c) {
	return (c << 3) | (c >> 2);
}

// Expand 6-bit color to 8-bit (for planar mode)
static inline uint8_t _etc_expand6(int32_t c) {
	return (c << 2) | (c >> 4);
}

// Expand 7-bit color to 8-bit (for planar mode)
static inline uint8_t _etc_expand7(int32_t c) {
	return (c << 1) | (c >> 6);
}

// Quantize 8-bit color to N bits with rounding
static inline int32_t _etc_quantize4(int32_t c8) {
	int32_t q = (c8 + 8) >> 4;
	return q > 15 ? 15 : q;
}
static inline int32_t _etc_quantize5(int32_t c8) {
	int32_t q = (c8 * 31 + 127) / 255;
	return q > 31 ? 31 : q;
}

// Pack 16 pixel indices (2-bit each) into MSB/LSB planes for ETC block
// Indices are in row-major order [y*4+x], output is column-major per spec
static inline void _etc_pack_indices(const uint8_t* indices, uint16_t* out_msb, uint16_t* out_lsb) {
	uint16_t msb = 0, lsb = 0;
	for (int32_t y = 0; y < 4; y++) {
		for (int32_t x = 0; x < 4; x++) {
			int32_t bit_idx = x * 4 + y;
			int32_t idx     = indices[y * 4 + x];
			msb |= ((idx >> 1) & 1) << bit_idx;
			lsb |= ((idx >> 0) & 1) << bit_idx;
		}
	}
	*out_msb = msb;
	*out_lsb = lsb;
}

// Write 64-bit block to bytes in big-endian order
static inline void _etc_write_block(uint64_t block, uint8_t* out) {
	out[0] = (block >> 56) & 0xFF;
	out[1] = (block >> 48) & 0xFF;
	out[2] = (block >> 40) & 0xFF;
	out[3] = (block >> 32) & 0xFF;
	out[4] = (block >> 24) & 0xFF;
	out[5] = (block >> 16) & 0xFF;
	out[6] = (block >>  8) & 0xFF;
	out[7] = (block >>  0) & 0xFF;
}

// Pack sub-block indices for individual/differential modes
// indices0/indices1 are 8 indices each in row-major sub-block order
// flip=0: vertical split (2x4 sub-blocks), flip=1: horizontal split (4x2)
static inline void _etc_pack_subblock_indices(const uint8_t* indices0, const uint8_t* indices1,
                                              int32_t flip, uint16_t* out_msb, uint16_t* out_lsb) {
	uint16_t msb = 0, lsb = 0;

	// Sub-block 0
	for (int32_t i = 0; i < 8; i++) {
		int32_t x = flip ? (i % 4) : (i % 2);
		int32_t y = flip ? (i / 4) : (i / 2);
		int32_t bit_idx = x * 4 + y;
		msb |= ((indices0[i] >> 1) & 1) << bit_idx;
		lsb |= ((indices0[i] >> 0) & 1) << bit_idx;
	}

	// Sub-block 1
	for (int32_t i = 0; i < 8; i++) {
		int32_t x = flip ? (i % 4) : (2 + i % 2);
		int32_t y = flip ? (2 + i / 4) : (i / 2);
		int32_t bit_idx = x * 4 + y;
		msb |= ((indices1[i] >> 1) & 1) << bit_idx;
		lsb |= ((indices1[i] >> 0) & 1) << bit_idx;
	}

	*out_msb = msb;
	*out_lsb = lsb;
}

// Compute squared error for a sub-block with given base color and table
#if ETC_USE_SSE
static int32_t _etc_subblock_error(const uint8_t* rgba, int32_t stride,
                                   int32_t x0, int32_t y0, int32_t w, int32_t h,
                                   int32_t base_r, int32_t base_g, int32_t base_b,
                                   int32_t table_idx, int32_t early_out,
                                   uint8_t* out_indices) {
	int32_t mod_small = _etc_modifier_table[table_idx][0];
	int32_t mod_large = _etc_modifier_table[table_idx][1];

	// Precompute the 4 candidate colors
	int32_t c0_r = _etc_clamp(base_r + mod_small), c0_g = _etc_clamp(base_g + mod_small), c0_b = _etc_clamp(base_b + mod_small);
	int32_t c1_r = _etc_clamp(base_r + mod_large), c1_g = _etc_clamp(base_g + mod_large), c1_b = _etc_clamp(base_b + mod_large);
	int32_t c2_r = _etc_clamp(base_r - mod_small), c2_g = _etc_clamp(base_g - mod_small), c2_b = _etc_clamp(base_b - mod_small);
	int32_t c3_r = _etc_clamp(base_r - mod_large), c3_g = _etc_clamp(base_g - mod_large), c3_b = _etc_clamp(base_b - mod_large);

	// Pack colors into SIMD vectors for parallel distance computation
	// Layout: [c0_r, c1_r, c2_r, c3_r] for each channel
	__m128i colors_r = _mm_set_epi32(c3_r, c2_r, c1_r, c0_r);
	__m128i colors_g = _mm_set_epi32(c3_g, c2_g, c1_g, c0_g);
	__m128i colors_b = _mm_set_epi32(c3_b, c2_b, c1_b, c0_b);

	int32_t total_error = 0;

	// Process 2 pixels at a time with interleaved operations to hide latency
	#define PROCESS_2PIXELS(p0, p1, i0, i1) do { \
		/* Load pixel as 32-bit, extract channels via shuffle */ \
		__m128i pix0 = _mm_cvtsi32_si128(*(const int32_t*)(p0)); \
		__m128i pix1 = _mm_cvtsi32_si128(*(const int32_t*)(p1)); \
		__m128i pr0 = _mm_shuffle_epi32(_mm_and_si128(pix0, _mm_set1_epi32(0xFF)), 0); \
		__m128i pg0 = _mm_shuffle_epi32(_mm_and_si128(_mm_srli_epi32(pix0, 8), _mm_set1_epi32(0xFF)), 0); \
		__m128i pb0 = _mm_shuffle_epi32(_mm_and_si128(_mm_srli_epi32(pix0, 16), _mm_set1_epi32(0xFF)), 0); \
		__m128i pr1 = _mm_shuffle_epi32(_mm_and_si128(pix1, _mm_set1_epi32(0xFF)), 0); \
		__m128i pg1 = _mm_shuffle_epi32(_mm_and_si128(_mm_srli_epi32(pix1, 8), _mm_set1_epi32(0xFF)), 0); \
		__m128i pb1 = _mm_shuffle_epi32(_mm_and_si128(_mm_srli_epi32(pix1, 16), _mm_set1_epi32(0xFF)), 0); \
		__m128i dr0 = _mm_sub_epi32(pr0, colors_r), dr1 = _mm_sub_epi32(pr1, colors_r); \
		__m128i dg0 = _mm_sub_epi32(pg0, colors_g), dg1 = _mm_sub_epi32(pg1, colors_g); \
		__m128i db0 = _mm_sub_epi32(pb0, colors_b), db1 = _mm_sub_epi32(pb1, colors_b); \
		__m128i sq_r0 = _mm_mullo_epi32(dr0, dr0), sq_r1 = _mm_mullo_epi32(dr1, dr1); \
		__m128i sq_g0 = _mm_mullo_epi32(dg0, dg0), sq_g1 = _mm_mullo_epi32(dg1, dg1); \
		__m128i sq_b0 = _mm_mullo_epi32(db0, db0), sq_b1 = _mm_mullo_epi32(db1, db1); \
		__m128i dist0 = _mm_add_epi32(_mm_add_epi32(sq_r0, sq_g0), sq_b0); \
		__m128i dist1 = _mm_add_epi32(_mm_add_epi32(sq_r1, sq_g1), sq_b1); \
		__m128i min0a = _mm_min_epi32(dist0, _mm_shuffle_epi32(dist0, _MM_SHUFFLE(2, 3, 0, 1))); \
		__m128i min1a = _mm_min_epi32(dist1, _mm_shuffle_epi32(dist1, _MM_SHUFFLE(2, 3, 0, 1))); \
		__m128i min0b = _mm_min_epi32(min0a, _mm_shuffle_epi32(min0a, _MM_SHUFFLE(1, 0, 3, 2))); \
		__m128i min1b = _mm_min_epi32(min1a, _mm_shuffle_epi32(min1a, _MM_SHUFFLE(1, 0, 3, 2))); \
		total_error += _mm_cvtsi128_si32(min0b) + _mm_cvtsi128_si32(min1b); \
		out_indices[i0] = __builtin_ctz(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(dist0, min0b)))); \
		out_indices[i1] = __builtin_ctz(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(dist1, min1b)))); \
	} while(0)

	const uint8_t* base = rgba + y0 * stride + x0 * 4;
	if (w == 4) {
		// 4x2 subblock: 2 rows of 4 pixels
		PROCESS_2PIXELS(base + 0,  base + 4,  0, 1);
		PROCESS_2PIXELS(base + 8,  base + 12, 2, 3);
		base += stride;
		PROCESS_2PIXELS(base + 0,  base + 4,  4, 5);
		PROCESS_2PIXELS(base + 8,  base + 12, 6, 7);
	} else {
		// 2x4 subblock: 4 rows of 2 pixels
		PROCESS_2PIXELS(base + 0, base + 4, 0, 1); base += stride;
		PROCESS_2PIXELS(base + 0, base + 4, 2, 3); base += stride;
		PROCESS_2PIXELS(base + 0, base + 4, 4, 5); base += stride;
		PROCESS_2PIXELS(base + 0, base + 4, 6, 7);
	}
	#undef PROCESS_2PIXELS

	return total_error;
}
#elif ETC_USE_NEON
static int32_t _etc_subblock_error(const uint8_t* rgba, int32_t stride,
                                   int32_t x0, int32_t y0, int32_t w, int32_t h,
                                   int32_t base_r, int32_t base_g, int32_t base_b,
                                   int32_t table_idx, int32_t early_out,
                                   uint8_t* out_indices) {
	int32_t mod_small = _etc_modifier_table[table_idx][0];
	int32_t mod_large = _etc_modifier_table[table_idx][1];

	// Precompute the 4 candidate colors
	int32_t c0_r = _etc_clamp(base_r + mod_small), c0_g = _etc_clamp(base_g + mod_small), c0_b = _etc_clamp(base_b + mod_small);
	int32_t c1_r = _etc_clamp(base_r + mod_large), c1_g = _etc_clamp(base_g + mod_large), c1_b = _etc_clamp(base_b + mod_large);
	int32_t c2_r = _etc_clamp(base_r - mod_small), c2_g = _etc_clamp(base_g - mod_small), c2_b = _etc_clamp(base_b - mod_small);
	int32_t c3_r = _etc_clamp(base_r - mod_large), c3_g = _etc_clamp(base_g - mod_large), c3_b = _etc_clamp(base_b - mod_large);

	// Pack colors into NEON vectors: [c0, c1, c2, c3]
	int32x4_t colors_r = {c0_r, c1_r, c2_r, c3_r};
	int32x4_t colors_g = {c0_g, c1_g, c2_g, c3_g};
	int32x4_t colors_b = {c0_b, c1_b, c2_b, c3_b};

	int32_t total_error = 0;

	#define PROCESS_PIXEL_NEON(p, idx) do { \
		int32x4_t pr = vdupq_n_s32((p)[0]); \
		int32x4_t pg = vdupq_n_s32((p)[1]); \
		int32x4_t pb = vdupq_n_s32((p)[2]); \
		int32x4_t dr = vsubq_s32(pr, colors_r); \
		int32x4_t dg = vsubq_s32(pg, colors_g); \
		int32x4_t db = vsubq_s32(pb, colors_b); \
		int32x4_t dist = vaddq_s32(vaddq_s32(vmulq_s32(dr, dr), vmulq_s32(dg, dg)), vmulq_s32(db, db)); \
		int32_t best_err = vminvq_s32(dist); \
		total_error += best_err; \
		uint32x4_t cmp = vceqq_s32(dist, vdupq_n_s32(best_err)); \
		uint32_t mask = (vgetq_lane_u32(cmp, 0) & 1) | ((vgetq_lane_u32(cmp, 1) & 1) << 1) | \
		                ((vgetq_lane_u32(cmp, 2) & 1) << 2) | ((vgetq_lane_u32(cmp, 3) & 1) << 3); \
		out_indices[idx] = __builtin_ctz(mask); \
	} while(0)

	const uint8_t* base = rgba + y0 * stride + x0 * 4;
	if (w == 4) {
		PROCESS_PIXEL_NEON(base + 0, 0); PROCESS_PIXEL_NEON(base + 4, 1);
		PROCESS_PIXEL_NEON(base + 8, 2); PROCESS_PIXEL_NEON(base + 12, 3);
		base += stride;
		PROCESS_PIXEL_NEON(base + 0, 4); PROCESS_PIXEL_NEON(base + 4, 5);
		PROCESS_PIXEL_NEON(base + 8, 6); PROCESS_PIXEL_NEON(base + 12, 7);
	} else {
		PROCESS_PIXEL_NEON(base + 0, 0); PROCESS_PIXEL_NEON(base + 4, 1); base += stride;
		PROCESS_PIXEL_NEON(base + 0, 2); PROCESS_PIXEL_NEON(base + 4, 3); base += stride;
		PROCESS_PIXEL_NEON(base + 0, 4); PROCESS_PIXEL_NEON(base + 4, 5); base += stride;
		PROCESS_PIXEL_NEON(base + 0, 6); PROCESS_PIXEL_NEON(base + 4, 7);
	}
	#undef PROCESS_PIXEL_NEON

	return total_error;
}
#else
static int32_t _etc_subblock_error(const uint8_t* rgba, int32_t stride,
                                   int32_t x0, int32_t y0, int32_t w, int32_t h,
                                   int32_t base_r, int32_t base_g, int32_t base_b,
                                   int32_t table_idx, int32_t early_out,
                                   uint8_t* out_indices) {
	int32_t mod_small = _etc_modifier_table[table_idx][0];
	int32_t mod_large = _etc_modifier_table[table_idx][1];

	// Precompute the 4 possible colors into local vars
	int32_t c0_r = _etc_clamp(base_r + mod_small);
	int32_t c0_g = _etc_clamp(base_g + mod_small);
	int32_t c0_b = _etc_clamp(base_b + mod_small);
	int32_t c1_r = _etc_clamp(base_r + mod_large);
	int32_t c1_g = _etc_clamp(base_g + mod_large);
	int32_t c1_b = _etc_clamp(base_b + mod_large);
	int32_t c2_r = _etc_clamp(base_r - mod_small);
	int32_t c2_g = _etc_clamp(base_g - mod_small);
	int32_t c2_b = _etc_clamp(base_b - mod_small);
	int32_t c3_r = _etc_clamp(base_r - mod_large);
	int32_t c3_g = _etc_clamp(base_g - mod_large);
	int32_t c3_b = _etc_clamp(base_b - mod_large);

	int32_t total_error = 0;
	int32_t idx = 0;

	for (int32_t y = y0; y < y0 + h; y++) {
		const uint8_t* row = rgba + y * stride;
		for (int32_t x = x0; x < x0 + w; x++) {
			const uint8_t* p = row + x * 4;
			int32_t pr = p[0], pg = p[1], pb = p[2];

			// Unrolled: compute all 4 distances
			int32_t d0 = (pr-c0_r)*(pr-c0_r) + (pg-c0_g)*(pg-c0_g) + (pb-c0_b)*(pb-c0_b);
			int32_t d1 = (pr-c1_r)*(pr-c1_r) + (pg-c1_g)*(pg-c1_g) + (pb-c1_b)*(pb-c1_b);
			int32_t d2 = (pr-c2_r)*(pr-c2_r) + (pg-c2_g)*(pg-c2_g) + (pb-c2_b)*(pb-c2_b);
			int32_t d3 = (pr-c3_r)*(pr-c3_r) + (pg-c3_g)*(pg-c3_g) + (pb-c3_b)*(pb-c3_b);

			// Find minimum
			int32_t best_err = d0, best_idx = 0;
			if (d1 < best_err) { best_err = d1; best_idx = 1; }
			if (d2 < best_err) { best_err = d2; best_idx = 2; }
			if (d3 < best_err) { best_err = d3; best_idx = 3; }

			total_error += best_err;
			if (out_indices) out_indices[idx] = best_idx;
			idx++;

			// Early exit if we can't beat current best
			if (total_error >= early_out) return INT32_MAX;
		}
	}

	return total_error;
}
#endif

// Compute average color of a sub-block region
#if ETC_USE_SSE
static void _etc_subblock_average(const uint8_t* rgba, int32_t stride,
                                  int32_t x0, int32_t y0, int32_t w, int32_t h,
                                  int32_t* out_r, int32_t* out_g, int32_t* out_b) {
	// 8 pixels total - either 4x2 or 2x4
	__m128i sum = _mm_setzero_si128();
	const uint8_t* base = rgba + y0 * stride + x0 * 4;

	if (w == 4) {
		// 4x2: 2 contiguous rows of 4 pixels
		__m128i row0 = _mm_loadu_si128((const __m128i*)(base));
		__m128i row1 = _mm_loadu_si128((const __m128i*)(base + stride));
		// Unpack to 16-bit and add
		sum = _mm_add_epi16(_mm_unpacklo_epi8(row0, _mm_setzero_si128()),
		                    _mm_unpackhi_epi8(row0, _mm_setzero_si128()));
		sum = _mm_add_epi16(sum, _mm_unpacklo_epi8(row1, _mm_setzero_si128()));
		sum = _mm_add_epi16(sum, _mm_unpackhi_epi8(row1, _mm_setzero_si128()));
	} else {
		// 2x4: 4 rows of 2 pixels (8 bytes each)
		__m128i row01 = _mm_set_epi64x(*(int64_t*)(base + stride), *(int64_t*)(base));
		__m128i row23 = _mm_set_epi64x(*(int64_t*)(base + 3*stride), *(int64_t*)(base + 2*stride));
		// Unpack to 16-bit and add
		sum = _mm_add_epi16(_mm_unpacklo_epi8(row01, _mm_setzero_si128()),
		                    _mm_unpackhi_epi8(row01, _mm_setzero_si128()));
		sum = _mm_add_epi16(sum, _mm_unpacklo_epi8(row23, _mm_setzero_si128()));
		sum = _mm_add_epi16(sum, _mm_unpackhi_epi8(row23, _mm_setzero_si128()));
	}

	// sum now has: [R_even, G_even, B_even, A_even, R_odd, G_odd, B_odd, A_odd]
	// where _even = pixels 0,2,4,6 and _odd = pixels 1,3,5,7
	// Shuffle to swap halves and add to combine even+odd
	sum = _mm_add_epi16(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
	// Now vals[0..3] contain the final sums (vals[4..7] are duplicates)
	int16_t vals[8];
	_mm_storeu_si128((__m128i*)vals, sum);

	*out_r = (vals[0] + 4) / 8;  // count is always 8
	*out_g = (vals[1] + 4) / 8;
	*out_b = (vals[2] + 4) / 8;
}
#elif ETC_USE_NEON
static void _etc_subblock_average(const uint8_t* rgba, int32_t stride,
                                  int32_t x0, int32_t y0, int32_t w, int32_t h,
                                  int32_t* out_r, int32_t* out_g, int32_t* out_b) {
	// 8 pixels total - either 4x2 or 2x4
	uint16x8_t sum = vdupq_n_u16(0);
	const uint8_t* base = rgba + y0 * stride + x0 * 4;

	if (w == 4) {
		// 4x2: 2 contiguous rows of 4 pixels (16 bytes each)
		uint8x16_t row0 = vld1q_u8(base);
		uint8x16_t row1 = vld1q_u8(base + stride);
		// Unpack to 16-bit and add
		sum = vaddl_u8(vget_low_u8(row0), vget_high_u8(row0));
		sum = vaddw_u8(sum, vget_low_u8(row1));
		sum = vaddw_u8(sum, vget_high_u8(row1));
	} else {
		// 2x4: 4 rows of 2 pixels (8 bytes each)
		uint8x8_t row0 = vld1_u8(base);
		uint8x8_t row1 = vld1_u8(base + stride);
		uint8x8_t row2 = vld1_u8(base + 2*stride);
		uint8x8_t row3 = vld1_u8(base + 3*stride);
		sum = vaddl_u8(row0, row1);
		sum = vaddw_u8(sum, row2);
		sum = vaddw_u8(sum, row3);
	}

	// sum now has interleaved RGBA sums for even/odd pixels
	// Extract and combine: sum[0]+sum[4]=R, sum[1]+sum[5]=G, sum[2]+sum[6]=B
	uint16_t vals[8];
	vst1q_u16(vals, sum);

	*out_r = (vals[0] + vals[4] + 4) / 8;
	*out_g = (vals[1] + vals[5] + 4) / 8;
	*out_b = (vals[2] + vals[6] + 4) / 8;
}
#else
static void _etc_subblock_average(const uint8_t* rgba, int32_t stride,
                                  int32_t x0, int32_t y0, int32_t w, int32_t h,
                                  int32_t* out_r, int32_t* out_g, int32_t* out_b) {
	int32_t sum_r = 0, sum_g = 0, sum_b = 0;
	int32_t count = w * h;

	for (int32_t y = y0; y < y0 + h; y++) {
		for (int32_t x = x0; x < x0 + w; x++) {
			const uint8_t* p = rgba + y * stride + x * 4;
			sum_r += p[0];
			sum_g += p[1];
			sum_b += p[2];
		}
	}

	*out_r = (sum_r + count/2) / count;
	*out_g = (sum_g + count/2) / count;
	*out_b = (sum_b + count/2) / count;
}
#endif

// Try encoding a sub-block with given base color, find best table
// Uses variance-based heuristic to try fewer tables
static int32_t _etc_encode_subblock(const uint8_t* rgba, int32_t stride,
                                    int32_t x0, int32_t y0, int32_t w, int32_t h,
                                    int32_t base_r, int32_t base_g, int32_t base_b,
                                    int32_t* out_table, uint8_t* out_indices) {
	// Quick pass: find max deviation from base color
	int32_t max_dev = 0;

#if ETC_USE_SSE
	// SSE path: find max channel deviation with unrolled loop
	__m128i base = _mm_set1_epi32(base_r | (base_g << 8) | (base_b << 16));
	__m128i max_sad = _mm_setzero_si128();

	#define MAX_DEV_PIXEL(p) do { \
		__m128i pixel = _mm_set1_epi32(*(const int32_t*)(p)); \
		__m128i diff = _mm_or_si128(_mm_subs_epu8(pixel, base), _mm_subs_epu8(base, pixel)); \
		__m128i diff_g = _mm_srli_epi32(diff, 8); \
		__m128i diff_b = _mm_srli_epi32(diff, 16); \
		max_sad = _mm_max_epu8(max_sad, _mm_max_epu8(_mm_max_epu8(diff, diff_g), diff_b)); \
	} while(0)

	const uint8_t* bp = rgba + y0 * stride + x0 * 4;
	if (w == 4) {
		// 4x2: 2 rows of 4 pixels
		MAX_DEV_PIXEL(bp + 0); MAX_DEV_PIXEL(bp + 4); MAX_DEV_PIXEL(bp + 8); MAX_DEV_PIXEL(bp + 12);
		bp += stride;
		MAX_DEV_PIXEL(bp + 0); MAX_DEV_PIXEL(bp + 4); MAX_DEV_PIXEL(bp + 8); MAX_DEV_PIXEL(bp + 12);
	} else {
		// 2x4: 4 rows of 2 pixels
		MAX_DEV_PIXEL(bp + 0); MAX_DEV_PIXEL(bp + 4); bp += stride;
		MAX_DEV_PIXEL(bp + 0); MAX_DEV_PIXEL(bp + 4); bp += stride;
		MAX_DEV_PIXEL(bp + 0); MAX_DEV_PIXEL(bp + 4); bp += stride;
		MAX_DEV_PIXEL(bp + 0); MAX_DEV_PIXEL(bp + 4);
	}
	#undef MAX_DEV_PIXEL

	max_dev = _mm_cvtsi128_si32(max_sad) & 0xFF;
#elif ETC_USE_NEON
	// NEON path: find max channel deviation with unrolled loop
	uint32x4_t base_u32 = vdupq_n_u32(base_r | (base_g << 8) | (base_b << 16));
	uint8x16_t base_u8  = vreinterpretq_u8_u32(base_u32);
	uint8x16_t max_sad  = vdupq_n_u8(0);

	#define MAX_DEV_PIXEL_NEON(p) do { \
		uint32x4_t pixel_u32 = vdupq_n_u32(*(const uint32_t*)(p)); \
		uint8x16_t pixel_u8  = vreinterpretq_u8_u32(pixel_u32); \
		uint8x16_t diff      = vabdq_u8(pixel_u8, base_u8); \
		uint32x4_t diff32    = vreinterpretq_u32_u8(diff); \
		uint32x4_t diff_g    = vshrq_n_u32(diff32, 8); \
		uint32x4_t diff_b    = vshrq_n_u32(diff32, 16); \
		uint8x16_t max_rgb   = vmaxq_u8(vmaxq_u8(diff, vreinterpretq_u8_u32(diff_g)), vreinterpretq_u8_u32(diff_b)); \
		max_sad = vmaxq_u8(max_sad, max_rgb); \
	} while(0)

	const uint8_t* bp = rgba + y0 * stride + x0 * 4;
	if (w == 4) {
		// 4x2: 2 rows of 4 pixels
		MAX_DEV_PIXEL_NEON(bp + 0); MAX_DEV_PIXEL_NEON(bp + 4); MAX_DEV_PIXEL_NEON(bp + 8); MAX_DEV_PIXEL_NEON(bp + 12);
		bp += stride;
		MAX_DEV_PIXEL_NEON(bp + 0); MAX_DEV_PIXEL_NEON(bp + 4); MAX_DEV_PIXEL_NEON(bp + 8); MAX_DEV_PIXEL_NEON(bp + 12);
	} else {
		// 2x4: 4 rows of 2 pixels
		MAX_DEV_PIXEL_NEON(bp + 0); MAX_DEV_PIXEL_NEON(bp + 4); bp += stride;
		MAX_DEV_PIXEL_NEON(bp + 0); MAX_DEV_PIXEL_NEON(bp + 4); bp += stride;
		MAX_DEV_PIXEL_NEON(bp + 0); MAX_DEV_PIXEL_NEON(bp + 4); bp += stride;
		MAX_DEV_PIXEL_NEON(bp + 0); MAX_DEV_PIXEL_NEON(bp + 4);
	}
	#undef MAX_DEV_PIXEL_NEON

	max_dev = vgetq_lane_u8(max_sad, 0);
#else
	for (int32_t y = y0; y < y0 + h; y++) {
		const uint8_t* row = rgba + y * stride;
		for (int32_t x = x0; x < x0 + w; x++) {
			const uint8_t* p = row + x * 4;
			int32_t dr = p[0] - base_r; if (dr < 0) dr = -dr;
			int32_t dg = p[1] - base_g; if (dg < 0) dg = -dg;
			int32_t db = p[2] - base_b; if (db < 0) db = -db;
			int32_t dev = dr > dg ? dr : dg;
			if (db > dev) dev = db;
			if (dev > max_dev) max_dev = dev;
		}
	}
#endif

	// Map max deviation to starting table (modifier_table[t][1] is the large value)
	// Tables: 8, 17, 29, 42, 60, 80, 106, 183
	int32_t start_t;
	if      (max_dev <=  8) start_t = 0;
	else if (max_dev <= 17) start_t = 1;
	else if (max_dev <= 29) start_t = 2;
	else if (max_dev <= 42) start_t = 3;
	else if (max_dev <= 60) start_t = 4;
	else if (max_dev <= 80) start_t = 5;
	else if (max_dev <= 106) start_t = 6;
	else start_t = 7;

	// Single table - just use the predicted one
	*out_table = start_t;
	return _etc_subblock_error(rgba, stride, x0, y0, w, h,
	                           base_r, base_g, base_b, start_t,
	                           INT32_MAX, out_indices);
}

// ETC2 T/H mode distance table (Table 131 in spec)
static const int32_t _etc_th_distance_table[8] = {
	3, 6, 11, 16, 23, 32, 41, 64
};

// Calculate error for T/H mode with 4 paint colors
// Returns INT32_MAX if error exceeds early_out threshold (skip losing configs)
static int32_t _etc_th_block_error(const uint8_t* rgba, int32_t stride,
                                   int32_t paint[4][3], int32_t early_out,
                                   uint8_t* out_indices) {
	// Preload paint colors to local vars (helps compiler optimize)
	int32_t p0_r = paint[0][0], p0_g = paint[0][1], p0_b = paint[0][2];
	int32_t p1_r = paint[1][0], p1_g = paint[1][1], p1_b = paint[1][2];
	int32_t p2_r = paint[2][0], p2_g = paint[2][1], p2_b = paint[2][2];
	int32_t p3_r = paint[3][0], p3_g = paint[3][1], p3_b = paint[3][2];

	int32_t total_error = 0;

	for (int32_t y = 0; y < 4; y++) {
		const uint8_t* row = rgba + y * stride;
		for (int32_t x = 0; x < 4; x++) {
			const uint8_t* p = row + x * 4;
			int32_t pr = p[0], pg = p[1], pb = p[2];

			// Unrolled: compute all 4 distances directly
			int32_t d0 = (pr-p0_r)*(pr-p0_r) + (pg-p0_g)*(pg-p0_g) + (pb-p0_b)*(pb-p0_b);
			int32_t d1 = (pr-p1_r)*(pr-p1_r) + (pg-p1_g)*(pg-p1_g) + (pb-p1_b)*(pb-p1_b);
			int32_t d2 = (pr-p2_r)*(pr-p2_r) + (pg-p2_g)*(pg-p2_g) + (pb-p2_b)*(pb-p2_b);
			int32_t d3 = (pr-p3_r)*(pr-p3_r) + (pg-p3_g)*(pg-p3_g) + (pb-p3_b)*(pb-p3_b);

			// Find minimum with branchless-friendly pattern
			int32_t best_err = d0;
			int32_t best_idx = 0;
			if (d1 < best_err) { best_err = d1; best_idx = 1; }
			if (d2 < best_err) { best_err = d2; best_idx = 2; }
			if (d3 < best_err) { best_err = d3; best_idx = 3; }

			total_error += best_err;
			if (out_indices) out_indices[y * 4 + x] = best_idx;

			// Early exit: this config can't beat current best
			if (total_error >= early_out) return INT32_MAX;
		}
	}
	return total_error;
}

// Color pair for T/H mode encoding (two base colors)
typedef struct {
	int32_t c1_r, c1_g, c1_b;
	int32_t c2_r, c2_g, c2_b;
} _etc_color_pair_t;

// Compute candidate color pairs for T/H mode using 3 strategies:
// [0] = min/max luminance pixels
// [1] = left half vs right half averages
// [2] = top half vs bottom half averages
static void _etc_compute_color_pairs(const uint8_t* rgba, int32_t stride,
                                     _etc_color_pair_t pairs[3]) {
	// Strategy 0: min/max luminance
	int32_t min_lum = INT32_MAX, max_lum = 0;
	int32_t min_r = 0, min_g = 0, min_b = 0;
	int32_t max_r = 0, max_g = 0, max_b = 0;

	// Strategy 1 & 2: half-block sums
	int32_t sum_l_r = 0, sum_l_g = 0, sum_l_b = 0;  // left  (x < 2)
	int32_t sum_r_r = 0, sum_r_g = 0, sum_r_b = 0;  // right (x >= 2)
	int32_t sum_t_r = 0, sum_t_g = 0, sum_t_b = 0;  // top   (y < 2)
	int32_t sum_b_r = 0, sum_b_g = 0, sum_b_b = 0;  // bottom (y >= 2)

	for (int32_t y = 0; y < 4; y++) {
		for (int32_t x = 0; x < 4; x++) {
			const uint8_t* p = rgba + y * stride + x * 4;
			int32_t r = p[0], g = p[1], b = p[2];
			int32_t lum = r * 2 + g * 4 + b;

			// Min/max luminance
			if (lum < min_lum) { min_lum = lum; min_r = r; min_g = g; min_b = b; }
			if (lum > max_lum) { max_lum = lum; max_r = r; max_g = g; max_b = b; }

			// Half-block sums
			if (x < 2) { sum_l_r += r; sum_l_g += g; sum_l_b += b; }
			else       { sum_r_r += r; sum_r_g += g; sum_r_b += b; }
			if (y < 2) { sum_t_r += r; sum_t_g += g; sum_t_b += b; }
			else       { sum_b_r += r; sum_b_g += g; sum_b_b += b; }
		}
	}

	// Strategy 0: min/max luminance
	pairs[0] = (_etc_color_pair_t){ min_r, min_g, min_b, max_r, max_g, max_b };

	// Strategy 1: left vs right (8 pixels each)
	pairs[1] = (_etc_color_pair_t){
		sum_l_r / 8, sum_l_g / 8, sum_l_b / 8,
		sum_r_r / 8, sum_r_g / 8, sum_r_b / 8
	};

	// Strategy 2: top vs bottom (8 pixels each)
	pairs[2] = (_etc_color_pair_t){
		sum_t_r / 8, sum_t_g / 8, sum_t_b / 8,
		sum_b_r / 8, sum_b_g / 8, sum_b_b / 8
	};
}

// Helper: Try one T-mode color pair configuration
static int32_t _etc_try_t_mode_config(const uint8_t* rgba, int32_t stride,
                                       int32_t c1_r, int32_t c1_g, int32_t c1_b,
                                       int32_t c2_r, int32_t c2_g, int32_t c2_b,
                                       uint8_t* out_block);

// Try T-mode encoding
// T-mode: paint0 = base1, paint1 = base2+d, paint2 = base2, paint3 = base2-d
// Triggered by R overflow
static int32_t _etc_try_t_mode(const uint8_t* rgba, int32_t stride, uint8_t* out_block) {
	_etc_color_pair_t pairs[3];
	_etc_compute_color_pairs(rgba, stride, pairs);

	int32_t best_error = INT32_MAX;
	for (int32_t s = 0; s < 3; s++) {
		uint8_t block[8];
		int32_t err = _etc_try_t_mode_config(rgba, stride,
			pairs[s].c1_r, pairs[s].c1_g, pairs[s].c1_b,
			pairs[s].c2_r, pairs[s].c2_g, pairs[s].c2_b, block);
		if (err < best_error) {
			best_error = err;
			memcpy(out_block, block, 8);
		}
	}
	return best_error;
}

// Try one T-mode configuration with given base colors
static int32_t _etc_try_t_mode_config(const uint8_t* rgba, int32_t stride,
                                       int32_t c1_r, int32_t c1_g, int32_t c1_b,
                                       int32_t c2_r, int32_t c2_g, int32_t c2_b,
                                       uint8_t* out_block) {
	// Quantize to 4-bit
	int32_t r1_4 = _etc_quantize4(c1_r), g1_4 = _etc_quantize4(c1_g), b1_4 = _etc_quantize4(c1_b);
	int32_t r2_4 = _etc_quantize4(c2_r), g2_4 = _etc_quantize4(c2_g), b2_4 = _etc_quantize4(c2_b);

	int32_t base1_r = _etc_expand4(r1_4), base1_g = _etc_expand4(g1_4), base1_b = _etc_expand4(b1_4);
	int32_t base2_r = _etc_expand4(r2_4), base2_g = _etc_expand4(g2_4), base2_b = _etc_expand4(b2_4);

	// Try all 8 distance values and pick best
	int32_t best_error = INT32_MAX;
	int32_t best_dist  = 0;
	uint8_t best_indices[16];

	for (int32_t di = 0; di < 8; di++) {
		int32_t d = _etc_th_distance_table[di];

		int32_t paint[4][3] = {
			{ base1_r,                   base1_g,                   base1_b                   },
			{ _etc_clamp(base2_r + d),   _etc_clamp(base2_g + d),   _etc_clamp(base2_b + d)   },
			{ base2_r,                   base2_g,                   base2_b                   },
			{ _etc_clamp(base2_r - d),   _etc_clamp(base2_g - d),   _etc_clamp(base2_b - d)   },
		};

		uint8_t indices[16];
		int32_t error = _etc_th_block_error(rgba, stride, paint, best_error, indices);

		if (error < best_error) {
			best_error = error;
			best_dist  = di;
			memcpy(best_indices, indices, 16);
		}
	}

	// Pack T-mode block - triggers R overflow
	// Strategy depends on R1 value: use positive overflow for {7,10,11,13,14,15}
	int32_t use_pos = (r1_4 == 7 || r1_4 == 10 || r1_4 == 11 ||
	                   r1_4 == 13 || r1_4 == 14 || r1_4 == 15);
	uint64_t block = 0;
	if (use_pos) block |= ((uint64_t)0x7) << 61;
	block |= ((uint64_t)((r1_4 >> 2) & 0x3)) << 59;
	if (!use_pos) block |= ((uint64_t)1) << 58;
	block |= ((uint64_t)(r1_4 & 0x3))          << 56;
	block |= ((uint64_t)(g1_4 & 0xF))          << 52;
	block |= ((uint64_t)(b1_4 & 0xF))          << 48;
	block |= ((uint64_t)(r2_4 & 0xF))          << 44;
	block |= ((uint64_t)(g2_4 & 0xF))          << 40;
	block |= ((uint64_t)(b2_4 & 0xF))          << 36;
	block |= ((uint64_t)((best_dist >> 1) & 0x3)) << 34;
	block |= ((uint64_t)1)                     << 33;
	block |= ((uint64_t)(best_dist & 0x1))     << 32;

	uint16_t msb, lsb;
	_etc_pack_indices(best_indices, &msb, &lsb);
	block |= ((uint64_t)msb) << 16;
	block |= ((uint64_t)lsb);

	_etc_write_block(block, out_block);
	return best_error;
}

// Helper: Try one H-mode color pair configuration
static int32_t _etc_try_h_mode_config(const uint8_t* rgba, int32_t stride,
                                       int32_t c1_r, int32_t c1_g, int32_t c1_b,
                                       int32_t c2_r, int32_t c2_g, int32_t c2_b,
                                       uint8_t* out_block);

// Try H-mode encoding
// H-mode: paint0 = base1+d, paint1 = base1-d, paint2 = base2+d, paint3 = base2-d
// Triggered by G overflow
static int32_t _etc_try_h_mode(const uint8_t* rgba, int32_t stride, uint8_t* out_block) {
	_etc_color_pair_t pairs[3];
	_etc_compute_color_pairs(rgba, stride, pairs);

	int32_t best_error = INT32_MAX;
	for (int32_t s = 0; s < 3; s++) {
		uint8_t block[8];
		int32_t err = _etc_try_h_mode_config(rgba, stride,
			pairs[s].c1_r, pairs[s].c1_g, pairs[s].c1_b,
			pairs[s].c2_r, pairs[s].c2_g, pairs[s].c2_b, block);
		if (err < best_error) {
			best_error = err;
			memcpy(out_block, block, 8);
		}
	}
	return best_error;
}

// Try one H-mode configuration with given base colors
static int32_t _etc_try_h_mode_config(const uint8_t* rgba, int32_t stride,
                                       int32_t c1_r, int32_t c1_g, int32_t c1_b,
                                       int32_t c2_r, int32_t c2_g, int32_t c2_b,
                                       uint8_t* out_block) {
	// Quantize to 4-bit
	int32_t r1_4 = _etc_quantize4(c1_r), g1_4 = _etc_quantize4(c1_g), b1_4 = _etc_quantize4(c1_b);
	int32_t r2_4 = _etc_quantize4(c2_r), g2_4 = _etc_quantize4(c2_g), b2_4 = _etc_quantize4(c2_b);

	int32_t base1_r = _etc_expand4(r1_4), base1_g = _etc_expand4(g1_4), base1_b = _etc_expand4(b1_4);
	int32_t base2_r = _etc_expand4(r2_4), base2_g = _etc_expand4(g2_4), base2_b = _etc_expand4(b2_4);

	// H-mode distance index LSB is determined by color ordering (val1 >= val2 -> 1)
	// Try both orderings and pick the best result
	int32_t best_error = INT32_MAX;
	int32_t best_dist  = 0;
	uint8_t best_indices[16];
	int32_t best_swapped = 0;

	for (int32_t swap = 0; swap <= 1; swap++) {
		int32_t ra = swap ? r2_4 : r1_4, ga = swap ? g2_4 : g1_4, ba = swap ? b2_4 : b1_4;
		int32_t rb = swap ? r1_4 : r2_4, gb = swap ? g1_4 : g2_4, bb = swap ? b1_4 : b2_4;
		int32_t ba_r = swap ? base2_r : base1_r, ba_g = swap ? base2_g : base1_g, ba_b = swap ? base2_b : base1_b;
		int32_t bb_r = swap ? base1_r : base2_r, bb_g = swap ? base1_g : base2_g, bb_b = swap ? base1_b : base2_b;

		int32_t val_a = (ra << 8) | (ga << 4) | ba;
		int32_t val_b = (rb << 8) | (gb << 4) | bb;
		int32_t ordering_bit = (val_a >= val_b) ? 1 : 0;

		for (int32_t di = 0; di < 8; di++) {
			if ((di & 1) != ordering_bit) continue;
			int32_t d = _etc_th_distance_table[di];

			int32_t paint[4][3] = {
				{ _etc_clamp(ba_r + d), _etc_clamp(ba_g + d), _etc_clamp(ba_b + d) },
				{ _etc_clamp(ba_r - d), _etc_clamp(ba_g - d), _etc_clamp(ba_b - d) },
				{ _etc_clamp(bb_r + d), _etc_clamp(bb_g + d), _etc_clamp(bb_b + d) },
				{ _etc_clamp(bb_r - d), _etc_clamp(bb_g - d), _etc_clamp(bb_b - d) },
			};

			uint8_t indices[16];
			int32_t error = _etc_th_block_error(rgba, stride, paint, best_error, indices);

			if (error < best_error) {
				best_error   = error;
				best_dist    = di;
				best_swapped = swap;
				memcpy(best_indices, indices, 16);
			}
		}
	}

	// Use the winning color order for packing
	if (best_swapped) {
		int32_t tmp;
		tmp = r1_4; r1_4 = r2_4; r2_4 = tmp;
		tmp = g1_4; g1_4 = g2_4; g2_4 = tmp;
		tmp = b1_4; b1_4 = b2_4; b2_4 = tmp;
	}

	// Pack H-mode block - triggers G overflow
	int32_t g_sum = ((g1_4 & 1) * 2) + (((b1_4 >> 2) & 1) * 2) + ((b1_4 >> 3) & 1) + ((b1_4 >> 1) & 1);
	int32_t use_pos = (g_sum >= 4);

	uint64_t block = 0;
	if ((g1_4 >> 3) & 1) block |= ((uint64_t)1) << 63;
	block |= ((uint64_t)(r1_4 & 0xF))          << 59;
	block |= ((uint64_t)((g1_4 >> 1) & 0x7))   << 56;
	if (use_pos) block |= ((uint64_t)0x7)      << 53;
	block |= ((uint64_t)(g1_4 & 0x1))          << 52;
	block |= ((uint64_t)((b1_4 >> 3) & 0x1))   << 51;
	if (!use_pos) block |= ((uint64_t)1)       << 50;
	block |= ((uint64_t)(b1_4 & 0x7))          << 47;
	block |= ((uint64_t)(r2_4 & 0xF))          << 43;
	block |= ((uint64_t)(g2_4 & 0xF))          << 39;
	block |= ((uint64_t)(b2_4 & 0xF))          << 35;
	block |= ((uint64_t)((best_dist >> 2) & 0x1)) << 34;
	block |= ((uint64_t)1)                     << 33;
	block |= ((uint64_t)((best_dist >> 1) & 0x1)) << 32;

	uint16_t msb, lsb;
	_etc_pack_indices(best_indices, &msb, &lsb);
	block |= ((uint64_t)msb) << 16;
	block |= ((uint64_t)lsb);

	_etc_write_block(block, out_block);
	return best_error;
}

// ETC2 spec-compliant planar mode bit layout
// The layout scatters color bits with gaps for "opcode" bits that trigger overflow.
// Based on Khronos spec and Granite shader decoder.
//
// 64-bit block layout (bit 63 = MSB of byte0):
//   RO (6 bits): bits [62:57] = byte0[6:1]
//   GO (7 bits): bits [56, 54:49] = byte0[0], byte1[6:1] (bit 55 is opcode)
//   BO (6 bits): bits [48, 44:43, 41:39] (bits 47,46,45,42 are opcode)
//   RH (6 bits): bits [38:34, 32] = scattered
//   GH (7 bits): bits [31:25]
//   BH (6 bits): bits [24:19]
//   RV (6 bits): bits [18:13]
//   GV (7 bits): bits [12:6]
//   BV (6 bits): bits [5:0]
//
// The opcode bits (55, 47, 46, 45, 42) are set to cause B overflow when
// interpreted as differential mode.

// Calculate planar mode error for given colors
static int32_t _etc_planar_error(const uint8_t* rgba, int32_t stride,
                                 int32_t o_r, int32_t o_g, int32_t o_b,
                                 int32_t h_r, int32_t h_g, int32_t h_b,
                                 int32_t v_r, int32_t v_g, int32_t v_b) {
	int32_t total_error = 0;
	for (int32_t y = 0; y < 4; y++) {
		for (int32_t x = 0; x < 4; x++) {
			const uint8_t* p = rgba + y * stride + x * 4;
			int32_t pr = _etc_clamp((x * (h_r - o_r) + y * (v_r - o_r) + 4 * o_r + 2) >> 2);
			int32_t pg = _etc_clamp((x * (h_g - o_g) + y * (v_g - o_g) + 4 * o_g + 2) >> 2);
			int32_t pb = _etc_clamp((x * (h_b - o_b) + y * (v_b - o_b) + 4 * o_b + 2) >> 2);
			int32_t dr = p[0] - pr;
			int32_t dg = p[1] - pg;
			int32_t db = p[2] - pb;
			total_error += dr*dr + dg*dg + db*db;
		}
	}
	return total_error;
}

// Pack planar block using ETC2 spec-compliant bit layout
// This layout ALWAYS triggers B overflow regardless of color values
//
// Key insight: opcode bits must be ONLY in byte2 to trigger B overflow
// while keeping R and G from overflowing.
//
// Byte layout (big-endian, bit 63 = byte0 MSB):
//   byte0[7:2] = RO[5:0]      - R base, kept in valid range
//   byte0[1:0] = GO[6:5]      - G high bits
//   byte1[7:3] = GO[4:0]      - G remaining bits
//   byte1[2:0] = BO[5:3]      - B high bits
//   byte2[7:5] = BO[2:0]      - B low bits
//   byte2[4:1] = 1111         - OPCODE bits to force B overflow
//   byte2[0]   = 1            - diff flag (must be 1 for planar detection)
//   byte3[7:2] = RH[5:0]      - H red
//   byte3[1:0] = GH[6:5]      - H green high
//   byte4[7:3] = GH[4:0]      - H green low
//   byte4[2:0] = BH[5:3]      - H blue high
//   byte5[7:5] = BH[2:0]      - H blue low
//   byte5[4:0] = RV[5:1]      - V red high
//   byte6[7]   = RV[0]        - V red low
//   byte6[6:0] = GV[6:0]      - V green
//   byte7[7:2] = BV[5:0]      - V blue
//   byte7[1:0] = 0            - unused
static void _etc_pack_planar_spec(int32_t o_r, int32_t o_g, int32_t o_b,
                                  int32_t h_r, int32_t h_g, int32_t h_b,
                                  int32_t v_r, int32_t v_g, int32_t v_b,
                                  uint8_t* out) {
	// ETC2 spec-compliant planar mode bit layout
	// Uses 64-bit block with opcode bits that guarantee B overflow detection
	uint64_t block = 0;

	// RO at bits [62:57] (6 bits)
	block |= ((uint64_t)(o_r & 0x3F)) << 57;

	// GO at bits [56, 54:49] - bit 55 is opcode (set to 0 to prevent G overflow)
	block |= ((uint64_t)((o_g >> 6) & 0x01)) << 56;  // GO[6]
	// bit 55 = 0 (opcode, prevents G overflow)
	block |= ((uint64_t)(o_g & 0x3F)) << 49;          // GO[5:0]

	// BO at bits [48, 44:43, 41:39] - bits 47,46,45=0, bit 42=1 for B overflow
	block |= ((uint64_t)((o_b >> 5) & 0x01)) << 48;  // BO[5]
	// bits [47:45] = 000 (opcode, makes b0_5 small)
	block |= ((uint64_t)((o_b >> 3) & 0x03)) << 43;  // BO[4:3]
	block |= ((uint64_t)1) << 42;                     // opcode = 1 (makes db negative)
	block |= ((uint64_t)(o_b & 0x07)) << 39;          // BO[2:0]

	// RH at bits [38:34, 32] - bit 33 is diff flag (must be 1)
	block |= ((uint64_t)((h_r >> 1) & 0x1F)) << 34;  // RH[5:1]
	block |= ((uint64_t)1) << 33;                     // diff flag = 1
	block |= ((uint64_t)(h_r & 0x01)) << 32;          // RH[0]

	// GH at bits [31:25] (7 bits)
	block |= ((uint64_t)(h_g & 0x7F)) << 25;

	// BH at bits [24:19] (6 bits)
	block |= ((uint64_t)(h_b & 0x3F)) << 19;

	// RV at bits [18:13] (6 bits)
	block |= ((uint64_t)(v_r & 0x3F)) << 13;

	// GV at bits [12:6] (7 bits)
	block |= ((uint64_t)(v_g & 0x7F)) << 6;

	// BV at bits [5:0] (6 bits)
	block |= ((uint64_t)(v_b & 0x3F)) << 0;

	_etc_write_block(block, out);
}

// Check if planar encoding will trigger B overflow (required for detection)
// With our layout: b0_5 = BO[4:3], db = -4 + BO[2:1]
// Overflow when: BO[4:3] + BO[2:1] < 4
static int32_t _etc_planar_will_overflow(int32_t o_b6) {
	int32_t bo_43 = (o_b6 >> 3) & 0x03;  // BO[4:3]
	int32_t bo_21 = (o_b6 >> 1) & 0x03;  // BO[2:1]
	return (bo_43 + bo_21) < 4;
}

// Check and fix colors to prevent R/G underflow in planar mode
// The decoder extracts:
//   r0_5 = bits[63:59], dr = bits[58:56] -> r1 = r0_5 + dr
//   g0_5 = bits[55:51], dg = bits[50:48] -> g1 = g0_5 + dg
// In planar encoding:
//   r0_5 = o_r[5:2], dr = o_r[1:0] || o_g[6] (GO[6])
//   g0_5 = 0 || o_g[5:2], dg = o_g[1:0] || o_b[5] (BO[5])
// R underflow when r0_5 + dr < 0 (dr is sign-extended)
// G underflow when g0_5 + dg < 0 (dg is sign-extended)
static void _etc_planar_fix_overflow(int32_t* ref_o_r6, int32_t* ref_o_g7, int32_t* ref_o_b6) {
	int32_t o_r6 = *ref_o_r6;
	int32_t o_g7 = *ref_o_g7;
	int32_t o_b6 = *ref_o_b6;

	// Check R underflow: r0_5 = o_r6[5:2], dr_raw = o_r6[1:0] || o_g7[6]
	int32_t r0_5   = (o_r6 >> 2) & 0xF;
	int32_t dr_raw = ((o_r6 & 0x3) << 1) | ((o_g7 >> 6) & 0x1);
	int32_t dr     = (dr_raw >= 4) ? (dr_raw - 8) : dr_raw;
	int32_t r1     = r0_5 + dr;

	if (r1 < 0) {
		// R would underflow - adjust o_r6 lower bits to prevent negative dr
		// Clear o_r6[1] to make dr_raw < 4
		o_r6 = (o_r6 & ~0x2);  // Clear bit 1
		*ref_o_r6 = o_r6;
	}

	// Check G underflow: g0_5 = 0 || o_g7[5:2], dg_raw = o_g7[1:0] || o_b6[5]
	int32_t g0_5   = (o_g7 >> 2) & 0xF;  // bit 55 is always 0, so MSB of g0_5 is 0
	int32_t dg_raw = ((o_g7 & 0x3) << 1) | ((o_b6 >> 5) & 0x1);
	int32_t dg     = (dg_raw >= 4) ? (dg_raw - 8) : dg_raw;
	int32_t g1     = g0_5 + dg;

	if (g1 < 0) {
		// G would underflow - adjust o_g7 lower bits to prevent negative dg
		// Clear o_g7[1] to make dg_raw < 4
		o_g7 = (o_g7 & ~0x2);  // Clear bit 1
		*ref_o_g7 = o_g7;
	}
}

// Find nearest valid BO value that triggers overflow
// Returns the valid o_b6 closest to the target 8-bit blue value
static int32_t _etc_find_best_overflow_bo(int32_t target_b8) {
	int32_t best_b6 = 0;
	int32_t best_error = INT32_MAX;

	// Try all 64 possible 6-bit values
	for (int32_t b6 = 0; b6 < 64; b6++) {
		if (!_etc_planar_will_overflow(b6)) continue;

		// Expand to 8-bit and compute error
		int32_t b8 = _etc_expand6(b6);
		int32_t error = (b8 - target_b8) * (b8 - target_b8);

		if (error < best_error) {
			best_error = error;
			best_b6 = b6;
		}
	}

	return best_b6;
}

// Fit a plane to the block using least-squares
// Computes O, H, V colors that minimize squared error
// The planar formula is: C(x,y) = O + x*(H-O)/4 + y*(V-O)/4
// Which rearranges to: C(x,y) = O*(1 - x/4 - y/4) + H*(x/4) + V*(y/4)
// We solve for O, H, V that minimize sum of (C(x,y) - pixel(x,y))^2
static void _etc_fit_plane(const uint8_t* rgba, int32_t stride, int32_t channel,
                           int32_t* out_o, int32_t* out_h, int32_t* out_v) {
	// Build normal equations for least squares: A^T*A * [O,H,V] = A^T*b
	// Coefficients for pixel at (x,y): O gets (4-x-y)/4, H gets x/4, V gets y/4
	// Multiply by 4 to avoid fractions: O gets (4-x-y), H gets x, V gets y
	//
	// A^T*A matrix (symmetric):
	//   [sum((4-x-y)^2), sum((4-x-y)*x), sum((4-x-y)*y)]
	//   [sum((4-x-y)*x), sum(x^2),       sum(x*y)      ]
	//   [sum((4-x-y)*y), sum(x*y),       sum(y^2)      ]
	//
	// These sums are constants for a 4x4 block:
	// sum(x) = 0+1+2+3 = 6 (per row, *4 rows = 24)
	// sum(y) = same = 24
	// sum(x^2) = 0+1+4+9 = 14 (per row, *4 = 56)
	// sum(y^2) = same = 56
	// sum(x*y) = sum_y(y * sum_x(x)) = 6 * 6 = 36
	// sum(4-x-y) = 16*4 - 24 - 24 = 16
	// sum((4-x-y)^2) = sum(16 - 8x - 8y + x^2 + 2xy + y^2)
	//               = 16*16 - 8*24 - 8*24 + 56 + 2*36 + 56 = 256 - 384 + 184 = 56
	// sum((4-x-y)*x) = sum(4x - x^2 - xy) = 4*24 - 56 - 36 = 4
	// sum((4-x-y)*y) = sum(4y - xy - y^2) = 4*24 - 36 - 56 = 4

	// A^T*A (scaled by 16 for the division by 4):
	// [56, 4, 4]
	// [4, 56, 36]
	// [4, 36, 56]

	// Compute A^T*b for each channel
	int32_t sum_o = 0, sum_h = 0, sum_v = 0;
	for (int32_t y = 0; y < 4; y++) {
		for (int32_t x = 0; x < 4; x++) {
			int32_t c = rgba[y * stride + x * 4 + channel];
			sum_o += (4 - x - y) * c;
			sum_h += x * c;
			sum_v += y * c;
		}
	}

	// Solve using Cramer's rule (matrix is well-conditioned)
	// det = 56*(56*56 - 36*36) - 4*(4*56 - 36*4) + 4*(4*36 - 56*4)
	//     = 56*(3136 - 1296) - 4*(224 - 144) + 4*(144 - 224)
	//     = 56*1840 - 4*80 - 4*80 = 103040 - 640 = 102400
	int32_t det = 102400;

	// Adjugate matrix (cofactors transposed):
	// [[1840, -80, -80],
	//  [-80, 3120, -2000],
	//  [-80, -2000, 3120]]
	// Solution: [O,H,V] = adj * [sum_o,sum_h,sum_v] / det
	int32_t o8 = (sum_o * 1840 - sum_h * 80 - sum_v * 80) / (det / 4);
	int32_t h8 = (-sum_o * 80 + sum_h * 3120 - sum_v * 2000) / (det / 4);
	int32_t v8 = (-sum_o * 80 - sum_h * 2000 + sum_v * 3120) / (det / 4);

	// Clamp to valid 8-bit range
	*out_o = o8 < 0 ? 0 : (o8 > 255 ? 255 : o8);
	*out_h = h8 < 0 ? 0 : (h8 > 255 ? 255 : h8);
	*out_v = v8 < 0 ? 0 : (v8 > 255 ? 255 : v8);
}

// Try planar mode encoding using spec-compliant layout
// Returns error, fills out_block with encoded data
static int32_t _etc_try_planar(const uint8_t* rgba, int32_t stride, uint8_t* out_block) {
	// Fit optimal plane to block using least-squares
	int32_t o_r8, o_g8, o_b8, h_r8, h_g8, h_b8, v_r8, v_g8, v_b8;
	_etc_fit_plane(rgba, stride, 0, &o_r8, &h_r8, &v_r8);
	_etc_fit_plane(rgba, stride, 1, &o_g8, &h_g8, &v_g8);
	_etc_fit_plane(rgba, stride, 2, &o_b8, &h_b8, &v_b8);

	// Quantize to 6/7/6 bits
	int32_t o_r6 = (o_r8 + 2) >> 2; if (o_r6 > 63) o_r6 = 63;
	int32_t o_g7 = (o_g8 + 1) >> 1; if (o_g7 > 127) o_g7 = 127;
	int32_t o_b6 = (o_b8 + 2) >> 2; if (o_b6 > 63) o_b6 = 63;
	int32_t h_r6 = (h_r8 + 2) >> 2; if (h_r6 > 63) h_r6 = 63;
	int32_t h_g7 = (h_g8 + 1) >> 1; if (h_g7 > 127) h_g7 = 127;
	int32_t h_b6 = (h_b8 + 2) >> 2; if (h_b6 > 63) h_b6 = 63;
	int32_t v_r6 = (v_r8 + 2) >> 2; if (v_r6 > 63) v_r6 = 63;
	int32_t v_g7 = (v_g8 + 1) >> 1; if (v_g7 > 127) v_g7 = 127;
	int32_t v_b6 = (v_b8 + 2) >> 2; if (v_b6 > 63) v_b6 = 63;

	// Find best BO that triggers overflow - search all valid values for minimum error
	// Only o_b6 matters - overflow is checked on byte2 which contains BO, not BH/BV
	if (!_etc_planar_will_overflow(o_b6)) {
		o_b6 = _etc_find_best_overflow_bo(o_b8);
	}

	// Fix R/G underflow issues - must be done after B overflow is ensured
	// because this may adjust o_g7 which affects G underflow check
	_etc_planar_fix_overflow(&o_r6, &o_g7, &o_b6);

	// Re-check B overflow after fixing R/G (o_b6 might have been affected)
	if (!_etc_planar_will_overflow(o_b6)) {
		o_b6 = _etc_find_best_overflow_bo(o_b8);
	}

	// Pack using spec-compliant layout
	_etc_pack_planar_spec(o_r6, o_g7, o_b6, h_r6, h_g7, h_b6, v_r6, v_g7, v_b6, out_block);

	// Calculate error
	int32_t o_r = _etc_expand6(o_r6);
	int32_t o_g = _etc_expand7(o_g7);
	int32_t o_b = _etc_expand6(o_b6);
	int32_t h_r = _etc_expand6(h_r6);
	int32_t h_g = _etc_expand7(h_g7);
	int32_t h_b = _etc_expand6(h_b6);
	int32_t v_r = _etc_expand6(v_r6);
	int32_t v_g = _etc_expand7(v_g7);
	int32_t v_b = _etc_expand6(v_b6);

	return _etc_planar_error(rgba, stride, o_r, o_g, o_b, h_r, h_g, h_b, v_r, v_g, v_b);
}

// DEBUG: Set to force specific mode only
// 0 = all modes, 1 = differential, 2 = individual, 3 = planar, 4 = T-mode, 5 = H-mode
#define ETC2_DEBUG_MODE_ONLY 0

// Quality mode:
//   0 = fast (quick select, no fallback)
//   1 = fast with fallback
//   2 = best (try all ETC2 modes)
//   3 = ETC1 + conservative planar (no T/H modes, planar only on smooth gradients)
#ifndef ETC2_QUALITY_MODE
#define ETC2_QUALITY_MODE 3
#endif

// Error threshold for fallback (if quick-selected mode error > this, try others)
#define ETC2_FALLBACK_THRESHOLD 1600  // ~100 error per pixel = noticeable

// Mode categories for quick selection
#define ETC2_CAT_SOLID      0  // Low variance -> differential
#define ETC2_CAT_GRADIENT   1  // Linear color change -> planar
#define ETC2_CAT_TWOTONE    2  // Two distinct colors -> T or H mode
#define ETC2_CAT_COMPLEX    3  // High variance, sub-blocks differ -> individual mode

// Quick block classifier - analyzes block and returns likely best mode category
// Simplified approach: classify by color range, biased toward planar for mid-range
static int32_t _etc_classify_block(const uint8_t* rgba, int32_t stride) {
	int32_t min_r = 255, max_r = 0;
	int32_t min_g = 255, max_g = 0;
	int32_t min_b = 255, max_b = 0;

	// Sub-block sums for individual mode detection
	int32_t sum_l_r = 0, sum_l_g = 0, sum_l_b = 0;
	int32_t sum_r_r = 0, sum_r_g = 0, sum_r_b = 0;

	for (int32_t y = 0; y < 4; y++) {
		const uint8_t* row = rgba + y * stride;
		for (int32_t x = 0; x < 4; x++) {
			int32_t r = row[x * 4 + 0];
			int32_t g = row[x * 4 + 1];
			int32_t b = row[x * 4 + 2];

			if (r < min_r) min_r = r; if (r > max_r) max_r = r;
			if (g < min_g) min_g = g; if (g > max_g) max_g = g;
			if (b < min_b) min_b = b; if (b > max_b) max_b = b;

			if (x < 2) { sum_l_r += r; sum_l_g += g; sum_l_b += b; }
			else       { sum_r_r += r; sum_r_g += g; sum_r_b += b; }
		}
	}

	int32_t range_r = max_r - min_r;
	int32_t range_g = max_g - min_g;
	int32_t range_b = max_b - min_b;
	int32_t max_range = range_r > range_g ? range_r : range_g;
	if (range_b > max_range) max_range = range_b;

	// Sub-block divergence check for individual mode
	int32_t diff_r = (sum_l_r - sum_r_r) / 8;
	int32_t diff_g = (sum_l_g - sum_r_g) / 8;
	int32_t diff_b = (sum_l_b - sum_r_b) / 8;
	int32_t subblock_diff = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;

	// Very high sub-block difference -> individual mode
	if (subblock_diff > 200) {
		return ETC2_CAT_COMPLEX;
	}

	// High contrast (sharp edges, text) -> T/H modes
	if (max_range > 80) {
		return ETC2_CAT_TWOTONE;
	}

	// Planar is surprisingly good for a wide range of blocks
	// Even subtle variation often encodes better with planar's 3-point interpolation
	// Only truly solid blocks (range < 8) should use differential
	if (max_range > 8) {
		return ETC2_CAT_GRADIENT;
	}

	// Nearly solid block - differential is perfect
	return ETC2_CAT_SOLID;
}

// Helper: Try individual mode (RGB444 + RGB444, independent sub-blocks)
static int32_t _etc_try_individual(const uint8_t* rgba, int32_t stride, uint8_t* out_block) {
	static const int32_t subblock_bounds[2][2][4] = {
		{{0, 0, 2, 4}, {2, 0, 2, 4}},
		{{0, 0, 4, 2}, {0, 2, 4, 2}},
	};

	int32_t best_error = INT32_MAX;

	for (int32_t flip = 0; flip <= 1; flip++) {
		int32_t x0_0 = subblock_bounds[flip][0][0], y0_0 = subblock_bounds[flip][0][1];
		int32_t w0   = subblock_bounds[flip][0][2], h0   = subblock_bounds[flip][0][3];
		int32_t x0_1 = subblock_bounds[flip][1][0], y0_1 = subblock_bounds[flip][1][1];
		int32_t w1   = subblock_bounds[flip][1][2], h1   = subblock_bounds[flip][1][3];

		int32_t avg0_r, avg0_g, avg0_b, avg1_r, avg1_g, avg1_b;
		_etc_subblock_average(rgba, stride, x0_0, y0_0, w0, h0, &avg0_r, &avg0_g, &avg0_b);
		_etc_subblock_average(rgba, stride, x0_1, y0_1, w1, h1, &avg1_r, &avg1_g, &avg1_b);

		int32_t r0_4 = _etc_quantize4(avg0_r), g0_4 = _etc_quantize4(avg0_g), b0_4 = _etc_quantize4(avg0_b);
		int32_t r1_4 = _etc_quantize4(avg1_r), g1_4 = _etc_quantize4(avg1_g), b1_4 = _etc_quantize4(avg1_b);
		int32_t base0_r = _etc_expand4(r0_4), base0_g = _etc_expand4(g0_4), base0_b = _etc_expand4(b0_4);
		int32_t base1_r = _etc_expand4(r1_4), base1_g = _etc_expand4(g1_4), base1_b = _etc_expand4(b1_4);

		int32_t table0, table1;
		uint8_t indices0[8], indices1[8];
		int32_t err0 = _etc_encode_subblock(rgba, stride, x0_0, y0_0, w0, h0, base0_r, base0_g, base0_b, &table0, indices0);
		int32_t err1 = _etc_encode_subblock(rgba, stride, x0_1, y0_1, w1, h1, base1_r, base1_g, base1_b, &table1, indices1);

		int32_t total_error = err0 + err1;
		if (total_error < best_error) {
			best_error = total_error;
			out_block[0] = (r0_4 << 4) | r1_4;
			out_block[1] = (g0_4 << 4) | g1_4;
			out_block[2] = (b0_4 << 4) | b1_4;
			out_block[3] = (table0 << 5) | (table1 << 2) | (0 << 1) | flip;
			uint16_t msb, lsb;
			_etc_pack_subblock_indices(indices0, indices1, flip, &msb, &lsb);
			out_block[4] = (msb >> 8) & 0xFF;
			out_block[5] = (msb >> 0) & 0xFF;
			out_block[6] = (lsb >> 8) & 0xFF;
			out_block[7] = (lsb >> 0) & 0xFF;
		}
	}
	return best_error;
}

// Helper: Try differential mode (the most common/versatile mode)
static int32_t _etc_try_differential(const uint8_t* rgba, int32_t stride, uint8_t* out_block) {
	static const int32_t subblock_bounds[2][2][4] = {
		{{0, 0, 2, 4}, {2, 0, 2, 4}},
		{{0, 0, 4, 2}, {0, 2, 4, 2}},
	};

	int32_t best_error = INT32_MAX;

	for (int32_t flip = 0; flip <= 1; flip++) {
		int32_t x0_0 = subblock_bounds[flip][0][0], y0_0 = subblock_bounds[flip][0][1];
		int32_t w0   = subblock_bounds[flip][0][2], h0   = subblock_bounds[flip][0][3];
		int32_t x0_1 = subblock_bounds[flip][1][0], y0_1 = subblock_bounds[flip][1][1];
		int32_t w1   = subblock_bounds[flip][1][2], h1   = subblock_bounds[flip][1][3];

		int32_t avg0_r, avg0_g, avg0_b, avg1_r, avg1_g, avg1_b;
		_etc_subblock_average(rgba, stride, x0_0, y0_0, w0, h0, &avg0_r, &avg0_g, &avg0_b);
		_etc_subblock_average(rgba, stride, x0_1, y0_1, w1, h1, &avg1_r, &avg1_g, &avg1_b);

		int32_t r0_5 = _etc_quantize5(avg0_r), g0_5 = _etc_quantize5(avg0_g), b0_5 = _etc_quantize5(avg0_b);
		int32_t r1_5_target = _etc_quantize5(avg1_r), g1_5_target = _etc_quantize5(avg1_g), b1_5_target = _etc_quantize5(avg1_b);

		int32_t dr = r1_5_target - r0_5; if (dr < -4) dr = -4; else if (dr > 3) dr = 3;
		int32_t dg = g1_5_target - g0_5; if (dg < -4) dg = -4; else if (dg > 3) dg = 3;
		int32_t db = b1_5_target - b0_5; if (db < -4) db = -4; else if (db > 3) db = 3;

		int32_t r1_5 = r0_5 + dr, g1_5 = g0_5 + dg, b1_5 = b0_5 + db;
		if (r1_5 < 0 || r1_5 > 31 || g1_5 < 0 || g1_5 > 31 || b1_5 < 0 || b1_5 > 31) continue;

		int32_t base0_r = _etc_expand5(r0_5), base0_g = _etc_expand5(g0_5), base0_b = _etc_expand5(b0_5);
		int32_t base1_r = _etc_expand5(r1_5), base1_g = _etc_expand5(g1_5), base1_b = _etc_expand5(b1_5);

		int32_t table0, table1;
		uint8_t indices0[8], indices1[8];
		int32_t err0 = _etc_encode_subblock(rgba, stride, x0_0, y0_0, w0, h0, base0_r, base0_g, base0_b, &table0, indices0);
		int32_t err1 = _etc_encode_subblock(rgba, stride, x0_1, y0_1, w1, h1, base1_r, base1_g, base1_b, &table1, indices1);

		int32_t total_error = err0 + err1;
		if (total_error < best_error) {
			best_error = total_error;
			out_block[0] = (r0_5 << 3) | (dr & 0x7);
			out_block[1] = (g0_5 << 3) | (dg & 0x7);
			out_block[2] = (b0_5 << 3) | (db & 0x7);
			out_block[3] = (table0 << 5) | (table1 << 2) | (1 << 1) | flip;
			uint16_t msb, lsb;
			_etc_pack_subblock_indices(indices0, indices1, flip, &msb, &lsb);
			out_block[4] = (msb >> 8) & 0xFF;
			out_block[5] = (msb >> 0) & 0xFF;
			out_block[6] = (lsb >> 8) & 0xFF;
			out_block[7] = (lsb >> 0) & 0xFF;
		}
	}
	return best_error;
}

// Encode a 4x4 block to ETC1/ETC2 format
static void _encode_etc2_block(const uint8_t* rgba, int32_t stride, uint8_t* out) {
#if ETC2_QUALITY_MODE == 2
	// Best quality: try all modes exhaustively
	int32_t best_error = INT32_MAX;
	uint8_t best_block[8];

	// Sub-block regions for flip=0 (2x4 side by side) and flip=1 (4x2 stacked)
	static const int32_t subblock_bounds[2][2][4] = {
		{{0, 0, 2, 4}, {2, 0, 2, 4}},
		{{0, 0, 4, 2}, {0, 2, 4, 2}},
	};

	for (int32_t flip = 0; flip <= 1; flip++) {
		int32_t x0_0 = subblock_bounds[flip][0][0], y0_0 = subblock_bounds[flip][0][1];
		int32_t w0   = subblock_bounds[flip][0][2], h0   = subblock_bounds[flip][0][3];
		int32_t x0_1 = subblock_bounds[flip][1][0], y0_1 = subblock_bounds[flip][1][1];
		int32_t w1   = subblock_bounds[flip][1][2], h1   = subblock_bounds[flip][1][3];

		int32_t avg0_r, avg0_g, avg0_b, avg1_r, avg1_g, avg1_b;
		_etc_subblock_average(rgba, stride, x0_0, y0_0, w0, h0, &avg0_r, &avg0_g, &avg0_b);
		_etc_subblock_average(rgba, stride, x0_1, y0_1, w1, h1, &avg1_r, &avg1_g, &avg1_b);

		// Try individual mode (RGB444 + RGB444)
		{
			int32_t r0_4 = _etc_quantize4(avg0_r), g0_4 = _etc_quantize4(avg0_g), b0_4 = _etc_quantize4(avg0_b);
			int32_t r1_4 = _etc_quantize4(avg1_r), g1_4 = _etc_quantize4(avg1_g), b1_4 = _etc_quantize4(avg1_b);
			int32_t base0_r = _etc_expand4(r0_4), base0_g = _etc_expand4(g0_4), base0_b = _etc_expand4(b0_4);
			int32_t base1_r = _etc_expand4(r1_4), base1_g = _etc_expand4(g1_4), base1_b = _etc_expand4(b1_4);

			int32_t table0, table1;
			uint8_t indices0[8], indices1[8];
			int32_t err0 = _etc_encode_subblock(rgba, stride, x0_0, y0_0, w0, h0, base0_r, base0_g, base0_b, &table0, indices0);
			int32_t err1 = _etc_encode_subblock(rgba, stride, x0_1, y0_1, w1, h1, base1_r, base1_g, base1_b, &table1, indices1);

			int32_t total_error = err0 + err1;
			if (total_error < best_error) {
				best_error = total_error;
				best_block[0] = (r0_4 << 4) | r1_4;
				best_block[1] = (g0_4 << 4) | g1_4;
				best_block[2] = (b0_4 << 4) | b1_4;
				best_block[3] = (table0 << 5) | (table1 << 2) | (0 << 1) | flip;
				uint16_t msb, lsb;
				_etc_pack_subblock_indices(indices0, indices1, flip, &msb, &lsb);
				best_block[4] = (msb >> 8) & 0xFF;
				best_block[5] = (msb >> 0) & 0xFF;
				best_block[6] = (lsb >> 8) & 0xFF;
				best_block[7] = (lsb >> 0) & 0xFF;
			}
		}

		// Try differential mode (RGB555 + delta RGB333)
		{
			int32_t r0_5 = _etc_quantize5(avg0_r), g0_5 = _etc_quantize5(avg0_g), b0_5 = _etc_quantize5(avg0_b);
			int32_t r1_5_target = _etc_quantize5(avg1_r), g1_5_target = _etc_quantize5(avg1_g), b1_5_target = _etc_quantize5(avg1_b);
			int32_t dr = r1_5_target - r0_5; if (dr < -4) dr = -4; else if (dr > 3) dr = 3;
			int32_t dg = g1_5_target - g0_5; if (dg < -4) dg = -4; else if (dg > 3) dg = 3;
			int32_t db = b1_5_target - b0_5; if (db < -4) db = -4; else if (db > 3) db = 3;
			int32_t r1_5 = r0_5 + dr, g1_5 = g0_5 + dg, b1_5 = b0_5 + db;

			if (!(r1_5 < 0 || r1_5 > 31 || g1_5 < 0 || g1_5 > 31 || b1_5 < 0 || b1_5 > 31)) {
				int32_t base0_r = _etc_expand5(r0_5), base0_g = _etc_expand5(g0_5), base0_b = _etc_expand5(b0_5);
				int32_t base1_r = _etc_expand5(r1_5), base1_g = _etc_expand5(g1_5), base1_b = _etc_expand5(b1_5);

				int32_t table0, table1;
				uint8_t indices0[8], indices1[8];
				int32_t err0 = _etc_encode_subblock(rgba, stride, x0_0, y0_0, w0, h0, base0_r, base0_g, base0_b, &table0, indices0);
				int32_t err1 = _etc_encode_subblock(rgba, stride, x0_1, y0_1, w1, h1, base1_r, base1_g, base1_b, &table1, indices1);

				int32_t total_error = err0 + err1;
				if (total_error < best_error) {
					best_error = total_error;
					best_block[0] = (r0_5 << 3) | (dr & 0x7);
					best_block[1] = (g0_5 << 3) | (dg & 0x7);
					best_block[2] = (b0_5 << 3) | (db & 0x7);
					best_block[3] = (table0 << 5) | (table1 << 2) | (1 << 1) | flip;
					uint16_t msb, lsb;
					_etc_pack_subblock_indices(indices0, indices1, flip, &msb, &lsb);
					best_block[4] = (msb >> 8) & 0xFF;
					best_block[5] = (msb >> 0) & 0xFF;
					best_block[6] = (lsb >> 8) & 0xFF;
					best_block[7] = (lsb >> 0) & 0xFF;
				}
			}
		}
	}

	// Try ETC2 extended modes
	uint8_t mode_block[8];
	int32_t mode_error;

	mode_error = _etc_try_planar(rgba, stride, mode_block);
	if (mode_error < best_error) { best_error = mode_error; memcpy(best_block, mode_block, 8); }

	mode_error = _etc_try_t_mode(rgba, stride, mode_block);
	if (mode_error < best_error) { best_error = mode_error; memcpy(best_block, mode_block, 8); }

	mode_error = _etc_try_h_mode(rgba, stride, mode_block);
	if (mode_error < best_error) { best_error = mode_error; memcpy(best_block, mode_block, 8); }

	memcpy(out, best_block, 8);

#elif ETC2_QUALITY_MODE == 3
	// ETC1-only mode + planar: quick flip + mode selection with conservative planar
	{
		// Quick edge detection: find max adjacent pixel difference (detects hard lines)
		int32_t hz_diff = 0, vt_diff = 0;
		int32_t max_adj_diff = 0;

#if ETC_USE_SSE
		// Load all 4 rows (each row is 4 pixels = 16 bytes)
		__m128i row0 = _mm_loadu_si128((const __m128i*)(rgba + 0 * stride));
		__m128i row1 = _mm_loadu_si128((const __m128i*)(rgba + 1 * stride));
		__m128i row2 = _mm_loadu_si128((const __m128i*)(rgba + 2 * stride));
		__m128i row3 = _mm_loadu_si128((const __m128i*)(rgba + 3 * stride));

		// Horizontal differences: shift each row right by 4 bytes (1 pixel) and SAD
		__m128i row0_r = _mm_srli_si128(row0, 4);
		__m128i row1_r = _mm_srli_si128(row1, 4);
		__m128i row2_r = _mm_srli_si128(row2, 4);
		__m128i row3_r = _mm_srli_si128(row3, 4);

		// Compute SAD for horizontal neighbors (only first 3 pixels matter, but SAD includes 4th with 0)
		__m128i hsad0 = _mm_sad_epu8(row0, row0_r);
		__m128i hsad1 = _mm_sad_epu8(row1, row1_r);
		__m128i hsad2 = _mm_sad_epu8(row2, row2_r);
		__m128i hsad3 = _mm_sad_epu8(row3, row3_r);

		// For max_adj_diff, we need per-pixel SAD, not row SAD
		// Use absolute difference and find max
		__m128i hdiff0 = _mm_or_si128(_mm_subs_epu8(row0, row0_r), _mm_subs_epu8(row0_r, row0));
		__m128i hdiff1 = _mm_or_si128(_mm_subs_epu8(row1, row1_r), _mm_subs_epu8(row1_r, row1));
		__m128i hdiff2 = _mm_or_si128(_mm_subs_epu8(row2, row2_r), _mm_subs_epu8(row2_r, row2));
		__m128i hdiff3 = _mm_or_si128(_mm_subs_epu8(row3, row3_r), _mm_subs_epu8(row3_r, row3));

		// Vertical differences between adjacent rows
		__m128i vdiff01 = _mm_or_si128(_mm_subs_epu8(row0, row1), _mm_subs_epu8(row1, row0));
		__m128i vdiff12 = _mm_or_si128(_mm_subs_epu8(row1, row2), _mm_subs_epu8(row2, row1));
		__m128i vdiff23 = _mm_or_si128(_mm_subs_epu8(row2, row3), _mm_subs_epu8(row3, row2));

		// Find max across all differences for max_adj_diff
		__m128i max_h = _mm_max_epu8(_mm_max_epu8(hdiff0, hdiff1), _mm_max_epu8(hdiff2, hdiff3));
		__m128i max_v = _mm_max_epu8(_mm_max_epu8(vdiff01, vdiff12), vdiff23);
		__m128i max_all = _mm_max_epu8(max_h, max_v);

		// Horizontal max within the register - need to sum RGB per pixel then find max
		// For simplicity, extract and compute in scalar (still faster than full scalar loop)
		uint8_t max_bytes[16];
		_mm_storeu_si128((__m128i*)max_bytes, max_all);
		for (int i = 0; i < 12; i += 4) {  // Only first 3 pixels have valid horizontal diffs
			int32_t d = max_bytes[i] + max_bytes[i+1] + max_bytes[i+2];
			if (d > max_adj_diff) max_adj_diff = d;
		}

		// Flip selection: SAD between row1/row2 (horizontal edge) and columns 1/2 (vertical edge)
		__m128i hz_sad = _mm_sad_epu8(row1, row2);
		hz_diff = _mm_cvtsi128_si32(hz_sad) + _mm_extract_epi16(hz_sad, 4);

		// Vertical edge: need pixels at x=1 and x=2 from each row
		// Extract middle two pixels from each row and compute SAD
		__m128i col1 = _mm_set_epi32(*(int32_t*)(rgba + 3*stride + 4), *(int32_t*)(rgba + 2*stride + 4),
		                            *(int32_t*)(rgba + 1*stride + 4), *(int32_t*)(rgba + 0*stride + 4));
		__m128i col2 = _mm_set_epi32(*(int32_t*)(rgba + 3*stride + 8), *(int32_t*)(rgba + 2*stride + 8),
		                            *(int32_t*)(rgba + 1*stride + 8), *(int32_t*)(rgba + 0*stride + 8));
		__m128i vt_sad = _mm_sad_epu8(col1, col2);
		vt_diff = _mm_cvtsi128_si32(vt_sad) + _mm_extract_epi16(vt_sad, 4);
#elif ETC_USE_NEON
		// Load all 4 rows (each row is 4 pixels = 16 bytes)
		uint8x16_t row0 = vld1q_u8(rgba + 0 * stride);
		uint8x16_t row1 = vld1q_u8(rgba + 1 * stride);
		uint8x16_t row2 = vld1q_u8(rgba + 2 * stride);
		uint8x16_t row3 = vld1q_u8(rgba + 3 * stride);

		// Horizontal differences: shift each row right by 4 bytes (1 pixel)
		uint8x16_t zeros = vdupq_n_u8(0);
		uint8x16_t row0_r = vextq_u8(row0, zeros, 4);
		uint8x16_t row1_r = vextq_u8(row1, zeros, 4);
		uint8x16_t row2_r = vextq_u8(row2, zeros, 4);
		uint8x16_t row3_r = vextq_u8(row3, zeros, 4);

		// Absolute differences for horizontal neighbors
		uint8x16_t hdiff0 = vabdq_u8(row0, row0_r);
		uint8x16_t hdiff1 = vabdq_u8(row1, row1_r);
		uint8x16_t hdiff2 = vabdq_u8(row2, row2_r);
		uint8x16_t hdiff3 = vabdq_u8(row3, row3_r);

		// Vertical differences between adjacent rows
		uint8x16_t vdiff01 = vabdq_u8(row0, row1);
		uint8x16_t vdiff12 = vabdq_u8(row1, row2);
		uint8x16_t vdiff23 = vabdq_u8(row2, row3);

		// Find max across all differences for max_adj_diff
		uint8x16_t max_h = vmaxq_u8(vmaxq_u8(hdiff0, hdiff1), vmaxq_u8(hdiff2, hdiff3));
		uint8x16_t max_v = vmaxq_u8(vmaxq_u8(vdiff01, vdiff12), vdiff23);
		uint8x16_t max_all = vmaxq_u8(max_h, max_v);

		// Extract and compute max_adj_diff (sum RGB per pixel, find max)
		uint8_t max_bytes[16];
		vst1q_u8(max_bytes, max_all);
		for (int i = 0; i < 12; i += 4) {
			int32_t d = max_bytes[i] + max_bytes[i+1] + max_bytes[i+2];
			if (d > max_adj_diff) max_adj_diff = d;
		}

		// Flip selection: SAD between row1/row2 (horizontal edge)
		uint8x16_t hz_abd = vabdq_u8(row1, row2);
		uint16x8_t hz_sum16 = vpaddlq_u8(hz_abd);
		uint32x4_t hz_sum32 = vpaddlq_u16(hz_sum16);
		uint64x2_t hz_sum64 = vpaddlq_u32(hz_sum32);
		hz_diff = (int32_t)(vgetq_lane_u64(hz_sum64, 0) + vgetq_lane_u64(hz_sum64, 1));

		// Vertical edge: need pixels at x=1 and x=2 from each row
		uint32_t c1[4] = {*(uint32_t*)(rgba + 0*stride + 4), *(uint32_t*)(rgba + 1*stride + 4),
		                  *(uint32_t*)(rgba + 2*stride + 4), *(uint32_t*)(rgba + 3*stride + 4)};
		uint32_t c2[4] = {*(uint32_t*)(rgba + 0*stride + 8), *(uint32_t*)(rgba + 1*stride + 8),
		                  *(uint32_t*)(rgba + 2*stride + 8), *(uint32_t*)(rgba + 3*stride + 8)};
		uint8x16_t col1 = vreinterpretq_u8_u32(vld1q_u32(c1));
		uint8x16_t col2 = vreinterpretq_u8_u32(vld1q_u32(c2));
		uint8x16_t vt_abd = vabdq_u8(col1, col2);
		uint16x8_t vt_sum16 = vpaddlq_u8(vt_abd);
		uint32x4_t vt_sum32 = vpaddlq_u16(vt_sum16);
		uint64x2_t vt_sum64 = vpaddlq_u32(vt_sum32);
		vt_diff = (int32_t)(vgetq_lane_u64(vt_sum64, 0) + vgetq_lane_u64(vt_sum64, 1));
#else
		for (int32_t y = 0; y < 4; y++) {
			for (int32_t x = 0; x < 4; x++) {
				const uint8_t* p = rgba + y * stride + x * 4;
				// Check right neighbor
				if (x < 3) {
					const uint8_t* r = p + 4;
					int32_t d = abs(p[0] - r[0]) + abs(p[1] - r[1]) + abs(p[2] - r[2]);
					if (d > max_adj_diff) max_adj_diff = d;
				}
				// Check bottom neighbor
				if (y < 3) {
					const uint8_t* b = p + stride;
					int32_t d = abs(p[0] - b[0]) + abs(p[1] - b[1]) + abs(p[2] - b[2]);
					if (d > max_adj_diff) max_adj_diff = d;
				}
			}
		}
		// Flip selection from middle edges
		for (int32_t x = 0; x < 4; x++) {
			const uint8_t* top = rgba + 1 * stride + x * 4;
			const uint8_t* bot = rgba + 2 * stride + x * 4;
			hz_diff += abs(top[0] - bot[0]) + abs(top[1] - bot[1]) + abs(top[2] - bot[2]);
		}
		for (int32_t y = 0; y < 4; y++) {
			const uint8_t* lft = rgba + y * stride + 1 * 4;
			const uint8_t* rgt = rgba + y * stride + 2 * 4;
			vt_diff += abs(lft[0] - rgt[0]) + abs(lft[1] - rgt[1]) + abs(lft[2] - rgt[2]);
		}
#endif

		// Conservative planar: only use for smooth gradients (no hard edges)
		// Threshold ~30 means max ~10 per channel difference between adjacent pixels
		if (max_adj_diff < 30 && (hz_diff > 20 || vt_diff > 20)) {
			_etc_try_planar(rgba, stride, out);
			goto etc1_done;
		}

		int32_t flip = (hz_diff > vt_diff) ? 1 : 0;

		// Subblock bounds: [flip][subblock][x,y,w,h]
		static const int32_t bounds[2][2][4] = {
			{{0, 0, 2, 4}, {2, 0, 2, 4}},  // flip=0
			{{0, 0, 4, 2}, {0, 2, 4, 2}},  // flip=1
		};

		int32_t avg0_r, avg0_g, avg0_b, avg1_r, avg1_g, avg1_b;
		_etc_subblock_average(rgba, stride, bounds[flip][0][0], bounds[flip][0][1], bounds[flip][0][2], bounds[flip][0][3], &avg0_r, &avg0_g, &avg0_b);
		_etc_subblock_average(rgba, stride, bounds[flip][1][0], bounds[flip][1][1], bounds[flip][1][2], bounds[flip][1][3], &avg1_r, &avg1_g, &avg1_b);

		// Quick mode selection: check if differential delta would be clamped
		int32_t r0_5 = _etc_quantize5(avg0_r), g0_5 = _etc_quantize5(avg0_g), b0_5 = _etc_quantize5(avg0_b);
		int32_t r1_5 = _etc_quantize5(avg1_r), g1_5 = _etc_quantize5(avg1_g), b1_5 = _etc_quantize5(avg1_b);
		int32_t dr_raw = r1_5 - r0_5, dg_raw = g1_5 - g0_5, db_raw = b1_5 - b0_5;

		// If any delta is outside -4..+3, use individual mode (RGB444 per subblock)
		int32_t use_individual = (dr_raw < -4 || dr_raw > 3 || dg_raw < -4 || dg_raw > 3 || db_raw < -4 || db_raw > 3);

		int32_t table0, table1;
		uint8_t indices0[8], indices1[8];

		if (use_individual) {
			// Individual mode: RGB444 + RGB444
			int32_t r0_4 = _etc_quantize4(avg0_r), g0_4 = _etc_quantize4(avg0_g), b0_4 = _etc_quantize4(avg0_b);
			int32_t r1_4 = _etc_quantize4(avg1_r), g1_4 = _etc_quantize4(avg1_g), b1_4 = _etc_quantize4(avg1_b);
			int32_t base0_r = _etc_expand4(r0_4), base0_g = _etc_expand4(g0_4), base0_b = _etc_expand4(b0_4);
			int32_t base1_r = _etc_expand4(r1_4), base1_g = _etc_expand4(g1_4), base1_b = _etc_expand4(b1_4);

			_etc_encode_subblock(rgba, stride, bounds[flip][0][0], bounds[flip][0][1], bounds[flip][0][2], bounds[flip][0][3], base0_r, base0_g, base0_b, &table0, indices0);
			_etc_encode_subblock(rgba, stride, bounds[flip][1][0], bounds[flip][1][1], bounds[flip][1][2], bounds[flip][1][3], base1_r, base1_g, base1_b, &table1, indices1);

			out[0] = (r0_4 << 4) | r1_4;
			out[1] = (g0_4 << 4) | g1_4;
			out[2] = (b0_4 << 4) | b1_4;
			out[3] = (table0 << 5) | (table1 << 2) | (0 << 1) | flip; // diff=0
		} else {
			// Differential mode: RGB555 + RGB333 delta
			int32_t dr = dr_raw, dg = dg_raw, db = db_raw;
			int32_t r1_5_actual = r0_5 + dr, g1_5_actual = g0_5 + dg, b1_5_actual = b0_5 + db;
			int32_t base0_r = _etc_expand5(r0_5), base0_g = _etc_expand5(g0_5), base0_b = _etc_expand5(b0_5);
			int32_t base1_r = _etc_expand5(r1_5_actual), base1_g = _etc_expand5(g1_5_actual), base1_b = _etc_expand5(b1_5_actual);

			_etc_encode_subblock(rgba, stride, bounds[flip][0][0], bounds[flip][0][1], bounds[flip][0][2], bounds[flip][0][3], base0_r, base0_g, base0_b, &table0, indices0);
			_etc_encode_subblock(rgba, stride, bounds[flip][1][0], bounds[flip][1][1], bounds[flip][1][2], bounds[flip][1][3], base1_r, base1_g, base1_b, &table1, indices1);

			out[0] = (r0_5 << 3) | (dr & 0x7);
			out[1] = (g0_5 << 3) | (dg & 0x7);
			out[2] = (b0_5 << 3) | (db & 0x7);
			out[3] = (table0 << 5) | (table1 << 2) | (1 << 1) | flip; // diff=1
		}

		uint16_t msb, lsb;
		_etc_pack_subblock_indices(indices0, indices1, flip, &msb, &lsb);
		out[4] = (msb >> 8) & 0xFF;
		out[5] = (msb >> 0) & 0xFF;
		out[6] = (lsb >> 8) & 0xFF;
		out[7] = (lsb >> 0) & 0xFF;
	}
	etc1_done:;

#else
	// Fast mode: quick classification + quick flip/mode selection
	int32_t category = _etc_classify_block(rgba, stride);
	int32_t best_error = INT32_MAX;
	uint8_t best_block[8];
	uint8_t mode_block[8];
	int32_t mode_error;

	switch (category) {
		case ETC2_CAT_SOLID:
		case ETC2_CAT_COMPLEX:
		{
			// Quick flip selection for differential/individual modes
			int32_t hz_diff = 0, vt_diff = 0;
			for (int32_t x = 0; x < 4; x++) {
				const uint8_t* top = rgba + 1 * stride + x * 4;
				const uint8_t* bot = rgba + 2 * stride + x * 4;
				hz_diff += abs(top[0] - bot[0]) + abs(top[1] - bot[1]) + abs(top[2] - bot[2]);
			}
			for (int32_t y = 0; y < 4; y++) {
				const uint8_t* lft = rgba + y * stride + 1 * 4;
				const uint8_t* rgt = rgba + y * stride + 2 * 4;
				vt_diff += abs(lft[0] - rgt[0]) + abs(lft[1] - rgt[1]) + abs(lft[2] - rgt[2]);
			}
			int32_t flip = (hz_diff > vt_diff) ? 1 : 0;

			static const int32_t bounds[2][2][4] = {
				{{0, 0, 2, 4}, {2, 0, 2, 4}},
				{{0, 0, 4, 2}, {0, 2, 4, 2}},
			};

			int32_t avg0_r, avg0_g, avg0_b, avg1_r, avg1_g, avg1_b;
			_etc_subblock_average(rgba, stride, bounds[flip][0][0], bounds[flip][0][1], bounds[flip][0][2], bounds[flip][0][3], &avg0_r, &avg0_g, &avg0_b);
			_etc_subblock_average(rgba, stride, bounds[flip][1][0], bounds[flip][1][1], bounds[flip][1][2], bounds[flip][1][3], &avg1_r, &avg1_g, &avg1_b);

			// Quick mode selection: check if differential delta would be clamped
			int32_t r0_5 = _etc_quantize5(avg0_r), g0_5 = _etc_quantize5(avg0_g), b0_5 = _etc_quantize5(avg0_b);
			int32_t r1_5 = _etc_quantize5(avg1_r), g1_5 = _etc_quantize5(avg1_g), b1_5 = _etc_quantize5(avg1_b);
			int32_t dr_raw = r1_5 - r0_5, dg_raw = g1_5 - g0_5, db_raw = b1_5 - b0_5;
			int32_t use_individual = (dr_raw < -4 || dr_raw > 3 || dg_raw < -4 || dg_raw > 3 || db_raw < -4 || db_raw > 3);

			int32_t table0, table1;
			uint8_t indices0[8], indices1[8];
			int32_t err0, err1;

			if (use_individual) {
				int32_t r0_4 = _etc_quantize4(avg0_r), g0_4 = _etc_quantize4(avg0_g), b0_4 = _etc_quantize4(avg0_b);
				int32_t r1_4 = _etc_quantize4(avg1_r), g1_4 = _etc_quantize4(avg1_g), b1_4 = _etc_quantize4(avg1_b);
				int32_t base0_r = _etc_expand4(r0_4), base0_g = _etc_expand4(g0_4), base0_b = _etc_expand4(b0_4);
				int32_t base1_r = _etc_expand4(r1_4), base1_g = _etc_expand4(g1_4), base1_b = _etc_expand4(b1_4);

				err0 = _etc_encode_subblock(rgba, stride, bounds[flip][0][0], bounds[flip][0][1], bounds[flip][0][2], bounds[flip][0][3], base0_r, base0_g, base0_b, &table0, indices0);
				err1 = _etc_encode_subblock(rgba, stride, bounds[flip][1][0], bounds[flip][1][1], bounds[flip][1][2], bounds[flip][1][3], base1_r, base1_g, base1_b, &table1, indices1);

				best_block[0] = (r0_4 << 4) | r1_4;
				best_block[1] = (g0_4 << 4) | g1_4;
				best_block[2] = (b0_4 << 4) | b1_4;
				best_block[3] = (table0 << 5) | (table1 << 2) | (0 << 1) | flip;
			} else {
				int32_t dr = dr_raw, dg = dg_raw, db = db_raw;
				int32_t r1_5_actual = r0_5 + dr, g1_5_actual = g0_5 + dg, b1_5_actual = b0_5 + db;
				int32_t base0_r = _etc_expand5(r0_5), base0_g = _etc_expand5(g0_5), base0_b = _etc_expand5(b0_5);
				int32_t base1_r = _etc_expand5(r1_5_actual), base1_g = _etc_expand5(g1_5_actual), base1_b = _etc_expand5(b1_5_actual);

				err0 = _etc_encode_subblock(rgba, stride, bounds[flip][0][0], bounds[flip][0][1], bounds[flip][0][2], bounds[flip][0][3], base0_r, base0_g, base0_b, &table0, indices0);
				err1 = _etc_encode_subblock(rgba, stride, bounds[flip][1][0], bounds[flip][1][1], bounds[flip][1][2], bounds[flip][1][3], base1_r, base1_g, base1_b, &table1, indices1);

				best_block[0] = (r0_5 << 3) | (dr & 0x7);
				best_block[1] = (g0_5 << 3) | (dg & 0x7);
				best_block[2] = (b0_5 << 3) | (db & 0x7);
				best_block[3] = (table0 << 5) | (table1 << 2) | (1 << 1) | flip;
			}
			best_error = err0 + err1;

			uint16_t msb, lsb;
			_etc_pack_subblock_indices(indices0, indices1, flip, &msb, &lsb);
			best_block[4] = (msb >> 8) & 0xFF;
			best_block[5] = (msb >> 0) & 0xFF;
			best_block[6] = (lsb >> 8) & 0xFF;
			best_block[7] = (lsb >> 0) & 0xFF;
			break;
		}

		case ETC2_CAT_GRADIENT:
			// Planar mode is best for gradients (no flip needed)
			best_error = _etc_try_planar(rgba, stride, best_block);
			break;

		case ETC2_CAT_TWOTONE:
			// T or H mode for two-tone blocks
			best_error = _etc_try_t_mode(rgba, stride, best_block);
			mode_error = _etc_try_h_mode(rgba, stride, mode_block);
			if (mode_error < best_error) {
				best_error = mode_error;
				memcpy(best_block, mode_block, 8);
			}
			break;
	}

#if ETC2_QUALITY_MODE == 1
	// Fallback: if error is too high, try other modes
	if (best_error > ETC2_FALLBACK_THRESHOLD) {
		if (category != ETC2_CAT_COMPLEX) {
			mode_error = _etc_try_individual(rgba, stride, mode_block);
			if (mode_error < best_error) { best_error = mode_error; memcpy(best_block, mode_block, 8); }
		}
		if (category != ETC2_CAT_SOLID) {
			mode_error = _etc_try_differential(rgba, stride, mode_block);
			if (mode_error < best_error) { best_error = mode_error; memcpy(best_block, mode_block, 8); }
		}
		if (category != ETC2_CAT_GRADIENT) {
			mode_error = _etc_try_planar(rgba, stride, mode_block);
			if (mode_error < best_error) { best_error = mode_error; memcpy(best_block, mode_block, 8); }
		}
		if (category != ETC2_CAT_TWOTONE) {
			mode_error = _etc_try_t_mode(rgba, stride, mode_block);
			if (mode_error < best_error) { best_error = mode_error; memcpy(best_block, mode_block, 8); }
			mode_error = _etc_try_h_mode(rgba, stride, mode_block);
			if (mode_error < best_error) { best_error = mode_error; memcpy(best_block, mode_block, 8); }
		}
	}
#endif

	memcpy(out, best_block, 8);
#endif
}

uint8_t* etc2_rgb8_compress(const uint8_t* rgba, int32_t width, int32_t height) {
	int32_t blocks_x  = (width  + 3) / 4;
	int32_t blocks_y  = (height + 3) / 4;
	int32_t etc_size  = blocks_x * blocks_y * 8;
	uint8_t* etc_data = malloc(etc_size);
	if (!etc_data) return NULL;

	int32_t stride = width * 4;

	// Temporary buffer for edge blocks
	uint8_t block_rgba[4 * 4 * 4];

	for (int32_t by = 0; by < blocks_y; by++) {
		for (int32_t bx = 0; bx < blocks_x; bx++) {
			int32_t px = bx * 4;
			int32_t py = by * 4;

			const uint8_t* block_ptr;
			int32_t        block_stride;

			// Handle edge blocks by copying with clamping
			if (px + 4 > width || py + 4 > height) {
				for (int32_t y = 0; y < 4; y++) {
					for (int32_t x = 0; x < 4; x++) {
						int32_t sx = px + x < width  ? px + x : width  - 1;
						int32_t sy = py + y < height ? py + y : height - 1;
						const uint8_t* src = rgba + sy * stride + sx * 4;
						uint8_t*       dst = block_rgba + y * 16 + x * 4;
						dst[0] = src[0];
						dst[1] = src[1];
						dst[2] = src[2];
						dst[3] = src[3];
					}
				}
				block_ptr    = block_rgba;
				block_stride = 16;
			} else {
				block_ptr    = rgba + py * stride + px * 4;
				block_stride = stride;
			}

			// Premultiply alpha so semi-transparent pixels fade to black
			uint8_t premul_rgba[4 * 4 * 4];
			for (int32_t y = 0; y < 4; y++) {
				for (int32_t x = 0; x < 4; x++) {
					const uint8_t* src = block_ptr + y * block_stride + x * 4;
					uint8_t*       dst = premul_rgba + y * 16 + x * 4;
					uint8_t a = src[3];
					dst[0] = (uint8_t)((src[0] * a + 127) / 255);
					dst[1] = (uint8_t)((src[1] * a + 127) / 255);
					dst[2] = (uint8_t)((src[2] * a + 127) / 255);
					dst[3] = a;
				}
			}

			uint8_t* out = etc_data + (by * blocks_x + bx) * 8;
			_encode_etc2_block(premul_rgba, 16, out);
		}
	}

	return etc_data;
}
