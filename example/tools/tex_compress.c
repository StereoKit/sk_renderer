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

