// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// L1h - UASTC LDR 4x4 -> BC7. Near-lossless, the one place this library accepts
// loss on a block-to-block path; ktx2_uastc_astc.c is preferred when the
// hardware offers ASTC, and this exists for BC-only desktop GPUs.
//
// UASTC is the ASTC/BC7 intersection, so partition patterns and the 2- and 3-bit
// weight tables carry across untouched. What does not:
//
//  - Endpoints. UASTC interpolates at 16 bits, BC7 at 8 and spends some of its
//    precision on p-bits. Requantizing is the whole source of loss, and KDF
//    25.8.2 pins exactly how, because the UASTC *encoder* assumed this
//    arithmetic when it estimated error. Deviating is worse, not just different.
//  - Weights, where the index widths disagree. The tables below are the spec's.
//  - Anchors. Both formats drop one index's high bit per subset but disagree on
//    which index, so a subset can need another endpoint swap and index
//    inversion, unrelated to the one ASTC needs.

#include "ktx2_internal.h"

#include <string.h>

typedef struct ktx2_bc7_mode_t {
	uint8_t subsets;
	uint8_t partition_bits;
	uint8_t rotation_bits;
	uint8_t index_sel_bits;
	uint8_t colour_bits;
	uint8_t alpha_bits;      // 0 means the mode has no alpha, which decodes as 255
	uint8_t endpoint_pbits;  // one p-bit per endpoint
	uint8_t shared_pbits;    // one p-bit per subset, shared by its two endpoints
	uint8_t index_bits;
	uint8_t index2_bits;     // secondary index set, alpha only
} ktx2_bc7_mode_t;

// KDF 1.4 Table 122. Modes 0 and 4 are never reached from UASTC.
static const ktx2_bc7_mode_t k_bc7_modes[8] = {
	{ 3, 4, 0, 0, 4, 0, 1, 0, 3, 0 },
	{ 2, 6, 0, 0, 6, 0, 0, 1, 3, 0 },
	{ 3, 6, 0, 0, 5, 0, 0, 0, 2, 0 },
	{ 2, 6, 0, 0, 7, 0, 1, 0, 2, 0 },
	{ 1, 0, 2, 1, 5, 6, 0, 0, 2, 3 },
	{ 1, 0, 2, 0, 7, 8, 0, 0, 2, 2 },
	{ 1, 0, 0, 0, 7, 7, 1, 0, 4, 0 },
	{ 2, 6, 0, 0, 5, 5, 1, 0, 2, 0 },
};

// The reference encoder's choices, trading endpoint precision against keeping
// weight indices exact. UASTC 1 takes a 2-subset BC7 mode with duplicated
// endpoints rather than mode 6, which keeps its 2-bit weights unwidened.
// Mode 8 is void extent, handled separately.
static const uint8_t k_bc7_mode_of[19] = {
	6, 3, 1, 2, 3, 6, 5, 2, 5, 7, 6, 5, 6, 5, 6, 6, 7, 5, 6,
};

// Only the widths that actually differ are populated.
static const uint8_t k_weight_1_to_2[ 2] = { 0, 3 };
static const uint8_t k_weight_2_to_4[ 4] = { 0, 5, 10, 15 };
static const uint8_t k_weight_3_to_4[ 8] = { 0, 2, 4, 6, 9, 11, 13, 15 };
static const uint8_t k_weight_5_to_4[32] = {
	0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7,
	8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
};

static uint32_t _convert_weight(uint32_t weight, uint32_t from_bits, uint32_t to_bits) {
	if (from_bits == to_bits) return weight;
	if (from_bits == 1 && to_bits == 2) return k_weight_1_to_2[weight];
	if (from_bits == 2 && to_bits == 4) return k_weight_2_to_4[weight];
	if (from_bits == 3 && to_bits == 4) return k_weight_3_to_4[weight];
	return k_weight_5_to_4[weight]; // 5 -> 4, UASTC mode 18
}

///////////////////////////////////////////////////////////////////////////////

// BC7 fields are strictly in stream order, so a cursor is all this needs.
static void _put(uint8_t block[16], uint32_t* ref_at, uint32_t width, uint32_t value) {
	ktx2_bits_put(block, *ref_at, width, value);
	*ref_at += width;
}

static int32_t _clamp(int32_t value, int32_t low, int32_t high) {
	return value < low ? low : (value > high ? high : value);
}

static float _square(float v) { return v * v; }

// KDF 25.8.2 reproduced, not reinvented: the encoder assumed this exact rounding,
// so a "better" quantizer here produces a worse texture than it predicted.
// The subset's two endpoints share one p-bit, so they are scored together.
static void _pbits_shared(uint32_t comps, uint32_t comp_bits, const float low[4], const float high[4],
                          uint8_t out_low[4], uint8_t out_high[4], uint8_t out_pbits[2]) {
	uint32_t total_bits = comp_bits + 1;
	int32_t  iscalep    = (1 << total_bits) - 1;
	float    scalep     = (float)iscalep;
	float    best       = 1e9f;

	for (int32_t p = 0; p < 2; p++) {
		uint8_t min_colour[4], max_colour[4], scaled_low[4], scaled_high[4];
		for (uint32_t c = 0; c < 4; c++) {
			min_colour[c] = (uint8_t)_clamp(((int32_t)((low [c] * scalep - (float)p) / 2.0f + 0.5f)) * 2 + p, p, iscalep - 1 + p);
			max_colour[c] = (uint8_t)_clamp(((int32_t)((high[c] * scalep - (float)p) / 2.0f + 0.5f)) * 2 + p, p, iscalep - 1 + p);
		}
		for (uint32_t c = 0; c < 4; c++) {
			scaled_low [c] = (uint8_t)(min_colour[c] << (8 - total_bits)); scaled_low [c] |= (uint8_t)(scaled_low [c] >> total_bits);
			scaled_high[c] = (uint8_t)(max_colour[c] << (8 - total_bits)); scaled_high[c] |= (uint8_t)(scaled_high[c] >> total_bits);
		}
		float error = 0;
		for (uint32_t c = 0; c < comps; c++)
			error += _square((float)scaled_low[c] / 255.0f - low[c]) + _square((float)scaled_high[c] / 255.0f - high[c]);

		if (error >= best) continue;
		best         = error;
		out_pbits[0] = out_pbits[1] = (uint8_t)p;
		for (uint32_t c = 0; c < 4; c++) {
			out_low [c] = (uint8_t)(min_colour[c] >> 1);
			out_high[c] = (uint8_t)(max_colour[c] >> 1);
		}
	}
}

// The same, but each endpoint has its own p-bit and so is scored alone.
static void _pbits_unique(uint32_t comps, uint32_t comp_bits, const float low[4], const float high[4],
                          uint8_t out_low[4], uint8_t out_high[4], uint8_t out_pbits[2]) {
	uint32_t total_bits = comp_bits + 1;
	int32_t  iscalep    = (1 << total_bits) - 1;
	float    scalep     = (float)iscalep;
	float    best_low   = 1e9f;
	float    best_high  = 1e9f;

	for (int32_t p = 0; p < 2; p++) {
		uint8_t min_colour[4], max_colour[4], scaled_low[4], scaled_high[4];
		for (uint32_t c = 0; c < 4; c++) {
			min_colour[c] = (uint8_t)_clamp(((int32_t)((low [c] * scalep - (float)p) / 2.0f + 0.5f)) * 2 + p, p, iscalep - 1 + p);
			max_colour[c] = (uint8_t)_clamp(((int32_t)((high[c] * scalep - (float)p) / 2.0f + 0.5f)) * 2 + p, p, iscalep - 1 + p);
		}
		for (uint32_t c = 0; c < 4; c++) {
			scaled_low [c] = (uint8_t)(min_colour[c] << (8 - total_bits)); scaled_low [c] |= (uint8_t)(scaled_low [c] >> total_bits);
			scaled_high[c] = (uint8_t)(max_colour[c] << (8 - total_bits)); scaled_high[c] |= (uint8_t)(scaled_high[c] >> total_bits);
		}
		float error_low = 0, error_high = 0;
		for (uint32_t c = 0; c < comps; c++) {
			error_low  += _square((float)scaled_low [c] - low [c] * 255.0f);
			error_high += _square((float)scaled_high[c] - high[c] * 255.0f);
		}
		if (error_low < best_low) {
			best_low     = error_low;
			out_pbits[0] = (uint8_t)p;
			for (uint32_t c = 0; c < 4; c++) out_low[c] = (uint8_t)(min_colour[c] >> 1);
		}
		if (error_high < best_high) {
			best_high    = error_high;
			out_pbits[1] = (uint8_t)p;
			for (uint32_t c = 0; c < 4; c++) out_high[c] = (uint8_t)(max_colour[c] >> 1);
		}
	}
}

// No p-bit: a plain scale with rounding, again exactly as the encoder assumed.
static uint8_t _scale_down(uint32_t value, uint32_t bits) {
	uint32_t maximum = (1u << bits) - 1u;
	return (uint8_t)((value * maximum + 127u) / 255u);
}

///////////////////////////////////////////////////////////////////////////////

// Void extent and anything unreadable. BC7 mode 5 hits any 8-bit colour exactly
// at a fixed weight of 1: the endpoints straddle the target so the interpolation
// lands back on it. KDF 25.8.2.1.
static void _write_solid(const uint8_t solid[4], uint8_t out_block[16]) {
	uint32_t at = 0;
	memset(out_block, 0, 16);
	_put(out_block, &at, 6, 1u << 5); // mode 5
	_put(out_block, &at, 2, 0);       // no rotation

	for (uint32_t c = 0; c < 3; c++) {
		uint32_t value = solid[c];
		uint32_t e0    = value >> 1;
		uint32_t e1    = e0;
		if      (value < 128 && (value & 1) != 0) e1 = e0 + 1;
		else if (value > 127 && (value & 1) == 0) e1 = e0 - 1;
		_put(out_block, &at, 7, e0);
		_put(out_block, &at, 7, e1);
	}
	_put(out_block, &at, 8, solid[3]); // alpha is 8 bits in mode 5, so exact
	_put(out_block, &at, 8, solid[3]);

	// Colour indices all 1 (the anchor is one bit narrower), alpha indices all 0.
	_put(out_block, &at, 1, 1);
	for (uint32_t texel = 1; texel < 16; texel++) _put(out_block, &at, 2, 1);
	at += 31; // alpha indices are already zero
}

void ktx2_uastc_write_bc7(const ktx2_uastc_t* block, uint8_t out_block[16]) {
	if (block->mode == KTX2_UASTC_MODE_INVALID) {
		uint8_t magenta[4] = { KTX2_UASTC_INVALID_R, KTX2_UASTC_INVALID_G, KTX2_UASTC_INVALID_B, 255 };
		_write_solid(magenta, out_block);
		return;
	}
	if (block->mode == KTX2_UASTC_MODE_VOID) { _write_solid(block->solid, out_block); return; }

	memset(out_block, 0, 16);

	uint32_t                 bc7      = k_bc7_mode_of[block->mode];
	const ktx2_bc7_mode_t*   info     = &k_bc7_modes[bc7];
	uint32_t                 subsets  = info->subsets;
	uint32_t                 pattern  = subsets > 1
		? ktx2_uastc_bc7_pattern(block->mode, block->subsets, block->pattern_index) : 0;

	// A single-subset block in a multi-subset BC7 mode duplicates its endpoints.
	// The partition is still 0, but its texels decide which indices narrow.
	uint32_t uastc_of[3] = { 0, 0, 0 };
	if (block->mode == 7) {
		for (uint32_t s = 0; s < 3; s++) uastc_of[s] = ktx2_uastc_bc7_collapse(block->pattern_index, s);
	} else if (block->subsets > 1) {
		for (uint32_t u = 0; u < block->subsets; u++)
			uastc_of[ktx2_uastc_bc7_subset(block->mode, block->subsets, block->pattern_index, u)] = u;
	}

	uint8_t src_low [3][4];
	uint8_t src_high[3][4];
	ktx2_uastc_endpoints(block, src_low, src_high);

	// ASTC names the component with its own weights; BC7 always gives the second
	// weight set to alpha and rotates the chosen component into alpha's slot. LA
	// blocks already select alpha, so they need neither swap nor rotation.
	uint32_t rotation = 0;
	if (block->planes == 2 && block->compsel != KTX2_UASTC_LA_COMPSEL) {
		rotation = block->compsel + 1u;
		for (uint32_t s = 0; s < 3; s++) {
			uint8_t swap = src_low [s][block->compsel]; src_low [s][block->compsel] = src_low [s][3]; src_low [s][3] = swap;
			swap         = src_high[s][block->compsel]; src_high[s][block->compsel] = src_high[s][3]; src_high[s][3] = swap;
		}
	}

	// Copied and re-indexed by BC7 subset, not UASTC subset. Two BC7 subsets can
	// share one UASTC subset - mode 7 by design, single-subset blocks in 2-subset
	// modes always - and each must stay separately swappable for the anchor fix-up.
	uint8_t low [3][4];
	uint8_t high[3][4];
	for (uint32_t s = 0; s < subsets; s++) {
		memcpy(low [s], src_low [uastc_of[s]], 4);
		memcpy(high[s], src_high[uastc_of[s]], 4);
	}

	// Converted to BC7's width. The second set exists only in mode 5, carrying
	// alpha; single plane blocks give it the same weights so alpha tracks colour.
	uint32_t index [16];
	uint32_t index2[16];
	for (uint32_t texel = 0; texel < 16; texel++) {
		uint32_t first  = block->weights[texel * block->planes];
		uint32_t second = block->planes == 2 ? block->weights[texel * 2 + 1] : first;
		index [texel] = _convert_weight(first,  block->weight_bits, info->index_bits);
		index2[texel] = info->index2_bits != 0
			? _convert_weight(second, block->weight_bits, info->index2_bits) : 0;
	}

	// BC7 picks a different anchor texel than UASTC, so a subset's anchor index can
	// arrive with its high bit set. Swapping and inverting that subset is exact.
	uint32_t texel_subset[16];
	for (uint32_t texel = 0; texel < 16; texel++)
		texel_subset[texel] = ktx2_uastc_bc7_texel_subset(subsets, pattern, texel);

	for (uint32_t s = 0; s < subsets; s++) {
		uint32_t anchor = ktx2_uastc_bc7_anchor(subsets, pattern, s);
		if ((index[anchor] >> (info->index_bits - 1)) == 0) continue;
		uint32_t top = (1u << info->index_bits) - 1u;
		for (uint32_t texel = 0; texel < 16; texel++)
			if (texel_subset[texel] == s) index[texel] = top - index[texel];
		// Mode 5 interpolates colour and alpha from different index sets, so a
		// colour-driven swap must leave alpha's endpoints alone.
		uint32_t channels = info->index2_bits != 0 ? 3u : 4u;
		for (uint32_t c = 0; c < channels; c++) {
			uint8_t swap = low[s][c]; low[s][c] = high[s][c]; high[s][c] = swap;
		}
	}
	if (info->index2_bits != 0 && (index2[0] >> (info->index2_bits - 1)) != 0) {
		uint32_t top = (1u << info->index2_bits) - 1u;
		for (uint32_t texel = 0; texel < 16; texel++) index2[texel] = top - index2[texel];
		uint8_t swap = low[0][3]; low[0][3] = high[0][3]; high[0][3] = swap;
	}

	// Requantize the endpoints, which is where the loss lives.
	uint8_t out_low  [3][4];
	uint8_t out_high [3][4];
	uint8_t out_pbits[3][2];
	memset(out_low, 0, sizeof(out_low)); memset(out_high, 0, sizeof(out_high));
	memset(out_pbits, 0, sizeof(out_pbits));

	for (uint32_t s = 0; s < subsets; s++) {
		if (info->endpoint_pbits != 0 || info->shared_pbits != 0) {
			float scaled_low[4], scaled_high[4];
			for (uint32_t c = 0; c < 4; c++) {
				scaled_low [c] = (float)low [s][c] / 255.0f;
				scaled_high[c] = (float)high[s][c] / 255.0f;
			}
			uint32_t comps = info->alpha_bits != 0 ? 4u : 3u;
			if (info->shared_pbits != 0)
				_pbits_shared(comps, info->colour_bits, scaled_low, scaled_high, out_low[s], out_high[s], out_pbits[s]);
			else
				_pbits_unique(comps, info->colour_bits, scaled_low, scaled_high, out_low[s], out_high[s], out_pbits[s]);
		} else {
			for (uint32_t c = 0; c < 3; c++) {
				out_low [s][c] = _scale_down(low [s][c], info->colour_bits);
				out_high[s][c] = _scale_down(high[s][c], info->colour_bits);
			}
			if (info->alpha_bits != 0) {
				out_low [s][3] = _scale_down(low [s][3], info->alpha_bits);
				out_high[s][3] = _scale_down(high[s][3], info->alpha_bits);
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////

	uint32_t at = 0;
	_put(out_block, &at, bc7 + 1u, 1u << bc7); // mode: that many zeroes, then a one
	_put(out_block, &at, info->partition_bits, pattern);
	_put(out_block, &at, info->rotation_bits,  rotation);
	_put(out_block, &at, info->index_sel_bits, 0);

	// Channel major, then subset, then endpoint - KDF 20.1.
	for (uint32_t c = 0; c < 3; c++)
		for (uint32_t s = 0; s < subsets; s++) {
			_put(out_block, &at, info->colour_bits, out_low [s][c]);
			_put(out_block, &at, info->colour_bits, out_high[s][c]);
		}
	if (info->alpha_bits != 0)
		for (uint32_t s = 0; s < subsets; s++) {
			_put(out_block, &at, info->alpha_bits, out_low [s][3]);
			_put(out_block, &at, info->alpha_bits, out_high[s][3]);
		}

	if (info->endpoint_pbits != 0)
		for (uint32_t s = 0; s < subsets; s++) {
			_put(out_block, &at, 1, out_pbits[s][0]);
			_put(out_block, &at, 1, out_pbits[s][1]);
		}
	if (info->shared_pbits != 0)
		for (uint32_t s = 0; s < subsets; s++) _put(out_block, &at, 1, out_pbits[s][0]);

	// Indices, raster order, with each subset's anchor one bit narrower.
	for (uint32_t texel = 0; texel < 16; texel++) {
		uint32_t s     = texel_subset[texel];
		bool     is_at = texel == ktx2_uastc_bc7_anchor(subsets, pattern, s);
		_put(out_block, &at, info->index_bits - (is_at ? 1u : 0u), index[texel]);
	}
	if (info->index2_bits != 0)
		for (uint32_t texel = 0; texel < 16; texel++)
			_put(out_block, &at, info->index2_bits - (texel == 0 ? 1u : 0u), index2[texel]);
}
