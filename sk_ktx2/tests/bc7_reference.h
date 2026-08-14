// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// A reference BC7 decoder, transcribed from KDF 20 and written separately from
// the transcoder it checks.
//
// Same reasoning as astc_reference.h: the `_to_bc7` goldens are the same eight
// files and reach only UASTC modes 0, 12 and 15. Comparing our own output
// against ktx2_uastc_to_rgba covers every mode - as a quality floor, not
// equality, since requantizing to BC7's precision is lossy by design.

#pragma once

#include "ktx2_internal.h"
#include "bc7_partitions.h"

#include <string.h>

static const uint8_t k_bc7_weight2[ 4] = { 0, 21, 43, 64 };
static const uint8_t k_bc7_weight3[ 8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const uint8_t k_bc7_weight4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };


static uint32_t bc7_ref_get(const uint8_t block[16], uint32_t* ref_at, uint32_t width) {
	uint32_t value = 0;
	for (uint32_t i = 0; i < width; i++, (*ref_at)++)
		if (block[*ref_at >> 3] & (1u << (*ref_at & 7))) value |= 1u << i;
	return value;
}

static uint8_t bc7_ref_expand(uint32_t value, uint32_t bits) {
	value <<= (8 - bits);
	return (uint8_t)(value | (value >> bits));
}

static uint8_t bc7_ref_lerp(uint32_t low, uint32_t high, uint32_t weight) {
	return (uint8_t)((low * (64 - weight) + high * weight + 32) >> 6);
}

// False for anything unexpected, which is itself a failure here.
static bool bc7_ref_decode(const uint8_t block[16], uint8_t out_texels[64]) {
	// Mode is a run of zeroes terminated by a one, in the low byte.
	uint32_t mode = 0;
	while (mode < 8 && (block[0] & (1u << mode)) == 0) mode++;
	if (mode >= 8) return false; // low byte zero is reserved

	static const uint8_t k_ns [8] = { 3, 2, 3, 2, 1, 1, 1, 2 };
	static const uint8_t k_pb [8] = { 4, 6, 6, 6, 0, 0, 0, 6 };
	static const uint8_t k_rb [8] = { 0, 0, 0, 0, 2, 2, 0, 0 };
	static const uint8_t k_isb[8] = { 0, 0, 0, 0, 1, 0, 0, 0 };
	static const uint8_t k_cb [8] = { 4, 6, 5, 7, 5, 7, 7, 5 };
	static const uint8_t k_ab [8] = { 0, 0, 0, 0, 6, 8, 7, 5 };
	static const uint8_t k_epb[8] = { 1, 0, 0, 1, 0, 0, 1, 1 };
	static const uint8_t k_spb[8] = { 0, 1, 0, 0, 0, 0, 0, 0 };
	static const uint8_t k_ib [8] = { 3, 3, 2, 2, 2, 2, 4, 2 };
	static const uint8_t k_ib2[8] = { 0, 0, 0, 0, 3, 2, 0, 0 };

	uint32_t subsets = k_ns[mode];
	uint32_t at      = mode + 1;
	uint32_t pattern = bc7_ref_get(block, &at, k_pb [mode]);
	uint32_t rotate  = bc7_ref_get(block, &at, k_rb [mode]);
	uint32_t index_s = bc7_ref_get(block, &at, k_isb[mode]);
	if (index_s != 0) return false; // never emitted; mode 4 only

	uint8_t low[3][4], high[3][4];
	for (uint32_t c = 0; c < 3; c++)
		for (uint32_t s = 0; s < subsets; s++) {
			low [s][c] = (uint8_t)bc7_ref_get(block, &at, k_cb[mode]);
			high[s][c] = (uint8_t)bc7_ref_get(block, &at, k_cb[mode]);
		}
	for (uint32_t s = 0; s < subsets; s++) {
		low [s][3] = 255;
		high[s][3] = 255;
	}
	if (k_ab[mode] != 0)
		for (uint32_t s = 0; s < subsets; s++) {
			low [s][3] = (uint8_t)bc7_ref_get(block, &at, k_ab[mode]);
			high[s][3] = (uint8_t)bc7_ref_get(block, &at, k_ab[mode]);
		}

	uint32_t pbit_low[3] = { 0, 0, 0 }, pbit_high[3] = { 0, 0, 0 };
	bool     has_pbit    = k_epb[mode] != 0 || k_spb[mode] != 0;
	if (k_epb[mode] != 0)
		for (uint32_t s = 0; s < subsets; s++) {
			pbit_low [s] = bc7_ref_get(block, &at, 1);
			pbit_high[s] = bc7_ref_get(block, &at, 1);
		}
	if (k_spb[mode] != 0)
		for (uint32_t s = 0; s < subsets; s++)
			pbit_low[s] = pbit_high[s] = bc7_ref_get(block, &at, 1);

	// The p-bit is the endpoint's least significant bit, below the stored ones.
	uint8_t full_low[3][4], full_high[3][4];
	for (uint32_t s = 0; s < subsets; s++) {
		for (uint32_t c = 0; c < 4; c++) {
			uint32_t colour = c < 3 || k_ab[mode] != 0;
			uint32_t bits   = c < 3 ? k_cb[mode] : k_ab[mode];
			if (!colour) { full_low[s][c] = 255; full_high[s][c] = 255; continue; }
			uint32_t lo = low[s][c], hi = high[s][c];
			if (has_pbit) {
				lo = (lo << 1) | pbit_low [s];
				hi = (hi << 1) | pbit_high[s];
				bits += 1;
			}
			full_low [s][c] = bc7_ref_expand(lo, bits);
			full_high[s][c] = bc7_ref_expand(hi, bits);
		}
	}

	uint32_t anchor[3] = { 0, 0, 0 };
	if (subsets == 2) anchor[1] = k_bc7_ref_anchor2[pattern];
	if (subsets == 3) { anchor[1] = k_bc7_ref_anchor32[pattern]; anchor[2] = k_bc7_ref_anchor33[pattern]; }

	uint32_t index[16], index2[16];
	for (uint32_t texel = 0; texel < 16; texel++) {
		uint32_t s     = subsets == 1 ? 0u
		               : (subsets == 2 ? k_bc7_pattern2[pattern][texel] : k_bc7_pattern3[pattern][texel]);
		bool     is_at = texel == anchor[s];
		index[texel]   = bc7_ref_get(block, &at, k_ib[mode] - (is_at ? 1u : 0u));
	}
	if (k_ib2[mode] != 0)
		for (uint32_t texel = 0; texel < 16; texel++)
			index2[texel] = bc7_ref_get(block, &at, k_ib2[mode] - (texel == 0 ? 1u : 0u));

	const uint8_t* table  = k_ib [mode] == 2 ? k_bc7_weight2 : (k_ib [mode] == 3 ? k_bc7_weight3 : k_bc7_weight4);
	const uint8_t* table2 = k_ib2[mode] == 2 ? k_bc7_weight2 : (k_ib2[mode] == 3 ? k_bc7_weight3 : k_bc7_weight4);

	for (uint32_t texel = 0; texel < 16; texel++) {
		uint32_t s = subsets == 1 ? 0u
		           : (subsets == 2 ? k_bc7_pattern2[pattern][texel] : k_bc7_pattern3[pattern][texel]);
		uint8_t  out[4];
		for (uint32_t c = 0; c < 4; c++) {
			uint32_t weight = (c == 3 && k_ib2[mode] != 0)
				? table2[index2[texel]]
				: table [index [texel]];
			out[c] = bc7_ref_lerp(full_low[s][c], full_high[s][c], weight);
		}
		// Rotation names a channel swapped into alpha at encode time.
		if (rotate != 0) { uint8_t swap = out[rotate - 1]; out[rotate - 1] = out[3]; out[3] = swap; }
		memcpy(out_texels + texel * 4, out, 4);
	}
	return true;
}
