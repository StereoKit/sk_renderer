// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// L1b / L1d - a decoded ETC1S block into output. ETC1S is a constrained ETC1:
// differential mode, zero colour deltas, equal intensity codewords, flip clear.
// So the ETC1 repack is bit assembly with no search and no loss.
//
// The trap is selector ordering. The spec calls the stored selectors "pixel
// index bits as defined by Khronos ETC1", which reads as though they were
// already wire values. They are not: the codebook holds sorted-modifier order
// {-large, -small, +small, +large} against ETC1's wire order {+small, +large,
// -small, -large}. Wrong still yields a well-formed block with a scrambled
// ramp, so it is invisible without a reference. Brute-forced against the CTS
// goldens, then cross-checked: ktx2_etc1_modifier[i][perm[s]] == sorted[i][s]
// for every intensity and selector.

#include "ktx2_internal.h"

// Declared in ktx2_internal.h; the grey and BC1 paths read this same table.
const uint8_t ktx2_selector_to_etc1[4] = { 3, 2, 0, 1 };

// [intensity][pixel index], columns in ETC1 wire order.
const int16_t ktx2_etc1_modifier[8][4] = {
	{  2,   8,  -2,   -8 },
	{  5,  17,  -5,  -17 },
	{  9,  29,  -9,  -29 },
	{ 13,  42, -13,  -42 },
	{ 18,  60, -18,  -60 },
	{ 24,  80, -24,  -80 },
	{ 33, 106, -33, -106 },
	{ 47, 183, -47, -183 },
};

static inline uint8_t _clamp_u8(int32_t value) {
	return (uint8_t)(value < 0 ? 0 : (value > 255 ? 255 : value));
}

// 5 to 8 bits by replication, as ETC1 hardware does for the differential base.
static inline uint8_t _expand5(uint8_t value) {
	return (uint8_t)((value << 3) | (value >> 2));
}

void ktx2_etc1s_write_etc1(const ktx2_endpoint_t* endpoint, const ktx2_selector_t* selector, uint8_t out_block[8]) {
	// Differential mode with a zero delta, so both subblocks resolve alike.
	out_block[0] = (uint8_t)(endpoint->color5[0] << 3);
	out_block[1] = (uint8_t)(endpoint->color5[1] << 3);
	out_block[2] = (uint8_t)(endpoint->color5[2] << 3);
	// Both intensity codewords equal, diff set, flip clear.
	out_block[3] = (uint8_t)((endpoint->inten << 5) | (endpoint->inten << 2) | 2);

	// MSB and LSB planes, 16 bits each, texel (x,y) at bit x*4+y. Column-major,
	// unlike the row-major selector bytes.
	uint32_t msb = 0;
	uint32_t lsb = 0;
	for (uint32_t y = 0; y < 4; y++) {
		uint32_t row = selector->rows[y];
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t index = ktx2_selector_to_etc1[(row >> (x * 2)) & 3];
			uint32_t bit   = x * 4 + y;
			msb |= (index >> 1) << bit;
			lsb |= (index &  1) << bit;
		}
	}
	out_block[4] = (uint8_t)(msb >> 8);
	out_block[5] = (uint8_t)(msb & 0xFF);
	out_block[6] = (uint8_t)(lsb >> 8);
	out_block[7] = (uint8_t)(lsb & 0xFF);
}

void ktx2_etc1s_write_rgba(const ktx2_endpoint_t* endpoint, const ktx2_selector_t* selector, uint8_t out_texels[64]) {
	int32_t base[3];
	for (int32_t c = 0; c < 3; c++) base[c] = _expand5(endpoint->color5[c]);

	for (uint32_t y = 0; y < 4; y++) {
		uint32_t row = selector->rows[y];
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t index    = ktx2_selector_to_etc1[(row >> (x * 2)) & 3];
			int32_t  modifier = ktx2_etc1_modifier[endpoint->inten][index];
			uint8_t* texel    = out_texels + (y * 4 + x) * 4;
			texel[0] = _clamp_u8(base[0] + modifier);
			texel[1] = _clamp_u8(base[1] + modifier);
			texel[2] = _clamp_u8(base[2] + modifier);
			texel[3] = 255;
		}
	}
}
