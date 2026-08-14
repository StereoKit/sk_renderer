// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// L1f - UASTC LDR 4x4 block decode. A 19 mode subset of ASTC in a flat 128-bit
// block, with no entropy coding and no codebook, so this file has no reader, no
// arena and no state.
//
// Three departures from ASTC proper, worth knowing before reading:
//
//  - BISE is simplified: trit and quint group codes first, then plain bits, with
//    none of KDF 18.2's interleaving. Weights skip BISE entirely.
//  - Weights are plain LSB-first bits after the endpoints, not reversed from the
//    end of the block the way ASTC packs them.
//  - Blue contraction is unused, so endpoints decode in the order given. UASTC
//    spends that freedom on anchor weights: each subset's first weight drops its
//    MSB, which is what lets the wider modes fit in 128 bits at all.

#include "ktx2_internal.h"

#include <string.h>

const ktx2_uastc_mode_t ktx2_uastc_modes[19] = {
	{ 4, 19, 1, 1, 3, 1, 1, 0 }, { 2, 20, 1, 1, 3, 1, 1, 0 }, { 3,  8, 2, 1, 3, 1, 1, 0 },
	{ 2,  7, 3, 1, 3, 1, 1, 0 }, { 2, 12, 2, 1, 3, 1, 1, 0 }, { 3, 20, 1, 1, 3, 1, 1, 0 },
	{ 2, 18, 1, 2, 3, 1, 1, 0 }, { 2, 12, 2, 1, 3, 1, 1, 0 }, { 0,  0, 0, 0, 0, 0, 0, 0 },
	{ 2,  8, 2, 1, 4, 1, 1, 1 }, { 4, 13, 1, 1, 4, 0, 0, 1 }, { 2, 13, 1, 2, 4, 0, 0, 1 },
	{ 3, 19, 1, 1, 4, 0, 0, 1 }, { 1, 20, 1, 2, 4, 1, 1, 1 }, { 2, 20, 1, 1, 4, 1, 1, 1 },
	{ 4, 20, 1, 1, 2, 1, 1, 1 }, { 2, 20, 2, 1, 2, 1, 1, 1 }, { 2, 20, 1, 2, 2, 1, 1, 1 },
	{ 5, 11, 1, 1, 3, 1, 1, 0 },
};

const ktx2_bise_t ktx2_bise_range[21] = {
	{ 1, 0 }, { 0, 1 }, { 2, 0 }, { 0, 2 }, { 1, 1 }, { 3, 0 }, { 1, 2 },
	{ 2, 1 }, { 4, 0 }, { 2, 2 }, { 3, 1 }, { 5, 0 }, { 3, 2 }, { 4, 1 },
	{ 6, 0 }, { 4, 2 }, { 5, 1 }, { 7, 0 }, { 5, 2 }, { 6, 1 }, { 8, 0 },
};

// An incomplete final group narrows to what its largest value needs, where ASTC
// would use a fixed-size packet.
static const uint8_t k_trit_group_bits [6] = { 0, 2, 4, 5, 7, 8 };
static const uint8_t k_quint_group_bits[4] = { 0, 3, 5, 7 };

// Weight dequantization to [0,64], shared with ASTC. Indexed by weight bits.
static const uint8_t k_weight1[ 2] = { 0, 64 };
static const uint8_t k_weight2[ 4] = { 0, 21, 43, 64 };
static const uint8_t k_weight3[ 8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const uint8_t k_weight4[16] = { 0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64 };
static const uint8_t k_weight5[32] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                      34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64 };

static const uint8_t* _weight_table(uint32_t weight_bits) {
	switch (weight_bits) {
	case 1:  return k_weight1;
	case 2:  return k_weight2;
	case 3:  return k_weight3;
	case 4:  return k_weight4;
	default: return k_weight5;
	}
}

///////////////////////////////////////////////////////////////////////////////

void ktx2_uastc_unpack(const uint8_t block[16], ktx2_uastc_t* out_block) {
	memset(out_block, 0, sizeof(*out_block));
	out_block->mode = KTX2_UASTC_MODE_INVALID;

	uint32_t mode = ktx2_uastc_mode_lut[block[0] & 0x7F];
	if (mode > 18) return; // 19 is reserved for a future format

	ktx2_bits_t bits;
	ktx2_bits_init(&bits, block, 16);
	ktx2_bits_get (&bits, ktx2_uastc_mode_bits[mode]);

	if (mode == KTX2_UASTC_MODE_VOID) {
		for (uint32_t c = 0; c < 4; c++)
			out_block->solid[c] = (uint8_t)ktx2_bits_get(&bits, 8);
		out_block->mode = (uint8_t)mode;
		return;
	}

	const ktx2_uastc_mode_t* info = &ktx2_uastc_modes[mode];

	// Transcode hints, skipped wholesale: they carry encoder search results for
	// ETC1/ETC2/BC1, which we only ever reach from ETC1S sources.
	ktx2_bits_get(&bits, 1 + info->bc1_hint1);       // BC1H0, BC1H1
	ktx2_bits_get(&bits, 8);                         // ETC1F, ETC1D, ETC1I0, ETC1I1
	ktx2_bits_get(&bits, info->etc1_bias ? 5 : 0);   // ETC1BIAS
	ktx2_bits_get(&bits, info->etc2_hint ? 8 : 0);   // ETC2TM

	if (info->planes == 2)
		out_block->compsel = ktx2_uastc_has_compsel(info)
			? (uint8_t)ktx2_bits_get(&bits, 2)
			: KTX2_UASTC_LA_COMPSEL;

	if (info->subsets > 1) {
		uint32_t pattern = ktx2_bits_get(&bits, info->subsets == 3 ? 4 : 5);
		out_block->pattern_index = (uint8_t)pattern;
		out_block->pattern       = ktx2_uastc_pattern(mode, info->subsets, pattern);
		if (out_block->pattern == NULL) return; // pattern index with nothing behind it
	}

	// Endpoints: every trit or quint group first, then one plain-bit code per
	// value, recombined into the quantized integer.
	uint32_t         count = (uint32_t)info->comps * 2 * info->subsets;
	const ktx2_bise_t bise = ktx2_bise_range[info->endpoint_range];
	uint8_t          extra[18];

	memset(extra, 0, sizeof(extra));
	if (bise.tq != 0) {
		uint32_t per   = bise.tq == 1 ? 5 : 3;
		uint32_t radix = bise.tq == 1 ? 3 : 5;
		for (uint32_t at = 0; at < count; at += per) {
			uint32_t size  = count - at < per ? count - at : per;
			uint32_t group = ktx2_bits_get(&bits, bise.tq == 1
				? k_trit_group_bits [size]
				: k_quint_group_bits[size]);
			for (uint32_t i = 0; i < size; i++) {
				extra[at + i] = (uint8_t)(group % radix);
				group /= radix;
			}
		}
	}
	for (uint32_t i = 0; i < count; i++)
		out_block->quant[i] = (uint8_t)((extra[i] << bise.bits) | ktx2_bits_get(&bits, bise.bits));

	// Each subset's first texel is an anchor: its weight drops the MSB, which the
	// encoder guaranteed zero by swapping endpoints. Dual plane anchors cover both.
	uint32_t seen = 0;
	for (uint32_t texel = 0; texel < 16; texel++) {
		uint32_t subset = out_block->pattern != NULL ? out_block->pattern[texel] : 0;
		int32_t  width  = info->weight_bits - ((seen & (1u << subset)) == 0 ? 1 : 0);
		seen |= 1u << subset;
		for (uint32_t plane = 0; plane < info->planes; plane++)
			out_block->weights[texel * info->planes + plane] = (uint8_t)ktx2_bits_get(&bits, width);
	}
	if (bits.overrun) return;

	out_block->mode           = (uint8_t)mode;
	out_block->subsets        = info->subsets;
	out_block->planes         = info->planes;
	out_block->comps          = info->comps;
	out_block->weight_bits    = info->weight_bits;
	out_block->endpoint_range = info->endpoint_range;
}

///////////////////////////////////////////////////////////////////////////////

// Trit and quint ranges do not decode in monotonic order, so they need the
// spec's table. Binary ranges bit-replicate, exact for the 4, 5 and 8 bit ones
// UASTC uses.
void ktx2_uastc_endpoints(const ktx2_uastc_t* block, uint8_t out_low[3][4], uint8_t out_high[3][4]) {
	const uint8_t* table = ktx2_uastc_dequant(block->endpoint_range);
	uint32_t       ebits = ktx2_bise_range[block->endpoint_range].bits;

	for (uint32_t subset = 0; subset < block->subsets; subset++) {
		uint8_t  value[8];
		uint32_t at = subset * block->comps * 2u;
		for (uint32_t i = 0; i < block->comps * 2u; i++) {
			uint32_t quant = block->quant[at + i];
			value[i] = table != NULL
				? table[quant]
				: (uint8_t)((quant << (8 - ebits)) | (quant >> (2 * ebits - 8)));
		}
		// LA replicates luminance across RGB; RGB pins alpha opaque, which
		// interpolates to itself and needs no special case downstream.
		bool la = block->comps == 2;
		out_low [subset][0] = out_low [subset][1] = out_low [subset][2] = value[0];
		out_high[subset][0] = out_high[subset][1] = out_high[subset][2] = value[1];
		if (!la) {
			out_low [subset][1] = value[2]; out_high[subset][1] = value[3];
			out_low [subset][2] = value[4]; out_high[subset][2] = value[5];
		}
		out_low [subset][3] = block->comps == 4 ? value[6] : (la ? value[2] : 255);
		out_high[subset][3] = block->comps == 4 ? value[7] : (la ? value[3] : 255);
	}
}

// ASTC's interpolation, widened to 16 bits and back. The spec's sRGB variant
// pads with 0x80 instead of replicating; no encoder enables it, so neither do we.
static inline uint8_t _interpolate(uint32_t low, uint32_t high, uint32_t weight) {
	low  = (low  << 8) | low;
	high = (high << 8) | high;
	return (uint8_t)(((low * (64 - weight) + high * weight + 32) >> 6) >> 8);
}

void ktx2_uastc_to_rgba(const ktx2_uastc_t* block, uint8_t out_texels[64]) {
	if (block->mode == KTX2_UASTC_MODE_INVALID) {
		for (uint32_t texel = 0; texel < 16; texel++) {
			out_texels[texel * 4 + 0] = KTX2_UASTC_INVALID_R;
			out_texels[texel * 4 + 1] = KTX2_UASTC_INVALID_G;
			out_texels[texel * 4 + 2] = KTX2_UASTC_INVALID_B;
			out_texels[texel * 4 + 3] = 255;
		}
		return;
	}
	if (block->mode == KTX2_UASTC_MODE_VOID) {
		for (uint32_t texel = 0; texel < 16; texel++)
			memcpy(out_texels + texel * 4, block->solid, 4);
		return;
	}

	uint8_t low [3][4];
	uint8_t high[3][4];
	ktx2_uastc_endpoints(block, low, high);

	const uint8_t* weights = _weight_table(block->weight_bits);
	for (uint32_t texel = 0; texel < 16; texel++) {
		uint32_t subset = block->pattern != NULL ? block->pattern[texel] : 0;
		uint32_t first  = weights[block->weights[texel * block->planes]];
		uint32_t second = block->planes == 2 ? weights[block->weights[texel * 2 + 1]] : first;
		for (uint32_t c = 0; c < 4; c++) {
			uint32_t weight = (block->planes == 2 && c == block->compsel) ? second : first;
			out_texels[texel * 4 + c] = _interpolate(low[subset][c], high[subset][c], weight);
		}
	}
}
