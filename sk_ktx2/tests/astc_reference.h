// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// A reference ASTC 4x4 LDR decoder, transcribed from KDF 23 and written
// separately from the transcoder it checks.
//
// The byte-exact ASTC goldens are thin: the eight CTS files use only UASTC modes
// 0, 12 and 15, so no golden covers a multi-subset block, a dual plane block, a
// void extent or a quint endpoint range. Decoding our own output against our
// RGBA output - which the spec's 64 block vectors pin exactly - reaches every
// mode on real images instead.
//
// It shares the dequantization tables with the library rather than restating
// them; those are already pinned by the block vectors, and the repacking around
// them is what is under test.

#pragma once

#include "ktx2_internal.h"

#include <string.h>

static uint32_t astc_ref_get(const uint8_t* block, uint32_t at, uint32_t width) {
	uint32_t value = 0;
	for (uint32_t i = 0; i < width; i++)
		if (block[(at + i) >> 3] & (1u << ((at + i) & 7))) value |= 1u << i;
	return value;
}

///////////////////////////////////////////////////////////////////////////////
// KDF 23.20. The partition index is generated from a seed, not stored, so a
// decoder has to reproduce the generator exactly.

static uint32_t astc_ref_hash52(uint32_t p) {
	p ^= p >> 15; p -= p << 17; p += p << 7; p += p << 4;
	p ^= p >>  5; p += p << 16; p ^= p >> 7; p ^= p >> 3;
	p ^= p <<  6; p ^= p >> 17;
	return p;
}

static uint32_t astc_ref_partition(uint32_t seed, uint32_t x, uint32_t y, uint32_t count) {
	x <<= 1; y <<= 1; // small_block: a 4x4 footprint is under 31 texels

	seed += (count - 1) * 1024;
	uint32_t rnum = astc_ref_hash52(seed);
	uint32_t s[12];
	s[ 0] = (rnum      ) & 0xF; s[ 1] = (rnum >>  4) & 0xF;
	s[ 2] = (rnum >>  8) & 0xF; s[ 3] = (rnum >> 12) & 0xF;
	s[ 4] = (rnum >> 16) & 0xF; s[ 5] = (rnum >> 20) & 0xF;
	s[ 6] = (rnum >> 24) & 0xF; s[ 7] = (rnum >> 28) & 0xF;
	s[ 8] = (rnum >> 18) & 0xF; s[ 9] = (rnum >> 22) & 0xF;
	s[10] = (rnum >> 26) & 0xF; s[11] = ((rnum >> 30) | (rnum << 2)) & 0xF;
	for (uint32_t i = 0; i < 12; i++) s[i] *= s[i];

	uint32_t sh1, sh2;
	if (seed & 1) { sh1 = (seed & 2) ? 4 : 5; sh2 = (count == 3) ? 6 : 5; }
	else          { sh1 = (count == 3) ? 6 : 5; sh2 = (seed & 2) ? 4 : 5; }
	uint32_t sh3 = (seed & 0x10) ? sh1 : sh2;

	s[0] >>= sh1; s[1] >>= sh2; s[2] >>= sh1; s[3] >>= sh2;
	s[4] >>= sh1; s[5] >>= sh2; s[6] >>= sh1; s[7] >>= sh2;
	s[8] >>= sh3; s[9] >>= sh3; s[10] >>= sh3; s[11] >>= sh3;

	uint32_t a = (s[0] * x + s[1] * y + (rnum >> 14)) & 0x3F;
	uint32_t b = (s[2] * x + s[3] * y + (rnum >> 10)) & 0x3F;
	uint32_t c = (s[4] * x + s[5] * y + (rnum >>  6)) & 0x3F;
	uint32_t d = (s[6] * x + s[7] * y + (rnum >>  2)) & 0x3F;
	if (count < 4) d = 0;
	if (count < 3) c = 0;

	if (a >= b && a >= c && a >= d) return 0;
	if (b >= c && b >= d)           return 1;
	if (c >= d)                     return 2;
	return 3;
}

///////////////////////////////////////////////////////////////////////////////
// KDF 23.12, decode direction: trits and quints interleaved among the value
// bits, not grouped ahead of them the way UASTC stores them.

static void astc_ref_trits(uint32_t t, uint8_t out_trit[5]) {
	uint32_t c;
	if (((t >> 2) & 7) == 7) {
		c = (((t >> 5) & 7) << 2) | (t & 3);
		out_trit[4] = out_trit[3] = 2;
	} else {
		c = t & 0x1F;
		if (((t >> 5) & 3) == 3) { out_trit[4] = 2;                       out_trit[3] = (uint8_t)((t >> 7) & 1); }
		else                     { out_trit[4] = (uint8_t)((t >> 7) & 1); out_trit[3] = (uint8_t)((t >> 5) & 3); }
	}
	if      ((c & 3)        == 3) { out_trit[2] = 2; out_trit[1] = (uint8_t)((c >> 4) & 1);
	                                out_trit[0] = (uint8_t)((((c >> 3) & 1) << 1) | (((c >> 2) & 1) & ~((c >> 3) & 1))); }
	else if (((c >> 2) & 3) == 3) { out_trit[2] = 2; out_trit[1] = 2; out_trit[0] = (uint8_t)(c & 3); }
	else                          { out_trit[2] = (uint8_t)((c >> 4) & 1); out_trit[1] = (uint8_t)((c >> 2) & 3);
	                                out_trit[0] = (uint8_t)((((c >> 1) & 1) << 1) | ((c & 1) & ~((c >> 1) & 1))); }
}

static void astc_ref_quints(uint32_t q, uint8_t out_quint[3]) {
	uint32_t c;
	if (((q >> 1) & 3) == 3 && ((q >> 5) & 3) == 0) {
		out_quint[2] = (uint8_t)(((q & 1) << 2) | ((((q >> 4) & 1) & ~q) << 1) | (((q >> 3) & 1) & ~q));
		out_quint[1] = out_quint[0] = 4;
		return;
	}
	if (((q >> 1) & 3) == 3) { out_quint[2] = 4; c = (((q >> 3) & 3) << 3) | ((~(q >> 5) & 3) << 1) | (q & 1); }
	else                     { out_quint[2] = (uint8_t)((q >> 5) & 3); c = q & 0x1F; }

	if ((c & 7) == 5) { out_quint[1] = 4;                       out_quint[0] = (uint8_t)((c >> 3) & 3); }
	else              { out_quint[1] = (uint8_t)((c >> 3) & 3); out_quint[0] = (uint8_t)(c & 7); }
}

static void astc_ref_read_bise(const uint8_t* block, uint32_t at, uint8_t* out_values,
                               uint32_t count, uint32_t bits, uint32_t tq) {
	if (tq == 0) {
		for (uint32_t i = 0; i < count; i++) out_values[i] = (uint8_t)astc_ref_get(block, at + i * bits, bits);
		return;
	}
	static const uint8_t k_trit_split [5] = { 2, 2, 1, 2, 1 };
	static const uint8_t k_quint_split[3] = { 3, 2, 2 };

	uint32_t per = tq == 1 ? 5 : 3;
	for (uint32_t base = 0; base < count; base += per) {
		uint8_t  low[5];
		uint32_t packed = 0, taken = 0;
		for (uint32_t i = 0; i < per && base + i < count; i++) {
			uint32_t width = tq == 1 ? k_trit_split[i] : k_quint_split[i];
			low[i]  = (uint8_t)astc_ref_get(block, at, bits);
			at     += bits;
			packed |= astc_ref_get(block, at, width) << taken;
			at     += width;
			taken  += width;
		}
		uint8_t extra[5] = { 0, 0, 0, 0, 0 };
		if (tq == 1) astc_ref_trits (packed, extra);
		else         astc_ref_quints(packed, extra);
		for (uint32_t i = 0; i < per && base + i < count; i++)
			out_values[base + i] = (uint8_t)((extra[i] << bits) | low[i]);
	}
}

///////////////////////////////////////////////////////////////////////////////

static uint32_t astc_ref_ise_bits(uint32_t count, ktx2_bise_t bise) {
	if (bise.tq == 1) return (count * 8 + 4) / 5 + count * bise.bits;
	if (bise.tq == 2) return (count * 7 + 2) / 3 + count * bise.bits;
	return count * bise.bits;
}

static uint8_t astc_ref_interpolate(uint32_t low, uint32_t high, uint32_t weight) {
	low  = (low  << 8) | low;
	high = (high << 8) | high;
	return (uint8_t)(((low * (64 - weight) + high * weight + 32) >> 6) >> 8);
}

// False for anything unexpected, which is itself a failure here.
static bool astc_ref_decode(const uint8_t block[16], uint8_t out_texels[64]) {
	static const uint8_t k_weight1[ 2] = { 0, 64 };
	static const uint8_t k_weight2[ 4] = { 0, 21, 43, 64 };
	static const uint8_t k_weight3[ 8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
	static const uint8_t k_weight4[16] = { 0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64 };
	static const uint8_t k_weight5[32] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
	                                      34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64 };

	uint32_t mode = astc_ref_get(block, 0, 11);

	// Void extent: bits [8..0] are 111111100.
	if ((mode & 0x1FF) == 0x1FC) {
		if ((mode >> 9) & 1) return false; // HDR
		for (uint32_t texel = 0; texel < 16; texel++)
			for (uint32_t c = 0; c < 4; c++)
				out_texels[texel * 4 + c] = (uint8_t)(astc_ref_get(block, 64 + c * 16, 16) >> 8);
		return true;
	}

	// KDF Table 161 row 1, the only shape a 4x4 weight grid can take.
	uint32_t planes    = ((mode >> 10) & 1) ? 2u : 1u;
	uint32_t precision =  (mode >>  9) & 1;
	uint32_t selector  = (((mode >> 1) & 1) << 2) | (((mode >> 0) & 1) << 1) | ((mode >> 4) & 1);
	if (((mode >> 2) & 3) != 0) return false; // not row 1
	if (((mode >> 7) & 3) != 0) return false; // weight grid width != 4
	if (((mode >> 5) & 3) != 2) return false; // weight grid height != 4
	if (selector < 2)           return false; // reserved

	uint32_t    weight_range = precision * 6 + (selector - 2);
	ktx2_bise_t weight_bise  = ktx2_bise_range[weight_range];
	if (weight_bise.tq != 0) return false; // UASTC never uses trit or quint weights

	uint32_t subsets = astc_ref_get(block, 11, 2) + 1;
	uint32_t seed    = 0;
	uint32_t cem;
	uint32_t endpoint_at;
	if (subsets > 1) {
		seed = astc_ref_get(block, 13, 10);
		if (astc_ref_get(block, 23, 2) != 0) return false; // per-partition CEMs
		cem         = astc_ref_get(block, 25, 4);
		endpoint_at = 29;
	} else {
		cem         = astc_ref_get(block, 13, 4);
		endpoint_at = 17;
	}
	if (cem != 4 && cem != 8 && cem != 12) return false;

	uint32_t comps       = cem == 4 ? 2u : (cem == 8 ? 3u : 4u);
	uint32_t count       = comps * 2 * subsets;
	uint32_t weight_span = 16 * planes * weight_bise.bits;
	uint32_t compsel     = planes == 2 ? astc_ref_get(block, 128 - weight_span - 2, 2) : 0;

	// The endpoint range is never stored; it is whatever the leftover bits allow.
	uint32_t config    = (subsets > 1 ? 29u : 17u) + (planes == 2 ? 2u : 0u);
	uint32_t remaining = 128 - config - weight_span;
	uint32_t range     = 0;
	for (uint32_t r = 0; r <= 20; r++)
		if (astc_ref_ise_bits(count, ktx2_bise_range[r]) <= remaining) range = r;

	uint8_t quant[18];
	memset(quant, 0, sizeof(quant));
	astc_ref_read_bise(block, endpoint_at, quant, count, ktx2_bise_range[range].bits, ktx2_bise_range[range].tq);

	// Weights run in reverse bit order from the top, so mirror them forward first.
	uint8_t forward[16];
	memset(forward, 0, sizeof(forward));
	for (uint32_t i = 0; i < weight_span; i++)
		if (block[(127 - i) >> 3] & (1u << ((127 - i) & 7)))
			forward[i >> 3] |= (uint8_t)(1u << (i & 7));

	const uint8_t* weight_table = weight_bise.bits == 1 ? k_weight1
	                            : weight_bise.bits == 2 ? k_weight2
	                            : weight_bise.bits == 3 ? k_weight3
	                            : weight_bise.bits == 4 ? k_weight4 : k_weight5;

	const uint8_t* dequant = ktx2_uastc_dequant(range);
	uint32_t       ebits   = ktx2_bise_range[range].bits;
	uint8_t        low [4][4];
	uint8_t        high[4][4];
	for (uint32_t subset = 0; subset < subsets; subset++) {
		uint8_t  value[8];
		uint32_t at = subset * comps * 2;
		for (uint32_t i = 0; i < comps * 2; i++) {
			uint32_t q = quant[at + i];
			value[i] = dequant != NULL
				? dequant[q]
				: (uint8_t)((q << (8 - ebits)) | (q >> (2 * ebits - 8)));
		}
		if (cem == 4) {
			low [subset][0] = low [subset][1] = low [subset][2] = value[0]; low [subset][3] = value[2];
			high[subset][0] = high[subset][1] = high[subset][2] = value[1]; high[subset][3] = value[3];
			continue;
		}
		// KDF 23.14.1.6 and 23.14.1.9: taken as written only when the high endpoint
		// sums at least as large, else blue-contracted and swapped. A transcoder
		// that leaves this reachable corrupts the block.
		uint32_t s0 = (uint32_t)value[0] + value[2] + value[4];
		uint32_t s1 = (uint32_t)value[1] + value[3] + value[5];
		uint8_t  a0 = cem == 12 ? value[6] : 255;
		uint8_t  a1 = cem == 12 ? value[7] : 255;
		if (s1 >= s0) {
			low [subset][0] = value[0]; low [subset][1] = value[2]; low [subset][2] = value[4]; low [subset][3] = a0;
			high[subset][0] = value[1]; high[subset][1] = value[3]; high[subset][2] = value[5]; high[subset][3] = a1;
		} else {
			low [subset][0] = (uint8_t)((value[1] + value[5]) >> 1);
			low [subset][1] = (uint8_t)((value[3] + value[5]) >> 1);
			low [subset][2] = value[5]; low [subset][3] = a1;
			high[subset][0] = (uint8_t)((value[0] + value[4]) >> 1);
			high[subset][1] = (uint8_t)((value[2] + value[4]) >> 1);
			high[subset][2] = value[4]; high[subset][3] = a0;
		}
	}

	for (uint32_t texel = 0; texel < 16; texel++) {
		uint32_t subset = subsets > 1
			? astc_ref_partition(seed, texel & 3, texel >> 2, subsets)
			: 0;
		uint32_t first  = weight_table[astc_ref_get(forward, (texel * planes) * weight_bise.bits, weight_bise.bits)];
		uint32_t second = planes == 2
			? weight_table[astc_ref_get(forward, (texel * 2 + 1) * weight_bise.bits, weight_bise.bits)]
			: first;
		for (uint32_t c = 0; c < 4; c++) {
			uint32_t weight = (planes == 2 && c == compsel) ? second : first;
			out_texels[texel * 4 + c] = astc_ref_interpolate(low[subset][c], high[subset][c], weight);
		}
	}
	return true;
}
