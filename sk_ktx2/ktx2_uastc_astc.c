// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// L1g - UASTC LDR 4x4 -> ASTC 4x4, lossless and without decoding pixels: UASTC
// is an ASTC subset, so this is a re-layout of fields that already exist. KDF
// 25.8.1 specifies it normatively, KDF 23 supplies the ASTC side.
//
// Four things move rather than copy:
//
//  - Weights are written in reverse bit order down from bit 127; UASTC packs
//    them forward, its one deliberate break from ASTC.
//  - Endpoints go back through real BISE, trits and quints interleaved among the
//    value bits (KDF 23.12). UASTC's grouped form exists to avoid exactly this.
//  - The UASTC pattern index becomes the ASTC seed generating the same shape.
//  - Blue contraction has to be suppressed: an ASTC decoder applies it whenever
//    the low endpoint sums higher, and UASTC endpoint order is arbitrary because
//    it spends that freedom on anchors. Swapping the pair and inverting the
//    subset's weights is exact, the weight tables being symmetric about 32.
//
// UASTC drops the MSB of each subset's first weight and ASTC stores every weight
// whole. There is room because ASTC carries none of the transcode hint bits.

#include "ktx2_internal.h"

#include <string.h>

///////////////////////////////////////////////////////////////////////////////

// Mirrors the weight stream into the top of the block. A forward byte lands
// whole in a mirrored byte position, so ten lookups replace an 80-iteration loop.
static const uint8_t k_reverse8[256] = {
	0x00,0x80,0x40,0xC0,0x20,0xA0,0x60,0xE0,0x10,0x90,0x50,0xD0,0x30,0xB0,0x70,0xF0,
	0x08,0x88,0x48,0xC8,0x28,0xA8,0x68,0xE8,0x18,0x98,0x58,0xD8,0x38,0xB8,0x78,0xF8,
	0x04,0x84,0x44,0xC4,0x24,0xA4,0x64,0xE4,0x14,0x94,0x54,0xD4,0x34,0xB4,0x74,0xF4,
	0x0C,0x8C,0x4C,0xCC,0x2C,0xAC,0x6C,0xEC,0x1C,0x9C,0x5C,0xDC,0x3C,0xBC,0x7C,0xFC,
	0x02,0x82,0x42,0xC2,0x22,0xA2,0x62,0xE2,0x12,0x92,0x52,0xD2,0x32,0xB2,0x72,0xF2,
	0x0A,0x8A,0x4A,0xCA,0x2A,0xAA,0x6A,0xEA,0x1A,0x9A,0x5A,0xDA,0x3A,0xBA,0x7A,0xFA,
	0x06,0x86,0x46,0xC6,0x26,0xA6,0x66,0xE6,0x16,0x96,0x56,0xD6,0x36,0xB6,0x76,0xF6,
	0x0E,0x8E,0x4E,0xCE,0x2E,0xAE,0x6E,0xEE,0x1E,0x9E,0x5E,0xDE,0x3E,0xBE,0x7E,0xFE,
	0x01,0x81,0x41,0xC1,0x21,0xA1,0x61,0xE1,0x11,0x91,0x51,0xD1,0x31,0xB1,0x71,0xF1,
	0x09,0x89,0x49,0xC9,0x29,0xA9,0x69,0xE9,0x19,0x99,0x59,0xD9,0x39,0xB9,0x79,0xF9,
	0x05,0x85,0x45,0xC5,0x25,0xA5,0x65,0xE5,0x15,0x95,0x55,0xD5,0x35,0xB5,0x75,0xF5,
	0x0D,0x8D,0x4D,0xCD,0x2D,0xAD,0x6D,0xED,0x1D,0x9D,0x5D,0xDD,0x3D,0xBD,0x7D,0xFD,
	0x03,0x83,0x43,0xC3,0x23,0xA3,0x63,0xE3,0x13,0x93,0x53,0xD3,0x33,0xB3,0x73,0xF3,
	0x0B,0x8B,0x4B,0xCB,0x2B,0xAB,0x6B,0xEB,0x1B,0x9B,0x5B,0xDB,0x3B,0xBB,0x7B,0xFB,
	0x07,0x87,0x47,0xC7,0x27,0xA7,0x67,0xE7,0x17,0x97,0x57,0xD7,0x37,0xB7,0x77,0xF7,
	0x0F,0x8F,0x4F,0xCF,0x2F,0xAF,0x6F,0xEF,0x1F,0x9F,0x5F,0xDF,0x3F,0xBF,0x7F,0xFF,
};

// Five trits into the 8-bit packed form of KDF 23.12, inverting the spec's decode
// rather than tabulating it. uastc_test.c round-trips all 243 tuples.
uint32_t ktx2_uastc_pack_trits(const uint8_t trit[5]) {
	uint32_t c;
	if      (trit[2] == 2 && trit[1] == 2) c = 0x0Cu | trit[0];                       // C[3:2] = 11 escape
	else if (trit[2] == 2)                 c = (uint32_t)(trit[1] << 4) | (uint32_t)(trit[0] << 2) | 3u;
	else                                   c = (uint32_t)(trit[2] << 4) | (uint32_t)(trit[1] << 2) | trit[0];

	if (trit[4] == 2 && trit[3] == 2) return ((c & 0x1Cu) << 3) | 0x1Cu | (c & 3u);   // T[4:2] = 111 escape
	if (trit[4] == 2)                 return ((uint32_t)trit[3] << 7) | 0x60u | c;
	return ((uint32_t)trit[4] << 7) | ((uint32_t)trit[3] << 5) | c;
}

// Three quints into the 7-bit packed form, same reasoning as above.
uint32_t ktx2_uastc_pack_quints(const uint8_t quint[3]) {
	// q0 and q1 both saturated is its own escape, scattering q2 across three
	// non-adjacent bits instead of storing it as a field.
	if (quint[0] == 4 && quint[1] == 4) {
		if (quint[2] == 4) return 6u | 1u;
		return (((uint32_t)quint[2] >> 1) << 4) | (((uint32_t)quint[2] & 1u) << 3) | 6u;
	}
	uint32_t c = quint[1] == 4
		? ((uint32_t)quint[0] << 3) | 5u                  // C[2:0] = 101 escape
		: ((uint32_t)quint[1] << 3) | quint[0];

	// C[2:1] is never 3, so neither branch below can be mistaken for the escape.
	if (quint[2] == 4) return ((~(c >> 1) & 3u) << 5) | (c & 0x18u) | 6u | (c & 1u);
	return ((uint32_t)quint[2] << 5) | c;
}

// One BISE stream at `at`, trits and quints interleaved among the value bits at
// the positions KDF Tables 169-171 fix.
//
// A short final group stops after its last value's fragment, which is the stream
// length ASTC computes. The decoder reads the missing high code bits as zeroes,
// consistent because the packers put low trits in low bits; writing the whole
// group would spill into the weight data.
static void _write_bise(uint8_t bits[16], uint32_t at, const uint8_t* values, uint32_t count, ktx2_bise_t bise) {
	uint32_t n    = bise.bits;
	uint32_t mask = n == 0 ? 0u : (1u << n) - 1u;
	if (bise.tq == 0) {
		for (uint32_t i = 0; i < count; i++) ktx2_bits_put(bits, at + i * n, n, values[i]);
		return;
	}

	// Bit widths of the packed-code fragments that follow each value.
	static const uint8_t k_trit_split [5] = { 2, 2, 1, 2, 1 };
	static const uint8_t k_quint_split[3] = { 3, 2, 2 };

	uint32_t per = bise.tq == 1 ? 5 : 3;
	for (uint32_t base = 0; base < count; base += per) {
		uint8_t extra[5] = { 0, 0, 0, 0, 0 };
		for (uint32_t i = 0; i < per && base + i < count; i++)
			extra[i] = (uint8_t)(values[base + i] >> n);

		uint32_t packed = bise.tq == 1 ? ktx2_uastc_pack_trits(extra) : ktx2_uastc_pack_quints(extra);
		uint32_t taken  = 0;
		for (uint32_t i = 0; i < per && base + i < count; i++) {
			uint32_t width = bise.tq == 1 ? k_trit_split[i] : k_quint_split[i];
			ktx2_bits_put(bits, at, n, values[base + i] & mask);
			at += n;
			ktx2_bits_put(bits, at, width, (packed >> taken) & ((1u << width) - 1u));
			at    += width;
			taken += width;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

// Weight ranges are the first twelve BISE ranges; UASTC reaches only pure-bit ones.
static uint32_t _weight_range(uint32_t weight_bits) {
	switch (weight_bits) {
	case 1:  return 0;
	case 2:  return 2;
	case 3:  return 5;
	case 4:  return 8;
	default: return 11;
	}
}

// The weight range is not stored; the block mode carries it as a 3-bit selector
// plus a precision bit (KDF Table 160). Every UASTC mode uses a 4x4 weight grid,
// which is KDF Table 161 row 1: width 0, height 2.
uint32_t ktx2_uastc_block_mode(uint32_t weight_bits, uint32_t planes) {
	uint32_t range     = _weight_range(weight_bits);
	uint32_t precision = range >= 6;
	uint32_t selector  = (range % 6) + 2;
	return ((planes == 2 ? 1u : 0u) << 10)
	     | (precision << 9)
	     | (0u << 7)  // weight grid width  - 4
	     | (2u << 5)  // weight grid height - 2
	     | ((selector & 1u) << 4)
	     | (((selector >> 2) & 1u) << 1)
	     | ((selector >> 1) & 1u);
}

void ktx2_uastc_write_astc(const ktx2_uastc_t* block, uint8_t out_block[16]) {
	memset(out_block, 0, 16);

	if (block->mode == KTX2_UASTC_MODE_VOID || block->mode == KTX2_UASTC_MODE_INVALID) {
		// Void extent, also where unreadable blocks land, as magenta - matching the
		// RGBA path. KDF 25.8.1.1: dynamic range clear, all extent bits set, the
		// 8-bit colour replicated to 16.
		uint8_t solid[4] = { KTX2_UASTC_INVALID_R, KTX2_UASTC_INVALID_G, KTX2_UASTC_INVALID_B, 255 };
		if (block->mode == KTX2_UASTC_MODE_VOID) memcpy(solid, block->solid, 4);

		ktx2_bits_put(out_block, 0, 12, 0xDFCu); // void extent signature, LDR
		ktx2_bits_put(out_block, 12, 13, 0x1FFFu);
		ktx2_bits_put(out_block, 25, 13, 0x1FFFu);
		ktx2_bits_put(out_block, 38, 13, 0x1FFFu);
		ktx2_bits_put(out_block, 51, 13, 0x1FFFu);
		for (uint32_t c = 0; c < 4; c++)
			ktx2_bits_put(out_block, 64 + c * 16, 16, ((uint32_t)solid[c] << 8) | solid[c]);
		return;
	}

	uint32_t weight_count = 16u * block->planes;
	uint32_t weight_span  = weight_count * block->weight_bits;
	uint32_t weight_top   = 128 - weight_span;

	ktx2_bits_put(out_block,  0, 11, ktx2_uastc_block_mode(block->weight_bits, block->planes));
	ktx2_bits_put(out_block, 11,  2, block->subsets - 1u);

	uint32_t endpoint_at;
	uint32_t cem = block->comps == 2 ? 4u : (block->comps == 3 ? 8u : 12u);
	if (block->subsets > 1) {
		ktx2_bits_put(out_block, 13, 10, ktx2_uastc_astc_seed(block->mode, block->subsets, block->pattern_index));
		// CEM selector 00 is one shared 4-bit mode for every partition, all UASTC
		// needs, and it keeps endpoint data at a fixed offset.
		ktx2_bits_put(out_block, 23,  2, 0);
		ktx2_bits_put(out_block, 25,  4, cem);
		endpoint_at = 29;
	} else {
		ktx2_bits_put(out_block, 13,  4, cem);
		endpoint_at = 17;
	}

	if (block->planes == 2)
		ktx2_bits_put(out_block, weight_top - 2, 2, block->compsel);

	// On the dequantized values, because that is what the decoder compares: trit
	// and quint ranges are not monotonic, so the quantized integers cannot stand in.
	uint8_t  quant  [18];
	uint8_t  weights[32];
	uint32_t invert = 0; // bit per subset
	memcpy(quant,   block->quant,   sizeof(quant));
	memcpy(weights, block->weights, sizeof(weights));

	if (cem != 4) {
		const uint8_t* table = ktx2_uastc_dequant(block->endpoint_range);
		uint32_t       ebits = ktx2_bise_range[block->endpoint_range].bits;
		for (uint32_t subset = 0; subset < block->subsets; subset++) {
			uint32_t at  = subset * block->comps * 2u;
			int32_t  sum = 0;
			for (uint32_t i = 0; i < 6; i++) {
				uint32_t value = quant[at + i];
				int32_t  unq   = table != NULL
					? table[value]
					: (int32_t)((value << (8 - ebits)) | (value >> (2 * ebits - 8)));
				sum += (i & 1) ? unq : -unq; // high endpoints positive, low negative
			}
			if (sum >= 0) continue;
			for (uint32_t i = 0; i < block->comps * 2u; i += 2) {
				uint8_t swap    = quant[at + i];
				quant[at + i]   = quant[at + i + 1];
				quant[at+i + 1] = swap;
			}
			invert |= 1u << subset;
		}
	}
	if (invert != 0) {
		uint32_t top = (1u << block->weight_bits) - 1u;
		for (uint32_t texel = 0; texel < 16; texel++) {
			uint32_t subset = block->pattern != NULL ? block->pattern[texel] : 0;
			if ((invert & (1u << subset)) == 0) continue;
			for (uint32_t plane = 0; plane < block->planes; plane++) {
				uint32_t at = texel * block->planes + plane;
				weights[at] = (uint8_t)(top - weights[at]);
			}
		}
	}

	_write_bise(out_block, endpoint_at, quant, (uint32_t)block->comps * 2u * block->subsets,
		ktx2_bise_range[block->endpoint_range]);

	// Weights fill downward from bit 127, so build forward and mirror: stream bit
	// i lands at block bit 127-i, or forward byte k reversed into block byte 15-k.
	uint8_t packed[16];
	memset(packed, 0, sizeof(packed));
	for (uint32_t i = 0; i < weight_count; i++)
		ktx2_bits_put(packed, i * block->weight_bits, block->weight_bits, weights[i]);
	for (uint32_t byte = 0; byte * 8 < weight_span; byte++)
		out_block[15 - byte] |= k_reverse8[packed[byte]];
}
