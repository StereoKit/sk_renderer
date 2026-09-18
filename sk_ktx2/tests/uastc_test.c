// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// UASTC LDR 4x4 block decode, gated on the spec's own 64 test vectors: random
// blocks with their exact decoded texels, invalid ones included. A stronger and
// more direct gate than the CTS files - no container, no corpus, and the block
// decoder is isolated from everything around it.
//
// The layout check is independent. Rather than tabulate field offsets, it
// re-derives each mode's endpoint, weight and total bit counts from the mode
// table and compares them against the totals the spec prints per mode. A wrong
// mode table entry almost always lands on the wrong total.

#include "ktx2_internal.h"
#include "uastc_vectors.h"
#include "astc_reference.h"
#include "bc7_partitions.h"
#include "bc7_reference.h"

#include <stdio.h>
#include <string.h>

static int32_t failures = 0;

// BC7 requantizes into fewer bits, so a channel can shift. Past this is
// structural - a wrong partition, a missed anchor swap - not rounding.
#define BC7_MAX_CHANNEL_ERROR 24

static double _rms(double sse, double count) {
	double x = sse / count, guess = x > 1.0 ? x : 1.0;
	for (int32_t i = 0; i < 40; i++) guess = 0.5 * (guess + x / guess);
	return guess;
}

///////////////////////////////////////////////////////////////////////////////

static uint32_t _group_bits(uint32_t tq, uint32_t size) {
	static const uint8_t trit [6] = { 0, 2, 4, 5, 7, 8 };
	static const uint8_t quint[4] = { 0, 3, 5, 7 };
	return tq == 1 ? trit[size] : quint[size];
}

static void _check_layout(void) {
	for (uint32_t mode = 0; mode < 19; mode++) {
		if (mode == KTX2_UASTC_MODE_VOID) continue; // void extent has no endpoints or weights

		const ktx2_uastc_mode_t* info = &ktx2_uastc_modes[mode];
		ktx2_bise_t              bise = ktx2_bise_range[info->endpoint_range];
		uint32_t                 count = (uint32_t)info->comps * 2 * info->subsets;

		uint32_t endpoint = count * bise.bits;
		if (bise.tq != 0) {
			uint32_t per = bise.tq == 1 ? 5 : 3;
			for (uint32_t at = 0; at < count; at += per)
				endpoint += _group_bits(bise.tq, count - at < per ? count - at : per);
		}

		// One anchor per subset; dual plane anchors drop a bit from both weights.
		uint32_t weight = 16 * info->planes * info->weight_bits - info->subsets * info->planes;

		uint32_t header = ktx2_uastc_mode_bits[mode]
			+ 1 + info->bc1_hint1 + 8
			+ (info->etc1_bias ? 5 : 0)
			+ (info->etc2_hint ? 8 : 0)
			+ (ktx2_uastc_has_compsel(info) ? 2 : 0)
			+ (info->subsets  > 1 ? (info->subsets == 3 ? 4 : 5) : 0);

		uint32_t total = header + endpoint + weight;
		if (endpoint != k_uastc_mode_budget[mode][1] ||
		    weight   != k_uastc_mode_budget[mode][2] ||
		    total    != k_uastc_mode_budget[mode][0]) {
			printf("  mode %2u layout: got total %u endpoint %u weight %u, spec says %u %u %u\n",
				mode, total, endpoint, weight,
				k_uastc_mode_budget[mode][0], k_uastc_mode_budget[mode][1], k_uastc_mode_budget[mode][2]);
			failures++;
		}
		if (total > 128) { printf("  mode %2u overflows the block\n", mode); failures++; }
	}
	printf("layout: 18 modes match the specification's stated bit budgets\n");
}

///////////////////////////////////////////////////////////////////////////////

static void _check_vectors(void) {
	uint32_t seen[21];
	uint32_t bad = 0;
	memset(seen, 0, sizeof(seen));

	for (uint32_t v = 0; v < UASTC_VECTOR_COUNT; v++) {
		ktx2_uastc_t block;
		uint8_t      texels[64];
		ktx2_uastc_unpack (k_uastc_vector_blocks[v], &block);
		ktx2_uastc_to_rgba(&block, texels);
		seen[block.mode]++;

		for (uint32_t texel = 0; texel < 16; texel++) {
			uint32_t want = k_uastc_vector_texels[v][texel];
			uint32_t got  = (uint32_t)texels[texel * 4 + 0]
			              | (uint32_t)texels[texel * 4 + 1] << 8
			              | (uint32_t)texels[texel * 4 + 2] << 16
			              | (uint32_t)texels[texel * 4 + 3] << 24;
			if (got == want) continue;
			if (bad < 12)
				printf("  block %2u mode %2u texel %2u: got %08X want %08X\n", v, block.mode, texel, got, want);
			bad++;
			break;
		}
	}

	printf("vectors: %u/%u blocks exact\n", UASTC_VECTOR_COUNT - bad, UASTC_VECTOR_COUNT);
	printf("         mode coverage:");
	for (uint32_t mode = 0; mode < 21; mode++)
		if (seen[mode] != 0) printf(" %u:%u", mode, seen[mode]);
	printf("\n");
	if (bad != 0) failures++;
}

///////////////////////////////////////////////////////////////////////////////
// Mode 7 is the one mode both the published vectors and the CTS corpus miss. It
// is field-for-field identical to mode 4 apart from which partition table its
// PAT index selects, so one synthesized block per pattern pins the difference.
//
// Every field is zero except PAT and subset 1's six endpoint bit codes, leaving
// the subsets on quantized endpoints 0 and 1. Those dequantize far apart, so a
// texel's value names its subset directly.

static void _put_bits(uint8_t block[16], uint32_t at, uint32_t width, uint32_t value) {
	for (uint32_t i = 0; i < width; i++, at++)
		if (value & (1u << i)) block[at >> 3] |= (uint8_t)(1u << (at & 7));
}

static void _check_mode7(void) {
	for (uint32_t pattern = 0; pattern < 19; pattern++) {
		uint8_t block[16];
		memset(block, 0, sizeof(block));
		_put_bits(block,  0, 5, 0x07);    // MODE, mode 7's Huffman code
		_put_bits(block, 20, 5, pattern); // PAT
		for (uint32_t i = 6; i < 12; i++)
			_put_bits(block, 53 + i * 3, 3, 1); // EBITS, subset 1 only

		ktx2_uastc_t unpacked;
		uint8_t      texels[64];
		ktx2_uastc_unpack (block, &unpacked);
		ktx2_uastc_to_rgba(&unpacked, texels);

		if (unpacked.mode != 7) {
			printf("  pattern %2u: unpacked as mode %u, wanted 7\n", pattern, unpacked.mode);
			failures++;
			continue;
		}
		// Without contrast between the subsets the map below proves nothing.
		bool contrast = false;
		for (uint32_t texel = 0; texel < 16; texel++)
			if (texels[texel * 4] != texels[0]) contrast = true;
		if (!contrast) { printf("  pattern %2u: both subsets decoded alike\n", pattern); failures++; continue; }

		const uint8_t* want = k_uastc_pattern7[pattern];
		for (uint32_t texel = 0; texel < 16; texel++) {
			if ((texels[texel * 4] != texels[0]) == (want[texel] != want[0])) continue;
			printf("  pattern %2u texel %2u: subset map disagrees with the specification\n", pattern, texel);
			failures++;
			break;
		}
	}
	printf("mode 7: 19 partition patterns match the specification\n");
}

///////////////////////////////////////////////////////////////////////////////
// The library packs trits and quints back into interleaved BISE by inverting the
// spec's decode rather than tabulating it. This keeps the two from drifting: it
// runs every tuple through the packer and back out through an independent
// transcription of KDF 23.12.
//
// It also checks the truncation property the repack relies on: a short final
// group stops after its last value, so the packed code's high bits are never
// written and must already be zero.

static void _unpack_trits(uint32_t t, uint8_t out_trit[5]) {
	uint32_t c;
	if (((t >> 2) & 7) == 7) {
		c = (((t >> 5) & 7) << 2) | (t & 3);
		out_trit[4] = out_trit[3] = 2;
	} else {
		c = t & 0x1F;
		if (((t >> 5) & 3) == 3) { out_trit[4] = 2;                     out_trit[3] = (uint8_t)((t >> 7) & 1); }
		else                     { out_trit[4] = (uint8_t)((t >> 7) & 1); out_trit[3] = (uint8_t)((t >> 5) & 3); }
	}
	if      ((c & 3)        == 3) { out_trit[2] = 2; out_trit[1] = (uint8_t)((c >> 4) & 1);
	                                out_trit[0] = (uint8_t)((((c >> 3) & 1) << 1) | (((c >> 2) & 1) & ~((c >> 3) & 1))); }
	else if (((c >> 2) & 3) == 3) { out_trit[2] = 2; out_trit[1] = 2; out_trit[0] = (uint8_t)(c & 3); }
	else                          { out_trit[2] = (uint8_t)((c >> 4) & 1); out_trit[1] = (uint8_t)((c >> 2) & 3);
	                                out_trit[0] = (uint8_t)((((c >> 1) & 1) << 1) | ((c & 1) & ~((c >> 1) & 1))); }
}

static void _unpack_quints(uint32_t q, uint8_t out_quint[3]) {
	uint32_t c;
	if (((q >> 1) & 3) == 3 && ((q >> 5) & 3) == 0) {
		out_quint[2] = (uint8_t)(((q & 1) << 2) | ((((q >> 4) & 1) & ~q) << 1) | (((q >> 3) & 1) & ~q));
		out_quint[1] = out_quint[0] = 4;
		return;
	}
	if (((q >> 1) & 3) == 3) { out_quint[2] = 4; c = (((q >> 3) & 3) << 3) | ((~(q >> 5) & 3) << 1) | (q & 1); }
	else                     { out_quint[2] = (uint8_t)((q >> 5) & 3); c = q & 0x1F; }

	if ((c & 7) == 5) { out_quint[1] = 4;                        out_quint[0] = (uint8_t)((c >> 3) & 3); }
	else              { out_quint[1] = (uint8_t)((c >> 3) & 3); out_quint[0] = (uint8_t)(c & 7); }
}

static void _check_ise(void) {
	// Bits a final group of each size has, from the interleave's fragment widths.
	static const uint8_t k_trit_room [6] = { 0, 2, 4, 5, 7, 8 };
	static const uint8_t k_quint_room[4] = { 0, 3, 5, 7 };
	int32_t trit_bad = 0, quint_bad = 0, room_bad = 0;

	for (uint32_t v = 0; v < 243; v++) {
		uint8_t  want[5], got[5];
		uint32_t rest = v;
		for (uint32_t i = 0; i < 5; i++) { want[i] = (uint8_t)(rest % 3); rest /= 3; }
		uint32_t packed = ktx2_uastc_pack_trits(want);
		_unpack_trits(packed, got);
		if (memcmp(want, got, 5) != 0) trit_bad++;

		uint32_t used = 5;
		while (used > 1 && want[used - 1] == 0) used--;
		if ((packed >> k_trit_room[used]) != 0) room_bad++;
	}
	for (uint32_t v = 0; v < 125; v++) {
		uint8_t  want[3], got[3];
		uint32_t rest = v;
		for (uint32_t i = 0; i < 3; i++) { want[i] = (uint8_t)(rest % 5); rest /= 5; }
		uint32_t packed = ktx2_uastc_pack_quints(want);
		_unpack_quints(packed, got);
		if (memcmp(want, got, 3) != 0) quint_bad++;

		uint32_t used = 3;
		while (used > 1 && want[used - 1] == 0) used--;
		if ((packed >> k_quint_room[used]) != 0) room_bad++;
	}

	if (trit_bad || quint_bad || room_bad) {
		printf("  ISE: %d/243 trit tuples, %d/125 quint tuples fail round trip, %d overflow a short group\n",
			trit_bad, quint_bad, room_bad);
		failures++;
	}
	printf("ISE: 243 trit and 125 quint tuples round-trip, short groups fit\n");
}

///////////////////////////////////////////////////////////////////////////////
// ASTC infers the endpoint range from the bits left after configuration and
// weights, so the repack is only correct if that inferred range is the one the
// UASTC mode used. True by design, enforced nowhere, so check every mode - and
// the block mode values against KDF Table 227 while here.

static uint32_t _ise_bits(uint32_t count, ktx2_bise_t bise) {
	if (bise.tq == 1) return (count * 8 + 4) / 5 + count * bise.bits;
	if (bise.tq == 2) return (count * 7 + 2) / 3 + count * bise.bits;
	return count * bise.bits;
}

static void _check_astc_layout(void) {
	for (uint32_t mode = 0; mode < 19; mode++) {
		if (mode == KTX2_UASTC_MODE_VOID) continue;
		const ktx2_uastc_mode_t* info = &ktx2_uastc_modes[mode];

		uint32_t got = ktx2_uastc_block_mode(info->weight_bits, info->planes);
		if (got != k_uastc_astc_block_mode[mode]) {
			printf("  mode %2u: ASTC block mode %u, KDF Table 227 says %u\n",
				mode, got, k_uastc_astc_block_mode[mode]);
			failures++;
		}

		uint32_t config    = info->subsets > 1 ? 29u : 17u;
		uint32_t ccs       = info->planes == 2 ? 2u : 0u;
		uint32_t weights   = 16u * info->planes * info->weight_bits;
		uint32_t count     = (uint32_t)info->comps * 2u * info->subsets;
		uint32_t remaining = 128u - config - ccs - weights;

		uint32_t derived = 0;
		for (uint32_t range = 0; range <= 20; range++)
			if (_ise_bits(count, ktx2_bise_range[range]) <= remaining) derived = range;

		if (derived != info->endpoint_range) {
			printf("  mode %2u: %u bits left fits range %u, but the mode encodes range %u\n",
				mode, remaining, derived, info->endpoint_range);
			failures++;
		}
	}
	printf("ASTC layout: 18 modes match KDF Table 227, endpoint ranges are inferable\n");
}

///////////////////////////////////////////////////////////////////////////////
// UASTC's 60 patterns are the ones ASTC and BC7 have in common, so every texel
// keeps its subset - but BC7 numbers subsets its own way, and mode 7 is 2-subset
// in ASTC while riding a 3-subset BC7 pattern. Three relations, checked against
// KDF Tables 127 and 128.
//
// Also checks the BC7 anchors land in the subset they claim. They are NOT the
// first texel of it - only 14 of 64 - so UASTC's rule must not be reused.

static uint32_t _collapse_3_to_2(uint32_t p, uint32_t k) {
	switch (k >> 1) {
	case 0:  p = p <= 1 ? 0u : 1u;             break;
	case 1:  p = p == 0 ? 0u : 1u;             break;
	default: p = (p == 0 || p == 2) ? 0u : 1u; break;
	}
	return (k & 1) ? 1u - p : p;
}

static void _check_bc7_partitions(void) {
	int32_t bad = 0, firsts = 0;

	for (uint32_t p = 0; p < 30; p++) {
		const uint8_t* uastc = ktx2_uastc_pattern(2, 2, p);
		uint32_t       bc7   = ktx2_uastc_bc7_pattern(2, 2, p);
		for (uint32_t t = 0; t < 16; t++)
			if (ktx2_uastc_bc7_subset(2, 2, p, uastc[t]) != k_bc7_pattern2[bc7][t]) bad++;
	}
	for (uint32_t p = 0; p < 11; p++) {
		const uint8_t* uastc = ktx2_uastc_pattern(3, 3, p);
		uint32_t       bc7   = ktx2_uastc_bc7_pattern(3, 3, p);
		for (uint32_t t = 0; t < 16; t++)
			if (ktx2_uastc_bc7_subset(3, 3, p, uastc[t]) != k_bc7_pattern3[bc7][t]) bad++;
	}
	// Mode 7 runs the other way: the BC7 pattern is the richer one.
	for (uint32_t p = 0; p < 19; p++) {
		const uint8_t* uastc = ktx2_uastc_pattern(7, 2, p);
		uint32_t       bc7   = ktx2_uastc_bc7_pattern(7, 2, p);
		for (uint32_t t = 0; t < 16; t++)
			if (uastc[t] != _collapse_3_to_2(k_bc7_pattern3[bc7][t], k_uastc_bc7_k72[p])) bad++;
		// and the library's own collapse must agree with the one written here
		for (uint32_t sub = 0; sub < 3; sub++)
			if (ktx2_uastc_bc7_collapse(p, sub) != _collapse_3_to_2(sub, k_uastc_bc7_k72[p])) bad++;
	}

	for (uint32_t b = 0; b < 64; b++) {
		if (k_bc7_pattern2[b][ktx2_uastc_bc7_anchor(2, b, 1)] != 1) bad++;
		if (k_bc7_pattern3[b][ktx2_uastc_bc7_anchor(3, b, 1)] != 1) bad++;
		if (k_bc7_pattern3[b][ktx2_uastc_bc7_anchor(3, b, 2)] != 2) bad++;
		uint32_t first = 0;
		while (first < 16 && k_bc7_pattern2[b][first] != 1) first++;
		if (ktx2_uastc_bc7_anchor(2, b, 1) == first) firsts++;
	}

	if (bad != 0) { printf("  BC7 partitions: %d mismatches\n", bad); failures++; }
	printf("BC7 mapping: 60 patterns agree with KDF Tables 127/128, anchors land correctly\n");
	printf("             %d/64 BC7 anchors are the first texel of their subset - do not assume it\n", firsts);
}

///////////////////////////////////////////////////////////////////////////////

// Every 16-byte input must land somewhere defined and repack to the same texels.
// Random blocks are what reach modes 2, 3, 7, 14 and 18, none of which occur in
// the CTS corpus, at field values no encoder would choose.
static void _check_random(void) {
	uint32_t state = 0x12345678u;
	uint32_t seen[21];
	uint32_t bad = 0, rejected = 0, bc7_bad = 0, bc7_rejected = 0, bc7_worst = 0;
	double   bc7_sse = 0;
	memset(seen, 0, sizeof(seen));

	for (uint32_t i = 0; i < 200000; i++) {
		uint8_t      raw[16], want[64], got[64], astc[16];
		ktx2_uastc_t block;
		for (uint32_t b = 0; b < 16; b++) {
			state = state * 1664525u + 1013904223u;
			raw[b] = (uint8_t)(state >> 24);
		}
		ktx2_uastc_unpack (raw, &block);
		ktx2_uastc_to_rgba(&block, want);
		seen[block.mode]++;

		// Near-lossless, so BC7 gets an error budget rather than equality.
		uint8_t bc7[16], bc7_texels[64];
		ktx2_uastc_write_bc7(&block, bc7);
		if (!bc7_ref_decode(bc7, bc7_texels)) bc7_rejected++;
		else {
			uint32_t worst = 0;
			for (uint32_t c = 0; c < 64; c++) {
				int32_t d = (int32_t)bc7_texels[c] - (int32_t)want[c];
				if (d < 0) d = -d;
				if ((uint32_t)d > worst) worst = (uint32_t)d;
				bc7_sse += (double)d * d;
			}
			if (worst > bc7_worst) bc7_worst = worst;
			if (worst > BC7_MAX_CHANNEL_ERROR) {
				if (bc7_bad == 0)
					printf("  block %u mode %u: BC7 off by %u on a channel\n", i, block.mode, worst);
				bc7_bad++;
			}
		}

		ktx2_uastc_write_astc(&block, astc);
		if (!astc_ref_decode(astc, got)) { rejected++; continue; }
		if (memcmp(want, got, 64) != 0) {
			if (bad == 0)
				printf("  block %u mode %u: ASTC decodes to %u,%u,%u,%u, UASTC to %u,%u,%u,%u\n",
					i, block.mode, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
			bad++;
		}
	}

	if (bad != 0 || rejected != 0) {
		printf("  random: %u blocks differ, %u rejected by the reference decoder\n", bad, rejected);
		failures++;
	}
	if (bc7_bad != 0 || bc7_rejected != 0) {
		printf("  random BC7: %u blocks over budget, %u rejected by the reference decoder\n",
			bc7_bad, bc7_rejected);
		failures++;
	}
	printf("random: 200000 blocks decode and round-trip through ASTC\n");
	printf("        BC7 worst channel error %u (budget %u), rms %.2f\n",
		bc7_worst, (uint32_t)BC7_MAX_CHANNEL_ERROR, _rms(bc7_sse, 200000.0 * 64.0));
	printf("        mode coverage:");
	for (uint32_t mode = 0; mode < 21; mode++)
		if (seen[mode] != 0) printf(" %u:%u", mode, seen[mode]);
	printf("\n");
	for (uint32_t mode = 0; mode < 19; mode++)
		if (seen[mode] == 0) { printf("  mode %u never generated\n", mode); failures++; }
}

int main(void) {
	_check_layout();
	_check_vectors();
	_check_mode7();
	_check_ise();
	_check_astc_layout();
	_check_bc7_partitions();
	_check_random();
	printf(failures == 0 ? "PASS\n" : "FAIL\n");
	return failures == 0 ? 0 : 1;
}
