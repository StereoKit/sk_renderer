// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// UASTC -> BC7 partition mapping and BC7 anchor indices.
//
// UASTC's 60 patterns are the ones ASTC and BC7 have in common, so each maps
// onto a BC7 pattern with no texel changing subset. Not the identity though:
// BC7 numbers subsets its own way, so the pattern index comes with an inversion
// flag (2-subset), a permutation (3-subset), or a 3-to-2 collapse (mode 7, which
// is 2-subset in ASTC but rides a 3-subset BC7 pattern). All three are verified
// against KDF Tables 127 and 128 in uastc_test.c.
//
// **BC7 anchors are a fixed table, not each subset's first texel**: only 14 of
// the 64 two-subset patterns coincide. Assuming UASTC's rule here silently
// writes the wrong index widths.

#include "ktx2_internal.h"

// UASTC 2-subset pattern -> BC7 pattern, and whether BC7 numbers the subsets
// the other way round.
static const uint8_t k_bc7_from2[30] = {
	  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
	 15,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  29,  32,  33,  52,
};
static const uint8_t k_bc7_invert2[30] = {
	  0,   0,   1,   0,   1,   0,   1,   1,   0,   1,   0,   1,   1,   1,   0,
	  1,   1,   1,   0,   0,   0,   1,   1,   0,   1,   0,   1,   1,   1,   1,
};

// UASTC 3-subset pattern -> BC7 pattern, plus which subset permutation applies.
static const uint8_t k_bc7_from3[11] = {
	  4,   8,   9,  10,  11,  12,  13,  20,  35,  36,  57,
};
static const uint8_t k_bc7_perm3[11] = {
	  0,   5,   5,   2,   2,   0,   4,   1,   1,   5,   0,
};

// UASTC mode 7: 2 subsets in ASTC, carried on a 3-subset BC7 pattern with two
// endpoint pairs set equal.
static const uint8_t k_bc7_from72[19] = {
	 10,  11,   0,   2,   8,  13,   1,  33,  40,  20,  21,  58,   3,  32,  59,  34,  20,  14,  31,
};
static const uint8_t k_bc7_k72[19] = {
	  4,   4,   3,   4,   5,   4,   2,   2,   3,   4,   0,   3,   0,   2,   1,   3,   1,   4,   3,
};

// ASTC subset -> BC7 subset, indexed by the permutation selector above.
static const uint8_t k_bc7_subset_perm[6][3] = {
	{ 0, 1, 2 }, { 1, 2, 0 }, { 2, 0, 1 }, { 2, 1, 0 }, { 0, 2, 1 }, { 1, 0, 2 },
};

// KDF Table 131: the anchor texel of subset 1 in each 2-subset pattern.
static const uint8_t k_bc7_anchor2[64] = {
	 15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,
	 15,   2,   8,   2,   2,   8,   8,  15,   2,   8,   2,   2,   8,   8,   2,   2,
	 15,  15,   6,   8,   2,   8,  15,  15,   2,   8,   2,   2,   2,  15,  15,   6,
	  6,   2,   6,   8,  15,  15,   2,   2,  15,  15,  15,  15,  15,   2,   2,  15,
};

// KDF Tables 129 and 130: subsets 1 and 2 of each 3-subset pattern. Subset 0
// always anchors on texel 0.
static const uint8_t k_bc7_anchor32[64] = {
	  3,   3,  15,  15,   8,   3,  15,  15,   8,   8,   6,   6,   6,   5,   3,   3,
	  3,   3,   8,  15,   3,   3,   6,  10,   5,   8,   8,   6,   8,   5,  15,  15,
	  8,  15,   3,   5,   6,  10,   8,  15,  15,   3,  15,   5,  15,  15,  15,  15,
	  3,  15,   5,   5,   5,   8,   5,  10,   5,  10,   8,  13,  15,  12,   3,   3,
};
static const uint8_t k_bc7_anchor33[64] = {
	 15,   8,   8,   3,  15,  15,   3,   8,  15,  15,  15,  15,  15,  15,  15,   8,
	 15,   8,  15,   3,  15,   8,  15,   8,   3,  15,   6,  10,  15,  15,  10,   8,
	 15,   3,  15,  10,  10,   8,   9,  10,   6,  15,   8,  15,   3,   6,   6,   8,
	 15,   3,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,   3,  15,  15,   8,
};

///////////////////////////////////////////////////////////////////////////////

uint32_t ktx2_uastc_bc7_pattern(uint32_t mode, uint32_t subsets, uint32_t pattern) {
	if (mode    == 7) return k_bc7_from72[pattern];
	if (subsets == 3) return k_bc7_from3 [pattern];
	return                   k_bc7_from2 [pattern];
}

// Maps a texel's UASTC subset onto the subset BC7 will assign it. The three
// mode families disagree about how, which is the whole reason this exists.
uint32_t ktx2_uastc_bc7_subset(uint32_t mode, uint32_t subsets, uint32_t pattern, uint32_t subset) {
	if (mode    == 7) return subset; // the collapse runs the other way; see the writer
	if (subsets == 3) return k_bc7_subset_perm[k_bc7_perm3[pattern]][subset];
	return k_bc7_invert2[pattern] ? 1u - subset : subset;
}

// Mode 7 only, and many-to-one: three BC7 subsets against two UASTC ones, so a
// texel's BC7 subset comes from the BC7 pattern and this says which UASTC
// endpoint pair it gets. Two BC7 subsets share endpoints, which is the point.
uint32_t ktx2_uastc_bc7_collapse(uint32_t pattern, uint32_t bc7_subset) {
	uint32_t k = k_bc7_k72[pattern];
	uint32_t p;
	switch (k >> 1) {
	case 0:  p = bc7_subset <= 1 ? 0u : 1u;                    break;
	case 1:  p = bc7_subset == 0 ? 0u : 1u;                    break;
	default: p = (bc7_subset == 0 || bc7_subset == 2) ? 0u : 1u; break;
	}
	return (k & 1) ? 1u - p : p;
}

uint32_t ktx2_uastc_bc7_anchor(uint32_t bc7_subsets, uint32_t pattern, uint32_t subset) {
	if (subset == 0)      return 0;
	if (bc7_subsets == 2) return k_bc7_anchor2[pattern];
	return subset == 1 ? k_bc7_anchor32[pattern] : k_bc7_anchor33[pattern];
}

///////////////////////////////////////////////////////////////////////////////
// BC7's own partition patterns, KDF Tables 127 and 128. The mapping above avoids
// these in the common case, but two situations need them:
//
//  - Mode 7 is many-to-one, so a texel's BC7 subset is only knowable here.
//  - A single-subset UASTC mode can target a 2-subset BC7 mode (UASTC 1 -> BC7 3,
//    keeping the 2-bit weights exact). Endpoints are duplicated, but the indices
//    still follow BC7 partition 0, so subset 1's anchor still narrows.

static const uint8_t k_bc7_pattern2[64][16] = {
	{ 0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1 },
	{ 0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1 },
	{ 0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1 },
	{ 0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1 },
	{ 0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1 },
	{ 0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1 },
	{ 0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1 },
	{ 0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1 },
	{ 0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1 },
	{ 0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1 },
	{ 0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1 },
	{ 0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1 },
	{ 0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1 },
	{ 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1 },
	{ 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1 },
	{ 0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1 },
	{ 0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1 },
	{ 0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0 },
	{ 0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0 },
	{ 0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0 },
	{ 0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0 },
	{ 0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0 },
	{ 0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0 },
	{ 0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1 },
	{ 0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0 },
	{ 0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0 },
	{ 0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0 },
	{ 0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0 },
	{ 0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0 },
	{ 0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0 },
	{ 0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0 },
	{ 0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0 },
	{ 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1 },
	{ 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 },
	{ 0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0 },
	{ 0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0 },
	{ 0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0 },
	{ 0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0 },
	{ 0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1 },
	{ 0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1 },
	{ 0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0 },
	{ 0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0 },
	{ 0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0 },
	{ 0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0 },
	{ 0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0 },
	{ 0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1 },
	{ 0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1 },
	{ 0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0 },
	{ 0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0 },
	{ 0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0 },
	{ 0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0 },
	{ 0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0 },
	{ 0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1 },
	{ 0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1 },
	{ 0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0 },
	{ 0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0 },
	{ 0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1 },
	{ 0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1 },
	{ 0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1 },
	{ 0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1 },
	{ 0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1 },
	{ 0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0 },
	{ 0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0 },
	{ 0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1 },
};

static const uint8_t k_bc7_pattern3[64][16] = {
	{ 0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2 },
	{ 0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1 },
	{ 0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1 },
	{ 0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1 },
	{ 0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2 },
	{ 0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2 },
	{ 0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1 },
	{ 0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1 },
	{ 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2 },
	{ 0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2 },
	{ 0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2 },
	{ 0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2 },
	{ 0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2 },
	{ 0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2 },
	{ 0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2 },
	{ 0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0 },
	{ 0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2 },
	{ 0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0 },
	{ 0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2 },
	{ 0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1 },
	{ 0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2 },
	{ 0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1 },
	{ 0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2 },
	{ 0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0 },
	{ 0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0 },
	{ 0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2 },
	{ 0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0 },
	{ 0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1 },
	{ 0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2 },
	{ 0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2 },
	{ 0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1 },
	{ 0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1 },
	{ 0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2 },
	{ 0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1 },
	{ 0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2 },
	{ 0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0 },
	{ 0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0 },
	{ 0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0 },
	{ 0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0 },
	{ 0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1 },
	{ 0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1 },
	{ 0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2 },
	{ 0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1 },
	{ 0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2 },
	{ 0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1 },
	{ 0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1 },
	{ 0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1 },
	{ 0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1 },
	{ 0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2 },
	{ 0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1 },
	{ 0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2 },
	{ 0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2 },
	{ 0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2 },
	{ 0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2 },
	{ 0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2 },
	{ 0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2 },
	{ 0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2 },
	{ 0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2 },
	{ 0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2 },
	{ 0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2 },
	{ 0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1 },
	{ 0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2 },
	{ 0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2 },
	{ 0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0 },
};

uint32_t ktx2_uastc_bc7_texel_subset(uint32_t bc7_subsets, uint32_t bc7_pattern, uint32_t texel) {
	if (bc7_subsets < 2) return 0;
	return bc7_subsets == 2 ? k_bc7_pattern2[bc7_pattern][texel] : k_bc7_pattern3[bc7_pattern][texel];
}
