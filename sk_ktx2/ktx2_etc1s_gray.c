// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Single-channel ETC1S output: EAC (ETC2 alpha, R11, RG11) and BC4.
//
// A block's four texel values follow from its 5-bit base and 3-bit intensity, so
// the format has only 32*8 ramps. Both targets are also "one base plus a fixed
// ramp", so this is a ramp-to-ramp fit tabulated once and applied with two
// lookups per block. Nothing decodes to pixels.
//
// It cannot be byte-identical to KTX-Software, whose EAC and BC4 paths run
// unspecified search encoders over decoded pixels. The gate is a quality bound
// instead; see tests/golden_test.c.

#include "ktx2_internal.h"

// ETC2 / EAC modifier table, indexed [table][3-bit index].
static const int16_t k_eac_modifier[16][8] = {
	{ -3,  -6,  -9, -15, 2, 5, 8, 14 }, { -3,  -7, -10, -13, 2, 6, 9, 12 },
	{ -2,  -5,  -8, -13, 1, 4, 7, 12 }, { -2,  -4,  -6, -13, 1, 3, 5, 12 },
	{ -3,  -6,  -8, -12, 2, 5, 7, 11 }, { -3,  -7,  -9, -11, 2, 6, 8, 10 },
	{ -4,  -7,  -8, -11, 3, 6, 7, 10 }, { -3,  -5,  -8, -11, 2, 4, 7, 10 },
	{ -2,  -6,  -8, -10, 1, 5, 7,  9 }, { -2,  -5,  -8, -10, 1, 4, 7,  9 },
	{ -2,  -4,  -8, -10, 1, 3, 7,  9 }, { -2,  -5,  -7, -10, 1, 4, 6,  9 },
	{ -3,  -4,  -7, -10, 2, 3, 6,  9 }, { -1,  -2,  -3, -10, 0, 1, 2,  9 },
	{ -4,  -6,  -8,  -9, 3, 5, 7,  8 }, { -3,  -5,  -7,  -9, 2, 4, 6,  8 },
};

static inline int32_t _clamp(int32_t value, int32_t low, int32_t high) {
	return value < low ? low : (value > high ? high : value);
}

static inline uint8_t _expand5(uint8_t value) {
	return (uint8_t)((value << 3) | (value >> 2));
}

// The four values a (base, intensity) pair produces, in selector order. Near
// black or white the ramp clips asymmetrically, hence indexing by base too.
static void _gray_levels(uint32_t base5, uint32_t inten, int32_t out_levels[4]) {
	int32_t base = _expand5((uint8_t)base5);
	for (int32_t s = 0; s < 4; s++)
		out_levels[s] = _clamp(base + ktx2_etc1_modifier[inten][ktx2_selector_to_etc1[s]], 0, 255);
}

///////////////////////////////////////////////////////////////////////////////

// (min selector, max selector) -> range slot. Shared with the BC1 path.
const uint8_t ktx2_range_slot[4][4] = {
	{ 0, 1, 2, 3 },
	{ 0, 4, 5, 6 },
	{ 0, 0, 7, 8 },
	{ 0, 0, 0, 9 },
};

uint32_t ktx2_selector_range(const ktx2_selector_t* selector) {
	uint32_t low = 3, high = 0;
	for (uint32_t y = 0; y < 4; y++) {
		uint32_t row = selector->rows[y];
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t s = (row >> (x * 2)) & 3;
			if (s < low ) low  = s;
			if (s > high) high = s;
		}
	}
	return ktx2_range_slot[low][high];
}

void ktx2_gray_fit_build(ktx2_gray_fit_t* out_fits) {
	// (intensity, range) outermost so base-independent work is solved once, not
	// 32 times. The other order measured 63 ms per load.
	for (uint32_t inten = 0; inten < 8; inten++) {
	for (uint32_t low   = 0; low   < 4; low++  ) {
	for (uint32_t high  = low; high < 4; high++) {
		uint32_t slot  = ktx2_range_slot[low][high];
		int32_t  count = (int32_t)(high - low + 1);

		// The EAC multiplier and table have to match the level *spacing*, which
		// the intensity table and range fix and the base does not, except where
		// the ramp clips. So search the 240 combinations once, on mid-grey.
		int32_t offsets[4];
		for (int32_t i = 0; i < count; i++)
			offsets[i] = ktx2_etc1_modifier[inten][ktx2_selector_to_etc1[low + i]];

		uint8_t shared_table = 0, shared_multiplier = 1;
		int32_t best_error   = 0x7FFFFFFF;
		for (uint32_t table = 0; table < 16 && best_error != 0; table++) {
			for (uint32_t multiplier = 1; multiplier < 16 && best_error != 0; multiplier++) {
				int32_t error = 0;
				for (int32_t i = 0; i < count; i++) {
					int32_t best_delta = 0x7FFFFFFF;
					for (uint32_t p = 0; p < 8; p++) {
						int32_t delta = k_eac_modifier[table][p] * (int32_t)multiplier - offsets[i];
						if (delta * delta < best_delta) best_delta = delta * delta;
					}
					error += best_delta;
				}
				if (error < best_error) {
					best_error        = error;
					shared_table      = (uint8_t)table;
					shared_multiplier = (uint8_t)multiplier;
				}
			}
		}

		// Likewise BC4, whose endpoints sit at fixed offsets from the base.
		// ETC1S puts its interior levels at 0.375 and 0.625 against BC4's
		// sevenths, worth ~20 levels on the widest tables, so widening the
		// endpoints buys more in the middle than it costs at the ends.
		// Pivoted on mid-grey rather than signed offsets: BC4 interpolates with
		// truncating division, so negative offsets would round differently to
		// what the writer does.
		int32_t bc4_lo_offset = offsets[0], bc4_hi_offset = offsets[count - 1];
		int32_t bc4_best      = 0x7FFFFFFF;
		for (int32_t hi = offsets[count - 1]; hi <= offsets[count - 1] + 8; hi++) {
			for (int32_t lo = offsets[0]; lo >= offsets[0] - 8; lo--) {
				int32_t high8 = 128 + hi, low8 = 128 + lo;
				if (high8 > 255 || low8 < 0) continue;
				int32_t palette[8];
				palette[0] = high8;
				palette[1] = low8;
				for (int32_t k = 1; k <= 6; k++) palette[1 + k] = ((7 - k) * high8 + k * low8) / 7;

				int32_t error = 0;
				for (int32_t i = 0; i < count; i++) {
					int32_t best_delta = 0x7FFFFFFF;
					for (int32_t p = 0; p < 8; p++) {
						int32_t delta = palette[p] - (128 + offsets[i]);
						if (delta * delta < best_delta) best_delta = delta * delta;
					}
					error += best_delta;
				}
				if (error < bc4_best) { bc4_best = error; bc4_lo_offset = lo; bc4_hi_offset = hi; }
			}
		}

		// Only the index assignments are re-derived per base, so clipping near
		// black and white - the one base-dependent effect - is respected.
		for (uint32_t base5 = 0; base5 < 32; base5++) {
			int32_t base = _expand5((uint8_t)base5);
			int32_t all_levels[4], levels[4];
			_gray_levels(base5, inten, all_levels);
			for (int32_t i = 0; i < count; i++) levels[i] = all_levels[low + i];

			ktx2_gray_fit_t* fit = &out_fits[(base5 * 8 + inten) * KTX2_BC1_RANGES + slot];
			fit->eac_table      = shared_table;
			fit->eac_multiplier = shared_multiplier;
			fit->bc4_lo         = (uint8_t)_clamp(base + bc4_lo_offset, 0, 255);
			fit->bc4_hi         = (uint8_t)_clamp(base + bc4_hi_offset, 0, 255);

			// Shared offsets assume a ramp symmetric about the base, which clipping
			// breaks - and alpha clips constantly against 255. Re-searching just
			// those entries is worth ~10 dB on them.
			if (base + offsets[0] < 0 || base + offsets[count - 1] > 255) {
				int32_t best = 0x7FFFFFFF;
				for (int32_t hi = levels[count - 1]; hi <= levels[count - 1] + 8 && hi <= 255; hi++) {
					for (int32_t lo = levels[0]; lo >= levels[0] - 8 && lo >= 0; lo--) {
						int32_t ramp[8];
						ramp[0] = hi;
						ramp[1] = lo;
						for (int32_t k = 1; k <= 6; k++) ramp[1 + k] = ((7 - k) * hi + k * lo) / 7;
						int32_t error = 0;
						for (int32_t i = 0; i < count; i++) {
							int32_t best_delta = 0x7FFFFFFF;
							for (int32_t p = 0; p < 8; p++) {
								int32_t delta = ramp[p] - levels[i];
								if (delta * delta < best_delta) best_delta = delta * delta;
							}
							error += best_delta;
						}
						if (error < best) { best = error; fit->bc4_hi = (uint8_t)hi; fit->bc4_lo = (uint8_t)lo; }
					}
				}
			}

			int32_t palette[8];
			palette[0] = fit->bc4_hi;
			palette[1] = fit->bc4_lo;
			for (int32_t k = 1; k <= 6; k++)
				palette[1 + k] = ((7 - k) * (int32_t)fit->bc4_hi + k * (int32_t)fit->bc4_lo) / 7;

			for (int32_t s = 0; s < 4; s++) {
				// Selectors outside the range cannot occur; clamp them anyway.
				int32_t used = s < (int32_t)low ? 0 : (s > (int32_t)high ? count - 1 : s - (int32_t)low);

				int32_t best_eac = 0x7FFFFFFF, best_bc4 = 0x7FFFFFFF;
				for (int32_t p = 0; p < 8; p++) {
					int32_t eac   = _clamp(base + k_eac_modifier[shared_table][p] * (int32_t)shared_multiplier, 0, 255) - levels[used];
					int32_t bc4   = palette[p] - levels[used];
					if (eac * eac < best_eac) { best_eac = eac * eac; fit->eac_index[s] = (uint8_t)p; }
					if (bc4 * bc4 < best_bc4) { best_bc4 = bc4 * bc4; fit->bc4_index[s] = (uint8_t)p; }
				}
			}
		}
	}}}
}

///////////////////////////////////////////////////////////////////////////////

void ktx2_etc1s_write_eac(const ktx2_gray_fit_t* fits, const ktx2_endpoint_t* endpoint,
                          const ktx2_selector_t* selector, uint8_t out_block[8]) {
	uint32_t               base5 = endpoint->color5[1]; // section 11: green carries the payload
	const ktx2_gray_fit_t* fit   = &fits[(base5 * 8 + endpoint->inten) * KTX2_BC1_RANGES
	                                   + ktx2_selector_range(selector)];

	// Big-endian: base, multiplier, table, then sixteen 3-bit indices from the
	// top, texel (x,y) at x*4+y as in ETC1.
	uint64_t block = ((uint64_t)_expand5((uint8_t)base5) << 56)
	               | ((uint64_t)fit->eac_multiplier      << 52)
	               | ((uint64_t)fit->eac_table           << 48);
	for (uint32_t y = 0; y < 4; y++) {
		uint32_t row = selector->rows[y];
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t index = fit->eac_index[(row >> (x * 2)) & 3];
			block |= (uint64_t)index << (45 - 3 * (x * 4 + y));
		}
	}
	for (int32_t i = 0; i < 8; i++) out_block[i] = (uint8_t)(block >> (56 - i * 8));
}

void ktx2_etc1s_write_bc4(const ktx2_gray_fit_t* fits, const ktx2_endpoint_t* endpoint,
                          const ktx2_selector_t* selector, uint8_t out_block[8]) {
	uint32_t               base5 = endpoint->color5[1];
	const ktx2_gray_fit_t* fit   = &fits[(base5 * 8 + endpoint->inten) * KTX2_BC1_RANGES
	                                   + ktx2_selector_range(selector)];

	out_block[0] = fit->bc4_hi; // red0 > red1 selects the 8-value palette
	out_block[1] = fit->bc4_lo;

	// Little-endian bit field, texel (x,y) at y*4+x, three bits each.
	uint64_t indices = 0;
	for (uint32_t y = 0; y < 4; y++) {
		uint32_t row = selector->rows[y];
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t index = fit->bc4_index[(row >> (x * 2)) & 3];
			indices |= (uint64_t)index << (3 * (y * 4 + x));
		}
	}
	for (int32_t i = 0; i < 6; i++) out_block[2 + i] = (uint8_t)(indices >> (i * 8));
}

void ktx2_etc1s_write_gray(const ktx2_endpoint_t* endpoint, const ktx2_selector_t* selector, uint8_t out_texels[16]) {
	int32_t levels[4];
	_gray_levels(endpoint->color5[1], endpoint->inten, levels);
	for (uint32_t y = 0; y < 4; y++) {
		uint32_t row = selector->rows[y];
		for (uint32_t x = 0; x < 4; x++)
			out_texels[y * 4 + x] = (uint8_t)levels[(row >> (x * 2)) & 3];
	}
}
