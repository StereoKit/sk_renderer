// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// L1e - ETC1S -> BC1, the only target with no shared block structure.
//
// ETC1S puts its four texel values on a line: one intensity modifier is added to
// every channel, so the levels are collinear along (1,1,1) and spaced
// non-uniformly. BC1 interpolates along a line too, but with four *uniformly*
// spaced values between RGB565 endpoints. That is a real fit, and the one place
// basisu resorts to tables from offline search.
//
// We compute an equivalent at load instead, which two properties make cheap:
//
//  - The modifier is common to all channels, so the fit decomposes per channel.
//    Only the selector-to-BC1-index assignment has to be shared.
//  - A channel's values are fixed by its 5-bit base and 3-bit intensity, so
//    there are 32*8 ramps per channel times ten contiguous selector spans. 2560
//    tiny least-squares problems, analytically seeded: about a millisecond.
//
// The span matters because the endpoints a block does not use are free to move.

#include "ktx2_internal.h"

#include <string.h>

// BC1 index of the p-th smallest interpolated value, colour0 being the high
// endpoint: 0 is colour0, 1 is colour1, 2 and 3 the thirds between.
static const uint8_t k_position_to_bc1[4] = { 1, 3, 2, 0 };

static inline int32_t _clamp(int32_t value, int32_t low, int32_t high) {
	return value < low ? low : (value > high ? high : value);
}

static inline int32_t _expand(int32_t value, int32_t bits) {
	return bits == 5 ? ((value << 3) | (value >> 2)) : ((value << 2) | (value >> 4));
}

// The four 8-bit values one channel takes, in selector order (ascending).
static void _channel_levels(int32_t base8, uint32_t inten, int32_t out_levels[4]) {
	for (int32_t s = 0; s < 4; s++)
		out_levels[s] = _clamp(base8 + ktx2_etc1_modifier[inten][ktx2_selector_to_etc1[s]], 0, 255);
}

// Squared error of a quantized endpoint pair against the targets.
static int32_t _ramp_error(int32_t lo, int32_t hi, int32_t bits,
                           const int32_t* levels, const uint8_t* positions, int32_t count) {
	int32_t lo8 = _expand(lo, bits);
	int32_t hi8 = _expand(hi, bits);
	int32_t error = 0;
	for (int32_t i = 0; i < count; i++) {
		int32_t p     = positions[i];
		int32_t value = ((3 - p) * lo8 + p * hi8) / 3;
		int32_t delta = value - levels[i];
		error += delta * delta;
	}
	return error;
}

// Least squares for the continuous endpoints, then a small integer search around
// them. The seed is what keeps this out of a 4096-pair brute force per entry.
static int32_t _fit_ramp(const int32_t* levels, const uint8_t* positions, int32_t count, int32_t bits,
                         uint8_t* out_lo, uint8_t* out_hi) {
	int32_t maximum = (1 << bits) - 1;

	double a = 0, b = 0, c = 0, d = 0, e = 0;
	for (int32_t i = 0; i < count; i++) {
		double t = positions[i] / 3.0;
		double u = 1.0 - t;
		a += u * u; b += u * t; c += t * t;
		d += u * levels[i]; e += t * levels[i];
	}
	double determinant = a * c - b * b;
	double lo_real, hi_real;
	if (determinant > -1e-9 && determinant < 1e-9) {
		double mean = 0;
		for (int32_t i = 0; i < count; i++) mean += levels[i];
		lo_real = hi_real = mean / count;
	} else {
		lo_real = (c * d - b * e) / determinant;
		hi_real = (a * e - b * d) / determinant;
	}

	int32_t seed_lo = _clamp((int32_t)((lo_real * maximum) / 255.0 + 0.5), 0, maximum);
	int32_t seed_hi = _clamp((int32_t)((hi_real * maximum) / 255.0 + 0.5), 0, maximum);

	int32_t best = 0x7FFFFFFF;
	for (int32_t lo = seed_lo - 3; lo <= seed_lo + 3; lo++) {
		if (lo < 0 || lo > maximum) continue;
		for (int32_t hi = seed_hi - 3; hi <= seed_hi + 3; hi++) {
			if (hi < 0 || hi > maximum || hi < lo) continue; // hi is colour0, and must not fall below lo
			int32_t error = _ramp_error(lo, hi, bits, levels, positions, count);
			if (error < best) { best = error; *out_lo = (uint8_t)lo; *out_hi = (uint8_t)hi; }
		}
	}
	return best;
}

///////////////////////////////////////////////////////////////////////////////

void ktx2_bc1_table_build(ktx2_bc1_table_t* out_table) {
	// The selector-to-position assignment is shared across channels, so it is
	// chosen once per (intensity, range) from the ascending subsets of {0,1,2,3}.
	//
	// Candidates are scored at 5-bit precision, the tighter width. Scoring an
	// unquantized ramp instead - a first attempt - costs ~9 dB: without
	// quantization a three-level block looks as happy at {0,1,2} as at {0,1,3},
	// when only the latter pins both extremes to an endpoint.
	for (uint32_t inten = 0; inten < 8; inten++) {
		int32_t levels_by_base[32][4];
		for (uint32_t base5 = 0; base5 < 32; base5++)
			_channel_levels(_expand((int32_t)base5, 5), inten, levels_by_base[base5]);

		for (uint32_t low = 0; low < 4; low++) {
			for (uint32_t high = low; high < 4; high++) {
				uint32_t slot  = ktx2_range_slot[low][high];
				int32_t  count = (int32_t)(high - low + 1);

				int32_t best       = 0x7FFFFFFF;
				uint8_t best_pos[4] = { 0, 1, 2, 3 };
				for (uint32_t mask = 0; mask < 16; mask++) {
					uint8_t positions[4];
					int32_t used = 0;
					for (uint32_t bit = 0; bit < 4; bit++)
						if (mask & (1u << bit)) { if (used < count) positions[used] = (uint8_t)bit; used++; }
					if (used != count) continue;

					// Every fourth base ranks assignments just as well: the
					// choice turns on ramp shape, which varies smoothly.
					int32_t error = 0;
					for (uint32_t base5 = 0; base5 < 32; base5 += 4) {
						int32_t targets[4];
						for (int32_t i = 0; i < count; i++) targets[i] = levels_by_base[base5][low + i];
						uint8_t lo, hi;
						error += _fit_ramp(targets, positions, count, 5, &lo, &hi);
					}
					if (error < best) { best = error; memcpy(best_pos, positions, sizeof(best_pos)); }
				}

				// Per selector, not per used-index, so the writer needs no arithmetic.
				// Out-of-range selectors cannot occur, so those entries are filler.
				for (uint32_t s = 0; s < 4; s++) {
					uint32_t clamped  = s < low ? low : (s > high ? high : s);
					uint32_t position = best_pos[clamped - low];
					out_table->index[inten][slot][s] = k_position_to_bc1[position];
				}

				for (uint32_t base5 = 0; base5 < 32; base5++) {
					int32_t targets5[4];
					for (int32_t i = 0; i < count; i++) targets5[i] = levels_by_base[base5][low + i];

					uint8_t positions[4];
					for (int32_t i = 0; i < count; i++) positions[i] = best_pos[i];

					_fit_ramp(targets5, positions, count, 5,
						&out_table->five[base5][inten][slot][0], &out_table->five[base5][inten][slot][1]);
					_fit_ramp(targets5, positions, count, 6,
						&out_table->six [base5][inten][slot][0], &out_table->six [base5][inten][slot][1]);
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

void ktx2_etc1s_write_bc1(const ktx2_bc1_table_t* table, const ktx2_endpoint_t* endpoint,
                          const ktx2_selector_t* selector, uint8_t out_block[8]) {
	// The levels this block reaches decide the endpoints, so find the span first.
	uint32_t slot  = ktx2_selector_range(selector);
	uint32_t inten = endpoint->inten;

	const uint8_t* red   = table->five[endpoint->color5[0]][inten][slot];
	const uint8_t* green = table->six [endpoint->color5[1]][inten][slot];
	const uint8_t* blue  = table->five[endpoint->color5[2]][inten][slot];

	uint32_t colour0 = ((uint32_t)red[1] << 11) | ((uint32_t)green[1] << 5) | blue[1];
	uint32_t colour1 = ((uint32_t)red[0] << 11) | ((uint32_t)green[0] << 5) | blue[0];

	out_block[0] = (uint8_t)(colour0 & 0xFF);
	out_block[1] = (uint8_t)(colour0 >> 8);
	out_block[2] = (uint8_t)(colour1 & 0xFF);
	out_block[3] = (uint8_t)(colour1 >> 8);

	// colour0 <= colour1 is BC1's three-colour mode, where index 3 is transparent
	// black. Every index must be 0 there, or a flat block turns into holes.
	bool flat = colour0 <= colour1;
	for (uint32_t y = 0; y < 4; y++) {
		uint32_t row  = selector->rows[y];
		uint32_t bits = 0;
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t index = flat ? 0 : table->index[inten][slot][(row >> (x * 2)) & 3];
			bits |= index << (x * 2);
		}
		out_block[4 + y] = (uint8_t)bits;
	}
}
