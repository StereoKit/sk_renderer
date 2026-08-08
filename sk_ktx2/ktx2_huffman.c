// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Canonical Huffman, plus the arena and VLC helpers the ETC1S bitstream leans
// on. Deflate-compatible, so RFC 1951 3.2.2 turns code lengths into codes.

#include "ktx2_internal.h"

#include <string.h>

void* ktx2_arena_alloc(ktx2_arena_t* ref_arena, size_t bytes) {
	size_t at = (ref_arena->at + 7) & ~(size_t)7;
	if (bytes > ref_arena->bytes || at > ref_arena->bytes - bytes) {
		ref_arena->exhausted = true;
		return NULL;
	}
	ref_arena->at = at + bytes;
	return ref_arena->data + at;
}

uint32_t ktx2_bits_vlc(ktx2_bits_t* ref_bits, int32_t chunk_bits) {
	uint32_t chunk_size = 1u << chunk_bits;
	uint32_t value      = 0;
	int32_t  shift      = 0;
	for (;;) {
		uint32_t chunk = ktx2_bits_get(ref_bits, chunk_bits + 1);
		value |= (chunk & (chunk_size - 1)) << shift;
		shift += chunk_bits;
		if ((chunk & chunk_size) == 0) break;
		if (shift >= 32)               break; // malformed; overrun catches it
		if (ref_bits->overrun)         break;
	}
	return value;
}

///////////////////////////////////////////////////////////////////////////////

static bool _huff_build(ktx2_huff_t* out_huff, const uint8_t* lengths, uint32_t count, uint16_t* symbols) {
	memset(out_huff->counts, 0, sizeof(out_huff->counts));
	for (uint32_t i = 0; i < count; i++) {
		if (lengths[i] > KTX2_HUFF_MAX_CODE_BITS) return false;
		out_huff->counts[lengths[i]]++;
	}
	out_huff->counts[0]     = 0; // unused symbols are not codes
	out_huff->symbols       = symbols;
	out_huff->symbol_count  = count;

	// Over-subscribed tables are rejected. Incomplete ones are legal - the spec
	// names the single-symbol case - and decode reports their gaps as -1.
	int32_t left = 1;
	for (int32_t len = 1; len <= KTX2_HUFF_MAX_CODE_BITS; len++) {
		left <<= 1;
		left  -= out_huff->counts[len];
		if (left < 0) return false;
	}

	uint16_t offsets[KTX2_HUFF_MAX_CODE_BITS + 2];
	offsets[1] = 0;
	for (int32_t len = 1; len <= KTX2_HUFF_MAX_CODE_BITS; len++)
		offsets[len + 1] = (uint16_t)(offsets[len] + out_huff->counts[len]);
	for (uint32_t i = 0; i < count; i++)
		if (lengths[i] != 0) symbols[offsets[lengths[i]]++] = (uint16_t)i;

	return true;
}

int32_t ktx2_huff_decode(ktx2_bits_t* ref_bits, const ktx2_huff_t* huff) {
	int32_t code = 0, first = 0, index = 0;
	for (int32_t len = 1; len <= KTX2_HUFF_MAX_CODE_BITS; len++) {
		code |= (int32_t)ktx2_bits_get(ref_bits, 1);
		int32_t count = huff->counts[len];
		if (code - first < count) return huff->symbols[index + (code - first)];
		index += count;
		first  = (first + count) << 1;
		code <<= 1;
	}
	return -1;
}

///////////////////////////////////////////////////////////////////////////////

// Ordered so the lengths most likely to be zero cluster at the end and can be
// omitted.
static const uint8_t k_codelength_order[21] = {
	17, 18, 19, 20, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 16
};

#define KTX2_SMALL_ZERO_RUN 17 // 3 extra bits, run of 3..10 zero lengths
#define KTX2_BIG_ZERO_RUN   18 // 7 extra bits, run of 11..138
#define KTX2_SMALL_REPEAT   19 // 2 extra bits, repeat previous 3..6 times
#define KTX2_BIG_REPEAT     20 // 7 extra bits, repeat previous 7..134 times

bool ktx2_huff_read(ktx2_bits_t* ref_bits, ktx2_arena_t* ref_arena, ktx2_huff_t* out_huff) {
	uint32_t total_syms       = ktx2_bits_get(ref_bits, 14);
	uint32_t codelength_codes = ktx2_bits_get(ref_bits, 5);
	if (total_syms == 0 || total_syms > KTX2_HUFF_MAX_SYMS) return false;
	if (codelength_codes == 0 || codelength_codes > 21)     return false;

	uint8_t codelength_lengths[21];
	memset(codelength_lengths, 0, sizeof(codelength_lengths));
	for (uint32_t i = 0; i < codelength_codes; i++)
		codelength_lengths[k_codelength_order[i]] = (uint8_t)ktx2_bits_get(ref_bits, 3);

	uint16_t   codelength_symbols[21];
	ktx2_huff_t codelength_huff;
	if (!_huff_build(&codelength_huff, codelength_lengths, 21, codelength_symbols)) return false;

	uint8_t* lengths = (uint8_t*)ktx2_arena_alloc(ref_arena, total_syms);
	if (lengths == NULL) return false;

	uint32_t at       = 0;
	uint8_t  previous = 0;
	while (at < total_syms) {
		int32_t symbol = ktx2_huff_decode(ref_bits, &codelength_huff);
		if (symbol < 0 || ref_bits->overrun) return false;

		if (symbol <= 16) {
			previous     = (uint8_t)symbol;
			lengths[at++] = previous;
			continue;
		}

		uint32_t run     = 0;
		bool     repeats = symbol == KTX2_SMALL_REPEAT || symbol == KTX2_BIG_REPEAT;
		switch (symbol) {
		case KTX2_SMALL_ZERO_RUN: run = ktx2_bits_get(ref_bits, 3) + 3;  break;
		case KTX2_BIG_ZERO_RUN:   run = ktx2_bits_get(ref_bits, 7) + 11; break;
		case KTX2_SMALL_REPEAT:   run = ktx2_bits_get(ref_bits, 2) + 3;  break;
		case KTX2_BIG_REPEAT:     run = ktx2_bits_get(ref_bits, 7) + 7;  break;
		default: return false;
		}
		// A repeat cannot lead, and cannot follow a zero length.
		if (repeats && (at == 0 || previous == 0)) return false;
		if (run > total_syms - at)                 return false;

		memset(lengths + at, repeats ? previous : 0, run);
		at += run;
	}

	uint16_t* symbols = (uint16_t*)ktx2_arena_alloc(ref_arena, total_syms * sizeof(uint16_t));
	if (symbols == NULL) return false;
	if (ref_bits->overrun) return false;

	return _huff_build(out_huff, lengths, total_syms, symbols);
}
