// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// L1a - BasisLZ / ETC1S entropy decode, from the ".basis File Format and ETC1S
// Texture Video Specification" 1.03, placed in the public domain by its author.
//
// Two global codebooks, endpoints and selectors, shared by every image, plus the
// per-block indices into them. The indices are the compressed part: spatial
// prediction over 2x2 groups for endpoints, approximate move-to-front with
// run-length coding for selectors.

#include "ktx2_internal.h"

#include <string.h>

// One symbol carries 2 bits per block of a 2x2 group, plus a "repeat" symbol.
#define KTX2_ENDPOINT_PRED_SYMS         (4 * 4 * 4 * 4 + 1)
#define KTX2_ENDPOINT_PRED_REPEAT       (KTX2_ENDPOINT_PRED_SYMS - 1)
#define KTX2_ENDPOINT_PRED_MIN_REPEAT   3
#define KTX2_ENDPOINT_PRED_VLC_BITS     4

#define KTX2_SELECTOR_RLE_THRESH        3
#define KTX2_SELECTOR_RLE_BITS          6
#define KTX2_SELECTOR_RLE_TOTAL         (1 << KTX2_SELECTOR_RLE_BITS)

// Endpoint colour DPCM: the previous component's magnitude picks the delta model.
#define KTX2_COLOR5_MODEL0_PREV_HI  9
#define KTX2_COLOR5_MODEL1_PREV_HI 21

///////////////////////////////////////////////////////////////////////////////

static ktx2_result_ _read_endpoints(const ktx2_reader_t* reader, ktx2_arena_t* ref_arena, ktx2_etc1s_t* ref_etc1s) {
	ktx2_bits_t bits;
	ktx2_bits_init(&bits, reader->sgd_endpoints, reader->sgd_endpoints_bytes);

	size_t      mark = ref_arena->at;
	ktx2_huff_t color_model[3];
	ktx2_huff_t inten_model;
	for (int32_t i = 0; i < 3; i++)
		if (!ktx2_huff_read(&bits, ref_arena, &color_model[i])) return ktx2_result_corrupt;
	if (!ktx2_huff_read(&bits, ref_arena, &inten_model)) return ktx2_result_corrupt;

	bool grayscale = ktx2_bits_get(&bits, 1) != 0;

	uint8_t previous_color[3] = { 16, 16, 16 };
	uint8_t previous_inten    = 0;
	for (uint32_t i = 0; i < ref_etc1s->endpoint_count; i++) {
		int32_t inten_delta = ktx2_huff_decode(&bits, &inten_model);
		if (inten_delta < 0) return ktx2_result_corrupt;
		previous_inten = (uint8_t)((inten_delta + previous_inten) & 7);
		ref_etc1s->endpoints[i].inten = previous_inten;

		uint32_t components = grayscale ? 1 : 3;
		for (uint32_t c = 0; c < components; c++) {
			const ktx2_huff_t* model = previous_color[c] <= KTX2_COLOR5_MODEL0_PREV_HI ? &color_model[0]
			                         : previous_color[c] <= KTX2_COLOR5_MODEL1_PREV_HI ? &color_model[1]
			                         :                                                   &color_model[2];
			int32_t delta = ktx2_huff_decode(&bits, model);
			if (delta < 0) return ktx2_result_corrupt;
			previous_color[c] = (uint8_t)((previous_color[c] + delta) & 31);
			ref_etc1s->endpoints[i].color5[c] = previous_color[c];
		}
		if (grayscale) {
			ref_etc1s->endpoints[i].color5[1] = ref_etc1s->endpoints[i].color5[0];
			ref_etc1s->endpoints[i].color5[2] = ref_etc1s->endpoints[i].color5[0];
			previous_color[1] = previous_color[0];
			previous_color[2] = previous_color[0];
		}
		if (bits.overrun) return ktx2_result_corrupt;
	}

	ref_arena->at = mark; // the four models are done with
	return ktx2_result_success;
}

static ktx2_result_ _read_selectors(const ktx2_reader_t* reader, ktx2_arena_t* ref_arena, ktx2_etc1s_t* ref_etc1s) {
	ktx2_bits_t bits;
	ktx2_bits_init(&bits, reader->sgd_selectors, reader->sgd_selectors_bytes);

	// Global and hybrid codebooks were specified but no encoder emits them.
	if (ktx2_bits_get(&bits, 1) != 0) return ktx2_result_unsupported;
	if (ktx2_bits_get(&bits, 1) != 0) return ktx2_result_unsupported;
	bool raw = ktx2_bits_get(&bits, 1) != 0;

	size_t      mark = ref_arena->at;
	ktx2_huff_t delta_model;
	if (!raw && !ktx2_huff_read(&bits, ref_arena, &delta_model)) return ktx2_result_corrupt;

	uint8_t previous[4] = { 0, 0, 0, 0 };
	for (uint32_t i = 0; i < ref_etc1s->selector_count; i++) {
		for (int32_t row = 0; row < 4; row++) {
			uint8_t value;
			if (raw || i == 0) {
				value = (uint8_t)ktx2_bits_get(&bits, 8);
			} else {
				int32_t delta = ktx2_huff_decode(&bits, &delta_model);
				if (delta < 0) return ktx2_result_corrupt;
				value = (uint8_t)(delta ^ previous[row]); // byte-wise XOR DPCM
			}
			previous[row] = value;
			ref_etc1s->selectors[i].rows[row] = value;
		}
		if (bits.overrun) return ktx2_result_corrupt;
	}

	ref_arena->at = mark; // the delta model is done with, if there was one
	return ktx2_result_success;
}

static ktx2_result_ _read_slice_tables(const ktx2_reader_t* reader, ktx2_arena_t* ref_arena, ktx2_etc1s_t* ref_etc1s) {
	ktx2_bits_t bits;
	ktx2_bits_init(&bits, reader->sgd_tables, reader->sgd_tables_bytes);

	if (!ktx2_huff_read(&bits, ref_arena, &ref_etc1s->endpoint_pred  )) return ktx2_result_corrupt;
	if (!ktx2_huff_read(&bits, ref_arena, &ref_etc1s->delta_endpoint )) return ktx2_result_corrupt;
	if (!ktx2_huff_read(&bits, ref_arena, &ref_etc1s->selector       )) return ktx2_result_corrupt;
	if (!ktx2_huff_read(&bits, ref_arena, &ref_etc1s->selector_rle   )) return ktx2_result_corrupt;

	ref_etc1s->history_size = ktx2_bits_get(&bits, 13);
	if (ref_etc1s->history_size > KTX2_MAX_SELECTOR_HISTORY) return ktx2_result_unsupported;
	if (bits.overrun)                                        return ktx2_result_corrupt;
	return ktx2_result_success;
}

ktx2_result_ ktx2_etc1s_read_codebooks(const ktx2_reader_t* reader, ktx2_arena_t* ref_arena, ktx2_etc1s_t* out_etc1s) {
	memset(out_etc1s, 0, sizeof(*out_etc1s));
	out_etc1s->endpoint_count = reader->endpoint_count;
	out_etc1s->selector_count = reader->selector_count;
	if (out_etc1s->endpoint_count == 0 || out_etc1s->selector_count == 0) return ktx2_result_corrupt;

	out_etc1s->endpoints = (ktx2_endpoint_t*)ktx2_arena_alloc(ref_arena, out_etc1s->endpoint_count * sizeof(ktx2_endpoint_t));
	out_etc1s->selectors = (ktx2_selector_t*)ktx2_arena_alloc(ref_arena, out_etc1s->selector_count * sizeof(ktx2_selector_t));
	if (ref_arena->exhausted) return ktx2_result_buffer_too_small;

	ktx2_result_ result = _read_endpoints(reader, ref_arena, out_etc1s);
	if (result != ktx2_result_success) return result;
	result = _read_selectors(reader, ref_arena, out_etc1s);
	if (result != ktx2_result_success) return result;
	result = _read_slice_tables(reader, ref_arena, out_etc1s);
	if (result != ktx2_result_success) return result;

	return ref_arena->exhausted ? ktx2_result_buffer_too_small : ktx2_result_success;
}

///////////////////////////////////////////////////////////////////////////////
// Approximate move-to-front: `use` swaps an entry halfway toward the front
// rather than moving it, and `add` overwrites at a rover cycling the back half.

typedef struct ktx2_mtf_t {
	int32_t  values[KTX2_MAX_SELECTOR_HISTORY];
	uint32_t size;
	uint32_t rover;
} ktx2_mtf_t;

static void _mtf_init(ktx2_mtf_t* out_mtf, uint32_t size) {
	memset(out_mtf->values, 0, sizeof(out_mtf->values));
	out_mtf->size  = size;
	out_mtf->rover = size / 2;
}

static void _mtf_add(ktx2_mtf_t* ref_mtf, int32_t value) {
	if (ref_mtf->size == 0) return;
	ref_mtf->values[ref_mtf->rover++] = value;
	if (ref_mtf->rover == ref_mtf->size) ref_mtf->rover = ref_mtf->size / 2;
}

static void _mtf_use(ktx2_mtf_t* ref_mtf, uint32_t index) {
	if (index == 0) return;
	int32_t swap                   = ref_mtf->values[index / 2];
	ref_mtf->values[index / 2]     = ref_mtf->values[index];
	ref_mtf->values[index]         = swap;
}

///////////////////////////////////////////////////////////////////////////////

ktx2_result_ ktx2_etc1s_decode_slice(ktx2_etc1s_t* ref_etc1s, const void* data, size_t bytes,
                                     uint32_t blocks_x, uint32_t blocks_y, uint32_t* out_blocks) {
	ktx2_bits_t bits;
	ktx2_bits_init(&bits, data, bytes);

	ktx2_mtf_t history;
	_mtf_init(&history, ref_etc1s->history_size);

	// Symbols past the codebook name history slots, and the one past those is
	// the run-length escape.
	uint32_t history_rle_symbol = ref_etc1s->selector_count + ref_etc1s->history_size;

	// Both run counts come from a VLC with no ceiling, so they are unsigned: the
	// file can name a run of four billion, and only the block loop bounds it.
	// Signed would be undefined on the addition below, not merely wrong.
	uint32_t selector_rle_count      = 0;
	uint32_t pred_repeat_count       = 0;
	uint32_t pred_bits               = 0;
	int32_t  previous_pred_symbol    = 0;
	uint32_t previous_endpoint_index = 0;

	memset(ref_etc1s->preds, 0, sizeof(ktx2_block_pred_t) * 2 * blocks_x);

	for (uint32_t block_y = 0; block_y < blocks_y; block_y++) {
		ktx2_block_pred_t* row      = ref_etc1s->preds + (size_t)(block_y & 1) * blocks_x;
		ktx2_block_pred_t* row_prev = ref_etc1s->preds + (size_t)((block_y & 1) ^ 1) * blocks_x;

		for (uint32_t block_x = 0; block_x < blocks_x; block_x++) {
			// Prediction arrives once per 2x2 group on even rows; odd rows
			// replay what the even row above stashed.
			if ((block_x & 1) == 0) {
				if ((block_y & 1) == 0) {
					if (pred_repeat_count != 0) {
						pred_repeat_count--;
						pred_bits = (uint32_t)previous_pred_symbol;
					} else {
						int32_t symbol = ktx2_huff_decode(&bits, &ref_etc1s->endpoint_pred);
						if (symbol < 0 || symbol >= KTX2_ENDPOINT_PRED_SYMS) return ktx2_result_corrupt;
						if (symbol == KTX2_ENDPOINT_PRED_REPEAT) {
							pred_repeat_count = ktx2_bits_vlc(&bits, KTX2_ENDPOINT_PRED_VLC_BITS)
							                  + KTX2_ENDPOINT_PRED_MIN_REPEAT - 1;
							pred_bits         = (uint32_t)previous_pred_symbol;
						} else {
							pred_bits            = (uint32_t)symbol;
							previous_pred_symbol = symbol;
						}
					}
					// The symbol's top half belongs to the two blocks beneath.
					row_prev[block_x].pred_bits = (uint8_t)(pred_bits >> 4);
				} else {
					pred_bits = row[block_x].pred_bits;
				}
			}

			uint32_t pred = pred_bits & 3;
			pred_bits >>= 2;

			uint32_t endpoint_index = 0;
			if (pred == 0) {
				if (block_x == 0) return ktx2_result_corrupt;
				endpoint_index = previous_endpoint_index;
			} else if (pred == 1) {
				if (block_y == 0) return ktx2_result_corrupt;
				endpoint_index = row_prev[block_x].endpoint_index;
			} else if (pred == 2) {
				// Texture video reuses the previous frame here; a 2D KTX2 has
				// none, so this is the upper-left block.
				if (block_x == 0 || block_y == 0) return ktx2_result_corrupt;
				endpoint_index = row_prev[block_x - 1].endpoint_index;
			} else {
				int32_t delta = ktx2_huff_decode(&bits, &ref_etc1s->delta_endpoint);
				if (delta < 0) return ktx2_result_corrupt;
				endpoint_index = (uint32_t)delta + previous_endpoint_index;
				if (endpoint_index >= ref_etc1s->endpoint_count)
					endpoint_index -= ref_etc1s->endpoint_count;
			}
			row[block_x].endpoint_index = (uint16_t)endpoint_index;
			previous_endpoint_index     = endpoint_index;

			uint32_t selector_index = 0;
			int32_t  selector_symbol;
			if (selector_rle_count > 0) {
				selector_rle_count--;
				selector_symbol = (int32_t)ref_etc1s->selector_count;
			} else {
				selector_symbol = ktx2_huff_decode(&bits, &ref_etc1s->selector);
				if (selector_symbol < 0) return ktx2_result_corrupt;

				if ((uint32_t)selector_symbol == history_rle_symbol) {
					int32_t run = ktx2_huff_decode(&bits, &ref_etc1s->selector_rle);
					if (run < 0) return ktx2_result_corrupt;
					selector_rle_count = run == KTX2_SELECTOR_RLE_TOTAL - 1
						? ktx2_bits_vlc(&bits, 7) + KTX2_SELECTOR_RLE_THRESH
						: (uint32_t)run          + KTX2_SELECTOR_RLE_THRESH;
					selector_symbol = (int32_t)ref_etc1s->selector_count;
					if (selector_rle_count == 0) return ktx2_result_corrupt;
					selector_rle_count--;
				}
			}

			if ((uint32_t)selector_symbol >= ref_etc1s->selector_count) {
				uint32_t history_index = (uint32_t)selector_symbol - ref_etc1s->selector_count;
				if (history_index >= history.size) return ktx2_result_corrupt;
				selector_index = (uint32_t)history.values[history_index];
				_mtf_use(&history, history_index);
			} else {
				selector_index = (uint32_t)selector_symbol;
				_mtf_add(&history, (int32_t)selector_index);
			}

			if (endpoint_index >= ref_etc1s->endpoint_count) return ktx2_result_corrupt;
			if (selector_index >= ref_etc1s->selector_count) return ktx2_result_corrupt;

			out_blocks[(size_t)block_y * blocks_x + block_x] = endpoint_index | (selector_index << 16);
		}
		if (bits.overrun) return ktx2_result_corrupt;
	}
	return ktx2_result_success;
}
