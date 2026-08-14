// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Crafted-file tests: KTX2 files that are structurally valid all the way down to
// the entropy coder, and hostile inside it.
//
// This exists because mutation fuzzing cannot get here. Reaching the ETC1S slice
// decoder at all needs four valid Huffman tables in the SGD, and every path
// worth attacking is behind a specific symbol followed by a specific bit
// pattern. Flipping bits in a real file destroys the tables long before it
// arrives, which is why a 60k-iteration ASan run finds nothing here.
//
// So the file is built rather than mutated. The writers below are the inverse of
// the decoder's readers - a bit writer, a flat canonical Huffman table, and a
// VLC - which is enough to say "emit exactly this symbol, then exactly these
// bits" and aim at one guard at a time.
//
// The pass condition is never a particular result code. It is that the library
// returns *something*, having neither crashed nor read out of bounds, so run
// this under ASan and UBSan like the rest.

#include "sk_ktx2.h"
#include "host_zstd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t g_failures = 0;

///////////////////////////////////////////////////////////////////////////////
// Bit writer, the inverse of ktx2_bits_get: LSB-first within each byte.

#define BW_MAX 4096

typedef struct bw_t {
	uint8_t  data[BW_MAX];
	uint32_t at_bit;
} bw_t;

static void bw_init(bw_t* out_bw) { memset(out_bw, 0, sizeof(*out_bw)); }

static void bw_put(bw_t* ref_bw, uint32_t value, uint32_t bits) {
	for (uint32_t i = 0; i < bits; i++, ref_bw->at_bit++) {
		if ((ref_bw->at_bit >> 3) >= BW_MAX) return;
		if (value & (1u << i)) ref_bw->data[ref_bw->at_bit >> 3] |= (uint8_t)(1u << (ref_bw->at_bit & 7));
	}
}

static uint32_t bw_bytes(const bw_t* bw) { return (bw->at_bit + 7) / 8; }

// ktx2_huff_decode consumes a code one bit at a time, shifting left, so the
// first bit written is the code's most significant.
static void bw_symbol(bw_t* ref_bw, uint32_t code, uint32_t code_bits) {
	for (uint32_t i = 0; i < code_bits; i++)
		bw_put(ref_bw, (code >> (code_bits - 1 - i)) & 1u, 1);
}

// ktx2_bits_vlc: chunk_bits of payload plus a continuation bit, repeated. It
// stops at 32 bits of shift, so `value` is reproduced exactly.
static void bw_vlc(bw_t* ref_bw, uint32_t value, uint32_t chunk_bits) {
	uint32_t shift = 0;
	for (;;) {
		uint32_t chunk = (value >> shift) & ((1u << chunk_bits) - 1u);
		shift += chunk_bits;
		bool more = shift < 32 && (value >> shift) != 0;
		bw_put(ref_bw, chunk | (more ? (1u << chunk_bits) : 0u), chunk_bits + 1);
		if (!more) return;
	}
}

// A canonical Huffman table where every symbol has the same code length, which
// makes symbol s the code s. The code length alphabet needs one entry, reachable
// by a single zero bit, so the per-symbol lengths are a run of zero bits.
static void bw_huff_flat(bw_t* ref_bw, uint32_t total_syms, uint32_t code_bits) {
	static const uint8_t k_order[21] = {
		17, 18, 19, 20, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 16
	};
	bw_put(ref_bw, total_syms, 14);
	bw_put(ref_bw, 21,          5);
	for (uint32_t i = 0; i < 21; i++) bw_put(ref_bw, k_order[i] == code_bits ? 1u : 0u, 3);
	for (uint32_t i = 0; i < total_syms; i++) bw_put(ref_bw, 0, 1);
}

///////////////////////////////////////////////////////////////////////////////
// Container assembly. Same layout as tests/synthetic_ktx2.h, but the SGD parts
// are supplied rather than fixed, which is the whole point.

#define CRAFT_ENDPOINTS 4
#define CRAFT_SELECTORS 4

// Code lengths chosen so each table is complete: total_syms <= 2^code_bits.
#define BITS_COLOR     5  // 32 delta symbols
#define BITS_INTEN     3  // 8
#define BITS_PRED      9  // 257, the endpoint prediction alphabet
#define BITS_SELECTOR  4  // selector_count + history_size + 1
#define BITS_RLE       6  // 64

typedef struct craft_t {
	uint8_t  data[8192];
	size_t   bytes;
} craft_t;

static void wr32(uint8_t* buffer, size_t at, uint32_t value) {
	buffer[at] = (uint8_t)value;         buffer[at + 1] = (uint8_t)(value >> 8);
	buffer[at + 2] = (uint8_t)(value >> 16); buffer[at + 3] = (uint8_t)(value >> 24);
}
static void wr64(uint8_t* buffer, size_t at, uint64_t value) {
	wr32(buffer, at, (uint32_t)value); wr32(buffer, at + 4, (uint32_t)(value >> 32));
}

// The endpoint codebook: four models, then a flat run of zero-delta symbols, so
// every endpoint is the same valid colour. None of the attacks are here.
static void craft_endpoints(bw_t* out_bw) {
	bw_init(out_bw);
	for (int32_t i = 0; i < 3; i++) bw_huff_flat(out_bw, 32, BITS_COLOR);
	bw_huff_flat(out_bw, 8, BITS_INTEN);
	bw_put(out_bw, 0, 1); // not grayscale
	for (int32_t i = 0; i < CRAFT_ENDPOINTS; i++) {
		bw_symbol(out_bw, 0, BITS_INTEN);
		for (int32_t c = 0; c < 3; c++) bw_symbol(out_bw, 0, BITS_COLOR);
	}
}

// Raw selectors, which skips the delta model entirely.
static void craft_selectors(bw_t* out_bw) {
	bw_init(out_bw);
	bw_put(out_bw, 0, 1); // not a global codebook
	bw_put(out_bw, 0, 1); // not hybrid
	bw_put(out_bw, 1, 1); // raw
	for (int32_t i = 0; i < CRAFT_SELECTORS; i++)
		for (int32_t row = 0; row < 4; row++) bw_put(out_bw, 0xE4, 8);
}

static void craft_tables(bw_t* out_bw, uint32_t delta_syms, uint32_t selector_syms, uint32_t history_size) {
	bw_init(out_bw);
	bw_huff_flat(out_bw, 257,           BITS_PRED);
	bw_huff_flat(out_bw, delta_syms,    BITS_COLOR);
	bw_huff_flat(out_bw, selector_syms, BITS_SELECTOR);
	bw_huff_flat(out_bw, 64,            BITS_RLE);
	bw_put(out_bw, history_size, 13);
}

// A minimal valid slice for any geometry, written from the decoder's grammar:
// one pred symbol per 2x2 group on even rows, 0xFF making every block
// delta-code, then per block a zero delta (endpoint 0) and raw selector 0, so
// no block depends on a neighbour and any block count decodes.
static void craft_slice(bw_t* out_slice, uint32_t blocks_x, uint32_t blocks_y) {
	bw_init(out_slice);
	for (uint32_t y = 0; y < blocks_y; y++) {
	for (uint32_t x = 0; x < blocks_x; x++) {
		if ((x & 1) == 0 && (y & 1) == 0) bw_symbol(out_slice, 0xFF, BITS_PRED);
		bw_symbol(out_slice, 0, BITS_COLOR);
		bw_symbol(out_slice, 0, BITS_SELECTOR);
	}}
}

// `slices` holds one slice per level; every image of a level points its desc at
// that whole slice, which the SGD bounds checks permit: identical bytes in,
// identical blocks out, per image. A longer level index moves the DFD and
// everything after it, which is why the offsets are computed rather than fixed.
static void craft_build_mips(craft_t* out_file, uint32_t width, uint32_t height,
                             uint32_t layer_count, uint32_t face_count, uint32_t level_count,
                             const bw_t* endpoints, const bw_t* selectors, const bw_t* tables,
                             const bw_t* slices) {
	static const uint8_t k_identifier[12] = {
		0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
	};
	uint32_t endpoint_bytes = bw_bytes(endpoints);
	uint32_t selector_bytes = bw_bytes(selectors);
	uint32_t table_bytes    = bw_bytes(tables);
	uint32_t per_level      = (layer_count ? layer_count : 1) * face_count;
	uint32_t images         = per_level * level_count;

	size_t dfd       = 80 + (size_t)level_count * 24;
	size_t sgd       = dfd + 44;
	size_t sgd_bytes = 20 + (size_t)images * 20 + endpoint_bytes + selector_bytes + table_bytes;
	size_t at        = (sgd + sgd_bytes + 7) & ~(size_t)7;

	memset(out_file, 0, sizeof(*out_file));
	memcpy(out_file->data, k_identifier, sizeof(k_identifier));
	wr32(out_file->data, 16, 1);      // typeSize
	wr32(out_file->data, 20, width);
	wr32(out_file->data, 24, height);
	wr32(out_file->data, 32, layer_count);
	wr32(out_file->data, 36, face_count);
	wr32(out_file->data, 40, level_count);
	wr32(out_file->data, 44, 1);      // supercompressionScheme = BasisLZ
	wr32(out_file->data, 48, (uint32_t)dfd);
	wr32(out_file->data, 52, 44);
	wr64(out_file->data, 64, sgd);
	wr64(out_file->data, 72, sgd_bytes);

	wr32(out_file->data, dfd,     44);
	wr32(out_file->data, dfd + 8, 40u << 16);
	out_file->data[dfd + 12] = 163; // ETC1S
	out_file->data[dfd + 13] = 1;   // BT709
	out_file->data[dfd + 14] = 2;   // sRGB
	out_file->data[dfd + 16] = 3;
	out_file->data[dfd + 17] = 3;
	out_file->data[dfd + 20] = 8;
	out_file->data[dfd + 30] = 63;
	out_file->data[dfd + 31] = 0;   // channel 0 = ETC1S RGB
	wr32(out_file->data, dfd + 40, 0xFFFFFFFF);

	wr32(out_file->data, sgd,      CRAFT_ENDPOINTS | (CRAFT_SELECTORS << 16));
	wr32(out_file->data, sgd +  4, endpoint_bytes);
	wr32(out_file->data, sgd +  8, selector_bytes);
	wr32(out_file->data, sgd + 12, table_bytes);
	wr32(out_file->data, sgd + 16, 0);
	for (uint32_t level = 0; level < level_count; level++) {
	for (uint32_t image = 0; image < per_level;   image++) {
		size_t desc = sgd + 20 + ((size_t)level * per_level + image) * 20;
		wr32(out_file->data, desc + 4, 0);                        // rgbSliceByteOffset
		wr32(out_file->data, desc + 8, bw_bytes(&slices[level])); // rgbSliceByteLength
	}}

	size_t blob = sgd + 20 + (size_t)images * 20;
	memcpy(out_file->data + blob, endpoints->data, endpoint_bytes); blob += endpoint_bytes;
	memcpy(out_file->data + blob, selectors->data, selector_bytes); blob += selector_bytes;
	memcpy(out_file->data + blob, tables->data,    table_bytes);

	for (uint32_t level = 0; level < level_count; level++) {
		uint32_t slice_bytes = bw_bytes(&slices[level]);
		wr64(out_file->data, 80 + (size_t)level * 24,      at);
		wr64(out_file->data, 80 + (size_t)level * 24 +  8, slice_bytes);
		wr64(out_file->data, 80 + (size_t)level * 24 + 16, slice_bytes);
		memcpy(out_file->data + at, slices[level].data, slice_bytes);
		at += slice_bytes;
	}
	out_file->bytes = at;
}

static void craft_build(craft_t* out_file, uint32_t width, uint32_t height,
                        const bw_t* endpoints, const bw_t* selectors, const bw_t* tables, const bw_t* slice) {
	craft_build_mips(out_file, width, height, 0, 1, 1, endpoints, selectors, tables, slice);
}

///////////////////////////////////////////////////////////////////////////////

// Transcodes to every target family. The result is not checked - only that the
// library returns rather than misbehaving - but it is printed, because a case
// that starts reporting `corrupt` when it used to succeed is worth noticing.
static void craft_run(const char* name, const craft_t* file) {
	static const ktx2_caps_ k_caps[] = { ktx2_caps_etc2, ktx2_caps_bc, ktx2_caps_none };

	ktx2_reader_t reader;
	ktx2_result_  result = ktx2_open(file->data, file->bytes, &reader);
	if (result != ktx2_result_success) {
		printf("  %-34s open: %s\n", name, ktx2_result_str(result));
		return;
	}

	printf("  %-34s", name);
	for (size_t i = 0; i < sizeof(k_caps) / sizeof(k_caps[0]); i++) {
		ktx2_plan_t plan;
		result = ktx2_plan(&reader, &k_host_context, k_caps[i], &plan);
		if (result != ktx2_result_success) { printf(" plan:%s", ktx2_result_str(result)); continue; }

		uint8_t* output  = (uint8_t*)malloc(plan.data_bytes    ? plan.data_bytes    : 1);
		uint8_t* scratch = (uint8_t*)malloc(plan.scratch_bytes ? plan.scratch_bytes : 1);
		result = ktx2_transcode(&plan, output, plan.data_bytes, scratch);
		printf(" %s:%s", ktx2_fmt_str(plan.format), ktx2_result_str(result));
		free(scratch);
		free(output);
	}
	printf("\n");
}

///////////////////////////////////////////////////////////////////////////////

// The endpoint prediction repeat count is a VLC with no ceiling, and it was
// being read into an int32_t and then had 2 added to it. A VLC of 0x7FFFFFFF is
// four bytes of slice data and overflows that, which is undefined rather than
// merely wrong. Nothing downstream misbehaves once it is unsigned - a repeat
// longer than the image just runs out of blocks.
static void case_pred_repeat_overflow(void) {
	bw_t endpoints, selectors, tables, slice;
	craft_endpoints(&endpoints);
	craft_selectors(&selectors);
	craft_tables(&tables, CRAFT_ENDPOINTS, CRAFT_SELECTORS + 8 + 1, 8);

	bw_init(&slice);
	bw_symbol(&slice, 256, BITS_PRED); // the repeat escape
	bw_vlc   (&slice, 0x7FFFFFFF, 4);  // int32_t's ceiling, minus one

	craft_t file;
	craft_build(&file, 4, 4, &endpoints, &selectors, &tables, &slice);
	craft_run("endpoint pred repeat overflow", &file);
}

// The same field at the other end of its range, where the unsigned addition
// wraps to a small number rather than a large one.
static void case_pred_repeat_wrap(void) {
	bw_t endpoints, selectors, tables, slice;
	craft_endpoints(&endpoints);
	craft_selectors(&selectors);
	craft_tables(&tables, CRAFT_ENDPOINTS, CRAFT_SELECTORS + 8 + 1, 8);

	bw_init(&slice);
	bw_symbol(&slice, 256, BITS_PRED);
	bw_vlc   (&slice, 0xFFFFFFFF, 4);

	craft_t file;
	craft_build(&file, 4, 4, &endpoints, &selectors, &tables, &slice);
	craft_run("endpoint pred repeat wrap", &file);
}

// A selector run-length far longer than the image has blocks. The run is only
// ever decremented once per block, so the block loop bounds it - but only
// because nothing else indexes with it. This one is expected to *succeed*: the
// image decodes, which is what proves the run was survived rather than escaped.
static void case_selector_rle_huge(void) {
	bw_t endpoints, selectors, tables, slice;
	craft_endpoints(&endpoints);
	craft_selectors(&selectors);
	craft_tables(&tables, CRAFT_ENDPOINTS, CRAFT_SELECTORS + 8 + 1, 8);

	bw_init(&slice);
	bw_symbol(&slice, 0xFF, BITS_PRED);            // every block predicts by delta
	bw_symbol(&slice, 0,    BITS_COLOR);           // endpoint delta for block 0
	bw_symbol(&slice, CRAFT_SELECTORS + 8, BITS_SELECTOR); // the RLE escape
	bw_symbol(&slice, 63,   BITS_RLE);             // escape to a VLC run length
	bw_vlc   (&slice, 0x10000, 7);                 // 65536 blocks, against an image of 4
	for (int32_t i = 0; i < 3; i++) bw_symbol(&slice, 0, BITS_COLOR); // blocks 1..3

	craft_t file;
	craft_build(&file, 8, 8, &endpoints, &selectors, &tables, &slice);
	craft_run("selector rle longer than image", &file);
}

// A delta_endpoint alphabet wider than the endpoint codebook, so a legal symbol
// names an endpoint that does not exist. The modular fixup does not bring it
// back into range, so only the explicit bounds check catches it.
static void case_endpoint_index_out_of_range(void) {
	bw_t endpoints, selectors, tables, slice;
	craft_endpoints(&endpoints);
	craft_selectors(&selectors);
	craft_tables(&tables, 32, CRAFT_SELECTORS + 8 + 1, 8); // 32 deltas, 4 endpoints

	bw_init(&slice);
	bw_symbol(&slice, 3,  BITS_PRED);   // pred 3, delta coded
	bw_symbol(&slice, 31, BITS_COLOR);  // delta 31 against a 4-entry codebook

	craft_t file;
	craft_build(&file, 4, 4, &endpoints, &selectors, &tables, &slice);
	craft_run("endpoint index past the codebook", &file);
}

// A selector alphabet that names more history slots than the header declares.
// The history buffer is a fixed 64 entries, so an unchecked index would read
// inside the struct rather than off the end of it - the quiet kind.
static void case_history_index_past_size(void) {
	bw_t endpoints, selectors, tables, slice;
	craft_endpoints(&endpoints);
	craft_selectors(&selectors);
	craft_tables(&tables, CRAFT_ENDPOINTS, CRAFT_SELECTORS + 8 + 1, 2); // 8 slots named, 2 declared

	bw_init(&slice);
	bw_symbol(&slice, 3,  BITS_PRED);
	bw_symbol(&slice, 0,  BITS_COLOR);
	bw_symbol(&slice, CRAFT_SELECTORS + 7, BITS_SELECTOR); // history slot 7 of a 2-slot history

	craft_t file;
	craft_build(&file, 4, 4, &endpoints, &selectors, &tables, &slice);
	craft_run("history index past history size", &file);
}

// A slice that simply stops. Every read past the end returns zero and latches
// overrun, so this has to end as corruption rather than as a plausible image
// decoded from nothing.
static void case_truncated_slice(void) {
	bw_t endpoints, selectors, tables, slice;
	craft_endpoints(&endpoints);
	craft_selectors(&selectors);
	craft_tables(&tables, CRAFT_ENDPOINTS, CRAFT_SELECTORS + 8 + 1, 8);

	bw_init(&slice);
	bw_symbol(&slice, 3, BITS_PRED); // and nothing after it

	craft_t file;
	craft_build(&file, 64, 64, &endpoints, &selectors, &tables, &slice);
	craft_run("slice truncated mid-image", &file);
}

// A well-formed file whose header claims the largest texture the reader accepts,
// against a few hundred bytes of slice data. The decode fails, but not before
// the caller has been told how much to allocate - so the sizes it reports have
// to stay finite and consistent. This is the amplification case, and the answer
// is that ktx2_plan reports it rather than acting on it.
static void case_dimension_amplification(void) {
	bw_t endpoints, selectors, tables, slice;
	craft_endpoints(&endpoints);
	craft_selectors(&selectors);
	craft_tables(&tables, CRAFT_ENDPOINTS, CRAFT_SELECTORS + 8 + 1, 8);

	bw_init(&slice);
	bw_symbol(&slice, 3, BITS_PRED);

	craft_t file;
	craft_build(&file, 16384, 16384, &endpoints, &selectors, &tables, &slice);

	ktx2_reader_t reader;
	if (ktx2_open(file.data, file.bytes, &reader) != ktx2_result_success) {
		printf("  %-34s open declined it\n", "16384x16384 from 500 bytes");
		return;
	}
	ktx2_plan_t plan;
	if (ktx2_plan(&reader, &k_host_context, ktx2_caps_etc2, &plan) != ktx2_result_success) {
		printf("  %-34s plan declined it\n", "16384x16384 from 500 bytes");
		return;
	}
	printf("  %-34s data %zu MB, scratch %zu MB from a %zu byte file\n", "16384x16384 from 500 bytes",
		plan.data_bytes / (1024 * 1024), plan.scratch_bytes / (1024 * 1024), file.bytes);

	// The figures have to be the honest size of that texture, not a wrapped one.
	if (plan.data_bytes != (size_t)4096 * 4096 * 8) {
		printf("  FAIL  data_bytes is %zu, expected %zu\n", plan.data_bytes, (size_t)4096 * 4096 * 8);
		g_failures++;
	}
}

// Open, plan against ETC2 caps, and transcode, expecting exactly `out_bytes`
// of output. Returns false on any refusal, so callers report one failure.
static bool craft_transcode(const craft_t* file, uint8_t* out_data, size_t out_bytes) {
	ktx2_reader_t reader;
	ktx2_plan_t   plan;
	if (ktx2_open(file->data, file->bytes, &reader)                    != ktx2_result_success) return false;
	if (ktx2_plan(&reader, &k_host_context, ktx2_caps_etc2, &plan)     != ktx2_result_success) return false;
	if (plan.data_bytes != out_bytes)                                                          return false;
	return ktx2_transcode(&plan, out_data, out_bytes, NULL) == ktx2_result_success;
}

// A mipped cubemap, each level's six imageDescs sharing that level's slice. The
// write cursor is proven by reference rather than inspection: every face sliced
// out of the cubemap output must equal the same slices transcoded as a plain 2D
// mip chain. This is the stride case a single-level or single-image file cannot
// reach: an indexing bug that swaps or overlaps images has to fail it.
static void case_cubemap_mips(void) {
	bw_t endpoints, selectors, tables;
	craft_endpoints(&endpoints);
	craft_selectors(&selectors);
	craft_tables(&tables, CRAFT_ENDPOINTS, CRAFT_SELECTORS + 8 + 1, 8);

	bw_t slices[2];
	craft_slice(&slices[0], 2, 2); // 8x8 mip 0
	craft_slice(&slices[1], 1, 1); // 4x4 mip 1

	craft_t cube, flat;
	craft_build_mips(&cube, 8, 8, 0, 6, 2, &endpoints, &selectors, &tables, slices);
	craft_build_mips(&flat, 8, 8, 0, 1, 2, &endpoints, &selectors, &tables, slices);

	const char* name = "mipped cubemap vs 2D reference";
	uint8_t cube_out[6 * 32 + 6 * 8]; // etc1: mip 0 is four blocks per face, mip 1 one
	uint8_t flat_out[32 + 8];
	if (!craft_transcode(&cube, cube_out, sizeof(cube_out)) ||
	    !craft_transcode(&flat, flat_out, sizeof(flat_out))) {
		printf("  FAIL  %s did not transcode\n", name); g_failures++; return;
	}

	for (int32_t face = 0; face < 6; face++) {
		if (memcmp(cube_out + face * 32,         flat_out,      32) != 0 ||
		    memcmp(cube_out + 6 * 32 + face * 8, flat_out + 32,  8) != 0) {
			printf("  FAIL  %s: face %d differs\n", name, face);
			g_failures++; return;
		}
	}
	printf("  %-34s six faces match the 2D reference\n", name);
}

int main(void) {
	printf("Crafted ETC1S bitstreams:\n");
	case_pred_repeat_overflow();
	case_pred_repeat_wrap();
	case_selector_rle_huge();
	case_endpoint_index_out_of_range();
	case_history_index_past_size();
	case_truncated_slice();
	case_dimension_amplification();
	case_cubemap_mips();

	if (g_failures == 0) printf("craft: all cases returned cleanly\n");
	else                 printf("%d failure(s)\n", g_failures);
	return g_failures == 0 ? 0 : 1;
}
