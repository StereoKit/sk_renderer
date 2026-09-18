// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Internals shared between the container, the ETC1S decoder, and the repacks.
// Not installed - `sk_ktx2.h` is the public surface.

#pragma once

#include "sk_ktx2.h"

// Both stated by the ETC1S bitstream specification.
#define KTX2_HUFF_MAX_CODE_BITS 16
#define KTX2_HUFF_MAX_SYMS      (1 << 14)
#define KTX2_HUFF_TABLES_LIVE   4  // most tables alive at once, sizes the arena

// KHR_DF_TRANSFER_*.
#define KTX2_TRANSFER_LINEAR 1
#define KTX2_TRANSFER_SRGB   2

// One BasisLZ imageDesc: flags, then the two slice offset/length pairs.
#define KTX2_IMAGE_DESC_BYTES 20

// Always 16 bytes per 4x4 block, so a level's inflated size follows from the
// geometry rather than from the file.
#define KTX2_UASTC_BLOCK_BYTES 16

///////////////////////////////////////////////////////////////////////////////

static inline uint32_t ktx2_rd_u32(const uint8_t* p) {
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint32_t ktx2_mip_size(uint32_t base, uint32_t level) {
	uint32_t size = base >> level;
	return size ? size : 1;
}

static inline uint32_t ktx2_blocks(uint32_t pixels) {
	return (pixels + 3) / 4;
}

// Index of one image in a file's level, then layer, then face order: the order
// KTX2 stores them within a level, and the order the output is packed in.
static inline uint32_t ktx2_image_index(const ktx2_reader_t* reader, uint32_t level, uint32_t layer, uint32_t face) {
	uint32_t layers = reader->layer_count ? reader->layer_count : 1;
	return (level * layers + layer) * reader->face_count + face;
}

// Shared so the plan's size and the transcode's write cursor cannot disagree.
size_t ktx2_level_bytes(ktx2_fmt_ format, uint32_t width, uint32_t height);

// `width` bits at absolute bit `at`. Blocks start zeroed and fields never
// overlap, so this ORs rather than masks, touching at most five bytes.
static inline void ktx2_bits_put(uint8_t* ref_block, uint32_t at, uint32_t width, uint32_t value) {
	if (width == 0) return;
	uint64_t bits = (uint64_t)(value & (width >= 32 ? 0xFFFFFFFFu : (1u << width) - 1u)) << (at & 7);
	uint32_t last = (at + width - 1) >> 3;
	for (uint32_t byte = at >> 3; byte <= last; byte++, bits >>= 8)
		ref_block[byte] |= (uint8_t)bits;
}

///////////////////////////////////////////////////////////////////////////////
// Bit reader: LSB-to-MSB within each byte, the opposite of JPEG.

typedef struct ktx2_bits_t {
	const uint8_t* data;
	size_t         bytes;
	size_t         at;      // next byte to pull
	uint64_t       buffer;  // 64-bit so a 32-bit request never overflows the shift
	int32_t        count;   // valid bits in buffer
	bool           overrun; // ran past the end; every later read is poisoned
} ktx2_bits_t;

static inline void ktx2_bits_init(ktx2_bits_t* out_bits, const void* data, size_t bytes) {
	out_bits->data    = (const uint8_t*)data;
	out_bits->bytes   = bytes;
	out_bits->at      = 0;
	out_bits->buffer  = 0;
	out_bits->count   = 0;
	out_bits->overrun = false;
}

// Up to 32 bits. Past the end yields zeroes and latches `overrun` instead of
// failing, so callers check once per section rather than once per read.
static inline uint32_t ktx2_bits_get(ktx2_bits_t* ref_bits, int32_t bit_count) {
	if (bit_count == 0) return 0;
	while (ref_bits->count < bit_count) {
		uint32_t byte = 0;
		if (ref_bits->at < ref_bits->bytes) byte = ref_bits->data[ref_bits->at++];
		else                                ref_bits->overrun = true;
		ref_bits->buffer |= (uint64_t)byte << ref_bits->count;
		ref_bits->count  += 8;
	}
	uint32_t value = (uint32_t)(ref_bits->buffer & ((bit_count == 32) ? 0xFFFFFFFFu : ((1u << bit_count) - 1u)));
	ref_bits->buffer >>= bit_count;
	ref_bits->count   -= bit_count;
	return value;
}

// Variable length code: chunk_bits of payload plus a continuation bit, repeated.
uint32_t ktx2_bits_vlc(ktx2_bits_t* ref_bits, int32_t chunk_bits);

///////////////////////////////////////////////////////////////////////////////
// Bump allocator over the caller's scratch buffer. Mark/reset lets the codebook
// Huffman tables be reclaimed before the slice tables are read.

typedef struct ktx2_arena_t {
	uint8_t* data;
	size_t   bytes;
	size_t   at;
	bool     exhausted;
} ktx2_arena_t;

void* ktx2_arena_alloc(ktx2_arena_t* ref_arena, size_t bytes);

///////////////////////////////////////////////////////////////////////////////
// Canonical Huffman, Deflate-compatible (RFC 1951 section 3.2.2).

typedef struct ktx2_huff_t {
	uint16_t  counts[KTX2_HUFF_MAX_CODE_BITS + 1]; // codes of each length
	uint16_t* symbols;                             // sorted by (length, symbol)
	uint32_t  symbol_count;
} ktx2_huff_t;

// Reads one RLE+Huffman-coded code length table and builds a decoder from it.
bool    ktx2_huff_read  (ktx2_bits_t* ref_bits, ktx2_arena_t* ref_arena, ktx2_huff_t* out_huff);
// Returns -1 on an invalid code, which is how a truncated stream surfaces.
int32_t ktx2_huff_decode(ktx2_bits_t* ref_bits, const ktx2_huff_t* huff);

///////////////////////////////////////////////////////////////////////////////
// ETC1S codebooks.

typedef struct ktx2_endpoint_t {
	uint8_t color5[3]; // 5-bit base colour, expanded on use
	uint8_t inten;     // 3-bit intensity table index
} ktx2_endpoint_t;

// One byte per texel row, four 2-bit selectors each, x=0 at the LSB. Stored in
// sorted-modifier order, NOT ETC1 wire order - see ktx2_etc1s_repack.c.
typedef struct ktx2_selector_t {
	uint8_t rows[4];
} ktx2_selector_t;

typedef struct ktx2_block_pred_t {
	uint16_t endpoint_index;
	uint8_t  pred_bits;
} ktx2_block_pred_t;

#define KTX2_MAX_SELECTOR_HISTORY 64

typedef struct ktx2_etc1s_t {
	ktx2_endpoint_t*   endpoints;
	uint32_t           endpoint_count;
	ktx2_selector_t*   selectors;
	uint32_t           selector_count;

	ktx2_huff_t        endpoint_pred;
	ktx2_huff_t        delta_endpoint;
	ktx2_huff_t        selector;
	ktx2_huff_t        selector_rle;
	uint32_t           history_size;

	ktx2_block_pred_t* preds;      // 2 rows of blocks_x
	uint32_t           preds_pitch;
} ktx2_etc1s_t;

// Endpoint codebook, selector codebook and slice Huffman tables, out of the
// supercompression global data. Everything after this is per-slice.
ktx2_result_ ktx2_etc1s_read_codebooks(const ktx2_reader_t* reader, ktx2_arena_t* ref_arena, ktx2_etc1s_t* out_etc1s);

// Decodes one slice into per-block codebook indices, `blocks_x * blocks_y`
// entries in raster order, packed as endpoint | selector << 16.
ktx2_result_ ktx2_etc1s_decode_slice(ktx2_etc1s_t* ref_etc1s, const void* data, size_t bytes,
                                     uint32_t blocks_x, uint32_t blocks_y, uint32_t* out_blocks);

///////////////////////////////////////////////////////////////////////////////
// Block output.

// [intensity][pixel index], in ETC1's wire order, not basisu's sorted order.
extern const int16_t ktx2_etc1_modifier[8][4];

// ETC1S selector -> ETC1 pixel index. Wrong still yields well-formed blocks, so
// it is defined once; ktx2_etc1s_repack.c has the note on how it was found.
extern const uint8_t ktx2_selector_to_etc1[4];

void ktx2_etc1s_write_etc1(const ktx2_endpoint_t* endpoint, const ktx2_selector_t* selector, uint8_t out_block[8]);
void ktx2_etc1s_write_rgba(const ktx2_endpoint_t* endpoint, const ktx2_selector_t* selector, uint8_t out_texels[64]);

///////////////////////////////////////////////////////////////////////////////
// Single-channel paths. A one-channel slice resolves to four grey levels fixed
// by the endpoint's 5-bit base and 3-bit intensity, so there are only 32 * 8
// ramps in the format and EAC or BC4 is a table lookup, not a search.
//
// ETC1S spec section 11: a second slice carries its payload in the *green*
// channel, whether that payload is alpha or green.

// Contiguous selector spans a block can occupy: 4 + 3 + 2 + 1.
#define KTX2_BC1_RANGES 10

// (min selector, max selector) -> range slot; only max >= min is reachable.
extern const uint8_t ktx2_range_slot[4][4];

// Span of selectors a block actually uses, as a range slot.
uint32_t ktx2_selector_range(const ktx2_selector_t* selector);

typedef struct ktx2_gray_fit_t {
	uint8_t eac_multiplier;
	uint8_t eac_table;
	uint8_t eac_index[4]; // ETC1S selector -> EAC 3-bit index
	uint8_t bc4_lo;       // relative to the block's base, applied when writing
	uint8_t bc4_hi;
	uint8_t bc4_index[4]; // ETC1S selector -> BC4 3-bit index
} ktx2_gray_fit_t;

// Bases x intensity tables x selector spans. The span matters as it does in
// BC1: a block reaching two levels wants its endpoints there, not on the full ramp.
#define KTX2_GRAY_FIT_COUNT (32 * 8 * KTX2_BC1_RANGES)

void ktx2_gray_fit_build(ktx2_gray_fit_t* out_fits);

// `base5` is the endpoint's green component, per section 11.
void ktx2_etc1s_write_eac (const ktx2_gray_fit_t* fits, const ktx2_endpoint_t* endpoint, const ktx2_selector_t* selector, uint8_t out_block[8]);
void ktx2_etc1s_write_bc4 (const ktx2_gray_fit_t* fits, const ktx2_endpoint_t* endpoint, const ktx2_selector_t* selector, uint8_t out_block[8]);
void ktx2_etc1s_write_gray(const ktx2_endpoint_t* endpoint, const ktx2_selector_t* selector, uint8_t out_texels[16]);

///////////////////////////////////////////////////////////////////////////////
// ETC1S -> BC1, the only target with no shared block structure and so the only
// real endpoint fit. Built at load, not baked; see ktx2_bc1.c for the indexing.

typedef struct ktx2_bc1_table_t {
	uint8_t index[8][KTX2_BC1_RANGES][4];       // [intensity][range][selector] -> BC1 index
	uint8_t five [32][8][KTX2_BC1_RANGES][2];   // red and blue endpoints, {lo, hi}
	uint8_t six  [32][8][KTX2_BC1_RANGES][2];   // green endpoints
} ktx2_bc1_table_t;

void ktx2_bc1_table_build(ktx2_bc1_table_t* out_table);

///////////////////////////////////////////////////////////////////////////////
// The two fit tables, carved out of the caller's context. Identical for every
// file and expensive to build, so each is built once and only when a format
// reads it; ETC1, ASTC, BC7 and the uncompressed decodes need neither.

#define KTX2_TABLE_GRAY 1u // bit in ktx2_context_t::tables_built
#define KTX2_TABLE_BC1  2u

const ktx2_gray_fit_t*  ktx2_context_gray(ktx2_context_t* ref_context);
const ktx2_bc1_table_t* ktx2_context_bc1 (ktx2_context_t* ref_context);
void ktx2_etc1s_write_bc1(const ktx2_bc1_table_t* table, const ktx2_endpoint_t* endpoint,
                          const ktx2_selector_t* selector, uint8_t out_block[8]);

///////////////////////////////////////////////////////////////////////////////
// UASTC LDR 4x4: a 19 mode subset of ASTC in a flat 128-bit block. No entropy
// coding and no codebook, so unpacking is a pure function of the 16 bytes.
//
// Endpoints stay *quantized* here. The ASTC repack copies them straight across,
// so dequantizing early would throw away what that path needs.

#define KTX2_UASTC_MODE_VOID    8   // solid colour, no endpoints or weights
#define KTX2_UASTC_MODE_INVALID 20  // 19 is reserved by the spec, 20 is ours

// Malformed blocks decode to opaque magenta. The spec's test vectors include
// invalid blocks and require this colour, so it is defined behaviour, not a choice.
#define KTX2_UASTC_INVALID_R 255
#define KTX2_UASTC_INVALID_G 0
#define KTX2_UASTC_INVALID_B 255

typedef struct ktx2_uastc_t {
	uint8_t mode;
	uint8_t subsets;        // 1-3
	uint8_t planes;         // 1, or 2 for dual plane modes
	uint8_t comps;          // 2 (LA), 3 (RGB), or 4 (RGBA)
	uint8_t compsel;        // ASTC colour component selector, dual plane only
	uint8_t weight_bits;    // 1-5, never BISE coded
	uint8_t endpoint_range; // index into the ASTC BISE range table
	uint8_t pattern_index;  // PAT as stored; ASTC needs a seed, not the pattern
	const uint8_t* pattern; // texel -> subset, NULL when there is one subset
	uint8_t quant  [18];    // BISE endpoint integers, ASTC order: RL RH GL GH ..
	uint8_t weights[32];    // raster order, p0 p1 interleaved when dual plane
	uint8_t solid  [4];     // void extent RGBA
} ktx2_uastc_t;

typedef struct ktx2_uastc_mode_t {
	uint8_t weight_bits;
	uint8_t endpoint_range;
	uint8_t subsets;
	uint8_t planes;
	uint8_t comps;      // 2 is LA (CEM 4), 3 is RGB (CEM 8), 4 is RGBA (CEM 12)
	uint8_t bc1_hint1;  // hint field presence; every non-void mode has hint 0
	uint8_t etc1_bias;
	uint8_t etc2_hint;
} ktx2_uastc_mode_t;

// Never fails: an unreadable block comes back as KTX2_UASTC_MODE_INVALID.
void ktx2_uastc_unpack (const uint8_t block[16], ktx2_uastc_t* out_block);
void ktx2_uastc_to_rgba(const ktx2_uastc_t* block, uint8_t out_texels[64]);

// Mode 8 is void extent and carries none of this; its row is zeroed.
extern const ktx2_uastc_mode_t ktx2_uastc_modes[19];

// Dual plane modes store a component selector, except LA: with two components
// the spec pins the second plane to alpha and spends no bits saying so.
#define KTX2_UASTC_LA_COMPSEL 3
static inline bool ktx2_uastc_has_compsel(const ktx2_uastc_mode_t* mode) {
	return mode->planes == 2 && mode->comps != 2;
}
// A 2-7 bit Huffman code at bit 0. Seven bits always resolve it, so the spec
// supplies a flat table instead of a decode loop.
extern const uint8_t ktx2_uastc_mode_lut [128];
extern const uint8_t ktx2_uastc_mode_bits[20];

// Lossless. Unreadable blocks become a magenta void extent, matching to_rgba.
void ktx2_uastc_write_astc(const ktx2_uastc_t* block, uint8_t out_block[16]);
// Near-lossless: BC7's endpoints are lower precision. Prefer ASTC when offered.
void ktx2_uastc_write_bc7 (const ktx2_uastc_t* block, uint8_t out_block[16]);

// Exposed for the tests: each inverts a decode rule transcribed from the spec,
// so each is worth pinning on its own.
uint32_t ktx2_uastc_pack_trits (const uint8_t trit [5]);
uint32_t ktx2_uastc_pack_quints(const uint8_t quint[3]);
uint32_t ktx2_uastc_block_mode (uint32_t weight_bits, uint32_t planes);

// UASTC's 60 patterns are the ones ASTC and BC7 share, so no texel changes
// subset - but BC7 may number subsets differently, and mode 7 rides a 3-subset
// BC7 pattern with two endpoint pairs equal. See ktx2_uastc_bc7_tables.c.
uint32_t ktx2_uastc_bc7_pattern(uint32_t mode, uint32_t subsets, uint32_t pattern);
uint32_t ktx2_uastc_bc7_subset (uint32_t mode, uint32_t subsets, uint32_t pattern, uint32_t subset);
uint32_t ktx2_uastc_bc7_anchor (uint32_t bc7_subsets, uint32_t pattern, uint32_t subset);
// Mode 7 only, BC7 subset -> UASTC subset. Many-to-one, so it does not invert.
uint32_t ktx2_uastc_bc7_collapse(uint32_t pattern, uint32_t bc7_subset);
uint32_t ktx2_uastc_bc7_texel_subset(uint32_t bc7_subsets, uint32_t bc7_pattern, uint32_t texel);

// Dequantized endpoint pairs per subset, RGBA. ASTC needs them for the blue
// contraction test, BC7 for requantization, the pixel decode to interpolate.
void ktx2_uastc_endpoints(const ktx2_uastc_t* block, uint8_t out_low[3][4], uint8_t out_high[3][4]);

const uint8_t* ktx2_uastc_pattern(uint32_t mode, uint32_t subsets, uint32_t pattern);
const uint8_t* ktx2_uastc_dequant(uint32_t range);
uint32_t       ktx2_uastc_astc_seed(uint32_t mode, uint32_t subsets, uint32_t pattern);
// Bits, then trits or quints, per ASTC BISE range. `tq` is 0 for a pure binary
// range, 1 for trits, 2 for quints.
typedef struct ktx2_bise_t { uint8_t bits, tq; } ktx2_bise_t;
extern const ktx2_bise_t ktx2_bise_range[21];
