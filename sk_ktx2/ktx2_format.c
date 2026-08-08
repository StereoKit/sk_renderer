// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Output format selection and sizing. ktx2_plan is the only place in the API
// where a decision gets made: given source, channels and caps there is one right
// answer, so the caller is not asked for one.

#include "ktx2_internal.h"

#include <string.h>

typedef struct ktx2_fmt_desc_t {
	const char* name;
	uint8_t     block_width;
	uint8_t     block_height;
	uint8_t     block_bytes;
} ktx2_fmt_desc_t;

static const ktx2_fmt_desc_t k_fmt[] = {
	{ "none",              0, 0,  0 },
	{ "etc1_rgb",          4, 4,  8 }, { "etc1_rgb_srgb",     4, 4,  8 },
	{ "etc2_rgba",         4, 4, 16 }, { "etc2_rgba_srgb",    4, 4, 16 },
	{ "eac_r11",           4, 4,  8 }, { "eac_rg11",          4, 4, 16 },
	{ "bc1_rgb",           4, 4,  8 }, { "bc1_rgb_srgb",      4, 4,  8 },
	{ "bc3_rgba",          4, 4, 16 }, { "bc3_rgba_srgb",     4, 4, 16 },
	{ "bc4_r",             4, 4,  8 }, { "bc5_rg",            4, 4, 16 },
	{ "bc7_rgba",          4, 4, 16 }, { "bc7_rgba_srgb",     4, 4, 16 },
	{ "astc4x4_rgba",      4, 4, 16 }, { "astc4x4_rgba_srgb", 4, 4, 16 },
	{ "r8",                1, 1,  1 }, { "rg8",               1, 1,  2 },
	{ "rgba32",            1, 1,  4 }, { "rgba32_srgb",       1, 1,  4 },
};

const char* ktx2_result_str(ktx2_result_ result) {
	switch (result) {
	case ktx2_result_success:             return "success";
	case ktx2_result_not_ktx2:            return "not a KTX2 file";
	case ktx2_result_corrupt:             return "corrupt";
	case ktx2_result_unsupported:         return "unsupported encoding";
	case ktx2_result_not_gltf_conformant: return "not KHR_texture_basisu conformant";
	case ktx2_result_no_target:           return "no format reachable with these capabilities";
	case ktx2_result_buffer_too_small:    return "output buffer too small";
	case ktx2_result_no_decompressor:     return "supercompressed, but no decompressor was supplied";
	default:                              return "unknown result";
	}
}

const char* ktx2_fmt_str(ktx2_fmt_ format) {
	if ((size_t)format >= sizeof(k_fmt) / sizeof(k_fmt[0])) return "invalid";
	return k_fmt[format].name;
}

const char* ktx2_source_str(ktx2_source_ source) {
	switch (source) {
	case ktx2_source_etc1s:          return "ETC1S";
	case ktx2_source_uastc_ldr_4x4:  return "UASTC LDR 4x4";
	case ktx2_source_astc:           return "ASTC";
	case ktx2_source_uastc_hdr_4x4:  return "UASTC HDR 4x4";
	case ktx2_source_uastc_hdr_6x6i: return "UASTC HDR 6x6 intermediate";
	default:                         return "unknown";
	}
}

void ktx2_fmt_block(ktx2_fmt_ format, int32_t* out_width, int32_t* out_height, int32_t* out_bytes) {
	const ktx2_fmt_desc_t* desc = (size_t)format < sizeof(k_fmt) / sizeof(k_fmt[0])
		? &k_fmt[format] : &k_fmt[0];
	*out_width  = desc->block_width;
	*out_height = desc->block_height;
	*out_bytes  = desc->block_bytes;
}

///////////////////////////////////////////////////////////////////////////////

// The mip tail reaches 2x2 and 1x1, and those still occupy a whole block.
// Rounding up here rather than per call site keeps it from being an off-by-one.
size_t ktx2_level_bytes(ktx2_fmt_ format, uint32_t width, uint32_t height) {
	int32_t block_w, block_h, block_bytes;
	ktx2_fmt_block(format, &block_w, &block_h, &block_bytes);
	uint32_t cols = ((width  + (uint32_t)block_w - 1) / (uint32_t)block_w);
	uint32_t rows = ((height + (uint32_t)block_h - 1) / (uint32_t)block_h);
	return (size_t)cols * (size_t)rows * (size_t)block_bytes;
}

// Match the output family to the SOURCE family, not one family per platform.
// ETC1S is already 4bpp ETC, so ETC targets are lossless at half what ASTC 4x4
// costs; UASTC is the ASTC/BC7 intersection, so both are bit rearrangement.
// ETC1S->ASTC and UASTC->ETC2 have no column: they need a real re-encode.
static ktx2_fmt_ _pick_format(ktx2_source_ source, ktx2_channels_ channels, bool is_srgb, ktx2_caps_ caps) {
	if (source == ktx2_source_etc1s) {
		switch (channels) {
		case ktx2_channels_rgb:
			if (caps & ktx2_caps_etc2) return is_srgb ? ktx2_fmt_etc1_rgb_srgb : ktx2_fmt_etc1_rgb;
			if (caps & ktx2_caps_bc  ) return is_srgb ? ktx2_fmt_bc1_rgb_srgb  : ktx2_fmt_bc1_rgb;
			return is_srgb ? ktx2_fmt_rgba32_srgb : ktx2_fmt_rgba32;
		case ktx2_channels_rgba:
			if (caps & ktx2_caps_etc2) return is_srgb ? ktx2_fmt_etc2_rgba_srgb : ktx2_fmt_etc2_rgba;
			if (caps & ktx2_caps_bc  ) return is_srgb ? ktx2_fmt_bc3_rgba_srgb  : ktx2_fmt_bc3_rgba;
			return is_srgb ? ktx2_fmt_rgba32_srgb : ktx2_fmt_rgba32;
		case ktx2_channels_r:
			if (caps & ktx2_caps_etc2) return ktx2_fmt_eac_r11;
			if (caps & ktx2_caps_bc  ) return ktx2_fmt_bc4_r;
			return ktx2_fmt_r8;
		case ktx2_channels_rg:
			if (caps & ktx2_caps_etc2) return ktx2_fmt_eac_rg11;
			if (caps & ktx2_caps_bc  ) return ktx2_fmt_bc5_rg;
			return ktx2_fmt_rg8;
		}
		return ktx2_fmt_none;
	}

	if (source == ktx2_source_uastc_ldr_4x4) {
		if (caps & ktx2_caps_astc_ldr) return is_srgb ? ktx2_fmt_astc4x4_rgba_srgb : ktx2_fmt_astc4x4_rgba;
		if (caps & ktx2_caps_bc      ) return is_srgb ? ktx2_fmt_bc7_rgba_srgb     : ktx2_fmt_bc7_rgba;
		// Both block targets carry all four channels, so only the uncompressed
		// fallback gains anything from narrowing - and there it saves 4x.
		switch (channels) {
		case ktx2_channels_r:  return ktx2_fmt_r8;
		case ktx2_channels_rg: return ktx2_fmt_rg8;
		default:               return is_srgb ? ktx2_fmt_rgba32_srgb : ktx2_fmt_rgba32;
		}
	}
	return ktx2_fmt_none;
}

#ifndef SK_KTX2_DECODE_UNCOMPRESSED
static bool _is_uncompressed(ktx2_fmt_ format) {
	return format == ktx2_fmt_r8     || format == ktx2_fmt_rg8 ||
	       format == ktx2_fmt_rgba32 || format == ktx2_fmt_rgba32_srgb;
}
#endif

ktx2_result_ ktx2_plan(const ktx2_reader_t* reader, ktx2_context_t* context,
                       ktx2_caps_ caps, ktx2_plan_t* out_plan) {
	memset(out_plan, 0, sizeof(*out_plan));

	if (reader->source != ktx2_source_etc1s && reader->source != ktx2_source_uastc_ldr_4x4)
		return ktx2_result_unsupported;
	if (!reader->channels_known)                                    return ktx2_result_unsupported;
	if (reader->layer_count != 0 || reader->face_count != 1 || reader->depth != 0)
		return ktx2_result_unsupported;                             // L1 is 2D only
	if (reader->source == ktx2_source_etc1s && reader->supercompression != 1)
		return ktx2_result_unsupported;
	if (reader->source == ktx2_source_uastc_ldr_4x4 && reader->supercompression != 0 && reader->supercompression != 2)
		return ktx2_result_unsupported;
	// Refuse here, not in transcode: by then the caller has been given a size, and
	// failing after that reads as a bug in the library.
	if (reader->supercompression == 2 && context->zstd == NULL)
		return ktx2_result_no_decompressor;
#ifndef SK_KTX2_UASTC
	if (reader->source == ktx2_source_uastc_ldr_4x4)                return ktx2_result_unsupported;
#endif

	bool      is_srgb = reader->transfer_fn == KTX2_TRANSFER_SRGB;
	ktx2_fmt_ format  = _pick_format(reader->source, reader->channels, is_srgb, caps);
	if (format == ktx2_fmt_none)                                    return ktx2_result_no_target;
#ifndef SK_KTX2_DECODE_UNCOMPRESSED
	if (_is_uncompressed(format))                                   return ktx2_result_no_target;
#endif

	uint32_t height = reader->height ? reader->height : 1;
	size_t   total  = 0;
	for (uint32_t level = 0; level < reader->level_stored; level++)
		total += ktx2_level_bytes(format, ktx2_mip_size(reader->width, level), ktx2_mip_size(height, level));

	out_plan->reader        = reader;
	out_plan->context       = context;
	out_plan->format        = format;
	out_plan->mip_count     = (int32_t)reader->level_stored;
	out_plan->data_bytes    = total;
	out_plan->scratch_bytes = 0;

	if (reader->source == ktx2_source_etc1s) {
		uint32_t blocks_x = ktx2_blocks(reader->width);
		uint32_t blocks_y = ktx2_blocks(height);
		// Codebooks, the largest slice's block indices, then two rows of endpoint
		// predictors. The Huffman allowance covers what is live at once: codebook
		// tables are reclaimed before slice tables are read, so this is a peak.
		uint32_t slices = reader->sample_count == 2 ? 2 : 1;
		out_plan->scratch_bytes =
			  (size_t)reader->endpoint_count * sizeof(ktx2_endpoint_t)
			+ (size_t)reader->selector_count * sizeof(ktx2_selector_t)
			+ (size_t)KTX2_HUFF_TABLES_LIVE  * KTX2_HUFF_MAX_SYMS * (sizeof(uint16_t) + sizeof(uint8_t))
			+ (size_t)blocks_x * blocks_y    * sizeof(uint32_t) * slices
			+ (size_t)blocks_x * 2           * sizeof(ktx2_block_pred_t)
			+ 256; // alignment slack across the individual allocations
		// The fit tables live on the context now - identical for every file, and
		// rebuilding them cost more than transcoding a small texture.
	} else if (reader->supercompression == 2) {
		// A level inflates whole, so scratch holds the largest - always mip 0 for a
		// block format. Sized from geometry, not from uncompressed_bytes, which is
		// whatever the file says; the transcode rejects a level that disagrees.
		out_plan->scratch_bytes = (size_t)ktx2_blocks(reader->width) * ktx2_blocks(height)
		                        * KTX2_UASTC_BLOCK_BYTES;
	}
	return ktx2_result_success;
}
