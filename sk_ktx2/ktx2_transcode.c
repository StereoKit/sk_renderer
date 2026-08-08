// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Transcode dispatch. Decodes the shared codebooks once, then walks the mip
// chain largest-first so the output lands in skr_tex_data_t's layout.

#include "ktx2_internal.h"

#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////

// One decoded level in `format`. `second` is the alpha or green slice, NULL for
// single-slice sources. Block counts drive this, not pixels: a 2x2 mip is still
// a whole block.
static ktx2_result_ _write_blocks(const ktx2_etc1s_t* etc1s, const ktx2_gray_fit_t* fits,
                                  const ktx2_bc1_table_t* bc1,
                                  const uint32_t* blocks, const uint32_t* second,
                                  uint32_t blocks_x, uint32_t blocks_y,
                                  ktx2_fmt_ format, uint32_t width, uint32_t height, uint8_t* out_data) {
	for (uint32_t block_y = 0; block_y < blocks_y; block_y++) {
		for (uint32_t block_x = 0; block_x < blocks_x; block_x++) {
			size_t   block  = (size_t)block_y * blocks_x + block_x;
			uint32_t packed = blocks[block];
			const ktx2_endpoint_t* endpoint = &etc1s->endpoints[packed & 0xFFFF];
			const ktx2_selector_t* selector = &etc1s->selectors[packed >> 16];

			const ktx2_endpoint_t* endpoint2 = NULL;
			const ktx2_selector_t* selector2 = NULL;
			if (second != NULL) {
				endpoint2 = &etc1s->endpoints[second[block] & 0xFFFF];
				selector2 = &etc1s->selectors[second[block] >> 16];
			}

			switch (format) {
			case ktx2_fmt_etc1_rgb:
			case ktx2_fmt_etc1_rgb_srgb:
				ktx2_etc1s_write_etc1(endpoint, selector, out_data + block * 8);
				break;

			// ETC2 RGBA is the EAC alpha block followed by the ETC1 colour block.
			case ktx2_fmt_etc2_rgba:
			case ktx2_fmt_etc2_rgba_srgb:
				ktx2_etc1s_write_eac (fits, endpoint2, selector2, out_data + block * 16);
				ktx2_etc1s_write_etc1(      endpoint,  selector,  out_data + block * 16 + 8);
				break;

			case ktx2_fmt_eac_r11:
				ktx2_etc1s_write_eac(fits, endpoint, selector, out_data + block * 8);
				break;
			case ktx2_fmt_eac_rg11:
				ktx2_etc1s_write_eac(fits, endpoint,  selector,  out_data + block * 16);
				ktx2_etc1s_write_eac(fits, endpoint2, selector2, out_data + block * 16 + 8);
				break;

			case ktx2_fmt_bc1_rgb:
			case ktx2_fmt_bc1_rgb_srgb:
				ktx2_etc1s_write_bc1(bc1, endpoint, selector, out_data + block * 8);
				break;

			// BC3 is the BC4 alpha block followed by the BC1 colour block.
			case ktx2_fmt_bc3_rgba:
			case ktx2_fmt_bc3_rgba_srgb:
				ktx2_etc1s_write_bc4(fits, endpoint2, selector2, out_data + block * 16);
				ktx2_etc1s_write_bc1(bc1,  endpoint,  selector,  out_data + block * 16 + 8);
				break;

			case ktx2_fmt_bc4_r:
				ktx2_etc1s_write_bc4(fits, endpoint, selector, out_data + block * 8);
				break;
			case ktx2_fmt_bc5_rg:
				ktx2_etc1s_write_bc4(fits, endpoint,  selector,  out_data + block * 16);
				ktx2_etc1s_write_bc4(fits, endpoint2, selector2, out_data + block * 16 + 8);
				break;

#ifdef SK_KTX2_DECODE_UNCOMPRESSED
			case ktx2_fmt_rgba32:
			case ktx2_fmt_rgba32_srgb:
			case ktx2_fmt_r8:
			case ktx2_fmt_rg8: {
				uint32_t stride = format == ktx2_fmt_r8 ? 1 : (format == ktx2_fmt_rg8 ? 2 : 4);
				uint8_t  texels[64];
				uint8_t  gray  [16];
				uint8_t  gray2 [16];
				if (stride == 4) ktx2_etc1s_write_rgba(endpoint, selector, texels);
				else             ktx2_etc1s_write_gray(endpoint, selector, gray);
				if (second != NULL) ktx2_etc1s_write_gray(endpoint2, selector2, gray2);

				// The last block of a row hangs off a mip whose size is not a
				// multiple of 4; copy only what lands inside.
				uint32_t copy_w = width  - block_x * 4 < 4 ? width  - block_x * 4 : 4;
				uint32_t copy_h = height - block_y * 4 < 4 ? height - block_y * 4 : 4;
				for (uint32_t y = 0; y < copy_h; y++) {
					for (uint32_t x = 0; x < copy_w; x++) {
						uint8_t* dst = out_data + ((size_t)(block_y * 4 + y) * width + block_x * 4 + x) * stride;
						if (stride == 4) {
							memcpy(dst, texels + (y * 4 + x) * 4, 4);
							if (second != NULL) dst[3] = gray2[y * 4 + x];
						} else {
							dst[0] = gray[y * 4 + x];
							if (stride == 2) dst[1] = second != NULL ? gray2[y * 4 + x] : 0;
						}
					}
				}
				break;
			}
#endif
			default: return ktx2_result_unsupported;
			}
		}
	}
	return ktx2_result_success;
}

static ktx2_result_ _transcode_etc1s(const ktx2_plan_t* plan, uint8_t* out_data, ktx2_arena_t* ref_arena) {
	const ktx2_reader_t* reader     = plan->reader;
	bool                 two_slices = reader->sample_count == 2;

	ktx2_etc1s_t etc1s;
	ktx2_result_ result = ktx2_etc1s_read_codebooks(reader, ref_arena, &etc1s);
	if (result != ktx2_result_success) return result;

	uint32_t height   = reader->height ? reader->height : 1;
	uint32_t blocks_x = ktx2_blocks(reader->width);
	size_t   slice    = sizeof(uint32_t) * (size_t)blocks_x * ktx2_blocks(height);

	etc1s.preds = (ktx2_block_pred_t*)ktx2_arena_alloc(ref_arena, sizeof(ktx2_block_pred_t) * 2 * blocks_x);
	uint32_t*        slice_blocks  = (uint32_t*)ktx2_arena_alloc(ref_arena, slice);
	uint32_t*        second_blocks = two_slices ? (uint32_t*)ktx2_arena_alloc(ref_arena, slice) : NULL;
	if (ref_arena->exhausted) return ktx2_result_buffer_too_small;

	// Both fit tables cost real time to derive and most targets need neither, so
	// ask the context only for what this format reads. It builds each at most once.
	bool needs_gray = false, needs_bc1 = false;
	switch (plan->format) {
	case ktx2_fmt_etc2_rgba: case ktx2_fmt_etc2_rgba_srgb:
	case ktx2_fmt_eac_r11:   case ktx2_fmt_eac_rg11:
	case ktx2_fmt_bc4_r:     case ktx2_fmt_bc5_rg:      needs_gray = true; break;
	case ktx2_fmt_bc1_rgb:   case ktx2_fmt_bc1_rgb_srgb: needs_bc1 = true; break;
	case ktx2_fmt_bc3_rgba:  case ktx2_fmt_bc3_rgba_srgb: needs_gray = true; needs_bc1 = true; break;
	default: break;
	}

	const ktx2_gray_fit_t*  fits = needs_gray ? ktx2_context_gray(plan->context) : NULL;
	const ktx2_bc1_table_t* bc1  = needs_bc1  ? ktx2_context_bc1 (plan->context) : NULL;

	size_t written = 0;
	for (uint32_t level = 0; level < reader->level_stored; level++) {
		uint32_t level_w  = ktx2_mip_size(reader->width, level);
		uint32_t level_h  = ktx2_mip_size(height,        level);
		uint32_t level_bx = ktx2_blocks(level_w);
		uint32_t level_by = ktx2_blocks(level_h);

		// 2D only, so the image index and the level index are the same thing.
		const uint8_t* desc      = reader->sgd_image_descs + (size_t)level * KTX2_IMAGE_DESC_BYTES;
		uint32_t       rgb_at    = ktx2_rd_u32(desc +  4);
		uint32_t       rgb_bytes = ktx2_rd_u32(desc +  8);
		uint32_t       aux_at    = ktx2_rd_u32(desc + 12);
		uint32_t       aux_bytes = ktx2_rd_u32(desc + 16);
		const uint8_t* level_at  = reader->data + reader->levels[level].offset;

		result = ktx2_etc1s_decode_slice(&etc1s, level_at + rgb_at, rgb_bytes, level_bx, level_by, slice_blocks);
		if (result != ktx2_result_success) return result;

		if (two_slices) {
			result = ktx2_etc1s_decode_slice(&etc1s, level_at + aux_at, aux_bytes, level_bx, level_by, second_blocks);
			if (result != ktx2_result_success) return result;
		}

		result = _write_blocks(&etc1s, fits, bc1, slice_blocks, second_blocks, level_bx, level_by,
			plan->format, level_w, level_h, out_data + written);
		if (result != ktx2_result_success) return result;

		written += ktx2_level_bytes(plan->format, level_w, level_h);
	}
	return ktx2_result_success;
}

///////////////////////////////////////////////////////////////////////////////

#ifdef SK_KTX2_UASTC

// UASTC needs no codebooks: every block stands alone, so this is a flat walk.
// Blocks are always 16 bytes and 4x4, hence no block-size arithmetic.
static ktx2_result_ _transcode_uastc(const ktx2_plan_t* plan, uint8_t* out_data, ktx2_arena_t* ref_arena) {
	const ktx2_reader_t* reader = plan->reader;
	if (reader->supercompression != 0 && reader->supercompression != 2)
		return ktx2_result_unsupported;

	uint32_t height = reader->height ? reader->height : 1;

	// One buffer reused across levels: each is an independent Zstd frame, and mip
	// 0 is the largest level of any block format.
	uint8_t* inflated = NULL;
	if (reader->supercompression == 2) {
		size_t largest = (size_t)ktx2_blocks(reader->width) * ktx2_blocks(height) * KTX2_UASTC_BLOCK_BYTES;
		inflated = (uint8_t*)ktx2_arena_alloc(ref_arena, largest);
		if (ref_arena->exhausted) return ktx2_result_buffer_too_small;
	}

	size_t written = 0;
	for (uint32_t level = 0; level < reader->level_stored; level++) {
		uint32_t level_w  = ktx2_mip_size(reader->width, level);
		uint32_t level_h  = ktx2_mip_size(height,        level);
		uint32_t level_bx = ktx2_blocks(level_w);
		uint32_t level_by = ktx2_blocks(level_h);
		uint64_t need     = (uint64_t)level_bx * level_by * KTX2_UASTC_BLOCK_BYTES;

		const uint8_t* src   = reader->data + reader->levels[level].offset;
		uint64_t       avail = reader->levels[level].bytes;
		if (inflated != NULL) {
			// Geometry fixes the inflated size, so the level index's figure is a
			// claim to check, not a size to honour - scratch cannot hold more.
			if (reader->levels[level].uncompressed_bytes != need) return ktx2_result_corrupt;
			size_t got = plan->context->zstd(plan->context->zstd_context,
				src, (size_t)reader->levels[level].bytes, inflated, (size_t)need);
			// A short inflate means the file lied about the size: corruption rather
			// than a host failure, and unusable either way.
			if (got != need) return ktx2_result_corrupt;
			src   = inflated;
			avail = got;
		}
		if (avail < need) return ktx2_result_corrupt;

		for (uint32_t block_y = 0; block_y < level_by; block_y++) {
			for (uint32_t block_x = 0; block_x < level_bx; block_x++) {
				size_t       block = (size_t)block_y * level_bx + block_x;
				ktx2_uastc_t unpacked;
				ktx2_uastc_unpack(src + block * 16, &unpacked);

				switch (plan->format) {
				case ktx2_fmt_astc4x4_rgba:
				case ktx2_fmt_astc4x4_rgba_srgb:
					ktx2_uastc_write_astc(&unpacked, out_data + written + block * 16);
					break;

				case ktx2_fmt_bc7_rgba:
				case ktx2_fmt_bc7_rgba_srgb:
					ktx2_uastc_write_bc7(&unpacked, out_data + written + block * 16);
					break;

#ifdef SK_KTX2_DECODE_UNCOMPRESSED
				case ktx2_fmt_rgba32:
				case ktx2_fmt_rgba32_srgb:
				case ktx2_fmt_r8:
				case ktx2_fmt_rg8: {
					uint8_t  texels[64];
					uint32_t stride = plan->format == ktx2_fmt_r8 ? 1 : (plan->format == ktx2_fmt_rg8 ? 2 : 4);
					ktx2_uastc_to_rgba(&unpacked, texels);

					uint32_t copy_w = level_w - block_x * 4 < 4 ? level_w - block_x * 4 : 4;
					uint32_t copy_h = level_h - block_y * 4 < 4 ? level_h - block_y * 4 : 4;
					for (uint32_t y = 0; y < copy_h; y++) {
						for (uint32_t x = 0; x < copy_w; x++) {
							const uint8_t* texel = texels + (y * 4 + x) * 4;
							uint8_t*       dst   = out_data + written
								+ ((size_t)(block_y * 4 + y) * level_w + block_x * 4 + x) * stride;
							if (stride == 4) { memcpy(dst, texel, 4); continue; }
							// One-channel sources replicate across RGB and two-channel
							// ones put the second in alpha, so green is never green.
							dst[0] = texel[0];
							if (stride == 2) dst[1] = texel[3];
						}
					}
					break;
				}
#endif
				default: return ktx2_result_unsupported;
				}
			}
		}
		written += ktx2_level_bytes(plan->format, level_w, level_h);
	}
	return ktx2_result_success;
}

#endif

///////////////////////////////////////////////////////////////////////////////

ktx2_result_ ktx2_transcode(const ktx2_plan_t* plan, void* out_data, size_t out_bytes, void* opt_scratch) {
	if (plan->reader == NULL)         return ktx2_result_corrupt;
	if (out_bytes < plan->data_bytes) return ktx2_result_buffer_too_small;

	void* scratch = opt_scratch;
	if (scratch == NULL && plan->scratch_bytes > 0) {
		scratch = malloc(plan->scratch_bytes);
		if (scratch == NULL) return ktx2_result_buffer_too_small;
	}

	ktx2_arena_t arena  = { (uint8_t*)scratch, plan->scratch_bytes, 0, false };
	ktx2_result_ result = ktx2_result_unsupported;
	if (plan->reader->source == ktx2_source_etc1s)
		result = _transcode_etc1s(plan, (uint8_t*)out_data, &arena);
#ifdef SK_KTX2_UASTC
	else if (plan->reader->source == ktx2_source_uastc_ldr_4x4)
		result = _transcode_uastc(plan, (uint8_t*)out_data, &arena);
#endif

	if (scratch != opt_scratch) free(scratch);
	return result;
}
