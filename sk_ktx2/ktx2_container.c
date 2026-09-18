// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// L0 - the KTX2 container: header, level index, DFD, KVD, BasisLZ global data.
// Reads from the caller's buffer and copies nothing but two short KV strings.
//
// Rejection happens in three separate layers, so a non-glTF caller can reuse the
// reader without unpicking validation logic:
//   ktx2_open              - well-formed KTX2 we can navigate?
//   ktx2_check_gltf_basisu - does KHR_texture_basisu permit it?
//   ktx2_plan              - can we produce something for this GPU?

#include "ktx2_internal.h"

#include <string.h>

#define KTX2_HEADER_BYTES 80  // identifier + header + index, level index follows
#define KTX2_LEVEL_BYTES  24
#define KTX2_SGD_HEADER_BYTES 20
#define KTX2_DFD_BLOCK_HEADER_BYTES 24
#define KTX2_DFD_SAMPLE_BYTES 16

// KHR_DF_MODEL_*
#define KTX2_MODEL_ETC1S 163
#define KTX2_MODEL_UASTC 166
#define KTX2_MODEL_UASTC_HDR_4X4  167
#define KTX2_MODEL_UASTC_HDR_6X6I 168
#define KTX2_MODEL_ASTC  162

// KHR_DF_PRIMARIES_*. The transfer function constants are in ktx2_internal.h;
// format selection needs them too.
#define KTX2_PRIMARIES_UNSPECIFIED 0
#define KTX2_PRIMARIES_BT709       1

// DFD channel IDs, verified against the CTS rather than recalled:
// r_reference_basis is {3}, rg_reference_basis {3,4}, alpha_simple_basis {0,15}.
// The only thing separating an ETC1S RGBA second slice from an RG one.
#define KTX2_ETC1S_RGB  0
#define KTX2_ETC1S_RRR  3
#define KTX2_ETC1S_GGG  4
#define KTX2_ETC1S_AAA 15
#define KTX2_UASTC_RGB  0
#define KTX2_UASTC_RGBA 3
#define KTX2_UASTC_RRR  4
#define KTX2_UASTC_RRRG 5

static const uint8_t k_ktx2_identifier[12] = {
	0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
};

///////////////////////////////////////////////////////////////////////////////

static uint64_t _rd_u64(const uint8_t* p) {
	return (uint64_t)ktx2_rd_u32(p) | ((uint64_t)ktx2_rd_u32(p + 4) << 32);
}

static uint16_t _rd_u16(const uint8_t* p) {
	return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}

// Overflow-safe: never forms offset+bytes, which is the whole point.
static bool _in_bounds(size_t total, uint64_t offset, uint64_t bytes) {
	return bytes <= (uint64_t)total && offset <= (uint64_t)total - bytes;
}

static uint32_t _full_mip_count(uint32_t width, uint32_t height) {
	uint32_t largest = width > height ? width : height;
	uint32_t count   = 1;
	while (largest > 1) { largest >>= 1; count++; }
	return count;
}

// 64-bit so a hostile layer count could never wrap the SGD length check, even
// though ktx2_open caps layerCount well below where that would matter.
static uint64_t _image_count(const ktx2_reader_t* reader) {
	uint64_t layers = reader->layer_count ? reader->layer_count : 1;
	return (uint64_t)reader->level_stored * layers * reader->face_count;
}

///////////////////////////////////////////////////////////////////////////////

// Naming the models we do not implement turns "unknown format" into a useful
// diagnostic, and makes adding them later purely additive.
static ktx2_source_ _source_from_model(uint8_t color_model) {
	switch (color_model) {
	case KTX2_MODEL_ETC1S:          return ktx2_source_etc1s;
	case KTX2_MODEL_UASTC:          return ktx2_source_uastc_ldr_4x4;
	case KTX2_MODEL_ASTC:           return ktx2_source_astc;
	case KTX2_MODEL_UASTC_HDR_4X4:  return ktx2_source_uastc_hdr_4x4;
	case KTX2_MODEL_UASTC_HDR_6X6I: return ktx2_source_uastc_hdr_6x6i;
	default:                        return ktx2_source_unknown;
	}
}

// Sets reader->channels, reports whether the IDs are a set we recognize. An
// unrecognized one is structurally decodable so it fails at ktx2_plan, not here.
// Guessing would silently decode every RG normal map as RGBA.
static bool _channels_from_dfd(ktx2_reader_t* ref_reader) {
	uint8_t a = ref_reader->channel_id[0];
	uint8_t b = ref_reader->channel_id[1];

	if (ref_reader->source == ktx2_source_etc1s) {
		if (ref_reader->sample_count == 1) {
			if (a == KTX2_ETC1S_RGB) { ref_reader->channels = ktx2_channels_rgb; return true; }
			if (a == KTX2_ETC1S_RRR) { ref_reader->channels = ktx2_channels_r;   return true; }
			return false;
		}
		if (a == KTX2_ETC1S_RGB && b == KTX2_ETC1S_AAA) { ref_reader->channels = ktx2_channels_rgba; return true; }
		if (a == KTX2_ETC1S_RRR && b == KTX2_ETC1S_GGG) { ref_reader->channels = ktx2_channels_rg;   return true; }
		return false;
	}

	if (ref_reader->source == ktx2_source_uastc_ldr_4x4) {
		if (ref_reader->sample_count != 1) return false;
		switch (a) {
		case KTX2_UASTC_RGB:  ref_reader->channels = ktx2_channels_rgb;  return true;
		case KTX2_UASTC_RGBA: ref_reader->channels = ktx2_channels_rgba; return true;
		case KTX2_UASTC_RRR:  ref_reader->channels = ktx2_channels_r;    return true;
		case KTX2_UASTC_RRRG: ref_reader->channels = ktx2_channels_rg;   return true;
		default:              return false;
		}
	}
	return false;
}

static ktx2_result_ _parse_dfd(ktx2_reader_t* ref_reader, uint32_t offset, uint32_t length) {
	const uint8_t* data = ref_reader->data;
	if (length < 4 + KTX2_DFD_BLOCK_HEADER_BYTES)             return ktx2_result_corrupt;
	if (!_in_bounds(ref_reader->bytes, offset, length))       return ktx2_result_corrupt;

	uint32_t total = ktx2_rd_u32(data + offset);
	if (total > length)                                       return ktx2_result_corrupt;

	const uint8_t* block       = data + offset + 4;
	uint32_t       vendor_word = ktx2_rd_u32(block);
	uint32_t       version     = ktx2_rd_u32(block + 4);
	uint32_t       vendor_id   = vendor_word & 0x1FFFF;
	uint32_t       desc_type   = vendor_word >> 17;
	uint32_t       block_bytes = version >> 16;

	if (vendor_id != 0 || desc_type != 0)                     return ktx2_result_unsupported;
	if (block_bytes < KTX2_DFD_BLOCK_HEADER_BYTES)            return ktx2_result_corrupt;
	if (block_bytes > length - 4)                             return ktx2_result_corrupt;
	if ((block_bytes - KTX2_DFD_BLOCK_HEADER_BYTES) % KTX2_DFD_SAMPLE_BYTES) return ktx2_result_corrupt;

	ref_reader->color_model     = block[8];
	ref_reader->color_primaries = block[9];
	ref_reader->transfer_fn     = block[10];
	ref_reader->dfd_flags       = block[11];

	uint32_t sample_count = (block_bytes - KTX2_DFD_BLOCK_HEADER_BYTES) / KTX2_DFD_SAMPLE_BYTES;
	if (sample_count < 1 || sample_count > 2)                 return ktx2_result_unsupported;

	ref_reader->sample_count = (uint8_t)sample_count;
	for (uint32_t i = 0; i < sample_count; i++) {
		const uint8_t* sample = block + KTX2_DFD_BLOCK_HEADER_BYTES + i * KTX2_DFD_SAMPLE_BYTES;
		ref_reader->channel_id[i] = sample[3] & 0x0F;
	}

	ref_reader->source = _source_from_model(ref_reader->color_model);
	if (ref_reader->source == ktx2_source_unknown)            return ktx2_result_unsupported;

	ref_reader->channels_known = _channels_from_dfd(ref_reader);
	return ktx2_result_success;
}

// Only KTXswizzle and KTXorientation matter; the rest is walked to prove the
// block is well formed.
static ktx2_result_ _parse_kvd(ktx2_reader_t* ref_reader, uint32_t offset, uint32_t length) {
	if (length == 0) return ktx2_result_success;
	if (!_in_bounds(ref_reader->bytes, offset, length)) return ktx2_result_corrupt;

	const uint8_t* data = ref_reader->data;
	uint32_t       at   = 0;
	while (at + 4 <= length) {
		uint32_t entry_bytes = ktx2_rd_u32(data + offset + at);
		at += 4;
		if (entry_bytes == 0 || entry_bytes > length - at) return ktx2_result_corrupt;

		const uint8_t* entry = data + offset + at;
		uint32_t       split = 0;
		while (split < entry_bytes && entry[split] != 0) split++;
		if (split == entry_bytes) return ktx2_result_corrupt; // key never terminated

		const char* key       = (const char*)entry;
		const char* value     = (const char*)entry + split + 1;
		uint32_t    value_len = entry_bytes - split - 1;

		char* dst = NULL;
		if      (strcmp(key, "KTXswizzle")     == 0) dst = ref_reader->swizzle;
		else if (strcmp(key, "KTXorientation") == 0) dst = ref_reader->orientation;
		if (dst != NULL) {
			uint32_t copy = value_len < 7 ? value_len : 7;
			for (uint32_t i = 0; i < copy && value[i] != 0; i++) dst[i] = value[i];
		}

		at += entry_bytes;
		at  = (at + 3) & ~3u; // entries are 4-byte aligned
	}
	return ktx2_result_success;
}

static ktx2_result_ _parse_sgd_basis(ktx2_reader_t* ref_reader, uint64_t offset, uint64_t length) {
	if (length < KTX2_SGD_HEADER_BYTES)                 return ktx2_result_corrupt;
	if (!_in_bounds(ref_reader->bytes, offset, length)) return ktx2_result_corrupt;

	const uint8_t* sgd = ref_reader->data + offset;
	ref_reader->endpoint_count       = _rd_u16(sgd + 0);
	ref_reader->selector_count       = _rd_u16(sgd + 2);
	ref_reader->sgd_endpoints_bytes  = ktx2_rd_u32(sgd + 4);
	ref_reader->sgd_selectors_bytes  = ktx2_rd_u32(sgd + 8);
	ref_reader->sgd_tables_bytes     = ktx2_rd_u32(sgd + 12);
	uint32_t extended_bytes          = ktx2_rd_u32(sgd + 16);

	// From the geometry, then checked against the blob size - a mismatch means the
	// file lies about one or the other.
	uint64_t image_count = _image_count(ref_reader);
	if (image_count * KTX2_IMAGE_DESC_BYTES > length) return ktx2_result_corrupt;
	ref_reader->sgd_image_count = (uint32_t)image_count;

	uint64_t need = (uint64_t)KTX2_SGD_HEADER_BYTES
	              + image_count * KTX2_IMAGE_DESC_BYTES
	              + ref_reader->sgd_endpoints_bytes
	              + ref_reader->sgd_selectors_bytes
	              + ref_reader->sgd_tables_bytes
	              + extended_bytes;
	if (need != length) return ktx2_result_corrupt;

	uint64_t at = KTX2_SGD_HEADER_BYTES;
	ref_reader->sgd_image_descs = sgd + at; at += (uint64_t)image_count * KTX2_IMAGE_DESC_BYTES;
	ref_reader->sgd_endpoints   = sgd + at; at += ref_reader->sgd_endpoints_bytes;
	ref_reader->sgd_selectors   = sgd + at; at += ref_reader->sgd_selectors_bytes;
	ref_reader->sgd_tables      = sgd + at;

	// Every image's slices must land inside the level blob they share.
	uint32_t layers = ref_reader->layer_count ? ref_reader->layer_count : 1;
	for (uint32_t level = 0; level < ref_reader->level_stored; level++) {
	for (uint32_t layer = 0; layer < layers;                  layer++) {
	for (uint32_t face  = 0; face  < ref_reader->face_count;  face ++) {
		uint32_t       image     = ktx2_image_index(ref_reader, level, layer, face);
		const uint8_t* desc      = ref_reader->sgd_image_descs + (size_t)image * KTX2_IMAGE_DESC_BYTES;
		uint64_t       rgb_at    = ktx2_rd_u32(desc + 4);
		uint64_t       rgb_bytes = ktx2_rd_u32(desc + 8);
		uint64_t       aux_at    = ktx2_rd_u32(desc + 12);
		uint64_t       aux_bytes = ktx2_rd_u32(desc + 16);
		uint64_t       level_len = ref_reader->levels[level].bytes;

		if (rgb_bytes == 0)                                   return ktx2_result_corrupt;
		if (!_in_bounds((size_t)level_len, rgb_at, rgb_bytes)) return ktx2_result_corrupt;
		if (aux_bytes != 0 && !_in_bounds((size_t)level_len, aux_at, aux_bytes)) return ktx2_result_corrupt;
		// A second slice in the file must agree with a second sample in the DFD.
		if ((aux_bytes != 0) != (ref_reader->sample_count == 2)) return ktx2_result_corrupt;
	}}}
	return ktx2_result_success;
}

///////////////////////////////////////////////////////////////////////////////

ktx2_result_ ktx2_open(const void* data, size_t bytes, ktx2_reader_t* out_reader) {
	memset(out_reader, 0, sizeof(*out_reader));

	if (bytes < KTX2_HEADER_BYTES)                                    return ktx2_result_not_ktx2;
	if (memcmp(data, k_ktx2_identifier, sizeof(k_ktx2_identifier)))   return ktx2_result_not_ktx2;

	const uint8_t* d = (const uint8_t*)data;
	out_reader->data             = d;
	out_reader->bytes            = bytes;
	out_reader->vk_format        = ktx2_rd_u32(d + 12);
	out_reader->type_size        = ktx2_rd_u32(d + 16);
	out_reader->width            = ktx2_rd_u32(d + 20);
	out_reader->height           = ktx2_rd_u32(d + 24);
	out_reader->depth            = ktx2_rd_u32(d + 28);
	out_reader->layer_count      = ktx2_rd_u32(d + 32);
	out_reader->face_count       = ktx2_rd_u32(d + 36);
	out_reader->level_count      = ktx2_rd_u32(d + 40);
	out_reader->supercompression = ktx2_rd_u32(d + 44);

	uint32_t dfd_offset = ktx2_rd_u32(d + 48);
	uint32_t dfd_bytes  = ktx2_rd_u32(d + 52);
	uint32_t kvd_offset = ktx2_rd_u32(d + 56);
	uint32_t kvd_bytes  = ktx2_rd_u32(d + 60);
	uint64_t sgd_offset = _rd_u64(d + 64);
	uint64_t sgd_bytes  = _rd_u64(d + 72);

	if (out_reader->width == 0)                                       return ktx2_result_corrupt;
	if (out_reader->face_count != 1 && out_reader->face_count != 6)   return ktx2_result_corrupt;
	if (out_reader->face_count == 6 && out_reader->width != out_reader->height) return ktx2_result_corrupt;
	if (out_reader->face_count == 6 && out_reader->depth != 0)        return ktx2_result_corrupt;
	if (out_reader->height == 0 && out_reader->depth != 0)            return ktx2_result_corrupt;
	if (out_reader->layer_count > KTX2_MAX_LAYERS)                    return ktx2_result_unsupported;
	if (out_reader->supercompression > 3)                             return ktx2_result_unsupported;
	if (out_reader->level_count > KTX2_MAX_LEVELS)                    return ktx2_result_unsupported;
	// Without this a file can name dimensions whose product overflows a size_t,
	// and the scratch sizing then agrees with the slice decoder on the wrapped figure.
	if (out_reader->width  > KTX2_MAX_DIMENSION ||
	    out_reader->height > KTX2_MAX_DIMENSION ||
	    out_reader->depth  > KTX2_MAX_DIMENSION)                      return ktx2_result_unsupported;

	uint32_t height = out_reader->height ? out_reader->height : 1;
	if (out_reader->level_count > _full_mip_count(out_reader->width, height)) return ktx2_result_corrupt;

	out_reader->level_stored = out_reader->level_count ? out_reader->level_count : 1;

	uint64_t index_bytes = (uint64_t)out_reader->level_stored * KTX2_LEVEL_BYTES;
	if (!_in_bounds(bytes, KTX2_HEADER_BYTES, index_bytes))           return ktx2_result_corrupt;

	for (uint32_t i = 0; i < out_reader->level_stored; i++) {
		const uint8_t* entry = d + KTX2_HEADER_BYTES + (size_t)i * KTX2_LEVEL_BYTES;
		ktx2_level_t*  level = &out_reader->levels[i];
		level->offset             = _rd_u64(entry);
		level->bytes              = _rd_u64(entry + 8);
		level->uncompressed_bytes = _rd_u64(entry + 16);

		if (level->bytes == 0)                                        return ktx2_result_corrupt;
		if (!_in_bounds(bytes, level->offset, level->bytes))          return ktx2_result_corrupt;
		if (out_reader->supercompression == 0 && level->uncompressed_bytes != level->bytes)
			return ktx2_result_corrupt;
	}

	ktx2_result_ result = _parse_dfd(out_reader, dfd_offset, dfd_bytes);
	if (result != ktx2_result_success) return result;

	result = _parse_kvd(out_reader, kvd_offset, kvd_bytes);
	if (result != ktx2_result_success) return result;

	if (out_reader->supercompression == 1) {
		result = _parse_sgd_basis(out_reader, sgd_offset, sgd_bytes);
		if (result != ktx2_result_success) return result;
	} else if (sgd_bytes != 0 && !_in_bounds(bytes, sgd_offset, sgd_bytes)) {
		return ktx2_result_corrupt;
	}

	// ETC1S and UASTC are block encodings with no Vulkan format of their own.
	if ((out_reader->source == ktx2_source_etc1s || out_reader->source == ktx2_source_uastc_ldr_4x4) &&
	    out_reader->vk_format != 0)
		return ktx2_result_corrupt;

	return ktx2_result_success;
}

ktx2_info_t ktx2_get_info(const ktx2_reader_t* reader) {
	ktx2_info_t info = {0};
	info.width       = (int32_t)reader->width;
	info.height      = (int32_t)(reader->height ? reader->height : 1);
	info.mip_count   = (int32_t)reader->level_stored;
	info.layer_count = (int32_t)(reader->layer_count ? reader->layer_count : 1);
	info.face_count  = (int32_t)reader->face_count;
	info.source      = reader->source;
	info.channels    = reader->channels;
	info.is_srgb     = reader->transfer_fn == KTX2_TRANSFER_SRGB;
	info.is_hdr      = reader->source == ktx2_source_uastc_hdr_4x4 ||
	                   reader->source == ktx2_source_uastc_hdr_6x6i;
	return info;
}

///////////////////////////////////////////////////////////////////////////////

ktx2_result_ ktx2_check_gltf_basisu(const ktx2_reader_t* reader) {
	// Encoding: ETC1S must be BasisLZ, UASTC must be uncompressed or Zstd.
	if (reader->source == ktx2_source_etc1s) {
		if (reader->supercompression != 1) return ktx2_result_not_gltf_conformant;
	} else if (reader->source == ktx2_source_uastc_ldr_4x4) {
		if (reader->supercompression != 0 && reader->supercompression != 2) return ktx2_result_not_gltf_conformant;
	} else {
		return ktx2_result_not_gltf_conformant; // raw ASTC and the HDR models are outside the extension
	}

	// 2D only: no arrays, no cubemaps, no 3D.
	if (reader->layer_count != 0 || reader->face_count != 1 || reader->depth != 0)
		return ktx2_result_not_gltf_conformant;

	// Base dimensions are multiples of 4. The mip tail is not, and need not be.
	uint32_t height = reader->height ? reader->height : 1;
	if ((reader->width % 4) != 0 || (height % 4) != 0)
		return ktx2_result_not_gltf_conformant;

	// A partial pyramid is an error, not just discouraged: CTS 3104 rejects 16x16
	// at 2, 3 and 4 levels. Either one level or all of them.
	if (reader->level_count > 1 && reader->level_count != _full_mip_count(reader->width, height))
		return ktx2_result_not_gltf_conformant;

	// The only place an RG normal map is distinguishable from RGBA. UASTC RRRG is
	// decodable, but the extension forbids it.
	if (!reader->channels_known) return ktx2_result_not_gltf_conformant;
	if (reader->source == ktx2_source_uastc_ldr_4x4 && reader->channel_id[0] == KTX2_UASTC_RRRG)
		return ktx2_result_not_gltf_conformant;

	// Colour is BT709 + sRGB, data is unspecified + linear. Mixed pairings are out.
	bool color = reader->color_primaries == KTX2_PRIMARIES_BT709        && reader->transfer_fn == KTX2_TRANSFER_SRGB;
	bool data  = reader->color_primaries == KTX2_PRIMARIES_UNSPECIFIED  && reader->transfer_fn == KTX2_TRANSFER_LINEAR;
	if (!color && !data) return ktx2_result_not_gltf_conformant;

	if (reader->swizzle    [0] != 0 && strcmp(reader->swizzle,     "rgba") != 0) return ktx2_result_not_gltf_conformant;
	if (reader->orientation[0] != 0 && strcmp(reader->orientation, "rd"  ) != 0) return ktx2_result_not_gltf_conformant;

	return ktx2_result_success;
}
