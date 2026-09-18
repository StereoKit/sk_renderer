// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// A valid 4x4 ETC1S RGB KTX2 file, built in memory: one level, one image, one
// endpoint, one selector. Written from the spec, not copied from the corpus, so
// it carries no third-party licensing and the tests run with no corpus present.
//
// unit_tests.c mutates one field at a time against it; fuzz_test.c uses it as a
// seed. Both want the offsets, hence a header rather than a static.

#pragma once

#include <stdint.h>
#include <string.h>

#define SYNTH_BYTES       208
#define SYNTH_LEVEL_INDEX  80
#define SYNTH_DFD         104
#define SYNTH_SGD         148
#define SYNTH_LEVEL_DATA  200

static void synth_u32(uint8_t* buffer, size_t at, uint32_t value) {
	buffer[at    ] = (uint8_t) (value        & 0xFF);
	buffer[at + 1] = (uint8_t)((value >>  8) & 0xFF);
	buffer[at + 2] = (uint8_t)((value >> 16) & 0xFF);
	buffer[at + 3] = (uint8_t)((value >> 24) & 0xFF);
}

static void synth_u64(uint8_t* buffer, size_t at, uint64_t value) {
	synth_u32(buffer, at,     (uint32_t) value);
	synth_u32(buffer, at + 4, (uint32_t)(value >> 32));
}

static const uint8_t k_synth_identifier[12] = {
	0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
};

static void synth_build(uint8_t* out_buffer) {
	memset(out_buffer, 0, SYNTH_BYTES);
	memcpy(out_buffer, k_synth_identifier, sizeof(k_synth_identifier));

	synth_u32(out_buffer, 12, 0);   // vkFormat UNDEFINED
	synth_u32(out_buffer, 16, 1);   // typeSize
	synth_u32(out_buffer, 20, 4);   // pixelWidth
	synth_u32(out_buffer, 24, 4);   // pixelHeight
	synth_u32(out_buffer, 28, 0);   // pixelDepth
	synth_u32(out_buffer, 32, 0);   // layerCount
	synth_u32(out_buffer, 36, 1);   // faceCount
	synth_u32(out_buffer, 40, 1);   // levelCount
	synth_u32(out_buffer, 44, 1);   // supercompressionScheme = BasisLZ
	synth_u32(out_buffer, 48, SYNTH_DFD);
	synth_u32(out_buffer, 52, 44);  // dfdByteLength
	synth_u32(out_buffer, 56, 0);   // kvdByteOffset
	synth_u32(out_buffer, 60, 0);   // kvdByteLength
	synth_u64(out_buffer, 64, SYNTH_SGD);
	synth_u64(out_buffer, 72, 49);  // sgdByteLength: 20 header + 20 imageDesc + 3+3+3

	synth_u64(out_buffer, SYNTH_LEVEL_INDEX,      SYNTH_LEVEL_DATA);
	synth_u64(out_buffer, SYNTH_LEVEL_INDEX +  8, 8);  // byteLength
	synth_u64(out_buffer, SYNTH_LEVEL_INDEX + 16, 8);  // uncompressedByteLength

	synth_u32(out_buffer, SYNTH_DFD,      44);          // dfdTotalSize
	synth_u32(out_buffer, SYNTH_DFD +  4, 0);           // vendorId 0, descriptorType 0
	synth_u32(out_buffer, SYNTH_DFD +  8, 40u << 16);   // version 0, descriptorBlockSize 40
	out_buffer[SYNTH_DFD + 12] = 163;                   // colorModel ETC1S
	out_buffer[SYNTH_DFD + 13] = 1;                     // colorPrimaries BT709
	out_buffer[SYNTH_DFD + 14] = 2;                     // transferFunction SRGB
	out_buffer[SYNTH_DFD + 15] = 0;                     // flags
	out_buffer[SYNTH_DFD + 16] = 3;                     // texelBlockDimension0 = 4
	out_buffer[SYNTH_DFD + 17] = 3;                     // texelBlockDimension1 = 4
	out_buffer[SYNTH_DFD + 20] = 8;                     // bytesPlane0
	// Samples begin after the 24-byte block header, so at dfd + 4 + 24.
	out_buffer[SYNTH_DFD + 30] = 63;                    // sample0 bitLength
	out_buffer[SYNTH_DFD + 31] = 0;                     // sample0 channelID = ETC1S RGB
	synth_u32(out_buffer, SYNTH_DFD + 40, 0xFFFFFFFF);  // sample0 sampleUpper

	synth_u32(out_buffer, SYNTH_SGD,      1u | (1u << 16)); // endpointCount 1, selectorCount 1
	synth_u32(out_buffer, SYNTH_SGD +  4, 3);               // endpointsByteLength
	synth_u32(out_buffer, SYNTH_SGD +  8, 3);               // selectorsByteLength
	synth_u32(out_buffer, SYNTH_SGD + 12, 3);               // tablesByteLength
	synth_u32(out_buffer, SYNTH_SGD + 16, 0);               // extendedByteLength
	synth_u32(out_buffer, SYNTH_SGD + 20, 0);               // imageDesc flags
	synth_u32(out_buffer, SYNTH_SGD + 24, 0);               // rgbSliceByteOffset
	synth_u32(out_buffer, SYNTH_SGD + 28, 8);               // rgbSliceByteLength
	synth_u32(out_buffer, SYNTH_SGD + 32, 0);               // alphaSliceByteOffset
	synth_u32(out_buffer, SYNTH_SGD + 36, 0);               // alphaSliceByteLength
}

// Like synth_build, but one level of `layer_count` layers by `face_count` faces.
// Each image gets its own imageDesc, all sharing the same 8-byte slice, which
// the SGD bounds checks permit and which keeps the level small. Returns the file
// size; out_buffer needs SYNTH_IMAGES_MAX_BYTES, or 0 if it would not fit.
#define SYNTH_IMAGES_MAX_BYTES 640

static inline size_t synth_build_images(uint8_t* out_buffer, uint32_t layer_count, uint32_t face_count) {
	uint32_t images    = (layer_count ? layer_count : 1) * face_count;
	uint32_t sgd_bytes = 20 + images * 20 + 9;              // header, imageDescs, 3-byte blobs
	size_t   level_at  = (SYNTH_SGD + sgd_bytes + 7) & ~(size_t)7;
	if (level_at + 8 > SYNTH_IMAGES_MAX_BYTES) return 0;

	memset(out_buffer, 0, SYNTH_IMAGES_MAX_BYTES);
	synth_build(out_buffer);
	synth_u32(out_buffer, 32, layer_count);
	synth_u32(out_buffer, 36, face_count);
	synth_u64(out_buffer, 72, sgd_bytes);
	synth_u64(out_buffer, SYNTH_LEVEL_INDEX, level_at);
	for (uint32_t i = 0; i < images; i++) {
		size_t desc = SYNTH_SGD + 20 + (size_t)i * 20;
		synth_u32(out_buffer, desc + 4, 0);                 // rgbSliceByteOffset
		synth_u32(out_buffer, desc + 8, 8);                 // rgbSliceByteLength
	}
	return level_at + 8;
}

// A UASTC KTX2 file: `size` x `size` with `level_count` levels, no
// supercompression, one 16-byte block per 4x4 of every image. `blocks` supplies
// the payload in the output's own order: level-major, then layer, then face, so
// each image can carry distinct content. Standalone rather than patched over
// synth_build, because a longer level index moves everything after it. Returns
// the file size, or 0 if it would not fit SYNTH_UASTC_MAX_BYTES.
#define SYNTH_UASTC_MAX_BYTES 1152

static inline size_t synth_build_uastc(uint8_t* out_buffer, uint32_t size,
                                       uint32_t layer_count, uint32_t face_count,
                                       uint32_t level_count, const uint8_t* blocks) {
	uint32_t images = (layer_count ? layer_count : 1) * face_count;
	size_t   dfd    = 80 + (size_t)level_count * 24;
	size_t   at     = (dfd + 44 + 7) & ~(size_t)7;

	size_t total = at;
	for (uint32_t level = 0; level < level_count; level++) {
		uint32_t span = ((size >> level ? size >> level : 1) + 3) / 4;
		total += (size_t)span * span * 16 * images;
	}
	if (total > SYNTH_UASTC_MAX_BYTES) return 0;

	memset(out_buffer, 0, SYNTH_UASTC_MAX_BYTES);
	memcpy(out_buffer, k_synth_identifier, sizeof(k_synth_identifier));
	synth_u32(out_buffer, 16, 1);                // typeSize
	synth_u32(out_buffer, 20, size);             // pixelWidth
	synth_u32(out_buffer, 24, size);             // pixelHeight
	synth_u32(out_buffer, 32, layer_count);
	synth_u32(out_buffer, 36, face_count);
	synth_u32(out_buffer, 40, level_count);
	synth_u32(out_buffer, 48, (uint32_t)dfd);
	synth_u32(out_buffer, 52, 44);               // dfdByteLength

	synth_u32(out_buffer, dfd,      44);         // dfdTotalSize
	synth_u32(out_buffer, dfd +  8, 40u << 16);  // version 0, descriptorBlockSize 40
	out_buffer[dfd + 12] = 166;                  // colorModel UASTC
	out_buffer[dfd + 13] = 1;                    // colorPrimaries BT709
	out_buffer[dfd + 14] = 2;                    // transferFunction SRGB
	out_buffer[dfd + 16] = 3;                    // texelBlockDimension0 = 4
	out_buffer[dfd + 17] = 3;                    // texelBlockDimension1 = 4
	out_buffer[dfd + 20] = 16;                   // bytesPlane0
	out_buffer[dfd + 30] = 127;                  // sample0 bitLength
	out_buffer[dfd + 31] = 0;                    // sample0 channelID = UASTC RGB
	synth_u32(out_buffer, dfd + 40, 0xFFFFFFFF); // sample0 sampleUpper

	const uint8_t* src = blocks;
	for (uint32_t level = 0; level < level_count; level++) {
		uint32_t span        = ((size >> level ? size >> level : 1) + 3) / 4;
		size_t   level_bytes = (size_t)span * span * 16 * images;
		synth_u64(out_buffer, 80 + (size_t)level * 24,      at);
		synth_u64(out_buffer, 80 + (size_t)level * 24 +  8, level_bytes);
		synth_u64(out_buffer, 80 + (size_t)level * 24 + 16, level_bytes);
		memcpy(out_buffer + at, src, level_bytes);
		src += level_bytes;
		at  += level_bytes;
	}
	return at;
}
