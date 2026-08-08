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

static void synth_build(uint8_t* out_buffer) {
	static const uint8_t identifier[12] = {
		0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
	};
	memset(out_buffer, 0, SYNTH_BYTES);
	memcpy(out_buffer, identifier, sizeof(identifier));

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
