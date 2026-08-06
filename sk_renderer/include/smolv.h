// SMOL-V, ported to C11 from smol-v by Aras Pranckevicius
// (https://github.com/aras-p/smol-v), MIT / public domain.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Rewrites SPIR-V smaller, and much friendlier to a general purpose compressor.
// Lossless apart from optional debug stripping.
// See https://aras-p.info/blog/2016/09/01/SPIR-V-Compression/
//
// The body is split into one stream per field kind rather than interleaved per
// instruction, which is worth ~6% off the zipped size but makes these blobs
// incompatible with upstream despite the shared "SMOL" magic.
//
// Blobs carry no version. They only live inside a .sks, so SKSC_FILE_VERSION
// already gates the encoding and anything changing the byte stream bumps that.
//
// Also unlike upstream: neither direction allocates, decoding bounds-checks
// every word it writes, and the stats API is gone.

typedef enum {
	smolv_encode_none             = 0,
	// Strips OpName/OpLine/OpSource and friends, so the decode is smaller than
	// the module that went in rather than identical to it.
	smolv_encode_strip_debug_info = 1 << 0,
} smolv_encode_;

///////////////////////////////////////////////////////////////////////////////

// Header sniff only, the body is not validated.
bool   smolv_is_smolv     (const void *data, size_t size);

// Worst-case encoded size. Varints can grow a word from 4 bytes to 5, so a
// pathological module encodes larger than it started.
size_t smolv_encode_bound (size_t spirv_size);

// Encodes into out_smolv, which needs smolv_encode_bound() bytes. spirv must be
// 4-byte aligned. Allocates nothing; false leaves partial output.
bool   smolv_encode       (const void *spirv, size_t spirv_size, void *out_smolv, size_t out_capacity, size_t *out_size, uint32_t flags);

// Buffer size smolv_decode() needs, or 0 if this isn't valid SMOL-V.
size_t smolv_decoded_size (const void *smolv_data, size_t smolv_size);

// out_spirv must be 4-byte aligned and smolv_decoded_size() bytes. Allocates
// nothing; false leaves partial output.
bool   smolv_decode       (const void *smolv_data, size_t smolv_size, void *out_spirv, size_t out_capacity);

#ifdef __cplusplus
}
#endif
