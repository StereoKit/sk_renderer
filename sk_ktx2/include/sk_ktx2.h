// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// sk_ktx2 - a KTX2 / Basis Universal transcoder in C11.
//
// Scope is glTF's KHR_texture_basisu: ETC1S and UASTC LDR 4x4, 2D only. Source
// blocks go straight to the destination block format, never through pixels, so
// there is no renderer dependency and no GPU.
//
//   static ktx2_context_t ctx = {0};  // ~43 KB, one per process; see below
//   ktx2_reader_t  r;
//   ktx2_plan_t    plan;
//   if (ktx2_open(bytes, size, &r) != ktx2_result_success) return;
//   if (ktx2_plan(&r, &ctx, caps, &plan) != ktx2_result_success) return;
//   void* data = malloc(plan.data_bytes);
//   ktx2_transcode(&plan, data, plan.data_bytes, NULL);
//
// The caller states what the hardware samples; the library picks the format.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
	#ifdef KTX2_BUILD_SHARED
		#define KTX2_API __declspec(dllexport)
	#else
		#define KTX2_API
	#endif
#elif defined(__GNUC__) || defined(__clang__)
	#define KTX2_API __attribute__((visibility("default")))
#else
	#define KTX2_API
#endif

// Keeps ktx2_reader_t a stack-friendly POD. A 16384 texture has 15 levels, so
// this has headroom; files claiming more are rejected, never truncated.
#define KTX2_MAX_LEVELS 24

// Bounds every `blocks * blocks * bytes` product downstream, so all of it can be
// plain size_t arithmetic. At 16384 the worst case - a 4 byte-per-texel decode -
// is 1 GiB, so the mip chain still fits a 32-bit size_t, which wasm32 has.
#define KTX2_MAX_DIMENSION 16384

///////////////////////////////////////////////////////////////////////////////

typedef enum ktx2_result_ {
	ktx2_result_success = 0,
	ktx2_result_not_ktx2,            // bad identifier
	ktx2_result_corrupt,             // bad offsets, truncated bitstream
	ktx2_result_unsupported,         // valid KTX2, but an encoding we don't implement
	ktx2_result_not_gltf_conformant, // we could decode it, but KHR_texture_basisu forbids it
	ktx2_result_no_target,           // caps offers nothing reachable
	ktx2_result_buffer_too_small,
	ktx2_result_no_decompressor,     // supercompressed, and the context supplies no inflate
} ktx2_result_;

// Granularity matches how hardware reports capability, which is why ASTC splits
// and BC does not: textureCompressionBC covers BC1-BC7 including BC6H in one
// bit, while ASTC HDR is a separate extension that WebGPU does not have at all.
typedef enum ktx2_caps_ {
	ktx2_caps_none     = 0,
	ktx2_caps_bc       = 1 << 0,
	ktx2_caps_etc2     = 1 << 1,
	ktx2_caps_astc_ldr = 1 << 2,
	ktx2_caps_astc_hdr = 1 << 3,
} ktx2_caps_;

// Append-only. The unimplemented ones are named so input reports something
// useful, and so adding them stays additive.
typedef enum ktx2_source_ {
	ktx2_source_unknown = 0,
	ktx2_source_etc1s,          // implemented
	ktx2_source_uastc_ldr_4x4,  // implemented
	ktx2_source_astc,           // standard ASTC blocks, upload as-is
	ktx2_source_uastc_hdr_4x4,  // standard ASTC HDR blocks, upload as-is
	ktx2_source_uastc_hdr_6x6i, // custom intermediate, needs a decoder
} ktx2_source_;

// From the DFD channel IDs and nowhere else: ETC1S RGBA and RG both store two
// slices, and only the IDs say whether the second one is alpha or green.
typedef enum ktx2_channels_ {
	ktx2_channels_rgb = 0,
	ktx2_channels_rgba,
	ktx2_channels_r,
	ktx2_channels_rg,
} ktx2_channels_;

// Output formats. Append-only; HDR targets slot in at the end when they arrive.
typedef enum ktx2_fmt_ {
	ktx2_fmt_none = 0,
	ktx2_fmt_etc1_rgb,     ktx2_fmt_etc1_rgb_srgb,
	ktx2_fmt_etc2_rgba,    ktx2_fmt_etc2_rgba_srgb,
	ktx2_fmt_eac_r11,      ktx2_fmt_eac_rg11,
	ktx2_fmt_bc1_rgb,      ktx2_fmt_bc1_rgb_srgb,
	ktx2_fmt_bc3_rgba,     ktx2_fmt_bc3_rgba_srgb,
	ktx2_fmt_bc4_r,        ktx2_fmt_bc5_rg,
	ktx2_fmt_bc7_rgba,     ktx2_fmt_bc7_rgba_srgb,
	ktx2_fmt_astc4x4_rgba, ktx2_fmt_astc4x4_rgba_srgb,
	ktx2_fmt_r8,           ktx2_fmt_rg8,
	ktx2_fmt_rgba32,       ktx2_fmt_rgba32_srgb,
} ktx2_fmt_;

// What ktx2_open found. It parses arrays, cubemaps and the HDR models, and
// ktx2_plan is what declines them, so these fields report more than we transcode.
typedef struct ktx2_info_t {
	int32_t        width;
	int32_t        height;
	int32_t        mip_count;
	int32_t        layer_count;
	int32_t        face_count;
	ktx2_source_   source;
	ktx2_channels_ channels;
	bool           is_srgb;
	bool           is_hdr;
} ktx2_info_t;

///////////////////////////////////////////////////////////////////////////////

typedef struct ktx2_level_t {
	uint64_t offset;             // byte offset into the caller's buffer
	uint64_t bytes;              // as stored, still supercompressed
	uint64_t uncompressed_bytes; // == bytes when supercompression is none
} ktx2_level_t;

// Parsed container. POD, no ownership: it points into the caller's buffer, which
// must outlive it. Fields are internal; ktx2_get_info is the supported view.
typedef struct ktx2_reader_t {
	const uint8_t* data;
	size_t         bytes;

	uint32_t vk_format;
	uint32_t type_size;
	uint32_t width;
	uint32_t height;
	uint32_t depth;            // 0 for 2D
	uint32_t layer_count;      // 0 for non-array
	uint32_t face_count;       // 1, or 6 for cubemaps
	uint32_t level_count;      // 0 asks the loader to generate mips
	uint32_t supercompression; // 0 none, 1 BasisLZ, 2 Zstd, 3 ZLIB

	uint32_t     level_stored;  // max(1, level_count), the real entry count
	ktx2_level_t levels[KTX2_MAX_LEVELS];

	uint8_t color_model;     // KHR_DF_MODEL_*, drives source dispatch
	uint8_t color_primaries;
	uint8_t transfer_fn;     // 1 linear, 2 sRGB
	uint8_t dfd_flags;       // bit 0 is KHR_DF_FLAG_ALPHA_PREMULTIPLIED
	uint8_t sample_count;    // 1 or 2 for the formats we accept
	uint8_t channel_id[2];   // DFD channel IDs, see ktx2_channels_
	bool    channels_known;  // false when channel_id is a set we don't recognize

	char swizzle    [8]; // KTXswizzle, "" when absent
	char orientation[8]; // KTXorientation, "" when absent

	ktx2_source_   source;
	ktx2_channels_ channels;

	// BasisLZ supercompression global data, present when supercompression == 1
	const uint8_t* sgd_endpoints;
	const uint8_t* sgd_selectors;
	const uint8_t* sgd_tables;
	const uint8_t* sgd_image_descs; // 20 bytes each, one per image
	uint32_t       sgd_endpoints_bytes;
	uint32_t       sgd_selectors_bytes;
	uint32_t       sgd_tables_bytes;
	uint32_t       sgd_image_count;
	uint16_t       endpoint_count;
	uint16_t       selector_count;
} ktx2_reader_t;

///////////////////////////////////////////////////////////////////////////////
// Host services. This library vendors no compression code, so Zstd
// supercompressed UASTC borrows the engine's decompressor.
//
// Each mip level is an independent frame of known size, so the contract needs
// no streaming and no allocation: one call, whole level. Returns bytes written,
// or 0 on failure.

typedef size_t (*ktx2_inflate_fn)(void* context, const void* src, size_t src_bytes,
                                  void* out_dst, size_t dst_bytes);

// Storage for the ETC1S conversion tables (EAC, BC1, BC3, BC4, BC5). Identical
// for every file, and building one costs more than transcoding a small texture.
#define KTX2_TABLE_BYTES (43 * 1024)

// Host services and cached tables, owned by the caller. `ktx2_context_t ctx =
// {0}` is the whole setup: tables fill on first use, and `zstd` stays NULL
// unless Zstd content shows up.
//
// Give it static or heap storage - it is ~43 KB - and keep one per process. One
// per load would rebuild the tables every time and defeat the point.
//
// Filling it is not thread safe. Transcode on one thread, give each thread its
// own, or call ktx2_context_prepare up front and then share it read-only.
typedef struct ktx2_context_t {
	ktx2_inflate_fn zstd;         // required for supercompressionScheme 2
	void*           zstd_context; // passed back untouched
	uint32_t        tables_built; // which tables are live; zero means none yet
	uint8_t         tables[KTX2_TABLE_BYTES];
} ktx2_context_t;

typedef struct ktx2_plan_t {
	const ktx2_reader_t*  reader;        // the plan knows its source
	ktx2_context_t*       context;       // and the services it was planned against
	ktx2_fmt_             format;        // chosen for you
	int32_t               mip_count;
	size_t                data_bytes;    // exact output size
	size_t                scratch_bytes; // 0 if none needed
} ktx2_plan_t;

///////////////////////////////////////////////////////////////////////////////

KTX2_API const char* ktx2_result_str(ktx2_result_ result);
KTX2_API const char* ktx2_fmt_str   (ktx2_fmt_    format);
KTX2_API const char* ktx2_source_str(ktx2_source_ source);

// Block footprint of an output format. 1x1 for the uncompressed formats.
KTX2_API void ktx2_fmt_block(ktx2_fmt_ format, int32_t* out_width, int32_t* out_height, int32_t* out_bytes);

// Parse and validate: no codebook decode, no bitstream walk. Rejects only what
// cannot be decoded at all. `data` must outlive `out_reader`.
KTX2_API ktx2_result_ ktx2_open    (const void* data, size_t bytes, ktx2_reader_t* out_reader);
KTX2_API ktx2_info_t  ktx2_get_info(const ktx2_reader_t* reader);

// KHR_texture_basisu's policy, on top of a reader that already parsed: 2D only,
// 4-aligned dimensions, permitted supercompression, rgba swizzle, rd
// orientation. Callers outside glTF simply don't call it.
KTX2_API ktx2_result_ ktx2_check_gltf_basisu(const ktx2_reader_t* reader);

// Picks the best format `caps` can reach and reports the exact buffer size. The
// only place in this API where a decision is made.
//
// `context` is stored in the plan and reused by ktx2_transcode, so a source
// needing a decompressor is refused here rather than mid-transcode. Required
// rather than optional: an argument that may be NULL is one a caller forgets,
// and then it only fails on the content that needs it.
//
// The sizes are bounded but not small, and a hostile file can be tiny while
// asking for both: a 362-byte ETC1S file naming 16384x16384 plans to 128 MB of
// output and 64 MB of scratch, and only fails once the decode runs out of bits.
// That is reported rather than acted on, so a caller loading untrusted content
// should compare `data_bytes` against what it is willing to spend before
// allocating - and pass its own `opt_scratch` to ktx2_transcode, which is
// otherwise the one allocation it never sees.
KTX2_API ktx2_result_ ktx2_plan(const ktx2_reader_t* reader, ktx2_context_t* context,
                                ktx2_caps_ caps, ktx2_plan_t* out_plan);

// Builds every cached table up front. Optional, but it keeps the cost off the
// first texture load and makes the context shareable across threads after.
KTX2_API void ktx2_context_prepare(ktx2_context_t* ref_context);

// The complete mip chain in one pass, mip 0 first and tightly packed - exactly
// skr_tex_data_t's layout. Levels are read in the file's native smallest-first
// order internally. opt_scratch may be NULL to allocate internally.
KTX2_API ktx2_result_ ktx2_transcode(const ktx2_plan_t* plan, void* out_data, size_t out_bytes, void* opt_scratch);

#ifdef __cplusplus
}
#endif
