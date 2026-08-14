// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Self-contained validation tests: they synthesize their own KTX2 files, so they
// run with no corpus present. They also cover what a fixed corpus covers badly -
// truncation at every offset, and one field wrong at a time.

#include "sk_ktx2.h"
#include "host_zstd.h"
#include "synthetic_ktx2.h"
#ifdef SK_KTX2_UASTC
	#include "uastc_vectors.h"
#endif

#include <stdio.h>
#include <string.h>

static int32_t g_failures = 0;

#define CHECK(cond, ...) do { \
	if (!(cond)) { g_failures++; printf("FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

static void test_baseline(void) {
	uint8_t       buffer[SYNTH_BYTES];
	ktx2_reader_t reader;
	synth_build(buffer);

	ktx2_result_ result = ktx2_open(buffer, SYNTH_BYTES, &reader);
	CHECK(result == ktx2_result_success, "baseline should open, got '%s'", ktx2_result_str(result));
	if (result != ktx2_result_success) return;

	ktx2_info_t info = ktx2_get_info(&reader);
	CHECK(info.width == 4 && info.height == 4, "expected 4x4, got %dx%d", info.width, info.height);
	CHECK(info.mip_count == 1,                 "expected 1 mip, got %d", info.mip_count);
	CHECK(info.source   == ktx2_source_etc1s,  "expected ETC1S, got %s", ktx2_source_str(info.source));
	CHECK(info.channels == ktx2_channels_rgb,  "expected RGB channels");
	CHECK(info.is_srgb,                        "expected sRGB");
	CHECK(!info.is_hdr,                        "expected LDR");

	result = ktx2_check_gltf_basisu(&reader);
	CHECK(result == ktx2_result_success, "baseline should be conformant, got '%s'", ktx2_result_str(result));

	ktx2_plan_t plan;
	result = ktx2_plan(&reader, &k_host_context, ktx2_caps_etc2, &plan);
	CHECK(result == ktx2_result_success,             "plan should succeed, got '%s'", ktx2_result_str(result));
	CHECK(plan.format     == ktx2_fmt_etc1_rgb_srgb, "expected etc1_rgb_srgb, got %s", ktx2_fmt_str(plan.format));
	CHECK(plan.data_bytes == 8,                      "expected 8 bytes, got %zu", plan.data_bytes);

	// Same source, different hardware, different answer - and no caller input.
	result = ktx2_plan(&reader, &k_host_context, ktx2_caps_bc, &plan);
	CHECK(result == ktx2_result_success && plan.format == ktx2_fmt_bc1_rgb_srgb,
		"BC-only should pick bc1_rgb_srgb, got %s", ktx2_fmt_str(plan.format));

	// The size the caller was handed is a promise: a buffer short of it is
	// refused outright, before any of the file is decoded.
	uint8_t small[8];
	CHECK(ktx2_transcode(&plan, small, plan.data_bytes - 1, NULL) == ktx2_result_buffer_too_small,
		"an undersized output buffer should be refused");
}

// Both the plan's sizing and the caller's upload stride come from this, and out
// of range has to land somewhere defined rather than reading past the table.
static void test_format_table(void) {
	for (int32_t format = 0; format <= ktx2_fmt_rgba32_srgb; format++) {
		int32_t width = 0, height = 0, bytes = 0;
		ktx2_fmt_block((ktx2_fmt_)format, &width, &height, &bytes);
		CHECK(strcmp(ktx2_fmt_str((ktx2_fmt_)format), "invalid") != 0, "format %d has no name", format);
		if (format == ktx2_fmt_none) continue;
		CHECK(width  == 1 || width  == 4, "format %s has block width %d",  ktx2_fmt_str((ktx2_fmt_)format), width);
		CHECK(height == width,            "format %s is not square",       ktx2_fmt_str((ktx2_fmt_)format));
		CHECK(bytes  >  0,                "format %s has no block size",   ktx2_fmt_str((ktx2_fmt_)format));
	}

	int32_t width = 9, height = 9, bytes = 9;
	ktx2_fmt_block((ktx2_fmt_)9999, &width, &height, &bytes);
	CHECK(width == 0 && height == 0 && bytes == 0, "an unknown format should describe nothing");
	CHECK(strcmp(ktx2_fmt_str((ktx2_fmt_)9999), "invalid") == 0, "an unknown format should name itself invalid");
}

// The tables build on first use anyway, so prepare only moves the cost - and it
// has to be idempotent, since a caller may call it once per load.
static void test_context_prepare(void) {
	static ktx2_context_t context;
	memset(&context, 0, sizeof(context));

	CHECK(context.tables_built == 0, "a zeroed context should claim no tables");
	ktx2_context_prepare(&context);
	uint32_t built = context.tables_built;
	CHECK(built != 0, "prepare should build the tables");
	ktx2_context_prepare(&context);
	CHECK(context.tables_built == built, "prepare should be idempotent");
}

typedef struct mutation_t {
	const char*  name;
	size_t       at;
	int32_t      width;    // 1, 4, or 8 byte write
	uint64_t     value;
	ktx2_result_ expect_open;
	ktx2_result_ expect_gltf; // only consulted when expect_open is success
} mutation_t;

static void test_mutations(void) {
	static const mutation_t mutations[] = {
		// Structural - ktx2_open must catch these.
		{ "zero width",            20,           4, 0,      ktx2_result_corrupt,     0 },
		{ "faceCount 2",           36,           4, 2,      ktx2_result_corrupt,     0 },
		{ "levelCount 99",         40,           4, 99,     ktx2_result_unsupported, 0 },
		{ "levelCount past full",  40,           4, 4,      ktx2_result_corrupt,     0 },
		{ "supercompression 9",    44,           4, 9,      ktx2_result_unsupported, 0 },
		{ "layerCount past cap",   32,           4, 100000, ktx2_result_unsupported, 0 },
		// Without the cap, the scratch sizing and the slice decoder agree on a
		// wrapped figure and the decode writes past its buffer.
		{ "width past the cap",    20,           4, 70000,  ktx2_result_unsupported, 0 },
		{ "height past the cap",   24,           4, 1u<<20, ktx2_result_unsupported, 0 },
		{ "dfd offset past end",   48,           4, 100000, ktx2_result_corrupt,     0 },
		{ "level offset past end", SYNTH_LEVEL_INDEX,     8, 100000, ktx2_result_corrupt, 0 },
		{ "level length past end", SYNTH_LEVEL_INDEX + 8, 8, 100000, ktx2_result_corrupt, 0 },
		{ "zero level length",     SYNTH_LEVEL_INDEX + 8, 8, 0,      ktx2_result_corrupt, 0 },
		{ "sgd length mismatch",   72,           8, 48,     ktx2_result_corrupt,     0 },
		{ "slice past level",      SYNTH_SGD + 28, 4, 99,     ktx2_result_corrupt,     0 },
		{ "unknown colour model",  SYNTH_DFD + 12, 4, 1,      ktx2_result_unsupported, 0 },
		{ "vkFormat set on ETC1S", 12,           4, 37,     ktx2_result_corrupt,     0 },
		{ "dfd block too small",   SYNTH_DFD + 8,  4, 16u<<16,ktx2_result_corrupt,     0 },

		// Decodable, but the extension says no - these must reach the policy
		// check, not be rejected by the parser.
		{ "3D texture",            28, 4, 1, ktx2_result_success, ktx2_result_not_gltf_conformant },
		{ "array texture",         32, 4, 1, ktx2_result_success, ktx2_result_not_gltf_conformant },
		{ "linear + BT709",        SYNTH_DFD + 14, 4, 1, ktx2_result_success, ktx2_result_not_gltf_conformant },
		{ "ETC1S AAA alone",       SYNTH_DFD + 31, 1, 15, ktx2_result_success, ktx2_result_not_gltf_conformant },
		{ "ETC1S RRR is R",        SYNTH_DFD + 31, 1, 3,  ktx2_result_success, ktx2_result_success },
	};

	for (size_t i = 0; i < sizeof(mutations) / sizeof(mutations[0]); i++) {
		const mutation_t* mutation = &mutations[i];
		uint8_t           buffer[SYNTH_BYTES];
		ktx2_reader_t     reader;
		synth_build(buffer);
		if      (mutation->width == 8) synth_u64(buffer, mutation->at, mutation->value);
		else if (mutation->width == 4) synth_u32(buffer, mutation->at, (uint32_t)mutation->value);
		else                           buffer[mutation->at] = (uint8_t)mutation->value;

		ktx2_result_ result = ktx2_open(buffer, SYNTH_BYTES, &reader);
		CHECK(result == mutation->expect_open, "%s: expected '%s' from open, got '%s'",
			mutation->name, ktx2_result_str(mutation->expect_open), ktx2_result_str(result));

		if (result == ktx2_result_success && mutation->expect_open == ktx2_result_success) {
			result = ktx2_check_gltf_basisu(&reader);
			CHECK(result == mutation->expect_gltf, "%s: expected '%s' from gltf check, got '%s'",
				mutation->name, ktx2_result_str(mutation->expect_gltf), ktx2_result_str(result));
		}
	}
}

// Arrays and cubemaps: the container parses them, the plan sizes them per image,
// and glTF policy still refuses them. The slices here are zeroes, so the
// transcode is only required to fail cleanly, never to produce pixels.
static void test_array_cubemap(void) {
	static const struct { uint32_t layers, faces; } shapes[] = { {2, 1}, {0, 6}, {2, 6} };

	for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
		uint8_t       buffer[SYNTH_IMAGES_MAX_BYTES];
		ktx2_reader_t reader;
		size_t        bytes  = synth_build_images(buffer, shapes[i].layers, shapes[i].faces);
		int32_t       layers = (int32_t)(shapes[i].layers ? shapes[i].layers : 1);
		size_t        images = (size_t)layers * shapes[i].faces;

		ktx2_result_ result = ktx2_open(buffer, bytes, &reader);
		CHECK(result == ktx2_result_success, "%ux%u images should open, got '%s'",
			shapes[i].layers, shapes[i].faces, ktx2_result_str(result));
		if (result != ktx2_result_success) continue;

		ktx2_info_t info = ktx2_get_info(&reader);
		CHECK(info.layer_count == layers,                "expected %d layers, got %d", layers, info.layer_count);
		CHECK(info.face_count  == (int32_t)shapes[i].faces, "expected %u faces, got %d", shapes[i].faces, info.face_count);

		CHECK(ktx2_check_gltf_basisu(&reader) == ktx2_result_not_gltf_conformant,
			"KHR_texture_basisu forbids arrays and cubemaps");

		ktx2_plan_t plan;
		result = ktx2_plan(&reader, &k_host_context, ktx2_caps_etc2, &plan);
		CHECK(result == ktx2_result_success,        "plan should accept it, got '%s'", ktx2_result_str(result));
		CHECK(plan.data_bytes == images * 8,        "expected %zu bytes, got %zu", images * 8, plan.data_bytes);

		uint8_t output[12 * 8];
		result = ktx2_transcode(&plan, output, plan.data_bytes, NULL);
		CHECK(result != ktx2_result_success, "zeroed codebooks should not decode");
	}

	// A cubemap with depth is nonsense the spec forbids outright.
	uint8_t       buffer[SYNTH_IMAGES_MAX_BYTES];
	ktx2_reader_t reader;
	size_t        bytes = synth_build_images(buffer, 0, 6);
	synth_u32(buffer, 28, 1); // pixelDepth on a cubemap
	CHECK(ktx2_open(buffer, bytes, &reader) == ktx2_result_corrupt, "a cubemap with depth should be corrupt");
}

#ifdef SK_KTX2_UASTC
// Two layers carrying two different spec blocks. UASTC needs no codebooks, so
// this transcode succeeds outright, and differing outputs prove each image was
// read from its own offset rather than the first image repeated.
static void test_uastc_array(void) {
	static const uint8_t blocks[2][16] = {
		{ 0x08, 0x18, 0x43, 0x67, 0x57, 0x4F, 0xB0, 0x8A, 0x5E, 0xB4, 0x62, 0x35, 0x42, 0xE2, 0x0E, 0x22 },
		{ 0xF1, 0xF0, 0x18, 0x8F, 0xE4, 0x6B, 0x3C, 0x3A, 0x90, 0xFB, 0x89, 0x5D, 0x56, 0x82, 0x9B, 0x6C },
	};
	uint8_t       buffer[SYNTH_UASTC_MAX_BYTES];
	ktx2_reader_t reader;
	size_t        bytes = synth_build_uastc(buffer, 4, 2, 1, 1, blocks[0]);

	ktx2_result_ result = ktx2_open(buffer, bytes, &reader);
	CHECK(result == ktx2_result_success, "UASTC array should open, got '%s'", ktx2_result_str(result));
	if (result != ktx2_result_success) return;

	ktx2_info_t info = ktx2_get_info(&reader);
	CHECK(info.source      == ktx2_source_uastc_ldr_4x4, "expected UASTC, got %s", ktx2_source_str(info.source));
	CHECK(info.layer_count == 2,                         "expected 2 layers, got %d", info.layer_count);

	ktx2_plan_t plan;
	result = ktx2_plan(&reader, &k_host_context, ktx2_caps_astc_ldr, &plan);
	CHECK(result == ktx2_result_success, "plan should accept it, got '%s'", ktx2_result_str(result));
	if (result != ktx2_result_success) return;
	CHECK(plan.data_bytes == 32, "expected 32 bytes, got %zu", plan.data_bytes);

	uint8_t output[32];
	result = ktx2_transcode(&plan, output, sizeof(output), NULL);
	CHECK(result == ktx2_result_success, "transcode should succeed, got '%s'", ktx2_result_str(result));
	CHECK(memcmp(output, output + 16, 16) != 0, "distinct layers decoded to identical blocks");
}

// The stride case where an indexing bug would hide: mips and multiple images at
// once, every image distinct spec blocks. Proven by reference rather than
// goldens: each image sliced from the array output must equal the same blocks
// transcoded as a plain 2D mip chain.
static void test_uastc_mipped_images(void) {
	// 8x8, two layers of six faces, two levels. Mip 0 is four blocks per image
	// (vectors 0..47), mip 1 one (vectors 48..59), level-major layer-then-face.
	uint8_t       buffer[SYNTH_UASTC_MAX_BYTES];
	ktx2_reader_t reader;
	size_t        bytes = synth_build_uastc(buffer, 8, 2, 6, 2, k_uastc_vector_blocks[0]);
	CHECK(bytes != 0, "the mipped array should fit the builder");

	ktx2_result_ result = ktx2_open(buffer, bytes, &reader);
	CHECK(result == ktx2_result_success, "mipped array should open, got '%s'", ktx2_result_str(result));
	if (result != ktx2_result_success) return;

	ktx2_plan_t plan;
	result = ktx2_plan(&reader, &k_host_context, ktx2_caps_astc_ldr, &plan);
	CHECK(result == ktx2_result_success, "plan should accept it, got '%s'", ktx2_result_str(result));
	if (result != ktx2_result_success) return;
	CHECK(plan.data_bytes == 12 * 64 + 12 * 16, "expected 960 bytes, got %zu", plan.data_bytes);

	uint8_t array_out[12 * 64 + 12 * 16];
	result = ktx2_transcode(&plan, array_out, sizeof(array_out), NULL);
	CHECK(result == ktx2_result_success, "transcode should succeed, got '%s'", ktx2_result_str(result));
	if (result != ktx2_result_success) return;

	for (int32_t image = 0; image < 12; image++) {
		uint8_t flat_blocks[5 * 16];
		memcpy(flat_blocks,      k_uastc_vector_blocks[image * 4], 4 * 16);
		memcpy(flat_blocks + 64, k_uastc_vector_blocks[48 + image],    16);

		uint8_t       flat_file[SYNTH_UASTC_MAX_BYTES];
		uint8_t       flat_out [64 + 16];
		ktx2_reader_t flat_reader;
		ktx2_plan_t   flat_plan;
		size_t        flat_bytes = synth_build_uastc(flat_file, 8, 0, 1, 2, flat_blocks);
		if (ktx2_open(flat_file, flat_bytes, &flat_reader)                            != ktx2_result_success ||
		    ktx2_plan(&flat_reader, &k_host_context, ktx2_caps_astc_ldr, &flat_plan)  != ktx2_result_success ||
		    ktx2_transcode(&flat_plan, flat_out, sizeof(flat_out), NULL)              != ktx2_result_success) {
			CHECK(false, "2D reference for image %d failed", image);
			continue;
		}
		CHECK(memcmp(array_out + image * 64,           flat_out,      64) == 0, "image %d mip 0 misplaced", image);
		CHECK(memcmp(array_out + 12 * 64 + image * 16, flat_out + 64, 16) == 0, "image %d mip 1 misplaced", image);
	}
}
#endif

// Every prefix must be rejected and none may read out of bounds; run under ASan
// for it to mean anything. Truncation is what a fixed corpus cannot enumerate.
static void test_truncation(void) {
	uint8_t buffer[SYNTH_BYTES];
	synth_build(buffer);

	for (size_t bytes = 0; bytes < SYNTH_BYTES; bytes++) {
		ktx2_reader_t reader;
		ktx2_result_  result = ktx2_open(buffer, bytes, &reader);
		CHECK(result != ktx2_result_success, "truncation to %zu bytes was accepted", bytes);
	}
}

// Not-KTX2 inputs, including the .basis magic we deliberately no longer read.
static void test_rejects_foreign(void) {
	ktx2_reader_t reader;
	uint8_t       basis[] = { 0x73, 0xB3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
	uint8_t       png  [] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };

	CHECK(ktx2_open(basis, sizeof(basis), &reader) == ktx2_result_not_ktx2, "basis magic should be not_ktx2");
	CHECK(ktx2_open(png,   sizeof(png),   &reader) == ktx2_result_not_ktx2, "png should be not_ktx2");
	CHECK(ktx2_open("",    0,             &reader) == ktx2_result_not_ktx2, "empty should be not_ktx2");
}

int main(void) {
	test_baseline();
	test_format_table();
	test_context_prepare();
	test_mutations();
	test_array_cubemap();
#ifdef SK_KTX2_UASTC
	test_uastc_array();
	test_uastc_mipped_images();
#endif
	test_truncation();
	test_rejects_foreign();

	if (g_failures == 0) printf("sk_ktx2 unit tests passed\n");
	else                 printf("%d failure(s)\n", g_failures);
	return g_failures == 0 ? 0 : 1;
}
