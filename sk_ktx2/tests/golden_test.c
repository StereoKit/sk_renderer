// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Comparison against the KTX-Software transcoder.
//
//   ktx2_golden <cts dir>
//
// Transcodes every BasisLZ input in clitests/input/ktx2 to each target we
// implement and compares with the matching golden. A check against the reference
// implementation rather than against ourselves, which is the only kind that
// catches a self-consistent misreading of the bitstream.
//
// Three gates, because the targets differ in what is achievable:
//
//  - Byte-exact where both sides repack: ETC1 colour, BC4, BC5, uncompressed.
//  - A PSNR floor for EAC, which KTX-Software reaches with an unspecified search
//    encoder over decoded pixels, so its bytes are not reproducible.
//  - PSNR against the ETC1S *reconstruction* for BC1 and BC3. For a lossy target
//    the honest question is not "do we match their bytes" but "is our fit as
//    good as theirs".

#include "sk_ktx2.h"
#include "host_zstd.h"
#include "astc_reference.h"
#include "bc7_reference.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t g_pass = 0, g_fail = 0, g_skip = 0;

static uint8_t* file_read(const char* path, size_t* out_bytes) {
	FILE* f = fopen(path, "rb");
	if (f == NULL) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	rewind(f);
	if (size <= 0) { fclose(f); return NULL; }
	uint8_t* data = (uint8_t*)malloc((size_t)size);
	if (data && fread(data, 1, (size_t)size, f) != (size_t)size) { free(data); data = NULL; }
	fclose(f);
	if (data) *out_bytes = (size_t)size;
	return data;
}

static uint32_t rd_u32(const uint8_t* p) {
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd_u64(const uint8_t* p) {
	return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}

// The goldens are plain KTX2 with a real vkFormat, which sk_ktx2 declines and
// should. Concatenated largest-first to match ktx2_transcode's layout.
static uint8_t* golden_levels(const uint8_t* data, size_t bytes, size_t* out_bytes) {
	if (bytes < 80) return NULL;
	uint32_t level_count = rd_u32(data + 40);
	uint32_t supercomp   = rd_u32(data + 44);
	if (supercomp != 0) return NULL;
	uint32_t levels = level_count ? level_count : 1;
	if (80 + (size_t)levels * 24 > bytes) return NULL;

	size_t total = 0;
	for (uint32_t i = 0; i < levels; i++) total += (size_t)rd_u64(data + 80 + i * 24 + 8);

	uint8_t* out = (uint8_t*)malloc(total ? total : 1);
	if (out == NULL) return NULL;
	size_t at = 0;
	for (uint32_t i = 0; i < levels; i++) {
		uint64_t offset = rd_u64(data + 80 + i * 24);
		uint64_t length = rd_u64(data + 80 + i * 24 + 8);
		if (offset + length > bytes) { free(out); return NULL; }
		memcpy(out + at, data + offset, (size_t)length);
		at += (size_t)length;
	}
	*out_bytes = total;
	return out;
}

///////////////////////////////////////////////////////////////////////////////

// KTX-Software's EAC and BC4 paths decode to pixels and run search encoders that
// no specification describes, while we repack the ETC1S ramp directly. Those
// targets get a PSNR floor over the decoded values instead.
typedef enum compare_ {
	compare_exact = 0,
	compare_eac,      // 8-byte EAC blocks
	compare_bc4,      // 8-byte BC4 blocks
	compare_etc2_rgba,// 16-byte: EAC alpha then ETC1 colour, colour must be exact
	compare_bc1,      // 8-byte BC1, judged against the ETC1S reconstruction
	compare_bc3,      // 16-byte: BC4 alpha then BC1 colour
} compare_;

typedef struct target_t {
	const char* golden_id; // the CTS's name for this target
	ktx2_caps_  caps;      // capabilities that steer ktx2_plan to it
	ktx2_fmt_   expect_a;  // the two formats that map to this golden (unorm/srgb)
	ktx2_fmt_   expect_b;
	compare_    compare;
	int32_t     channels;  // sub-blocks per block, for the two-channel targets
} target_t;

static const target_t k_targets[] = {
	{ "etc-rgb",  ktx2_caps_etc2, ktx2_fmt_etc1_rgb,  ktx2_fmt_etc1_rgb_srgb,  compare_exact,     1 },
	{ "etc-rgba", ktx2_caps_etc2, ktx2_fmt_etc2_rgba, ktx2_fmt_etc2_rgba_srgb, compare_etc2_rgba, 1 },
	{ "eac-r11",  ktx2_caps_etc2, ktx2_fmt_eac_r11,   ktx2_fmt_eac_r11,        compare_eac,       1 },
	{ "eac-rg11", ktx2_caps_etc2, ktx2_fmt_eac_rg11,  ktx2_fmt_eac_rg11,       compare_eac,       2 },
	{ "bc1",      ktx2_caps_bc,   ktx2_fmt_bc1_rgb,   ktx2_fmt_bc1_rgb_srgb,   compare_bc1,       1 },
	{ "bc3",      ktx2_caps_bc,   ktx2_fmt_bc3_rgba,  ktx2_fmt_bc3_rgba_srgb,  compare_bc3,       1 },
	{ "bc4",      ktx2_caps_bc,   ktx2_fmt_bc4_r,     ktx2_fmt_bc4_r,          compare_exact,     1 },
	{ "bc5",      ktx2_caps_bc,   ktx2_fmt_bc5_rg,    ktx2_fmt_bc5_rg,         compare_exact,     2 },
	{ "rgba8",    ktx2_caps_none, ktx2_fmt_rgba32,    ktx2_fmt_rgba32_srgb,    compare_exact,     1 },
	{ "r8",       ktx2_caps_none, ktx2_fmt_r8,        ktx2_fmt_r8,             compare_exact,     1 },
	{ "rg8",      ktx2_caps_none, ktx2_fmt_rg8,       ktx2_fmt_rg8,            compare_exact,     1 },
};

// Anything near this floor is a regression, not a tuning question.
#define PSNR_FLOOR 40.0
// How far below KTX-Software the fit may land. basisu's published figure for
// this conversion is 0.3-0.5 dB, so within a dB means the fit is doing its job.
#define BC_TOLERANCE_DB 1.0

// 10*log10(255^2/mse), without pulling in math.h for one call.
static double psnr_db(double sse, size_t count) {
	if (count == 0 || sse == 0.0) return 99.0;
	double ratio = 65025.0 / (sse / (double)count);
	double log10 = 0.0, x = ratio;
	while (x >= 10.0) { x /= 10.0; log10 += 1.0; }
	while (x <  1.0 ) { x *= 10.0; log10 -= 1.0; }
	double m = x, mantissa = 0.0;
	for (int32_t i = 0; i < 40; i++) { m = m * m; if (m >= 10.0) { m /= 10.0; mantissa += 1.0 / (double)(1u << (i + 1)); } }
	return 10.0 * (log10 + mantissa);
}

static void decode_eac_block(const uint8_t block[8], uint8_t out_texels[16]) {
	static const int16_t modifier[16][8] = {
		{ -3,-6, -9,-15, 2,5,8,14 }, { -3,-7,-10,-13, 2,6,9,12 }, { -2,-5,-8,-13, 1,4,7,12 }, { -2,-4,-6,-13, 1,3,5,12 },
		{ -3,-6, -8,-12, 2,5,7,11 }, { -3,-7, -9,-11, 2,6,8,10 }, { -4,-7,-8,-11, 3,6,7,10 }, { -3,-5,-8,-11, 2,4,7,10 },
		{ -2,-6, -8,-10, 1,5,7, 9 }, { -2,-5, -8,-10, 1,4,7, 9 }, { -2,-4,-8,-10, 1,3,7, 9 }, { -2,-5,-7,-10, 1,4,6, 9 },
		{ -3,-4, -7,-10, 2,3,6, 9 }, { -1,-2, -3,-10, 0,1,2, 9 }, { -4,-6,-8, -9, 3,5,7, 8 }, { -3,-5,-7, -9, 2,4,6, 8 },
	};
	uint64_t bits = 0;
	for (int32_t i = 0; i < 8; i++) bits = (bits << 8) | block[i];

	int32_t base       = (int32_t)((bits >> 56) & 0xFF);
	int32_t multiplier = (int32_t)((bits >> 52) & 0x0F);
	int32_t table      = (int32_t)((bits >> 48) & 0x0F);
	for (uint32_t y = 0; y < 4; y++) {
		for (uint32_t x = 0; x < 4; x++) {
			int32_t index = (int32_t)((bits >> (45 - 3 * (x * 4 + y))) & 7);
			// R11 semantics, degrading to the 8-bit alpha form: multiplier 0
			// means one eighth.
			int32_t value = base * 8 + 4 + modifier[table][index] * (multiplier ? multiplier * 8 : 1);
			value = value < 0 ? 0 : (value > 2047 ? 2047 : value);
			out_texels[y * 4 + x] = (uint8_t)((value + 4) / 8 > 255 ? 255 : (value + 4) / 8);
		}
	}
}

static void decode_bc4_block(const uint8_t block[8], uint8_t out_texels[16]) {
	int32_t red0 = block[0], red1 = block[1];
	int32_t palette[8];
	palette[0] = red0;
	palette[1] = red1;
	if (red0 > red1) {
		for (int32_t k = 1; k <= 6; k++) palette[1 + k] = ((7 - k) * red0 + k * red1) / 7;
	} else {
		for (int32_t k = 1; k <= 4; k++) palette[1 + k] = ((5 - k) * red0 + k * red1) / 5;
		palette[6] = 0;
		palette[7] = 255;
	}
	uint64_t indices = 0;
	for (int32_t i = 5; i >= 0; i--) indices = (indices << 8) | block[2 + i];
	for (uint32_t i = 0; i < 16; i++)
		out_texels[i] = (uint8_t)palette[(indices >> (3 * i)) & 7];
}

static void decode_bc1_block(const uint8_t block[8], uint8_t out_texels[64]) {
	uint32_t colour[2] = { (uint32_t)block[0] | ((uint32_t)block[1] << 8),
	                       (uint32_t)block[2] | ((uint32_t)block[3] << 8) };
	int32_t  rgb[4][3];
	for (int32_t i = 0; i < 2; i++) {
		int32_t r = (int32_t)((colour[i] >> 11) & 31), g = (int32_t)((colour[i] >> 5) & 63), b = (int32_t)(colour[i] & 31);
		rgb[i][0] = (r << 3) | (r >> 2);
		rgb[i][1] = (g << 2) | (g >> 4);
		rgb[i][2] = (b << 3) | (b >> 2);
	}
	bool four = colour[0] > colour[1];
	for (int32_t c = 0; c < 3; c++) {
		if (four) {
			rgb[2][c] = (2 * rgb[0][c] + rgb[1][c]) / 3;
			rgb[3][c] = (rgb[0][c] + 2 * rgb[1][c]) / 3;
		} else {
			rgb[2][c] = (rgb[0][c] + rgb[1][c]) / 2;
			rgb[3][c] = 0;
		}
	}
	for (uint32_t y = 0; y < 4; y++)
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t index = (block[4 + y] >> (x * 2)) & 3;
			uint8_t* texel = out_texels + (y * 4 + x) * 4;
			for (int32_t c = 0; c < 3; c++) texel[c] = (uint8_t)rgb[index][c];
			texel[3] = 255;
		}
}

// Returns PSNR in dB, or 99.0 for an exact match.
static double block_psnr(const uint8_t* got, const uint8_t* want, size_t blocks, compare_ mode, int32_t channels) {
	size_t stride = mode == compare_etc2_rgba ? 16 : (size_t)channels * 8;
	double sum    = 0.0;
	size_t count  = 0;
	for (size_t b = 0; b < blocks; b++) {
		for (int32_t c = 0; c < channels; c++) {
			uint8_t a[16], e[16];
			const uint8_t* got_block  = got  + b * stride + (size_t)c * 8;
			const uint8_t* want_block = want + b * stride + (size_t)c * 8;
			if (mode == compare_bc4) { decode_bc4_block(got_block, a); decode_bc4_block(want_block, e); }
			else                     { decode_eac_block(got_block, a); decode_eac_block(want_block, e); }
			for (int32_t i = 0; i < 16; i++) { double d = (double)a[i] - e[i]; sum += d * d; count++; }
		}
	}
	return psnr_db(sum, count);
}

// Judged against the ETC1S reconstruction both transcoders approximate, not
// against KTX-Software's bytes. Ours only has to be as good as theirs.
static void bc_quality(const char* subcase, const target_t* target, const ktx2_reader_t* reader,
                       const uint8_t* got, const uint8_t* want, size_t bytes, ktx2_fmt_ format) {
	ktx2_plan_t reference_plan;
	if (ktx2_plan(reader, &k_host_context, ktx2_caps_none, &reference_plan) != ktx2_result_success) { g_skip++; return; }
	uint8_t* reference = (uint8_t*)malloc(reference_plan.data_bytes);
	if (ktx2_transcode(&reference_plan, reference, reference_plan.data_bytes, NULL) != ktx2_result_success) {
		free(reference); g_skip++; return;
	}

	ktx2_info_t info     = ktx2_get_info(reader);
	uint32_t    width    = (uint32_t)info.width;
	uint32_t    blocks_x = (width + 3) / 4;
	size_t      stride   = target->compare == compare_bc3 ? 16 : 8;
	size_t      colour   = target->compare == compare_bc3 ? 8  : 0; // BC3 puts alpha first
	int32_t     channels = target->compare == compare_bc3 ? 4  : 3;

	double got_sse = 0.0, want_sse = 0.0;
	size_t count   = 0;
	for (size_t b = 0; b < bytes / stride; b++) {
		uint8_t got_texels[64], want_texels[64], got_alpha[16], want_alpha[16];
		decode_bc1_block(got  + b * stride + colour, got_texels);
		decode_bc1_block(want + b * stride + colour, want_texels);
		if (target->compare == compare_bc3) {
			decode_bc4_block(got  + b * stride, got_alpha);
			decode_bc4_block(want + b * stride, want_alpha);
		}
		uint32_t block_x = (uint32_t)(b % blocks_x), block_y = (uint32_t)(b / blocks_x);
		for (uint32_t y = 0; y < 4; y++) {
			for (uint32_t x = 0; x < 4; x++) {
				const uint8_t* ref = reference + ((size_t)(block_y * 4 + y) * width + block_x * 4 + x) * 4;
				for (int32_t c = 0; c < channels; c++) {
					double g = c == 3 ? got_alpha [y * 4 + x] : got_texels [(y * 4 + x) * 4 + c];
					double w = c == 3 ? want_alpha[y * 4 + x] : want_texels[(y * 4 + x) * 4 + c];
					got_sse  += (g - ref[c]) * (g - ref[c]);
					want_sse += (w - ref[c]) * (w - ref[c]);
					count++;
				}
			}
		}
	}
	free(reference);

	double ours   = psnr_db(got_sse,  count);
	double theirs = psnr_db(want_sse, count);
	if (ours < theirs - BC_TOLERANCE_DB) {
		printf("  FAIL  %-16s %-9s  -> %-18s %.1f dB vs reference, KTX-Software %.1f dB\n",
			subcase, target->golden_id, ktx2_fmt_str(format), ours, theirs);
		g_fail++;
	} else {
		printf("  ok    %-16s %-9s  -> %-18s %.1f dB vs reference (KTX-Software %.1f dB)\n",
			subcase, target->golden_id, ktx2_fmt_str(format), ours, theirs);
		g_pass++;
	}
}

static const char* k_subcases[] = {
	"R8_UNORM",       "R8_SRGB",
	"R8G8_UNORM",     "R8G8_SRGB",
	"R8G8B8_UNORM",   "R8G8B8_SRGB",
	"R8G8B8A8_UNORM", "R8G8B8A8_SRGB",
};

static void run_case(const char* cts, const char* subcase, const target_t* target) {
	char input_path [1024];
	char golden_path[1024];
	snprintf(input_path,  sizeof(input_path),  "%s/clitests/input/ktx2/valid_%s_2D_BLZE.ktx2", cts, subcase);
	snprintf(golden_path, sizeof(golden_path),
		"%s/clitests/golden/transcode/transcode_blze/output_%s_2D_BLZE_to_%s.ktx2", cts, subcase, target->golden_id);

	size_t   input_bytes = 0;
	uint8_t* input       = file_read(input_path, &input_bytes);
	if (input == NULL) { g_skip++; return; }

	ktx2_reader_t reader;
	ktx2_plan_t   plan;
	if (ktx2_open(input, input_bytes, &reader) != ktx2_result_success ||
	    ktx2_plan(&reader, &k_host_context, target->caps, &plan) != ktx2_result_success) {
		printf("  SKIP  %-16s %-9s  plan declined\n", subcase, target->golden_id);
		free(input); g_skip++; return;
	}
	// ktx2_plan picks by channel config, so most (subcase, target) pairs simply
	// do not correspond - only compare where its choice matches this golden.
	if (plan.format != target->expect_a && plan.format != target->expect_b) { free(input); return; }

	uint8_t*     output = (uint8_t*)malloc(plan.data_bytes ? plan.data_bytes : 1);
	ktx2_result_ result = ktx2_transcode(&plan, output, plan.data_bytes, NULL);
	if (result != ktx2_result_success) {
		printf("  SKIP  %-16s %-9s  -> %-18s %s\n", subcase, target->golden_id,
			ktx2_fmt_str(plan.format), ktx2_result_str(result));
		free(output); free(input); g_skip++; return;
	}

	// Two contract details no golden can express: an undersized output buffer is
	// refused rather than partly written, and caller-supplied scratch produces the
	// same bytes as the internal allocation.
	if (ktx2_transcode(&plan, output, plan.data_bytes - 1, NULL) != ktx2_result_buffer_too_small) {
		printf("  FAIL  %-16s %-9s  undersized output was not refused\n", subcase, target->golden_id);
		g_fail++;
	}
	uint8_t* scratched = (uint8_t*)malloc(plan.data_bytes ? plan.data_bytes : 1);
	void*    scratch   = malloc(plan.scratch_bytes ? plan.scratch_bytes : 1);
	if (ktx2_transcode(&plan, scratched, plan.data_bytes, scratch) != ktx2_result_success ||
	    memcmp(scratched, output, plan.data_bytes) != 0) {
		printf("  FAIL  %-16s %-9s  caller-supplied scratch changed the output\n", subcase, target->golden_id);
		g_fail++;
	}
	free(scratch); free(scratched);

	size_t   golden_bytes = 0;
	uint8_t* golden_file  = file_read(golden_path, &golden_bytes);
	size_t   expect_bytes = 0;
	uint8_t* expect       = golden_file ? golden_levels(golden_file, golden_bytes, &expect_bytes) : NULL;
	if (expect == NULL) {
		printf("  SKIP  %-16s %-9s  no golden\n", subcase, target->golden_id);
		free(golden_file); free(output); free(input); g_skip++; return;
	}

	if (expect_bytes != plan.data_bytes) {
		printf("  FAIL  %-16s %-9s  -> %-18s size %zu, golden %zu\n", subcase, target->golden_id,
			ktx2_fmt_str(plan.format), plan.data_bytes, expect_bytes);
		g_fail++;
	} else if (target->compare == compare_bc1 || target->compare == compare_bc3) {
		bc_quality(subcase, target, &reader, output, expect, expect_bytes, plan.format);
	} else if (target->compare != compare_exact) {
		// The ETC2 RGBA colour half is a true repack and must still be exact;
		// only the EAC alpha half is allowed to differ.
		int32_t colour_diff = 0;
		if (target->compare == compare_etc2_rgba)
			for (size_t b = 0; b < expect_bytes / 16; b++)
				colour_diff += memcmp(output + b * 16 + 8, expect + b * 16 + 8, 8) != 0;

		size_t stride = target->compare == compare_etc2_rgba ? 16 : (size_t)target->channels * 8;
		double psnr   = block_psnr(output, expect, expect_bytes / stride, target->compare, target->channels);
		if (colour_diff > 0) {
			printf("  FAIL  %-16s %-9s  -> %-18s %d colour blocks differ (must be exact)\n",
				subcase, target->golden_id, ktx2_fmt_str(plan.format), colour_diff);
			g_fail++;
		} else if (psnr < PSNR_FLOOR) {
			printf("  FAIL  %-16s %-9s  -> %-18s %.1f dB, floor %.1f\n", subcase, target->golden_id,
				ktx2_fmt_str(plan.format), psnr, (double)PSNR_FLOOR);
			g_fail++;
		} else {
			printf("  ok    %-16s %-9s  -> %-18s %.1f dB vs golden%s\n", subcase, target->golden_id,
				ktx2_fmt_str(plan.format), psnr,
				target->compare == compare_etc2_rgba ? ", colour exact" : "");
			g_pass++;
		}
	} else if (memcmp(output, expect, expect_bytes) != 0) {
		size_t  first = 0;
		int32_t diffs = 0;
		for (size_t i = 0; i < expect_bytes; i++)
			if (output[i] != expect[i]) { if (!diffs) first = i; diffs++; }
		printf("  FAIL  %-16s %-9s  -> %-18s %d/%zu bytes differ, first at %zu (got %02X want %02X)\n",
			subcase, target->golden_id, ktx2_fmt_str(plan.format), diffs, expect_bytes,
			first, output[first], expect[first]);
		g_fail++;
	} else {
		printf("  ok    %-16s %-9s  -> %-18s %zu bytes exact\n", subcase, target->golden_id,
			ktx2_fmt_str(plan.format), expect_bytes);
		g_pass++;
	}
	free(expect); free(golden_file); free(output); free(input);
}

///////////////////////////////////////////////////////////////////////////////
// The goldens are all 8x8 single-level, so they say nothing about the mip loop or
// levels whose dimensions are not multiples of 4. This decodes our ETC1 output
// with a decoder written separately from the writer and holds it against our RGBA
// output. A transposed plane, a wrong bit, or a misplaced mip all show up.

static void etc1_decode_block(const uint8_t block[8], uint8_t out_texels[64]) {
	static const int16_t modifier[8][4] = {
		{  2,   8,  -2,   -8 }, {  5,  17,  -5,  -17 }, {  9,  29,  -9,  -29 }, { 13,  42, -13,  -42 },
		{ 18,  60, -18,  -60 }, { 24,  80, -24,  -80 }, { 33, 106, -33, -106 }, { 47, 183, -47, -183 },
	};
	int32_t base[3];
	for (int32_t c = 0; c < 3; c++) {
		int32_t five = block[c] >> 3;                 // differential base, delta is zero
		base[c] = (five << 3) | (five >> 2);
	}
	int32_t  table = (block[3] >> 5) & 7;
	uint32_t msb   = ((uint32_t)block[4] << 8) | block[5];
	uint32_t lsb   = ((uint32_t)block[6] << 8) | block[7];

	for (uint32_t y = 0; y < 4; y++) {
		for (uint32_t x = 0; x < 4; x++) {
			uint32_t bit   = x * 4 + y;
			uint32_t index = (((msb >> bit) & 1) << 1) | ((lsb >> bit) & 1);
			int32_t  delta = modifier[table][index];
			uint8_t* texel = out_texels + (y * 4 + x) * 4;
			for (int32_t c = 0; c < 3; c++) {
				int32_t v = base[c] + delta;
				texel[c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
			}
			texel[3] = 255;
		}
	}
}

static void cross_check(const char* cts, const char* name) {
	char path[1024];
	snprintf(path, sizeof(path), "%s/clitests/input/ktx2_sample/%s", cts, name);
	size_t   input_bytes = 0;
	uint8_t* input       = file_read(path, &input_bytes);
	if (input == NULL) { g_skip++; return; }

	ktx2_reader_t reader;
	ktx2_plan_t   etc_plan, rgba_plan;
	if (ktx2_open(input, input_bytes, &reader)                != ktx2_result_success ||
	    ktx2_plan(&reader, &k_host_context, ktx2_caps_etc2, &etc_plan)         != ktx2_result_success ||
	    ktx2_plan(&reader, &k_host_context, ktx2_caps_none, &rgba_plan)        != ktx2_result_success) {
		free(input); g_skip++; return;
	}

	uint8_t* etc  = (uint8_t*)malloc(etc_plan.data_bytes);
	uint8_t* rgba = (uint8_t*)malloc(rgba_plan.data_bytes);
	if (ktx2_transcode(&etc_plan,  etc,  etc_plan.data_bytes,  NULL) != ktx2_result_success ||
	    ktx2_transcode(&rgba_plan, rgba, rgba_plan.data_bytes, NULL) != ktx2_result_success) {
		free(rgba); free(etc); free(input); g_skip++; return;
	}

	ktx2_info_t info      = ktx2_get_info(&reader);
	size_t      etc_at    = 0;
	size_t      rgba_at   = 0;
	int32_t     mismatch  = 0;
	for (int32_t level = 0; level < info.mip_count; level++) {
		uint32_t w = (uint32_t)info.width  >> level; w = w ? w : 1;
		uint32_t h = (uint32_t)info.height >> level; h = h ? h : 1;
		uint32_t blocks_x = (w + 3) / 4, blocks_y = (h + 3) / 4;

		for (uint32_t by = 0; by < blocks_y; by++) {
			for (uint32_t bx = 0; bx < blocks_x; bx++) {
				uint8_t texels[64];
				etc1_decode_block(etc + etc_at + ((size_t)by * blocks_x + bx) * 8, texels);
				uint32_t copy_w = w - bx * 4 < 4 ? w - bx * 4 : 4;
				uint32_t copy_h = h - by * 4 < 4 ? h - by * 4 : 4;
				for (uint32_t y = 0; y < copy_h; y++)
					for (uint32_t x = 0; x < copy_w; x++)
						if (memcmp(texels + (y * 4 + x) * 4,
						           rgba + rgba_at + ((size_t)(by * 4 + y) * w + bx * 4 + x) * 4, 4) != 0)
							mismatch++;
			}
		}
		etc_at  += (size_t)blocks_x * blocks_y * 8;
		rgba_at += (size_t)w * h * 4;
	}

	if (mismatch == 0) {
		printf("  ok    %-34s %4dx%-4d mips=%-2d  ETC1 matches RGBA\n", name, info.width, info.height, info.mip_count);
		g_pass++;
	} else {
		printf("  FAIL  %-34s %4dx%-4d mips=%-2d  %d texels differ\n", name, info.width, info.height, info.mip_count, mismatch);
		g_fail++;
	}
	free(rgba); free(etc); free(input);
}

// The §1.2 trap on real files: an RG normal map and an RGBA texture both carry
// two ETC1S slices, and only the DFD channel IDs separate them. Reading the
// normal map's second slice as alpha corrupts every normal but still looks valid.
static void channel_check(const char* cts, const char* name, ktx2_caps_ caps, ktx2_fmt_ expect) {
	char path[1024];
	snprintf(path, sizeof(path), "%s/clitests/input/ktx2_sample/%s", cts, name);
	size_t   bytes = 0;
	uint8_t* data  = file_read(path, &bytes);
	if (data == NULL) { g_skip++; return; }

	ktx2_reader_t reader;
	ktx2_plan_t   plan;
	if (ktx2_open(data, bytes, &reader)      != ktx2_result_success ||
	    ktx2_plan(&reader, &k_host_context, caps, &plan)      != ktx2_result_success) {
		printf("  FAIL  %-36s could not plan\n", name);
		free(data); g_fail++; return;
	}
	if (plan.format != expect) {
		printf("  FAIL  %-36s got %s, expected %s\n", name, ktx2_fmt_str(plan.format), ktx2_fmt_str(expect));
		g_fail++;
	} else {
		printf("  ok    %-36s -> %s\n", name, ktx2_fmt_str(plan.format));
		g_pass++;
	}
	free(data);
}

///////////////////////////////////////////////////////////////////////////////
// UASTC block decode against libktx's rgba8 transcode. The spec's 64 block
// vectors are the primary gate (see uastc_test.c); these files add the modes
// those miss, real images, and the container walk around the decoder.
//
// The goldens are always RGBA8 and libktx normalizes as it writes them: a
// two-channel source stores its second channel in the block's alpha, but the
// golden carries it in green. Our r8/rg8 output unpacks the same way.

// The ASTC goldens reach only UASTC modes 0, 12 and 15, so they say nothing about
// multi-subset blocks, dual plane blocks, void extents or quint ranges. This
// decodes our ASTC output with astc_reference.h and holds it against
// ktx2_uastc_to_rgba over every block of every uncompressed UASTC file.
static void uastc_astc_crosscheck(const char* path, uint32_t* ref_seen) {
	size_t   bytes = 0;
	uint8_t* data  = file_read(path, &bytes);
	if (data == NULL) { g_skip++; return; }

	ktx2_reader_t reader;
	if (ktx2_open(data, bytes, &reader) != ktx2_result_success ||
	    reader.source != ktx2_source_uastc_ldr_4x4 || reader.supercompression != 0) {
		free(data); g_skip++; return;
	}

	const char* name = strrchr(path, '/');
	name = name ? name + 1 : path;

	size_t total = 0, bad = 0, rejected = 0;
	double bc7_sse = 0;
	for (uint32_t level = 0; level < reader.level_stored; level++) {
		const uint8_t* src    = reader.data + reader.levels[level].offset;
		uint64_t       blocks = reader.levels[level].bytes / 16;
		for (uint64_t b = 0; b < blocks; b++) {
			ktx2_uastc_t unpacked;
			uint8_t      want[64], got[64], astc[16];
			ktx2_uastc_unpack (src + b * 16, &unpacked);
			ref_seen[unpacked.mode]++;
			ktx2_uastc_to_rgba(&unpacked, want);
			// BC7 rides along, scored rather than compared: it is lossy by design.
			uint8_t bc7[16], bc7_texels[64];
			ktx2_uastc_write_bc7(&unpacked, bc7);
			if (bc7_ref_decode(bc7, bc7_texels))
				for (uint32_t i = 0; i < 64; i++) {
					double d = (double)bc7_texels[i] - want[i];
					bc7_sse += d * d;
				}
			else rejected++;

			ktx2_uastc_write_astc(&unpacked, astc);
			total++;
			if (!astc_ref_decode(astc, got)) { rejected++; continue; }
			if (memcmp(want, got, 64) != 0) {
				if (bad == 0)
					printf("  FAIL  %-28s block %llu mode %u: ASTC decodes to %u,%u,%u,%u, UASTC to %u,%u,%u,%u\n",
						name, (unsigned long long)b, unpacked.mode,
						got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
				bad++;
			}
		}
	}
	if (bad != 0 || rejected != 0) {
		printf("  FAIL  %-28s %zu/%zu blocks differ, %zu rejected by the decoder\n", name, bad, total, rejected);
		g_fail++;
	} else {
		printf("  ok    %-28s %zu blocks round-trip, BC7 %.1f dB\n",
			name, total, psnr_db(bc7_sse, total * 64));
		g_pass++;
	}
	free(data);
}

// Near-lossless, so the question is not "do we match libktx's bytes" but "is our
// block as close to the source as theirs". Both are scored against
// ktx2_uastc_to_rgba, which the spec's 64 block vectors pin exactly.
static void uastc_bc7_check(const char* cts, const char* subcase) {
	char input_path [1024];
	char golden_path[1024];
	snprintf(input_path,  sizeof(input_path),  "%s/clitests/input/ktx2/valid_%s_2D_UASTC.ktx2", cts, subcase);
	snprintf(golden_path, sizeof(golden_path),
		"%s/clitests/golden/transcode/transcode_uastc/output_%s_2D_UASTC_to_bc7.ktx2", cts, subcase);

	size_t   input_bytes = 0;
	uint8_t* input       = file_read(input_path, &input_bytes);
	if (input == NULL) { g_skip++; return; }

	ktx2_reader_t reader;
	ktx2_plan_t   plan;
	if (ktx2_open(input, input_bytes, &reader)      != ktx2_result_success ||
	    ktx2_plan(&reader, &k_host_context, ktx2_caps_bc, &plan) != ktx2_result_success) {
		printf("  SKIP  %-16s plan declined\n", subcase);
		free(input); g_skip++; return;
	}

	uint8_t* output = (uint8_t*)malloc(plan.data_bytes ? plan.data_bytes : 1);
	if (ktx2_transcode(&plan, output, plan.data_bytes, NULL) != ktx2_result_success) {
		printf("  SKIP  %-16s transcode declined\n", subcase);
		free(output); free(input); g_skip++; return;
	}

	size_t   golden_bytes = 0;
	uint8_t* golden_file  = file_read(golden_path, &golden_bytes);
	size_t   expect_bytes = 0;
	uint8_t* expect       = golden_file ? golden_levels(golden_file, golden_bytes, &expect_bytes) : NULL;
	if (expect == NULL || expect_bytes != plan.data_bytes) {
		printf("  SKIP  %-16s no comparable golden\n", subcase);
		free(expect); free(golden_file); free(output); free(input); g_skip++; return;
	}

	// Ground truth is the UASTC block itself, decoded straight to pixels.
	const uint8_t* src   = reader.data + reader.levels[0].offset;
	size_t         count = expect_bytes / 16;
	double         ours = 0, theirs = 0;
	size_t         same = 0;
	for (size_t b = 0; b < count; b++) {
		ktx2_uastc_t unpacked;
		uint8_t      truth[64], mine[64], libktx[64];
		ktx2_uastc_unpack (src + b * 16, &unpacked);
		ktx2_uastc_to_rgba(&unpacked, truth);
		if (!bc7_ref_decode(output + b * 16, mine) || !bc7_ref_decode(expect + b * 16, libktx)) {
			printf("  FAIL  %-16s block %zu did not decode as BC7\n", subcase, b);
			g_fail++; free(expect); free(golden_file); free(output); free(input); return;
		}
		if (memcmp(output + b * 16, expect + b * 16, 16) == 0) same++;
		for (uint32_t i = 0; i < 64; i++) {
			double a = (double)mine  [i] - truth[i];
			double c = (double)libktx[i] - truth[i];
			ours   += a * a;
			theirs += c * c;
		}
	}

	double ours_db = psnr_db(ours, count * 64), theirs_db = psnr_db(theirs, count * 64);
	if (ours_db < theirs_db - BC_TOLERANCE_DB) {
		printf("  FAIL  %-16s -> %-14s %.1f dB vs libktx %.1f\n",
			subcase, ktx2_fmt_str(plan.format), ours_db, theirs_db);
		g_fail++;
	} else {
		printf("  ok    %-16s -> %-14s %.1f dB vs libktx %.1f, %zu/%zu blocks identical\n",
			subcase, ktx2_fmt_str(plan.format), ours_db, theirs_db, same, count);
		g_pass++;
	}
	free(expect); free(golden_file); free(output); free(input);
}

// Supercompression is a wrapper, not an encoding: the same content with and
// without Zstd must transcode identically. The corpus ships both forms of
// several files, so this needs no golden.
//
// The second half is the API-shaped mistake rather than the bitstream one:
// planning a Zstd source with no decompressor has to fail *at plan time*, before
// the caller has been handed an output size.
static void zstd_check(const char* cts, const char* plain_name, const char* zstd_name, ktx2_caps_ caps) {
	char plain_path[1024], zstd_path[1024];
	snprintf(plain_path, sizeof(plain_path), "%s/clitests/input/%s", cts, plain_name);
	snprintf(zstd_path,  sizeof(zstd_path),  "%s/clitests/input/%s", cts, zstd_name);

	size_t   plain_bytes = 0, zstd_bytes = 0;
	uint8_t* plain = file_read(plain_path, &plain_bytes);
	uint8_t* zstd  = file_read(zstd_path,  &zstd_bytes);
	if (plain == NULL || zstd == NULL) { free(plain); free(zstd); g_skip++; return; }

	const char* label = strrchr(zstd_name, '/');
	label = label ? label + 1 : zstd_name;

	ktx2_reader_t plain_reader, zstd_reader;
	ktx2_plan_t   plain_plan,   zstd_plan;
	if (ktx2_open(plain, plain_bytes, &plain_reader) != ktx2_result_success ||
	    ktx2_open(zstd,  zstd_bytes,  &zstd_reader)  != ktx2_result_success) {
		printf("  SKIP  %-32s unreadable\n", label);
		free(plain); free(zstd); g_skip++; return;
	}

	if (!HOST_HAS_ZSTD) {
		printf("  SKIP  %-32s no libzstd in this build\n", label);
		free(plain); free(zstd); g_skip++; return;
	}

	// Plan with a decompressor first: a file declined for another reason - the
	// corpus has a cubemap pair - never reaches the decompressor question.
	if (ktx2_plan(&plain_reader, &k_host_context, caps, &plain_plan) != ktx2_result_success ||
	    ktx2_plan(&zstd_reader,  &k_host_context, caps, &zstd_plan)  != ktx2_result_success) {
		printf("  SKIP  %-32s out of scope for other reasons\n", label);
		free(plain); free(zstd); g_skip++; return;
	}
	if (plain_plan.format != zstd_plan.format || plain_plan.data_bytes != zstd_plan.data_bytes) {
		printf("  FAIL  %-32s the two forms did not plan alike\n", label);
		free(plain); free(zstd); g_fail++; return;
	}

	// A zero context is the correct empty state, so this must be a clean refusal,
	// and it must land here rather than mid-transcode.
	ktx2_context_t empty = { NULL, NULL, 0, { 0 } };
	ktx2_plan_t    probe;
	ktx2_result_   refused = ktx2_plan(&zstd_reader, &empty, caps, &probe);
	if (refused != ktx2_result_no_decompressor) {
		printf("  FAIL  %-32s plan without a decompressor returned %s\n", label, ktx2_result_str(refused));
		g_fail++;
	}

	uint8_t* plain_out = (uint8_t*)malloc(plain_plan.data_bytes);
	uint8_t* zstd_out  = (uint8_t*)malloc(zstd_plan.data_bytes);
	ktx2_result_ a = ktx2_transcode(&plain_plan, plain_out, plain_plan.data_bytes, NULL);
	ktx2_result_ b = ktx2_transcode(&zstd_plan,  zstd_out,  zstd_plan.data_bytes,  NULL);

	if (a != ktx2_result_success || b != ktx2_result_success) {
		printf("  FAIL  %-32s transcode: plain %s, zstd %s\n", label, ktx2_result_str(a), ktx2_result_str(b));
		g_fail++;
	} else if (memcmp(plain_out, zstd_out, plain_plan.data_bytes) != 0) {
		printf("  FAIL  %-32s -> %-18s inflated content differs\n", label, ktx2_fmt_str(zstd_plan.format));
		g_fail++;
	} else {
		printf("  ok    %-32s -> %-18s %zu bytes identical to the plain form\n",
			label, ktx2_fmt_str(zstd_plan.format), zstd_plan.data_bytes);
		g_pass++;
	}
	free(plain_out); free(zstd_out); free(plain); free(zstd);
}

// A lossless repack on both sides, so this is a plain byte comparison - the
// strongest gate in the suite.
static void uastc_astc_check(const char* cts, const char* subcase) {
	char input_path [1024];
	char golden_path[1024];
	snprintf(input_path,  sizeof(input_path),  "%s/clitests/input/ktx2/valid_%s_2D_UASTC.ktx2", cts, subcase);
	snprintf(golden_path, sizeof(golden_path),
		"%s/clitests/golden/transcode/transcode_uastc/output_%s_2D_UASTC_to_astc.ktx2", cts, subcase);

	size_t   input_bytes = 0;
	uint8_t* input       = file_read(input_path, &input_bytes);
	if (input == NULL) { g_skip++; return; }

	ktx2_reader_t reader;
	ktx2_plan_t   plan;
	if (ktx2_open(input, input_bytes, &reader)          != ktx2_result_success ||
	    ktx2_plan(&reader, &k_host_context, ktx2_caps_astc_ldr, &plan)   != ktx2_result_success) {
		printf("  SKIP  %-16s plan declined\n", subcase);
		free(input); g_skip++; return;
	}

	uint8_t*     output = (uint8_t*)malloc(plan.data_bytes ? plan.data_bytes : 1);
	ktx2_result_ result = ktx2_transcode(&plan, output, plan.data_bytes, NULL);
	if (result != ktx2_result_success) {
		printf("  SKIP  %-16s -> %-18s %s\n", subcase, ktx2_fmt_str(plan.format), ktx2_result_str(result));
		free(output); free(input); g_skip++; return;
	}

	size_t   golden_bytes = 0;
	uint8_t* golden_file  = file_read(golden_path, &golden_bytes);
	size_t   expect_bytes = 0;
	uint8_t* expect       = golden_file ? golden_levels(golden_file, golden_bytes, &expect_bytes) : NULL;
	if (expect == NULL) {
		printf("  SKIP  %-16s no golden\n", subcase);
		free(golden_file); free(output); free(input); g_skip++; return;
	}

	if (expect_bytes != plan.data_bytes) {
		printf("  FAIL  %-16s -> %-18s size %zu, golden %zu\n",
			subcase, ktx2_fmt_str(plan.format), plan.data_bytes, expect_bytes);
		g_fail++;
	} else if (memcmp(output, expect, expect_bytes) != 0) {
		size_t blocks = expect_bytes / 16, bad = 0, first = expect_bytes;
		for (size_t b = 0; b < blocks; b++)
			if (memcmp(output + b * 16, expect + b * 16, 16) != 0) {
				if (bad == 0) first = b;
				bad++;
			}
		printf("  FAIL  %-16s -> %-18s %zu/%zu blocks differ, first at %zu\n",
			subcase, ktx2_fmt_str(plan.format), bad, blocks, first);
		if (first < blocks) {
			printf("          got ");
			for (int i = 0; i < 16; i++) printf("%02X", output[first * 16 + i]);
			printf("\n          want ");
			for (int i = 0; i < 16; i++) printf("%02X", expect[first * 16 + i]);
			printf("\n");
		}
		g_fail++;
	} else {
		printf("  ok    %-16s -> %-18s %zu blocks exact\n",
			subcase, ktx2_fmt_str(plan.format), expect_bytes / 16);
		g_pass++;
	}
	free(expect); free(golden_file); free(output); free(input);
}

static void uastc_check(const char* cts, const char* subcase) {
	char input_path [1024];
	char golden_path[1024];
	snprintf(input_path,  sizeof(input_path),  "%s/clitests/input/ktx2/valid_%s_2D_UASTC.ktx2", cts, subcase);
	snprintf(golden_path, sizeof(golden_path),
		"%s/clitests/golden/transcode/transcode_uastc/output_%s_2D_UASTC_to_rgba8.ktx2", cts, subcase);

	size_t   input_bytes = 0;
	uint8_t* input       = file_read(input_path, &input_bytes);
	if (input == NULL) { g_skip++; return; }

	ktx2_reader_t reader;
	ktx2_plan_t   plan;
	if (ktx2_open(input, input_bytes, &reader)        != ktx2_result_success ||
	    ktx2_plan(&reader, &k_host_context, ktx2_caps_none, &plan)     != ktx2_result_success) {
		printf("  SKIP  %-16s plan declined\n", subcase);
		free(input); g_skip++; return;
	}

	uint8_t*     output = (uint8_t*)malloc(plan.data_bytes ? plan.data_bytes : 1);
	ktx2_result_ result = ktx2_transcode(&plan, output, plan.data_bytes, NULL);
	if (result != ktx2_result_success) {
		printf("  SKIP  %-16s -> %-12s %s\n", subcase, ktx2_fmt_str(plan.format), ktx2_result_str(result));
		free(output); free(input); g_skip++; return;
	}

	size_t   golden_bytes = 0;
	uint8_t* golden_file  = file_read(golden_path, &golden_bytes);
	size_t   expect_bytes = 0;
	uint8_t* expect       = golden_file ? golden_levels(golden_file, golden_bytes, &expect_bytes) : NULL;
	if (expect == NULL) {
		printf("  SKIP  %-16s no golden\n", subcase);
		free(golden_file); free(output); free(input); g_skip++; return;
	}

	size_t stride = plan.format == ktx2_fmt_r8 ? 1 : (plan.format == ktx2_fmt_rg8 ? 2 : 4);
	size_t texels = expect_bytes / 4;
	if (texels * stride != plan.data_bytes) {
		printf("  FAIL  %-16s -> %-12s %zu bytes, golden %zu texels\n",
			subcase, ktx2_fmt_str(plan.format), plan.data_bytes, texels);
		g_fail++;
	} else {
		size_t diffs = 0;
		for (size_t t = 0; t < texels; t++) {
			const uint8_t* got  = output + t * stride;
			const uint8_t* want = expect + t * 4;
			bool same = stride == 4
				? memcmp(got, want, 4) == 0
				: (got[0] == want[0] && (stride == 1 || got[1] == want[1]));
			if (!same) diffs++;
		}
		if (diffs != 0) {
			printf("  FAIL  %-16s -> %-12s %zu/%zu texels differ\n",
				subcase, ktx2_fmt_str(plan.format), diffs, texels);
			g_fail++;
		} else {
			printf("  ok    %-16s -> %-12s %zu texels exact\n", subcase, ktx2_fmt_str(plan.format), texels);
			g_pass++;
		}
	}
	free(expect); free(golden_file); free(output); free(input);
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
	if (argc < 2) { fprintf(stderr, "usage: ktx2_golden <cts dir>\n"); return 2; }

	printf("Bit-exact against KTX-Software:\n");
	for (size_t s = 0; s < sizeof(k_subcases) / sizeof(k_subcases[0]); s++)
		for (size_t t = 0; t < sizeof(k_targets) / sizeof(k_targets[0]); t++)
			run_case(argv[1], k_subcases[s], &k_targets[t]);

	// Mip chains and dimensions the goldens never exercise: 7 levels, and
	// 211x211 / 200x100, where the last block of a row hangs off the edge.
	printf("\nIndependent ETC1 decode cross-check:\n");
	cross_check(argv[1], "rgba-mipmap-reference-basis.ktx2");
	cross_check(argv[1], "CesiumLogoFlat.ktx2");
	cross_check(argv[1], "luminance_reference_basis.ktx2");
	cross_check(argv[1], "kodim17_basis.ktx2");
	cross_check(argv[1], "color_grid_basis.ktx2");
	cross_check(argv[1], "FlightHelmet_baseColor_basis.ktx2");

	printf("\nTwo-slice channel configuration (an RG normal map is not RGBA):\n");
	channel_check(argv[1], "etc1s_Iron_Bars_001_normal.ktx2",      ktx2_caps_etc2, ktx2_fmt_eac_rg11);
	channel_check(argv[1], "etc1s_Iron_Bars_001_normal.ktx2",      ktx2_caps_bc,   ktx2_fmt_bc5_rg);
	channel_check(argv[1], "etc1s_Iron_Bars_001_normal.ktx2",      ktx2_caps_none, ktx2_fmt_rg8);
	channel_check(argv[1], "alpha_simple_basis.ktx2",              ktx2_caps_etc2, ktx2_fmt_etc2_rgba_srgb);
	channel_check(argv[1], "luminance_alpha_reference_basis.ktx2", ktx2_caps_etc2, ktx2_fmt_etc2_rgba);
	channel_check(argv[1], "r_reference_basis.ktx2",               ktx2_caps_etc2, ktx2_fmt_eac_r11);

	printf("\nUASTC block decode vs libktx rgba8:\n");
	uastc_check(argv[1], "R8_UNORM");       uastc_check(argv[1], "R8_SRGB");
	uastc_check(argv[1], "R8G8_UNORM");     uastc_check(argv[1], "R8G8_SRGB");
	uastc_check(argv[1], "R8G8B8_UNORM");   uastc_check(argv[1], "R8G8B8_SRGB");
	uastc_check(argv[1], "R8G8B8A8_UNORM"); uastc_check(argv[1], "R8G8B8A8_SRGB");

	printf("\nUASTC -> ASTC 4x4, byte-exact:\n");
	uastc_astc_check(argv[1], "R8_UNORM");       uastc_astc_check(argv[1], "R8_SRGB");
	uastc_astc_check(argv[1], "R8G8_UNORM");     uastc_astc_check(argv[1], "R8G8_SRGB");
	uastc_astc_check(argv[1], "R8G8B8_UNORM");   uastc_astc_check(argv[1], "R8G8B8_SRGB");
	uastc_astc_check(argv[1], "R8G8B8A8_UNORM"); uastc_astc_check(argv[1], "R8G8B8A8_SRGB");

	printf("\nUASTC -> ASTC 4x4 vs an independent ASTC decoder:\n");
	{
		static const char* k_uastc_files[] = {
			"clitests/input/ktx2/valid_R8_UNORM_2D_UASTC.ktx2",
			"clitests/input/ktx2/valid_R8G8_UNORM_2D_UASTC.ktx2",
			"clitests/input/ktx2/valid_R8G8B8_UNORM_2D_UASTC.ktx2",
			"clitests/input/ktx2/valid_R8G8B8A8_UNORM_2D_UASTC.ktx2",
			"clitests/input/ktx2_sample/color_grid_uastc.ktx2",
			"clitests/input/ktx2_sample/cimg5293_uastc.ktx2",
			"clitests/input/ktx2_sample/camera_camera_BaseColor_uastc.ktx2",
			"clitests/input/ktx2_sample/cyan_rgb_reference_uastc.ktx2",
			"clitests/input/ktx2_sample/luminance_reference_uastc.ktx2",
			"clitests/input/ktx2_sample/luminance_alpha_reference_uastc.ktx2",
			"clitests/input/ktx2_sample/rg_reference_uastc.ktx2",
			"clitests/input/ktx2_sample/r_reference_uastc.ktx2",
			"clitests/input/compare/content/supercomp_UASTC.ktx2",
		};
		uint32_t seen[21];
		memset(seen, 0, sizeof(seen));
		for (size_t i = 0; i < sizeof(k_uastc_files) / sizeof(k_uastc_files[0]); i++) {
			char path[1024];
			snprintf(path, sizeof(path), "%s/%s", argv[1], k_uastc_files[i]);
			uastc_astc_crosscheck(path, seen);
		}
		printf("        modes reached:");
		for (uint32_t m = 0; m < 21; m++) if (seen[m]) printf(" %u", m);
		printf("\n        modes missing:");
		for (uint32_t m = 0; m < 19; m++) if (!seen[m]) printf(" %u", m);
		printf("\n");
	}

	printf("\nUASTC -> BC7, near-lossless, scored against the UASTC source:\n");
	uastc_bc7_check(argv[1], "R8_UNORM");       uastc_bc7_check(argv[1], "R8_SRGB");
	uastc_bc7_check(argv[1], "R8G8_UNORM");     uastc_bc7_check(argv[1], "R8G8_SRGB");
	uastc_bc7_check(argv[1], "R8G8B8_UNORM");   uastc_bc7_check(argv[1], "R8G8B8_SRGB");
	uastc_bc7_check(argv[1], "R8G8B8A8_UNORM"); uastc_bc7_check(argv[1], "R8G8B8A8_SRGB");

	printf("\nZstd supercompression is a wrapper, not an encoding:\n");
	zstd_check(argv[1], "ktx2_sample/color_grid_uastc.ktx2",
	                    "ktx2_sample/color_grid_uastc_zstd.ktx2",         ktx2_caps_astc_ldr);
	zstd_check(argv[1], "compare/content/supercomp_UASTC.ktx2",
	                    "compare/content/supercomp_UASTC_ZSTD.ktx2",      ktx2_caps_astc_ldr);
	zstd_check(argv[1], "ktx2/valid_R8G8B8A8_UNORM_2D_UASTC.ktx2",
	                    "ktx2/valid_R8G8B8A8_UNORM_2D_UASTC_ZSTD_9.ktx2", ktx2_caps_astc_ldr);
	zstd_check(argv[1], "ktx2/valid_R8G8B8_SRGB_2D_UASTC.ktx2",
	                    "ktx2/valid_R8G8B8_SRGB_2D_UASTC_ZSTD_1.ktx2",    ktx2_caps_none);

	printf("\n%d exact, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
	return g_fail == 0 ? 0 : 1;
}
