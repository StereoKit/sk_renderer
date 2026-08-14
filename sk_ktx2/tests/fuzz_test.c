// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Fuzzing for the paths that touch untrusted bytes. A glTF downloaded off the
// network reaches ktx2_open and, if it parses, the ETC1S entropy decoder - so
// those are the threat model, and neither the CTS's fixed malformed set nor the
// truncation sweep in unit_tests.c explores them the way mutation does.
//
// One target, two drivers:
//
//  - libFuzzer, when built with -DSK_KTX2_FUZZ_LIBFUZZER by a clang that has it.
//    Coverage-guided, run for as long as you like.
//  - A deterministic mutation loop otherwise, so ctest always runs *some* of
//    this rather than the fuzzing being a thing someone remembers to do. Same
//    target function, seeds from the CTS when it is present.
//
// Neither driver checks results: every input is a legitimate input, and the only
// wrong answers are a crash, a hang, or an out-of-bounds access. Run it under
// ASan and UBSan or it proves very little.

#include "sk_ktx2.h"
#include "host_zstd.h"
#include "synthetic_ktx2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Each of these steers ktx2_plan to a different output format, and so to a
// different set of writers.
static const ktx2_caps_ k_caps[] = {
	ktx2_caps_none,
	ktx2_caps_bc,
	ktx2_caps_etc2,
	ktx2_caps_astc_ldr,
	ktx2_caps_bc | ktx2_caps_etc2 | ktx2_caps_astc_ldr,
};

// One caps set per input rather than all five, which is 5x the inputs per
// second for the same coverage across a run. Chosen from the bytes so the target
// stays a pure function of its input, which libFuzzer requires to reproduce.
static uint32_t caps_for(const uint8_t* data, size_t bytes) {
	uint32_t hash = 2166136261u;
	for (size_t i = 0; i < bytes && i < 64; i++) hash = (hash ^ data[i]) * 16777619u;
	return hash % (uint32_t)(sizeof(k_caps) / sizeof(k_caps[0]));
}

// The whole public surface, in the order a caller would use it. Transcoding is
// the point: stopping at ktx2_plan would leave the entropy decoder untouched,
// and that is the part walking attacker-controlled bits.
static void fuzz_one(const uint8_t* data, size_t bytes) {
	ktx2_reader_t reader;
	if (ktx2_open(data, bytes, &reader) != ktx2_result_success) return;

	ktx2_get_info(&reader);
	ktx2_check_gltf_basisu(&reader);

	ktx2_plan_t plan;
	if (ktx2_plan(&reader, &k_host_context, k_caps[caps_for(data, bytes)], &plan) != ktx2_result_success)
		return;

	// A malformed header can name a large texture in a small file, so bound the
	// allocation rather than the dimensions: the decoders run per block, so a
	// 1 GB output exercises nothing a 4 MB one does not.
	if (plan.data_bytes > 4u * 1024 * 1024) return;

	uint8_t* output  = (uint8_t*)malloc(plan.data_bytes    ? plan.data_bytes    : 1);
	uint8_t* scratch = (uint8_t*)malloc(plan.scratch_bytes ? plan.scratch_bytes : 1);
	if (output != NULL && scratch != NULL)
		ktx2_transcode(&plan, output, plan.data_bytes, scratch);
	free(scratch);
	free(output);
}

#ifdef SK_KTX2_FUZZ_LIBFUZZER

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
	fuzz_one(data, size);
	return 0;
}

#else

///////////////////////////////////////////////////////////////////////////////
// Standalone driver. Deterministic, so a failure reproduces from its iteration
// number alone, and bounded, so it can sit in ctest.

#define FUZZ_MAX_SEEDS  64
#define FUZZ_MAX_BYTES  (4 * 1024 * 1024)

typedef struct seed_t {
	uint8_t* data;
	size_t   bytes;
} seed_t;

static uint32_t g_state = 0x9E3779B9u;

static uint32_t rnd(uint32_t bound) {
	g_state = g_state * 1664525u + 1013904223u;
	return bound ? (g_state >> 8) % bound : 0;
}

// Byte-level mutations plus one structural one. The structural case matters
// most: KTX2 is offsets and lengths, and a plain bit flipper rarely lands on a
// length field with a value that is wrong in an interesting way.
static size_t mutate(const seed_t* seed, uint8_t* out_data, size_t capacity) {
	size_t bytes = seed->bytes < capacity ? seed->bytes : capacity;
	memcpy(out_data, seed->data, bytes);

	uint32_t rounds = 1 + rnd(8);
	for (uint32_t i = 0; i < rounds && bytes > 0; i++) {
		switch (rnd(5)) {
		case 0: out_data[rnd((uint32_t)bytes)] ^= (uint8_t)(1u << rnd(8)); break;
		case 1: out_data[rnd((uint32_t)bytes)]  = (uint8_t)rnd(256);       break;
		case 2: bytes = 1 + rnd((uint32_t)bytes); break; // truncate
		case 3: { // a 32-bit field, set to a value that tends to break arithmetic
			static const uint32_t k_interesting[] = {
				0, 1, 2, 3, 4, 0x7F, 0x80, 0xFF, 0x100, 0xFFFF, 0x10000,
				0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE,
			};
			if (bytes < 4) break;
			size_t at = (rnd((uint32_t)(bytes - 3)) / 4) * 4;
			synth_u32(out_data, at, k_interesting[rnd(sizeof(k_interesting) / sizeof(k_interesting[0]))]);
			break;
		}
		default: { // splice a run from somewhere else in the same file
			if (bytes < 8) break;
			uint32_t run = 1 + rnd(16);
			uint32_t src = rnd((uint32_t)bytes);
			uint32_t dst = rnd((uint32_t)bytes);
			if (src + run > bytes || dst + run > bytes) break;
			memmove(out_data + dst, out_data + src, run);
			break;
		}
		}
	}
	return bytes;
}

static uint8_t* file_read(const char* path, size_t* out_bytes) {
	FILE* f = fopen(path, "rb");
	if (f == NULL) return NULL;
	if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
	long size = ftell(f);
	rewind(f);
	if (size <= 0 || size > FUZZ_MAX_BYTES) { fclose(f); return NULL; }

	uint8_t* data = (uint8_t*)malloc((size_t)size);
	if (data != NULL && fread(data, 1, (size_t)size, f) != (size_t)size) { free(data); data = NULL; }
	fclose(f);
	if (data != NULL) *out_bytes = (size_t)size;
	return data;
}

// Real files make far better seeds than the synthetic one: their Huffman tables
// and codebooks are valid, so a mutation lands the entropy decoder in states it
// cannot otherwise reach. Optional, because the CTS is an optional checkout.
static int32_t add_seeds(seed_t* out_seeds, int32_t count, const char* cts, const char* const* names, int32_t name_count) {
	for (int32_t i = 0; i < name_count && count < FUZZ_MAX_SEEDS; i++) {
		char   path[1024];
		size_t bytes = 0;
		snprintf(path, sizeof(path), "%s/clitests/input/%s", cts, names[i]);
		uint8_t* data = file_read(path, &bytes);
		if (data == NULL) continue;
		out_seeds[count].data  = data;
		out_seeds[count].bytes = bytes;
		count++;
	}
	return count;
}

int main(int argc, char** argv) {
	static const char* k_seed_names[] = {
		"ktx2_sample/color_grid_basis.ktx2",
		"ktx2_sample/alpha_simple_basis.ktx2",
		"ktx2_sample/rg_reference_basis.ktx2",
		"ktx2_sample/r_reference_basis.ktx2",
		"ktx2_sample/kodim17_basis.ktx2",
		"ktx2_sample/color_grid_uastc.ktx2",
		"ktx2_sample/color_grid_uastc_zstd.ktx2",
		"ktx2_sample/luminance_alpha_reference_uastc.ktx2",
		"ktx2/valid_R8G8B8A8_UNORM_2D_UASTC.ktx2",
		"ktx2/valid_R8G8_UNORM_2D_BLZE.ktx2",
	};

	int32_t iterations = argc > 1 ? atoi(argv[1]) : 20000;
	seed_t  seeds[FUZZ_MAX_SEEDS];
	int32_t seed_count = 0;

	uint8_t* synthetic = (uint8_t*)malloc(SYNTH_BYTES);
	synth_build(synthetic);
	seeds[seed_count].data  = synthetic;
	seeds[seed_count].bytes = SYNTH_BYTES;
	seed_count++;

	if (argc > 2)
		seed_count = add_seeds(seeds, seed_count, argv[2], k_seed_names,
			(int32_t)(sizeof(k_seed_names) / sizeof(k_seed_names[0])));

	size_t   capacity = 0;
	for (int32_t i = 0; i < seed_count; i++)
		if (seeds[i].bytes > capacity) capacity = seeds[i].bytes;
	uint8_t* buffer = (uint8_t*)malloc(capacity);

	printf("fuzzing %d iterations over %d seed(s)\n", iterations, seed_count);
	for (int32_t i = 0; i < iterations; i++) {
		const seed_t* seed  = &seeds[rnd((uint32_t)seed_count)];
		size_t        bytes = mutate(seed, buffer, capacity);
		fuzz_one(buffer, bytes);
	}
	printf("fuzz: %d iterations, no crash\n", iterations);

	free(buffer);
	for (int32_t i = 0; i < seed_count; i++) free(seeds[i].data);
	return 0;
}

#endif
