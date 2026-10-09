// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// texenc_bench: times sk_texenc's encoders, headless, on desktop or Android.
//
// Usage: texenc_bench <image> <format> [options]
//   image   .png/.jpg/... (sRGB), .hdr (float), or .raw: u32 width, u32 height,
//           u32 fmt (0 rgba8 UNORM, 1 rgba32f), then the pixels
//   format  bc1 bc1_alpha bc7 bc6h astc4x4 astc6x6 astc8x8hdr
//   -iters N      timed encodes (default 30)
//   -warm MS      encode for MS first, so the GPU clock settles (Adreno needs ~1000)
//   -shader FILE  encode with this .sks instead of the embedded one, e.g. a
//                 different compiler's build of the encoder (sk_texenc_debug_shader)
//   -linear       sk_texenc_flags_linear
//   -out FILE     the encoded blocks: an .astc file for ASTC formats, raw blocks for BC
//
// Prints the first encode's time (pipeline creation included), the GPU time of
// the timed encodes, and a hash of the blocks, so two encoder builds compare by
// hash. Check quality on desktop with astc_validate (../validate).

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "astc_io.h"

#include <sk_renderer.h>
#include <sk_texenc.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const struct { const char* name; sk_texenc_fmt_ format; int32_t block; bool astc; } _formats[] = {
	{ "bc1",        sk_texenc_fmt_bc1,        4, false },
	{ "bc1_alpha",  sk_texenc_fmt_bc1_alpha,  4, false },
	{ "bc7",        sk_texenc_fmt_bc7,        4, false },
	{ "bc6h",       sk_texenc_fmt_bc6h,       4, false },
	{ "astc4x4",    sk_texenc_fmt_astc4x4,    4, true  },
	{ "astc6x6",    sk_texenc_fmt_astc6x6,    6, true  },
	{ "astc8x8hdr", sk_texenc_fmt_astc8x8hdr, 8, true  },
};

static size_t _inflate(void* context, const void* src, size_t src_bytes, void* out_dst, size_t dst_bytes) {
	(void)context;
	int32_t written = stbi_zlib_decode_buffer((char*)out_dst, (int32_t)dst_bytes, (const char*)src, (int32_t)src_bytes);
	return written < 0 ? 0 : (size_t)written;
}

static void* _read_file(const char* path, size_t* out_size) {
	FILE* f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	void* data = size > 0 ? malloc((size_t)size) : NULL;
	if (data && fread(data, 1, (size_t)size, f) != (size_t)size) { free(data); data = NULL; }
	fclose(f);
	*out_size = data ? (size_t)size : 0;
	return data;
}

static bool _ends_with(const char* str, const char* suffix) {
	size_t a = strlen(str), b = strlen(suffix);
	return a >= b && strcmp(str + a - b, suffix) == 0;
}

// The source texture. Formats follow the encoder's contract: sRGB and float
// sources Load as linear light, a .raw's UNORM bytes count as gamma-encoded.
static bool _load_source(const char* path, skr_tex_t* out_tex) {
	int32_t      w = 0, h = 0;
	skr_tex_fmt_ format = skr_tex_fmt_none;
	void*        pixels = NULL;
	if (_ends_with(path, ".raw")) {
		size_t   size = 0;
		uint8_t* raw  = (uint8_t*)_read_file(path, &size);
		uint32_t hdr[3];
		if (!raw || size < sizeof(hdr)) { free(raw); return false; }
		memcpy(hdr, raw, sizeof(hdr));
		size_t bytes = (size_t)hdr[0] * hdr[1] * (hdr[2] ? 16 : 4);
		if (size < sizeof(hdr) + bytes) { free(raw); return false; }
		w      = (int32_t)hdr[0];
		h      = (int32_t)hdr[1];
		format = hdr[2] ? skr_tex_fmt_rgba128f : skr_tex_fmt_rgba32_linear;
		pixels = malloc(bytes);
		memcpy(pixels, raw + sizeof(hdr), bytes);
		free(raw);
	} else if (stbi_is_hdr(path)) {
		pixels = stbi_loadf(path, &w, &h, NULL, 4);
		format = skr_tex_fmt_rgba128f;
	} else {
		pixels = stbi_load(path, &w, &h, NULL, 4);
		format = skr_tex_fmt_rgba32_srgb;
	}
	if (!pixels) return false;
	skr_err_ err = skr_tex_create(format, skr_tex_flags_readable, (skr_tex_sampler_t){0}, (skr_vec3i_t){ w, h, 1 }, 1, 1,
		&(skr_tex_data_t){ .data = pixels, .mip_count = 1, .layer_count = 1 }, out_tex);
	free(pixels);
	return err == skr_err_success;
}

static int _cmp_u64(const void* a, const void* b) {
	uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
	return x < y ? -1 : x > y;
}

// The Adreno GPU clock (kgsl), or 0 where the driver doesn't expose it
static double _gpu_clock_mhz(void) {
	FILE*     f  = fopen("/sys/class/kgsl/kgsl-3d0/gpuclk", "r");
	long long hz = 0;
	if (!f) return 0;
	if (fscanf(f, "%lld", &hz) != 1) hz = 0;
	fclose(f);
	return (double)hz / 1e6;
}

// One encode as its own frame, waited on, so the frame's GPU timestamps bracket it alone
static void _encode_frame(skr_tex_t* source, sk_texenc_fmt_ format, sk_texenc_flags_ flags, skr_buffer_t* blocks) {
	skr_renderer_frame_begin();
	sk_texenc_debug_encode(source, format, flags, 0, blocks);
	skr_renderer_frame_end(NULL, 0);
	skr_future_t future = skr_future_get();
	skr_future_wait(&future);
}

int main(int argc, char** argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: texenc_bench <image> <format> [-iters N] [-warm MS] [-shader FILE] [-linear] [-out FILE]\n");
		return 2;
	}
	const char*      image   = argv[1];
	int32_t          fi      = -1;
	int32_t          iters   = 30, warm_ms = 0;
	const char*      shader  = NULL;
	const char*      out     = NULL;
	sk_texenc_flags_ flags   = sk_texenc_flags_none;
	for (int32_t i = 0; i < (int32_t)(sizeof(_formats) / sizeof(_formats[0])); i++)
		if (strcmp(argv[2], _formats[i].name) == 0) fi = i;
	if (fi < 0) { fprintf(stderr, "unknown format '%s'\n", argv[2]); return 2; }
	for (int32_t i = 3; i < argc; i++) {
		bool more = i + 1 < argc;
		if      (strcmp(argv[i], "-iters")  == 0 && more) iters   = atoi(argv[++i]);
		else if (strcmp(argv[i], "-warm")   == 0 && more) warm_ms = atoi(argv[++i]);
		else if (strcmp(argv[i], "-shader") == 0 && more) shader  = argv[++i];
		else if (strcmp(argv[i], "-out")    == 0 && more) out     = argv[++i];
		else if (strcmp(argv[i], "-linear") == 0)         flags   = sk_texenc_flags_linear;
		else { fprintf(stderr, "unknown or incomplete option '%s'\n", argv[i]); return 2; }
	}
	if (iters < 1) iters = 1;
	sk_texenc_fmt_ format = _formats[fi].format;

	if (!skr_init((skr_settings_t){ .app_name = "texenc_bench" })) { fprintf(stderr, "skr_init failed\n"); return 1; }
	sk_texenc_init(_inflate, NULL);

	int32_t   rc     = 1;
	skr_tex_t source = {0};
	if (!_load_source(image, &source)) { fprintf(stderr, "can't load '%s'\n", image); goto done; }
	if (shader) {
		size_t size = 0;
		void*  sks  = _read_file(shader, &size);
		bool   ok   = sks && sk_texenc_debug_shader(format, sks, size);
		free(sks);
		if (!ok) { fprintf(stderr, "can't use '%s' as the %s encoder\n", shader, _formats[fi].name); goto done; }
	}

	uint32_t     bytes = sk_texenc_debug_size(&source, format, 0);
	skr_buffer_t blocks;
	if (skr_buffer_create(NULL, bytes, 1, skr_buffer_type_storage, (skr_use_)(skr_use_dynamic | skr_use_compute_readwrite | skr_use_uninitialized), &blocks) != skr_err_success) {
		fprintf(stderr, "can't create the block buffer\n");
		goto done;
	}

	// The first encode builds the pipeline
	uint64_t start = skr_time_now_ns();
	_encode_frame(&source, format, flags, &blocks);
	double first_ms = (double)(skr_time_now_ns() - start) / 1e6;

	for (uint64_t warm_start = skr_time_now_ns(); (double)(skr_time_now_ns() - warm_start) / 1e6 < warm_ms;)
		_encode_frame(&source, format, flags, &blocks);

	// Frame timing arrives a few frames late; every frame is the same encode, so
	// the last `iters` reports are the timed encodes'
	uint64_t* gpu_ns = (uint64_t*)calloc((size_t)iters, sizeof(uint64_t));
	int32_t   count  = 0;
	double    clk    = 0;
	for (int32_t i = 0; i < iters + 4; i++) {
		_encode_frame(&source, format, flags, &blocks);
		skr_frame_timing_t timing;
		if (i >= 4 && skr_renderer_get_frame_timing(&timing) && timing.gpu_time_ns) {
			gpu_ns[count++] = timing.gpu_time_ns;
			clk            += _gpu_clock_mhz();
		}
	}
	if (count == 0) { fprintf(stderr, "no GPU timing came back\n"); free(gpu_ns); skr_buffer_destroy(&blocks); goto done; }
	qsort(gpu_ns, (size_t)count, sizeof(uint64_t), _cmp_u64);

	uint8_t* data = (uint8_t*)malloc(bytes);
	skr_buffer_get(&blocks, data, bytes);
	uint64_t hash = 0xcbf29ce484222325ull; // FNV-1a
	for (uint32_t i = 0; i < bytes; i++) hash = (hash ^ data[i]) * 0x100000001b3ull;

	skr_vec3i_t size = skr_tex_get_size(&source);
	printf("BENCH %s %s %dx%d first %.1f ms, encode median %.3f ms min %.3f p90 %.3f (n=%d)",
		_formats[fi].name, image, size.x, size.y, first_ms, gpu_ns[count / 2] / 1e6, gpu_ns[0] / 1e6, gpu_ns[(count * 9) / 10] / 1e6, count);
	if (clk > 0) printf(" gpuclk %.0f MHz", clk / count);
	printf(" hash %016llx\n", (unsigned long long)hash);

	rc = 0;
	if (out) {
		bool ok = false;
		if (_formats[fi].astc) {
			ok = astc_write_file(out, size.x, size.y, _formats[fi].block, _formats[fi].block, data, bytes);
		} else {
			FILE* f = fopen(out, "wb");
			ok = f && fwrite(data, 1, bytes, f) == bytes;
			if (f) fclose(f);
		}
		if (!ok) { fprintf(stderr, "can't write '%s'\n", out); rc = 1; }
	}
	free(data);
	free(gpu_ns);
	skr_buffer_destroy(&blocks);

done:
	if (skr_tex_is_valid(&source)) skr_tex_destroy(&source);
	sk_texenc_shutdown();
	skr_shutdown();
	return rc;
}
