// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// ASTC encoder validation tool
//
// Usage: validate <original.png> <ours.astc> [--reference <quality>] [--astcenc <path>]
//
// For each .astc input, shells out to astcenc to decode back to a PNG, then
// computes PSNR/SSIM vs the original. With --reference, also encodes the
// original with astcenc at the specified quality (e.g. "medium") to provide
// a "what a slow encoder gets" quality target.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "metrics.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
	#define PATH_SEP '\\'
	#define MKTMPDIR(s) _mkdir(s)
#else
	#include <unistd.h>
	#define PATH_SEP '/'
	#define MKTMPDIR(s) mkdir(s, 0700)
#endif

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

static const char* _find_astcenc(const char* override_path) {
	static const char* candidates[] = {
		NULL,  // filled from override
		"astcenc-avx2",
		"astcenc-sse4.1",
		"astcenc-sse2",
		"astcenc",
		"/home/koujaku/Apps/astcenc/astcenc-avx2",
		"/home/koujaku/Apps/astcenc/astcenc-sse4.1",
		"/home/koujaku/Apps/astcenc/astcenc-sse2",
		NULL
	};
	candidates[0] = override_path ? override_path : getenv("ASTCENC_BIN");

	for (int i = 0; i < (int)(sizeof(candidates) / sizeof(*candidates)); i++) {
		if (!candidates[i]) continue;
		// Probe with -h (short help — exits 0; --help exits 1 on astcenc).
		char cmd[1024];
		snprintf(cmd, sizeof(cmd), "\"%s\" -h >/dev/null 2>&1", candidates[i]);
		if (system(cmd) == 0) return candidates[i];
	}
	return NULL;
}

static const char* _basename_no_ext(const char* path, char* out, size_t out_size) {
	const char* sl = strrchr(path, '/');
	const char* bs = strrchr(path, '\\');
	const char* n  = path;
	if (sl && sl >= n) n = sl + 1;
	if (bs && bs >= n) n = bs + 1;
	strncpy(out, n, out_size - 1);
	out[out_size - 1] = '\0';
	char* dot = strrchr(out, '.');
	if (dot) *dot = '\0';
	return out;
}

static int64_t _file_size(const char* path) {
	FILE* f = fopen(path, "rb");
	if (!f) return -1;
	fseek(f, 0, SEEK_END);
	long s = ftell(f);
	fclose(f);
	return s;
}

// Decode an .astc file to an RGBA PNG via astcenc. Returns true on success.
static bool _astcenc_decode(const char* astcenc, const char* astc_path, const char* png_path) {
	char cmd[2048];
	snprintf(cmd, sizeof(cmd), "\"%s\" -dl \"%s\" \"%s\" >/dev/null 2>&1", astcenc, astc_path, png_path);
	return system(cmd) == 0;
}

// Encode a PNG to an .astc file via astcenc. Block size is fixed at 6x6 since
// that's the only footprint we currently validate.
static bool _astcenc_encode(const char* astcenc, const char* png_path, const char* astc_path, const char* quality) {
	char cmd[2048];
	snprintf(cmd, sizeof(cmd), "\"%s\" -cl \"%s\" \"%s\" 6x6 -%s >/dev/null 2>&1", astcenc, png_path, astc_path, quality);
	return system(cmd) == 0;
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

static void _print_row(const char* label, int64_t bytes, const quality_metrics_t* m) {
	if (bytes >= 0) {
		printf("  %-24s  %8lld B   PSNR(RGB) %5.2f dB   PSNR(R/G/B) %5.2f/%5.2f/%5.2f   SSIM %.4f\n",
			label, (long long)bytes, m->psnr_rgb, m->psnr_r, m->psnr_g, m->psnr_b, m->ssim_rgb);
	} else {
		printf("  %-24s  (missing)\n", label);
	}
}

int main(int argc, char** argv) {
	if (argc < 3) {
		printf("Usage: validate <original.png> <ours.astc> [--reference <quality>] [--astcenc <path>]\n");
		printf("  quality: one of astcenc's presets (fastest, fast, medium, thorough, exhaustive)\n");
		printf("\n");
		printf("Environment:\n");
		printf("  ASTCENC_BIN  — override astcenc executable path\n");
		return 1;
	}

	const char* original_png = argv[1];
	const char* ours_astc    = argv[2];
	const char* reference_q  = NULL;
	const char* astcenc_over = NULL;

	for (int i = 3; i < argc; i++) {
		if      (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) reference_q  = argv[++i];
		else if (strcmp(argv[i], "--astcenc")   == 0 && i + 1 < argc) astcenc_over = argv[++i];
		else {
			fprintf(stderr, "Unknown argument: %s\n", argv[i]);
			return 1;
		}
	}

	const char* astcenc = _find_astcenc(astcenc_over);
	if (!astcenc) {
		fprintf(stderr, "Could not find astcenc. Pass --astcenc <path> or set ASTCENC_BIN.\n");
		return 1;
	}
	printf("Using astcenc: %s\n", astcenc);

	// Load original.
	int32_t w, h, c;
	uint8_t* orig = stbi_load(original_png, &w, &h, &c, 4);
	if (!orig) {
		fprintf(stderr, "Failed to load original: %s\n", original_png);
		return 1;
	}
	int64_t orig_bytes = (int64_t)w * h * 4;
	printf("\n%s  %dx%d  %lld bytes (RGBA8)\n\n", original_png, w, h, (long long)orig_bytes);

	// Scratch directory for intermediate PNGs.
	char tmp_dir[] = "astc_validate_tmp";
	MKTMPDIR(tmp_dir);

	char stem[256];
	_basename_no_ext(original_png, stem, sizeof(stem));

	// Decode our .astc and compare.
	char ours_png[512];
	snprintf(ours_png, sizeof(ours_png), "%s%c%s_ours.png", tmp_dir, PATH_SEP, stem);
	if (!_astcenc_decode(astcenc, ours_astc, ours_png)) {
		fprintf(stderr, "astcenc failed to decode %s\n", ours_astc);
		stbi_image_free(orig);
		return 1;
	}
	int32_t ow, oh, oc;
	uint8_t* ours = stbi_load(ours_png, &ow, &oh, &oc, 4);
	if (!ours || ow != w || oh != h) {
		fprintf(stderr, "Decoded size mismatch or load failure: %dx%d vs %dx%d\n", ow, oh, w, h);
		stbi_image_free(orig);
		if (ours) stbi_image_free(ours);
		return 1;
	}
	quality_metrics_t ours_m = quality_compare_rgba8(orig, ours, w, h);
	int64_t ours_bytes = _file_size(ours_astc);
	_print_row("ours.astc", ours_bytes, &ours_m);

	// Optional reference encode from astcenc itself.
	if (reference_q) {
		char ref_astc[512], ref_png[512];
		snprintf(ref_astc, sizeof(ref_astc), "%s%c%s_ref.astc", tmp_dir, PATH_SEP, stem);
		snprintf(ref_png,  sizeof(ref_png),  "%s%c%s_ref.png",  tmp_dir, PATH_SEP, stem);
		if (!_astcenc_encode(astcenc, original_png, ref_astc, reference_q)) {
			fprintf(stderr, "astcenc failed to encode reference\n");
		} else if (!_astcenc_decode(astcenc, ref_astc, ref_png)) {
			fprintf(stderr, "astcenc failed to decode reference\n");
		} else {
			int32_t rw, rh, rc;
			uint8_t* ref = stbi_load(ref_png, &rw, &rh, &rc, 4);
			if (ref && rw == w && rh == h) {
				quality_metrics_t ref_m = quality_compare_rgba8(orig, ref, w, h);
				int64_t ref_bytes = _file_size(ref_astc);
				char label[64];
				snprintf(label, sizeof(label), "astcenc-%s.astc", reference_q);
				_print_row(label, ref_bytes, &ref_m);

				printf("\n  delta (ours - ref):       PSNR %+5.2f dB   SSIM %+.4f\n",
					ours_m.psnr_rgb - ref_m.psnr_rgb,
					ours_m.ssim_rgb - ref_m.ssim_rgb);
			}
			if (ref) stbi_image_free(ref);
		}
	}

	stbi_image_free(orig);
	stbi_image_free(ours);
	printf("\n");
	return 0;
}
