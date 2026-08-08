// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Conformance runner for the Khronos KTX-Software-CTS corpus.
//
//   ktx2_cts [--expect reject|accept|cts] [--quiet] [--no-gltf] [--caps bc,etc2,astc] <path>...
//
// Runs every .ktx2 under each path through open, the glTF check and plan, and
// reports where it stopped. Headless, which is most of why sk_ktx2 has no
// sk_renderer dependency.
//
// --no-gltf skips the conformance check. That matters: the CTS's R and RG ETC1S
// references carry KTXswizzle "r001"/"ra01" so are not glTF-legal, yet they are
// the only real two-slice files to test the RGBA-vs-RG distinction against.
//
// --expect cts reads the corpus's own expectation off the filename: error_* and
// fatal_* must be rejected, warning_* may be accepted. A surviving error_* file
// is a validation hole. Run it under ASan.

#include "sk_ktx2.h"
#include "host_zstd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
	#include <windows.h>
#else
	#include <dirent.h>
	#include <sys/stat.h>
#endif

typedef enum stage_ {
	stage_open = 0,
	stage_gltf,
	stage_plan,
	stage_transcode,
	stage_accepted,
	stage_count,
} stage_;

static const char* k_stage_name[stage_count] = {
	"open", "gltf", "plan", "transcode", "accepted"
};

typedef struct stats_t {
	int32_t files;
	int32_t unreadable;
	int32_t accepted;
	int32_t stopped_at [stage_count];
	int32_t by_result  [8];
	int32_t violations;
} stats_t;

typedef enum expect_ {
	expect_none = 0,
	expect_reject,
	expect_accept,
	expect_cts,   // read it off the CTS filename prefix
} expect_;

#define ALLOW_MAX 8

typedef struct options_t {
	bool        quiet;
	bool        skip_gltf;   // exercise the transcode path on non-glTF sources
	bool        transcode;   // actually decode, not just plan
	expect_     expect;
	ktx2_caps_  caps;
	const char* allow[ALLOW_MAX]; // filename substrings exempt from --expect
	int32_t     allow_count;
} options_t;

static const char* base_name(const char* path) {
	const char* name = path;
	for (const char* c = path; *c; c++)
		if (*c == '/' || *c == '\\') name = c + 1;
	return name;
}

// error_/fatal_ must be rejected; warning_ is merely unusual, and libktx accepts
// those too.
static expect_ expect_for(const options_t* opt, const char* path) {
	const char* name = base_name(path);
	for (int32_t i = 0; i < opt->allow_count; i++)
		if (strstr(name, opt->allow[i]) != NULL) return expect_none;

	if (opt->expect != expect_cts) return opt->expect;
	if (strncmp(name, "error_", 6) == 0 || strncmp(name, "fatal_", 6) == 0) return expect_reject;
	return expect_none;
}

///////////////////////////////////////////////////////////////////////////////

static uint8_t* file_read(const char* path, size_t* out_bytes) {
	FILE* f = fopen(path, "rb");
	if (f == NULL) return NULL;
	if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
	long size = ftell(f);
	if (size < 0) { fclose(f); return NULL; }
	rewind(f);

	uint8_t* data = (uint8_t*)malloc((size_t)size ? (size_t)size : 1);
	if (data == NULL) { fclose(f); return NULL; }
	if (fread(data, 1, (size_t)size, f) != (size_t)size) { free(data); fclose(f); return NULL; }
	fclose(f);
	*out_bytes = (size_t)size;
	return data;
}

static bool ends_with_ktx2(const char* name) {
	size_t len = strlen(name);
	return len > 5 && strcmp(name + len - 5, ".ktx2") == 0;
}

static void check_file(const char* path, const options_t* opt, stats_t* ref_stats) {
	size_t   bytes = 0;
	uint8_t* data  = file_read(path, &bytes);
	if (data == NULL) {
		printf("UNREADABLE  %s\n", path);
		ref_stats->unreadable++;
		return;
	}
	ref_stats->files++;

	ktx2_reader_t reader;
	ktx2_plan_t   plan;
	stage_        stopped = stage_accepted;
	ktx2_result_  result  = ktx2_open(data, bytes, &reader);
	if (result != ktx2_result_success) {
		stopped = stage_open;
	} else {
		result = opt->skip_gltf ? ktx2_result_success : ktx2_check_gltf_basisu(&reader);
		if (result != ktx2_result_success) {
			stopped = stage_gltf;
		} else {
			result = ktx2_plan(&reader, &k_host_context, opt->caps, &plan);
			if (result != ktx2_result_success) {
				stopped = stage_plan;
			} else if (opt->transcode) {
				// Scratch sized at exactly plan.scratch_bytes. The goldens use the
				// internal-malloc path, so this is what holds that figure to being
				// sufficient: too small surfaces as buffer_too_small.
				void* output  = malloc(plan.data_bytes    ? plan.data_bytes    : 1);
				void* scratch = malloc(plan.scratch_bytes ? plan.scratch_bytes : 1);
				result = output != NULL && scratch != NULL
					? ktx2_transcode(&plan, output, plan.data_bytes, scratch)
					: ktx2_result_buffer_too_small;
				if (result != ktx2_result_success) stopped = stage_transcode;
				free(scratch);
				free(output);
			}
		}
	}

	ref_stats->stopped_at[stopped]++;
	if ((size_t)result < sizeof(ref_stats->by_result) / sizeof(ref_stats->by_result[0]))
		ref_stats->by_result[result]++;

	bool    accepted = stopped == stage_accepted;
	expect_ expect   = expect_for(opt, path);
	if (accepted) ref_stats->accepted++;

	bool violation = (expect == expect_reject &&  accepted)
	              || (expect == expect_accept && !accepted);
	if (violation) ref_stats->violations++;

	if (!opt->quiet || violation) {
		if (accepted) {
			ktx2_info_t info = ktx2_get_info(&reader);
			printf("%-9s %-8s  %-14s %4dx%-4d mips=%-2d -> %-18s %8zu B  %s\n",
				violation ? "VIOLATION" : "accepted", k_stage_name[stopped],
				ktx2_source_str(info.source), info.width, info.height, info.mip_count,
				ktx2_fmt_str(plan.format), plan.data_bytes, path);
		} else {
			printf("%-9s %-8s  %-38s %s\n",
				violation ? "VIOLATION" : "rejected", k_stage_name[stopped],
				ktx2_result_str(result), path);
		}
	}
	free(data);
}

///////////////////////////////////////////////////////////////////////////////

static void check_path(const char* path, const options_t* opt, stats_t* ref_stats);

#if defined(_WIN32) || defined(_WIN64)

static bool path_is_dir(const char* path) {
	DWORD attr = GetFileAttributesA(path);
	return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
}

static void walk_dir(const char* path, const options_t* opt, stats_t* ref_stats) {
	char pattern[1024];
	snprintf(pattern, sizeof(pattern), "%s\\*", path);

	WIN32_FIND_DATAA entry;
	HANDLE           find = FindFirstFileA(pattern, &entry);
	if (find == INVALID_HANDLE_VALUE) return;
	do {
		if (strcmp(entry.cFileName, ".") == 0 || strcmp(entry.cFileName, "..") == 0) continue;
		char child[1024];
		snprintf(child, sizeof(child), "%s\\%s", path, entry.cFileName);
		check_path(child, opt, ref_stats);
	} while (FindNextFileA(find, &entry));
	FindClose(find);
}

#else

static bool path_is_dir(const char* path) {
	struct stat info;
	return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
}

static void walk_dir(const char* path, const options_t* opt, stats_t* ref_stats) {
	DIR* dir = opendir(path);
	if (dir == NULL) return;

	struct dirent* entry;
	while ((entry = readdir(dir)) != NULL) {
		if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
		char child[1024];
		snprintf(child, sizeof(child), "%s/%s", path, entry->d_name);
		check_path(child, opt, ref_stats);
	}
	closedir(dir);
}

#endif

static void check_path(const char* path, const options_t* opt, stats_t* ref_stats) {
	if (path_is_dir(path))          walk_dir  (path, opt, ref_stats);
	else if (ends_with_ktx2(path))  check_file(path, opt, ref_stats);
}

///////////////////////////////////////////////////////////////////////////////

static ktx2_caps_ parse_caps(const char* text) {
	ktx2_caps_ caps = ktx2_caps_none;
	if (strstr(text, "bc"      )) caps |= ktx2_caps_bc;
	if (strstr(text, "etc2"    )) caps |= ktx2_caps_etc2;
	if (strstr(text, "astc"    )) caps |= ktx2_caps_astc_ldr;
	if (strstr(text, "astc_hdr")) caps |= ktx2_caps_astc_hdr;
	return caps;
}

int main(int argc, char** argv) {
	options_t opt = {0};
	opt.caps = ktx2_caps_bc | ktx2_caps_etc2 | ktx2_caps_astc_ldr;

	int first_path = argc;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--quiet") == 0) {
			opt.quiet = true;
		} else if (strcmp(argv[i], "--no-gltf") == 0) {
			opt.skip_gltf = true;
		} else if (strcmp(argv[i], "--transcode") == 0) {
			opt.transcode = true;
		} else if (strcmp(argv[i], "--expect") == 0 && i + 1 < argc) {
			const char* mode = argv[++i];
			if      (strcmp(mode, "accept") == 0) opt.expect = expect_accept;
			else if (strcmp(mode, "reject") == 0) opt.expect = expect_reject;
			else if (strcmp(mode, "cts"   ) == 0) opt.expect = expect_cts;
			else { fprintf(stderr, "unknown --expect mode '%s'\n", mode); return 2; }
		} else if (strcmp(argv[i], "--caps") == 0 && i + 1 < argc) {
			opt.caps = parse_caps(argv[++i]);
		} else if (strcmp(argv[i], "--allow") == 0 && i + 1 < argc) {
			if (opt.allow_count >= ALLOW_MAX) { fprintf(stderr, "too many --allow entries\n"); return 2; }
			opt.allow[opt.allow_count++] = argv[++i];
		} else {
			first_path = i;
			break;
		}
	}
	if (first_path >= argc) {
		fprintf(stderr, "usage: ktx2_cts [--expect reject|accept|cts] [--allow <substr>] [--quiet] [--no-gltf] [--caps bc,etc2,astc] <path>...\n");
		return 2;
	}

	stats_t stats = {0};
	for (int i = first_path; i < argc; i++) check_path(argv[i], &opt, &stats);

	printf("\n%d files, %d accepted, %d rejected", stats.files, stats.accepted, stats.files - stats.accepted);
	if (stats.unreadable) printf(", %d unreadable", stats.unreadable);
	printf("\n  stopped at:");
	for (int32_t i = 0; i < stage_count; i++) printf(" %s=%d", k_stage_name[i], stats.stopped_at[i]);
	printf("\n");

	if (opt.expect != expect_none) printf("  violations: %d\n", stats.violations);
	if (stats.unreadable > 0 || stats.violations > 0) return 1;
	return 0;
}
