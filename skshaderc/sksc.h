// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include <sksc_file.h>

// Symbol visibility macros
#if defined(_WIN32) || defined(_WIN64)
	#ifdef SKSC_BUILD_SHARED
		#define SKSC_API __declspec(dllexport)
	#else
		#define SKSC_API
	#endif
#elif defined(__GNUC__) || defined(__clang__)
	#define SKSC_API __attribute__((visibility("default")))
#else
	#define SKSC_API
#endif

///////////////////////////////////////////

typedef struct sksc_settings_t {
	bool        debug;
	bool        silent_info;
	bool        silent_err;
	bool        silent_warn;
	int32_t     optimize;       // 0=none, 1=size, 2+=performance
	char        vs_entrypoint[64];
	char        ps_entrypoint[64];
	char        cs_entrypoint[64];
	char**      include_folders;
	int32_t     include_folder_ct;
	bool        target_langs[6]; // Indexed by skr_shader_lang_
	bool        use_svsl;        // compile with the SVSL backend instead of glslang
} sksc_settings_t;

typedef struct sksc_log_item_t {
	int32_t     level;
	int32_t     line;
	int32_t     column;
	const char* text;
} sksc_log_item_t;

typedef enum sksc_log_level_ {
	sksc_log_level_info,
	sksc_log_level_warn,
	sksc_log_level_err,
	sksc_log_level_err_pre,
} sksc_log_level_;

///////////////////////////////////////////

SKSC_API void            sksc_init            (void);
SKSC_API void            sksc_shutdown        (void);
SKSC_API bool            sksc_include_walk    (const char *filename, const sksc_settings_t *settings, void (*on_dep)(void *user, const char *path), void *user);
SKSC_API bool            sksc_compile         (const char *filename, const char *hlsl_text, sksc_settings_t *settings, sksc_shader_file_t *out_file);
// SPIR-V stages are stored SMOL-V encoded (smolv.h) and decoded by the loader.
// keep_debug_names keeps OpName/OpMemberName, ~8% of a module, for debug builds.
SKSC_API void            sksc_build_file      (const sksc_shader_file_t *file, bool keep_debug_names, void **out_data, uint32_t *out_size);
SKSC_API char*           sksc_shader_file_info(const sksc_shader_file_t *file); // Returns malloc'd string, caller must free

SKSC_API void            sksc_log        (sksc_log_level_ level, const char* text, ...);
SKSC_API void            sksc_log_at     (sksc_log_level_ level, int32_t line, int32_t column, const char *text, ...);
SKSC_API void            sksc_log_print  (const char* file, const sksc_settings_t* settings);
SKSC_API void            sksc_log_clear  (void);
SKSC_API int32_t         sksc_log_count  (void);
SKSC_API sksc_log_item_t sksc_log_get    (int32_t index);

#ifdef __cplusplus
}
#endif