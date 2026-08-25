#pragma once

#ifndef _CRT_INTERNAL_NONSTDC_NAMES
#define _CRT_INTERNAL_NONSTDC_NAMES 1 // must beat MSVC's sys/stat.h to the unprefixed S_IF* names
#endif
#include <sys/stat.h>
#include <sksc_file.h>

#include "sksc.h"
#include "array.h"

// MSVC's sys/stat.h has no S_ISREG/S_ISDIR, and falls back to the underscored
// constants when something included it before the define above landed.
#if !defined(S_ISREG)
  #if   defined(S_IFMT)  && defined(S_IFREG)
    #define S_ISREG(m) (((m) &  S_IFMT) ==  S_IFREG)
  #elif defined(_S_IFMT) && defined(_S_IFREG)
    #define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
  #endif
#endif
#if !defined(S_ISDIR)
  #if   defined(S_IFMT)  && defined(S_IFDIR)
    #define S_ISDIR(m) (((m) &  S_IFMT) ==  S_IFDIR)
  #elif defined(_S_IFMT) && defined(_S_IFDIR)
    #define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
  #endif
#endif

enum compile_result_ {
	compile_result_success = 1,
	compile_result_fail    = 0,
	compile_result_skip    = -1,
};

struct sksc_meta_item_t {
	char name [32];
	char tag  [64];
	char value[512];
	int32_t row, col;
};

struct sksc_ast_default_t {
	char    name[32];
	double  values[16];
	int32_t value_count;
};

// Shared, so a long path can't resolve for one caller and truncate for another
#define SKSC_PATH_MAX 2048

// Searches the requester's own folder, then each -i folder in order
bool                        sksc_include_resolve       (const char *path, const char *requester, const sksc_settings_t *settings, char *out_full, size_t full_size);
char                       *sksc_file_read             (const char *path, int32_t *opt_out_len); // malloc'd, NUL terminated

void                        sksc_glslang_init          ();
void                        sksc_glslang_shutdown      ();
compile_result_             sksc_hlsl_to_spirv         (const char *filename, const char *hlsl, const sksc_settings_t *settings, skr_stage_ type, sksc_shader_file_stage_t *out_stage);

array_t<sksc_ast_default_t> sksc_hlsl_find_initializers(const char *hlsl_text);
bool                        sksc_hlsl_to_bytecode      (const char *filename, const char *hlsl_text, const sksc_settings_t *settings, skr_stage_ type, sksc_shader_file_stage_t *out_stage);

array_t<sksc_meta_item_t>   sksc_meta_find_defaults    (const char *hlsl_text);
void                        sksc_meta_assign_defaults  (array_t<sksc_ast_default_t> ast_defaults, array_t<sksc_meta_item_t> comment_overrides, sksc_shader_meta_t *ref_meta);
bool                        sksc_meta_check_dup_buffers  (const sksc_shader_meta_t *ref_meta);
bool                        sksc_meta_check_dup_resources(const sksc_shader_meta_t *ref_meta, const char **out_name1, const char **out_name2, uint32_t *out_slot);
bool                        sksc_spirv_to_meta           (const sksc_shader_file_stage_t *spirv_stage, sksc_shader_meta_t *meta);

bool                        sksc_spirv_to_glsl         (const sksc_shader_file_stage_t *src_stage, const sksc_settings_t *settings, skr_shader_lang_ lang, sksc_shader_file_stage_t *out_stage, const sksc_shader_meta_t *meta, array_t<sksc_meta_item_t> var_meta);

#ifdef SKSC_HAS_SVSL
// Optional SVSL backend (skshaderc/sksc_svsl.cpp). Compiles the source with
// libsvsl instead of the glslang pipeline and fills out_file from SVSL's SKS
// container output.
bool                        sksc_svsl_compile          (const char *filename, const char *hlsl_text, const sksc_settings_t *settings, sksc_shader_file_t *out_file);
#endif
