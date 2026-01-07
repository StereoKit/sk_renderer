#pragma once

#include <sksc_file.h>

#include "sksc.h"
#include "array.h"

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

void                        sksc_glslang_init          ();
void                        sksc_glslang_shutdown      ();
compile_result_             sksc_hlsl_to_spirv         (const char *filename, const char *hlsl, const sksc_settings_t *settings, skr_stage_ type, const char** defines, int32_t define_count, sksc_shader_file_stage_t *out_stage);

array_t<sksc_ast_default_t> sksc_hlsl_find_initializers(const char *hlsl_text);
bool                        sksc_hlsl_to_bytecode      (const char *filename, const char *hlsl_text, const sksc_settings_t *settings, skr_stage_ type, sksc_shader_file_stage_t *out_stage);

array_t<sksc_meta_item_t>   sksc_meta_find_defaults    (const char *hlsl_text);
void                        sksc_meta_assign_defaults  (array_t<sksc_ast_default_t> ast_defaults, array_t<sksc_meta_item_t> comment_overrides, sksc_shader_meta_t *ref_meta);
bool                        sksc_meta_check_dup_buffers  (const sksc_shader_meta_t *ref_meta);
bool                        sksc_meta_check_dup_resources(const sksc_shader_meta_t *ref_meta, const char **out_name1, const char **out_name2, uint32_t *out_slot);
bool                        sksc_spirv_to_meta           (const sksc_shader_file_stage_t *spirv_stage, sksc_shader_meta_t *meta);

bool                        sksc_spirv_to_glsl         (const sksc_shader_file_stage_t *src_stage, const sksc_settings_t *settings, skr_shader_lang_ lang, sksc_shader_file_stage_t *out_stage, const sksc_shader_meta_t *meta, array_t<sksc_meta_item_t> var_meta);
