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
// split_samplers compiles a WGSL-ready variant: HLSL SamplerState objects stay
// separate standalone samplers (shifted to s+400) instead of merging into
// combined image samplers. The Vulkan path always passes false.
compile_result_             sksc_hlsl_to_spirv         (const char *filename, const char *hlsl, const sksc_settings_t *settings, skr_stage_ type, const char** defines, int32_t define_count, bool split_samplers, sksc_shader_file_stage_t *out_stage);

// Device-feature mask scanned from a SPIR-V blob's OpCapability/OpExtension
// declarations (sksc_feature_bit_ values). Implemented in sksc.cpp.
uint64_t                    sksc_spirv_features        (const uint32_t *words, uint32_t word_count);
uint64_t                    sksc_wgsl_unsupported_features(void); // feature bits with no browser-WebGPU counterpart

// sksc_wgsl.cpp — WGSL stage generation for the WebGPU backend. Compiles a
// split-sampler SPIR-V variant, rewrites ViewIndex to the sk_view_index spec
// constant, reflects standalone samplers into ref_meta, and (when built with
// Tint, SKSC_HAS_TINT) emits validated WGSL text as out_stage. Returns skip
// (with a warning logged) for constructs WebGPU can't express — the .sks then
// simply carries no WGSL blob for the stage.
compile_result_             sksc_wgsl_compile_stage    (const char *filename, const char *hlsl, const sksc_settings_t *settings, skr_stage_ type, sksc_shader_meta_t *ref_meta, sksc_shader_file_stage_t *out_stage);
// SPIR-V -> WGSL directly (rewrites + Tint), for backends that already have
// final SPIR-V (SVSL). The SPIR-V must use split samplers or none at all.
compile_result_             sksc_wgsl_from_spirv       (const uint32_t *words, uint32_t word_count, skr_stage_ type, sksc_shader_meta_t *ref_meta, sksc_shader_file_stage_t *out_stage);

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
