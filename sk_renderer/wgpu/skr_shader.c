// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////
// Shader stage creation — WGSL text stages from the .sks container (v12).
// Each stage was compiled separately, so its module holds a single entry
// point and pipelines can leave entryPoint unset.

static skr_shader_stage_t _skr_shader_file_create_stage(const sksc_shader_file_t* file, skr_stage_ stage_type) {
	skr_shader_stage_t stage = { .type = stage_type };

	for (uint32_t i = 0; i < file->stage_count; i++) {
		if (file->stages[i].stage != stage_type || file->stages[i].language != skr_shader_lang_wgsl || file->stages[i].code_size == 0)
			continue;

		const char* code = (const char*)file->stages[i].code;
		size_t      len  = strnlen(code, file->stages[i].code_size);

		// Only multiview-using stages declare the override; pipelines may
		// only pass constants that exist in the module. A plain substring
		// probe is safe here because the input is SVSL-generated WGSL — no
		// comments or string literals to false-positive on.
		stage.has_view_index = strstr(code, "sk_view_index") != NULL;

		WGPUShaderSourceWGSL source = {
			.chain = { .sType = WGPUSType_ShaderSourceWGSL },
			.code  = { code, len },
		};
		WGPUShaderModuleDescriptor desc = {
			.nextInChain = &source.chain,
			.label       = { file->meta.name, strnlen(file->meta.name, sizeof(file->meta.name)) },
		};
		stage.shader = wgpuDeviceCreateShaderModule(_skr_wgpu.device, &desc);
		return stage;
	}
	return stage;
}

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_shader_create(const void* shader_data, uint32_t data_size, skr_shader_t* out_shader) {
	if (out_shader == NULL) return skr_err_invalid_parameter;
	*out_shader = (skr_shader_t){0};
	if (shader_data == NULL || data_size == 0) return skr_err_invalid_parameter;

	sksc_shader_file_t file   = {0};
	sksc_result_       result = sksc_shader_file_load_memory(shader_data, data_size, &file);
	if (result != sksc_result_success) {
		const char* extra;
		skr_err_    err;
		switch (result) {
			case sksc_result_bad_format:    extra = "unrecognized format"; err = skr_err_unsupported;    break;
			case sksc_result_old_version:   extra = "old version";         err = skr_err_unsupported;    break;
			case sksc_result_out_of_memory: extra = "out of memory";       err = skr_err_out_of_memory;  break;
			case sksc_result_corrupt_data:  extra = "corrupt data";        err = skr_err_invalid_parameter; break;
			default:                        extra = "unknown";             err = skr_err_failure;        break;
		}
		skr_log(skr_log_critical, "Failed to load shader file: %s", extra);
		return err;
	}

	bool has_wgsl = false;
	for (uint32_t i = 0; i < file.stage_count; i++)
		if (file.stages[i].language == skr_shader_lang_wgsl) has_wgsl = true;
	if (!has_wgsl) {
		skr_log(skr_log_critical, "Shader '%s' has no WGSL stages — compile it with skshaderc '-t sw' (or the shader uses features WebGPU can't express)", file.meta.name);
		sksc_shader_file_destroy(&file);
		return skr_err_unsupported;
	}

	// A stage present in another language but missing from WGSL was skipped at
	// compile time (a construct WGSL can't express, e.g. a vertex stage writing
	// storage buffers). The shader can't run as authored, so creation fails —
	// which routes apps into the same fallback paths an unsupported Vulkan
	// feature would (e.g. the fog scatter QCOM -> taps fallback).
	for (uint32_t i = 0; i < file.stage_count; i++) {
		if (file.stages[i].language == skr_shader_lang_wgsl) continue;
		bool stage_has_wgsl = false;
		for (uint32_t j = 0; j < file.stage_count; j++)
			if (file.stages[j].language == skr_shader_lang_wgsl && file.stages[j].stage == file.stages[i].stage) stage_has_wgsl = true;
		if (!stage_has_wgsl) {
			skr_log(skr_log_warning, "Shader '%s' is missing its WGSL %s stage (skipped at compile — see skshaderc warnings), so it can't run on WebGPU",
				file.meta.name,
				file.stages[i].stage == skr_stage_vertex ? "vertex" : file.stages[i].stage == skr_stage_pixel ? "pixel" : "compute");
			sksc_shader_file_destroy(&file);
			return skr_err_unsupported;
		}
	}

	skr_shader_stage_t v_stage = _skr_shader_file_create_stage(&file, skr_stage_vertex);
	skr_shader_stage_t p_stage = _skr_shader_file_create_stage(&file, skr_stage_pixel);
	skr_shader_stage_t c_stage = _skr_shader_file_create_stage(&file, skr_stage_compute);

	// Move meta ownership to the shader; zero file.meta so destroy skips it
	out_shader->meta          = file.meta;
	file.meta                 = (sksc_shader_meta_t){0};
	out_shader->vertex_stage  = v_stage;
	out_shader->pixel_stage   = p_stage;
	out_shader->compute_stage = c_stage;
	sksc_shader_file_destroy(&file);

	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////

bool skr_shader_is_valid(const skr_shader_t* shader) {
	return shader != NULL &&
	       (shader->vertex_stage.shader != NULL || shader->compute_stage.shader != NULL);
}

void skr_shader_destroy(skr_shader_t* ref_shader) {
	if (ref_shader == NULL) return;
	if (ref_shader->vertex_stage .shader) wgpuShaderModuleRelease(ref_shader->vertex_stage .shader);
	if (ref_shader->pixel_stage  .shader) wgpuShaderModuleRelease(ref_shader->pixel_stage  .shader);
	if (ref_shader->compute_stage.shader) wgpuShaderModuleRelease(ref_shader->compute_stage.shader);
	sksc_shader_meta_free(&ref_shader->meta);
	memset(ref_shader, 0, sizeof(*ref_shader));
}

///////////////////////////////////////////////////////////////////////////////

skr_bind_t skr_shader_get_bind(const skr_shader_t* shader, const char* bind_name) {
	if (shader == NULL) { skr_bind_t empty = {0}; return empty; }
	return sksc_shader_meta_get_bind(&shader->meta, bind_name);
}

bool skr_shader_get_param_info(const skr_shader_t* shader, const char* param_name, skr_shader_param_info_t* opt_out_info) {
	if (opt_out_info) memset(opt_out_info, 0, sizeof(*opt_out_info));
	if (shader == NULL || param_name == NULL) return false;

	int32_t idx = sksc_shader_meta_get_var_index(&shader->meta, param_name);
	if (idx < 0) return false;
	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(&shader->meta, idx);
	if (var == NULL) return false;

	if (opt_out_info) {
		opt_out_info->type  = (sksc_shader_var_)var->type;
		opt_out_info->count = var->type_count;
		opt_out_info->size  = var->size;
	}
	return true;
}

bool skr_shader_get_tex_info(const skr_shader_t* shader, const char* tex_name, skr_shader_tex_info_t* opt_out_info) {
	if (opt_out_info) memset(opt_out_info, 0, sizeof(*opt_out_info));
	if (shader == NULL || tex_name == NULL) return false;

	uint64_t hash = skr_hash(tex_name);
	for (uint32_t i = 0; i < shader->meta.resource_count; i++) {
		if (shader->meta.resources[i].name_hash == hash) {
			if (opt_out_info) opt_out_info->default_value = shader->meta.resources[i].value;
			return true;
		}
	}
	return false;
}

void skr_shader_set_name(skr_shader_t* ref_shader, const char* name) {
	if (ref_shader == NULL || name == NULL) return;
	WGPUStringView sv = { name, strlen(name) };
	if (ref_shader->vertex_stage .shader) wgpuShaderModuleSetLabel(ref_shader->vertex_stage .shader, sv);
	if (ref_shader->pixel_stage  .shader) wgpuShaderModuleSetLabel(ref_shader->pixel_stage  .shader, sv);
	if (ref_shader->compute_stage.shader) wgpuShaderModuleSetLabel(ref_shader->compute_stage.shader, sv);
}
