// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Shader stage creation
///////////////////////////////////////////////////////////////////////////////

skr_shader_stage_t _skr_shader_stage_create(VkDevice device, const void* shader_data, uint32_t shader_size, skr_stage_ type) {
	skr_shader_stage_t stage = {0};
	stage.type               = type;

	VkShaderModuleCreateInfo create_info = {
		.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = shader_size,
		.pCode    = (const uint32_t*)shader_data,
	};

	VkResult vr = vkCreateShaderModule(device, &create_info, NULL, &stage.shader);
	SKR_VK_CHECK_RET(vr, "vkCreateShaderModule", stage);

	return stage;
}

void _skr_shader_stage_destroy(skr_shader_stage_t* ref_stage) {
	if (!ref_stage) return;

	_skr_cmd_destroy_shader_module(NULL, ref_stage->shader);
	*ref_stage = (skr_shader_stage_t){0};
}

static skr_shader_stage_t _skr_shader_file_create_stage(VkDevice device, const sksc_shader_file_t* file, skr_stage_ stage_type) {
	for (uint32_t i = 0; i < file->stage_count; i++) {
		if (file->stages[i].stage == stage_type && file->stages[i].code_size > 0)
			return _skr_shader_stage_create(device, file->stages[i].code, file->stages[i].code_size, stage_type);
	}
	skr_shader_stage_t empty = {0};
	return empty;
}

///////////////////////////////////////////////////////////////////////////////
// Shader creation
///////////////////////////////////////////////////////////////////////////////

skr_shader_t _skr_shader_create_manual(const sksc_shader_meta_t* meta, skr_shader_stage_t v_shader,
                                       skr_shader_stage_t p_shader, skr_shader_stage_t c_shader) {
	skr_shader_t shader  = {0};
	if (meta) shader.meta = *meta;
	shader.vertex_stage  = v_shader;
	shader.pixel_stage   = p_shader;
	shader.compute_stage = c_shader;

	return shader;
}

skr_err_ skr_shader_create(const void* shader_data, uint32_t data_size, skr_shader_t* out_shader) {
	if (!out_shader) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_shader = (skr_shader_t){0};

	if (!shader_data || data_size == 0) {
		return skr_err_invalid_parameter;
	}

	sksc_shader_file_t file = {0};

	sksc_result_ result = sksc_shader_file_load_memory(shader_data, data_size, &file);
	if (result != sksc_result_success) {
		const char* extra;
		skr_err_ err = skr_err_failure;
		switch(result) {
			case sksc_result_bad_format:    extra = "unrecognized format"; err = skr_err_unsupported; break;
			case sksc_result_old_version:   extra = "old version";         err = skr_err_unsupported; break;
			case sksc_result_out_of_memory: extra = "out of memory";       err = skr_err_out_of_memory; break;
			case sksc_result_corrupt_data:  extra = "corrupt data";        err = skr_err_invalid_parameter; break;
			default:                        extra = "unknown";             err = skr_err_failure; break;
		}
		skr_log(skr_log_critical, "Failed to load shader file: %s", extra);
		return err;
	}

	// Create shader stages from the file
	skr_shader_stage_t v_stage = _skr_shader_file_create_stage(_skr_vk.device, &file, skr_stage_vertex);
	skr_shader_stage_t p_stage = _skr_shader_file_create_stage(_skr_vk.device, &file, skr_stage_pixel);
	skr_shader_stage_t c_stage = _skr_shader_file_create_stage(_skr_vk.device, &file, skr_stage_compute);

	// Move meta ownership to the shader, zero file.meta so
	// sksc_shader_file_destroy doesn't double-free.
	sksc_shader_meta_t meta = file.meta;
	file.meta = (sksc_shader_meta_t){0};
	sksc_shader_file_destroy(&file);

	*out_shader = _skr_shader_create_manual(&meta, v_stage, p_stage, c_stage);

	return skr_err_success;
}

bool skr_shader_is_valid(const skr_shader_t* shader) {
	if (!shader) return false;
	return
		shader->vertex_stage.shader  != VK_NULL_HANDLE ||
		shader->pixel_stage.shader   != VK_NULL_HANDLE ||
		shader->compute_stage.shader != VK_NULL_HANDLE;
}

void skr_shader_destroy(skr_shader_t* ref_shader) {
	if (!ref_shader) return;

	_skr_shader_stage_destroy(&ref_shader->vertex_stage);
	_skr_shader_stage_destroy(&ref_shader->pixel_stage);
	_skr_shader_stage_destroy(&ref_shader->compute_stage);

	sksc_shader_meta_free(&ref_shader->meta);
	*ref_shader = (skr_shader_t){0};
}

skr_bind_t skr_shader_get_bind(const skr_shader_t* shader, const char* bind_name) {
	if (!shader) {
		skr_bind_t empty = {0};
		return empty;
	}

	return sksc_shader_meta_get_bind(&shader->meta, bind_name);
}

bool skr_shader_get_param_info(const skr_shader_t* shader, const char* param_name, skr_shader_param_info_t* opt_out_info) {
	if (!shader) return false;

	int32_t idx = sksc_shader_meta_get_var_index(&shader->meta, param_name);
	if (idx < 0) return false;

	if (opt_out_info) {
		const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(&shader->meta, idx);
		*opt_out_info = (skr_shader_param_info_t){
			.type  = var->type,
			.count = var->type_count,
			.size  = var->size,
		};
	}
	return true;
}

bool skr_shader_get_tex_info(const skr_shader_t* shader, const char* tex_name, skr_shader_tex_info_t* opt_out_info) {
	if (!shader) return false;

	uint64_t hash = skr_hash(tex_name);
	for (uint32_t i = 0; i < shader->meta.resource_count; i++) {
		if (shader->meta.resources[i].name_hash == hash) {
			if (opt_out_info) {
				*opt_out_info = (skr_shader_tex_info_t){
					.default_value = shader->meta.resources[i].value,
				};
			}
			return true;
		}
	}
	return false;
}

void skr_shader_set_name(skr_shader_t* ref_shader, const char* name) {
	if (!ref_shader) return;

	char stage_name[256];

	if (ref_shader->vertex_stage.shader != VK_NULL_HANDLE) {
		snprintf(stage_name, sizeof(stage_name), "%s_vert", name);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)ref_shader->vertex_stage.shader, stage_name);
	}
	if (ref_shader->pixel_stage.shader != VK_NULL_HANDLE) {
		snprintf(stage_name, sizeof(stage_name), "%s_frag", name);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)ref_shader->pixel_stage.shader, stage_name);
	}
	if (ref_shader->compute_stage.shader != VK_NULL_HANDLE) {
		snprintf(stage_name, sizeof(stage_name), "%s_comp", name);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)ref_shader->compute_stage.shader, stage_name);
	}
}

// Shader defaults with user overrides applied by name. out_values is indexed
// by meta order, matching the map entries built at pipeline creation.
void _skr_shader_resolve_spec_constants(const sksc_shader_meta_t* meta, const skr_spec_constant_t* specs, uint32_t spec_count, uint32_t out_values[SKR_MAX_SPEC_CONSTANTS]) {
	memset(out_values, 0, sizeof(uint32_t) * SKR_MAX_SPEC_CONSTANTS);

	uint32_t count = meta->spec_constant_count;
	if (count > SKR_MAX_SPEC_CONSTANTS) {
		skr_log(skr_log_warning, "Shader '%s' has %u spec constants, max is %d; extras keep their defaults", meta->name, count, SKR_MAX_SPEC_CONSTANTS);
		count = SKR_MAX_SPEC_CONSTANTS;
	}
	for (uint32_t i = 0; i < count; i++)
		out_values[i] = meta->spec_constants[i].default_value;

	for (uint32_t s = 0; s < spec_count; s++) {
		if (!specs[s].name) continue;

		uint64_t hash  = skr_hash(specs[s].name);
		int32_t  found = -1;
		for (uint32_t i = 0; i < count; i++) {
			if (meta->spec_constants[i].name_hash == hash) { found = (int32_t)i; break; }
		}
		if (found < 0) {
			skr_log(skr_log_warning, "Spec constant '%s' not found in shader '%s'", specs[s].name, meta->name);
			continue;
		}

		switch (meta->spec_constants[found].type) {
			case sksc_shader_var_uint:  { uint32_t v = (uint32_t)specs[s].value; memcpy(&out_values[found], &v, sizeof(v)); } break;
			case sksc_shader_var_float: { float    v = (float   )specs[s].value; memcpy(&out_values[found], &v, sizeof(v)); } break;
			default:                    { int32_t  v = (int32_t )specs[s].value; memcpy(&out_values[found], &v, sizeof(v)); } break;
		}
	}
}

const VkSpecializationInfo* _skr_shader_make_spec_info(const sksc_shader_meta_t* meta, const uint32_t* spec_values, VkSpecializationMapEntry out_entries[SKR_MAX_SPEC_CONSTANTS], VkSpecializationInfo* out_info) {
	uint32_t spec_count = meta->spec_constant_count < SKR_MAX_SPEC_CONSTANTS ? meta->spec_constant_count : SKR_MAX_SPEC_CONSTANTS;
	if (spec_count == 0) return NULL;

	// Each 32-bit value is packed contiguously; offset i*4 matches the layout
	// _skr_shader_resolve_spec_constants writes into spec_values.
	for (uint32_t i = 0; i < spec_count; i++) {
		out_entries[i] = (VkSpecializationMapEntry){
			.constantID = meta->spec_constants[i].constant_id,
			.offset     = i * (uint32_t)sizeof(uint32_t),
			.size       = sizeof(uint32_t),
		};
	}
	*out_info = (VkSpecializationInfo){
		.mapEntryCount = spec_count,
		.pMapEntries   = out_entries,
		.dataSize      = spec_count * sizeof(uint32_t),
		.pData         = spec_values,
	};
	return out_info;
}

// Returns pointer to the immutable sampler for a given binding slot, or NULL if none.
// The returned pointer is stable (points into the caller's array) for use in pImmutableSamplers.
static const VkSampler* _skr_find_immutable_sampler(int32_t slot, const VkSampler* samplers, const int32_t* slots, int32_t count) {
	for (int32_t i = 0; i < count; i++) {
		if (slots[i] == slot) return &samplers[i];
	}
	return NULL;
}

VkDescriptorSetLayout _skr_shader_make_layout(VkDevice device, bool has_push_descriptors, const sksc_shader_meta_t* meta, skr_stage_ stage_mask, const VkSampler* immutable_samplers, const int32_t* immutable_sampler_slots, int32_t immutable_sampler_count) {
	if (meta->buffer_count == 0 && meta->resource_count == 0) {
		return VK_NULL_HANDLE;
	}

	VkDescriptorSetLayoutBinding bindings[32];
	uint32_t                     binding_count = 0;

	// Add buffer bindings
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		skr_bind_t bind = meta->buffers[i].bind;
		bind.stage_bits = bind.stage_bits & stage_mask;
		if (!bind.stage_bits) continue;

		// Determine descriptor type based on register type
		VkDescriptorType desc_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
		switch (bind.register_type) {
			case skr_register_constant:         desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         break;
			case skr_register_texture:          desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; break;
			case skr_register_read_buffer:      desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         break; // (StructuredBuffer)
			case skr_register_readwrite:        desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         break; // (RWStructuredBuffer)
			case skr_register_readwrite_tex:    desc_type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          break; // (RWTexture)
			case skr_register_input_attachment: desc_type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;       break; // (SubpassInput)
			default:                            desc_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;               break;
		}

		VkShaderStageFlags stages = 0;
		if (bind.stage_bits & skr_stage_vertex ) stages |= VK_SHADER_STAGE_VERTEX_BIT;
		if (bind.stage_bits & skr_stage_pixel  ) stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
		if (bind.stage_bits & skr_stage_compute) stages |= VK_SHADER_STAGE_COMPUTE_BIT;

		bindings[binding_count++] = (VkDescriptorSetLayoutBinding){
			.binding            = bind.slot,
			.descriptorType     = desc_type,
			.descriptorCount    = 1,
			.stageFlags         = stages,
			.pImmutableSamplers = _skr_find_immutable_sampler(bind.slot, immutable_samplers, immutable_sampler_slots, immutable_sampler_count),
		};
	}

	// Add resource bindings (textures and storage buffers)
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		skr_bind_t bind = meta->resources[i].bind;
		bind.stage_bits = bind.stage_bits & stage_mask;
		if (!bind.stage_bits) continue;

		// Determine descriptor type based on register type
		VkDescriptorType desc_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
		switch (bind.register_type) {
			case skr_register_constant:         desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         break;
			case skr_register_texture:          desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; break;
			case skr_register_read_buffer:      desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         break; // (StructuredBuffer)
			case skr_register_readwrite:        desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         break; // (RWStructuredBuffer)
			case skr_register_readwrite_tex:    desc_type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          break; // (RWTexture)
			case skr_register_input_attachment: desc_type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;       break; // (SubpassInput)
			default:                            desc_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;               break;
		}

		VkShaderStageFlags stages = 0;
		if (bind.stage_bits & skr_stage_vertex ) stages |= VK_SHADER_STAGE_VERTEX_BIT;
		if (bind.stage_bits & skr_stage_pixel  ) stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
		if (bind.stage_bits & skr_stage_compute) stages |= VK_SHADER_STAGE_COMPUTE_BIT;

		const VkSampler* found_sampler = _skr_find_immutable_sampler(bind.slot, immutable_samplers, immutable_sampler_slots, immutable_sampler_count);
		bindings[binding_count++] = (VkDescriptorSetLayoutBinding){
			.binding            = bind.slot,
			.descriptorType     = desc_type,
			.descriptorCount    = 1,
			.stageFlags         = stages,
			.pImmutableSamplers = found_sampler,
		};
	}

	VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.flags        = has_push_descriptors ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
		.bindingCount = binding_count,
		.pBindings    = bindings,
	};

	VkDescriptorSetLayout layout;
	VkResult vr = vkCreateDescriptorSetLayout(device, &layout_info, NULL, &layout);
	SKR_VK_CHECK_RET(vr, "vkCreateDescriptorSetLayout", VK_NULL_HANDLE);

	return layout;
}
