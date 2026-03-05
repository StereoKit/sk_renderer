// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// SPIRV patching
///////////////////////////////////////////////////////////////////////////////

// SPIRV opcodes
#define SPIRV_OP_EXTENSION  10
#define SPIRV_OP_CAPABILITY 17
#define SPIRV_OP_VARIABLE   59
#define SPIRV_OP_DECORATE   71

// SPIRV capabilities
#define SPIRV_CAPABILITY_GEOMETRY        2
#define SPIRV_CAPABILITY_VIEWPORT_LAYER  5254

// SPIRV decorations
#define SPIRV_DECORATION_BUILTIN  11
#define SPIRV_DECORATION_FLAT     14
#define SPIRV_DECORATION_LOCATION 30

// SPIRV built-ins
#define SPIRV_BUILTIN_PRIMITIVE_ID 7
#define SPIRV_BUILTIN_LAYER        9

// SPIRV storage classes
#define SPIRV_STORAGE_INPUT  1
#define SPIRV_STORAGE_OUTPUT 3

// Convert the Layer built-in to a regular Location-decorated flat
// interpolant when VK_EXT_shader_viewport_index_layer is absent. The
// variable, its stores, loads, and entry-point reference all remain
// intact — only the decoration changes so the value passes between
// stages as a flat varying. The buffer must have room for up to 3 extra
// words (one Flat decoration instruction). Returns the new word count.
static uint32_t _skr_spirv_strip_viewport_layer(uint32_t* code, uint32_t word_count) {
	// --- Phase 1: Analyze ---

	// Scan capabilities
	bool has_vl_cap   = false;
	bool has_geom_cap = false;
	for (uint32_t i = 5; i < word_count; ) {
		uint32_t opcode   = code[i] & 0xFFFF;
		uint32_t word_len = code[i] >> 16;
		if (word_len == 0 || opcode != SPIRV_OP_CAPABILITY) break;
		if (word_len >= 2) {
			if (code[i + 1] == SPIRV_CAPABILITY_VIEWPORT_LAYER) has_vl_cap   = true;
			if (code[i + 1] == SPIRV_CAPABILITY_GEOMETRY)        has_geom_cap = true;
		}
		i += word_len;
	}

	// Find layer variable ID from OpDecorate BuiltIn Layer
	uint32_t layer_id = 0;
	for (uint32_t i = 5; i < word_count; ) {
		uint32_t opcode   = code[i] & 0xFFFF;
		uint32_t word_len = code[i] >> 16;
		if (word_len == 0) break;
		if (opcode == SPIRV_OP_DECORATE && word_len >= 4 &&
		    code[i + 2] == SPIRV_DECORATION_BUILTIN &&
		    code[i + 3] == SPIRV_BUILTIN_LAYER) {
			layer_id = code[i + 1];
			break;
		}
		i += word_len;
	}
	if (layer_id == 0) return word_count;

	// Find layer variable's storage class, collect interface variable IDs
	// (same storage class), and check for PrimitiveId / existing Flat.
	uint32_t layer_storage    = 0;
	bool     has_prim_id      = false;
	bool     has_flat          = false;
	uint32_t iface_vars[32];
	uint32_t iface_count      = 0;
	for (uint32_t i = 5; i < word_count; ) {
		uint32_t opcode   = code[i] & 0xFFFF;
		uint32_t word_len = code[i] >> 16;
		if (word_len == 0) break;
		if (opcode == SPIRV_OP_VARIABLE && word_len >= 4) {
			if (code[i + 2] == layer_id)
				layer_storage = code[i + 3];
		}
		if (opcode == SPIRV_OP_DECORATE && word_len >= 3 &&
		    code[i + 1] == layer_id &&
		    code[i + 2] == SPIRV_DECORATION_FLAT)
			has_flat = true;
		if (opcode == SPIRV_OP_DECORATE && word_len >= 4 &&
		    code[i + 2] == SPIRV_DECORATION_BUILTIN &&
		    code[i + 3] == SPIRV_BUILTIN_PRIMITIVE_ID)
			has_prim_id = true;
		i += word_len;
	}

	// VS: has viewport layer cap + Output layer variable
	// PS: Input layer variable (reading it back)
	bool is_output = has_vl_cap && layer_storage == SPIRV_STORAGE_OUTPUT;
	bool is_input  = layer_storage == SPIRV_STORAGE_INPUT;
	if (!is_output && !is_input) return word_count;

	// Collect interface variable IDs (same storage class as layer).
	for (uint32_t i = 5; i < word_count; ) {
		uint32_t opcode   = code[i] & 0xFFFF;
		uint32_t word_len = code[i] >> 16;
		if (word_len == 0) break;
		if (opcode == SPIRV_OP_VARIABLE && word_len >= 4 &&
		    code[i + 3] == layer_storage &&
		    code[i + 2] != layer_id &&
		    iface_count < 32)
			iface_vars[iface_count++] = code[i + 2];
		i += word_len;
	}

	// Find max Location among interface variables so the patched layer
	// gets a non-conflicting slot. Both VS and PS will compute the same
	// value because HLSL compilers assign matching locations to VS
	// outputs and PS inputs.
	uint32_t max_loc   = 0;
	bool     found_loc = false;
	for (uint32_t i = 5; i < word_count; ) {
		uint32_t opcode   = code[i] & 0xFFFF;
		uint32_t word_len = code[i] >> 16;
		if (word_len == 0) break;
		if (opcode == SPIRV_OP_DECORATE && word_len >= 4 &&
		    code[i + 2] == SPIRV_DECORATION_LOCATION) {
			uint32_t var_id = code[i + 1];
			for (uint32_t j = 0; j < iface_count; j++) {
				if (iface_vars[j] == var_id) {
					if (!found_loc || code[i + 3] > max_loc)
						max_loc = code[i + 3];
					found_loc = true;
					break;
				}
			}
		}
		i += word_len;
	}
	uint32_t new_loc = found_loc ? max_loc + 1 : 0;

	// --- Phase 2: Patch ---
	// Convert BuiltIn Layer → Location + Flat. For VS, also strip the
	// extension and capability. For PS, strip Geometry capability if it
	// was only there for Layer reading.

	bool strip_geom = is_input && has_geom_cap && !has_prim_id;
	bool need_flat  = !has_flat; // emit Flat if the shader doesn't have it

	uint32_t src = 5;
	uint32_t dst = 5;
	while (src < word_count) {
		uint32_t opcode       = code[src] & 0xFFFF;
		uint32_t word_len     = code[src] >> 16;
		uint32_t word_len_src = word_len;
		if (word_len == 0) break;

		bool strip = false;

		// VS: strip ShaderViewportIndexLayerEXT capability
		if (is_output && opcode == SPIRV_OP_CAPABILITY && word_len >= 2 &&
		    code[src + 1] == SPIRV_CAPABILITY_VIEWPORT_LAYER) {
			strip = true;
		}
		// VS: strip the extension string
		else if (is_output && opcode == SPIRV_OP_EXTENSION && word_len >= 2) {
			const char* name = (const char*)&code[src + 1];
			if (strncmp(name, "SPV_EXT_shader_viewport_index_layer", (word_len - 1) * 4) == 0)
				strip = true;
		}
		// PS: strip Geometry capability if only used for Layer reading
		else if (strip_geom && opcode == SPIRV_OP_CAPABILITY && word_len >= 2 &&
		         code[src + 1] == SPIRV_CAPABILITY_GEOMETRY) {
			strip = true;
		}
		// Convert BuiltIn Layer → Location + Flat
		else if (opcode == SPIRV_OP_DECORATE && word_len >= 4 &&
		         code[src + 1] == layer_id &&
		         code[src + 2] == SPIRV_DECORATION_BUILTIN &&
		         code[src + 3] == SPIRV_BUILTIN_LAYER) {
			// Replace BuiltIn decoration with Location (same 4-word size)
			code[dst + 0] = (4 << 16) | SPIRV_OP_DECORATE;
			code[dst + 1] = layer_id;
			code[dst + 2] = SPIRV_DECORATION_LOCATION;
			code[dst + 3] = new_loc;
			dst += 4;
			// Emit Flat decoration if not already present. The buffer
			// was allocated with 3 extra words of headroom for this.
			if (need_flat) {
				code[dst + 0] = (3 << 16) | SPIRV_OP_DECORATE;
				code[dst + 1] = layer_id;
				code[dst + 2] = SPIRV_DECORATION_FLAT;
				dst += 3;
			}
			src += word_len_src;
			continue;
		}

		if (strip) {
			src += word_len_src;
			continue;
		}

		if (dst != src)
			memmove(&code[dst], &code[src], word_len * sizeof(uint32_t));
		dst += word_len;
		src += word_len_src;
	}

	return dst;
}

///////////////////////////////////////////////////////////////////////////////
// Shader stage creation
///////////////////////////////////////////////////////////////////////////////

skr_shader_stage_t _skr_shader_stage_create(VkDevice device, const void* shader_data, uint32_t shader_size, skr_stage_ type) {
	skr_shader_stage_t stage = {0};
	stage.type               = type;

	const void* final_data = shader_data;
	uint32_t*   patched    = NULL;

	// Convert the Layer built-in to a regular interpolant when the
	// viewport layer extension is not available. The variable stays
	// intact — only the decoration changes from BuiltIn to Location.
	uint32_t final_size = shader_size;
	if (!_skr_vk.has_viewport_layer && (type == skr_stage_vertex || type == skr_stage_pixel)) {
		uint32_t word_count = shader_size / sizeof(uint32_t);
		// Extra 3 words of headroom for a potential Flat decoration insert
		patched = (uint32_t*)_skr_malloc(shader_size + 3 * sizeof(uint32_t));
		memcpy(patched, shader_data, shader_size);
		uint32_t new_word_count = _skr_spirv_strip_viewport_layer(patched, word_count);
		final_size = new_word_count * sizeof(uint32_t);
		final_data = patched;
	}

	VkShaderModuleCreateInfo create_info = {
		.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = final_size,
		.pCode    = (const uint32_t*)final_data,
	};

	VkResult vr = vkCreateShaderModule(device, &create_info, NULL, &stage.shader);
	_skr_free(patched);
	SKR_VK_CHECK_RET(vr, "vkCreateShaderModule", stage);

	return stage;
}

void _skr_shader_stage_destroy(skr_shader_stage_t* ref_stage) {
	if (!ref_stage) return;

	_skr_cmd_destroy_shader_module(NULL, ref_stage->shader);
	*ref_stage = (skr_shader_stage_t){0};
}

skr_shader_stage_t _skr_shader_file_create_stage(VkDevice device, const sksc_shader_file_t* file, skr_stage_ stage) {
	for (uint32_t i = 0; i < file->stage_count; i++) {
		if (file->stages[i].language == skr_shader_lang_spirv && file->stages[i].stage == stage)
			return _skr_shader_stage_create(device, file->stages[i].code, file->stages[i].code_size, stage);
	}
	skr_shader_stage_t empty = {0};
	return empty;
}

///////////////////////////////////////////////////////////////////////////////
// Shader creation
///////////////////////////////////////////////////////////////////////////////

skr_shader_t _skr_shader_create_manual(sksc_shader_meta_t* meta, skr_shader_stage_t v_shader,
                                       skr_shader_stage_t p_shader, skr_shader_stage_t c_shader) {
	skr_shader_t shader  = {0};
	shader.meta          = meta;
	shader.vertex_stage  = v_shader;
	shader.pixel_stage   = p_shader;
	shader.compute_stage = c_shader;

	if (meta) {
		sksc_shader_meta_reference(meta);
	}

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

	// Create shader stages based on what's in the file
	skr_shader_stage_t v_stage = _skr_shader_file_create_stage(_skr_vk.device, &file, skr_stage_vertex);
	skr_shader_stage_t p_stage = _skr_shader_file_create_stage(_skr_vk.device, &file, skr_stage_pixel);
	skr_shader_stage_t c_stage = _skr_shader_file_create_stage(_skr_vk.device, &file, skr_stage_compute);

	*out_shader = _skr_shader_create_manual(file.meta, v_stage, p_stage, c_stage);

	// Don't destroy meta here, it's now owned by the shader
	// Just clean up the file structure
	for (uint32_t i = 0; i < file.stage_count; i++) {
		_skr_free(file.stages[i].code);
	}
	_skr_free(file.stages);

	return skr_err_success;
}

bool skr_shader_is_valid(const skr_shader_t* shader) {
	if (!shader) return false;
	return
		shader->meta                 == NULL           ||
		shader->vertex_stage.shader  != VK_NULL_HANDLE ||
		shader->pixel_stage.shader   != VK_NULL_HANDLE ||
		shader->compute_stage.shader != VK_NULL_HANDLE;
}

void skr_shader_destroy(skr_shader_t* ref_shader) {
	if (!ref_shader) return;

	_skr_shader_stage_destroy(&ref_shader->vertex_stage);
	_skr_shader_stage_destroy(&ref_shader->pixel_stage);
	_skr_shader_stage_destroy(&ref_shader->compute_stage);

	if (ref_shader->meta) {
		sksc_shader_meta_release(ref_shader->meta);
		ref_shader->meta = NULL;
	}

	*ref_shader = (skr_shader_t){0};
}

skr_bind_t skr_shader_get_bind(const skr_shader_t* shader, const char* bind_name) {
	if (!shader || !shader->meta) {
		skr_bind_t empty = {0};
		return empty;
	}

	return sksc_shader_meta_get_bind(shader->meta, bind_name);
}

bool skr_shader_get_param_info(const skr_shader_t* shader, const char* param_name, skr_shader_param_info_t* opt_out_info) {
	if (!shader || !shader->meta) return false;

	int32_t idx = sksc_shader_meta_get_var_index(shader->meta, param_name);
	if (idx < 0) return false;

	if (opt_out_info) {
		const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(shader->meta, idx);
		*opt_out_info = (skr_shader_param_info_t){
			.type  = var->type,
			.count = var->type_count,
			.size  = var->size,
		};
	}
	return true;
}

bool skr_shader_get_tex_info(const skr_shader_t* shader, const char* tex_name, skr_shader_tex_info_t* opt_out_info) {
	if (!shader || !shader->meta) return false;

	uint64_t hash = skr_hash(tex_name);
	for (uint32_t i = 0; i < shader->meta->resource_count; i++) {
		if (shader->meta->resources[i].name_hash == hash) {
			if (opt_out_info) {
				*opt_out_info = (skr_shader_tex_info_t){
					.default_value = shader->meta->resources[i].value,
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
			case skr_register_constant:      desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         break;
			case skr_register_texture:       desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; break;
			case skr_register_read_buffer:   desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         break; // (StructuredBuffer)
			case skr_register_readwrite:     desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         break; // (RWStructuredBuffer)
			case skr_register_readwrite_tex: desc_type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          break; // (RWTexture)
			default:                         desc_type = VK_DESCRIPTOR_TYPE_MAX_ENUM; break;
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
			case skr_register_constant:      desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         break;
			case skr_register_texture:       desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; break;
			case skr_register_read_buffer:   desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         break; // (StructuredBuffer)
			case skr_register_readwrite:     desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         break; // (RWStructuredBuffer)
			case skr_register_readwrite_tex: desc_type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          break; // (RWTexture)
			default:                         desc_type = VK_DESCRIPTOR_TYPE_MAX_ENUM; break;
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
