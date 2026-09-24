// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "../include/sk_renderer.h"
#include "skr_vulkan.h"
#include "_sk_renderer.h"

#include <stdlib.h>
#include <string.h>

// Builds a pipeline into *out_pipeline from the compute's shader stage, applying
// wave_size and info's resolved spec constants. Reads compute->shader/->layout
// only; never touches compute->pipeline, so callers can build into a temp and
// swap on success. On failure returns skr_err_device_error, *out_pipeline stays
// VK_NULL_HANDLE. Assumes compute->layout is already valid.
static skr_err_ _skr_compute_build_pipeline(const skr_compute_t* compute, skr_compute_info_t info, VkPipeline* out_pipeline) {
	const skr_shader_t* shader = compute->shader;

	// Optionally pin the compute subgroup/wave size. Driven by `//--wave_size = N`
	// in the shader; requires VK_EXT_subgroup_size_control. Mismatched values
	// fall back to the implementation default and emit a warning.
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT required_subgroup_size = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
	};
	void* stage_pNext = NULL;
	if (shader->meta.wave_size != 0) {
		if (!_skr_vk.has_subgroup_size_control) {
			skr_log(skr_log_warning, "Shader '%s' requests wave_size=%u but VK_EXT_subgroup_size_control is unavailable; using implementation default", shader->meta.name, shader->meta.wave_size);
		} else if (shader->meta.wave_size < _skr_vk.min_subgroup_size || shader->meta.wave_size > _skr_vk.max_subgroup_size) {
			skr_log(skr_log_warning, "Shader '%s' requests wave_size=%u but device subgroup range is [%u, %u]; using implementation default", shader->meta.name, shader->meta.wave_size, _skr_vk.min_subgroup_size, _skr_vk.max_subgroup_size);
		} else if ((_skr_vk.required_subgroup_size_stages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
			skr_log(skr_log_warning, "Shader '%s' requests wave_size=%u but the device does not allow required subgroup size on compute stages; using implementation default", shader->meta.name, shader->meta.wave_size);
		} else {
			required_subgroup_size.requiredSubgroupSize = shader->meta.wave_size;
			stage_pNext = &required_subgroup_size;
		}
	}

	uint32_t spec_values[SKR_MAX_SPEC_CONSTANTS];
	_skr_shader_resolve_spec_constants(&shader->meta, info.spec_constants, info.spec_constant_count, spec_values);

	VkSpecializationMapEntry    spec_entries[SKR_MAX_SPEC_CONSTANTS];
	VkSpecializationInfo        spec_info;
	const VkSpecializationInfo* spec = _skr_shader_make_spec_info(&shader->meta, spec_values, spec_entries, &spec_info);

	VkComputePipelineCreateInfo pipeline_info = {
		.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage  = (VkPipelineShaderStageCreateInfo){
			.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext               = stage_pNext,
			.stage               = VK_SHADER_STAGE_COMPUTE_BIT,
			.module              = shader->compute_stage.shader,
			.pName               = "cs",
			.pSpecializationInfo = spec,
		},
		.layout = compute->layout,
	};

	VkResult vr = vkCreateComputePipelines(_skr_vk.device, _skr_vk.pipeline_cache, 1, &pipeline_info, NULL, out_pipeline);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkCreateComputePipelines");
		*out_pipeline = VK_NULL_HANDLE;
		return skr_err_device_error;
	}
	return skr_err_success;
}

skr_err_ skr_compute_create(const skr_shader_t* shader, skr_compute_info_t info, skr_compute_t* out_compute) {
	if (!out_compute) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_compute = (skr_compute_t){0};

	if (!shader || !skr_shader_is_valid(shader) || shader->compute_stage.shader == VK_NULL_HANDLE) {
		skr_log(skr_log_critical, "Invalid shader or no compute stage");
		return skr_err_invalid_parameter;
	}

	out_compute->shader = shader;

	// Must keep producing no dynamic descriptors: skr_compute_execute binds with
	// zero offsets, and a compute $Global at b0 collides with material_slot, so
	// _skr_shader_make_layout would silently make it dynamic.
	if (shader->meta.buffer_count > 0 || shader->meta.resource_count > 0) {
		VkDescriptorSetLayoutBinding bindings[32];
		uint32_t binding_count = 0;

		// Add buffer bindings
		for (uint32_t i = 0; i < shader->meta.buffer_count; i++) {
			VkDescriptorType desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			if (shader->meta.buffers[i].bind.register_type == skr_register_readwrite) {
				desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}

			bindings[binding_count++] = (VkDescriptorSetLayoutBinding){
				.binding            = shader->meta.buffers[i].bind.slot,
				.descriptorType     = desc_type,
				.descriptorCount    = 1,
				.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = NULL,
			};
		}

		// Add resource bindings (textures, storage buffers, and storage images)
		for (uint32_t i = 0; i < shader->meta.resource_count; i++) {
			VkDescriptorType desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

			skr_register_ reg_type = shader->meta.resources[i].bind.register_type;
			if (reg_type == skr_register_readwrite_tex) {
				desc_type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			} else if (reg_type == skr_register_texture) {
				desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			} else if (reg_type == skr_register_readwrite || reg_type == skr_register_read_buffer) {
				// Both StructuredBuffer and RWStructuredBuffer map to storage buffers in Vulkan
				desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}

			bindings[binding_count++] = (VkDescriptorSetLayoutBinding){
				.binding            = shader->meta.resources[i].bind.slot,
				.descriptorType     = desc_type,
				.descriptorCount    = 1,
				.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = NULL,
			};
		}

		VkDescriptorSetLayoutCreateInfo layout_info = {
			.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.flags        = _skr_vk.has_push_descriptors ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
			.bindingCount = binding_count,
			.pBindings    = bindings,
		};

		VkResult vr = vkCreateDescriptorSetLayout(_skr_vk.device, &layout_info, NULL, &out_compute->descriptor_layout);
		SKR_VK_CHECK_RET(vr, "vkCreateDescriptorSetLayout", skr_err_device_error);
	}

	// Create pipeline layout
	VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = out_compute->descriptor_layout != VK_NULL_HANDLE ? 1 : 0,
		.pSetLayouts    = out_compute->descriptor_layout != VK_NULL_HANDLE ? &out_compute->descriptor_layout : NULL,
	};

	VkResult vr = vkCreatePipelineLayout(_skr_vk.device, &pipeline_layout_info, NULL, &out_compute->layout);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkCreatePipelineLayout");
		if (out_compute->descriptor_layout) {
			vkDestroyDescriptorSetLayout(_skr_vk.device, out_compute->descriptor_layout, NULL);
		}
		*out_compute = (skr_compute_t){0};
		return skr_err_device_error;
	}

	skr_err_ err = _skr_compute_build_pipeline(out_compute, info, &out_compute->pipeline);
	if (err != skr_err_success) {
		vkDestroyPipelineLayout(_skr_vk.device, out_compute->layout, NULL);
		if (out_compute->descriptor_layout) {
			vkDestroyDescriptorSetLayout(_skr_vk.device, out_compute->descriptor_layout, NULL);
		}
		*out_compute = (skr_compute_t){0};
		return err;
	}

	// The bind pool slice doubles as this dispatch's descriptor cache key
	out_compute->bind_count = shader->meta.resource_count + shader->meta.buffer_count;
	out_compute->bind_start = _skr_bind_pool_alloc(out_compute->bind_count);
	skr_material_bind_t* binds = _skr_bind_pool_get(out_compute->bind_start);
	for (uint32_t i = 0; i < shader->meta.buffer_count;   i++) binds[i                          ].bind = shader->meta.buffers  [i].bind;
	for (uint32_t i = 0; i < shader->meta.resource_count; i++) binds[i+shader->meta.buffer_count].bind = shader->meta.resources[i].bind;

	// Initialize parameter buffer if shader has $Global cbuffer
	const sksc_shader_meta_t* meta = &shader->meta;
	if (meta->global_buffer_id >= 0) {
		sksc_shader_buffer_t* global_buffer = &meta->buffers[meta->global_buffer_id];

		out_compute->param_buffer_size = global_buffer->size;
		out_compute->param_buffer      = _skr_malloc(out_compute->param_buffer_size);

		// Initialize with defaults from shader (or zero)
		if (global_buffer->defaults) {
			memcpy(out_compute->param_buffer, global_buffer->defaults, out_compute->param_buffer_size);
		} else {
			memset(out_compute->param_buffer, 0, out_compute->param_buffer_size);
		}

		// Mark as dirty to force initial upload
		out_compute->param_dirty = true;
	}

	return skr_err_success;
}

void skr_compute_set_pipeline(skr_compute_t* ref_compute, skr_compute_info_t info) {
	if (!ref_compute || !ref_compute->shader) return;

	// The compute is locked to its create-time shader; only spec values change.
	// layout, descriptor_layout, binds, and param_buffer are all retained.
	const skr_shader_t* shader = ref_compute->shader;

	// Build into a temp first: a failed rebuild must leave the still-working
	// pipeline in place rather than stranding the object with no pipeline.
	VkPipeline new_pipeline = VK_NULL_HANDLE;
	skr_err_   err          = _skr_compute_build_pipeline(ref_compute, info, &new_pipeline);
	if (err != skr_err_success) {
		skr_log(skr_log_critical, "skr_compute_set_pipeline: failed to rebuild pipeline for '%s'; keeping previous pipeline", shader->meta.name);
		return;
	}

	// Swap in the new pipeline, deferred-destroy the old (safe for in-flight
	// command buffers).
	_skr_cmd_destroy_pipeline(NULL, ref_compute->pipeline);
	ref_compute->pipeline = new_pipeline;
}

bool skr_compute_is_valid(const skr_compute_t* compute) {
	return compute && compute->pipeline != VK_NULL_HANDLE;
}

skr_bind_t skr_compute_get_bind(const skr_compute_t* compute, const char* bind_name) {
	if (!compute || !compute->shader) {
		skr_bind_t empty = {0};
		return empty;
	}
	return sksc_shader_meta_get_bind(&compute->shader->meta, bind_name);
}

void skr_compute_destroy(skr_compute_t* ref_compute) {
	if (!ref_compute) return;

	_skr_cmd_destroy_pipeline             (NULL, ref_compute->pipeline);
	_skr_cmd_destroy_pipeline_layout      (NULL, ref_compute->layout);
	_skr_cmd_destroy_descriptor_set_layout(NULL, ref_compute->descriptor_layout);

	_skr_cmd_destroy_bind_pool_slots(NULL, ref_compute->bind_start, ref_compute->bind_count);
	_skr_free(ref_compute->param_buffer);

	*ref_compute = (skr_compute_t){0};
}

void skr_compute_set_buffer(skr_compute_t* ref_compute, const char* name, skr_buffer_t* buffer) {
	const sksc_shader_meta_t *meta = &ref_compute->shader->meta;

	int32_t  idx  = -1;
	uint64_t hash = skr_hash(name);
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		if (meta->buffers[i].name_hash == hash) {
			idx = i;
			break; 
		}
	}

	skr_material_bind_t* binds = _skr_bind_pool_get(ref_compute->bind_start);
	if (idx >= 0) {
		binds[idx].buffer = buffer;
		return;
	}

	// StructuredBuffers look like buffers, but HLSL treats them like textures/resources
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		if (meta->resources[i].name_hash == hash) {
			idx = i;
			break;
		}
	}

	if (idx >= 0) {
		binds[meta->buffer_count + idx].buffer = buffer;
	} else {
		skr_log(skr_log_warning, "Buffer name '%s' not found", name);
	}
	return;
}

void skr_compute_set_tex(skr_compute_t* ref_compute, const char* name, skr_tex_t* texture) {
	const sksc_shader_meta_t *meta = &ref_compute->shader->meta;

	int32_t  idx  = -1;
	uint64_t hash = skr_hash(name);
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		if (meta->resources[i].name_hash == hash) {
			idx = i;
			break;
		}
	}

	if (idx == -1) {
		skr_log(skr_log_warning, "Texture name '%s' not found", name);
		return;
	}

	_skr_bind_pool_get(ref_compute->bind_start)[meta->buffer_count + idx].texture = texture;
}

///////////////////////////////////////////////////////////////////////////////
// Compute parameter setters/getters
///////////////////////////////////////////////////////////////////////////////

static uint32_t _skr_shader_var_size(sksc_shader_var_ type) {
	switch (type) {
		case sksc_shader_var_int:    return sizeof(int32_t);
		case sksc_shader_var_uint:   return sizeof(uint32_t);
		case sksc_shader_var_uint8:  return sizeof(uint8_t);
		case sksc_shader_var_float:  return sizeof(float);
		case sksc_shader_var_double: return sizeof(double);
		default:                     return 0;
	}
}

void skr_compute_set_params(skr_compute_t* ref_compute, const void* data, uint32_t size) {
	if (!ref_compute || !ref_compute->param_buffer) {
		skr_log(skr_log_warning, "compute_set_params: compute has no $Global buffer");
		return;
	}
	if (size != ref_compute->param_buffer_size) {
		skr_log(skr_log_warning, "compute_set_params: incorrect size! Expected %u, got %u", ref_compute->param_buffer_size, size);
		return;
	}
	memcpy(ref_compute->param_buffer, data, size);
	ref_compute->param_dirty = true;
}

void skr_compute_set_param(skr_compute_t* ref_compute, const char* name, sksc_shader_var_ type, uint32_t count, const void* data) {
	if (!ref_compute || !ref_compute->shader || !ref_compute->param_buffer) return;

	int32_t var_index = sksc_shader_meta_get_var_index(&ref_compute->shader->meta, name);
	if (var_index < 0) {
		skr_log(skr_log_warning, "Compute parameter '%s' not found", name);
		return;
	}

	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(&ref_compute->shader->meta, var_index);
	if (!var) return;

	// When type is uint8, treat count as raw byte count and skip type check
	uint32_t copy_size;
	if (type == sksc_shader_var_uint8) {
		copy_size = count;
	} else {
		if (var->type != type) {
			skr_log(skr_log_warning, "Compute parameter '%s' type mismatch", name);
			return;
		}
		copy_size = _skr_shader_var_size(type) * count;
	}

	if (var->offset + copy_size > ref_compute->param_buffer_size) {
		skr_log(skr_log_warning, "Compute parameter '%s' write would exceed buffer size", name);
		return;
	}

	memcpy((uint8_t*)ref_compute->param_buffer + var->offset, data, copy_size);
	ref_compute->param_dirty = true;
}

void skr_compute_get_param(const skr_compute_t* compute, const char* name, sksc_shader_var_ type, uint32_t count, void* out_data) {
	if (!compute || !compute->shader || !compute->param_buffer) return;

	int32_t var_index = sksc_shader_meta_get_var_index(&compute->shader->meta, name);
	if (var_index < 0) {
		skr_log(skr_log_warning, "Compute parameter '%s' not found", name);
		return;
	}

	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(&compute->shader->meta, var_index);
	if (!var) return;

	// When type is uint8, treat count as raw byte count and skip type check
	uint32_t copy_size;
	if (type == sksc_shader_var_uint8) {
		copy_size = count;
	} else {
		if (var->type != type) {
			skr_log(skr_log_warning, "Compute parameter '%s' type mismatch", name);
			return;
		}
		copy_size = _skr_shader_var_size(type) * count;
	}

	if (var->offset + copy_size > compute->param_buffer_size) {
		skr_log(skr_log_warning, "Compute parameter '%s' read would exceed buffer size", name);
		return;
	}

	memcpy(out_data, (uint8_t*)compute->param_buffer + var->offset, copy_size);
}

// Resolves a dispatch's $Global block to the declared size, padding short
// params with the shader's defaults. Returns NULL when there's no $Global.
static const void* _skr_compute_params(const sksc_shader_meta_t* meta, const void* opt_params, uint32_t params_size, uint8_t* ref_pad, uint32_t pad_size, void** out_heap, uint32_t* out_size) {
	*out_heap = NULL;
	*out_size = 0;
	if (meta->global_buffer_id < 0) return NULL;

	const sksc_shader_buffer_t* global = &meta->buffers[meta->global_buffer_id];
	*out_size = global->size;
	if (opt_params && params_size >= global->size) return opt_params;

	uint8_t* dst = ref_pad;
	if (global->size > pad_size) dst = (uint8_t*)(*out_heap = _skr_malloc(global->size));
	if (global->defaults) memcpy(dst, global->defaults, global->size);
	else                  memset(dst, 0, global->size);
	if (opt_params) memcpy(dst, opt_params, params_size);
	return dst;
}

// Shared by execute, execute_indirect, and dispatch. Everything per-dispatch
// arrives as arguments, so this never writes to the compute.
static bool _skr_compute_record(const skr_compute_t* compute, const skr_material_bind_t* binds, uint32_t bind_count, const void* opt_params, uint32_t params_size, uint32_t x, uint32_t y, uint32_t z, const skr_buffer_t* opt_indirect) {
	const sksc_shader_meta_t* meta = &compute->shader->meta;

	_skr_cmd_ctx_t  ctx = _skr_cmd_acquire();
	VkCommandBuffer cmd = ctx.cmd;
	if (!cmd) {
		skr_log(skr_log_warning, "Compute dispatch failed to acquire command buffer");
		return false;
	}

	// $Global params go straight into this dispatch's writes rather than
	// through the bind slot, so the slot is skipped below
	_skr_desc_writes_t desc;
	_skr_desc_writes_begin(&desc, -1); // compute layouts have no dynamic bindings
	int32_t global_slot = -1;
	if (opt_params && meta->global_buffer_id >= 0) {
		skr_bump_result_t result = _skr_bump_ring_write(ctx.const_ring, opt_params, params_size);
		if (!result.buffer) {
			skr_log(skr_log_warning, "Compute dispatch: bump allocator failed");
			_skr_cmd_release(cmd);
			return false;
		}
		global_slot = meta->buffers[meta->global_buffer_id].bind.slot;
		_skr_write_buffer(&desc, (uint32_t)global_slot, false, result.buffer, result.offset, params_size);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute->pipeline);

	for (uint32_t i = 0; i < bind_count; i++) {
		const skr_material_bind_t *res = &binds[i];
		if      (res->bind.register_type == skr_register_readwrite_tex && res->texture) {_skr_tex_transition_for_storage    (cmd, res->texture); }
		else if (res->bind.register_type == skr_register_texture       && res->texture) {_skr_tex_transition_for_shader_read(cmd, res->texture, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT); }
	}

	if (opt_indirect) {
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
			0, 1, &(VkMemoryBarrier){
				.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
			}, 0, NULL, 0, NULL);
	}

	int32_t fail_idx = _skr_material_add_writes(binds, bind_count, skr_stage_compute, &global_slot, global_slot >= 0 ? 1 : 0, &desc);
	if (fail_idx >= 0) {
		skr_log(skr_log_critical, "Compute dispatch missing binding '%s' in shader '%s'", _skr_material_bind_name(meta, fail_idx), meta->name);
		_skr_cmd_release(cmd);
		return false;
	}

	if (_skr_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute->bind_start, compute->layout, compute->descriptor_layout, &desc)) {
		if (opt_indirect) vkCmdDispatchIndirect(cmd, opt_indirect->buffer, 0);
		else              vkCmdDispatch        (cmd, x, y, z);
	}

	// Compute→compute memory barrier (immediate). Compute→graphics barrier is
	// deferred via pending_compute_barrier and flushed before the next render pass.
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 1, &(VkMemoryBarrier){
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		}, 0, NULL, 0, NULL);
	_skr_store_u32(&_skr_vk.pending_compute_barrier, 1);

	_skr_cmd_release(cmd);
	return true;
}

void skr_compute_execute(skr_compute_t* ref_compute, uint32_t x, uint32_t y, uint32_t z) {
	if (!skr_compute_is_valid(ref_compute)) return;
	if (_skr_compute_record(ref_compute, _skr_bind_pool_get(ref_compute->bind_start), ref_compute->bind_count, ref_compute->param_buffer, ref_compute->param_buffer_size, x, y, z, NULL))
		ref_compute->param_dirty = false;
}

void skr_compute_execute_indirect(skr_compute_t* ref_compute, skr_buffer_t* indirect_args) {
	if (!skr_compute_is_valid(ref_compute) || !indirect_args) return;
	if (_skr_compute_record(ref_compute, _skr_bind_pool_get(ref_compute->bind_start), ref_compute->bind_count, ref_compute->param_buffer, ref_compute->param_buffer_size, 0, 0, 0, indirect_args))
		ref_compute->param_dirty = false;
}

void skr_compute_dispatch(const skr_compute_t* compute, const skr_compute_bind_t* binds, uint32_t bind_count, const void* opt_params, uint32_t params_size, uint32_t x, uint32_t y, uint32_t z) {
	if (!skr_compute_is_valid(compute)) return;
	const sksc_shader_meta_t* meta = &compute->shader->meta;

	// Laid out like the compute's own bind slice, so a slot the caller skips
	// fails the same missing-binding check execute does.
	skr_material_bind_t slots[32] = {0};
	uint32_t            slot_count = compute->bind_count;
	if (slot_count > sizeof(slots) / sizeof(slots[0])) {
		skr_log(skr_log_critical, "skr_compute_dispatch: shader '%s' has %u binds, above the %u it supports", meta->name, slot_count, (uint32_t)(sizeof(slots) / sizeof(slots[0])));
		return;
	}
	const skr_material_bind_t* own = _skr_bind_pool_get(compute->bind_start);
	for (uint32_t i = 0; i < slot_count; i++) slots[i].bind = own[i].bind;
	for (uint32_t b = 0; b < bind_count; b++) {
		for (uint32_t i = 0; i < slot_count; i++) {
			if (slots[i].bind.slot != binds[b].bind.slot || slots[i].bind.register_type != binds[b].bind.register_type) continue;
			// StructuredBuffers sit among the resources, so the register decides
			uint8_t reg = binds[b].bind.register_type;
			if (reg == skr_register_constant || reg == skr_register_read_buffer || reg == skr_register_readwrite) slots[i].buffer  = binds[b].buffer;
			else                                                                                                  slots[i].texture = binds[b].tex;
			break;
		}
	}

	uint8_t     pad[256];
	void*       heap = NULL;
	uint32_t    size = 0;
	const void* params = _skr_compute_params(meta, opt_params, params_size, pad, sizeof(pad), &heap, &size);
	_skr_compute_record(compute, slots, slot_count, params, size, x, y, z, NULL);
	_skr_free(heap);
}
