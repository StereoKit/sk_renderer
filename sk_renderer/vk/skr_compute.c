// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "../include/sk_renderer.h"
#include "skr_vulkan.h"
#include "_sk_renderer.h"

#include <stdlib.h>
#include <string.h>

skr_err_ skr_compute_create(const skr_shader_t* shader, skr_compute_t* out_compute) {
	if (!out_compute) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_compute = (skr_compute_t){};

	if (!shader || !skr_shader_is_valid(shader) || shader->compute_stage.shader == VK_NULL_HANDLE || !shader->meta) {
		skr_log(skr_log_critical, "Invalid shader or no compute stage");
		return skr_err_invalid_parameter;
	}

	out_compute->shader = shader;

	// Create descriptor set layout
	if (shader->meta->buffer_count > 0 || shader->meta->resource_count > 0) {
		VkDescriptorSetLayoutBinding bindings[32];
		uint32_t binding_count = 0;

		// Add buffer bindings
		for (uint32_t i = 0; i < shader->meta->buffer_count; i++) {
			VkDescriptorType desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			if (shader->meta->buffers[i].bind.register_type == skr_register_readwrite) {
				desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}

			bindings[binding_count++] = (VkDescriptorSetLayoutBinding){
				.binding            = shader->meta->buffers[i].bind.slot,
				.descriptorType     = desc_type,
				.descriptorCount    = 1,
				.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = NULL,
			};
		}

		// Add resource bindings (textures, storage buffers, and storage images)
		for (uint32_t i = 0; i < shader->meta->resource_count; i++) {
			VkDescriptorType desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

			skr_register_ reg_type = shader->meta->resources[i].bind.register_type;
			if (reg_type == skr_register_readwrite_tex) {
				desc_type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			} else if (reg_type == skr_register_texture) {
				desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			} else if (reg_type == skr_register_readwrite || reg_type == skr_register_read_buffer) {
				// Both StructuredBuffer and RWStructuredBuffer map to storage buffers in Vulkan
				desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}

			bindings[binding_count++] = (VkDescriptorSetLayoutBinding){
				.binding            = shader->meta->resources[i].bind.slot,
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
		*out_compute = (skr_compute_t){};
		return skr_err_device_error;
	}

	// Create compute pipeline
	VkComputePipelineCreateInfo pipeline_info = {
		.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage  = (VkPipelineShaderStageCreateInfo){
			.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage  = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shader->compute_stage.shader,
			.pName  = "cs",
		},
		.layout = out_compute->layout,
	};

	vr = vkCreateComputePipelines(_skr_vk.device, _skr_vk.pipeline_cache, 1, &pipeline_info, NULL, &out_compute->pipeline);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkCreateComputePipelines");
		vkDestroyPipelineLayout(_skr_vk.device, out_compute->layout, NULL);
		if (out_compute->descriptor_layout) {
			vkDestroyDescriptorSetLayout(_skr_vk.device, out_compute->descriptor_layout, NULL);
		}
		*out_compute = (skr_compute_t){};
		return skr_err_device_error;
	}

	// Allocate memory for our resource binds
	out_compute->bind_count = shader->meta->resource_count + shader->meta->buffer_count;
	out_compute->binds      = (skr_material_bind_t*)_skr_calloc(out_compute->bind_count, sizeof(skr_material_bind_t));
	for (uint32_t i = 0; i < shader->meta->buffer_count;   i++) out_compute->binds[i                           ].bind = shader->meta->buffers  [i].bind;
	for (uint32_t i = 0; i < shader->meta->resource_count; i++) out_compute->binds[i+shader->meta->buffer_count].bind = shader->meta->resources[i].bind;

	// Initialize parameter buffer if shader has $Global cbuffer
	const sksc_shader_meta_t* meta = shader->meta;
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

bool skr_compute_is_valid(const skr_compute_t* compute) {
	return compute && compute->pipeline != VK_NULL_HANDLE;
}

skr_bind_t skr_compute_get_bind(const skr_compute_t* compute, const char* bind_name) {
	if (!compute || !compute->shader || !compute->shader->meta) {
		skr_bind_t empty = {0};
		return empty;
	}
	return sksc_shader_meta_get_bind(compute->shader->meta, bind_name);
}

void skr_compute_destroy(skr_compute_t* ref_compute) {
	if (!ref_compute) return;

	_skr_cmd_destroy_pipeline             (NULL, ref_compute->pipeline);
	_skr_cmd_destroy_pipeline_layout      (NULL, ref_compute->layout);
	_skr_cmd_destroy_descriptor_set_layout(NULL, ref_compute->descriptor_layout);

	_skr_free(ref_compute->binds);
	_skr_free(ref_compute->param_buffer);

	*ref_compute = (skr_compute_t){};
}

void skr_compute_set_buffer(skr_compute_t* ref_compute, const char* name, skr_buffer_t* buffer) {
	const sksc_shader_meta_t *meta = ref_compute->shader->meta;

	int32_t  idx  = -1;
	uint64_t hash = skr_hash(name);
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		if (meta->buffers[i].name_hash == hash) {
			idx = i;
			break; 
		}
	}

	if (idx >= 0) {
		ref_compute->binds[idx].buffer = buffer;
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
		ref_compute->binds[meta->buffer_count + idx].buffer = buffer;
	} else {
		skr_log(skr_log_warning, "Buffer name '%s' not found", name);
	}
	return;
}

void skr_compute_set_tex(skr_compute_t* ref_compute, const char* name, skr_tex_t* texture) {
	const sksc_shader_meta_t *meta = ref_compute->shader->meta;

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

	ref_compute->binds[meta->buffer_count + idx].texture = texture;
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
	if (!ref_compute || !ref_compute->shader || !ref_compute->shader->meta || !ref_compute->param_buffer) return;

	int32_t var_index = sksc_shader_meta_get_var_index(ref_compute->shader->meta, name);
	if (var_index < 0) {
		skr_log(skr_log_warning, "Compute parameter '%s' not found", name);
		return;
	}

	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(ref_compute->shader->meta, var_index);
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
	if (!compute || !compute->shader || !compute->shader->meta || !compute->param_buffer) return;

	int32_t var_index = sksc_shader_meta_get_var_index(compute->shader->meta, name);
	if (var_index < 0) {
		skr_log(skr_log_warning, "Compute parameter '%s' not found", name);
		return;
	}

	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(compute->shader->meta, var_index);
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

void skr_compute_execute(skr_compute_t* ref_compute, uint32_t x, uint32_t y, uint32_t z) {
	if (!skr_compute_is_valid(ref_compute)) return;

	const sksc_shader_meta_t* meta = ref_compute->shader->meta;

	// Acquire command buffer first so we have a valid future
	_skr_cmd_ctx_t  ctx = _skr_cmd_acquire();
	VkCommandBuffer cmd = ctx.cmd;
	if (!cmd) {
		skr_log(skr_log_warning, "skr_compute_execute failed to acquire command buffer");
		return;
	}

	// Upload parameter buffer to bump allocator if it exists
	if (ref_compute->param_buffer && meta->global_buffer_id >= 0) {
		// Write to command buffer's bump allocator
		skr_bump_result_t result = _skr_bump_alloc_write(ctx.const_bump, ref_compute->param_buffer, ref_compute->param_buffer_size);

		if (!result.buffer) {
			skr_log(skr_log_warning, "skr_compute_execute: bump allocator failed");
			_skr_cmd_release(cmd);
			return;
		}

		// Set up bind with offset
		ref_compute->binds[meta->global_buffer_id].buffer        = result.buffer;
		ref_compute->binds[meta->global_buffer_id].buffer_offset = result.offset;
		ref_compute->binds[meta->global_buffer_id].buffer_range  = ref_compute->param_buffer_size;
		ref_compute->param_dirty = false;
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ref_compute->pipeline);

	// Transition all bound textures to appropriate layouts before dispatch
	for (uint32_t i = 0; i < ref_compute->bind_count; i++) {
		skr_material_bind_t *res = &ref_compute->binds[i];
		if      (res->bind.register_type == skr_register_readwrite_tex && res->texture) {_skr_tex_transition_for_storage    (cmd, res->texture); }
		else if (res->bind.register_type == skr_register_texture       && res->texture) {_skr_tex_transition_for_shader_read(cmd, res->texture, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT); }
	}

	VkWriteDescriptorSet   writes      [32];
	VkDescriptorBufferInfo buffer_infos[16];
	VkDescriptorImageInfo  image_infos [16];
	uint32_t write_ct  = 0;
	uint32_t buffer_ct = 0;
	uint32_t image_ct  = 0;
	int32_t fail_idx = _skr_material_add_writes(ref_compute->binds, ref_compute->bind_count, NULL, 0,
		writes,       sizeof(writes      )/sizeof(writes      [0]),
		buffer_infos, sizeof(buffer_infos)/sizeof(buffer_infos[0]),
		image_infos,  sizeof(image_infos )/sizeof(image_infos [0]),
		&write_ct, &buffer_ct, &image_ct);
	if (fail_idx >= 0) {
		skr_log(skr_log_critical, "Compute dispatch missing binding '%s' in shader '%s'", _skr_material_bind_name(meta, fail_idx), meta->name);
		_skr_cmd_release(cmd);
		return;
	}

	//_skr_log_descriptor_writes(writes, buffer_infos, image_infos, write_ct, buffer_ct, image_ct);

	_skr_bind_descriptors(cmd, ctx.descriptor_pool, VK_PIPELINE_BIND_POINT_COMPUTE,
	                      ref_compute->layout, ref_compute->descriptor_layout, writes, write_ct);

	vkCmdDispatch(cmd, x, y, z);

	// Add memory barrier for storage resources to ensure writes are visible to next operation
	// This includes compute→compute, compute→vertex, and compute→fragment transitions
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0, 1, &(VkMemoryBarrier){
			.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		}, 0, NULL, 0, NULL);

	_skr_cmd_release(cmd);
}

void skr_compute_execute_indirect(skr_compute_t* ref_compute, skr_buffer_t* indirect_args) {
	if (!skr_compute_is_valid(ref_compute) || !indirect_args) return;

	const sksc_shader_meta_t* meta = ref_compute->shader->meta;

	// Acquire command buffer first so we have a valid future
	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();
	VkCommandBuffer cmd = ctx.cmd;
	if (!cmd) {
		skr_log(skr_log_warning, "skr_compute_execute_indirect failed to acquire command buffer");
		return;
	}

	// Upload parameter buffer to bump allocator if it exists
	if (ref_compute->param_buffer && meta->global_buffer_id >= 0) {
		// Write to command buffer's bump allocator
		skr_bump_result_t result = _skr_bump_alloc_write(ctx.const_bump, ref_compute->param_buffer, ref_compute->param_buffer_size);

		if (!result.buffer) {
			skr_log(skr_log_warning, "skr_compute_execute_indirect: bump allocator failed");
			_skr_cmd_release(cmd);
			return;
		}

		// Set up bind with offset
		ref_compute->binds[meta->global_buffer_id].buffer        = result.buffer;
		ref_compute->binds[meta->global_buffer_id].buffer_offset = result.offset;
		ref_compute->binds[meta->global_buffer_id].buffer_range  = ref_compute->param_buffer_size;
		ref_compute->param_dirty = false;
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ref_compute->pipeline);

	VkWriteDescriptorSet   writes      [32];
	VkDescriptorBufferInfo buffer_infos[16];
	VkDescriptorImageInfo  image_infos [16];
	uint32_t write_ct = 0;
	uint32_t buffer_ct = 0;
	uint32_t image_ct = 0;
	int32_t fail_idx = _skr_material_add_writes(ref_compute->binds, ref_compute->bind_count, NULL, 0,
		writes,       sizeof(writes      )/sizeof(writes      [0]),
		buffer_infos, sizeof(buffer_infos)/sizeof(buffer_infos[0]),
		image_infos,  sizeof(image_infos )/sizeof(image_infos [0]),
		&write_ct, &buffer_ct, &image_ct);
	if (fail_idx >= 0) {
		skr_log(skr_log_critical, "Compute indirect dispatch missing binding '%s' in shader '%s'", _skr_material_bind_name(meta, fail_idx), meta->name);
		_skr_cmd_release(cmd);
		return;
	}

	_skr_bind_descriptors(cmd, ctx.descriptor_pool, VK_PIPELINE_BIND_POINT_COMPUTE,
	                      ref_compute->layout, ref_compute->descriptor_layout, writes, write_ct);

	vkCmdDispatchIndirect(cmd, indirect_args->buffer, 0);
	_skr_cmd_release(cmd);
}
