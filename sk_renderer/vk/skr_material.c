// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "sk_renderer.h"
#include "_sk_renderer.h"

#include "skr_vulkan.h"
#include "skr_pipeline.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

///////////////////////////////////////////////////////////////////////////////
// Bind pool implementation
///////////////////////////////////////////////////////////////////////////////

#define SKR_BIND_POOL_INITIAL_CAPACITY 256
#define SKR_BIND_POOL_FREE_INITIAL     16

void _skr_bind_pool_init(void) {
	_skr_bind_pool_t* pool = &_skr_vk.bind_pool;

	mtx_init(&pool->mutex, mtx_plain);

	pool->capacity            = SKR_BIND_POOL_INITIAL_CAPACITY;
	pool->binds               = (_skr_calloc(pool->capacity, sizeof(skr_material_bind_t)));
	pool->free_range_capacity = SKR_BIND_POOL_FREE_INITIAL;
	pool->free_range_count    = 1;
	pool->free_ranges         = (_skr_malloc(pool->free_range_capacity * sizeof(_skr_bind_range_t)));

	// Initially all slots are free as one contiguous range
	pool->free_ranges[0] = (_skr_bind_range_t){ .start = 0, .count = pool->capacity };
}

void _skr_bind_pool_shutdown(void) {
	_skr_bind_pool_t* pool = &_skr_vk.bind_pool;

	mtx_destroy(&pool->mutex);

	_skr_free(pool->binds);
	_skr_free(pool->free_ranges);
	*pool = (_skr_bind_pool_t){0};
}

int32_t _skr_bind_pool_alloc(uint32_t count) {
	if (count == 0) return -1;

	_skr_bind_pool_t* pool = &_skr_vk.bind_pool;
	int32_t result = -1;

	mtx_lock(&pool->mutex);

	// First-fit search through free ranges
	for (uint32_t i = 0; i < pool->free_range_count; i++) {
		if (pool->free_ranges[i].count >= count) {
			uint32_t start = pool->free_ranges[i].start;

			// Shrink or remove this range
			if (pool->free_ranges[i].count == count) {
				// Remove range by swapping with last
				pool->free_ranges[i] = pool->free_ranges[--pool->free_range_count];
			} else {
				// Shrink range
				pool->free_ranges[i].start += count;
				pool->free_ranges[i].count -= count;
			}

			// Zero out the allocated slots
			memset(&pool->binds[start], 0, count * sizeof(skr_material_bind_t));
			result = (int32_t)start;
			goto done;
		}
	}

	// No suitable range found, grow the pool
	uint32_t old_capacity = pool->capacity;
	uint32_t new_capacity = pool->capacity * 2;
	while (new_capacity - old_capacity < count) {
		new_capacity *= 2;
	}

	skr_material_bind_t* new_binds = _skr_realloc(pool->binds, new_capacity * sizeof(skr_material_bind_t));
	if (!new_binds) {
		skr_log(skr_log_critical, "Failed to grow bind pool");
		goto done;
	}
	pool->binds    = new_binds;
	pool->capacity = new_capacity;

	// Zero new memory
	memset(&pool->binds[old_capacity], 0, (new_capacity - old_capacity) * sizeof(skr_material_bind_t));

	// Add the new free space (minus what we're about to allocate)
	uint32_t remaining = (new_capacity - old_capacity) - count;
	if (remaining > 0) {
		// Grow free range array if needed
		if (pool->free_range_count >= pool->free_range_capacity) {
			uint32_t new_range_capacity = pool->free_range_capacity * 2;
			_skr_bind_range_t* new_ranges = _skr_realloc(pool->free_ranges, new_range_capacity * sizeof(_skr_bind_range_t));
			if (!new_ranges) {
				skr_log(skr_log_warning, "Failed to grow bind pool free range array");
				// Not fatal - we just can't track this free space
			} else {
				pool->free_ranges         = new_ranges;
				pool->free_range_capacity = new_range_capacity;
				pool->free_ranges[pool->free_range_count++] = (_skr_bind_range_t){
					.start = old_capacity + count,
					.count = remaining
				};
			}
		} else {
			pool->free_ranges[pool->free_range_count++] = (_skr_bind_range_t){
				.start = old_capacity + count,
				.count = remaining
			};
		}
	}

	result = (int32_t)old_capacity;

done:
	mtx_unlock(&pool->mutex);
	return result;
}

void _skr_bind_pool_free(int32_t start, uint32_t count) {
	if (start < 0 || count == 0) return;

	_skr_bind_pool_t* pool = &_skr_vk.bind_pool;

	mtx_lock(&pool->mutex);

	// Clear the freed slots (helps catch use-after-free)
	memset(&pool->binds[start], 0, count * sizeof(skr_material_bind_t));

	// Try to coalesce with adjacent ranges
	uint32_t freed_start = (uint32_t)start;
	uint32_t freed_end   = freed_start + count;

	for (uint32_t i = 0; i < pool->free_range_count; i++) {
		uint32_t range_end = pool->free_ranges[i].start + pool->free_ranges[i].count;

		// Check if this range is immediately before the freed region
		if (range_end == freed_start) {
			pool->free_ranges[i].count += count;

			// Check if we can also merge with a range after
			for (uint32_t j = 0; j < pool->free_range_count; j++) {
				if (j != i && pool->free_ranges[j].start == freed_end) {
					pool->free_ranges[i].count += pool->free_ranges[j].count;
					pool->free_ranges[j] = pool->free_ranges[--pool->free_range_count];
					break;
				}
			}
			goto done;
		}

		// Check if this range is immediately after the freed region
		if (pool->free_ranges[i].start == freed_end) {
			pool->free_ranges[i].start  = freed_start;
			pool->free_ranges[i].count += count;
			goto done;
		}
	}

	// No adjacent range found, add new range
	if (pool->free_range_count >= pool->free_range_capacity) {
		uint32_t new_capacity = pool->free_range_capacity * 2;
		_skr_bind_range_t* new_ranges = _skr_realloc(pool->free_ranges, new_capacity * sizeof(_skr_bind_range_t));
		if (!new_ranges) {
			skr_log(skr_log_warning, "Failed to grow bind pool free range array during free");
			goto done;  // Can't track this free space, but memory is cleared
		}
		pool->free_ranges         = new_ranges;
		pool->free_range_capacity = new_capacity;
	}
	pool->free_ranges[pool->free_range_count++] = (_skr_bind_range_t){
		.start = freed_start,
		.count = count
	};

done:
	mtx_unlock(&pool->mutex);
}

skr_material_bind_t* _skr_bind_pool_get(int32_t start) {
	// NOTE: Caller should hold the pool lock via _skr_bind_pool_lock() if there's
	// any chance of concurrent _skr_bind_pool_alloc() calls that might grow the pool.
	if (start < 0 || (uint32_t)start >= _skr_vk.bind_pool.capacity) return NULL;
	return &_skr_vk.bind_pool.binds[start];
}

void _skr_bind_pool_lock(void) {
	mtx_lock(&_skr_vk.bind_pool.mutex);
}

void _skr_bind_pool_unlock(void) {
	mtx_unlock(&_skr_vk.bind_pool.mutex);
}

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_material_create(skr_material_info_t info, skr_material_t* out_material) {
	if (!out_material) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_material = (skr_material_t){};

	if (!info.shader || !skr_shader_is_valid(info.shader)) {
		skr_log(skr_log_warning, "Cannot create material with invalid shader");
		return skr_err_invalid_parameter;
	}

	// Store pipeline-affecting state in key, queue_offset separately
	out_material->key = (_skr_pipeline_material_key_t){
		.shader            = info.shader,
		.cull              = info.cull,
		.write_mask        = info.write_mask ? info.write_mask : skr_write_default,
		.depth_test        = info.depth_test,
		.blend_state       = info.blend_state,
		.alpha_to_coverage = info.alpha_to_coverage,
		.depth_clamp       = info.depth_clamp,
		.stencil_front     = info.stencil_front,
		.stencil_back      = info.stencil_back,
	};
	out_material->queue_offset = info.queue_offset;

	if (out_material->key.shader->meta) {
		sksc_shader_meta_reference(out_material->key.shader->meta);
	}

	// Allocate material parameter buffer if shader has $Global buffer
	const sksc_shader_meta_t* meta = out_material->key.shader->meta;
	if (meta && meta->global_buffer_id >= 0) {
		sksc_shader_buffer_t* global_buffer = &meta->buffers[meta->global_buffer_id];
		out_material->param_buffer_size = global_buffer->size;
		out_material->param_buffer = _skr_malloc(out_material->param_buffer_size);

		if (!out_material->param_buffer) {
			skr_log(skr_log_critical, "Failed to allocate material parameter buffer");
			if (out_material->key.shader->meta) {
				sksc_shader_meta_release(out_material->key.shader->meta);
			}
			*out_material = (skr_material_t){};
			return skr_err_out_of_memory;
		}

		// Initialize with default values if available
		if (global_buffer->defaults) {
			memcpy(out_material->param_buffer, global_buffer->defaults, out_material->param_buffer_size);
		} else {
			memset(out_material->param_buffer, 0, out_material->param_buffer_size);
		}
	}

	// Allocate bindings from global pool
	out_material->bind_count = meta->resource_count + meta->buffer_count;
	out_material->bind_start = _skr_bind_pool_alloc(out_material->bind_count);
	if (out_material->bind_start < 0 && out_material->bind_count > 0) {
		skr_log(skr_log_critical, "Failed to allocate material bindings from pool");
		_skr_free(out_material->param_buffer);
		if (out_material->key.shader->meta) {
			sksc_shader_meta_release(out_material->key.shader->meta);
		}
		*out_material = (skr_material_t){};
		return skr_err_out_of_memory;
	}
	skr_material_bind_t* binds = _skr_bind_pool_get(out_material->bind_start);
	for (uint32_t i = 0; i < meta->buffer_count;   i++) binds[i                   ].bind = meta->buffers  [i].bind;
	for (uint32_t i = 0; i < meta->resource_count; i++) binds[i+meta->buffer_count].bind = meta->resources[i].bind;

	// Check if we have a buffer bound to the system buffer slot
	out_material->has_system_buffer = false;
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		if (meta->buffers[i].bind.slot == SKR_BIND_SHIFT_BUFFER + _skr_vk.bind_settings.system_slot && meta->buffers[i].bind.stage_bits != 0) {
			out_material->has_system_buffer = true;
			break;
		}
	}

	// Check if we have a StructuredBuffer bound to the instance buffer slot
	out_material->instance_buffer_stride = 0;
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		if (meta->resources[i].bind.slot == SKR_BIND_SHIFT_TEXTURE + _skr_vk.bind_settings.instance_slot && meta->resources[i].bind.stage_bits != 0) {
			out_material->instance_buffer_stride = meta->resources[i].element_size;
			break;
		}
	}

	// Register material with pipeline system
	out_material->pipeline_material_idx = _skr_pipeline_register_material(&out_material->key);

	if (out_material->pipeline_material_idx < 0) {
		skr_log(skr_log_critical, "Failed to register material with pipeline system");
		_skr_bind_pool_free(out_material->bind_start, out_material->bind_count);
		_skr_free(out_material->param_buffer);
		if (out_material->key.shader->meta) {
			sksc_shader_meta_release(out_material->key.shader->meta);
		}
		*out_material = (skr_material_t){};
		return skr_err_device_error;
	}

	// Fill out default textures
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		skr_tex_t* tex = &_skr_vk.default_tex_white;
		if      (strcmp(meta->resources[i].value, "black") == 0) tex = &_skr_vk.default_tex_black;
		else if (strcmp(meta->resources[i].value, "gray" ) == 0) tex = &_skr_vk.default_tex_gray;
		else if (strcmp(meta->resources[i].value, "grey" ) == 0) tex = &_skr_vk.default_tex_gray;
		skr_material_set_tex(out_material, meta->resources[i].name, tex);
	}

	return skr_err_success;
}

bool skr_material_is_valid(const skr_material_t* material) {
	return material && material->pipeline_material_idx >= 0;
}

void skr_material_destroy(skr_material_t* ref_material) {
	if (!ref_material) return;

	// Unregister from pipeline system
	if (ref_material->pipeline_material_idx >= 0) {
		_skr_pipeline_unregister_material(ref_material->pipeline_material_idx);
	}

	// Free allocated memory (param_buffer is CPU-side, can free immediately)
	_skr_free(ref_material->param_buffer);

	// Defer bind pool slot release until GPU is done with this material
	_skr_cmd_destroy_bind_pool_slots(NULL, ref_material->bind_start, ref_material->bind_count);

	if (ref_material->key.shader && ref_material->key.shader->meta) {
		sksc_shader_meta_release(ref_material->key.shader->meta);
	}

	*ref_material = (skr_material_t){};
	ref_material->pipeline_material_idx = -1;
	ref_material->bind_start            = -1;
}

void skr_material_set_tex(skr_material_t* ref_material, const char* name, skr_tex_t* texture) {
	const sksc_shader_meta_t *meta = ref_material->key.shader->meta;

	int32_t  idx  = -1;
	uint64_t hash = skr_hash(name);
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		if (meta->resources[i].name_hash == hash) {
			idx  = i;
			break;
		}
	}

	if (idx == -1) {
		skr_log(skr_log_warning, "Texture name '%s' not found", name);
		return;
	}

	_skr_bind_pool_lock();
	skr_material_bind_t* binds = _skr_bind_pool_get(ref_material->bind_start);
	binds[meta->buffer_count + idx].texture = texture;
	_skr_bind_pool_unlock();
}

void skr_material_set_buffer(skr_material_t* ref_material, const char* name, skr_buffer_t* buffer) {
	const sksc_shader_meta_t *meta = ref_material->key.shader->meta;

	int32_t  idx  = -1;
	uint64_t hash = skr_hash(name);
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		if (meta->buffers[i].name_hash == hash) {
			idx = i;
			break;
		}
	}

	_skr_bind_pool_lock();
	skr_material_bind_t* binds = _skr_bind_pool_get(ref_material->bind_start);
	if (idx >= 0) {
		binds[idx].buffer = buffer;
		_skr_bind_pool_unlock();
		return;
	}

	// StructuredBuffers look like buffers, but HLSL treats them like textures/resources
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		if (meta->resources[i].name_hash == hash) {
			idx  = i;
			break;
		}
	}

	if (idx >= 0) {
		binds[meta->buffer_count + idx].buffer = buffer;
	} else {
		skr_log(skr_log_warning, "Buffer name '%s' not found", name);
	}
	_skr_bind_pool_unlock();
}

void skr_material_set_params(skr_material_t* ref_material, const void* data, uint32_t size) {
	if (size != ref_material->param_buffer_size) {
		skr_log(skr_log_warning, "material_set_params: incorrect size!");
		return;
	}
	memcpy(ref_material->param_buffer, data, size);
}


///////////////////////////////////////////////////////////////////////////////
// Material parameter setters/getters
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

void skr_material_set_param(skr_material_t* material, const char* name, sksc_shader_var_ type, uint32_t count, const void* data) {
	if (!material || !material->key.shader || !material->key.shader->meta || !material->param_buffer) return;

	int32_t var_index = sksc_shader_meta_get_var_index(material->key.shader->meta, name);
	if (var_index < 0) {
		skr_log(skr_log_warning, "Material parameter '%s' not found", name);
		return;
	}

	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(material->key.shader->meta, var_index);
	if (!var) return;

	// When type is uint8, treat count as raw byte count and skip type check
	uint32_t copy_size;
	if (type == sksc_shader_var_uint8) {
		copy_size = count;
	} else {
		if (var->type != type) {
			skr_log(skr_log_warning, "Material parameter '%s' type mismatch", name);
			return;
		}
		copy_size = _skr_shader_var_size(type) * count;
	}

	if (var->offset + copy_size > material->param_buffer_size) {
		skr_log(skr_log_warning, "Material parameter '%s' write would exceed buffer size", name);
		return;
	}

	memcpy((uint8_t*)material->param_buffer + var->offset, data, copy_size);
}

void skr_material_get_param(const skr_material_t* material, const char* name, sksc_shader_var_ type, uint32_t count, void* out_data) {
	if (!material || !material->key.shader || !material->key.shader->meta || !material->param_buffer) return;

	int32_t var_index = sksc_shader_meta_get_var_index(material->key.shader->meta, name);
	if (var_index < 0) {
		skr_log(skr_log_warning, "Material parameter '%s' not found", name);
		return;
	}

	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(material->key.shader->meta, var_index);
	if (!var) return;

	// When type is uint8, treat count as raw byte count and skip type check
	uint32_t copy_size;
	if (type == sksc_shader_var_uint8) {
		copy_size = count;
	} else {
		if (var->type != type) {
			skr_log(skr_log_warning, "Material parameter '%s' type mismatch", name);
			return;
		}
		copy_size = _skr_shader_var_size(type) * count;
	}

	if (var->offset + copy_size > material->param_buffer_size) {
		skr_log(skr_log_warning, "Material parameter '%s' read would exceed buffer size", name);
		return;
	}

	memcpy(out_data, (uint8_t*)material->param_buffer + var->offset, copy_size);
}

const char* _skr_material_bind_name(const sksc_shader_meta_t* meta, int32_t bind_idx) {
	if (!meta || bind_idx < 0) return "unknown";
	if ((uint32_t)bind_idx < meta->buffer_count) {
		return meta->buffers[bind_idx].name;
	}
	uint32_t res_idx = (uint32_t)bind_idx - meta->buffer_count;
	if (res_idx < meta->resource_count) {
		return meta->resources[res_idx].name;
	}
	return "unknown";
}

int32_t _skr_material_add_writes(const skr_material_bind_t* binds, uint32_t bind_ct, const int32_t* ignore_slots, int32_t ignore_ct, VkWriteDescriptorSet* ref_writes, uint32_t write_max, VkDescriptorBufferInfo* ref_buffer_infos, uint32_t buffer_max, VkDescriptorImageInfo* ref_image_infos, uint32_t image_max, uint32_t* ref_write_ct, uint32_t* ref_buffer_ct, uint32_t* ref_image_ct) {
 	for (uint32_t i = 0; i < bind_ct; i++) {
		int32_t       slot          = binds[i].bind.slot;
		skr_register_ register_type = binds[i].bind.register_type;

		bool skip = false;
		for (int32_t s = 0; s < ignore_ct; s++) {
			if (slot == ignore_slots[s]) {
				skip = true;
				break;
			}
		}
		if (skip) continue;

		switch(register_type) {
		case skr_register_constant: { // cbuffer, (b in HLSL)
			if (*ref_write_ct >= write_max || *ref_buffer_ct >= buffer_max) continue;

			skr_buffer_t* buffer  = _skr_vk.global_buffers[slot-SKR_BIND_SHIFT_BUFFER];
			uint32_t      offset  = 0;
			uint32_t      range   = 0;
			if (!buffer) {
				buffer = binds[i].buffer;
				offset = binds[i].buffer_offset;
				range  = binds[i].buffer_range;
			}
			if (!buffer) return (int32_t)i;

			ref_buffer_infos[*ref_buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = buffer->buffer,
				.offset = offset,
				.range  = range > 0 ? range : buffer->size,
			};
			ref_writes[*ref_write_ct] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo     = &ref_buffer_infos[*ref_buffer_ct],
			};
			(*ref_write_ct)++;
			(*ref_buffer_ct)++;
		} break;
		case skr_register_read_buffer: { // StructuredBuffer, (t in HLSL)
			if (*ref_write_ct >= write_max || *ref_buffer_ct >= buffer_max) continue;

			skr_buffer_t* buffer  = _skr_vk.global_buffers[slot-SKR_BIND_SHIFT_TEXTURE];
			uint32_t      offset  = 0;
			uint32_t      range   = 0;
			if (!buffer) {
				buffer = binds[i].buffer;
				offset = binds[i].buffer_offset;
				range  = binds[i].buffer_range;
			}
			if (!buffer) return (int32_t)i;

			ref_buffer_infos[*ref_buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = buffer->buffer,
				.offset = offset,
				.range  = range > 0 ? range : buffer->size,
			};
			ref_writes[*ref_write_ct] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo     = &ref_buffer_infos[*ref_buffer_ct],
			};
			(*ref_write_ct)++;
			(*ref_buffer_ct)++;
		} break;
		case skr_register_texture: { // Textures (Texture2D, etc.) (t in HLSL)
			if (*ref_write_ct >= write_max || *ref_image_ct >= image_max) continue;

			skr_tex_t* tex = _skr_vk.global_textures[slot-SKR_BIND_SHIFT_TEXTURE];
			if (!tex)  tex = binds[i].texture;
			if (!tex) return (int32_t)i;

			ref_image_infos[*ref_image_ct] = (VkDescriptorImageInfo){
				.sampler     = tex->sampler,
				.imageView   = tex->view,
				.imageLayout = (tex->flags & skr_tex_flags_compute)
					? VK_IMAGE_LAYOUT_GENERAL
					: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			ref_writes[*ref_write_ct] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo      = &ref_image_infos[*ref_image_ct],
			};
			(*ref_write_ct)++;
			(*ref_image_ct)++;
		} break;
		case skr_register_readwrite: { // RWStructuredBuffer (u in HLSL)
			if (*ref_write_ct >= write_max || *ref_buffer_ct >= buffer_max) continue;

			skr_buffer_t* buffer  = _skr_vk.global_buffers[slot-SKR_BIND_SHIFT_UAV];
			uint32_t      offset  = 0;
			uint32_t      range   = 0;
			if (!buffer) {
				buffer = binds[i].buffer;
				offset = binds[i].buffer_offset;
				range  = binds[i].buffer_range;
			}
			if (!buffer) return (int32_t)i;

			ref_buffer_infos[*ref_buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = buffer->buffer,
				.offset = offset,
				.range  = range > 0 ? range : buffer->size,
			};
			ref_writes[*ref_write_ct] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo     = &ref_buffer_infos[*ref_buffer_ct],
			};
			(*ref_write_ct)++;
			(*ref_buffer_ct)++;
		} break;
		case skr_register_readwrite_tex: { // Storage images (RWTexture2D, etc.)
			if (*ref_write_ct >= write_max || *ref_image_ct >= image_max) continue;

			skr_tex_t* tex = _skr_vk.global_textures[slot-SKR_BIND_SHIFT_UAV];
			if (!tex)  tex = binds[i].texture;
			if (!tex) return (int32_t)i;

			ref_image_infos[*ref_image_ct] = (VkDescriptorImageInfo){
				.sampler     = tex->sampler,
				.imageView   = tex->view,
				.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
			};
			ref_writes[*ref_write_ct] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo      = &ref_image_infos[*ref_image_ct],
			};
			(*ref_write_ct)++;
			(*ref_image_ct)++;
		} break;
		default: break;}
	}
	return -1;
}