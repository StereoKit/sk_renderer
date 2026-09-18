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
// Bind pool: slices live in fixed chunks that never move, so readers index
// them lock-free while one mutex serializes allocation and the destroy list
// defers frees past the frame. Setters write slot fields unlocked, so a
// material must not be modified while another thread is drawing with it.

#define _SKR_BIND_CHUNK_SHIFT 8
#define _SKR_BIND_CHUNK_SIZE  (1u << _SKR_BIND_CHUNK_SHIFT)
#define _SKR_BIND_CHUNK_MASK  (_SKR_BIND_CHUNK_SIZE - 1)
#define _SKR_BIND_MAX_CHUNKS  1024 // 256k binds; slices recycle, so this is headroom, not a leak budget

typedef struct { uint32_t start, count; } _skr_bind_range_t;

static _skr_atomic(skr_material_bind_t*) _bind_chunks[_SKR_BIND_MAX_CHUNKS];
static uint32_t           _bind_chunk_count; // writer mutex from here down
static uint32_t           _bind_chunk_used;  // entries used in the newest chunk
static _skr_bind_range_t* _bind_free;
static uint32_t           _bind_free_count;
static uint32_t           _bind_free_capacity;
static mtx_t              _bind_mutex;

void _skr_bind_pool_init(void) {
	mtx_init(&_bind_mutex, mtx_plain);
}

void _skr_bind_pool_shutdown(void) {
	for (uint32_t i = 0; i < _bind_chunk_count; i++) {
		_skr_free(_skr_load_acquire(&_bind_chunks[i]));
		_skr_store_release(&_bind_chunks[i], (skr_material_bind_t*)NULL);
	}
	_skr_free(_bind_free);
	_bind_free          = NULL;
	_bind_free_count    = 0;
	_bind_free_capacity = 0;
	_bind_chunk_count   = 0;
	_bind_chunk_used    = 0;
	mtx_destroy(&_bind_mutex);
}

static void _skr_bind_range_push(_skr_bind_range_t range) {
	if (_bind_free_count >= _bind_free_capacity) {
		_bind_free_capacity = _bind_free_capacity == 0 ? 16 : _bind_free_capacity * 2;
		_bind_free          = _skr_realloc(_bind_free, sizeof(_skr_bind_range_t) * _bind_free_capacity);
	}
	_bind_free[_bind_free_count++] = range;
}

int32_t _skr_bind_pool_alloc(uint32_t count) {
	if (count == 0) return -1;
	if (count > _SKR_BIND_CHUNK_SIZE) {
		skr_log(skr_log_critical, "Material needs %u binds, above the pool's %u limit", count, _SKR_BIND_CHUNK_SIZE);
		return -1;
	}
	mtx_lock(&_bind_mutex);

	int32_t result = -1;
	for (uint32_t i = 0; i < _bind_free_count; i++) {
		if (_bind_free[i].count < count) continue;
		result = (int32_t)_bind_free[i].start;
		if (_bind_free[i].count == count) _bind_free[i] = _bind_free[--_bind_free_count];
		else { _bind_free[i].start += count; _bind_free[i].count -= count; }
		break;
	}
	if (result < 0) {
		// A slice that doesn't fit the newest chunk opens another, tail freed
		if (_bind_chunk_count == 0 || _bind_chunk_used + count > _SKR_BIND_CHUNK_SIZE) {
			if (_bind_chunk_count >= _SKR_BIND_MAX_CHUNKS) {
				skr_log(skr_log_critical, "Bind pool is out of chunks");
				mtx_unlock(&_bind_mutex);
				return -1;
			}
			uint32_t tail = _bind_chunk_count > 0 ? _SKR_BIND_CHUNK_SIZE - _bind_chunk_used : 0;
			if (tail > 0)
				_skr_bind_range_push((_skr_bind_range_t){ .start = ((_bind_chunk_count - 1) << _SKR_BIND_CHUNK_SHIFT) + _bind_chunk_used, .count = tail });
			skr_material_bind_t* chunk = _skr_calloc(_SKR_BIND_CHUNK_SIZE, sizeof(skr_material_bind_t));
			_skr_store_release(&_bind_chunks[_bind_chunk_count], chunk);
			_bind_chunk_count += 1;
			_bind_chunk_used   = 0;
		}
		result = (int32_t)(((_bind_chunk_count - 1) << _SKR_BIND_CHUNK_SHIFT) + _bind_chunk_used);
		_bind_chunk_used += count;
	}
	skr_material_bind_t* chunk = _skr_load_acquire(&_bind_chunks[result >> _SKR_BIND_CHUNK_SHIFT]);
	memset(&chunk[result & _SKR_BIND_CHUNK_MASK], 0, sizeof(skr_material_bind_t) * count);

	mtx_unlock(&_bind_mutex);
	return result;
}

void _skr_bind_pool_free(int32_t start, uint32_t count) {
	if (start < 0 || count == 0) return;
	uint32_t ustart = (uint32_t)start;
	uint32_t end    = ustart + count;
	uint32_t chunk  = ustart >> _SKR_BIND_CHUNK_SHIFT;
	mtx_lock(&_bind_mutex);

	memset(_skr_bind_pool_get(start), 0, sizeof(skr_material_bind_t) * count); // catches use-after-free

	// Coalesce with an adjacent free range, but never across a chunk boundary
	bool merged = false;
	for (uint32_t i = 0; i < _bind_free_count && !merged; i++) {
		if ((_bind_free[i].start >> _SKR_BIND_CHUNK_SHIFT) != chunk) continue;
		if (_bind_free[i].start + _bind_free[i].count == ustart) {
			_bind_free[i].count += count;
			for (uint32_t j = 0; j < _bind_free_count; j++)
				if (j != i && _bind_free[j].start == end && (_bind_free[j].start >> _SKR_BIND_CHUNK_SHIFT) == chunk) {
					_bind_free[i].count += _bind_free[j].count;
					_bind_free[j] = _bind_free[--_bind_free_count];
					break;
				}
			merged = true;
		} else if (_bind_free[i].start == end) {
			_bind_free[i].start  = ustart;
			_bind_free[i].count += count;
			merged = true;
		}
	}
	if (!merged)
		_skr_bind_range_push((_skr_bind_range_t){ .start = ustart, .count = count });

	mtx_unlock(&_bind_mutex);
}

skr_material_bind_t* _skr_bind_pool_get(int32_t start) {
	if (start < 0) return NULL;
	skr_material_bind_t* chunk = _skr_load_acquire(&_bind_chunks[start >> _SKR_BIND_CHUNK_SHIFT]);
	return chunk ? &chunk[start & _SKR_BIND_CHUNK_MASK] : NULL;
}

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_material_create(skr_material_info_t info, skr_material_t* out_material) {
	if (!out_material) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_material = (skr_material_t){0};

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
		.wireframe         = info.wireframe,
		.stencil_front     = info.stencil_front,
		.stencil_back      = info.stencil_back,
	};
	_skr_shader_resolve_spec_constants(&info.shader->meta, info.spec_constants, info.spec_constant_count, out_material->key.spec_constant_values);
	out_material->queue_offset = info.queue_offset;

	const sksc_shader_meta_t* meta = &out_material->key.shader->meta;

	// Allocate material parameter buffer if shader has $Global buffer
	if (meta->global_buffer_id >= 0) {
		sksc_shader_buffer_t* global_buffer = &meta->buffers[meta->global_buffer_id];
		out_material->param_buffer_size = global_buffer->size;
		out_material->param_buffer = _skr_malloc(out_material->param_buffer_size);

		if (!out_material->param_buffer) {
			skr_log(skr_log_critical, "Failed to allocate material parameter buffer");
			*out_material = (skr_material_t){0};
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
		*out_material = (skr_material_t){0};
		return skr_err_out_of_memory;
	}
	skr_material_bind_t* binds = _skr_bind_pool_get(out_material->bind_start);
	for (uint32_t i = 0; i < meta->buffer_count;   i++) binds[i                   ].bind = meta->buffers  [i].bind;
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		binds[i+meta->buffer_count].bind = meta->resources[i].bind;
		// Shape bit 6 on sampled textures: the shader uses QCOM image-processing
		// ops, which require the dedicated IMAGE_PROCESSING sampler at descriptor
		// time. On storage images the same bit records write usage instead.
		binds[i+meta->buffer_count].image_proc_sampler = meta->resources[i].bind.register_type == skr_register_texture && (meta->resources[i].shape & SKSC_SHAPE_WRITTEN) != 0; // QCOM sampler bit on sampled textures
	}

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
		*out_material = (skr_material_t){0};
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

void skr_material_set_pipeline(skr_material_t* ref_material, skr_material_info_t info) {
	_skr_pipeline_unregister_material(ref_material->pipeline_material_idx);

	ref_material->key.cull              = info.cull;
	ref_material->key.write_mask        = info.write_mask ? info.write_mask : skr_write_default;
	ref_material->key.depth_test        = info.depth_test;
	ref_material->key.blend_state       = info.blend_state;
	ref_material->key.alpha_to_coverage = info.alpha_to_coverage;
	ref_material->key.depth_clamp       = info.depth_clamp;
	ref_material->key.wireframe         = info.wireframe;
	ref_material->key.stencil_front     = info.stencil_front;
	ref_material->key.stencil_back      = info.stencil_back;
	_skr_shader_resolve_spec_constants(&ref_material->key.shader->meta, info.spec_constants, info.spec_constant_count, ref_material->key.spec_constant_values);
	ref_material->queue_offset          = info.queue_offset;
	ref_material->pipeline_material_idx = _skr_pipeline_register_material(&ref_material->key);
}

bool skr_material_is_valid(const skr_material_t* material) {
	// The shader check matters for zero-initialized materials that were never
	// created: pipeline_material_idx is 0 there, which reads as a valid index.
	return material && material->key.shader != NULL && material->pipeline_material_idx >= 0;
}

void skr_material_destroy(skr_material_t* ref_material) {
	if (!ref_material || !ref_material->key.shader) return;

	// Unregister from pipeline system
	if (ref_material->pipeline_material_idx >= 0) {
		_skr_pipeline_unregister_material(ref_material->pipeline_material_idx);
	}

	// Free allocated memory (param_buffer is CPU-side, can free immediately)
	_skr_free(ref_material->param_buffer);

	// Defer bind pool slot release until GPU is done with this material
	_skr_cmd_destroy_bind_pool_slots(NULL, ref_material->bind_start, ref_material->bind_count);

	*ref_material = (skr_material_t){0};
}

void skr_material_set_tex(skr_material_t* ref_material, const char* name, skr_tex_t* texture) {
	const sksc_shader_meta_t *meta = &ref_material->key.shader->meta;

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

	skr_material_bind_t* binds = _skr_bind_pool_get(ref_material->bind_start);
	binds[meta->buffer_count + idx].texture = texture;

	// Auto-detect YCbCr immutable sampler: if the texture carries a ycbcr_sampler,
	// bake it into the descriptor set layout for this binding slot. This is required
	// by Vulkan for VkSamplerYcbcrConversion — the sampler must be specified at
	// layout creation time, not at descriptor write time.
	VkSampler new_ycbcr = texture ? texture->ycbcr_sampler : VK_NULL_HANDLE;
	int32_t   slot      = meta->resources[idx].bind.slot;

	// Build updated immutable sampler list: copy existing, update/add/remove for this slot
	_skr_pipeline_material_key_t new_key = ref_material->key;
	int32_t found = -1;
	for (int32_t i = 0; i < new_key.immutable_sampler_count; i++) {
		if (new_key.immutable_sampler_slots[i] == slot) { found = i; break; }
	}

	bool changed = false;
	if (new_ycbcr != VK_NULL_HANDLE) {
		if (found >= 0) {
			// Update existing entry
			if (new_key.immutable_samplers[found] != new_ycbcr) {
				new_key.immutable_samplers[found] = new_ycbcr;
				changed = true;
			}
		} else if (new_key.immutable_sampler_count < SKR_MAX_IMMUTABLE_SAMPLERS) {
			// Insert new entry, keep sorted by slot
			int32_t insert = new_key.immutable_sampler_count;
			for (int32_t i = 0; i < new_key.immutable_sampler_count; i++) {
				if (slot < new_key.immutable_sampler_slots[i]) { insert = i; break; }
			}
			// Shift entries after insert point
			for (int32_t i = new_key.immutable_sampler_count; i > insert; i--) {
				new_key.immutable_samplers[i]     = new_key.immutable_samplers[i - 1];
				new_key.immutable_sampler_slots[i] = new_key.immutable_sampler_slots[i - 1];
			}
			new_key.immutable_samplers[insert]     = new_ycbcr;
			new_key.immutable_sampler_slots[insert] = slot;
			new_key.immutable_sampler_count++;
			changed = true;
		} else {
			skr_log(skr_log_warning, "skr_material_set_tex: too many YCbCr textures (max %d)", SKR_MAX_IMMUTABLE_SAMPLERS);
		}
	} else if (found >= 0) {
		// Remove entry for this slot
		for (int32_t i = found; i < new_key.immutable_sampler_count - 1; i++) {
			new_key.immutable_samplers[i]     = new_key.immutable_samplers[i + 1];
			new_key.immutable_sampler_slots[i] = new_key.immutable_sampler_slots[i + 1];
		}
		new_key.immutable_sampler_count--;
		new_key.immutable_samplers[new_key.immutable_sampler_count]     = VK_NULL_HANDLE;
		new_key.immutable_sampler_slots[new_key.immutable_sampler_count] = 0;
		changed = true;
	}

	if (changed) {
		_skr_pipeline_unregister_material(ref_material->pipeline_material_idx);
		ref_material->key = new_key;
		ref_material->pipeline_material_idx = _skr_pipeline_register_material(&ref_material->key);
	}
}

void skr_material_set_buffer(skr_material_t* ref_material, const char* name, skr_buffer_t* buffer) {
	const sksc_shader_meta_t *meta = &ref_material->key.shader->meta;

	int32_t  idx  = -1;
	uint64_t hash = skr_hash(name);
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		if (meta->buffers[i].name_hash == hash) {
			idx = i;
			break;
		}
	}

	skr_material_bind_t* binds = _skr_bind_pool_get(ref_material->bind_start);
	if (idx >= 0) {
		binds[idx].buffer = buffer;
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

// Packs a shader parameter into any param buffer with the shader's $Global
// layout, so per-draw parameter blocks can be built without staging them
// through a material's own (possibly shared) buffer.
void _skr_shader_param_write(const sksc_shader_meta_t* meta, void* param_buffer, uint32_t param_buffer_size, const char* name, sksc_shader_var_ type, uint32_t count, const void* data) {

	int32_t var_index = sksc_shader_meta_get_var_index(meta, name);
	if (var_index < 0) {
		skr_log(skr_log_warning, "Material parameter '%s' not found", name);
		return;
	}

	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(meta, var_index);
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

	if (var->offset + copy_size > param_buffer_size) {
		skr_log(skr_log_warning, "Material parameter '%s' write would exceed buffer size", name);
		return;
	}

	memcpy((uint8_t*)param_buffer + var->offset, data, copy_size);
}

void skr_material_set_param(skr_material_t* material, const char* name, sksc_shader_var_ type, uint32_t count, const void* data) {
	_skr_shader_param_write(&material->key.shader->meta, material->param_buffer, material->param_buffer_size, name, type, count, data);
}

void skr_material_get_param(const skr_material_t* material, const char* name, sksc_shader_var_ type, uint32_t count, void* out_data) {

	int32_t var_index = sksc_shader_meta_get_var_index(&material->key.shader->meta, name);
	if (var_index < 0) {
		skr_log(skr_log_warning, "Material parameter '%s' not found", name);
		return;
	}

	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(&material->key.shader->meta, var_index);
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

bool skr_material_get_param_info(const skr_material_t* material, const char* param_name, skr_shader_param_info_t* opt_out_info) {
	if (!material) return false;
	return skr_shader_get_param_info(material->key.shader, param_name, opt_out_info);
}

bool skr_material_get_tex_info(const skr_material_t* material, const char* tex_name, skr_shader_tex_info_t* opt_out_info) {
	if (!material) return false;
	return skr_shader_get_tex_info(material->key.shader, tex_name, opt_out_info);
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

int32_t _skr_material_add_writes(const skr_material_bind_t* binds, uint32_t bind_ct, skr_stage_ stage_mask, const int32_t* ignore_slots, int32_t ignore_ct, _skr_desc_writes_t* ref_writes) {
	for (uint32_t i = 0; i < bind_ct; i++) {
		int32_t       slot          = binds[i].bind.slot;
		skr_register_ register_type = binds[i].bind.register_type;

		// Resources outside the pipeline's stages get no descriptor: a graphics
		// bind must not write a vs/ps/cs shader's compute-only storage image
		// (often holding a storage-less default texture), and vice versa.
		if ((binds[i].bind.stage_bits & stage_mask) == 0) continue;

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
			skr_buffer_t* buffer  = _skr_vk.global_buffers[slot-SKR_BIND_SHIFT_BUFFER];
			uint32_t      offset  = 0;
			uint32_t      range   = 0;
			if (!buffer) {
				buffer = binds[i].buffer;
				offset = binds[i].buffer_offset;
				range  = binds[i].buffer_range;
			}
			if (!buffer) return (int32_t)i;

			_skr_write_buffer(ref_writes, (uint32_t)slot, false, buffer->buffer, offset, range > 0 ? range : buffer->size);
		} break;
		case skr_register_read_buffer: { // StructuredBuffer, (t in HLSL)
			skr_buffer_t* buffer  = _skr_vk.global_buffers[slot-SKR_BIND_SHIFT_TEXTURE];
			uint32_t      offset  = 0;
			uint32_t      range   = 0;
			if (!buffer) {
				buffer = binds[i].buffer;
				offset = binds[i].buffer_offset;
				range  = binds[i].buffer_range;
			}
			if (!buffer) return (int32_t)i;

			_skr_write_buffer(ref_writes, (uint32_t)slot, true, buffer->buffer, offset, range > 0 ? range : buffer->size);
		} break;
		case skr_register_texture: { // Textures (Texture2D, etc.) (t in HLSL)
			skr_tex_t* tex = _skr_vk.global_textures[slot-SKR_BIND_SHIFT_TEXTURE];
			if (!tex)  tex = binds[i].texture;
			if (!tex) return (int32_t)i;

			// imageLayout must match the layout the GPU sees when the descriptor
			// is accessed (VUID-VkDescriptorImageInfo-imageLayout-00344). We
			// can't snapshot tex->current_layout: it's a CPU-side trajectory
			// mutated by every transition, so a concurrent upload or an
			// intermediate state (e.g. TRANSFER_DST mid-upload, or an external
			// producer in TRANSFER_SRC after a copy) would get baked into the
			// descriptor and mismatch the actual layout at draw time — and
			// non-sampling layouts like TRANSFER_SRC are illegal for SAMPLED
			// descriptors anyway. Instead, _skr_tex_sample_layout derives the
			// destination sampling layout from texture properties, and the
			// transition helpers use the same function — so the GPU state and
			// the descriptor agree by construction. The producer of an external
			// texture is responsible for transitioning it to this layout before
			// any draw that samples it.
			// Image-processing bindings (BoxFilterQCOM etc.) are only legal with
			// the dedicated IMAGE_PROCESSING sampler, never the texture's own.
			// The fallback to tex->sampler only happens when the device lacks the
			// extension — in which case skr_shader_check_support already failed
			// this shader and the pipeline is expected to be rejected anyway.
			VkSampler sampler = (binds[i].image_proc_sampler && _skr_vk.sampler_image_proc != VK_NULL_HANDLE)
				? _skr_vk.sampler_image_proc : tex->sampler;
			_skr_write_image(ref_writes, (uint32_t)slot, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampler, tex->view, _skr_tex_sample_layout(tex));
		} break;
		case skr_register_readwrite: { // RWStructuredBuffer (u in HLSL)
			skr_buffer_t* buffer  = _skr_vk.global_buffers[slot-SKR_BIND_SHIFT_UAV];
			uint32_t      offset  = 0;
			uint32_t      range   = 0;
			if (!buffer) {
				buffer = binds[i].buffer;
				offset = binds[i].buffer_offset;
				range  = binds[i].buffer_range;
			}
			if (!buffer) return (int32_t)i;

			_skr_write_buffer(ref_writes, (uint32_t)slot, true, buffer->buffer, offset, range > 0 ? range : buffer->size);
		} break;
		case skr_register_readwrite_tex: { // Storage images (RWTexture2D, etc.)
			skr_tex_t* tex = _skr_vk.global_textures[slot-SKR_BIND_SHIFT_UAV];
			if (!tex)  tex = binds[i].texture;
			if (!tex) return (int32_t)i;

			// _skr_tex_sample_layout returns GENERAL for compute-flagged textures —
			// same value the spec mandates for STORAGE_IMAGE, but routes through the
			// single canonical-layout helper for consistency with the sampled path.
			_skr_write_image(ref_writes, (uint32_t)slot, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, tex->sampler, tex->view, _skr_tex_sample_layout(tex));
		} break;
		case skr_register_input_attachment: { // Input attachments (SubpassInput)
			skr_tex_t* tex = binds[i].texture;
			if (!tex) return (int32_t)i;

			_skr_write_image(ref_writes, (uint32_t)slot, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VK_NULL_HANDLE, tex->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		} break;
		case skr_register_tile_sampled:      // [tile_attachment] Texture2D — VK_QCOM_tile_shading
		case skr_register_tile_storage: {    // [tile_attachment] RWTexture2D
			skr_tex_t* tex = binds[i].texture;
			if (!tex) return (int32_t)i;

			// The bound image must be an attachment of the current tile shading
			// render pass, and the descriptor layout must match the image's
			// layout during the reading subpass. Sampled: the postfx chain also
			// references it as an input attachment there, which puts it in
			// SHADER_READ_ONLY_OPTIMAL — the same layout _skr_tex_sample_layout
			// reports for a readable color texture. Storage: storage-image
			// descriptors require GENERAL unconditionally
			// (VUID-VkWriteDescriptorSet-descriptorType-04152), so the caller
			// must keep the attachment in GENERAL across the reading subpass.
			bool sampled = register_type == skr_register_tile_sampled;
			_skr_write_image(ref_writes, (uint32_t)slot,
				sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				sampled ? tex->sampler : VK_NULL_HANDLE, tex->view,
				sampled ? _skr_tex_sample_layout(tex) : VK_IMAGE_LAYOUT_GENERAL);
		} break;
		default: break;}
	}
	return -1;
}
