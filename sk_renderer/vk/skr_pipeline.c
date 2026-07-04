// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"

#include "skr_pipeline.h"
#include "skr_conversions.h"

#include <threads.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

typedef struct {
	_skr_pipeline_material_key_t     key;
	VkPipelineLayout                 layout;
	VkDescriptorSetLayout            descriptor_layout;
	int32_t                          ref_count;
} _skr_pipeline_material_slot_t;

typedef struct {
	skr_pipeline_renderpass_key_t    key;
	VkRenderPass                     render_pass;
	int32_t                          ref_count;
} _skr_pipeline_renderpass_slot_t;

typedef struct {
	skr_vert_type_t                  vert_type;
	int32_t                          ref_count;
} _skr_pipeline_vertformat_slot_t;

typedef struct {
	_skr_pipeline_material_slot_t*   materials;
	_skr_pipeline_renderpass_slot_t* renderpasses;
	_skr_pipeline_vertformat_slot_t* vertformats;
	VkPipeline*                      pipelines;       // 3D array: [material][renderpass][vertformat]
	int32_t                          material_count;
	int32_t                          material_capacity;
	int32_t                          renderpass_count;
	int32_t                          renderpass_capacity;
	int32_t                          vertformat_count;
	int32_t                          vertformat_capacity;
	mtx_t                            mutex;           // Thread safety for cache access
} _skr_pipeline_cache_t;

///////////////////////////////////////////////////////////////////////////////
// State
///////////////////////////////////////////////////////////////////////////////

static _skr_pipeline_cache_t _skr_pipeline_cache = {0};

///////////////////////////////////////////////////////////////////////////////
// Forward declarations
///////////////////////////////////////////////////////////////////////////////

static VkRenderPass     _skr_pipeline_create_renderpass(const skr_pipeline_renderpass_key_t* key);
static VkPipelineLayout _skr_pipeline_create_layout    (VkDescriptorSetLayout descriptor_layout);
static VkPipeline       _skr_pipeline_create           (int32_t material_idx, int32_t renderpass_idx, int32_t vertformat_idx);

///////////////////////////////////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////////////////////////////////

static inline int32_t _skr_pipeline_index_3d(int32_t m, int32_t r, int32_t v, int32_t renderpass_cap, int32_t vertfmt_cap) {
	return (m * renderpass_cap * vertfmt_cap) +
	       (r * vertfmt_cap) +
	       v;
}

// Shared pipeline 3D array grow logic
static void _skr_pipeline_grow_pipelines_array(VkPipeline** ref_pipelines, int32_t old_m, int32_t new_m, int32_t old_r, int32_t new_r, int32_t old_v, int32_t new_v) {
	int32_t old_size = old_m * old_r * old_v;
	int32_t new_size = new_m * new_r * new_v;

	if (new_size == 0) return;

	VkPipeline* new_pipelines = _skr_calloc(new_size, sizeof(VkPipeline));

	// Copy existing pipelines to new layout
	if (*ref_pipelines && old_size > 0) {
		for (int32_t m = 0; m < old_m; m++) {
			for (int32_t r = 0; r < old_r; r++) {
				for (int32_t v = 0; v < old_v; v++) {
					int32_t old_idx = (m * old_r * old_v) + (r * old_v) + v;
					int32_t new_idx = (m * new_r * new_v) + (r * new_v) + v;
					new_pipelines[new_idx] = (*ref_pipelines)[old_idx];
				}
			}
		}
		_skr_free(*ref_pipelines);
	}
	*ref_pipelines = new_pipelines;
}

///////////////////////////////////////////////////////////////////////////////

void _skr_pipeline_init(void) {
	_skr_pipeline_cache = (_skr_pipeline_cache_t){0};
	mtx_init(&_skr_pipeline_cache.mutex, mtx_plain);
}

void _skr_pipeline_lock(void) {
	mtx_lock(&_skr_pipeline_cache.mutex);
}

void _skr_pipeline_unlock(void) {
	mtx_unlock(&_skr_pipeline_cache.mutex);
}

void _skr_pipeline_shutdown(void) {
	// This happens during shutdown, so it's safe, and preferable to directly
	// destroy Vulkan asssets, instead of using the deferred asset destroy
	// system.

	// Destroy all pipelines
	if (_skr_pipeline_cache.pipelines) {
		for (int32_t m = 0; m < _skr_pipeline_cache.material_capacity; m++) {
			for (int32_t r = 0; r < _skr_pipeline_cache.renderpass_capacity; r++) {
				for (int32_t v = 0; v < _skr_pipeline_cache.vertformat_capacity; v++) {
					int32_t idx = _skr_pipeline_index_3d(m, r, v, _skr_pipeline_cache.renderpass_capacity, _skr_pipeline_cache.vertformat_capacity);
					if (_skr_pipeline_cache.pipelines[idx] != VK_NULL_HANDLE) {
						vkDestroyPipeline(_skr_vk.device, _skr_pipeline_cache.pipelines[idx], NULL);
					}
				}
			}
		}
		_skr_free(_skr_pipeline_cache.pipelines);
	}

	// Destroy material resources
	if (_skr_pipeline_cache.materials) {
		for (int32_t m = 0; m < _skr_pipeline_cache.material_capacity; m++) {
			if (_skr_pipeline_cache.materials[m].ref_count > 0) {
				if (_skr_pipeline_cache.materials[m].layout != VK_NULL_HANDLE) {
					vkDestroyPipelineLayout(_skr_vk.device, _skr_pipeline_cache.materials[m].layout, NULL);
				}
				if (_skr_pipeline_cache.materials[m].descriptor_layout != VK_NULL_HANDLE) {
					vkDestroyDescriptorSetLayout(_skr_vk.device, _skr_pipeline_cache.materials[m].descriptor_layout, NULL);
				}
			}
		}
		_skr_free(_skr_pipeline_cache.materials);
	}

	// Destroy render passes (shared handles may appear in multiple slots)
	if (_skr_pipeline_cache.renderpasses) {
		for (int32_t r = 0; r < _skr_pipeline_cache.renderpass_capacity; r++) {
			VkRenderPass rp = _skr_pipeline_cache.renderpasses[r].render_pass;
			if (_skr_pipeline_cache.renderpasses[r].ref_count <= 0 || rp == VK_NULL_HANDLE) continue;

			vkDestroyRenderPass(_skr_vk.device, rp, NULL);
			// Null out any other slots sharing this handle to avoid double-free
			for (int32_t j = r + 1; j < _skr_pipeline_cache.renderpass_capacity; j++) {
				if (_skr_pipeline_cache.renderpasses[j].render_pass == rp)
					_skr_pipeline_cache.renderpasses[j].render_pass = VK_NULL_HANDLE;
			}
		}
		_skr_free(_skr_pipeline_cache.renderpasses);
	}

	// Free vertex formats
	if (_skr_pipeline_cache.vertformats) {
		_skr_free(_skr_pipeline_cache.vertformats);
	}

	mtx_destroy(&_skr_pipeline_cache.mutex);
	_skr_pipeline_cache = (_skr_pipeline_cache_t){0};
}

static void _skr_pipeline_grow_materials(_skr_pipeline_cache_t* ref_cache, int32_t min_capacity) {
	if (min_capacity <= ref_cache->material_capacity) return;

	int32_t old_capacity = ref_cache->material_capacity;
	int32_t new_capacity = old_capacity == 0 ? 8 : old_capacity * 2;
	while (new_capacity < min_capacity) {
		new_capacity *= 2;
	}

	// Grow materials array
	ref_cache->materials = _skr_realloc(ref_cache->materials, new_capacity * sizeof(_skr_pipeline_material_slot_t));
	memset(&ref_cache->materials[old_capacity], 0, (new_capacity - old_capacity) * sizeof(_skr_pipeline_material_slot_t));

	// Grow pipelines 3D array
	_skr_pipeline_grow_pipelines_array(
		&ref_cache->pipelines,
		old_capacity, new_capacity,
		ref_cache->renderpass_capacity, ref_cache->renderpass_capacity,
		ref_cache->vertformat_capacity, ref_cache->vertformat_capacity
	);

	ref_cache->material_capacity = new_capacity;
}

int32_t _skr_pipeline_register_material(const _skr_pipeline_material_key_t* key) {
	mtx_lock(&_skr_pipeline_cache.mutex);

	// Find existing or free slot
	int32_t free_slot = -1;
	for (int32_t i = 0; i < _skr_pipeline_cache.material_capacity; i++) {
		if (_skr_pipeline_cache.materials[i].ref_count > 0) {
			// Check if this material already exists
			if (memcmp(&_skr_pipeline_cache.materials[i].key, key, sizeof(_skr_pipeline_material_key_t)) == 0) {
				_skr_pipeline_cache.materials[i].ref_count++;
				mtx_unlock(&_skr_pipeline_cache.mutex);
				return i;
			}
		} else if (free_slot == -1) {
			free_slot = i;
		}
	}

	// If no free slot, grow the array
	if (free_slot == -1) {
		free_slot = _skr_pipeline_cache.material_capacity;
		_skr_pipeline_grow_materials(&_skr_pipeline_cache, free_slot + 1);
	}

	// Register new material
	_skr_pipeline_cache.materials[free_slot].key               = *key;
	_skr_pipeline_cache.materials[free_slot].descriptor_layout = _skr_shader_make_layout    (_skr_vk.device, _skr_vk.has_push_descriptors, &key->shader->meta, skr_stage_vertex | skr_stage_pixel | skr_stage_compute, key->immutable_samplers, key->immutable_sampler_slots, key->immutable_sampler_count);
	_skr_pipeline_cache.materials[free_slot].layout            = _skr_pipeline_create_layout(_skr_pipeline_cache.materials[free_slot].descriptor_layout);
	_skr_pipeline_cache.materials[free_slot].ref_count         = 1;

	if (free_slot >= _skr_pipeline_cache.material_count) {
		_skr_pipeline_cache.material_count = free_slot + 1;
	}

	// Generate and set debug name for pipeline layout
	char name[256];
	const char* shader_name = key->shader->meta.name[0] ? key->shader->meta.name : "unknown";
	snprintf(name, sizeof(name), "layout_%s_", shader_name);
	_skr_append_material_config(name, sizeof(name), key);
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)_skr_pipeline_cache.materials[free_slot].layout, name);

	// Generate debug name based on shader
	snprintf(name, sizeof(name), "layoutdesc_%s_", shader_name);
	_skr_append_material_config(name, sizeof(name), key);
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, (uint64_t)_skr_pipeline_cache.materials[free_slot].descriptor_layout, name);

	mtx_unlock(&_skr_pipeline_cache.mutex);
	return free_slot;
}

static void _skr_pipeline_grow_renderpasses(_skr_pipeline_cache_t* ref_cache, int32_t min_capacity) {
	if (min_capacity <= ref_cache->renderpass_capacity) return;

	int32_t old_capacity = ref_cache->renderpass_capacity;
	int32_t new_capacity = old_capacity == 0 ? 4 : old_capacity * 2;
	while (new_capacity < min_capacity) {
		new_capacity *= 2;
	}

	// Grow renderpasses array
	ref_cache->renderpasses = _skr_realloc(ref_cache->renderpasses, new_capacity * sizeof(_skr_pipeline_renderpass_slot_t));
	memset(&ref_cache->renderpasses[old_capacity], 0, (new_capacity - old_capacity) * sizeof(_skr_pipeline_renderpass_slot_t));

	// Grow pipelines 3D array
	_skr_pipeline_grow_pipelines_array(
		&ref_cache->pipelines,
		ref_cache->material_capacity, ref_cache->material_capacity,
		old_capacity, new_capacity,
		ref_cache->vertformat_capacity, ref_cache->vertformat_capacity
	);

	ref_cache->renderpass_capacity = new_capacity;
}

// Compare two renderpass keys ignoring subpass_index. Keys that match on all
// other fields describe the same VkRenderPass — subpass_index only affects
// which subpass a pipeline targets within that renderpass.
static bool _skr_renderpass_key_matches_base(const skr_pipeline_renderpass_key_t* a, const skr_pipeline_renderpass_key_t* b) {
	return a->color_format             == b->color_format
	    && a->depth_format             == b->depth_format
	    && a->resolve_format           == b->resolve_format
	    && a->samples                  == b->samples
	    && a->depth_store_op           == b->depth_store_op
	    && a->color_load_op            == b->color_load_op
	    && a->view_mask                == b->view_mask
	    && a->correlation_mask         == b->correlation_mask
	    && a->postfx_count             == b->postfx_count
	    && a->has_resolve_subpass      == b->has_resolve_subpass
	    && a->use_custom_resolve_flags == b->use_custom_resolve_flags
	    && a->postfx_output_format     == b->postfx_output_format
	    && a->final_color_layout       == b->final_color_layout
	    && a->final_resolve_layout     == b->final_resolve_layout
	    && a->final_depth_layout       == b->final_depth_layout;
}

// Unlocked version - caller must hold the mutex via _skr_pipeline_lock()
int32_t _skr_pipeline_register_renderpass_unlocked(const skr_pipeline_renderpass_key_t* key) {
	// Find existing or free slot
	int32_t free_slot  = -1;
	int32_t share_slot = -1; // Slot with same base key (different subpass_index) to share VkRenderPass from
	for (int32_t i = 0; i < _skr_pipeline_cache.renderpass_capacity; i++) {
		if (_skr_pipeline_cache.renderpasses[i].ref_count > 0) {
			// Exact match (including subpass_index): reuse this slot entirely
			if (memcmp(&_skr_pipeline_cache.renderpasses[i].key, key, sizeof(skr_pipeline_renderpass_key_t)) == 0) {
				_skr_pipeline_cache.renderpasses[i].ref_count++;
				return i;
			}
			// Base match: candidate to borrow the VkRenderPass from
			if (share_slot == -1 && _skr_renderpass_key_matches_base(&_skr_pipeline_cache.renderpasses[i].key, key)) {
				share_slot = i;
			}
		} else if (free_slot == -1) {
			free_slot = i;
		}
	}

	// If no free slot, grow the array
	if (free_slot == -1) {
		free_slot = _skr_pipeline_cache.renderpass_capacity;
		_skr_pipeline_grow_renderpasses(&_skr_pipeline_cache, free_slot + 1);
	}

	// Register new slot. Share the VkRenderPass if another slot with the same
	// base key already exists (different subpass_index, same renderpass object).
	_skr_pipeline_cache.renderpasses[free_slot].key       = *key;
	_skr_pipeline_cache.renderpasses[free_slot].ref_count = 1;
	if (share_slot >= 0) {
		_skr_pipeline_cache.renderpasses[free_slot].render_pass = _skr_pipeline_cache.renderpasses[share_slot].render_pass;
	} else {
		_skr_pipeline_cache.renderpasses[free_slot].render_pass = _skr_pipeline_create_renderpass(key);
	}

	if (free_slot >= _skr_pipeline_cache.renderpass_count) {
		_skr_pipeline_cache.renderpass_count = free_slot + 1;
	}

	return free_slot;
}

int32_t _skr_pipeline_register_renderpass(const skr_pipeline_renderpass_key_t* key) {
	mtx_lock(&_skr_pipeline_cache.mutex);
	int32_t result = _skr_pipeline_register_renderpass_unlocked(key);
	mtx_unlock(&_skr_pipeline_cache.mutex);
	return result;
}

void _skr_pipeline_unregister_material(int32_t material_idx) {
	mtx_lock(&_skr_pipeline_cache.mutex);

	if (material_idx < 0 || material_idx >= _skr_pipeline_cache.material_capacity) { mtx_unlock(&_skr_pipeline_cache.mutex); return; }
	if (_skr_pipeline_cache.materials[material_idx].ref_count <= 0)                { mtx_unlock(&_skr_pipeline_cache.mutex); return; }

	_skr_pipeline_cache.materials[material_idx].ref_count--;
	if (_skr_pipeline_cache.materials[material_idx].ref_count > 0) { mtx_unlock(&_skr_pipeline_cache.mutex); return; }

	// Destroy all pipelines using this material
	for (int32_t r = 0; r < _skr_pipeline_cache.renderpass_capacity; r++) {
		for (int32_t v = 0; v < _skr_pipeline_cache.vertformat_capacity; v++) {
			int32_t idx = _skr_pipeline_index_3d(material_idx, r, v, _skr_pipeline_cache.renderpass_capacity, _skr_pipeline_cache.vertformat_capacity);
			_skr_cmd_destroy_pipeline(NULL, _skr_pipeline_cache.pipelines[idx]);
			_skr_pipeline_cache.pipelines[idx] = VK_NULL_HANDLE;
		}
	}

	// Destroy material resources
	_skr_cmd_destroy_pipeline_layout      (NULL, _skr_pipeline_cache.materials[material_idx].layout);
	_skr_cmd_destroy_descriptor_set_layout(NULL, _skr_pipeline_cache.materials[material_idx].descriptor_layout);

	mtx_unlock(&_skr_pipeline_cache.mutex);
}

void _skr_pipeline_unregister_renderpass(int32_t renderpass_idx) {
	mtx_lock(&_skr_pipeline_cache.mutex);

	if (renderpass_idx < 0 || renderpass_idx >= _skr_pipeline_cache.renderpass_capacity) { mtx_unlock(&_skr_pipeline_cache.mutex); return; }
	if (_skr_pipeline_cache.renderpasses[renderpass_idx].ref_count <= 0)                 { mtx_unlock(&_skr_pipeline_cache.mutex); return; }

	_skr_pipeline_cache.renderpasses[renderpass_idx].ref_count--;
	if (_skr_pipeline_cache.renderpasses[renderpass_idx].ref_count > 0) { mtx_unlock(&_skr_pipeline_cache.mutex); return; }

	// Destroy all pipelines using this render pass
	for (int32_t m = 0; m < _skr_pipeline_cache.material_capacity; m++) {
		for (int32_t v = 0; v < _skr_pipeline_cache.vertformat_capacity; v++) {
			int32_t idx = _skr_pipeline_index_3d(m, renderpass_idx, v, _skr_pipeline_cache.renderpass_capacity, _skr_pipeline_cache.vertformat_capacity);
			_skr_cmd_destroy_pipeline(NULL, _skr_pipeline_cache.pipelines[idx]);
			_skr_pipeline_cache.pipelines[idx] = VK_NULL_HANDLE;
		}
	}

	// Only destroy VkRenderPass if no other slot shares the same handle
	VkRenderPass rp = _skr_pipeline_cache.renderpasses[renderpass_idx].render_pass;
	bool shared = false;
	for (int32_t i = 0; i < _skr_pipeline_cache.renderpass_capacity; i++) {
		if (i == renderpass_idx) continue;
		if (_skr_pipeline_cache.renderpasses[i].ref_count > 0 && _skr_pipeline_cache.renderpasses[i].render_pass == rp) {
			shared = true;
			break;
		}
	}
	if (!shared) {
		_skr_cmd_destroy_render_pass(NULL, rp);
	}
	_skr_pipeline_cache.renderpasses[renderpass_idx].render_pass = VK_NULL_HANDLE;

	mtx_unlock(&_skr_pipeline_cache.mutex);
}

static void _skr_pipeline_grow_vertformats(_skr_pipeline_cache_t* ref_cache, int32_t min_capacity) {
	if (min_capacity <= ref_cache->vertformat_capacity) return;

	int32_t old_capacity = ref_cache->vertformat_capacity;
	int32_t new_capacity = old_capacity == 0 ? 4 : old_capacity * 2;
	while (new_capacity < min_capacity) {
		new_capacity *= 2;
	}

	// Grow vertformats array
	ref_cache->vertformats = _skr_realloc(ref_cache->vertformats, new_capacity * sizeof(_skr_pipeline_vertformat_slot_t));
	memset(&ref_cache->vertformats[old_capacity], 0, (new_capacity - old_capacity) * sizeof(_skr_pipeline_vertformat_slot_t));

	// Grow pipelines 3D array
	_skr_pipeline_grow_pipelines_array(
		&ref_cache->pipelines,
		ref_cache->material_capacity, ref_cache->material_capacity,
		ref_cache->renderpass_capacity, ref_cache->renderpass_capacity,
		old_capacity, new_capacity
	);

	ref_cache->vertformat_capacity = new_capacity;
}

static bool _skr_vert_type_equals(const skr_vert_type_t* a, const skr_vert_type_t* b) {
	if (a->binding_count   != b->binding_count  ) return false;
	if (a->component_count != b->component_count) return false;

	// Compare bindings (deep comparison)
	if (a->binding_count > 0 && memcmp(a->bindings, b->bindings, sizeof(VkVertexInputBindingDescription) * a->binding_count) != 0)
		return false;

	// Compare attributes (deep comparison)
	if (a->component_count > 0 && memcmp(a->attributes, b->attributes, sizeof(VkVertexInputAttributeDescription) * a->component_count) != 0)
		return false;

	return true;
}

// Unlocked version - caller must hold the mutex via _skr_pipeline_lock()
int32_t _skr_pipeline_register_vertformat_unlocked(skr_vert_type_t vert_type) {
	// Find existing or free slot
	int32_t free_slot = -1;
	for (int32_t i = 0; i < _skr_pipeline_cache.vertformat_capacity; i++) {
		if (_skr_pipeline_cache.vertformats[i].ref_count > 0) {
			// Check if this vertex format already exists (deep comparison)
			if (_skr_vert_type_equals(&_skr_pipeline_cache.vertformats[i].vert_type, &vert_type)) {
				_skr_pipeline_cache.vertformats[i].ref_count++;
				return i;
			}
		} else if (free_slot == -1) {
			free_slot = i;
		}
	}

	// If no free slot, grow the array
	if (free_slot == -1) {
		free_slot = _skr_pipeline_cache.vertformat_capacity;
		_skr_pipeline_grow_vertformats(&_skr_pipeline_cache, free_slot + 1);
	}

	// Register new vertex format (just store copy)
	_skr_pipeline_cache.vertformats[free_slot].vert_type  = vert_type;
	_skr_pipeline_cache.vertformats[free_slot].ref_count  = 1;

	if (free_slot >= _skr_pipeline_cache.vertformat_count) {
		_skr_pipeline_cache.vertformat_count = free_slot + 1;
	}

	return free_slot;
}

int32_t _skr_pipeline_register_vertformat(skr_vert_type_t vert_type) {
	mtx_lock(&_skr_pipeline_cache.mutex);
	int32_t result = _skr_pipeline_register_vertformat_unlocked(vert_type);
	mtx_unlock(&_skr_pipeline_cache.mutex);
	return result;
}

void _skr_pipeline_unregister_vertformat(int32_t vertformat_idx) {
	mtx_lock(&_skr_pipeline_cache.mutex);

	if (vertformat_idx < 0 || vertformat_idx >= _skr_pipeline_cache.vertformat_capacity) { mtx_unlock(&_skr_pipeline_cache.mutex); return; }
	if (_skr_pipeline_cache.vertformats[vertformat_idx].ref_count <= 0)                  { mtx_unlock(&_skr_pipeline_cache.mutex); return; }

	_skr_pipeline_cache.vertformats[vertformat_idx].ref_count--;
	if (_skr_pipeline_cache.vertformats[vertformat_idx].ref_count > 0) { mtx_unlock(&_skr_pipeline_cache.mutex); return; }

	// Destroy all pipelines using this vertex format
	for (int32_t m = 0; m < _skr_pipeline_cache.material_capacity; m++) {
		for (int32_t r = 0; r < _skr_pipeline_cache.renderpass_capacity; r++) {
			int32_t idx = _skr_pipeline_index_3d(m, r, vertformat_idx, _skr_pipeline_cache.renderpass_capacity, _skr_pipeline_cache.vertformat_capacity);
			_skr_cmd_destroy_pipeline(NULL, _skr_pipeline_cache.pipelines[idx]);
			_skr_pipeline_cache.pipelines[idx] = VK_NULL_HANDLE;
		}
	}

	mtx_unlock(&_skr_pipeline_cache.mutex);
}

VkPipeline _skr_pipeline_get(int32_t material_idx, int32_t renderpass_idx, int32_t vertformat_idx) {
	if (material_idx   < 0 || material_idx   >= _skr_pipeline_cache.material_capacity)   return VK_NULL_HANDLE;
	if (renderpass_idx < 0 || renderpass_idx >= _skr_pipeline_cache.renderpass_capacity) return VK_NULL_HANDLE;
	if (vertformat_idx < 0 || vertformat_idx >= _skr_pipeline_cache.vertformat_capacity) return VK_NULL_HANDLE;
	if (_skr_pipeline_cache.materials   [material_idx  ].ref_count <= 0)                 return VK_NULL_HANDLE;
	if (_skr_pipeline_cache.renderpasses[renderpass_idx].ref_count <= 0)                 return VK_NULL_HANDLE;
	if (_skr_pipeline_cache.vertformats [vertformat_idx].ref_count <= 0)                 return VK_NULL_HANDLE;

	// Check if pipeline already exists
	int32_t idx = _skr_pipeline_index_3d(material_idx, renderpass_idx, vertformat_idx, _skr_pipeline_cache.renderpass_capacity, _skr_pipeline_cache.vertformat_capacity);
	if (_skr_pipeline_cache.pipelines[idx] != VK_NULL_HANDLE) {
		return _skr_pipeline_cache.pipelines[idx];
	}

	// Create pipeline
	VkPipeline pipeline = _skr_pipeline_create(material_idx, renderpass_idx, vertformat_idx);
	_skr_pipeline_cache.pipelines[idx] = pipeline;

	return pipeline;
}

VkPipelineLayout _skr_pipeline_get_layout(int32_t material_idx) {
	if (material_idx < 0 || material_idx >= _skr_pipeline_cache.material_capacity) return VK_NULL_HANDLE;
	if (_skr_pipeline_cache.materials[material_idx].ref_count <= 0)                return VK_NULL_HANDLE;

	return _skr_pipeline_cache.materials[material_idx].layout;
}

VkDescriptorSetLayout _skr_pipeline_get_descriptor_layout(int32_t material_idx) {
	if (material_idx < 0 || material_idx >= _skr_pipeline_cache.material_capacity) return VK_NULL_HANDLE;
	if (_skr_pipeline_cache.materials[material_idx].ref_count <= 0)                return VK_NULL_HANDLE;

	return _skr_pipeline_cache.materials[material_idx].descriptor_layout;
}

VkRenderPass _skr_pipeline_get_renderpass(int32_t renderpass_idx) {
	if (renderpass_idx < 0 || renderpass_idx >= _skr_pipeline_cache.renderpass_capacity) return VK_NULL_HANDLE;
	if (_skr_pipeline_cache.renderpasses[renderpass_idx].ref_count <= 0)                 return VK_NULL_HANDLE;

	return _skr_pipeline_cache.renderpasses[renderpass_idx].render_pass;
}

///////////////////////////////////////////////////////////////////////////////
// Internal helpers
///////////////////////////////////////////////////////////////////////////////

static VkRenderPass _skr_pipeline_create_multisubpass_renderpass(const skr_pipeline_renderpass_key_t* key) {
	VkAttachmentDescription attachments[SKR_POSTFX_MAX_ATTACHMENTS];
	uint32_t                attachment_count = 0;

	bool use_msaa  = key->samples > VK_SAMPLE_COUNT_1_BIT && key->resolve_format != VK_FORMAT_UNDEFINED;
	bool has_color = key->color_format != VK_FORMAT_UNDEFINED;
	bool has_depth = key->depth_format != VK_FORMAT_UNDEFINED;

	// --- Attachment indices (assigned as we go) ---
	int32_t color_idx   = -1;
	int32_t resolve_idx = -1;
	int32_t depth_idx   = -1;
	int32_t output_idx  = -1;
	int32_t intermediate_start = -1;

	// [0] Color attachment (MSAA or direct) — geometry subpass output
	if (has_color) {
		color_idx = (int32_t)attachment_count;
		attachments[attachment_count++] = (VkAttachmentDescription){
			.format         = key->color_format,
			.samples        = key->samples,
			.loadOp         = key->color_load_op,
			.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE, // postfx reads via input attachment, no need to store
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = (key->color_load_op == VK_ATTACHMENT_LOAD_OP_LOAD) ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
	}

	// [1] Resolve attachment (for MSAA) — resolved result, then input for postfx
	// When resolve-only (no postfx), this IS the final output and must be stored.
	bool resolve_is_final = use_msaa && key->has_resolve_subpass && key->postfx_count == 0;
	if (use_msaa) {
		resolve_idx = (int32_t)attachment_count;
		attachments[attachment_count++] = (VkAttachmentDescription){
			.format         = key->resolve_format,
			.samples        = VK_SAMPLE_COUNT_1_BIT,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp        = resolve_is_final ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
	}

	// [2] Depth attachment
	VkImageLayout depth_final = key->final_depth_layout ? key->final_depth_layout : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	if (has_depth) {
		bool has_stencil = _skr_format_has_stencil(key->depth_format);
		depth_idx = (int32_t)attachment_count;
		attachments[attachment_count++] = (VkAttachmentDescription){
			.format         = key->depth_format,
			.samples        = key->samples,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = key->depth_store_op,
			.stencilLoadOp  = has_stencil ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = has_stencil ? key->depth_store_op         : VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = key->depth_store_op == VK_ATTACHMENT_STORE_OP_STORE
				? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL  // Readable depth: explicitly transitioned before render pass
				: VK_IMAGE_LAYOUT_UNDEFINED,                        // Transient discard: loadOp=CLEAR handles it
			.finalLayout    = depth_final,
		};
	}

	// [3..] Intermediate transient color attachments for postfx chaining
	VkFormat intermediate_format = use_msaa ? key->resolve_format : key->color_format;
	if (key->postfx_output_format != VK_FORMAT_UNDEFINED)
		intermediate_format = key->postfx_output_format;

	uint32_t intermediate_count = key->postfx_count > 1 ? key->postfx_count - 1 : 0;
	if (intermediate_count > 0) {
		intermediate_start = (int32_t)attachment_count;
		for (uint32_t i = 0; i < intermediate_count; i++) {
			attachments[attachment_count++] = (VkAttachmentDescription){
				.format         = intermediate_format,
				.samples        = VK_SAMPLE_COUNT_1_BIT,
				.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE, // transient, consumed as input
				.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			};
		}
	}

	// [last] Final postfx output attachment (skip when resolve-only — resolve attachment is final)
	VkImageLayout final_output_layout = key->final_color_layout ? key->final_color_layout : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	if (!resolve_is_final) {
		VkFormat output_format = key->postfx_output_format != VK_FORMAT_UNDEFINED
			? key->postfx_output_format
			: (use_msaa ? key->resolve_format : key->color_format);
		output_idx = (int32_t)attachment_count;
		attachments[attachment_count++] = (VkAttachmentDescription){
			.format         = output_format,
			.samples        = VK_SAMPLE_COUNT_1_BIT,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp        = VK_ATTACHMENT_STORE_OP_STORE, // final output must be stored
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = final_output_layout,
		};
	} else {
		// resolve-only: the resolve attachment is the final output
		// Update its finalLayout to match the requested final layout
		if (resolve_idx >= 0 && key->final_resolve_layout) {
			attachments[resolve_idx].finalLayout = key->final_resolve_layout;
		}
	}

	// --- Build subpass descriptions ---
	uint32_t resolve_subpass_count = key->has_resolve_subpass ? 1 : 0;
	uint32_t subpass_count = 1 + resolve_subpass_count + key->postfx_count;

	// Attachment references (pre-allocate max needed)
	VkAttachmentReference color_refs  [SKR_POSTFX_MAX_SUBPASSES];
	VkAttachmentReference resolve_refs[SKR_POSTFX_MAX_SUBPASSES];
	VkAttachmentReference input_refs  [SKR_POSTFX_MAX_SUBPASSES];
	VkAttachmentReference depth_ref = { .attachment = VK_ATTACHMENT_UNUSED, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
	if (depth_idx >= 0) depth_ref.attachment = (uint32_t)depth_idx;

	VkSubpassDescription subpasses[SKR_POSTFX_MAX_SUBPASSES];
	memset(subpasses, 0, sizeof(subpasses));

	// Subpass 0: Geometry
	// When manual resolve is active, NO pResolveAttachments — the resolve subpass
	// reads the MSAA color directly as an input attachment instead.
	color_refs[0] = (VkAttachmentReference){
		.attachment = has_color ? (uint32_t)color_idx : VK_ATTACHMENT_UNUSED,
		.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};
	resolve_refs[0] = (VkAttachmentReference){
		.attachment = use_msaa ? (uint32_t)resolve_idx : VK_ATTACHMENT_UNUSED,
		.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};
	bool use_auto_resolve = use_msaa && !key->has_resolve_subpass;
	subpasses[0] = (VkSubpassDescription){
		.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount    = has_color ? 1 : 0,
		.pColorAttachments       = has_color ? &color_refs[0] : NULL,
		.pResolveAttachments     = use_auto_resolve ? &resolve_refs[0] : NULL,
		.pDepthStencilAttachment = has_depth ? &depth_ref : NULL,
	};

	// Subpass 1 (optional): Manual MSAA resolve
	// Reads MSAA color as multisampled input attachment, writes 1x resolved output.
	// No depth — frees tile memory after geometry subpass.
	uint32_t next_sp = 1;
	if (key->has_resolve_subpass) {
		input_refs[0] = (VkAttachmentReference){
			.attachment = (uint32_t)color_idx,
			.layout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
		color_refs[1] = (VkAttachmentReference){
			.attachment = (uint32_t)resolve_idx,
			.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		subpasses[1] = (VkSubpassDescription){
			.flags                   = key->use_custom_resolve_flags
				? (VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_QCOM | VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM)
				: 0,
			.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount    = 1,
			.pInputAttachments       = &input_refs[0],
			.colorAttachmentCount    = 1,
			.pColorAttachments       = &color_refs[1],
		};
		next_sp = 2;
	}

	// PostFX subpasses: read previous output as input attachment
	int32_t prev_color_attachment = (use_msaa || key->has_resolve_subpass) ? resolve_idx : color_idx;

	for (uint32_t p = 0; p < key->postfx_count; p++) {
		uint32_t sp      = next_sp + p;
		bool     is_last = (p == key->postfx_count - 1);
		uint32_t iref    = resolve_subpass_count + p; // offset past resolve input_ref

		// Input: previous color only. Depth is NOT included as input attachment
		// to avoid forcing the driver to keep MSAA depth in tile memory.
		input_refs[iref] = (VkAttachmentReference){
			.attachment = (uint32_t)prev_color_attachment,
			.layout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		// Output: last postfx writes to final output, others to intermediate
		int32_t this_output = is_last ? output_idx : intermediate_start + (int32_t)p;

		color_refs[sp] = (VkAttachmentReference){
			.attachment = (uint32_t)this_output,
			.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		subpasses[sp] = (VkSubpassDescription){
			.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount    = 1,
			.pInputAttachments       = &input_refs[iref],
			.colorAttachmentCount    = 1,
			.pColorAttachments       = &color_refs[sp],
		};

		prev_color_attachment = this_output;
	}

	// --- Subpass dependencies ---
	VkSubpassDependency dependencies[SKR_POSTFX_MAX_SUBPASSES + 4]; // +4: 2 external→0, N-1 chains, up to 2 subpass→external
	uint32_t dep_count = 0;

	// External → subpass 0: color
	dependencies[dep_count++] = (VkSubpassDependency){
		.srcSubpass    = VK_SUBPASS_EXTERNAL,
		.dstSubpass    = 0,
		.srcStageMask  = key->color_load_op == VK_ATTACHMENT_LOAD_OP_CLEAR
			? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
			: VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	};
	// External → subpass 0: depth
	if (has_depth) {
		dependencies[dep_count++] = (VkSubpassDependency){
			.srcSubpass    = VK_SUBPASS_EXTERNAL,
			.dstSubpass    = 0,
			.srcStageMask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		};
	}

	// Subpass N → N+1: tile-local dependency (covers resolve + postfx chain)
	for (uint32_t s = 0; s + 1 < subpass_count; s++) {
		dependencies[dep_count++] = (VkSubpassDependency){
			.srcSubpass      = s,
			.dstSubpass      = s + 1,
			.srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask   = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
		};
	}

	// Subpass → EXTERNAL: ensure writes complete before downstream shader reads
	// when finalLayout transitions to a readable layout (free on tilers).
	if (final_output_layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
	    (resolve_is_final && key->final_resolve_layout && key->final_resolve_layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)) {
		dependencies[dep_count++] = (VkSubpassDependency){
			.srcSubpass    = subpass_count - 1,
			.dstSubpass    = VK_SUBPASS_EXTERNAL,
			.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
	}
	if (has_depth && depth_final != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		dependencies[dep_count++] = (VkSubpassDependency){
			.srcSubpass    = 0,  // Depth is only used in geometry subpass
			.dstSubpass    = VK_SUBPASS_EXTERNAL,
			.srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
	}

	// --- Multiview ---
	uint32_t view_masks[SKR_POSTFX_MAX_SUBPASSES];
	for (uint32_t s = 0; s < subpass_count; s++)
		view_masks[s] = key->view_mask;

	VkRenderPassMultiviewCreateInfo multiview_info = {
		.sType                = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
		.subpassCount         = subpass_count,
		.pViewMasks           = view_masks,
		.correlationMaskCount = key->correlation_mask ? 1 : 0,
		.pCorrelationMasks    = key->correlation_mask ? &key->correlation_mask : NULL,
	};

	VkRenderPassCreateInfo render_pass_info = {
		.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext           = key->view_mask != 0 ? &multiview_info : NULL,
		.attachmentCount = attachment_count,
		.pAttachments    = attachments,
		.subpassCount    = subpass_count,
		.pSubpasses      = subpasses,
		.dependencyCount = dep_count,
		.pDependencies   = dependencies,
	};

	VkRenderPass render_pass;
	VkResult vr = vkCreateRenderPass(_skr_vk.device, &render_pass_info, NULL, &render_pass);
	SKR_VK_CHECK_RET(vr, "vkCreateRenderPass (postfx)", VK_NULL_HANDLE);

	char name[256];
	snprintf(name, sizeof(name), "rpass_%s%s%u_",
		key->has_resolve_subpass ? "resolve_" : "",
		key->use_custom_resolve_flags ? "cr_" : "",
		key->postfx_count);
	_skr_append_renderpass_config(name, sizeof(name), key);
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_RENDER_PASS, (uint64_t)render_pass, name);

	return render_pass;
}

static VkRenderPass _skr_pipeline_create_renderpass(const skr_pipeline_renderpass_key_t* key) {
	// Multi-subpass path for postfx and/or manual resolve
	if (key->postfx_count > 0 || key->has_resolve_subpass)
		return _skr_pipeline_create_multisubpass_renderpass(key);

	VkAttachmentDescription attachments[3];
	uint32_t attachment_count = 0;

	bool use_msaa  = key->samples > VK_SAMPLE_COUNT_1_BIT && key->resolve_format != VK_FORMAT_UNDEFINED;
	bool has_color = key->color_format != VK_FORMAT_UNDEFINED;

	// Color attachment (MSAA if samples > 1) - only if we have a color format
	VkAttachmentReference color_ref = {0};
	if (has_color) {
		// When loading previous contents, initialLayout must match the current layout
		VkImageLayout color_initial = (key->color_load_op == VK_ATTACHMENT_LOAD_OP_LOAD)
			? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
			: VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[attachment_count] = (VkAttachmentDescription){
			.format         = key->color_format,
			.samples        = key->samples,
			.loadOp         = key->color_load_op,
			.storeOp        = use_msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = color_initial,
			.finalLayout    = key->final_color_layout ? key->final_color_layout : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		color_ref.attachment = attachment_count;
		color_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachment_count++;
	}

	// Resolve attachment (for MSAA)
	VkAttachmentReference resolve_ref = {0};
	if (use_msaa) {
		attachments[attachment_count] = (VkAttachmentDescription){
			.format         = key->resolve_format,  // Use the actual resolve target format
			.samples        = VK_SAMPLE_COUNT_1_BIT,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = key->final_resolve_layout ? key->final_resolve_layout : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		resolve_ref.attachment = attachment_count;
		resolve_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachment_count++;
	}

	// Depth attachment (if present)
	VkAttachmentReference depth_ref = {0};
	VkImageLayout depth_final = key->final_depth_layout ? key->final_depth_layout : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	if (key->depth_format != VK_FORMAT_UNDEFINED) {
		// Check if this is a depth+stencil format
		bool has_stencil = _skr_format_has_stencil(key->depth_format);

		attachments[attachment_count] = (VkAttachmentDescription){
			.format         = key->depth_format,
			.samples        = key->samples,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp        = key->depth_store_op,  // Use the store op from the key (based on readable flag)
			.stencilLoadOp  = has_stencil ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = has_stencil ? key->depth_store_op         : VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = key->depth_store_op == VK_ATTACHMENT_STORE_OP_STORE
				? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL  // Readable depth: explicitly transitioned before render pass
				: VK_IMAGE_LAYOUT_UNDEFINED,                        // Transient discard: never explicitly transitioned, loadOp=CLEAR handles it
			.finalLayout    = depth_final,
		};

		depth_ref.attachment = attachment_count;
		depth_ref.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachment_count++;
	}

	// Subpass
	VkSubpassDescription subpass = {
		.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount    = has_color ? 1 : 0,
		.pColorAttachments       = has_color ? &color_ref : NULL,
		.pResolveAttachments     = use_msaa ? &resolve_ref : NULL,
		.pDepthStencilAttachment = key->depth_format != VK_FORMAT_UNDEFINED ? &depth_ref : NULL,
	};

	// Subpass dependencies
	// Use TOP_OF_PIPE for clears to avoid unnecessary execution dependency on
	// prior fragment/color work. loadOp=CLEAR discards previous contents so
	// there's no data hazard with earlier passes.
	VkSubpassDependency dependencies[4];
	uint32_t dep_count = 0;

	dependencies[dep_count++] = (VkSubpassDependency){
		.srcSubpass    = VK_SUBPASS_EXTERNAL,
		.dstSubpass    = 0,
		.srcStageMask  = key->color_load_op == VK_ATTACHMENT_LOAD_OP_CLEAR
			? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
			: VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	};
	dependencies[dep_count++] = (VkSubpassDependency){
		.srcSubpass    = VK_SUBPASS_EXTERNAL,
		.dstSubpass    = 0,
		.srcStageMask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // Depth always clears, no prior dependency needed
		.dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
	};

	// When finalLayout differs from the in-pass layout, the render pass performs
	// an implicit transition. Add subpass→EXTERNAL dependency to ensure writes
	// complete before downstream shader reads. This is free on tile-based GPUs.
	VkImageLayout color_final   = key->final_color_layout   ? key->final_color_layout   : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	VkImageLayout resolve_final = key->final_resolve_layout  ? key->final_resolve_layout  : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	if ((has_color && color_final != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) ||
	    (use_msaa  && resolve_final != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)) {
		dependencies[dep_count++] = (VkSubpassDependency){
			.srcSubpass    = 0,
			.dstSubpass    = VK_SUBPASS_EXTERNAL,
			.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
	}
	if (key->depth_format != VK_FORMAT_UNDEFINED && depth_final != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		dependencies[dep_count++] = (VkSubpassDependency){
			.srcSubpass    = 0,
			.dstSubpass    = VK_SUBPASS_EXTERNAL,
			.srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
	}

	// Chain multiview info when view_mask is set
	VkRenderPassMultiviewCreateInfo multiview_info = {
		.sType                = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
		.subpassCount         = 1,
		.pViewMasks           = &key->view_mask,
		.correlationMaskCount = key->correlation_mask ? 1 : 0,
		.pCorrelationMasks    = key->correlation_mask ? &key->correlation_mask : NULL,
	};

	VkRenderPassCreateInfo render_pass_info = {
		.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext           = key->view_mask != 0 ? &multiview_info : NULL,
		.attachmentCount = attachment_count,
		.pAttachments    = attachments,
		.subpassCount    = 1,
		.pSubpasses      = &subpass,
		.dependencyCount = dep_count,
		.pDependencies   = dependencies,
	};

	VkRenderPass render_pass;
	VkResult vr = vkCreateRenderPass(_skr_vk.device, &render_pass_info, NULL, &render_pass);
	SKR_VK_CHECK_RET(vr, "vkCreateRenderPass", VK_NULL_HANDLE);

	// Generate debug name based on render pass configuration
	char name[256];
	snprintf(name, sizeof(name), "rpass_");
	_skr_append_renderpass_config(name, sizeof(name), key);
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_RENDER_PASS, (uint64_t)render_pass, name);

	return render_pass;
}

static VkPipelineLayout _skr_pipeline_create_layout(VkDescriptorSetLayout descriptor_layout) {
	VkPipelineLayoutCreateInfo layout_info = {
		.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = descriptor_layout != VK_NULL_HANDLE ? 1 : 0,
		.pSetLayouts    = descriptor_layout != VK_NULL_HANDLE ? &descriptor_layout : NULL,
	};

	VkPipelineLayout layout;
	VkResult vr = vkCreatePipelineLayout(_skr_vk.device, &layout_info, NULL, &layout);
	SKR_VK_CHECK_RET(vr, "vkCreatePipelineLayout", VK_NULL_HANDLE);

	// Pipeline layouts are created per-material, name will be set during material registration
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)layout, "pipeline_layout");

	return layout;
}

static VkPipeline _skr_pipeline_create(int32_t material_idx, int32_t renderpass_idx, int32_t vertformat_idx) {
	const _skr_pipeline_material_key_t*  mat_key   = &_skr_pipeline_cache.materials   [material_idx  ].key;
	const skr_pipeline_renderpass_key_t* rp_key    = &_skr_pipeline_cache.renderpasses[renderpass_idx].key;
	const skr_vert_type_t*               vert_type = &_skr_pipeline_cache.vertformats [vertformat_idx].vert_type;
	const VkPipelineLayout               layout    =  _skr_pipeline_cache.materials   [material_idx  ].layout;
	const VkRenderPass                   rp        =  _skr_pipeline_cache.renderpasses[renderpass_idx].render_pass;

	// One specialization info is shared by both stages; Vulkan ignores map
	// entries whose constantID isn't present in a stage's module.
	const sksc_shader_meta_t* meta       = &mat_key->shader->meta;
	uint32_t                  spec_count = meta->spec_constant_count < SKR_MAX_SPEC_CONSTANTS ? meta->spec_constant_count : SKR_MAX_SPEC_CONSTANTS;
	VkSpecializationMapEntry  spec_entries[SKR_MAX_SPEC_CONSTANTS];
	VkSpecializationInfo      spec_info;
	const VkSpecializationInfo* spec = NULL;
	if (spec_count > 0) {
		for (uint32_t i = 0; i < spec_count; i++) {
			spec_entries[i] = (VkSpecializationMapEntry){
				.constantID = meta->spec_constants[i].constant_id,
				.offset     = i * (uint32_t)sizeof(uint32_t),
				.size       = sizeof(uint32_t),
			};
		}
		spec_info = (VkSpecializationInfo){
			.mapEntryCount = spec_count,
			.pMapEntries   = spec_entries,
			.dataSize      = spec_count * sizeof(uint32_t),
			.pData         = mat_key->spec_constant_values,
		};
		spec = &spec_info;
	}

	// Shader stages
	VkPipelineShaderStageCreateInfo shader_stages[2];
	uint32_t stage_count = 0;

	if (mat_key->shader->vertex_stage.shader != VK_NULL_HANDLE) {
		shader_stages[stage_count++] = (VkPipelineShaderStageCreateInfo){
			.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage               = VK_SHADER_STAGE_VERTEX_BIT,
			.module              = mat_key->shader->vertex_stage.shader,
			.pName               = "vs",
			.pSpecializationInfo = spec,
		};
	}

	if (mat_key->shader->pixel_stage.shader != VK_NULL_HANDLE) {
		shader_stages[stage_count++] = (VkPipelineShaderStageCreateInfo){
			.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage               = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module              = mat_key->shader->pixel_stage.shader,
			.pName               = "ps",
			.pSpecializationInfo = spec,
		};
	}

	// Vertex input - baked from vertex type
	VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount   = vert_type ? vert_type->binding_count : 0,
		.pVertexBindingDescriptions      = vert_type ? vert_type->bindings : NULL,
		.vertexAttributeDescriptionCount = vert_type ? vert_type->component_count : 0,
		.pVertexAttributeDescriptions    = vert_type ? vert_type->attributes : NULL,
	};

	// Input assembly
	VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	// Viewport state (dynamic)
	VkPipelineViewportStateCreateInfo viewport_state = {
		.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount  = 1,
	};

	// Rasterization
	VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable        = mat_key->depth_clamp && _skr_vk.has_depth_clamp ? VK_TRUE : VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode             = mat_key->wireframe && _skr_vk.has_fill_mode_non_solid ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL,
		.cullMode                = _skr_to_vk_cull(mat_key->cull),
		.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable         = VK_FALSE,
		.lineWidth               = 1.0f,
	};

	// Multisampling — all non-geometry subpasses (resolve + postfx) rasterize at 1x
	VkSampleCountFlagBits pipeline_samples = (rp_key->subpass_index > 0)
		? VK_SAMPLE_COUNT_1_BIT : rp_key->samples;
	VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples  = pipeline_samples,
		.sampleShadingEnable   = VK_FALSE,
		.alphaToCoverageEnable = mat_key->alpha_to_coverage ? VK_TRUE : VK_FALSE,
	};

	// Depth/stencil
	bool stencil_enabled = (mat_key->write_mask & skr_write_stencil) ||
	                       mat_key->stencil_front.compare != skr_compare_none ||
	                       mat_key->stencil_back.compare != skr_compare_none;

	VkStencilOpState front_stencil = {
		.failOp      = _skr_to_vk_stencil_op(mat_key->stencil_front.fail_op),
		.passOp      = _skr_to_vk_stencil_op(mat_key->stencil_front.pass_op),
		.depthFailOp = _skr_to_vk_stencil_op(mat_key->stencil_front.depth_fail_op),
		.compareOp   = _skr_to_vk_compare(mat_key->stencil_front.compare),
		.compareMask = mat_key->stencil_front.compare_mask,
		.writeMask   = mat_key->stencil_front.write_mask,
		.reference   = mat_key->stencil_front.reference,
	};

	VkStencilOpState back_stencil = {
		.failOp      = _skr_to_vk_stencil_op(mat_key->stencil_back.fail_op),
		.passOp      = _skr_to_vk_stencil_op(mat_key->stencil_back.pass_op),
		.depthFailOp = _skr_to_vk_stencil_op(mat_key->stencil_back.depth_fail_op),
		.compareOp   = _skr_to_vk_compare(mat_key->stencil_back.compare),
		.compareMask = mat_key->stencil_back.compare_mask,
		.writeMask   = mat_key->stencil_back.write_mask,
		.reference   = mat_key->stencil_back.reference,
	};

	VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable       = mat_key->depth_test != skr_compare_none ? VK_TRUE : VK_FALSE,
		.depthWriteEnable      = (mat_key->write_mask & skr_write_depth) ? VK_TRUE : VK_FALSE,
		.depthCompareOp        = _skr_to_vk_compare(mat_key->depth_test),
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable     = stencil_enabled ? VK_TRUE : VK_FALSE,
		.front                 = front_stencil,
		.back                  = back_stencil,
	};

	// Color blending - check if blend is enabled by seeing if any factors are non-zero
	// Zero-initialized blend state means "no blending" - pass source through unchanged
	bool blend_enabled = (mat_key->blend_state.src_color_factor != skr_blend_zero ||
	                      mat_key->blend_state.dst_color_factor != skr_blend_zero ||
	                      mat_key->blend_state.src_alpha_factor != skr_blend_zero ||
	                      mat_key->blend_state.dst_alpha_factor != skr_blend_zero);

	// When blend is disabled, always use ONE for src and ZERO for dst (pass through source)
	VkBlendFactor src_color = blend_enabled ? _skr_to_vk_blend_factor(mat_key->blend_state.src_color_factor) : VK_BLEND_FACTOR_ONE;
	VkBlendFactor dst_color = blend_enabled ? _skr_to_vk_blend_factor(mat_key->blend_state.dst_color_factor) : VK_BLEND_FACTOR_ZERO;
	VkBlendFactor src_alpha = blend_enabled ? _skr_to_vk_blend_factor(mat_key->blend_state.src_alpha_factor) : VK_BLEND_FACTOR_ONE;
	VkBlendFactor dst_alpha = blend_enabled ? _skr_to_vk_blend_factor(mat_key->blend_state.dst_alpha_factor) : VK_BLEND_FACTOR_ZERO;

	VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable         = blend_enabled ? VK_TRUE : VK_FALSE,
		.srcColorBlendFactor = src_color,
		.dstColorBlendFactor = dst_color,
		.colorBlendOp        = _skr_to_vk_blend_op(mat_key->blend_state.color_op),
		.srcAlphaBlendFactor = src_alpha,
		.dstAlphaBlendFactor = dst_alpha,
		.alphaBlendOp        = _skr_to_vk_blend_op(mat_key->blend_state.alpha_op),
		.colorWriteMask      =
			((mat_key->write_mask & skr_write_r) ? VK_COLOR_COMPONENT_R_BIT : 0) |
			((mat_key->write_mask & skr_write_g) ? VK_COLOR_COMPONENT_G_BIT : 0) |
			((mat_key->write_mask & skr_write_b) ? VK_COLOR_COMPONENT_B_BIT : 0) |
			((mat_key->write_mask & skr_write_a) ? VK_COLOR_COMPONENT_A_BIT : 0),
	};

	VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable   = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments    = &color_blend_attachment,
	};

	// Dynamic state
	VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates    = dynamic_states,
	};

	// Create pipeline
	VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount          = stage_count,
		.pStages             = shader_stages,
		.pVertexInputState   = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState      = &viewport_state,
		.pRasterizationState = &rasterizer,
		.pMultisampleState   = &multisampling,
		.pDepthStencilState  = &depth_stencil,
		.pColorBlendState    = &color_blending,
		.pDynamicState       = &dynamic_state,
		.layout              = layout,
		.renderPass          = rp,
		.subpass             = rp_key->subpass_index,
	};

	// Build debug name before creation so it's available for error logging
	char name[256];
	const char* shader_name = mat_key->shader->meta.name[0]
		? mat_key->shader->meta.name
		: "shader";

	snprintf(name, sizeof(name), "pipeline_%s_(", shader_name);
	_skr_append_material_config(name, sizeof(name), mat_key);
	strcat(name, ")_(");
	_skr_append_renderpass_config(name, sizeof(name), rp_key);
	strcat(name, ")_(");
	_skr_append_vertex_format    (name, sizeof(name), vert_type->components, vert_type->component_count);
	strcat(name, ")");

	VkPipeline pipeline;
	VkResult   result = vkCreateGraphicsPipelines(_skr_vk.device, _skr_vk.pipeline_cache, 1, &pipeline_info, NULL, &pipeline);
	if (result != VK_SUCCESS) {
		skr_log(skr_log_critical, "Failed to create graphics pipeline: %s (VkResult %d, stages %u, view_mask 0x%x)", name, result, stage_count, rp_key->view_mask);
		return VK_NULL_HANDLE;
	}

	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_PIPELINE, (uint64_t)pipeline, name);

	return pipeline;
}

///////////////////////////////////////////////////////////////////////////////
// Framebuffer creation
///////////////////////////////////////////////////////////////////////////////

VkFramebuffer _skr_create_framebuffer(VkDevice device, VkRenderPass render_pass, skr_tex_t* color, skr_tex_t* depth, skr_tex_t* opt_resolve) {
	VkImageView attachments[3];
	uint32_t    attachment_count = 0;
	uint32_t    width            = 1;
	uint32_t    height           = 1;
	uint32_t    layers           = 1;

	if (color) {
		attachments[attachment_count++] = color->view;
		width                           = color->size.x;
		height                          = color->size.y;
		// Multiview renderpasses require layers=1; the view_mask controls
		// which array layers are rendered to. The image view already covers
		// all layers, so we always use layers=1 here.
	}

	// Resolve attachment comes after color but before depth
	if (opt_resolve && color && color->samples > VK_SAMPLE_COUNT_1_BIT) {
		attachments[attachment_count++] = opt_resolve->view;
	}

	if (depth) {
		attachments[attachment_count++] = depth->view;
		if (width == 1 && height == 1) {
			width  = depth->size.x;
			height = depth->size.y;
		}
		// Multiview handles array layers via view_mask, so layers stays 1
	}

	VkFramebufferCreateInfo framebuffer_info = {
		.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass      = render_pass,
		.attachmentCount = attachment_count,
		.pAttachments    = attachments,
		.width           = width,
		.height          = height,
		.layers          = layers,
	};

	VkFramebuffer framebuffer;
	VkResult vr = vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer);
	SKR_VK_CHECK_RET(vr, "vkCreateFramebuffer", VK_NULL_HANDLE);

	return framebuffer;
}
