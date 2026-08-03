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

// A VkRenderPass and the key that built it. subpass_index is not part of a
// VkRenderPass's identity, so object keys always have subpass_index == 0.
typedef struct {
	skr_pipeline_renderpass_key_t    key;
	VkRenderPass                     render_pass;
	int32_t                          ref_count;  // Number of slots pointing here
} _skr_pipeline_renderpass_object_t;

// A cache slot is (renderpass identity + subpass_index), and its index is the
// renderpass axis of the pipelines 3D array.
typedef struct {
	skr_pipeline_renderpass_key_t    key;         // Full key, including subpass_index
	int32_t                          object_idx;  // Index into renderpass_objects
	int32_t                          ref_count;
} _skr_pipeline_renderpass_slot_t;

typedef struct {
	skr_vert_type_t                  vert_type;
	int32_t                          ref_count;
} _skr_pipeline_vertformat_slot_t;

typedef struct {
	_skr_pipeline_material_slot_t*     materials;
	_skr_pipeline_renderpass_slot_t*   renderpasses;
	_skr_pipeline_renderpass_object_t* renderpass_objects;
	_skr_pipeline_vertformat_slot_t*   vertformats;
	VkPipeline*                        pipelines;       // 3D array: [material][renderpass][vertformat]
	int32_t                            material_capacity;
	int32_t                            renderpass_capacity;
	int32_t                            renderpass_object_capacity;
	int32_t                            vertformat_capacity;
	mtx_t                              mutex;           // Thread safety for cache access
} _skr_pipeline_cache_t;

///////////////////////////////////////////////////////////////////////////////
// State
///////////////////////////////////////////////////////////////////////////////

static _skr_pipeline_cache_t _skr_pipeline_cache = {0};

// Stored in the pipelines[] cache to mark a slot whose creation permanently
// failed (incompatible mesh/shader vertex formats), distinct from
// VK_NULL_HANDLE which means "not created yet". Without it a failed slot stays
// VK_NULL_HANDLE and _skr_pipeline_get re-runs the whole remap — and re-logs
// the error — on every affected draw, every frame. 0xFFFF… is never a real
// Vulkan handle. It never escapes the cache: _skr_pipeline_get maps it back to
// VK_NULL_HANDLE for callers, and the destroy paths skip it.
#define SKR_PIPELINE_FAILED ((VkPipeline)(uintptr_t)-1)

// A cache slot holds a real, destroyable pipeline only when it is neither the
// "not created" nor the "creation failed" sentinel.
static inline bool _skr_pipeline_is_real(VkPipeline p) {
	return p != VK_NULL_HANDLE && p != SKR_PIPELINE_FAILED;
}

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
					if (_skr_pipeline_is_real(_skr_pipeline_cache.pipelines[idx])) {
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

	// Destroy render passes, each handle lives in exactly one object entry
	if (_skr_pipeline_cache.renderpass_objects) {
		for (int32_t r = 0; r < _skr_pipeline_cache.renderpass_object_capacity; r++) {
			VkRenderPass rp = _skr_pipeline_cache.renderpass_objects[r].render_pass;
			if (_skr_pipeline_cache.renderpass_objects[r].ref_count > 0 && rp != VK_NULL_HANDLE)
				vkDestroyRenderPass(_skr_vk.device, rp, NULL);
		}
		_skr_free(_skr_pipeline_cache.renderpass_objects);
	}
	if (_skr_pipeline_cache.renderpasses) {
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

// Object indices are not a pipeline array axis, unlike slot indices, so this
// grows independently of the pipelines 3D array.
static void _skr_pipeline_grow_renderpass_objects(_skr_pipeline_cache_t* ref_cache, int32_t min_capacity) {
	if (min_capacity <= ref_cache->renderpass_object_capacity) return;

	int32_t old_capacity = ref_cache->renderpass_object_capacity;
	int32_t new_capacity = old_capacity == 0 ? 4 : old_capacity * 2;
	while (new_capacity < min_capacity) {
		new_capacity *= 2;
	}

	ref_cache->renderpass_objects = _skr_realloc(ref_cache->renderpass_objects, new_capacity * sizeof(_skr_pipeline_renderpass_object_t));
	memset(&ref_cache->renderpass_objects[old_capacity], 0, (new_capacity - old_capacity) * sizeof(_skr_pipeline_renderpass_object_t));

	ref_cache->renderpass_object_capacity = new_capacity;
}

// Unlocked version - caller must hold the mutex via _skr_pipeline_lock()
int32_t _skr_pipeline_register_renderpass_unlocked(const skr_pipeline_renderpass_key_t* key) {
	int32_t free_slot = -1;
	for (int32_t i = 0; i < _skr_pipeline_cache.renderpass_capacity; i++) {
		if (_skr_pipeline_cache.renderpasses[i].ref_count > 0) {
			if (memcmp(&_skr_pipeline_cache.renderpasses[i].key, key, sizeof(skr_pipeline_renderpass_key_t)) == 0) {
				_skr_pipeline_cache.renderpasses[i].ref_count++;
				return i;
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

	// Find or create the VkRenderPass object shared by every subpass_index
	skr_pipeline_renderpass_key_t obj_key = *key;
	obj_key.subpass_index = 0;
	int32_t object_idx = -1;
	int32_t free_obj   = -1;
	for (int32_t i = 0; i < _skr_pipeline_cache.renderpass_object_capacity; i++) {
		if (_skr_pipeline_cache.renderpass_objects[i].ref_count > 0) {
			if (memcmp(&_skr_pipeline_cache.renderpass_objects[i].key, &obj_key, sizeof(skr_pipeline_renderpass_key_t)) == 0) {
				object_idx = i;
				break;
			}
		} else if (free_obj == -1) {
			free_obj = i;
		}
	}
	if (object_idx >= 0) {
		_skr_pipeline_cache.renderpass_objects[object_idx].ref_count++;
	} else {
		if (free_obj == -1) {
			free_obj = _skr_pipeline_cache.renderpass_object_capacity;
			_skr_pipeline_grow_renderpass_objects(&_skr_pipeline_cache, free_obj + 1);
		}
		_skr_pipeline_cache.renderpass_objects[free_obj].key         = obj_key;
		_skr_pipeline_cache.renderpass_objects[free_obj].render_pass = _skr_pipeline_create_renderpass(&obj_key);
		_skr_pipeline_cache.renderpass_objects[free_obj].ref_count   = 1;
		object_idx = free_obj;
	}

	_skr_pipeline_cache.renderpasses[free_slot].key        = *key;
	_skr_pipeline_cache.renderpasses[free_slot].object_idx = object_idx;
	_skr_pipeline_cache.renderpasses[free_slot].ref_count  = 1;

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
			if (_skr_pipeline_is_real(_skr_pipeline_cache.pipelines[idx]))
				_skr_cmd_destroy_pipeline(NULL, _skr_pipeline_cache.pipelines[idx]);
			_skr_pipeline_cache.pipelines[idx] = VK_NULL_HANDLE;
		}
	}

	// Destroy material resources
	_skr_cmd_destroy_pipeline_layout      (NULL, _skr_pipeline_cache.materials[material_idx].layout);
	_skr_cmd_destroy_descriptor_set_layout(NULL, _skr_pipeline_cache.materials[material_idx].descriptor_layout);
	_skr_pipeline_cache.materials[material_idx].layout            = VK_NULL_HANDLE;
	_skr_pipeline_cache.materials[material_idx].descriptor_layout = VK_NULL_HANDLE;

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
			if (_skr_pipeline_is_real(_skr_pipeline_cache.pipelines[idx]))
				_skr_cmd_destroy_pipeline(NULL, _skr_pipeline_cache.pipelines[idx]);
			_skr_pipeline_cache.pipelines[idx] = VK_NULL_HANDLE;
		}
	}

	// Release the slot's object; the last slot out destroys the VkRenderPass
	int32_t object_idx = _skr_pipeline_cache.renderpasses[renderpass_idx].object_idx;
	if (object_idx >= 0 && object_idx < _skr_pipeline_cache.renderpass_object_capacity) {
		_skr_pipeline_renderpass_object_t* obj = &_skr_pipeline_cache.renderpass_objects[object_idx];
		obj->ref_count--;
		if (obj->ref_count <= 0) {
			if (obj->render_pass != VK_NULL_HANDLE)
				_skr_cmd_destroy_render_pass(NULL, obj->render_pass);
			obj->render_pass = VK_NULL_HANDLE;
		}
	}
	_skr_pipeline_cache.renderpasses[renderpass_idx].object_idx = -1;

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

	// Compare the semantic-carrying fields the baked attributes don't capture.
	// The pipeline remap (_skr_pipeline_create) now wires attributes to shader
	// inputs by semantic + slot, reading them from the *cached* vert type. Two
	// vert types with identical baked attributes but a different semantic ->
	// offset mapping (e.g. position/normal vs normal/position, same formats)
	// must not alias to one cache slot, or a mesh would silently inherit the
	// other's wiring. (location rides in the struct too; compare it for
	// completeness — a field-by-field loop also sidesteps struct padding.)
	for (uint32_t i = 0; i < a->component_count; i++) {
		if (a->components[i].semantic      != b->components[i].semantic      ||
		    a->components[i].semantic_slot != b->components[i].semantic_slot ||
		    a->components[i].location      != b->components[i].location)
			return false;
	}

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
			if (_skr_pipeline_is_real(_skr_pipeline_cache.pipelines[idx]))
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

	// Check the cache slot. A prior creation that failed is remembered as
	// SKR_PIPELINE_FAILED so we neither retry it nor re-log its error every
	// frame; report it to callers as VK_NULL_HANDLE (their "skip this draw"
	// signal), never the raw sentinel.
	int32_t    idx    = _skr_pipeline_index_3d(material_idx, renderpass_idx, vertformat_idx, _skr_pipeline_cache.renderpass_capacity, _skr_pipeline_cache.vertformat_capacity);
	VkPipeline cached = _skr_pipeline_cache.pipelines[idx];
	if (cached == SKR_PIPELINE_FAILED) return VK_NULL_HANDLE;
	if (cached != VK_NULL_HANDLE)      return cached;

	// First request for this slot — create, and remember a failure so the
	// descriptive error (logged inside _skr_pipeline_create) fires just once.
	VkPipeline pipeline = _skr_pipeline_create(material_idx, renderpass_idx, vertformat_idx);
	_skr_pipeline_cache.pipelines[idx] = (pipeline == VK_NULL_HANDLE) ? SKR_PIPELINE_FAILED : pipeline;

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

	int32_t object_idx = _skr_pipeline_cache.renderpasses[renderpass_idx].object_idx;
	if (object_idx < 0 || object_idx >= _skr_pipeline_cache.renderpass_object_capacity)  return VK_NULL_HANDLE;

	return _skr_pipeline_cache.renderpass_objects[object_idx].render_pass;
}

///////////////////////////////////////////////////////////////////////////////
// Internal helpers
///////////////////////////////////////////////////////////////////////////////

// Down-converts a VkRenderPassCreateInfo2 to a core-1.0 vkCreateRenderPass
// call for devices without VK_KHR_create_renderpass2. Depth-feature keys
// never get here — per-reference aspect masks and depth-stencil resolve
// chains would be silently dropped by this conversion.
static VkRenderPass _skr_renderpass_create_compat(const VkRenderPassCreateInfo2* info, uint32_t view_mask) {
	VkAttachmentDescription attachments [SKR_POSTFX_MAX_ATTACHMENTS];
	VkSubpassDescription    subpasses   [SKR_POSTFX_MAX_SUBPASSES];
	VkSubpassDependency     dependencies[SKR_POSTFX_MAX_SUBPASSES + 10];
	VkAttachmentReference   refs        [SKR_POSTFX_MAX_SUBPASSES * 5]; // ≤2 input + color + resolve + depth per subpass
	uint32_t                ref_count = 0;

	for (uint32_t i = 0; i < info->attachmentCount; i++) {
		const VkAttachmentDescription2* a = &info->pAttachments[i];
		attachments[i] = (VkAttachmentDescription){
			.flags = a->flags,          .format         = a->format,
			.samples = a->samples,      .loadOp         = a->loadOp,
			.storeOp = a->storeOp,      .stencilLoadOp  = a->stencilLoadOp,
			.stencilStoreOp = a->stencilStoreOp,
			.initialLayout  = a->initialLayout,
			.finalLayout    = a->finalLayout,
		};
	}

	for (uint32_t s = 0; s < info->subpassCount; s++) {
		const VkSubpassDescription2* sp = &info->pSubpasses[s];

		const VkAttachmentReference* inputs = &refs[ref_count];
		for (uint32_t i = 0; i < sp->inputAttachmentCount; i++)
			refs[ref_count++] = (VkAttachmentReference){ sp->pInputAttachments[i].attachment, sp->pInputAttachments[i].layout };
		const VkAttachmentReference* colors = &refs[ref_count];
		for (uint32_t i = 0; i < sp->colorAttachmentCount; i++)
			refs[ref_count++] = (VkAttachmentReference){ sp->pColorAttachments[i].attachment, sp->pColorAttachments[i].layout };
		const VkAttachmentReference* resolves = NULL;
		if (sp->pResolveAttachments) {
			resolves = &refs[ref_count];
			for (uint32_t i = 0; i < sp->colorAttachmentCount; i++)
				refs[ref_count++] = (VkAttachmentReference){ sp->pResolveAttachments[i].attachment, sp->pResolveAttachments[i].layout };
		}
		const VkAttachmentReference* depth = NULL;
		if (sp->pDepthStencilAttachment) {
			depth = &refs[ref_count];
			refs[ref_count++] = (VkAttachmentReference){ sp->pDepthStencilAttachment->attachment, sp->pDepthStencilAttachment->layout };
		}

		subpasses[s] = (VkSubpassDescription){
			.flags                   = sp->flags,
			.pipelineBindPoint       = sp->pipelineBindPoint,
			.inputAttachmentCount    = sp->inputAttachmentCount,
			.pInputAttachments       = sp->inputAttachmentCount ? inputs : NULL,
			.colorAttachmentCount    = sp->colorAttachmentCount,
			.pColorAttachments       = sp->colorAttachmentCount ? colors : NULL,
			.pResolveAttachments     = resolves,
			.pDepthStencilAttachment = depth,
		};
	}

	for (uint32_t d = 0; d < info->dependencyCount; d++) {
		const VkSubpassDependency2* dep = &info->pDependencies[d];
		dependencies[d] = (VkSubpassDependency){
			.srcSubpass      = dep->srcSubpass,    .dstSubpass      = dep->dstSubpass,
			.srcStageMask    = dep->srcStageMask,  .dstStageMask    = dep->dstStageMask,
			.srcAccessMask   = dep->srcAccessMask, .dstAccessMask   = dep->dstAccessMask,
			.dependencyFlags = dep->dependencyFlags,
		};
	}

	uint32_t view_masks[SKR_POSTFX_MAX_SUBPASSES];
	for (uint32_t s = 0; s < info->subpassCount; s++)
		view_masks[s] = view_mask;

	VkRenderPassMultiviewCreateInfo multiview_info = {
		.sType                = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
		.subpassCount         = info->subpassCount,
		.pViewMasks           = view_masks,
		.correlationMaskCount = info->correlatedViewMaskCount,
		.pCorrelationMasks    = info->pCorrelatedViewMasks,
	};

	VkRenderPassCreateInfo render_pass_info = {
		.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext           = view_mask != 0 ? &multiview_info : NULL,
		.attachmentCount = info->attachmentCount,
		.pAttachments    = attachments,
		.subpassCount    = info->subpassCount,
		.pSubpasses      = subpasses,
		.dependencyCount = info->dependencyCount,
		.pDependencies   = dependencies,
	};

	VkRenderPass render_pass;
	VkResult vr = vkCreateRenderPass(_skr_vk.device, &render_pass_info, NULL, &render_pass);
	SKR_VK_CHECK_RET(vr, "vkCreateRenderPass (postfx compat)", VK_NULL_HANDLE);
	return render_pass;
}

// Built with renderpass2 structs so postfx depth input attachments get
// per-reference aspect masks and the geometry subpass can chain a
// VkSubpassDescriptionDepthStencilResolve. Devices without
// VK_KHR_create_renderpass2 go through _skr_renderpass_create_compat above,
// which only ever sees keys without depth features (callers gate on the
// extension flags).
static VkRenderPass _skr_pipeline_create_multisubpass_renderpass(const skr_pipeline_renderpass_key_t* key) {
	VkAttachmentDescription2 attachments[SKR_POSTFX_MAX_ATTACHMENTS];
	uint32_t                 attachment_count = 0;

	bool use_msaa      = key->samples > VK_SAMPLE_COUNT_1_BIT && key->resolve_format != VK_FORMAT_UNDEFINED;
	bool has_color     = key->color_format != VK_FORMAT_UNDEFINED;
	bool has_depth     = key->depth_format != VK_FORMAT_UNDEFINED;
	bool reads_depth   = (key->flags & skr_rp_flag_postfx_reads_depth) && has_depth;
	// MS-declared postfx depth reads the MSAA attachment directly, so no
	// resolve attachment and no 1x transient - the depth input reference below
	// falls through to depth_idx exactly as it does in a single-sample pass.
	bool resolve_depth = reads_depth && key->samples > VK_SAMPLE_COUNT_1_BIT && !(key->flags & skr_rp_flag_postfx_depth_ms);

	if (reads_depth && !_skr_vk.has_create_renderpass2) {
		skr_log(skr_log_critical, "PostFX depth read requires VK_KHR_create_renderpass2");
		return VK_NULL_HANDLE;
	}
	if (resolve_depth && !_skr_vk.has_depth_stencil_resolve) {
		skr_log(skr_log_critical, "PostFX depth read with MSAA requires VK_KHR_depth_stencil_resolve");
		return VK_NULL_HANDLE;
	}

	// Contents are undefined under both, but DONT_CARE still counts as a store
	// *access* that races the subpass-boundary transition. NONE performs none.
	VkAttachmentStoreOp discard_op = _skr_vk.has_store_op_none
		? VK_ATTACHMENT_STORE_OP_NONE
		: VK_ATTACHMENT_STORE_OP_DONT_CARE;

	// --- Attachment indices (assigned as we go) ---
	int32_t color_idx         = -1;
	int32_t resolve_idx       = -1;
	int32_t depth_idx         = -1;
	int32_t depth_resolve_idx = -1;
	int32_t output_idx        = -1;
	int32_t intermediate_start = -1;

	// [0] Color attachment (MSAA or direct) — geometry subpass output
	if (has_color) {
		color_idx = (int32_t)attachment_count;
		attachments[attachment_count++] = (VkAttachmentDescription2){
			.sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
			.format         = key->color_format,
			.samples        = key->samples,
			.loadOp         = key->color_load_op,
			.storeOp        = discard_op, // postfx reads via input attachment, no need to store
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = (key->color_load_op == VK_ATTACHMENT_LOAD_OP_LOAD) ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
	}

	// [1] Resolve attachment (for MSAA) — resolved result, then input for postfx
	// When resolve-only (no postfx), this IS the final output and must be stored.
	// A resolve subpass implies MSAA, so this never falls through to the
	// postfx_output_format read in the final-output block below.
	bool resolve_is_final = use_msaa && (key->flags & skr_rp_flag_resolve_subpass) && key->postfx_count == 0;
	if (use_msaa) {
		resolve_idx = (int32_t)attachment_count;
		attachments[attachment_count++] = (VkAttachmentDescription2){
			.sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
			.format         = key->resolve_format,
			.samples        = VK_SAMPLE_COUNT_1_BIT,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp        = resolve_is_final ? VK_ATTACHMENT_STORE_OP_STORE : discard_op,
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
			// Matches its last use as a postfx input. Any mismatch makes
			// EndRenderPass transition it again, conflicting with its own store.
			.finalLayout    = (!resolve_is_final && key->postfx_count > 0)
				? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
				: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
	}

	// [2] Depth attachment
	VkImageLayout depth_final = key->final_depth_layout ? key->final_depth_layout : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	if (has_depth) {
		bool has_stencil = _skr_format_has_stencil(key->depth_format);
		depth_idx = (int32_t)attachment_count;
		attachments[attachment_count++] = (VkAttachmentDescription2){
			.sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
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

	// [3] Depth resolve attachment — pooled 1x transient the geometry subpass
	// resolves depth into, read by postfx as an input attachment.
	if (resolve_depth) {
		depth_resolve_idx = (int32_t)attachment_count;
		attachments[attachment_count++] = (VkAttachmentDescription2){
			.sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
			.format         = key->depth_format,
			.samples        = VK_SAMPLE_COUNT_1_BIT,
			.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp        = discard_op, // transient, consumed as input
			.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
	}

	// [4..] Intermediate transient color attachments for postfx chaining
	VkFormat intermediate_format = use_msaa ? key->resolve_format : key->color_format;
	if (key->postfx_output_format != VK_FORMAT_UNDEFINED)
		intermediate_format = key->postfx_output_format;

	uint32_t intermediate_count = key->postfx_count > 1 ? key->postfx_count - 1 : 0;
	if (intermediate_count > 0) {
		intermediate_start = (int32_t)attachment_count;
		for (uint32_t i = 0; i < intermediate_count; i++) {
			attachments[attachment_count++] = (VkAttachmentDescription2){
				.sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
				.format         = intermediate_format,
				.samples        = VK_SAMPLE_COUNT_1_BIT,
				.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.storeOp        = discard_op, // transient, consumed as input
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
		attachments[attachment_count++] = (VkAttachmentDescription2){
			.sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
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
	uint32_t resolve_subpass_count = (key->flags & skr_rp_flag_resolve_subpass) ? 1 : 0;
	uint32_t subpass_count = 1 + resolve_subpass_count + key->postfx_count;

	VkImageAspectFlags depth_aspect = has_depth
		? (VK_IMAGE_ASPECT_DEPTH_BIT | (_skr_format_has_stencil(key->depth_format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0))
		: 0;

	// Attachment references (pre-allocate max needed; input_refs is
	// [subpass][color, depth] — InputAttachmentIndex 0 = color, 1 = depth)
	VkAttachmentReference2 color_refs  [SKR_POSTFX_MAX_SUBPASSES];
	VkAttachmentReference2 resolve_refs[SKR_POSTFX_MAX_SUBPASSES];
	VkAttachmentReference2 input_refs  [SKR_POSTFX_MAX_SUBPASSES][2];
	VkAttachmentReference2 depth_ref = {
		.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
		.attachment = depth_idx >= 0 ? (uint32_t)depth_idx : VK_ATTACHMENT_UNUSED,
		.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		.aspectMask = depth_aspect,
	};

	VkSubpassDescription2 subpasses[SKR_POSTFX_MAX_SUBPASSES];
	memset(subpasses, 0, sizeof(subpasses));

	// Subpass 0: Geometry
	// When manual resolve is active, NO pResolveAttachments — the resolve subpass
	// reads the MSAA color directly as an input attachment instead.
	color_refs[0] = (VkAttachmentReference2){
		.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
		.attachment = has_color ? (uint32_t)color_idx : VK_ATTACHMENT_UNUSED,
		.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	};
	resolve_refs[0] = (VkAttachmentReference2){
		.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
		.attachment = use_msaa ? (uint32_t)resolve_idx : VK_ATTACHMENT_UNUSED,
		.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	};
	// On-tile depth resolve: geometry subpass resolves depth into the 1x
	// transient so postfx reads single-sample depth at MSAA settings too.
	VkAttachmentReference2 depth_resolve_ref = {
		.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
		.attachment = resolve_depth ? (uint32_t)depth_resolve_idx : VK_ATTACHMENT_UNUSED,
		.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
	};
	VkSubpassDescriptionDepthStencilResolve depth_resolve_info = {
		.sType                          = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE,
		.depthResolveMode               = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT, // The only mode Vulkan mandates
		.stencilResolveMode             = VK_RESOLVE_MODE_NONE,
		.pDepthStencilResolveAttachment = &depth_resolve_ref,
	};
	bool use_auto_resolve = use_msaa && !(key->flags & skr_rp_flag_resolve_subpass);
	// Tile shading apron: the subpasses that *produce* the tile data (geometry,
	// and manual resolve below) rasterize into the apron border so a later
	// tile-attachment read finds valid neighbors at tile edges. The reading
	// postfx subpasses do NOT get the bit — their fragments only run on real
	// tile pixels, keeping every read within tile + apron.
	VkSubpassDescriptionFlags apron_flag = ((key->flags & skr_rp_flag_tile_shading) && (key->tile_apron[0] | key->tile_apron[1]))
		? VK_SUBPASS_DESCRIPTION_TILE_SHADING_APRON_BIT_QCOM : 0;
	subpasses[0] = (VkSubpassDescription2){
		.sType                   = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
		.pNext                   = resolve_depth ? &depth_resolve_info : NULL,
		.flags                   = apron_flag,
		.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.viewMask                = key->view_mask,
		.colorAttachmentCount    = has_color ? 1 : 0,
		.pColorAttachments       = has_color ? &color_refs[0] : NULL,
		.pResolveAttachments     = use_auto_resolve ? &resolve_refs[0] : NULL,
		.pDepthStencilAttachment = has_depth ? &depth_ref : NULL,
	};

	// Subpass 1 (optional): Manual MSAA resolve
	// Reads MSAA color as multisampled input attachment, writes 1x resolved output.
	// No depth — frees tile memory after geometry subpass.
	uint32_t next_sp = 1;
	if (key->flags & skr_rp_flag_resolve_subpass) {
		input_refs[1][0] = (VkAttachmentReference2){
			.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
			.attachment = (uint32_t)color_idx,
			.layout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		};
		color_refs[1] = (VkAttachmentReference2){
			.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
			.attachment = (uint32_t)resolve_idx,
			.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		};
		subpasses[1] = (VkSubpassDescription2){
			.sType                   = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
			.flags                   = apron_flag | ((key->flags & skr_rp_flag_custom_resolve)
				? (VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_QCOM | VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM)
				: 0),
			.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.viewMask                = key->view_mask,
			.inputAttachmentCount    = 1,
			.pInputAttachments       = &input_refs[1][0],
			.colorAttachmentCount    = 1,
			.pColorAttachments       = &color_refs[1],
		};
		next_sp = 2;
	}

	// PostFX subpasses: read previous output (and optionally depth) as input attachments
	int32_t prev_color_attachment = (use_msaa || (key->flags & skr_rp_flag_resolve_subpass)) ? resolve_idx : color_idx;
	int32_t depth_input_attachment = resolve_depth ? depth_resolve_idx : depth_idx;

	for (uint32_t p = 0; p < key->postfx_count; p++) {
		uint32_t sp      = next_sp + p;
		bool     is_last = (p == key->postfx_count - 1);

		// InputAttachmentIndex 0: previous color
		input_refs[sp][0] = (VkAttachmentReference2){
			.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
			.attachment = (uint32_t)prev_color_attachment,
			.layout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		};
		// InputAttachmentIndex 1: depth, only when a postfx shader asks for it —
		// otherwise depth stays out of the input refs to free tile memory.
		if (reads_depth) {
			input_refs[sp][1] = (VkAttachmentReference2){
				.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
				.attachment = (uint32_t)depth_input_attachment,
				.layout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			};
		}

		// Output: last postfx writes to final output, others to intermediate
		int32_t this_output = is_last ? output_idx : intermediate_start + (int32_t)p;

		color_refs[sp] = (VkAttachmentReference2){
			.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
			.attachment = (uint32_t)this_output,
			.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		};

		subpasses[sp] = (VkSubpassDescription2){
			.sType                   = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
			.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.viewMask                = key->view_mask,
			.inputAttachmentCount    = reads_depth ? 2u : 1u,
			.pInputAttachments       = &input_refs[sp][0],
			.colorAttachmentCount    = 1,
			.pColorAttachments       = &color_refs[sp],
		};

		prev_color_attachment = this_output;
	}

	// --- Subpass dependencies ---
	VkSubpassDependency2 dependencies[SKR_POSTFX_MAX_SUBPASSES * 4 + 8]; // per subpass: chain, depth, intermediate-external, plus a few fixed
	uint32_t dep_count = 0;

	// External → subpass 0: color. The scene color/resolve here may be a
	// pooled transient still in use by an earlier pass on this queue, so
	// unlike the single-subpass path, this always orders against prior
	// color writes and input attachment reads.
	dependencies[dep_count++] = (VkSubpassDependency2){
		.sType         = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
		.srcSubpass    = VK_SUBPASS_EXTERNAL,
		.dstSubpass    = 0,
		.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	};
	// External → subpass 0: depth
	if (has_depth) {
		dependencies[dep_count++] = (VkSubpassDependency2){
			.sType         = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
			.srcSubpass    = VK_SUBPASS_EXTERNAL,
			.dstSubpass    = 0,
			.srcStageMask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		};
	}
	// External → subpass 0: pooled depth-resolve transient — order its
	// UNDEFINED transition after any prior-pass use on this queue.
	if (resolve_depth) {
		dependencies[dep_count++] = (VkSubpassDependency2){
			.sType         = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
			.srcSubpass    = VK_SUBPASS_EXTERNAL,
			.dstSubpass    = 0,
			.srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		};
	}

	// External → intermediate-writing postfx subpasses: the intermediates are
	// pooled and may still be in use by an earlier pass on this queue — order
	// their UNDEFINED layout transition after that work completes.
	for (uint32_t i = 0; i < intermediate_count; i++) {
		dependencies[dep_count++] = (VkSubpassDependency2){
			.sType         = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
			.srcSubpass    = VK_SUBPASS_EXTERNAL,
			.dstSubpass    = next_sp + i,
			.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
		};
	}

	// Attachments first referenced by the last subpass transition at that
	// boundary rather than at pass begin, so external→0 never covers them.
	if (output_idx >= 0 && subpass_count > 1) {
		dependencies[dep_count++] = (VkSubpassDependency2){
			.sType         = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
			.srcSubpass    = VK_SUBPASS_EXTERNAL,
			.dstSubpass    = subpass_count - 1,
			.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
		};
	}

	// Subpass N → N+1: tile-local dependency (covers resolve + postfx chain).
	// In a tile shading pass the consuming subpass reads the attachment as a
	// tile attachment too; that access is a 64-bit sync2 flag, so it's carried
	// by a chained VkMemoryBarrier2 (which supersedes the dependency's own
	// masks). BY_REGION in a tile shading pass is tile-granular rather than
	// pixel-granular — exactly what makes the neighborhood reads legal.
	VkMemoryBarrier2 tile_read_barriers[SKR_POSTFX_MAX_SUBPASSES];
	for (uint32_t s = 0; s + 1 < subpass_count; s++) {
		if (key->flags & skr_rp_flag_tile_shading) {
			tile_read_barriers[s] = (VkMemoryBarrier2){
				.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
				.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
				.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				.dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_TILE_ATTACHMENT_READ_BIT_QCOM,
			};
		}
		dependencies[dep_count++] = (VkSubpassDependency2){
			.sType           = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
			.pNext           = (key->flags & skr_rp_flag_tile_shading) ? &tile_read_barriers[s] : NULL,
			.srcSubpass      = s,
			.dstSubpass      = s + 1,
			.srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask   = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
		};
	}

	// Geometry → each postfx subpass: depth writes (and the on-tile depth
	// resolve, which Vulkan places in late fragment tests / color output)
	// must land before postfx reads depth as an input attachment.
	if (reads_depth) {
		for (uint32_t p = 0; p < key->postfx_count; p++) {
			dependencies[dep_count++] = (VkSubpassDependency2){
				.sType           = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
				.srcSubpass      = 0,
				.dstSubpass      = next_sp + p,
				.srcStageMask    = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				.srcAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				.dstAccessMask   = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
				.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
			};
		}
	}

	// Subpass → EXTERNAL: ensure writes complete before downstream shader reads
	// when finalLayout transitions to a readable layout (free on tilers).
	if (final_output_layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
	    (resolve_is_final && key->final_resolve_layout && key->final_resolve_layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)) {
		dependencies[dep_count++] = (VkSubpassDependency2){
			.sType         = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
			.srcSubpass    = subpass_count - 1,
			.dstSubpass    = VK_SUBPASS_EXTERNAL,
			.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
	}
	if (has_depth && depth_final != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		dependencies[dep_count++] = (VkSubpassDependency2){
			.sType         = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
			.srcSubpass    = reads_depth ? subpass_count - 1 : 0,  // Without depth read, depth is only used in geometry subpass
			.dstSubpass    = VK_SUBPASS_EXTERNAL,
			.srcStageMask  = reads_depth ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT : VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
	}

	// --- Create ---
	VkRenderPassCreateInfo2 render_pass_info = {
		.sType                   = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
		.attachmentCount         = attachment_count,
		.pAttachments            = attachments,
		.subpassCount            = subpass_count,
		.pSubpasses              = subpasses,
		.dependencyCount         = dep_count,
		.pDependencies           = dependencies,
		.correlatedViewMaskCount = key->correlation_mask ? 1 : 0,
		.pCorrelatedViewMasks    = key->correlation_mask ? &key->correlation_mask : NULL,
	};

	// Mark the whole render pass as a tile shading pass with the requested apron
	VkRenderPassTileShadingCreateInfoQCOM tile_shading_info = {
		.sType         = VK_STRUCTURE_TYPE_RENDER_PASS_TILE_SHADING_CREATE_INFO_QCOM,
		.flags         = VK_TILE_SHADING_RENDER_PASS_ENABLE_BIT_QCOM,
		.tileApronSize = { key->tile_apron[0], key->tile_apron[1] },
	};
	if (key->flags & skr_rp_flag_tile_shading) {
		tile_shading_info.pNext = (void*)render_pass_info.pNext;
		render_pass_info.pNext  = &tile_shading_info;
	}

	// Subpass merge feedback — drivers that report it tell us whether this
	// chain actually merged into one tile pass. Tile-locality of the whole
	// postfx system depends on merging, so surface it in the log.
	VkRenderPassCreationFeedbackInfoEXT       merge_feedback      = {0};
	VkRenderPassCreationFeedbackCreateInfoEXT merge_feedback_info = {
		.sType               = VK_STRUCTURE_TYPE_RENDER_PASS_CREATION_FEEDBACK_CREATE_INFO_EXT,
		.pRenderPassFeedback = &merge_feedback,
	};
	VkRenderPassSubpassFeedbackInfoEXT       subpass_feedback     [SKR_POSTFX_MAX_SUBPASSES] = {0};
	VkRenderPassSubpassFeedbackCreateInfoEXT subpass_feedback_info[SKR_POSTFX_MAX_SUBPASSES];
	bool merge_report = _skr_vk.has_subpass_merge_feedback && _skr_vk.has_create_renderpass2;
	if (merge_report) {
		for (uint32_t s = 0; s < subpass_count; s++) {
			subpass_feedback_info[s] = (VkRenderPassSubpassFeedbackCreateInfoEXT){
				.sType            = VK_STRUCTURE_TYPE_RENDER_PASS_SUBPASS_FEEDBACK_CREATE_INFO_EXT,
				.pNext            = subpasses[s].pNext,
				.pSubpassFeedback = &subpass_feedback[s],
			};
			subpasses[s].pNext = &subpass_feedback_info[s];
		}
		merge_feedback_info.pNext = render_pass_info.pNext;
		render_pass_info.pNext    = &merge_feedback_info;
	}

	VkRenderPass render_pass;
	if (_skr_vk.has_create_renderpass2) {
		PFN_vkCreateRenderPass2 create2 = vkCreateRenderPass2 ? vkCreateRenderPass2 : vkCreateRenderPass2KHR;
		VkResult vr = create2(_skr_vk.device, &render_pass_info, NULL, &render_pass);
		SKR_VK_CHECK_RET(vr, "vkCreateRenderPass2 (postfx)", VK_NULL_HANDLE);

		// Render passes are cached, so this logs once per unique pass config
		if (merge_report) {
			skr_log(skr_log_info, "Postfx render pass: %u subpasses ran as %u after merging", subpass_count, merge_feedback.postMergeSubpassCount);
			for (uint32_t s = 0; s < subpass_count; s++) {
				if (subpass_feedback[s].subpassMergeStatus != VK_SUBPASS_MERGE_STATUS_MERGED_EXT)
					skr_log(skr_log_info, "  subpass %u unmerged (status %u): %s", s, (uint32_t)subpass_feedback[s].subpassMergeStatus, subpass_feedback[s].description);
			}
		}
	} else {
		render_pass = _skr_renderpass_create_compat(&render_pass_info, key->view_mask);
		if (render_pass == VK_NULL_HANDLE) return VK_NULL_HANDLE;
	}

	char name[256];
	snprintf(name, sizeof(name), "rpass_%s%s%s%u_",
		(key->flags & skr_rp_flag_resolve_subpass)    ? "resolve_" : "",
		(key->flags & skr_rp_flag_custom_resolve)     ? "cr_"      : "",
		(key->flags & skr_rp_flag_postfx_reads_depth) ? "din_"     : "",
		key->postfx_count);
	_skr_append_renderpass_config(name, sizeof(name), key);
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_RENDER_PASS, (uint64_t)render_pass, name);

	return render_pass;
}

static VkRenderPass _skr_pipeline_create_renderpass(const skr_pipeline_renderpass_key_t* key) {
	// Multi-subpass path for postfx and/or manual resolve
	if (key->postfx_count > 0 || (key->flags & skr_rp_flag_resolve_subpass))
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
	// srcStageMask must cover the implicit initialLayout transition, not just the
	// loadOp. loadOp=CLEAR discards previous *contents*, but the transition is
	// itself a full-image write that happens inside this dependency — so a
	// TOP_OF_PIPE source (which orders against nothing) leaves it unsynchronized
	// against both the acquire semaphore and the prior frame's storeOp.
	// Each dependency sources from the stage matching its own attachment type, so
	// depth work doesn't get serialized behind unrelated color work, or vice versa.
	VkSubpassDependency dependencies[4];
	uint32_t dep_count = 0;

	dependencies[dep_count++] = (VkSubpassDependency){
		.srcSubpass    = VK_SUBPASS_EXTERNAL,
		.dstSubpass    = 0,
		.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		// The transition/loadOp write is WAW against the prior frame's storeOp on
		// the same image, so this needs a memory dependency, not just execution.
		.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	};
	dependencies[dep_count++] = (VkSubpassDependency){
		.srcSubpass    = VK_SUBPASS_EXTERNAL,
		.dstSubpass    = 0,
		.srcStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		.dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
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
			.dstStageMask  = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		};
	}
	if (key->depth_format != VK_FORMAT_UNDEFINED && depth_final != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		dependencies[dep_count++] = (VkSubpassDependency){
			.srcSubpass    = 0,
			.dstSubpass    = VK_SUBPASS_EXTERNAL,
			.srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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

// Max vertex attributes a pipeline can wire. Vulkan guarantees
// maxVertexInputAttributes >= 16, which comfortably covers every builtin and
// realistic mesh format.
#define SKR_MAX_VERTEX_ATTRIBUTES 16

static const char* _skr_vert_class_name(skr_vert_class_ cls) {
	switch (cls) {
		case skr_vert_class_sint: return "int";
		case skr_vert_class_uint: return "uint";
		default:                  return "float";
	}
}

// Logs a descriptive vertex-layout error naming both sides by semantic, so a
// mismatch reads as a diagnosable message instead of garbage geometry or a
// validation-layer crash. `reason` states which input failed and why.
static void _skr_log_vertex_layout_error(const skr_shader_t* shader, const skr_vert_type_t* vert_type, const char* reason) {
	char   shader_wants[256];
	char   mesh_has    [256];
	size_t pos;

	const sksc_shader_meta_t* meta = &shader->meta;
	pos = 0;
	shader_wants[0] = '\0';
	for (int32_t i = 0; i < meta->vertex_input_count && pos < sizeof(shader_wants); i++) {
		const skr_vert_component_t* v = &meta->vertex_inputs[i];
		pos += snprintf(shader_wants + pos, sizeof(shader_wants) - pos, "%s%s%d(loc%d)",
			i == 0 ? "" : ", ", _skr_semantic_name(v->semantic), v->semantic_slot, v->location);
	}
	pos = 0;
	mesh_has[0] = '\0';
	for (uint32_t i = 0; i < vert_type->component_count && pos < sizeof(mesh_has); i++) {
		const skr_vert_component_t* v = &vert_type->components[i];
		pos += snprintf(mesh_has + pos, sizeof(mesh_has) - pos, "%s%s%d",
			i == 0 ? "" : ", ", _skr_semantic_name(v->semantic), v->semantic_slot);
	}

	skr_log(skr_log_critical, "Shader '%s' vertex input mismatch: %s\n  shader inputs: %s\n  mesh provides: %s",
		meta->name[0] ? meta->name : "shader", reason, shader_wants, mesh_has);
}

static VkPipeline _skr_pipeline_create(int32_t material_idx, int32_t renderpass_idx, int32_t vertformat_idx) {
	const _skr_pipeline_material_key_t*  mat_key   = &_skr_pipeline_cache.materials   [material_idx  ].key;
	const skr_pipeline_renderpass_key_t* rp_key    = &_skr_pipeline_cache.renderpasses[renderpass_idx].key;
	const skr_vert_type_t*               vert_type = &_skr_pipeline_cache.vertformats [vertformat_idx].vert_type;
	const VkPipelineLayout               layout    =  _skr_pipeline_cache.materials   [material_idx  ].layout;
	const VkRenderPass                   rp        =  _skr_pipeline_get_renderpass(renderpass_idx);

	// One specialization info is shared by both stages; Vulkan ignores map
	// entries whose constantID isn't present in a stage's module.
	VkSpecializationMapEntry  spec_entries[SKR_MAX_SPEC_CONSTANTS];
	VkSpecializationInfo      spec_info;
	const VkSpecializationInfo* spec = _skr_shader_make_spec_info(&mat_key->shader->meta, mat_key->spec_constant_values, spec_entries, &spec_info);

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

	// Vertex input — wire mesh components to shader inputs by semantic + slot
	// rather than by array position, so a mesh's component order is free of the
	// shader's input declaration order. The attribute's location comes from the
	// shader meta (the SPIR-V input location); format/offset/binding are reused
	// from the vert type's baked attributes. Mesh components the shader does not
	// consume are simply not emitted. See docs/PLAN_attribute_remap.md.
	VkVertexInputAttributeDescription vert_attrs[SKR_MAX_VERTEX_ATTRIBUTES];
	uint32_t                          vert_attr_count = 0;
	if (vert_type) {
		const sksc_shader_meta_t* meta = &mat_key->shader->meta;
		for (int32_t in = 0; in < meta->vertex_input_count; in++) {
			const skr_vert_component_t* input = &meta->vertex_inputs[in];

			// Find the mesh component that carries this input's semantic + slot.
			int32_t match = -1;
			for (uint32_t c = 0; c < vert_type->component_count; c++) {
				if (vert_type->components[c].semantic      == input->semantic &&
				    vert_type->components[c].semantic_slot == input->semantic_slot) {
					match = (int32_t)c;
					break;
				}
			}
			if (match == -1) {
				char reason[128];
				snprintf(reason, sizeof(reason), "no mesh component supplies input %s%d",
					_skr_semantic_name(input->semantic), input->semantic_slot);
				_skr_log_vertex_layout_error(mat_key->shader, vert_type, reason);
				return VK_NULL_HANDLE;
			}

			// A buffer format's numeric class (int/uint/float, normalized ->
			// float) must match the shader input's, or Vulkan reads garbage.
			skr_vert_class_ mesh_class  = _skr_vert_fmt_class(vert_type->components[match].format);
			skr_vert_class_ input_class = _skr_vert_fmt_class(input->format);
			if (mesh_class != input_class) {
				char reason[128];
				snprintf(reason, sizeof(reason), "input %s%d expects %s but mesh supplies %s",
					_skr_semantic_name(input->semantic), input->semantic_slot,
					_skr_vert_class_name(input_class), _skr_vert_class_name(mesh_class));
				_skr_log_vertex_layout_error(mat_key->shader, vert_type, reason);
				return VK_NULL_HANDLE;
			}

			if (vert_attr_count >= SKR_MAX_VERTEX_ATTRIBUTES) {
				skr_log(skr_log_critical, "Shader '%s' declares more than %d vertex inputs",
					meta->name[0] ? meta->name : "shader", SKR_MAX_VERTEX_ATTRIBUTES);
				return VK_NULL_HANDLE;
			}
			vert_attrs[vert_attr_count++] = (VkVertexInputAttributeDescription){
				.location = input->location,
				.binding  = vert_type->attributes[match].binding,
				.format   = vert_type->attributes[match].format,
				.offset   = vert_type->attributes[match].offset,
			};
		}
	}

	VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount   = vert_type ? vert_type->binding_count : 0,
		.pVertexBindingDescriptions      = vert_type ? vert_type->bindings : NULL,
		.vertexAttributeDescriptionCount = vert_attr_count,
		.pVertexAttributeDescriptions    = vert_attr_count ? vert_attrs : NULL,
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
