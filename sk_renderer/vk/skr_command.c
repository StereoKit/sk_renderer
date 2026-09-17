// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"

#include <assert.h>
#include <threads.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////////

thread_local int32_t _skr_thread_idx = -1;

///////////////////////////////////////////////////////////////////////////////

bool _skr_cmd_init(void) {
	memset(_skr_vk.thread_pools, 0, sizeof(_skr_vk.thread_pools));
	return true;
}

///////////////////////////////////////////////////////////////////////////////

void _skr_cmd_shutdown(void) {
	_skr_device_wait_idle();

	// Destroy thread command pools and per-thread command ring fences
	mtx_lock(&_skr_vk.thread_pool_mutex);
	for (uint32_t i = 0; i < skr_MAX_THREAD_POOLS; i++) {
		_skr_vk_thread_t *thread = &_skr_vk.thread_pools[i];

		// Clear active_cmd and last_submitted so buffer destroys go directly to immediate destruction
		thread->active_cmd     = NULL;
		thread->last_submitted = NULL;

		for (uint32_t c = 0; c < skr_MAX_COMMAND_RING; c++) {
			// Execute and free any remaining destroy lists
			_skr_destroy_list_execute(&thread->cmd_ring[c].destroy_list);
			_skr_destroy_list_free   (&thread->cmd_ring[c].destroy_list);

			if (thread->cmd_ring[c].fence != VK_NULL_HANDLE)
				vkDestroyFence(_skr_vk.device, thread->cmd_ring[c].fence, NULL);
		}
		_skr_bump_ring_destroy(&thread->const_ring);
		_skr_bump_ring_destroy(&thread->storage_ring);
		_skr_desc_cache_destroy(&thread->desc_cache);

		if (thread->cmd_pool != VK_NULL_HANDLE)
			vkDestroyCommandPool(_skr_vk.device, thread->cmd_pool, NULL);

		*thread = (_skr_vk_thread_t){0};
	}
	mtx_unlock(&_skr_vk.thread_pool_mutex);

	// Each thread's pool index is a thread_local this loop can't reach. Reset
	// the calling thread's so a later skr_init can re-register it.
	_skr_thread_idx = -1;
}

///////////////////////////////////////////////////////////////////////////////

_skr_vk_thread_t* _skr_cmd_get_thread(void) {
	if (_skr_thread_idx >= 0) {
		return &_skr_vk.thread_pools[_skr_thread_idx];
	}
	return NULL;
}

///////////////////////////////////////////////////////////////////////////////

void skr_thread_init(void) {
	// Already initialized for this thread
	if (_skr_thread_idx >= 0) {
		skr_log(skr_log_critical, "Thread already initialized with index %d", _skr_thread_idx);
		return;
	}

	// Create command pool first (outside the lock)
	_skr_vk_thread_t thread = {
		.alive = true,
	};
	VkResult vr = vkCreateCommandPool(_skr_vk.device, &(VkCommandPoolCreateInfo){
		.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = _skr_vk.graphics_queue_family,
	}, NULL, &thread.cmd_pool);
	SKR_VK_CHECK_RET(vr, "vkCreateCommandPool",);

	// Lock and find an available slot
	mtx_lock(&_skr_vk.thread_pool_mutex);

	// Search for first available slot (either never used or marked non-alive)
	int32_t thread_idx = -1;
	for (uint32_t i = 0; i < skr_MAX_THREAD_POOLS; i++) {
		if (!_skr_vk.thread_pools[i].alive) {
			thread_idx = i;
			break;
		}
	}

	if (thread_idx < 0) {
		mtx_unlock(&_skr_vk.thread_pool_mutex);
		vkDestroyCommandPool(_skr_vk.device, thread.cmd_pool, NULL);
		skr_log(skr_log_critical, "Exceeded maximum thread pools (%d)", skr_MAX_THREAD_POOLS);
		return;
	}

	// Register thread - set thread_idx and copy to array atomically
	_skr_thread_idx                  = thread_idx;
	thread.thread_idx                = thread_idx;
	_skr_vk.thread_pools[thread_idx] = thread;
	_skr_vk_thread_t* pool = &_skr_vk.thread_pools[thread_idx];
	_skr_bump_ring_init(&pool->const_ring,   skr_buffer_type_constant, _skr_vk.min_ubo_offset_align,  pool->cmd_ring);
	_skr_bump_ring_init(&pool->storage_ring, skr_buffer_type_storage,  _skr_vk.min_ssbo_offset_align, pool->cmd_ring);

	mtx_unlock(&_skr_vk.thread_pool_mutex);

	char name[64];
	snprintf(name, sizeof(name), "CommandPool_thr%d", thread_idx);
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_COMMAND_POOL, (uint64_t)thread.cmd_pool, name);

	return;
}

///////////////////////////////////////////////////////////////////////////////

bool skr_thread_is_initialized(void) {
	return _skr_thread_idx >= 0;
}

///////////////////////////////////////////////////////////////////////////////

void skr_thread_shutdown(void) {
	if (_skr_thread_idx < 0) {
		skr_log(skr_log_warning, "Thread not initialized, nothing to shutdown");
		return;
	}

	mtx_lock(&_skr_vk.thread_pool_mutex);

	_skr_vk_thread_t *thread = &_skr_vk.thread_pools[_skr_thread_idx];

	// Clear active_cmd and last_submitted so buffer destroys go directly to immediate destruction
	thread->active_cmd     = NULL;
	thread->last_submitted = NULL;

	// Clean up command ring - wait on each fence individually and destroy
	for (uint32_t c = 0; c < skr_MAX_COMMAND_RING; c++) {
		if (thread->cmd_ring[c].fence != VK_NULL_HANDLE) {
			vkWaitForFences(_skr_vk.device, 1, &thread->cmd_ring[c].fence, VK_TRUE, UINT64_MAX);
		}

		_skr_destroy_list_execute(&thread->cmd_ring[c].destroy_list);
		_skr_destroy_list_free   (&thread->cmd_ring[c].destroy_list);

		if (thread->cmd_ring[c].fence != VK_NULL_HANDLE)
			vkDestroyFence(_skr_vk.device, thread->cmd_ring[c].fence, NULL);
	}
	_skr_bump_ring_destroy(&thread->const_ring);
	_skr_bump_ring_destroy(&thread->storage_ring);
	_skr_desc_cache_destroy(&thread->desc_cache);

	// Destroy command pool
	if (thread->cmd_pool != VK_NULL_HANDLE)
		vkDestroyCommandPool(_skr_vk.device, thread->cmd_pool, NULL);

	// Mark as non-alive for reuse (don't zero out the whole struct)
	thread->alive           = false;
	thread->cmd_pool        = VK_NULL_HANDLE;
	thread->active_cmd      = NULL;
	thread->cmd_ring_index  = 0;
	thread->ref_count       = 0;
	memset(thread->cmd_ring, 0, sizeof(thread->cmd_ring));

	_skr_thread_idx = -1;

	mtx_unlock(&_skr_vk.thread_pool_mutex);
}

///////////////////////////////////////////////////////////////////////////////

static _skr_cmd_ring_slot_t *_skr_cmd_ring_begin(_skr_vk_thread_t* ref_pool) {
	// Find available slot in the per-thread command ring
	_skr_cmd_ring_slot_t* slot      = NULL;
	uint32_t              start_idx = ref_pool->cmd_ring_index;

	uint32_t idx;
	for (uint32_t i = 0; i < skr_MAX_COMMAND_RING; i++) {
		idx = (start_idx + i) % skr_MAX_COMMAND_RING;
		_skr_cmd_ring_slot_t* curr = &ref_pool->cmd_ring[idx];

		// Use this slot if available
		if (!curr->alive) {
			slot        = curr;
			slot->alive = true;
			ref_pool->cmd_ring_index = (idx + 1) % skr_MAX_COMMAND_RING;
			break;
		}
	}

	// If no slots available, wait for oldest one
	if (!slot) {
		idx         = start_idx;
		slot        = &ref_pool->cmd_ring[start_idx];
		slot->alive = true;
		vkWaitForFences(_skr_vk.device, 1, &slot->fence, VK_TRUE, UINT64_MAX);
		ref_pool->cmd_ring_index = (start_idx + 1) % skr_MAX_COMMAND_RING;

		// Fence is done, make sure we free its assets too
		_skr_destroy_list_execute(&slot->destroy_list);

		// Increment generation to invalidate old futures referencing this fence
		slot->generation++;
	}

	// Allocate command buffer if needed
	if (slot->cmd == VK_NULL_HANDLE) {
		vkAllocateCommandBuffers(_skr_vk.device, &(VkCommandBufferAllocateInfo){
			.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandPool        = ref_pool->cmd_pool,
			.commandBufferCount = 1,
		}, &slot->cmd);
		VkExportFenceCreateInfo export_fence_info = {
			.sType       = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
			.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT,
		};
		vkCreateFence(_skr_vk.device, &(VkFenceCreateInfo){
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = _skr_vk.has_external_fence_fd ? &export_fence_info : NULL,
		}, NULL, &slot->fence);
		slot->destroy_list = _skr_destroy_list_create();

		char name[64];
		snprintf(name,sizeof(name), "CommandBuffer_thr%u_%u", ref_pool->thread_idx, idx);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_COMMAND_BUFFER, (uint64_t)slot->cmd, name);

		snprintf(name,sizeof(name), "Command_Fence_thr%u_%u", ref_pool->thread_idx, idx);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_FENCE, (uint64_t)slot->fence, name);
	} else {
		vkResetCommandBuffer(slot->cmd, 0);
		vkResetFences       (_skr_vk.device, 1, &slot->fence);
		// The GPU is done with this slot, so the sets it retired can free
		_skr_desc_cache_retire(&ref_pool->desc_cache, idx);
	}
	_skr_bump_ring_slot_begin(&ref_pool->const_ring,   idx);
	_skr_bump_ring_slot_begin(&ref_pool->storage_ring, idx);

	vkBeginCommandBuffer(slot->cmd, &(VkCommandBufferBeginInfo){
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	});

	return slot;
}

///////////////////////////////////////////////////////////////////////////////
// Descriptor writes

void _skr_desc_writes_begin(_skr_desc_writes_t* out_writes, int32_t material_idx) {
	out_writes->write_ct  = 0;
	out_writes->buffer_ct = 0;
	out_writes->image_ct  = 0;
	out_writes->dyn       = (_skr_dyn_offsets_t){0};
	out_writes->dyn.count = _skr_pipeline_get_dyn_bindings(material_idx, out_writes->dyn.bindings);
}

// Every buffer descriptor goes through here, so the layout's choice of dynamic
// vs plain is honored no matter which caller supplies the buffer
void _skr_write_buffer(_skr_desc_writes_t* ref_writes, uint32_t binding, bool storage, VkBuffer buffer, uint32_t offset, uint32_t range) {
	if (ref_writes->write_ct >= _SKR_DESC_WRITES_MAX || ref_writes->buffer_ct >= _SKR_DESC_INFOS_MAX) return;

	// A dynamic binding keeps the buffer at offset 0 in the set and takes the
	// draw's offset at bind time, so the set stays valid across frames
	bool dynamic = false;
	for (uint32_t i = 0; i < ref_writes->dyn.count; i++) {
		if (ref_writes->dyn.bindings[i] != binding) continue;
		ref_writes->dyn.offsets[i] = offset;
		dynamic = true;
	}
	VkDescriptorType type = storage
		? (dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
		: (dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	ref_writes->buffer_infos[ref_writes->buffer_ct] = (VkDescriptorBufferInfo){
		.buffer = buffer,
		.offset = dynamic ? 0 : offset,
		.range  = range == UINT32_MAX ? VK_WHOLE_SIZE : range,
	};
	ref_writes->writes[ref_writes->write_ct] = (VkWriteDescriptorSet){
		.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstBinding      = binding,
		.descriptorCount = 1,
		.descriptorType  = type,
		.pBufferInfo     = &ref_writes->buffer_infos[ref_writes->buffer_ct],
	};
	ref_writes->buffer_ct++;
	ref_writes->write_ct++;
}

void _skr_write_image(_skr_desc_writes_t* ref_writes, uint32_t binding, VkDescriptorType type, VkSampler sampler, VkImageView view, VkImageLayout layout) {
	if (ref_writes->write_ct >= _SKR_DESC_WRITES_MAX || ref_writes->image_ct >= _SKR_DESC_INFOS_MAX) return;

	ref_writes->image_infos[ref_writes->image_ct] = (VkDescriptorImageInfo){
		.sampler     = sampler,
		.imageView   = view,
		.imageLayout = layout,
	};
	ref_writes->writes[ref_writes->write_ct] = (VkWriteDescriptorSet){
		.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstBinding      = binding,
		.descriptorCount = 1,
		.descriptorType  = type,
		.pImageInfo      = &ref_writes->image_infos[ref_writes->image_ct],
	};
	ref_writes->image_ct++;
	ref_writes->write_ct++;
}

static inline uint64_t _skr_hash_word(uint64_t hash, uint64_t word) {
	hash ^= word;
	hash *= 0x9E3779B97F4A7C15ull;
	return hash ^ (hash >> 29);
}

// Fingerprint of a set's contents: the layout it targets and every handle,
// offset and range it would be written with. Never returns 0.
static uint64_t _skr_hash_writes(VkDescriptorSetLayout layout, const VkWriteDescriptorSet* writes, uint32_t write_count) {
	uint64_t hash = _skr_hash_word(0x9E3779B97F4A7C15ull, (uint64_t)layout);
	for (uint32_t i = 0; i < write_count; i++) {
		hash = _skr_hash_word(hash, ((uint64_t)writes[i].dstBinding << 32) | (uint32_t)writes[i].descriptorType);
		if (writes[i].pBufferInfo) {
			const VkDescriptorBufferInfo* info = writes[i].pBufferInfo;
			hash = _skr_hash_word(hash, (uint64_t)info->buffer);
			hash = _skr_hash_word(hash, info->offset);
			hash = _skr_hash_word(hash, info->range);
		} else if (writes[i].pImageInfo) {
			const VkDescriptorImageInfo* info = writes[i].pImageInfo;
			hash = _skr_hash_word(hash, (uint64_t)info->sampler);
			hash = _skr_hash_word(hash, (uint64_t)info->imageView);
			hash = _skr_hash_word(hash, (uint64_t)info->imageLayout);
		}
	}
	return hash ? hash : 1;
}

///////////////////////////////////////////////////////////////////////////////
// Descriptor set cache, see _skr_desc_cache_t

#define _SKR_DESC_TYPE_COUNT 9
static const VkDescriptorType _skr_desc_types[_SKR_DESC_TYPE_COUNT] = {
	VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
	VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
	VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM,
	VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM,
};
#define _SKR_DESC_QCOM_FIRST    7 // index of the first type that needs VK_QCOM_image_processing
#define _SKR_DESC_POOL_SETS     512
#define _SKR_DESC_POOL_PER_TYPE 2048

static VkDescriptorPool _skr_desc_pool_create(void) {
	VkDescriptorPoolSize sizes[_SKR_DESC_TYPE_COUNT];
	uint32_t             size_ct = 0;
	for (int32_t i = 0; i < _SKR_DESC_TYPE_COUNT; i++) {
		if (i >= _SKR_DESC_QCOM_FIRST && !_skr_vk.has_qcom_image_proc) continue;
		sizes[size_ct++] = (VkDescriptorPoolSize){ .type = _skr_desc_types[i], .descriptorCount = _SKR_DESC_POOL_PER_TYPE };
	}
	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkResult vr = vkCreateDescriptorPool(_skr_vk.device, &(VkDescriptorPoolCreateInfo){
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets       = _SKR_DESC_POOL_SETS,
		.poolSizeCount = size_ct,
		.pPoolSizes    = sizes,
	}, NULL, &pool);
	if (vr != VK_SUCCESS) skr_log(skr_log_critical, "vkCreateDescriptorPool failed: %d", vr);
	return pool;
}

static _skr_desc_cache_entry_t* _skr_desc_cache_entry(_skr_desc_cache_t* ref_cache, int32_t bind_start) {
	uint32_t chunk_idx = (uint32_t)bind_start >> _SKR_DESC_CACHE_SHIFT;
	if (bind_start < 0 || chunk_idx >= _SKR_DESC_CACHE_MAX_CHUNKS) return NULL;
	if (ref_cache->chunks[chunk_idx] == NULL)
		ref_cache->chunks[chunk_idx] = _skr_calloc(_SKR_DESC_CACHE_CHUNK, sizeof(_skr_desc_cache_entry_t));
	return &ref_cache->chunks[chunk_idx][(uint32_t)bind_start & (_SKR_DESC_CACHE_CHUNK - 1)];
}

static bool _skr_desc_cache_alloc(_skr_desc_cache_t* ref_cache, VkDescriptorSetLayout layout, VkDescriptorSet* out_set, uint8_t* out_pool) {
	VkDescriptorSetAllocateInfo info = {
		.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorSetCount = 1,
		.pSetLayouts        = &layout,
	};
	// Freed sets go back to the pool they came from, so any pool may have room.
	// One that fails with sets left has run out of a descriptor type, not sets.
	for (uint32_t i = 0; i < ref_cache->pool_count; i++) {
		if (ref_cache->pool_used[i] >= _SKR_DESC_POOL_SETS || ref_cache->pool_no_room[i]) continue;
		info.descriptorPool = ref_cache->pools[i];
		if (vkAllocateDescriptorSets(_skr_vk.device, &info, out_set) != VK_SUCCESS) {
			ref_cache->pool_no_room[i] = true;
			continue;
		}
		ref_cache->pool_used[i]++;
		*out_pool = (uint8_t)i;
		return true;
	}
	if (ref_cache->pool_count >= 255) {
		skr_log(skr_log_critical, "Descriptor set cache is out of pools");
		return false;
	}
	VkDescriptorPool pool = _skr_desc_pool_create();
	if (pool == VK_NULL_HANDLE) return false;
	if (ref_cache->pool_count == ref_cache->pool_capacity) {
		ref_cache->pool_capacity = ref_cache->pool_capacity == 0 ? 4 : ref_cache->pool_capacity * 2;
		ref_cache->pools         = _skr_realloc(ref_cache->pools,        ref_cache->pool_capacity * sizeof(VkDescriptorPool));
		ref_cache->pool_used     = _skr_realloc(ref_cache->pool_used,    ref_cache->pool_capacity * sizeof(uint16_t));
		ref_cache->pool_no_room  = _skr_realloc(ref_cache->pool_no_room, ref_cache->pool_capacity * sizeof(bool));
	}
	uint32_t idx = ref_cache->pool_count++;
	ref_cache->pools       [idx] = pool;
	ref_cache->pool_used   [idx] = 0;
	ref_cache->pool_no_room[idx] = false;
	info.descriptorPool = pool;
	if (vkAllocateDescriptorSets(_skr_vk.device, &info, out_set) != VK_SUCCESS) {
		skr_log(skr_log_critical, "vkAllocateDescriptorSets failed on a fresh cache pool");
		return false;
	}
	ref_cache->pool_used[idx] = 1;
	*out_pool = (uint8_t)idx;
	return true;
}

bool _skr_bind_descriptors(VkCommandBuffer cmd, VkPipelineBindPoint bind_point, int32_t bind_start, VkPipelineLayout layout, VkDescriptorSetLayout desc_layout, _skr_desc_writes_t* ref_writes) {
	if (ref_writes->write_ct == 0) return true;

	if (_skr_vk.has_push_descriptors) {
		vkCmdPushDescriptorSetKHR(cmd, bind_point, layout, 0, ref_writes->write_ct, ref_writes->writes);
		return true;
	}

	// One entry per bind slice, so a caller that varies its writes within one
	// material (mipgen's per-mip views) misses every time and churns a set per call
	_skr_vk_thread_t*        thread = _skr_cmd_get_thread();
	uint32_t                 slot   = (uint32_t)(thread->active_cmd - thread->cmd_ring);
	_skr_desc_cache_t*       cache  = &thread->desc_cache;
	_skr_desc_cache_entry_t* entry  = _skr_desc_cache_entry(cache, bind_start);
	if (entry == NULL) {
		skr_log(skr_log_critical, "Descriptor bind without a bind slice");
		return false;
	}

	uint64_t hash = _skr_hash_writes(desc_layout, ref_writes->writes, ref_writes->write_ct);
	if (entry->hash != hash) {
		if (entry->set != VK_NULL_HANDLE) {
			if (cache->retired_count[slot] == cache->retired_capacity[slot]) {
				cache->retired_capacity[slot] = cache->retired_capacity[slot] == 0 ? 16 : cache->retired_capacity[slot] * 2;
				cache->retired[slot]          = _skr_realloc(cache->retired[slot], cache->retired_capacity[slot] * sizeof(_skr_desc_retired_t));
			}
			cache->retired[slot][cache->retired_count[slot]++] = (_skr_desc_retired_t){ .set = entry->set, .pool = entry->pool };
		}
		entry->set  = VK_NULL_HANDLE;
		entry->hash = 0;
		if (!_skr_desc_cache_alloc(cache, desc_layout, &entry->set, &entry->pool))
			return false;
		for (uint32_t i = 0; i < ref_writes->write_ct; i++)
			ref_writes->writes[i].dstSet = entry->set;
		vkUpdateDescriptorSets(_skr_vk.device, ref_writes->write_ct, ref_writes->writes, 0, NULL);
		entry->hash = hash;
	}
	vkCmdBindDescriptorSets(cmd, bind_point, layout, 0, 1, &entry->set, ref_writes->dyn.count, ref_writes->dyn.offsets);
	return true;
}

void _skr_desc_cache_retire(_skr_desc_cache_t* ref_cache, uint32_t slot) {
	for (uint32_t i = 0; i < ref_cache->retired_count[slot]; i++) {
		_skr_desc_retired_t retired = ref_cache->retired[slot][i];
		vkFreeDescriptorSets(_skr_vk.device, ref_cache->pools[retired.pool], 1, &retired.set);
		ref_cache->pool_used   [retired.pool]--;
		ref_cache->pool_no_room[retired.pool] = false;
	}
	ref_cache->retired_count[slot] = 0;
}

void _skr_desc_cache_destroy(_skr_desc_cache_t* ref_cache) {
	for (uint32_t i = 0; i < _SKR_DESC_CACHE_MAX_CHUNKS; i++)
		_skr_free(ref_cache->chunks[i]);
	for (uint32_t i = 0; i < skr_MAX_COMMAND_RING; i++)
		_skr_free(ref_cache->retired[i]);
	for (uint32_t i = 0; i < ref_cache->pool_count; i++)
		vkDestroyDescriptorPool(_skr_vk.device, ref_cache->pools[i], NULL);
	_skr_free(ref_cache->pools);
	_skr_free(ref_cache->pool_used);
	_skr_free(ref_cache->pool_no_room);
	*ref_cache = (_skr_desc_cache_t){0};
}

///////////////////////////////////////////////////////////////////////////////

_skr_cmd_ctx_t _skr_cmd_begin(void) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	(void)pool; // only read by the asserts below
	assert(pool);
	assert(pool->ref_count == 0 && "Ref count should be 0 at batch start");

	return _skr_cmd_acquire();
}

///////////////////////////////////////////////////////////////////////////////

bool _skr_cmd_try_get_active(_skr_cmd_ctx_t* out_ctx) {
	*out_ctx = (_skr_cmd_ctx_t){0};

	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	if (pool->active_cmd) {
		*out_ctx = (_skr_cmd_ctx_t){
			.cmd             = pool->active_cmd->cmd,
			.destroy_list    = &pool->active_cmd->destroy_list,
			.const_ring      = &pool->const_ring,
			.storage_ring    = &pool->storage_ring,
		};
		return true;
	}
	return false;
}

///////////////////////////////////////////////////////////////////////////////

_skr_cmd_ctx_t _skr_cmd_acquire(void) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	if (pool->ref_count == 0)
		pool->active_cmd = _skr_cmd_ring_begin(pool);

	pool->ref_count++;
	return (_skr_cmd_ctx_t){
		.cmd             = pool->active_cmd->cmd,
		.destroy_list    = &pool->active_cmd->destroy_list,
		.const_ring      = &pool->const_ring,
		.storage_ring    = &pool->storage_ring,
	};
}

///////////////////////////////////////////////////////////////////////////////

void _skr_cmd_release(VkCommandBuffer buffer) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	pool->ref_count--;
	assert(pool->ref_count       >= 0      && "Unbalanced acquire/release");
	assert(pool->active_cmd->cmd == buffer && "Shouldn't release someone else's buffer!");

	if (pool->ref_count == 0) {
		// Outside a batch: submit the command buffer from the ring
		// The ring will handle waiting when it needs to reuse a slot
		vkEndCommandBuffer(pool->active_cmd->cmd);

		mtx_lock(_skr_vk.graphics_queue_mutex);
		vkQueueSubmit(_skr_vk.graphics_queue, 1, &(VkSubmitInfo){
			.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers    = &pool->active_cmd->cmd,
		}, pool->active_cmd->fence);
		mtx_unlock(_skr_vk.graphics_queue_mutex);

		// Track this as the most recently submitted command
		pool->last_submitted = pool->active_cmd;
		pool->active_cmd     = NULL;
	}
}

///////////////////////////////////////////////////////////////////////////////

VkCommandBuffer _skr_cmd_end(void) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	pool->ref_count--;
	assert(pool->ref_count == 0 && "Unbalanced acquire/release - ref count should be 0");

	// Track this as the most recently used command (not yet submitted, but will be soon)
	pool->last_submitted = pool->active_cmd;

	return pool->active_cmd->cmd;
}

///////////////////////////////////////////////////////////////////////////////

skr_future_t _skr_cmd_end_submit(const VkSemaphore* wait_semaphores, uint32_t wait_count, const VkSemaphore* signal_semaphores, uint32_t signal_count) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool && pool->active_cmd);

	pool->ref_count--;
	assert(pool->ref_count == 0 && "Unbalanced acquire/release - ref count should be 0");

	// End the command buffer
	vkEndCommandBuffer(pool->active_cmd->cmd);

	// Build wait stages (one per wait semaphore)
	assert(wait_count <= SKR_MAX_SURFACES && "Wait count exceeds maximum surfaces");
	assert(signal_count <= SKR_MAX_SURFACES && "Signal count exceeds maximum surfaces");

	VkPipelineStageFlags wait_stages[SKR_MAX_SURFACES];
	for (uint32_t i = 0; i < wait_count; i++) {
		wait_stages[i] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	}

	// Submit with command buffer's fence
	mtx_lock(_skr_vk.graphics_queue_mutex);
	vkQueueSubmit(_skr_vk.graphics_queue, 1, &(VkSubmitInfo){
		.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount   = 1,
		.pCommandBuffers      = &pool->active_cmd->cmd,
		.waitSemaphoreCount   = wait_count,
		.pWaitSemaphores      = wait_semaphores,
		.pWaitDstStageMask    = wait_count > 0 ? wait_stages : NULL,
		.signalSemaphoreCount = signal_count,
		.pSignalSemaphores    = signal_semaphores,
	}, pool->active_cmd->fence);  // Always use command buffer's fence
	mtx_unlock(_skr_vk.graphics_queue_mutex);

	// Create future for this submission
	skr_future_t future = {
		.slot       = pool->active_cmd,
		.generation = pool->active_cmd->generation,
	};

	// Track this as the most recently submitted command
	pool->last_submitted = pool->active_cmd;
	pool->active_cmd     = NULL;

	return future;
}

//TODO: all of this is just using the graphics_queue! It should be configurable

///////////////////////////////////////////////////////////////////////////////
// Future API - for GPU/CPU synchronization
///////////////////////////////////////////////////////////////////////////////

skr_future_t skr_future_get(void) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();

	// Invalid future if not on an initialized thread
	if (!pool || !pool->alive) {
		return (skr_future_t){ .slot = NULL, .generation = 0 };
	}

	// Prefer active_cmd if we're currently recording, otherwise use last_submitted
	_skr_cmd_ring_slot_t* target = pool->active_cmd ? pool->active_cmd : pool->last_submitted;

	// Return invalid if no command has been submitted yet
	if (!target || target->fence == VK_NULL_HANDLE) {
		return (skr_future_t){ .slot = NULL, .generation = 0 };
	}

	return (skr_future_t){
		.slot       = target,
		.generation = target->generation,
	};
}

int32_t skr_renderer_frame_fence_fd(void) {
	if (!_skr_vk.has_external_fence_fd) return -1;

	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	if (!pool || !pool->alive) return -1;

	_skr_cmd_ring_slot_t* slot = pool->last_submitted;
	if (!slot || slot->fence == VK_NULL_HANDLE) return -1;

	VkFenceGetFdInfoKHR fd_info = {
		.sType      = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
		.fence      = slot->fence,
		.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT,
	};
	int fd = -1;
	if (vkGetFenceFdKHR(_skr_vk.device, &fd_info, &fd) != VK_SUCCESS) {
		return -1;
	}
	return fd;
}

bool skr_future_check(const skr_future_t* future) {
	if (!future || !future->slot) {
		return true; // Invalid futures are considered "done"
	}

	_skr_cmd_ring_slot_t* slot = (_skr_cmd_ring_slot_t*)future->slot;

	// If generation doesn't match, the slot was reused, so the original work is done
	if (slot->generation != future->generation) {
		return true;
	}

	// Query fence status (non-blocking)
	VkResult result = vkGetFenceStatus(_skr_vk.device, slot->fence);
	return result == VK_SUCCESS; // VK_SUCCESS = signaled, VK_NOT_READY = not signaled
}

void skr_future_wait(const skr_future_t* future) {
	if (!future || !future->slot) {
		return; // Invalid futures are no-op
	}

	_skr_cmd_ring_slot_t* slot = (_skr_cmd_ring_slot_t*)future->slot;

	// If generation doesn't match, the slot was reused, so work is already done
	if (slot->generation != future->generation) {
		return;
	}

	// Block until fence signals
	vkWaitForFences(_skr_vk.device, 1, &slot->fence, VK_TRUE, UINT64_MAX);
}

///////////////////////////////////////////////////////////////////////////////
// Command Batching API - for grouping multiple GPU operations
///////////////////////////////////////////////////////////////////////////////

void skr_cmd_begin(void) {
	_skr_cmd_acquire();
}

skr_future_t skr_cmd_end(void) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool && pool->ref_count > 0 && "Unbalanced skr_cmd_begin/end");

	// Capture future before potentially clearing active_cmd
	skr_future_t future = {
		.slot       = pool->active_cmd,
		.generation = pool->active_cmd->generation,
	};

	_skr_cmd_release(pool->active_cmd->cmd);

	return future;
}

///////////////////////////////////////////////////////////////////////////////

bool skr_cmd_is_active(void) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	return pool && pool->ref_count > 0;
}

skr_future_t skr_cmd_flush(void) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	// Nothing to flush if not recording
	if (pool->active_cmd == NULL || pool->ref_count == 0) {
		return (skr_future_t){ .slot = NULL, .generation = 0 };
	}

	// Save ref_count - we need to restore this level after starting new batch
	int32_t saved_ref_count = pool->ref_count;

	// End recording
	vkEndCommandBuffer(pool->active_cmd->cmd);

	// Submit with no semaphores (mid-frame, not tied to surface)
	mtx_lock(_skr_vk.graphics_queue_mutex);
	vkQueueSubmit(_skr_vk.graphics_queue, 1, &(VkSubmitInfo){
		.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers    = &pool->active_cmd->cmd,
	}, pool->active_cmd->fence);
	mtx_unlock(_skr_vk.graphics_queue_mutex);

	// Create future before clearing active_cmd
	skr_future_t result = {
		.slot       = pool->active_cmd,
		.generation = pool->active_cmd->generation,
	};

	// Mark this slot as submitted and clear
	pool->last_submitted = pool->active_cmd;
	pool->active_cmd     = NULL;
	pool->ref_count      = 0;

	// Immediately start a new batch at the same ref level
	// This ensures outstanding acquires still have a valid command buffer
	pool->active_cmd = _skr_cmd_ring_begin(pool);
	pool->ref_count  = saved_ref_count;

	return result;
}
