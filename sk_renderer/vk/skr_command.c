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

static void         _skr_cmd_slot_retire(_skr_cmd_ring_slot_t* ref_slot, bool destroy_fence);
static skr_future_t _skr_cmd_submit     (_skr_vk_thread_t* ref_pool, const VkSemaphore* opt_waits, uint32_t wait_count, const VkSemaphore* opt_signals, uint32_t signal_count);

///////////////////////////////////////////////////////////////////////////////

bool _skr_cmd_init(void) {
	memset(_skr_vk.thread_pools, 0, sizeof(_skr_vk.thread_pools));
	return true;
}

///////////////////////////////////////////////////////////////////////////////

void _skr_cmd_shutdown(void) {
	_skr_device_wait_idle();

	mtx_lock(&_skr_vk.thread_pool_mutex);
	for (uint32_t i = 0; i < skr_MAX_THREAD_POOLS; i++) {
		_skr_vk_thread_t *thread = &_skr_vk.thread_pools[i];
		_skr_bump_ring_destroy(&thread->const_ring);
		_skr_bump_ring_destroy(&thread->storage_ring);
		for (uint32_t c = 0; c < skr_MAX_COMMAND_RING; c++) {
			if (thread->cmd_ring[c].fence == VK_NULL_HANDLE) continue;
			_skr_destroy_list_execute(&thread->cmd_ring[c].destroy_list, &thread->desc_cache);
			_skr_free(thread->cmd_ring[c].destroy_list.items);
			vkDestroyFence(_skr_vk.device, thread->cmd_ring[c].fence, NULL);
		}
		_skr_destroy_list_execute(&thread->shared_open, &thread->desc_cache);
		_skr_free(thread->shared_open.items);
		_skr_desc_cache_destroy(&thread->desc_cache); // after the lists, which free sets into it
		if (thread->cmd_pool != VK_NULL_HANDLE)
			vkDestroyCommandPool(_skr_vk.device, thread->cmd_pool, NULL);

		*thread = (_skr_vk_thread_t){0};
	}
	mtx_unlock(&_skr_vk.thread_pool_mutex);
	_skr_destroy_drain();

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
		.cur   = skr_MAX_COMMAND_RING - 1, // so the first begin takes slot 0
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

	// Generations outlive the pool's previous owner, so its futures never match
	for (uint32_t c = 0; c < skr_MAX_COMMAND_RING; c++)
		thread.cmd_ring[c].generation = _skr_vk.thread_pools[thread_idx].cmd_ring[c].generation;

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

	_skr_vk_thread_t *thread = &_skr_vk.thread_pools[_skr_thread_idx];

	// An open scope here is an app bug, but its buffer holds real work and
	// its fence would never signal, so submit it rather than hang below
	assert(thread->ref_count == 0 && "skr_thread_shutdown inside an open skr_cmd_begin scope");
	if (thread->ref_count > 0) {
		skr_log(skr_log_warning, "skr_thread_shutdown inside an open command scope, submitting it");
		_skr_cmd_submit(thread, NULL, 0, NULL, 0);
		_skr_store_u32(&thread->ref_count, 0);
	}

	mtx_lock(&_skr_vk.thread_pool_mutex);

	for (uint32_t c = 0; c < skr_MAX_COMMAND_RING; c++) {
		if (thread->cmd_ring[c].fence != VK_NULL_HANDLE)
			vkWaitForFences(_skr_vk.device, 1, &thread->cmd_ring[c].fence, VK_TRUE, UINT64_MAX);
	}

	_skr_bump_ring_destroy(&thread->const_ring);
	_skr_bump_ring_destroy(&thread->storage_ring);

	for (uint32_t c = 0; c < skr_MAX_COMMAND_RING; c++) {
		if (thread->cmd_ring[c].fence == VK_NULL_HANDLE) continue;
		_skr_destroy_list_execute(&thread->cmd_ring[c].destroy_list, &thread->desc_cache);
		_skr_free(thread->cmd_ring[c].destroy_list.items);
	}
	_skr_desc_cache_destroy(&thread->desc_cache); // after the lists, which free sets into it

	// Shared destroys may still be bound in the frame, so they stay pending,
	// and blocks stamped with these fences lose the stamp before the fences go
	mtx_lock(_skr_vk.graphics_queue_mutex);
	_skr_destroy_block_t* done = _skr_destroy_retire(&thread->shared_open, (skr_future_t){0}, false);
	_skr_destroy_disown(thread->cmd_ring);
	mtx_unlock(_skr_vk.graphics_queue_mutex);
	_skr_destroy_execute_blocks(done);
	for (uint32_t c = 0; c < skr_MAX_COMMAND_RING; c++) {
		if (thread->cmd_ring[c].fence != VK_NULL_HANDLE)
			_skr_cmd_slot_retire(&thread->cmd_ring[c], true);
	}

	if (thread->cmd_pool != VK_NULL_HANDLE)
		vkDestroyCommandPool(_skr_vk.device, thread->cmd_pool, NULL);

	// Mark as non-alive for reuse. Generations stay, so a future from this
	// thread can't match the next one to take the pool.
	thread->alive    = false;
	thread->cmd_pool = VK_NULL_HANDLE;
	thread->cur      = 0;
	_skr_store_u32(&thread->ref_count, 0);
	for (uint32_t c = 0; c < skr_MAX_COMMAND_RING; c++) {
		thread->cmd_ring[c].cmd          = VK_NULL_HANDLE;
		thread->cmd_ring[c].fence        = VK_NULL_HANDLE;
		thread->cmd_ring[c].destroy_list = (skr_destroy_list_t){0};
	}

	_skr_thread_idx = -1;

	mtx_unlock(&_skr_vk.thread_pool_mutex);
}

///////////////////////////////////////////////////////////////////////////////

// Ends the slot's generation, then resets or destroys its fence, all under
// the queue mutex: other threads read generation and the fence under it, and
// a thread blocked in skr_future_wait holds `waiters`, which only drains once
// the fence has signaled, so the spin is short.
static void _skr_cmd_slot_retire(_skr_cmd_ring_slot_t* ref_slot, bool destroy_fence) {
	mtx_lock(_skr_vk.graphics_queue_mutex);
	ref_slot->generation++;
	while (_skr_load_u32(&ref_slot->waiters) > 0)
		thrd_yield();
	if (destroy_fence) {
		vkDestroyFence(_skr_vk.device, ref_slot->fence, NULL);
		ref_slot->fence = VK_NULL_HANDLE;
	} else {
		vkResetFences(_skr_vk.device, 1, &ref_slot->fence);
	}
	mtx_unlock(_skr_vk.graphics_queue_mutex);
}

static _skr_cmd_ring_slot_t *_skr_cmd_ring_begin(_skr_vk_thread_t* ref_pool) {
	uint32_t              idx  = (ref_pool->cur + 1) % skr_MAX_COMMAND_RING;
	_skr_cmd_ring_slot_t* slot = &ref_pool->cmd_ring[idx];
	ref_pool->cur = idx;

	// A slot that has been submitted before is the oldest in flight
	if (slot->fence != VK_NULL_HANDLE) {
		vkWaitForFences(_skr_vk.device, 1, &slot->fence, VK_TRUE, UINT64_MAX);
		_skr_destroy_list_execute(&slot->destroy_list, &ref_pool->desc_cache);
		_skr_cmd_slot_retire(slot, false);
		vkResetCommandBuffer(slot->cmd, 0);
	} else {
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
		char name[64];
		snprintf(name,sizeof(name), "CommandBuffer_thr%u_%u", ref_pool->thread_idx, idx);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_COMMAND_BUFFER, (uint64_t)slot->cmd, name);

		snprintf(name,sizeof(name), "Command_Fence_thr%u_%u", ref_pool->thread_idx, idx);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_FENCE, (uint64_t)slot->fence, name);
		if (slot->generation == 0) slot->generation = 1;
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
void _skr_write_buffer(_skr_desc_writes_t* ref_writes, uint32_t binding, bool storage, VkBuffer buffer, uint64_t uid, uint32_t offset, uint32_t range) {
	if (ref_writes->write_ct >= _SKR_DESC_WRITES_MAX || ref_writes->buffer_ct >= _SKR_DESC_INFOS_MAX) return;
	assert((buffer == VK_NULL_HANDLE || uid != 0) && "Buffer without an identity, see _skr_uid_new");

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
	ref_writes->uids[ref_writes->write_ct] = uid;
	ref_writes->buffer_ct++;
	ref_writes->write_ct++;
}

void _skr_write_image(_skr_desc_writes_t* ref_writes, uint32_t binding, VkDescriptorType type, VkSampler sampler, VkImageView view, uint64_t uid, VkImageLayout layout) {
	if (ref_writes->write_ct >= _SKR_DESC_WRITES_MAX || ref_writes->image_ct >= _SKR_DESC_INFOS_MAX) return;
	assert((view == VK_NULL_HANDLE || uid != 0) && "View without an identity, see _skr_uid_new");

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
	ref_writes->uids[ref_writes->write_ct] = uid;
	ref_writes->image_ct++;
	ref_writes->write_ct++;
}

static inline uint64_t _skr_hash_word(uint64_t hash, uint64_t word) {
	hash ^= word;
	hash *= 0x9E3779B97F4A7C15ull;
	return hash ^ (hash >> 29);
}

// Keys on identities rather than handles, see _skr_uid_new. Never returns 0.
static uint64_t _skr_hash_writes(uint64_t layout_uid, const _skr_desc_writes_t* writes) {
	uint64_t hash = _skr_hash_word(0x9E3779B97F4A7C15ull, layout_uid);
	for (uint32_t i = 0; i < writes->write_ct; i++) {
		const VkWriteDescriptorSet* write = &writes->writes[i];
		hash = _skr_hash_word(hash, ((uint64_t)write->dstBinding << 32) | (uint32_t)write->descriptorType);
		hash = _skr_hash_word(hash, writes->uids[i]);
		if (write->pBufferInfo) {
			hash = _skr_hash_word(hash, write->pBufferInfo->offset);
			hash = _skr_hash_word(hash, write->pBufferInfo->range);
		} else if (write->pImageInfo) {
			hash = _skr_hash_word(hash, (uint64_t)write->pImageInfo->imageLayout);
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

bool _skr_bind_descriptors(VkCommandBuffer cmd, VkPipelineBindPoint bind_point, int32_t bind_start, VkPipelineLayout layout, _skr_desc_layout_t desc_layout, _skr_desc_writes_t* ref_writes) {
	if (ref_writes->write_ct == 0) return true;

	if (_skr_vk.has_push_descriptors) {
		vkCmdPushDescriptorSetKHR(cmd, bind_point, layout, 0, ref_writes->write_ct, ref_writes->writes);
		return true;
	}

	// One entry per bind slice, so a caller that varies its writes within one
	// material (mipgen's per-mip views) misses every time and churns a set per call
	_skr_desc_cache_t*       cache = &_skr_cmd_get_thread()->desc_cache;
	_skr_desc_cache_entry_t* entry = _skr_desc_cache_entry(cache, bind_start);
	if (entry == NULL) {
		skr_log(skr_log_critical, "Descriptor bind without a bind slice");
		return false;
	}

	uint64_t hash = _skr_hash_writes(desc_layout.uid, ref_writes);
	if (entry->hash != hash) {
		// The replaced set may still be bound in an in-flight buffer, so it
		// frees with this slot's fence, which orders after every earlier submit
		if (entry->set != VK_NULL_HANDLE)
			_skr_destroy_private_desc_set(entry->set, entry->pool);
		entry->set  = VK_NULL_HANDLE;
		entry->hash = 0;
		if (!_skr_desc_cache_alloc(cache, desc_layout.handle, &entry->set, &entry->pool))
			return false;
		for (uint32_t i = 0; i < ref_writes->write_ct; i++)
			ref_writes->writes[i].dstSet = entry->set;
		vkUpdateDescriptorSets(_skr_vk.device, ref_writes->write_ct, ref_writes->writes, 0, NULL);
		entry->hash = hash;
	}
	vkCmdBindDescriptorSets(cmd, bind_point, layout, 0, 1, &entry->set, ref_writes->dyn.count, ref_writes->dyn.offsets);
	return true;
}

void _skr_desc_cache_destroy(_skr_desc_cache_t* ref_cache) {
	for (uint32_t i = 0; i < _SKR_DESC_CACHE_MAX_CHUNKS; i++)
		_skr_free(ref_cache->chunks[i]);
	for (uint32_t i = 0; i < ref_cache->pool_count; i++)
		vkDestroyDescriptorPool(_skr_vk.device, ref_cache->pools[i], NULL);
	_skr_free(ref_cache->pools);
	_skr_free(ref_cache->pool_used);
	_skr_free(ref_cache->pool_no_room);
	*ref_cache = (_skr_desc_cache_t){0};
}

// A set freed back to its pool reopens that pool for allocation
void _skr_desc_cache_free(_skr_desc_cache_t* ref_cache, VkDescriptorSet set, uint8_t pool) {
	vkFreeDescriptorSets(_skr_vk.device, ref_cache->pools[pool], 1, &set);
	ref_cache->pool_used   [pool]--;
	ref_cache->pool_no_room[pool] = false;
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

// Other threads read ref_count at submit, so writes go through the atomic
static void _skr_ref_add(_skr_vk_thread_t* ref_pool, int32_t delta) {
	_skr_store_u32(&ref_pool->ref_count, ref_pool->ref_count + delta);
}

_skr_cmd_ctx_t _skr_cmd_acquire(void) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	if (pool->ref_count == 0)
		_skr_cmd_ring_begin(pool);

	_skr_ref_add(pool, 1);
	return (_skr_cmd_ctx_t){
		.cmd          = pool->cmd_ring[pool->cur].cmd,
		.const_ring   = &pool->const_ring,
		.storage_ring = &pool->storage_ring,
	};
}

///////////////////////////////////////////////////////////////////////////////

// Ends and submits the recording slot. Callers own ref_count and keep it
// above zero until this returns, see _skr_cmd_release.
static skr_future_t _skr_cmd_submit(_skr_vk_thread_t* ref_pool, const VkSemaphore* opt_waits, uint32_t wait_count, const VkSemaphore* opt_signals, uint32_t signal_count) {
	_skr_cmd_ring_slot_t* slot = &ref_pool->cmd_ring[ref_pool->cur];
	vkEndCommandBuffer(slot->cmd);

	assert(wait_count   <= SKR_MAX_SURFACES && "Wait count exceeds maximum surfaces");
	assert(signal_count <= SKR_MAX_SURFACES && "Signal count exceeds maximum surfaces");
	VkPipelineStageFlags wait_stages[SKR_MAX_SURFACES];
	for (uint32_t i = 0; i < wait_count; i++)
		wait_stages[i] = _SKR_ACQUIRE_WAIT_STAGE;

	mtx_lock(_skr_vk.graphics_queue_mutex);
	vkQueueSubmit(_skr_vk.graphics_queue, 1, &(VkSubmitInfo){
		.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount   = 1,
		.pCommandBuffers      = &slot->cmd,
		.waitSemaphoreCount   = wait_count,
		.pWaitSemaphores      = opt_waits,
		.pWaitDstStageMask    = wait_count > 0 ? wait_stages : NULL,
		.signalSemaphoreCount = signal_count,
		.pSignalSemaphores    = opt_signals,
	}, slot->fence);
	// Still under the mutex: this fence covers every earlier submission, and
	// main's open frame is the only buffer that can outlive one
	skr_future_t future    = { .slot = slot, .generation = slot->generation };
	uint32_t     main_idx  = _skr_load_u32(&_skr_vk.main_pool_idx);
	bool         may_stamp = ref_pool->thread_idx == main_idx || _skr_load_u32(&_skr_vk.thread_pools[main_idx].ref_count) == 0;
	_skr_destroy_block_t* done = _skr_destroy_retire(&ref_pool->shared_open, future, may_stamp);
	mtx_unlock(_skr_vk.graphics_queue_mutex);
	_skr_destroy_execute_blocks(done);

	return future;
}

// The slot most recently handed to the queue; its fence is null when none has been
static _skr_cmd_ring_slot_t* _skr_cmd_last_submitted(_skr_vk_thread_t* pool) {
	uint32_t idx = pool->ref_count > 0 ? (pool->cur + skr_MAX_COMMAND_RING - 1) % skr_MAX_COMMAND_RING : pool->cur;
	return &pool->cmd_ring[idx];
}

///////////////////////////////////////////////////////////////////////////////

void _skr_cmd_release(VkCommandBuffer buffer) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	assert(pool->ref_count > 0 && "Unbalanced acquire/release");
	assert(pool->cmd_ring[pool->cur].cmd == buffer && "Shouldn't release someone else's buffer!");

	// Submit before the count drops: a worker that sees main at zero stamps
	// against its own fence, which only covers what is already in the queue
	if (pool->ref_count == 1)
		_skr_cmd_submit(pool, NULL, 0, NULL, 0);
	_skr_ref_add(pool, -1);
}

///////////////////////////////////////////////////////////////////////////////

skr_future_t _skr_cmd_end_submit(const VkSemaphore* opt_waits, uint32_t wait_count, const VkSemaphore* opt_signals, uint32_t signal_count) {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	assert(pool->ref_count == 1 && "Unbalanced acquire/release - ref count should be 1");

	skr_future_t future = _skr_cmd_submit(pool, opt_waits, wait_count, opt_signals, signal_count);
	_skr_ref_add(pool, -1);
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

	// The recording slot if there is one, otherwise the last submitted
	_skr_cmd_ring_slot_t* target = &pool->cmd_ring[pool->cur];

	// Return invalid if no command has been submitted yet
	if (target->fence == VK_NULL_HANDLE) {
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

	_skr_cmd_ring_slot_t* slot = _skr_cmd_last_submitted(pool);
	if (slot->fence == VK_NULL_HANDLE) return -1;

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

// Invalid futures count as done. A generation mismatch means the slot moved
// on, which its owner only does after waiting the fence.
bool skr_future_check(const skr_future_t* future) {
	if (!future || !future->slot) return true;
	_skr_cmd_ring_slot_t* slot = (_skr_cmd_ring_slot_t*)future->slot;

	mtx_lock(_skr_vk.graphics_queue_mutex);
	bool done = slot->generation != future->generation
	         || vkGetFenceStatus(_skr_vk.device, slot->fence) == VK_SUCCESS;
	mtx_unlock(_skr_vk.graphics_queue_mutex);
	return done;
}

void skr_future_wait(const skr_future_t* future) {
	if (!future || !future->slot) return;
	_skr_cmd_ring_slot_t* slot = (_skr_cmd_ring_slot_t*)future->slot;

	// Registered under the mutex, so the owner either sees us before it
	// retires the generation, or we see the retired generation
	mtx_lock(_skr_vk.graphics_queue_mutex);
	bool done = slot->generation != future->generation;
	if (!done) _skr_add_u32(&slot->waiters, 1);
	mtx_unlock(_skr_vk.graphics_queue_mutex);
	if (done) return;

	vkWaitForFences(_skr_vk.device, 1, &slot->fence, VK_TRUE, UINT64_MAX);
	_skr_add_u32(&slot->waiters, -1);
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

	_skr_cmd_ring_slot_t* slot   = &pool->cmd_ring[pool->cur];
	skr_future_t          future = { .slot = slot, .generation = slot->generation };

	_skr_cmd_release(slot->cmd);

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
	if (pool->ref_count == 0) {
		return (skr_future_t){ .slot = NULL, .generation = 0 };
	}

	// Outstanding acquires keep their ref level and get the next slot
	skr_future_t result = _skr_cmd_submit(pool, NULL, 0, NULL, 0);
	_skr_cmd_ring_begin(pool);
	return result;
}
