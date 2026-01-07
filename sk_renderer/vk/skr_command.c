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

bool _skr_cmd_init() {
	memset(_skr_vk.thread_pools, 0, sizeof(_skr_vk.thread_pools));
	return true;
}

///////////////////////////////////////////////////////////////////////////////

void _skr_cmd_shutdown() {
	vkDeviceWaitIdle(_skr_vk.device);

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

			// Destroy bump allocators
			_skr_bump_alloc_destroy(&thread->cmd_ring[c].const_bump);
			_skr_bump_alloc_destroy(&thread->cmd_ring[c].storage_bump);

			if (thread->cmd_ring[c].fence != VK_NULL_HANDLE)
				vkDestroyFence(_skr_vk.device, thread->cmd_ring[c].fence, NULL);
			if (thread->cmd_ring[c].descriptor_pool != VK_NULL_HANDLE)
				vkDestroyDescriptorPool(_skr_vk.device, thread->cmd_ring[c].descriptor_pool, NULL);
		}

		if (thread->cmd_pool != VK_NULL_HANDLE)
			vkDestroyCommandPool(_skr_vk.device, thread->cmd_pool, NULL);

		*thread = (_skr_vk_thread_t){};
	}
	mtx_unlock(&_skr_vk.thread_pool_mutex);
}

///////////////////////////////////////////////////////////////////////////////

_skr_vk_thread_t* _skr_cmd_get_thread() {
	if (_skr_thread_idx >= 0) {
		return &_skr_vk.thread_pools[_skr_thread_idx];
	}
	return NULL;
}

///////////////////////////////////////////////////////////////////////////////

void skr_thread_init() {
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

	mtx_unlock(&_skr_vk.thread_pool_mutex);

	char name[64];
	snprintf(name, sizeof(name), "CommandPool_thr%d", thread_idx);
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_COMMAND_POOL, (uint64_t)thread.cmd_pool, name);

	return;
}

///////////////////////////////////////////////////////////////////////////////

void skr_thread_shutdown() {
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

		// Destroy bump allocators
		_skr_bump_alloc_destroy(&thread->cmd_ring[c].const_bump);
		_skr_bump_alloc_destroy(&thread->cmd_ring[c].storage_bump);

		if (thread->cmd_ring[c].fence != VK_NULL_HANDLE)
			vkDestroyFence(_skr_vk.device, thread->cmd_ring[c].fence, NULL);
		if (thread->cmd_ring[c].descriptor_pool != VK_NULL_HANDLE)
			vkDestroyDescriptorPool(_skr_vk.device, thread->cmd_ring[c].descriptor_pool, NULL);
	}

	// Destroy command pool
	if (thread->cmd_pool != VK_NULL_HANDLE)
		vkDestroyCommandPool(_skr_vk.device, thread->cmd_pool, NULL);

	skr_log(skr_log_info, "Thread #%d shutdown, marking as non-alive for reuse", _skr_thread_idx);

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
		_skr_destroy_list_clear  (&slot->destroy_list);

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
		vkCreateFence(_skr_vk.device, &(VkFenceCreateInfo){
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		}, NULL, &slot->fence);
		slot->destroy_list = _skr_destroy_list_create();
		_skr_bump_alloc_init(&slot->const_bump,   skr_buffer_type_constant, 256);  // 256-byte alignment for UBOs
		_skr_bump_alloc_init(&slot->storage_bump, skr_buffer_type_storage,  16);   // 16-byte alignment for storage

		// Create descriptor pool for non-push-descriptor fallback
		if (!_skr_vk.has_push_descriptors) {
			VkDescriptorPoolSize pool_sizes[] = {
				{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount = 1000 },
				{ .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          .descriptorCount = 1000 },
				{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1000 },
				{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         .descriptorCount = 1000 },
			};
			VkDescriptorPoolCreateInfo pool_info = {
				.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
				.maxSets       = 2000,
				.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
				.pPoolSizes    = pool_sizes,
			};
			VkResult vr = vkCreateDescriptorPool(_skr_vk.device, &pool_info, NULL, &slot->descriptor_pool);
			SKR_VK_CHECK_NRET(vr, "vkCreateDescriptorPool");
		}

		char name[64];
		snprintf(name,sizeof(name), "CommandBuffer_thr%u_%u", ref_pool->thread_idx, idx);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_COMMAND_BUFFER, (uint64_t)slot->cmd, name);

		snprintf(name,sizeof(name), "Command_Fence_thr%u_%u", ref_pool->thread_idx, idx);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_FENCE, (uint64_t)slot->fence, name);

		if (slot->descriptor_pool != VK_NULL_HANDLE) {
			snprintf(name,sizeof(name), "DescriptorPool_thr%u_%u", ref_pool->thread_idx, idx);
			_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_DESCRIPTOR_POOL, (uint64_t)slot->descriptor_pool, name);
		}
	} else {
		vkResetCommandBuffer(slot->cmd, 0);
		vkResetFences       (_skr_vk.device, 1, &slot->fence);
		// Reset descriptor pool when reusing command buffer slot
		if (slot->descriptor_pool != VK_NULL_HANDLE) {
			vkResetDescriptorPool(_skr_vk.device, slot->descriptor_pool, 0);
		}
		// Reset bump allocators - resizes main buffer if needed, clears overflow
		_skr_bump_alloc_reset(&slot->const_bump);
		_skr_bump_alloc_reset(&slot->storage_bump);
	}

	vkBeginCommandBuffer(slot->cmd, &(VkCommandBufferBeginInfo){
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	});

	return slot;
}

///////////////////////////////////////////////////////////////////////////////

// Helper to bind descriptors (handles push descriptors vs descriptor set allocation)
void _skr_bind_descriptors(VkCommandBuffer cmd, VkDescriptorPool pool, VkPipelineBindPoint bind_point, 
                            VkPipelineLayout layout, VkDescriptorSetLayout desc_layout, 
                            VkWriteDescriptorSet* writes, uint32_t write_count) {
	if (write_count == 0) return;

	if (_skr_vk.has_push_descriptors) {
		vkCmdPushDescriptorSetKHR(cmd, bind_point, layout, 0, write_count, writes);
	} else {
		// Fallback: allocate and bind descriptor set from command buffer's pool
		VkDescriptorSetAllocateInfo alloc_info = {
			.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool     = pool,
			.descriptorSetCount = 1,
			.pSetLayouts        = &desc_layout,
		};
		VkDescriptorSet desc_set;
		VkResult vr = vkAllocateDescriptorSets(_skr_vk.device, &alloc_info, &desc_set);
		if (vr == VK_SUCCESS) {
			for (uint32_t i = 0; i < write_count; i++) {
				writes[i].dstSet = desc_set;
			}
			vkUpdateDescriptorSets(_skr_vk.device, write_count, writes, 0, NULL);
			vkCmdBindDescriptorSets(cmd, bind_point, layout, 0, 1, &desc_set, 0, NULL);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

_skr_cmd_ctx_t _skr_cmd_begin() {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);
	assert(pool->ref_count == 0 && "Ref count should be 0 at batch start");

	return _skr_cmd_acquire();
}

///////////////////////////////////////////////////////////////////////////////

bool _skr_cmd_try_get_active(_skr_cmd_ctx_t* out_ctx) {
	*out_ctx = (_skr_cmd_ctx_t){};

	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	if (pool->active_cmd) {
		*out_ctx = (_skr_cmd_ctx_t){
			.cmd             = pool->active_cmd->cmd,
			.descriptor_pool = pool->active_cmd->descriptor_pool,
			.destroy_list    = &pool->active_cmd->destroy_list,
			.const_bump      = &pool->active_cmd->const_bump,
			.storage_bump    = &pool->active_cmd->storage_bump,
		};
		return true;
	}
	return false;
}

///////////////////////////////////////////////////////////////////////////////

_skr_cmd_ctx_t _skr_cmd_acquire() {
	_skr_vk_thread_t* pool = _skr_cmd_get_thread();
	assert(pool);

	if (pool->ref_count == 0)
		pool->active_cmd = _skr_cmd_ring_begin(pool);

	pool->ref_count++;
	return (_skr_cmd_ctx_t){
		.cmd             = pool->active_cmd->cmd,
		.descriptor_pool = pool->active_cmd->descriptor_pool,
		.destroy_list    = &pool->active_cmd->destroy_list,
		.const_bump      = &pool->active_cmd->const_bump,
		.storage_bump    = &pool->active_cmd->storage_bump,
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

VkCommandBuffer _skr_cmd_end() {
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

skr_future_t skr_future_get() {
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

void skr_cmd_begin() {
	_skr_cmd_acquire();
}

skr_future_t skr_cmd_end() {
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