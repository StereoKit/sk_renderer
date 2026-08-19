// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"
#include "skr_conversions.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////////////////////////////////

static uint32_t _skr_find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
		if ((type_filter & (1 << i)) &&
		    (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	skr_log(skr_log_critical, "Failed to find suitable memory type");
	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Buffer creation and destruction
///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_buffer_create(const void* opt_data, uint32_t size_count, uint32_t size_stride,
                            skr_buffer_type_ type, skr_use_ use, skr_buffer_t* out_buffer) {
	if (!out_buffer) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_buffer = (skr_buffer_t){0};

	// Validate inputs
	if (size_count == 0 || size_stride == 0) {
		return skr_err_invalid_parameter;
	}

	out_buffer->size = size_count * size_stride;
	out_buffer->type = type;
	out_buffer->use  = use;

	VkBufferUsageFlags usage = _skr_to_vk_buffer_usage(type);

	// Add transfer dst for initial data upload (unless dynamic)
	if (opt_data != NULL && !(use & skr_use_dynamic)) {
		usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	}

	// Create buffer
	VkResult vr = vkCreateBuffer(_skr_vk.device, &(VkBufferCreateInfo){
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = out_buffer->size,
		.usage       = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	}, NULL, &out_buffer->buffer);
	SKR_VK_CHECK_RET(vr, "vkCreateBuffer", skr_err_device_error);

	// Allocate memory
	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(_skr_vk.device, out_buffer->buffer, &mem_requirements);

	VkMemoryPropertyFlags mem_properties = (use & skr_use_dynamic)
		? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		: VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	vr = vkAllocateMemory(_skr_vk.device, &(VkMemoryAllocateInfo){
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = mem_requirements.size,
		.memoryTypeIndex = _skr_find_memory_type(_skr_vk.physical_device, mem_requirements.memoryTypeBits, mem_properties),
	}, NULL, &out_buffer->memory);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkAllocateMemory");
		vkDestroyBuffer(_skr_vk.device, out_buffer->buffer, NULL);
		*out_buffer = (skr_buffer_t){0};
		return skr_err_out_of_memory;
	}

	vkBindBufferMemory(_skr_vk.device, out_buffer->buffer, out_buffer->memory, 0);

	// Upload initial data
	if (opt_data != NULL) {
		if (use & skr_use_dynamic) {
			// Direct map and copy for dynamic buffers
			void* mapped;
			vkMapMemory(_skr_vk.device, out_buffer->memory, 0, out_buffer->size, 0, &mapped);
			memcpy(mapped, opt_data, out_buffer->size);
			vkUnmapMemory(_skr_vk.device, out_buffer->memory);
		} else {
			// Use staging buffer for static buffers
			VkBuffer staging_buffer;
			vr = vkCreateBuffer(_skr_vk.device, &(VkBufferCreateInfo){
				.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size        = out_buffer->size,
				.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			}, NULL, &staging_buffer);
			if (vr != VK_SUCCESS) {
				SKR_VK_CHECK_NRET(vr, "vkCreateBuffer");
				vkDestroyBuffer(_skr_vk.device, out_buffer->buffer, NULL);
				vkFreeMemory(_skr_vk.device, out_buffer->memory, NULL);
				*out_buffer = (skr_buffer_t){0};
				return skr_err_device_error;
			}

			VkMemoryRequirements staging_mem_req;
			vkGetBufferMemoryRequirements(_skr_vk.device, staging_buffer, &staging_mem_req);

			VkDeviceMemory staging_memory;
			vr = vkAllocateMemory(_skr_vk.device, &(VkMemoryAllocateInfo){
				.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
				.allocationSize  = staging_mem_req.size,
				.memoryTypeIndex = _skr_find_memory_type(_skr_vk.physical_device, staging_mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
			}, NULL, &staging_memory);
			if (vr != VK_SUCCESS) {
				SKR_VK_CHECK_NRET(vr, "vkAllocateMemory");
				vkDestroyBuffer(_skr_vk.device, staging_buffer, NULL);
				vkDestroyBuffer(_skr_vk.device, out_buffer->buffer, NULL);
				vkFreeMemory(_skr_vk.device, out_buffer->memory, NULL);
				*out_buffer = (skr_buffer_t){0};
				return skr_err_out_of_memory;
			}
			vkBindBufferMemory(_skr_vk.device, staging_buffer, staging_memory, 0);

			// Copy data to staging buffer
			void* mapped;
			vkMapMemory(_skr_vk.device, staging_memory, 0, out_buffer->size, 0, &mapped);
			memcpy(mapped, opt_data, out_buffer->size);
			vkUnmapMemory(_skr_vk.device, staging_memory);

			_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

			vkCmdCopyBuffer(ctx.cmd, staging_buffer, out_buffer->buffer, 1, &(VkBufferCopy){
				.size = out_buffer->size,
			});

			// The copy carries no implicit dependency with later reads. Those reads
			// may land in this same command buffer (_skr_cmd_acquire hands back the
			// active one, so creation can nest inside an open context), or in a
			// later submission on this queue — submissions overlap freely. Either
			// way the barrier below is what orders them.
			VkAccessFlags        dst_access = 0;
			VkPipelineStageFlags dst_stage  = 0;
			if (type & skr_buffer_type_vertex) {
				dst_access |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
				dst_stage  |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			}
			if (type & skr_buffer_type_index) {
				dst_access |= VK_ACCESS_INDEX_READ_BIT;
				dst_stage  |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			}
			if (type & skr_buffer_type_constant) {
				dst_access |= VK_ACCESS_UNIFORM_READ_BIT;
				dst_stage  |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			}
			if (type & skr_buffer_type_storage) {
				// SHADER_WRITE covers the WAW against a compute pass that writes the
				// buffer on its first dispatch. Storage buffers also double as
				// indirect args (skr_compute_execute_indirect), so cover that read.
				dst_access |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
				dst_stage  |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
			}
			if (dst_stage == 0) {
				// A skr_buffer_type_ with no mapping above. Fail loud in debug, and
				// stay conservative in release — dropping the barrier here would
				// surface as intermittent corruption, not a clean failure.
				assert(false && "skr_buffer_type_ has no barrier mapping, add one above");
				dst_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
				dst_stage  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
			}
			VkBufferMemoryBarrier buffer_barrier = {
				.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
				.dstAccessMask       = dst_access,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.buffer              = out_buffer->buffer,
				.offset              = 0,
				.size                = out_buffer->size,
			};
			vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, dst_stage, 0, 0, NULL, 1, &buffer_barrier, 0, NULL);

			// Destroy list is LIFO — push memory before buffer so vkFreeMemory
			// runs after vkDestroyBuffer (VUID-vkFreeMemory-memory-00677).
			_skr_cmd_destroy_memory(ctx.destroy_list, staging_memory);
			_skr_cmd_destroy_buffer(ctx.destroy_list, staging_buffer);
			_skr_cmd_release       (ctx.cmd);
		}
	}

	// Keep dynamic buffers mapped
	if (use & skr_use_dynamic) {
		vkMapMemory(_skr_vk.device, out_buffer->memory, 0, out_buffer->size, 0, &out_buffer->mapped);
	}

	return skr_err_success;
}

bool skr_buffer_is_valid(const skr_buffer_t* buffer) {
	return buffer && buffer->buffer != VK_NULL_HANDLE;
}

// Helper to allocate a new ring slot for dynamic buffer updates
static bool _skr_buffer_alloc_ring_slot(skr_buffer_t* ref_buffer, uint8_t slot_idx) {
	VkBufferUsageFlags usage = _skr_to_vk_buffer_usage(ref_buffer->type);

	VkResult vr = vkCreateBuffer(_skr_vk.device, &(VkBufferCreateInfo){
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = ref_buffer->size,
		.usage       = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	}, NULL, &ref_buffer->_ring[slot_idx].buffer);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkCreateBuffer (ring slot)");
		return false;
	}

	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(_skr_vk.device, ref_buffer->_ring[slot_idx].buffer, &mem_requirements);

	vr = vkAllocateMemory(_skr_vk.device, &(VkMemoryAllocateInfo){
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = mem_requirements.size,
		.memoryTypeIndex = _skr_find_memory_type(_skr_vk.physical_device, mem_requirements.memoryTypeBits,
		                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
	}, NULL, &ref_buffer->_ring[slot_idx].memory);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkAllocateMemory (ring slot)");
		vkDestroyBuffer(_skr_vk.device, ref_buffer->_ring[slot_idx].buffer, NULL);
		ref_buffer->_ring[slot_idx].buffer = VK_NULL_HANDLE;
		return false;
	}

	vkBindBufferMemory(_skr_vk.device, ref_buffer->_ring[slot_idx].buffer, ref_buffer->_ring[slot_idx].memory, 0);
	vkMapMemory(_skr_vk.device, ref_buffer->_ring[slot_idx].memory, 0, ref_buffer->size, 0, &ref_buffer->_ring[slot_idx].mapped);

	return true;
}

void skr_buffer_set(skr_buffer_t* ref_buffer, const void* data, uint32_t size_bytes) {
	if (!ref_buffer || !data) return;

	if (!(ref_buffer->use & skr_use_dynamic)) {
		skr_log(skr_log_critical, "skr_buffer_set only supports dynamic buffers");
		return;
	}

	uint32_t copy_size = size_bytes < ref_buffer->size ? size_bytes : ref_buffer->size;

	// First update: initialize ring buffer system
	if (ref_buffer->_ring_count == 0) {
		// Migrate existing buffer to ring[0]
		ref_buffer->_ring[0].buffer = ref_buffer->buffer;
		ref_buffer->_ring[0].memory = ref_buffer->memory;
		ref_buffer->_ring[0].mapped = ref_buffer->mapped;
		ref_buffer->_ring_count     = 1;
		ref_buffer->_ring_index     = 0;

		// Allocate ring[1] for this write
		if (!_skr_buffer_alloc_ring_slot(ref_buffer, 1)) {
			// Fallback: write directly (unsafe but better than crash)
			memcpy(ref_buffer->mapped, data, copy_size);
			return;
		}
		ref_buffer->_ring_count = 2;

		// Write to ring[1] and make it current
		memcpy(ref_buffer->_ring[1].mapped, data, copy_size);
		ref_buffer->_ring_index = 1;
		ref_buffer->buffer      = ref_buffer->_ring[1].buffer;
		ref_buffer->memory      = ref_buffer->_ring[1].memory;
		ref_buffer->mapped      = ref_buffer->_ring[1].mapped;
		return;
	}

	// Subsequent updates: advance to next slot in ring
	uint8_t next_idx = (ref_buffer->_ring_index + 1) % SKR_DYNAMIC_BUFFER_COPIES;

	// Allocate slot if not yet allocated
	if (next_idx >= ref_buffer->_ring_count) {
		if (!_skr_buffer_alloc_ring_slot(ref_buffer, next_idx)) {
			// Fallback: write to current slot (unsafe but better than crash)
			memcpy(ref_buffer->mapped, data, copy_size);
			return;
		}
		ref_buffer->_ring_count = next_idx + 1;
	}

	// Write to the new slot and make it current
	memcpy(ref_buffer->_ring[next_idx].mapped, data, copy_size);
	ref_buffer->_ring_index = next_idx;
	ref_buffer->buffer      = ref_buffer->_ring[next_idx].buffer;
	ref_buffer->memory      = ref_buffer->_ring[next_idx].memory;
	ref_buffer->mapped      = ref_buffer->_ring[next_idx].mapped;
}

void skr_buffer_get(const skr_buffer_t *buffer, void *ref_buffer, uint32_t buffer_size) {
	if (!buffer || !ref_buffer) return;

	if (!(buffer->use & skr_use_dynamic)) {
		skr_log(skr_log_critical, "skr_buffer_get only supports dynamic buffers");
		return;
	}

	if (!buffer->mapped) {
		skr_log(skr_log_critical, "Dynamic buffer is not mapped");
		return;
	}

	// Copy min of requested size and actual buffer size
	uint32_t copy_size = buffer_size < buffer->size ? buffer_size : buffer->size;
	memcpy(ref_buffer, buffer->mapped, copy_size);
}

uint32_t skr_buffer_get_size(const skr_buffer_t* buffer) {
	return buffer ? buffer->size : 0;
}

// Internal state for readback: the host-visible staging snapshot
typedef struct _skr_buffer_readback_internal_t {
	VkBuffer       staging_buffer;
	VkDeviceMemory staging_memory;
} _skr_buffer_readback_internal_t;

skr_err_ skr_buffer_readback(const skr_buffer_t* buffer, skr_buffer_readback_t* out_readback) {
	if (!buffer || !out_readback) return skr_err_invalid_parameter;
	memset(out_readback, 0, sizeof(*out_readback));

	// Storage is the only type both backends can copy out of; see the header
	if (!(buffer->type & skr_buffer_type_storage)) {
		skr_log(skr_log_critical, "skr_buffer_readback needs a storage-type buffer");
		return skr_err_unsupported;
	}

	VkBuffer staging_buffer;
	VkResult vr = vkCreateBuffer(_skr_vk.device, &(VkBufferCreateInfo){
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = buffer->size,
		.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	}, NULL, &staging_buffer);
	SKR_VK_CHECK_RET(vr, "vkCreateBuffer", skr_err_device_error);

	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(_skr_vk.device, staging_buffer, &mem_requirements);

	VkDeviceMemory staging_memory;
	vr = vkAllocateMemory(_skr_vk.device, &(VkMemoryAllocateInfo){
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = mem_requirements.size,
		.memoryTypeIndex = _skr_find_memory_type(_skr_vk.physical_device, mem_requirements.memoryTypeBits,
		                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
	}, NULL, &staging_memory);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkAllocateMemory");
		vkDestroyBuffer(_skr_vk.device, staging_buffer, NULL);
		return skr_err_out_of_memory;
	}
	vr = vkBindBufferMemory(_skr_vk.device, staging_buffer, staging_memory, 0);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkBindBufferMemory");
		vkFreeMemory   (_skr_vk.device, staging_memory, NULL);
		vkDestroyBuffer(_skr_vk.device, staging_buffer, NULL);
		return skr_err_device_error;
	}

	void* mapped;
	vr = vkMapMemory(_skr_vk.device, staging_memory, 0, buffer->size, 0, &mapped);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkMapMemory");
		vkFreeMemory   (_skr_vk.device, staging_memory, NULL);
		vkDestroyBuffer(_skr_vk.device, staging_buffer, NULL);
		return skr_err_device_error;
	}
	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

	VkPipelineStageFlags shader_stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	// Order the copy after shader writes and the initial-data upload
	vkCmdPipelineBarrier(ctx.cmd, shader_stages | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &(VkBufferMemoryBarrier){
		.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer              = buffer->buffer,
		.offset              = 0,
		.size                = buffer->size,
	}, 0, NULL);

	vkCmdCopyBuffer(ctx.cmd, buffer->buffer, staging_buffer, 1, &(VkBufferCopy){ .size = buffer->size });

	// And later shader writes after the copy (WAR with the next dispatch)
	vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, shader_stages, 0, 0, NULL, 1, &(VkBufferMemoryBarrier){
		.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
		.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer              = buffer->buffer,
		.offset              = 0,
		.size                = buffer->size,
	}, 0, NULL);

	// Future before release, matching skr_tex_readback: release submits this
	// command buffer when the readback isn't nested inside a larger batch
	skr_future_t future = skr_future_get();
	_skr_cmd_release(ctx.cmd);

	_skr_buffer_readback_internal_t* internal = (_skr_buffer_readback_internal_t*)_skr_malloc(sizeof(_skr_buffer_readback_internal_t));
	internal->staging_buffer = staging_buffer;
	internal->staging_memory = staging_memory;

	out_readback->data      = mapped;
	out_readback->size      = buffer->size;
	out_readback->future    = future;
	out_readback->_internal = internal;
	return skr_err_success;
}

void skr_buffer_readback_destroy(skr_buffer_readback_t* ref_readback) {
	if (!ref_readback || !ref_readback->_internal) return;
	_skr_buffer_readback_internal_t* internal = (_skr_buffer_readback_internal_t*)ref_readback->_internal;

	// No wait: the copy may sit in a command buffer this thread has yet to
	// submit, and blocking on that here deadlocks. Deferred destruction
	// releases the staging once the GPU is done with it (LIFO: memory first).
	vkUnmapMemory(_skr_vk.device, internal->staging_memory);
	_skr_cmd_destroy_memory(NULL, internal->staging_memory);
	_skr_cmd_destroy_buffer(NULL, internal->staging_buffer);
	_skr_free(internal);

	*ref_readback = (skr_buffer_readback_t){0};
}

void skr_buffer_set_name(skr_buffer_t* ref_buffer, const char* name) {
	if (!ref_buffer || ref_buffer->buffer == VK_NULL_HANDLE) return;
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_BUFFER, (uint64_t)ref_buffer->buffer, name);
}

void skr_buffer_destroy(skr_buffer_t* ref_buffer) {
	if (!ref_buffer || ref_buffer->buffer == VK_NULL_HANDLE) return;

	// Destroy list is LIFO — push memory before buffer so vkFreeMemory runs
	// after vkDestroyBuffer (VUID-vkFreeMemory-memory-00677).
	if (ref_buffer->_ring_count > 0) {
		// Ring buffer mode: destroy all allocated ring slots
		for (uint8_t i = 0; i < ref_buffer->_ring_count; i++) {
			if (ref_buffer->_ring[i].mapped) {
				vkUnmapMemory(_skr_vk.device, ref_buffer->_ring[i].memory);
			}
			_skr_cmd_destroy_memory(NULL, ref_buffer->_ring[i].memory);
			_skr_cmd_destroy_buffer(NULL, ref_buffer->_ring[i].buffer);
		}
	} else {
		// Single buffer mode: destroy top-level fields
		if (ref_buffer->mapped) {
			vkUnmapMemory(_skr_vk.device, ref_buffer->memory);
		}
		_skr_cmd_destroy_memory(NULL, ref_buffer->memory);
		_skr_cmd_destroy_buffer(NULL, ref_buffer->buffer);
	}

	*ref_buffer = (skr_buffer_t){0};
}

///////////////////////////////////////////////////////////////////////////////
// Bump Allocator
///////////////////////////////////////////////////////////////////////////////

void _skr_bump_alloc_init(skr_bump_alloc_t* ref_alloc, skr_buffer_type_ type, uint32_t alignment) {
	*ref_alloc = (skr_bump_alloc_t){
		.buffer_type    = type,
		.alignment      = alignment > 0 ? alignment : 1,
		.high_water_mark = 0,
	};
}

void _skr_bump_alloc_destroy(skr_bump_alloc_t* ref_alloc) {
	if (!ref_alloc) return;

	// Destroy main buffer
	if (ref_alloc->main_valid) {
		skr_buffer_destroy(&ref_alloc->main_buffer);
	}

	// Destroy all overflow buffers (individually allocated)
	for (uint32_t i = 0; i < ref_alloc->overflow_count; i++) {
		skr_buffer_destroy(ref_alloc->overflow[i]);
		_skr_free(ref_alloc->overflow[i]);
	}
	_skr_free(ref_alloc->overflow);

	*ref_alloc = (skr_bump_alloc_t){0};
}

void _skr_bump_alloc_reset(skr_bump_alloc_t* ref_alloc) {
	if (!ref_alloc) return;

	// Resize main buffer if high-water mark exceeds current capacity
	uint32_t main_capacity = ref_alloc->main_valid ? ref_alloc->main_buffer.size : 0;
	if (ref_alloc->high_water_mark > main_capacity) {
		// Destroy old main buffer
		if (ref_alloc->main_valid) {
			skr_buffer_destroy(&ref_alloc->main_buffer);
			ref_alloc->main_valid = false;
		}

		// Create new buffer sized to high-water mark (with some headroom)
		uint32_t new_size = ref_alloc->high_water_mark + (ref_alloc->high_water_mark / 4);  // +25% headroom
		if (new_size < 4096) new_size = 4096;  // Minimum 4KB

		skr_buffer_create(NULL, new_size, 1, ref_alloc->buffer_type, skr_use_dynamic, &ref_alloc->main_buffer);
		ref_alloc->main_valid = true;
	}

	// Reset main buffer offset
	ref_alloc->main_used = 0;

	// Destroy overflow buffers from previous frame (GPU is done with them now)
	for (uint32_t i = 0; i < ref_alloc->overflow_count; i++) {
		skr_buffer_destroy(ref_alloc->overflow[i]);
		_skr_free(ref_alloc->overflow[i]);
	}
	ref_alloc->overflow_count = 0;

	// Reset high-water mark for this frame
	ref_alloc->high_water_mark = 0;
}

skr_bump_result_t _skr_bump_alloc_write(skr_bump_alloc_t* ref_alloc, const void* data, uint32_t size) {
	skr_bump_result_t result = { .buffer = NULL, .offset = 0 };
	if (!ref_alloc || !data || size == 0) return result;

	// Align the allocation
	uint32_t aligned_offset = (ref_alloc->main_used + ref_alloc->alignment - 1) & ~(ref_alloc->alignment - 1);
	uint32_t main_capacity = ref_alloc->main_valid ? ref_alloc->main_buffer.size : 0;

	// Try to allocate from main buffer
	if (ref_alloc->main_valid && aligned_offset + size <= main_capacity) {
		// Write to main buffer
		memcpy((uint8_t*)ref_alloc->main_buffer.mapped + aligned_offset, data, size);
		ref_alloc->main_used = aligned_offset + size;

		// Track high-water mark
		if (ref_alloc->main_used > ref_alloc->high_water_mark) {
			ref_alloc->high_water_mark = ref_alloc->main_used;
		}

		result.buffer = &ref_alloc->main_buffer;
		result.offset = aligned_offset;
		return result;
	}

	// Main buffer is full or doesn't exist - create overflow buffer
	// Grow pointer array if needed (only the pointer array can realloc,
	// individual buffers are stable so previous results stay valid)
	if (ref_alloc->overflow_count >= ref_alloc->overflow_capacity) {
		uint32_t new_cap = ref_alloc->overflow_capacity == 0 ? 4 : ref_alloc->overflow_capacity * 2;
		skr_buffer_t** new_overflow = _skr_realloc(ref_alloc->overflow, new_cap * sizeof(skr_buffer_t*));
		if (!new_overflow) {
			skr_log(skr_log_critical, "Failed to grow bump allocator overflow array");
			return result;
		}
		ref_alloc->overflow = new_overflow;
		ref_alloc->overflow_capacity = new_cap;
	}

	// Allocate individual overflow buffer (stable address, never moved by realloc)
	skr_buffer_t* overflow = _skr_malloc(sizeof(skr_buffer_t));
	if (!overflow) {
		skr_log(skr_log_critical, "Failed to allocate bump overflow buffer");
		return result;
	}
	*overflow = (skr_buffer_t){0};

	skr_buffer_create(data, size, 1, ref_alloc->buffer_type, skr_use_dynamic, overflow);
	ref_alloc->overflow[ref_alloc->overflow_count] = overflow;
	ref_alloc->overflow_count++;

	// Track total usage in high-water mark (main + all overflow buffers)
	uint32_t overflow_total = 0;
	for (uint32_t i = 0; i < ref_alloc->overflow_count; i++) {
		overflow_total += ref_alloc->overflow[i]->size;
	}
	uint32_t total_used = ref_alloc->main_used + overflow_total;
	if (total_used > ref_alloc->high_water_mark) {
		ref_alloc->high_water_mark = total_used;
	}

	result.buffer = overflow;
	result.offset = 0;
	return result;
}
