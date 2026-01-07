// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"
#include "skr_conversions.h"

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
	*out_buffer = (skr_buffer_t){};

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
		*out_buffer = (skr_buffer_t){};
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
				*out_buffer = (skr_buffer_t){};
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
				*out_buffer = (skr_buffer_t){};
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

			_skr_cmd_destroy_buffer(ctx.destroy_list, staging_buffer);
			_skr_cmd_destroy_memory(ctx.destroy_list, staging_memory);
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

void skr_buffer_set(skr_buffer_t* ref_buffer, const void* data, uint32_t size_bytes) {
	if (!ref_buffer || !data) return;

	if ((ref_buffer->use & skr_use_dynamic) && ref_buffer->mapped) {
		memcpy(ref_buffer->mapped, data, size_bytes < ref_buffer->size ? size_bytes : ref_buffer->size);
	} else {
		skr_log(skr_log_critical, "skr_buffer_set only supports dynamic buffers");
	}
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

void skr_buffer_set_name(skr_buffer_t* ref_buffer, const char* name) {
	if (!ref_buffer || ref_buffer->buffer == VK_NULL_HANDLE) return;
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_BUFFER, (uint64_t)ref_buffer->buffer, name);
}

void skr_buffer_destroy(skr_buffer_t* ref_buffer) {
	if (!ref_buffer || ref_buffer->buffer == VK_NULL_HANDLE) return;

	if (ref_buffer->mapped) {
		vkUnmapMemory(_skr_vk.device, ref_buffer->memory);
		ref_buffer->mapped = NULL;
	}

	_skr_cmd_destroy_buffer(NULL, ref_buffer->buffer);
	_skr_cmd_destroy_memory(NULL, ref_buffer->memory);

	*ref_buffer = (skr_buffer_t){};
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

	// Destroy all overflow buffers
	for (uint32_t i = 0; i < ref_alloc->overflow_count; i++) {
		skr_buffer_destroy(&ref_alloc->overflow[i]);
	}
	_skr_free(ref_alloc->overflow);

	*ref_alloc = (skr_bump_alloc_t){};
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
		skr_buffer_destroy(&ref_alloc->overflow[i]);
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
	// Grow overflow array if needed
	if (ref_alloc->overflow_count >= ref_alloc->overflow_capacity) {
		uint32_t new_cap = ref_alloc->overflow_capacity == 0 ? 4 : ref_alloc->overflow_capacity * 2;
		skr_buffer_t* new_overflow = _skr_realloc(ref_alloc->overflow, new_cap * sizeof(skr_buffer_t));
		if (!new_overflow) {
			skr_log(skr_log_critical, "Failed to grow bump allocator overflow array");
			return result;
		}
		ref_alloc->overflow = new_overflow;
		ref_alloc->overflow_capacity = new_cap;
	}

	// Create overflow buffer for this allocation
	skr_buffer_t* overflow = &ref_alloc->overflow[ref_alloc->overflow_count];
	*overflow = (skr_buffer_t){};

	skr_buffer_create(data, size, 1, ref_alloc->buffer_type, skr_use_dynamic, overflow);
	ref_alloc->overflow_count++;

	// Track total usage in high-water mark (main + all overflow buffers)
	uint32_t overflow_total = 0;
	for (uint32_t i = 0; i < ref_alloc->overflow_count; i++) {
		overflow_total += ref_alloc->overflow[i].size;
	}
	uint32_t total_used = ref_alloc->main_used + overflow_total;
	if (total_used > ref_alloc->high_water_mark) {
		ref_alloc->high_water_mark = total_used;
	}

	result.buffer = overflow;
	result.offset = 0;
	return result;
}
