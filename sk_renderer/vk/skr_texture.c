// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "sk_renderer.h"
#include "_sk_renderer.h"

#include "skr_vulkan.h"
#include "skr_conversions.h"
#include "skr_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

///////////////////////////////////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////////////////////////////////

// Find memory type index with required properties, returns UINT32_MAX if not found
static uint32_t _skr_find_memory_type(VkPhysicalDevice phys_device, VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags required_props) {
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_properties);

	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
		if ((mem_requirements.memoryTypeBits & (1 << i)) &&
		    (mem_properties.memoryTypes[i].propertyFlags & required_props) == required_props) {
			return i;
		}
	}
	return UINT32_MAX;
}

// Allocate device memory for an image, trying lazily-allocated first for transient attachments
static VkDeviceMemory _skr_allocate_image_memory(VkDevice device, VkPhysicalDevice phys_device, VkImage image, bool is_transient_attachment, VkDeviceMemory* out_memory) {
	VkMemoryRequirements mem_requirements;
	vkGetImageMemoryRequirements(device, image, &mem_requirements);

	uint32_t memory_type_index = UINT32_MAX;

	// For transient MSAA attachments, prefer lazily allocated memory
	if (is_transient_attachment) {
		memory_type_index = _skr_find_memory_type(phys_device, mem_requirements, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
	}

	// Fallback to device local memory
	if (memory_type_index == UINT32_MAX) {
		memory_type_index = _skr_find_memory_type(phys_device, mem_requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	}

	if (memory_type_index == UINT32_MAX) {
		return VK_NULL_HANDLE;
	}

	VkMemoryAllocateInfo alloc_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = mem_requirements.size,
		.memoryTypeIndex = memory_type_index,
	};

	VkResult vr = vkAllocateMemory(device, &alloc_info, NULL, out_memory);
	SKR_VK_CHECK_RET(vr, "vkAllocateMemory", VK_NULL_HANDLE);

	return *out_memory;
}

// Create staging buffer and memory for texture uploads
typedef struct {
	VkBuffer       buffer;
	VkDeviceMemory memory;
	void*          mapped_data;
	bool           valid;
} staging_buffer_t;

static staging_buffer_t _skr_create_staging_buffer(VkDevice device, VkPhysicalDevice phys_device, VkDeviceSize size) {
	staging_buffer_t result = {0};

	VkBufferCreateInfo buffer_info = {
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = size,
		.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	VkResult vr = vkCreateBuffer(device, &buffer_info, NULL, &result.buffer);
	SKR_VK_CHECK_RET(vr, "vkCreateBuffer", result);

	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(device, result.buffer, &mem_requirements);

	uint32_t memory_type_index = _skr_find_memory_type(phys_device, mem_requirements,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	if (memory_type_index == UINT32_MAX) {
		vkDestroyBuffer(device, result.buffer, NULL);
		return result;
	}

	VkMemoryAllocateInfo alloc_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = mem_requirements.size,
		.memoryTypeIndex = memory_type_index,
	};

	vr = vkAllocateMemory(device, &alloc_info, NULL, &result.memory);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkAllocateMemory");
		vkDestroyBuffer(device, result.buffer, NULL);
		return result;
	}

	vkBindBufferMemory(device, result.buffer, result.memory, 0);

	vr = vkMapMemory(device, result.memory, 0, size, 0, &result.mapped_data);
	if (vr != VK_SUCCESS) {
		SKR_VK_CHECK_NRET(vr, "vkMapMemory");
		vkFreeMemory   (device, result.memory, NULL);
		vkDestroyBuffer(device, result.buffer, NULL);
		return result;
	}

	result.valid = true;
	return result;
}

// Transition image layout with a pipeline barrier (low-level helper)
static void _skr_transition_image_layout(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect_mask,
                                          uint32_t base_mip, uint32_t mip_count, uint32_t layer_count,
                                          VkImageLayout old_layout, VkImageLayout new_layout,
                                          VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                                          VkAccessFlags src_access, VkAccessFlags dst_access) {
	VkImageMemoryBarrier barrier = {
		.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.oldLayout           = old_layout,
		.newLayout           = new_layout,
		.srcAccessMask       = src_access,
		.dstAccessMask       = dst_access,
		.image               = image,
		.subresourceRange    = {
			.aspectMask     = aspect_mask,
			.baseMipLevel   = base_mip,
			.levelCount     = mip_count,
			.baseArrayLayer = 0,
			.layerCount     = layer_count,
		},
	};

	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &barrier);
}

// Helper: Convert layout to typical source stage flags
static VkPipelineStageFlags _layout_to_src_stage(VkImageLayout layout) {
	switch (layout) {
		case VK_IMAGE_LAYOUT_UNDEFINED:
			return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			return VK_PIPELINE_STAGE_TRANSFER_BIT;
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
			return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			return VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		case VK_IMAGE_LAYOUT_GENERAL:
			return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
			return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		default:
			return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	}
}

// Helper: Convert layout to typical access flags
static VkAccessFlags _layout_to_access_flags(VkImageLayout layout) {
	switch (layout) {
		case VK_IMAGE_LAYOUT_UNDEFINED:
			return 0;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			return VK_ACCESS_TRANSFER_READ_BIT;
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			return VK_ACCESS_TRANSFER_WRITE_BIT;
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
			return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			return VK_ACCESS_SHADER_READ_BIT;
		case VK_IMAGE_LAYOUT_GENERAL:
			return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
			return 0;
		default:
			return 0;
	}
}

#ifdef SKR_DEBUG
// Helper: Layout to string for debug logging
static const char* _layout_to_string(VkImageLayout layout) {
	switch (layout) {
		case VK_IMAGE_LAYOUT_UNDEFINED: return "UNDEFINED";
		case VK_IMAGE_LAYOUT_GENERAL: return "GENERAL";
		case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return "COLOR_ATTACHMENT";
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return "DEPTH_STENCIL_ATTACHMENT";
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: return "DEPTH_STENCIL_READ_ONLY";
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return "SHADER_READ_ONLY";
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return "TRANSFER_SRC";
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return "TRANSFER_DST";
		case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: return "PRESENT_SRC";
		default: return "UNKNOWN";
	}
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Automatic Layout Transition System
///////////////////////////////////////////////////////////////////////////////

// Check if texture needs transition for given type (without requiring command buffer)
bool _skr_tex_needs_transition(const skr_tex_t* tex, uint8_t type) {
	if (!tex || !tex->image) return false;

	// Transient discard textures always need transitions (conceptually always UNDEFINED)
	if (tex->is_transient_discard) return true;

	// Determine target layout based on type
	VkImageLayout target_layout;
	if (type == 1) {  // storage
		target_layout = VK_IMAGE_LAYOUT_GENERAL;
	} else {  // shader_read (type == 0)
		// Storage images use GENERAL layout, regular textures use SHADER_READ_ONLY_OPTIMAL
		target_layout = (tex->flags & skr_tex_flags_compute)
			? VK_IMAGE_LAYOUT_GENERAL
			: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	// Check if already in target layout
	return tex->current_layout != target_layout;
}

// General-purpose automatic transition - tracks state and inserts barrier if needed
void _skr_tex_transition(VkCommandBuffer cmd, skr_tex_t* ref_tex, VkImageLayout new_layout, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
	if (!ref_tex || !ref_tex->image) return;

	// For transient discard textures (non-readable depth/MSAA), always use UNDEFINED as old layout
	VkImageLayout old_layout = ref_tex->is_transient_discard ? VK_IMAGE_LAYOUT_UNDEFINED : ref_tex->current_layout;

	// Skip if already in target layout (unless it's a transient discard texture)
	if (!ref_tex->is_transient_discard && ref_tex->current_layout == new_layout) {
		return;
	}

#ifdef SKR_DEBUG
	if (ref_tex->current_layout != VK_IMAGE_LAYOUT_UNDEFINED && ref_tex->current_layout != old_layout) {
		skr_log(skr_log_warning, "Texture layout mismatch: tracked=%s, using=%s for transition to %s",
			_layout_to_string(ref_tex->current_layout), _layout_to_string(old_layout), _layout_to_string(new_layout));
	}
#endif

	// Determine source stage and access from old layout
	VkPipelineStageFlags src_stage  = _layout_to_src_stage   (old_layout);
	VkAccessFlags        src_access = _layout_to_access_flags(old_layout);

	// Perform transition
	_skr_transition_image_layout(cmd, ref_tex->image, ref_tex->aspect_mask,
		0, ref_tex->mip_levels, ref_tex->layer_count,
		old_layout, new_layout,
		src_stage,  dst_stage,
		src_access, dst_access);

	// Update tracked state (unless it's transient discard - always stays UNDEFINED conceptually)
	if (!ref_tex->is_transient_discard) {
		ref_tex->current_layout = new_layout;
	}
	ref_tex->first_use = false;
}

// Specialized: Transition for shader read (most common case)
void _skr_tex_transition_for_shader_read(VkCommandBuffer cmd, skr_tex_t* ref_tex, VkPipelineStageFlags dst_stage) {
	if (!ref_tex) return;

	// Storage images use GENERAL layout, regular textures use SHADER_READ_ONLY_OPTIMAL
	VkImageLayout target_layout = (ref_tex->flags & skr_tex_flags_compute)
		? VK_IMAGE_LAYOUT_GENERAL
		: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	_skr_tex_transition(cmd, ref_tex, target_layout, dst_stage, VK_ACCESS_SHADER_READ_BIT);
}

// Specialized: Transition for storage image (compute RWTexture)
void _skr_tex_transition_for_storage(VkCommandBuffer cmd, skr_tex_t* ref_tex) {
	if (!ref_tex) return;
	_skr_tex_transition(cmd, ref_tex, VK_IMAGE_LAYOUT_GENERAL,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

// Notify the system that a render pass has performed an implicit layout transition
// This updates tracked state without issuing a barrier
void _skr_tex_transition_notify_layout(skr_tex_t* ref_tex, VkImageLayout new_layout) {
	if (!ref_tex) return;

	// Don't update transient discard textures - they conceptually stay in UNDEFINED
	if (!ref_tex->is_transient_discard) {
		ref_tex->current_layout = new_layout;
	}
	ref_tex->first_use = false;
}

// Queue family ownership transfer (for future async upload)
void _skr_tex_transition_queue_family(VkCommandBuffer cmd, skr_tex_t* ref_tex,
                                     uint32_t src_queue_family, uint32_t dst_queue_family,
                                     VkImageLayout layout) {
	if (!ref_tex || !ref_tex->image || src_queue_family == dst_queue_family) return;

	VkImageLayout old_layout = ref_tex->is_transient_discard ? VK_IMAGE_LAYOUT_UNDEFINED : ref_tex->current_layout;
	VkAccessFlags src_access = _layout_to_access_flags(old_layout);
	VkAccessFlags dst_access = _layout_to_access_flags(layout);

	VkImageMemoryBarrier barrier = {
		.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcQueueFamilyIndex = src_queue_family,
		.dstQueueFamilyIndex = dst_queue_family,
		.oldLayout           = old_layout,
		.newLayout           = layout,
		.srcAccessMask       = src_access,
		.dstAccessMask       = dst_access,
		.image               = ref_tex->image,
		.subresourceRange    = {
			.aspectMask     = ref_tex->aspect_mask,
			.baseMipLevel   = 0,
			.levelCount     = ref_tex->mip_levels,
			.baseArrayLayer = 0,
			.layerCount     = ref_tex->layer_count,
		},
	};

	VkPipelineStageFlags src_stage = _layout_to_src_stage(old_layout);
	VkPipelineStageFlags dst_stage = _layout_to_src_stage(layout);

	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &barrier);

	// Update tracked state
	if (!ref_tex->is_transient_discard) {
		ref_tex->current_layout = layout;
	}
	ref_tex->current_queue_family = dst_queue_family;
	ref_tex->first_use = false;
}

///////////////////////////////////////////////////////////////////////////////

static void _skr_tex_generate_mips_blit  (VkPhysicalDevice phys_device, skr_tex_t* tex, int32_t mip_levels);
static void _skr_tex_generate_mips_render(VkDevice         device,      skr_tex_t* tex, int32_t mip_levels, const skr_shader_t* fragment_shader);

// Upload texture data from skr_tex_data_t descriptor
// Handles multiple mips and layers in mip-major layout
static skr_err_ _skr_tex_upload_data(skr_tex_t* ref_tex, const skr_tex_data_t* data) {
	if (!ref_tex || !data || !data->data) return skr_err_invalid_parameter;
	if (data->mip_count == 0 || data->layer_count == 0) return skr_err_invalid_parameter;

	// row_pitch only valid for single-mip uploads
	if (data->row_pitch != 0 && data->mip_count > 1) {
		skr_log(skr_log_warning, "Texture upload: row_pitch only valid when mip_count == 1");
		return skr_err_invalid_parameter;
	}

	// Validate ranges
	if (data->base_mip + data->mip_count > ref_tex->mip_levels) {
		skr_log(skr_log_warning, "Texture upload: mip range [%u, %u) exceeds texture mip count %u",
			data->base_mip, data->base_mip + data->mip_count, ref_tex->mip_levels);
		return skr_err_invalid_parameter;
	}
	if (data->base_layer + data->layer_count > ref_tex->layer_count) {
		skr_log(skr_log_warning, "Texture upload: layer range [%u, %u) exceeds texture layer count %u",
			data->base_layer, data->base_layer + data->layer_count, ref_tex->layer_count);
		return skr_err_invalid_parameter;
	}

	// size.z is always the actual depth (1 for non-3D textures)
	skr_vec3i_t base_size = ref_tex->size;

	// Calculate total staging buffer size (always tightly packed)
	VkDeviceSize total_size = 0;
	for (uint32_t m = 0; m < data->mip_count; m++) {
		uint32_t mip        = data->base_mip + m;
		uint64_t layer_size = skr_tex_calc_mip_size(ref_tex->format, base_size, mip);
		total_size += layer_size * data->layer_count;
	}

	// Create staging buffer
	staging_buffer_t staging = _skr_create_staging_buffer(_skr_vk.device, _skr_vk.physical_device, total_size);
	if (!staging.valid) {
		skr_log(skr_log_critical, "Failed to create staging buffer for texture upload (%llu bytes)", (unsigned long long)total_size);
		return skr_err_out_of_memory;
	}

	// Copy data to staging buffer, handling row_pitch if needed
	if (data->row_pitch > 0 && data->mip_count == 1) {
		// Single mip with row pitch - copy row by row to make tightly packed
		skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(base_size, data->base_mip);

		uint32_t block_w, block_h, block_bytes;
		skr_tex_fmt_block_info(ref_tex->format, &block_w, &block_h, &block_bytes);
		uint32_t blocks_x   = (mip_size.x + block_w - 1) / block_w;
		uint32_t dst_pitch  = blocks_x * block_bytes;
		uint32_t row_count  = ((mip_size.y + block_h - 1) / block_h) * mip_size.z * data->layer_count;

		const uint8_t* src = (const uint8_t*)data->data;
		uint8_t*       dst = (uint8_t*)staging.mapped_data;
		for (uint32_t row = 0; row < row_count; row++) {
			memcpy(dst, src, dst_pitch);
			src += data->row_pitch;
			dst += dst_pitch;
		}
	} else {
		// Tightly packed - single memcpy
		memcpy(staging.mapped_data, data->data, total_size);
	}

	// Build copy regions (one per mip level, covering all layers for that mip)
	VkBufferImageCopy* regions = _skr_malloc(sizeof(VkBufferImageCopy) * data->mip_count);
	if (!regions) {
		vkUnmapMemory  (_skr_vk.device, staging.memory);
		vkFreeMemory   (_skr_vk.device, staging.memory, NULL);
		vkDestroyBuffer(_skr_vk.device, staging.buffer, NULL);
		return skr_err_out_of_memory;
	}

	VkDeviceSize offset = 0;
	for (uint32_t m = 0; m < data->mip_count; m++) {
		uint32_t    mip        = data->base_mip + m;
		skr_vec3i_t mip_size   = skr_tex_calc_mip_dimensions(base_size, mip);
		uint64_t    layer_size = skr_tex_calc_mip_size(ref_tex->format, base_size, mip);

		regions[m] = (VkBufferImageCopy){
			.bufferOffset      = offset,
			.bufferRowLength   = 0,  // Tightly packed
			.bufferImageHeight = 0,  // Tightly packed
			.imageSubresource  = {
				.aspectMask     = ref_tex->aspect_mask,
				.mipLevel       = mip,
				.baseArrayLayer = data->base_layer,
				.layerCount     = data->layer_count,
			},
			.imageOffset = {0, 0, 0},
			.imageExtent = {mip_size.x, mip_size.y, mip_size.z},
		};

		offset += layer_size * data->layer_count;
	}

	// Create command buffer and upload
	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

	// Transition to TRANSFER_DST
	_skr_tex_transition(ctx.cmd, ref_tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

	// Copy all regions
	vkCmdCopyBufferToImage(ctx.cmd, staging.buffer, ref_tex->image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, data->mip_count, regions);

	// Transition to shader read
	VkPipelineStageFlags shader_stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	if (ref_tex->flags & skr_tex_flags_compute) {
		shader_stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}
	_skr_tex_transition_for_shader_read(ctx.cmd, ref_tex, shader_stages);

	// Defer cleanup
	_skr_cmd_destroy_buffer(ctx.destroy_list, staging.buffer);
	_skr_cmd_destroy_memory(ctx.destroy_list, staging.memory);
	_skr_cmd_release(ctx.cmd);

	_skr_free(regions);
	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_tex_create(skr_tex_fmt_ format, skr_tex_flags_ flags, skr_tex_sampler_t sampler, skr_vec3i_t size, int32_t multisample, int32_t mip_count, const skr_tex_data_t* opt_data, skr_tex_t* out_tex) {
	if (!out_tex) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_tex = (skr_tex_t){};

	// Validate parameters
	if (size.x <= 0 || size.y <= 0 || size.z <= 0) {
		skr_log(skr_log_warning, "Invalid texture size");
		return skr_err_invalid_parameter;
	}

	out_tex->flags            = flags;
	out_tex->sampler_settings = sampler;
	out_tex->format           = format;

	VkFormat vk_format = skr_tex_fmt_to_native(format);
	if (vk_format == VK_FORMAT_UNDEFINED) {
		return skr_err_unsupported;
	}

	// Determine image type, layer count, and normalize size
	// After this block, size.z is always the actual depth dimension (1 for 2D textures)
	// and layer_count holds the array layer count
	VkImageType image_type;
	if (out_tex->flags & skr_tex_flags_3d) {
		image_type           = VK_IMAGE_TYPE_3D;
		out_tex->layer_count = 1;
		out_tex->size        = size;  // z is actual depth
	} else if (out_tex->flags & skr_tex_flags_array) {
		image_type           = VK_IMAGE_TYPE_2D;
		out_tex->layer_count = size.z;
		out_tex->size        = (skr_vec3i_t){ size.x, size.y, 1 };
	} else if (out_tex->flags & skr_tex_flags_cubemap) {
		image_type           = VK_IMAGE_TYPE_2D;
		out_tex->layer_count = 6;
		out_tex->size        = (skr_vec3i_t){ size.x, size.y, 1 };
	} else {
		image_type           = VK_IMAGE_TYPE_2D;
		out_tex->layer_count = 1;
		out_tex->size        = (skr_vec3i_t){ size.x, size.y, 1 };
	}

	// Normalize mip count (default to 1 if not specified or zero)
	out_tex->mip_levels = mip_count <= 0 ? 1 : mip_count;

	// Normalize sample count
	out_tex->samples = (multisample > 1) ? (VkSampleCountFlagBits)multisample : VK_SAMPLE_COUNT_1_BIT;

	// Check if this is a depth format
	bool is_depth = (format == skr_tex_fmt_depth16 || format == skr_tex_fmt_depth32 || format == skr_tex_fmt_depth32s8 || format == skr_tex_fmt_depth24s8 || format == skr_tex_fmt_depth16s8);

	// Determine usage flags
	// Don't add SAMPLED_BIT for MSAA depth textures - not supported by some drivers (NVIDIA returns size=0)
	bool is_msaa_depth = (out_tex->samples > VK_SAMPLE_COUNT_1_BIT) && is_depth;
	VkImageUsageFlags usage = is_msaa_depth ? 0 : VK_IMAGE_USAGE_SAMPLED_BIT;
	
	if (out_tex->flags & skr_tex_flags_readable) {
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	// Determine aspect mask based on format
	out_tex->aspect_mask  = 0;
	if (is_depth)                           { out_tex->aspect_mask  |= VK_IMAGE_ASPECT_DEPTH_BIT;   }
	if (_skr_format_has_stencil(vk_format)) { out_tex->aspect_mask  |= VK_IMAGE_ASPECT_STENCIL_BIT; }
	if (out_tex->aspect_mask == 0)          { out_tex->aspect_mask   = VK_IMAGE_ASPECT_COLOR_BIT;   }


	// For MSAA attachments, add transient bit for in-tile resolve optimization
	// But only if the texture is NOT readable (transient means no memory backing)
	bool is_msaa_attachment = out_tex->samples > VK_SAMPLE_COUNT_1_BIT && (out_tex->flags & skr_tex_flags_writeable) && !(out_tex->flags & skr_tex_flags_readable);

	if (out_tex->flags & skr_tex_flags_writeable) {
		if (is_depth) {
			usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		} else {
			usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
			// TRANSIENT_ATTACHMENT_BIT can't be combined with TRANSFER_DST_BIT
			if (!is_msaa_attachment) {
				usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			}
		}
	}
	if (opt_data && opt_data->data) {
		usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT; // Need to upload data
	}

	// Only use transient attachment if format+usage combination is supported
	// AND lazily allocated memory is available (required for transient attachments)
	if (is_msaa_attachment) {
		VkImageUsageFlags test_usage = usage | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
		test_usage &= ~(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

		VkImageFormatProperties format_props;
		VkResult result = vkGetPhysicalDeviceImageFormatProperties(
			_skr_vk.physical_device,
			vk_format,
			image_type,
			VK_IMAGE_TILING_OPTIMAL,
			test_usage,
			0,
			&format_props);

		// Check if lazily allocated memory is available
		bool has_lazy_memory = false;
		if (result == VK_SUCCESS) {
			VkPhysicalDeviceMemoryProperties mem_properties;
			vkGetPhysicalDeviceMemoryProperties(_skr_vk.physical_device, &mem_properties);

			// Check if any memory type has LAZILY_ALLOCATED_BIT
			for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
				if (mem_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
					has_lazy_memory = true;
					break;
				}
			}
		}

		// Only use transient attachment if both format is supported AND lazy memory is available
		if (result == VK_SUCCESS && has_lazy_memory) {
			usage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
			// Remove SAMPLED_BIT and TRANSFER_DST_BIT for transient attachments
			usage &= ~(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
			// is_msaa_attachment stays true - request lazy memory allocation
		} else {
			// Transient not supported, fall back to regular memory
			is_msaa_attachment = false;
		}
	}

	// For compute shader storage images (RWTexture2D)
	if (out_tex->flags & skr_tex_flags_compute) {
		usage |= VK_IMAGE_USAGE_STORAGE_BIT;
	}

	// For dynamic textures (updated via skr_tex_set_data)
	if (out_tex->flags & skr_tex_flags_dynamic) {
		usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}

	// For mipmap generation
	if (out_tex->flags & skr_tex_flags_gen_mips) {
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		// Check if format supports STORAGE_BIT (needed for compute-based mipmap filters)
		VkFormatProperties format_props;
		vkGetPhysicalDeviceFormatProperties(_skr_vk.physical_device, vk_format, &format_props);
		if (format_props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {
			usage |= VK_IMAGE_USAGE_STORAGE_BIT;
		}

		// Add color attachment usage for render-based mipmap generation
		if (!is_depth) {
			usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		}

		// Auto-calculate mip count if not specified or is 1
		if (out_tex->mip_levels == 1) {
			out_tex->mip_levels = skr_tex_calc_mip_count(out_tex->size);
		}
	}

	// Create image (use normalized out_tex->size where z is always depth)
	VkImageCreateInfo image_info = {
		.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType     = image_type,
		.format        = vk_format,
		.extent        = { .width = out_tex->size.x, .height = out_tex->size.y, .depth = out_tex->size.z },
		.mipLevels     = out_tex->mip_levels,
		.arrayLayers   = out_tex->layer_count,
		.samples       = out_tex->samples,
		.tiling        = VK_IMAGE_TILING_OPTIMAL,
		.usage         = usage,
		.sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.flags         = (out_tex->flags & skr_tex_flags_cubemap) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
	};

	VkResult vr = vkCreateImage(_skr_vk.device, &image_info, NULL, &out_tex->image);
	if (vr != VK_SUCCESS) {
		skr_log(skr_log_critical, "vkCreateImage failed");
		return skr_err_device_error;
	}

	// Allocate memory using helper
	if (_skr_allocate_image_memory(_skr_vk.device, _skr_vk.physical_device, out_tex->image, is_msaa_attachment, &out_tex->memory) == VK_NULL_HANDLE) {
		skr_log(skr_log_critical, "Failed to allocate texture memory - Format: %d, Size: %dx%dx%d, Mips: %d, Layers: %d, Samples: %d, Usage: 0x%x, Flags: 0x%x",
			format, size.x, size.y, size.z, out_tex->mip_levels, out_tex->layer_count, out_tex->samples, usage, out_tex->flags);
		vkDestroyImage(_skr_vk.device, out_tex->image, NULL);
		*out_tex = (skr_tex_t){};
		return skr_err_out_of_memory;
	}

	vkBindImageMemory(_skr_vk.device, out_tex->image, out_tex->memory, 0);

	// Initialize layout tracking BEFORE any transitions
	// This must happen before _skr_tex_upload_data or _skr_tex_transition calls
	// since those functions update current_layout
	out_tex->current_layout       = VK_IMAGE_LAYOUT_UNDEFINED;
	out_tex->current_queue_family = _skr_vk.graphics_queue_family;
	out_tex->first_use            = true;
	// Transient discard optimization for non-readable depth/MSAA (tile GPU optimization)
	out_tex->is_transient_discard = (is_msaa_attachment || (is_depth && !(flags & skr_tex_flags_readable)));

	// Upload texture data if provided (or just transition to shader read layout)
	if (opt_data && opt_data->data) {
		skr_err_ upload_err = _skr_tex_upload_data(out_tex, opt_data);
		if (upload_err != skr_err_success) {
			vkFreeMemory  (_skr_vk.device, out_tex->memory, NULL);
			vkDestroyImage(_skr_vk.device, out_tex->image,  NULL);
			*out_tex = (skr_tex_t){};
			return upload_err;
		}
	} else if (!is_msaa_attachment && !(out_tex->flags & skr_tex_flags_writeable)) {
		// No data provided, transition to appropriate layout for read-only textures
		// Skip for transient MSAA attachments - they don't need initial layout transition
		// Skip for writeable textures - let the first render pass handle the transition

		_skr_cmd_ctx_t ctx = _skr_cmd_acquire();
		// Use automatic transition system - handles storage vs regular textures
		if (out_tex->flags & skr_tex_flags_compute) { _skr_tex_transition_for_storage    (ctx.cmd, out_tex); }
		else                                        { _skr_tex_transition_for_shader_read(ctx.cmd, out_tex, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT); }
		_skr_cmd_release(ctx.cmd);
	}

	// Create image view
	VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
	if      (out_tex->flags & skr_tex_flags_3d     ) view_type = VK_IMAGE_VIEW_TYPE_3D;
	else if (out_tex->flags & skr_tex_flags_cubemap) view_type = VK_IMAGE_VIEW_TYPE_CUBE;
	else if (out_tex->flags & skr_tex_flags_array  ) view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;

	VkImageViewCreateInfo view_info = {
		.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image    = out_tex->image,
		.viewType = view_type,
		.format   = vk_format,
		.subresourceRange = {
			.aspectMask     = out_tex->aspect_mask,
			.baseMipLevel   = 0,
			.levelCount     = image_info.mipLevels,
			.baseArrayLayer = 0,
			.layerCount     = image_info.arrayLayers,
		},
	};

	vr = vkCreateImageView(_skr_vk.device, &view_info, NULL, &out_tex->view);
	if (vr != VK_SUCCESS) {
		skr_log(skr_log_critical, "vkCreateImageView failed");
		vkFreeMemory  (_skr_vk.device, out_tex->memory, NULL);
		vkDestroyImage(_skr_vk.device, out_tex->image,  NULL);
		*out_tex = (skr_tex_t){};
		return skr_err_device_error;
	}

	// Store texture properties
	out_tex->sampler = _skr_sampler_cache_acquire(sampler);

	return skr_err_success;
}

void skr_tex_destroy(skr_tex_t* ref_tex) {
	if (!ref_tex) return;

	_skr_cmd_destroy_framebuffer(NULL, ref_tex->framebuffer);
	_skr_cmd_destroy_framebuffer(NULL, ref_tex->framebuffer_depth);
	_skr_sampler_cache_release(ref_tex->sampler_settings);
	_skr_cmd_destroy_image_view (NULL, ref_tex->view);

	// Only destroy image/memory if we own them (not external)
	if (!ref_tex->is_external) {
		_skr_cmd_destroy_image (NULL, ref_tex->image);
		_skr_cmd_destroy_memory(NULL, ref_tex->memory);
	}
	*ref_tex = (skr_tex_t){};
}

bool skr_tex_is_valid(const skr_tex_t* tex) {
	return tex && tex->image != VK_NULL_HANDLE;
}

skr_vec3i_t skr_tex_get_size(const skr_tex_t* tex) {
	return tex ? tex->size : (skr_vec3i_t){0, 0, 0};
}

uint32_t skr_tex_get_array_count(const skr_tex_t* tex) {
	return tex ? tex->layer_count : 0;
}

skr_tex_fmt_ skr_tex_get_format(const skr_tex_t* tex) {
	return tex ? tex->format : skr_tex_fmt_none;
}

skr_tex_flags_ skr_tex_get_flags(const skr_tex_t* tex) {
	return tex ? tex->flags : 0;
}

int32_t skr_tex_get_multisample(const skr_tex_t* tex) {
	return tex ? tex->samples : VK_SAMPLE_COUNT_1_BIT;
}

skr_tex_sampler_t skr_tex_get_sampler(const skr_tex_t* tex) {
	return tex ? tex->sampler_settings : (skr_tex_sampler_t){0};
}

void skr_tex_set_sampler(skr_tex_t* ref_tex, skr_tex_sampler_t sampler) {
	if (!ref_tex || ref_tex->image == VK_NULL_HANDLE) return;

	// Release old sampler from cache
	if (ref_tex->sampler != VK_NULL_HANDLE) {
		_skr_sampler_cache_release(ref_tex->sampler_settings);
	}

	// Acquire new sampler from cache and update settings
	ref_tex->sampler          = _skr_sampler_cache_acquire(sampler);
	ref_tex->sampler_settings = sampler;
}

skr_err_ skr_tex_set_data(skr_tex_t* ref_tex, const skr_tex_data_t* data) {
	if (!ref_tex || !data || !data->data) return skr_err_invalid_parameter;
	if (ref_tex->image == VK_NULL_HANDLE) return skr_err_invalid_parameter;

	return _skr_tex_upload_data(ref_tex, data);
}

void skr_tex_set_name(skr_tex_t* ref_tex, const char* name) {
	if (!ref_tex || ref_tex->image == VK_NULL_HANDLE) return;

	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_IMAGE, (uint64_t)ref_tex->image, name);

	// Also name the image view if it exists
	if (ref_tex->view != VK_NULL_HANDLE) {
		char view_name[256];
		snprintf(view_name, sizeof(view_name), "%s_view", name);
		_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)ref_tex->view, view_name);
	}
}

void skr_tex_generate_mips(skr_tex_t* ref_tex, const skr_shader_t* opt_shader) {
	if (!skr_tex_is_valid(ref_tex)) {
		skr_log(skr_log_warning, "Cannot generate mipmaps for invalid texture");
		return;
	}

	// size.z is always the actual depth (1 for non-3D textures)
	int32_t mip_levels = skr_tex_calc_mip_count(ref_tex->size);

	if (mip_levels <= 1) {
		skr_log(skr_log_info, "Texture only has 1 mip level, nothing to generate");
		return;
	}

	// Route to appropriate implementation
	// If a custom shader is provided, use render-based mipmap generation (fragment shader)
	// Otherwise fall back to simple blit
	if (opt_shader == NULL) {
		_skr_tex_generate_mips_blit  (_skr_vk.physical_device, ref_tex, mip_levels);
	} else {
		_skr_tex_generate_mips_render(_skr_vk.device, ref_tex, mip_levels, opt_shader);
	}
}

static void _skr_tex_generate_mips_blit(VkPhysicalDevice phys_device, skr_tex_t* ref_tex, int32_t mip_levels) {
	// Check format support for blit operations
	VkFormatProperties format_properties;
	VkFormat           vk_format = skr_tex_fmt_to_native(ref_tex->format);
	vkGetPhysicalDeviceFormatProperties(phys_device, vk_format, &format_properties);

	if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ||
	    !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
		skr_log(skr_log_critical, "Texture format doesn't support blit operations for mipmap generation");
		return;
	}

	// Check if format supports linear filtering during blit
	VkFilter filter_mode = VK_FILTER_LINEAR;
	if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
		skr_log(skr_log_info, "Format doesn't support linear filtering, using nearest");
		filter_mode = VK_FILTER_NEAREST;
	}

	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

	// Transition mip 0 to TRANSFER_SRC_OPTIMAL (automatic system tracks current layout)
	_skr_tex_transition(ctx.cmd, ref_tex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	// Generate each mip level by blitting from the previous level
	int32_t mip_width  = ref_tex->size.x;
	int32_t mip_height = ref_tex->size.y;

	for (int32_t i = 1; i < mip_levels; i++) {
		int32_t next_mip_width  = mip_width  > 1 ? mip_width  / 2 : 1;
		int32_t next_mip_height = mip_height > 1 ? mip_height / 2 : 1;

		// Transition mip i from UNDEFINED to TRANSFER_DST_OPTIMAL
		_skr_transition_image_layout(ctx.cmd, ref_tex->image, ref_tex->aspect_mask, i, 1, ref_tex->layer_count,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, VK_ACCESS_TRANSFER_WRITE_BIT);

		// Blit from mip i-1 to mip i (for all layers)
		VkImageBlit blit = {
			.srcSubresource = {
				.aspectMask     = ref_tex->aspect_mask,
				.mipLevel       = i - 1,
				.baseArrayLayer = 0,
				.layerCount     = ref_tex->layer_count,
			},
			.srcOffsets[0] = {0, 0, 0},
			.srcOffsets[1] = {mip_width, mip_height, 1},
			.dstSubresource = {
				.aspectMask     = ref_tex->aspect_mask,
				.mipLevel       = i,
				.baseArrayLayer = 0,
				.layerCount     = ref_tex->layer_count,
			},
			.dstOffsets[0] = {0, 0, 0},
			.dstOffsets[1] = {next_mip_width, next_mip_height, 1},
		};

		vkCmdBlitImage(ctx.cmd,
			ref_tex->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			ref_tex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &blit, filter_mode);

		// Transition mip i from TRANSFER_DST to TRANSFER_SRC for next iteration
		_skr_transition_image_layout(ctx.cmd, ref_tex->image, ref_tex->aspect_mask, i, 1, ref_tex->layer_count,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

		mip_width  = next_mip_width;
		mip_height = next_mip_height;
	}

	// Transition back to shader read layout (automatic system handles this)
	_skr_tex_transition_for_shader_read(ctx.cmd, ref_tex, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	_skr_cmd_release(ctx.cmd);
}

static void _skr_tex_generate_mips_render(VkDevice device, skr_tex_t* ref_tex, int32_t mip_levels, const skr_shader_t* fragment_shader) {
	if (!skr_shader_is_valid(fragment_shader)) {
		skr_log(skr_log_warning, "Invalid fragment shader provided for mipmap generation");
		return;
	}

	// Validate that shader has required 'src_tex' binding
	skr_bind_t bind_source = skr_shader_get_bind(fragment_shader, "src_tex");
	if ((bind_source.stage_bits & skr_stage_pixel) == 0) {
		skr_log(skr_log_warning, "Mip shader missing 'src_tex' binding");
		return;
	}

	// Create material from shader (handles pipeline registration automatically)
	skr_material_t material;
	skr_material_create((skr_material_info_t){
		.shader      = fragment_shader,
		.cull        = skr_cull_none,
		.depth_test  = skr_compare_always,
		.write_mask  = skr_write_rgba,
	}, &material);
	if (!skr_material_is_valid(&material)) {
		skr_log(skr_log_warning, "Failed to create material for mipmap generation");
		return;
	}

	// Lock pipeline cache for the duration of this mipmap generation
	_skr_pipeline_lock();

	// Register render pass format with pipeline system (cached for reuse)
	VkFormat format = skr_tex_fmt_to_native(ref_tex->format);
	skr_pipeline_renderpass_key_t rp_key = {
		.color_format   = format,
		.depth_format   = VK_FORMAT_UNDEFINED,
		.resolve_format = VK_FORMAT_UNDEFINED,
		.samples        = VK_SAMPLE_COUNT_1_BIT,
		.depth_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.color_load_op  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,  // Full blit
	};
	int32_t renderpass_idx = _skr_pipeline_register_renderpass(&rp_key);
	int32_t vert_idx       = _skr_pipeline_register_vertformat((skr_vert_type_t){});

	// Get cached render pass
	VkRenderPass render_pass = _skr_pipeline_get_renderpass(renderpass_idx);
	if (render_pass == VK_NULL_HANDLE) {
		_skr_pipeline_unlock();
		skr_material_destroy(&material);
		return;
	}

	// Acquire command buffer context
	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();
	if (!ctx.cmd) {
		skr_log(skr_log_warning, "Failed to acquire command buffer for mipmap generation");
		_skr_pipeline_unlock();
		skr_material_destroy(&material);
		return;
	}

	// Determine view type for layer rendering
	VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
	if (ref_tex->flags & skr_tex_flags_cubemap) {
		view_type = VK_IMAGE_VIEW_TYPE_CUBE;
	} else if (ref_tex->flags & skr_tex_flags_array) {
		view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	}

	// Get cached pipeline (lazy creation via 3D cache: material × renderpass × vertformat)
	VkPipeline pipeline = _skr_pipeline_get(material.pipeline_material_idx, renderpass_idx, vert_idx);
	if (pipeline == VK_NULL_HANDLE) {
		skr_log(skr_log_warning, "Failed to get pipeline for mipmap generation");
		_skr_cmd_release(ctx.cmd);
		_skr_pipeline_unlock();
		skr_material_destroy(&material);
		return;
	}

	// Pre-populate parameter buffers for all mip levels
	// Use material API to set values (handles different $Global layouts per shader)
	int32_t      num_mips      = mip_levels - 1;
	uint8_t*     all_params    = NULL;
	skr_buffer_t params_buffer = {0};

	if (material.param_buffer_size > 0) {
		all_params = _skr_calloc(num_mips, material.param_buffer_size);

		for (int32_t mip = 1; mip < mip_levels; mip++) {
			skr_vec3i_t dst_dims = skr_tex_calc_mip_dimensions(ref_tex->size, mip);
			skr_vec3i_t src_dims = skr_tex_calc_mip_dimensions(ref_tex->size, mip - 1);

			// Use material API to populate values (handles different shader layouts)
			skr_vec2i_t src_size = {src_dims.x, src_dims.y};
			skr_vec2i_t dst_size = {dst_dims.x, dst_dims.y};
			uint32_t src_mip = mip - 1;
			skr_material_set_param(&material, "src_size", sksc_shader_var_uint, 2, &src_size);
			skr_material_set_param(&material, "dst_size", sksc_shader_var_uint, 2, &dst_size);
			skr_material_set_param(&material, "src_mip_level", sksc_shader_var_uint, 1, &src_mip);
			skr_material_set_param(&material, "mip_max", sksc_shader_var_uint, 1, &mip_levels);

			// Copy material's parameter buffer for this mip (preserves other values)
			memcpy(all_params + (mip - 1) * material.param_buffer_size,
			       material.param_buffer,
			       material.param_buffer_size);
		}

		// Create GPU buffer with all mip parameters
		skr_buffer_create(all_params, num_mips, material.param_buffer_size, skr_buffer_type_constant, skr_use_static, &params_buffer);
		_skr_free(all_params);
	}

	// Generate each mip level by rendering from previous mip
	for (int32_t mip = 1; mip < mip_levels; mip++) {
		skr_vec3i_t mip_dims   = skr_tex_calc_mip_dimensions(ref_tex->size, mip);
		uint32_t    mip_width  = mip_dims.x;
		uint32_t    mip_height = mip_dims.y;

		// Create image view for this mip level (target)
		VkImageView mip_view = VK_NULL_HANDLE;
		{
			VkImageViewCreateInfo view_info = {
				.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image      = ref_tex->image,
				.viewType   = view_type,
				.format     = format,
				.subresourceRange = {
					.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel   = mip,
					.levelCount     = 1,
					.baseArrayLayer = 0,
					.layerCount     = ref_tex->layer_count,
				},
			};
			VkResult vr = vkCreateImageView(device, &view_info, NULL, &mip_view);
			if (vr != VK_SUCCESS) {
				SKR_VK_CHECK_NRET(vr, "vkCreateImageView");
				continue;
			}
			_skr_cmd_destroy_image_view(ctx.destroy_list, mip_view);
		}

		// Create framebuffer for this mip level
		VkFramebuffer framebuffer = VK_NULL_HANDLE;
		{
			VkFramebufferCreateInfo fb_info = {
				.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass      = render_pass,
				.attachmentCount = 1,
				.pAttachments    = &mip_view,
				.width           = mip_width,
				.height          = mip_height,
				.layers          = ref_tex->layer_count,
			};
			VkResult vr = vkCreateFramebuffer(device, &fb_info, NULL, &framebuffer);
			if (vr != VK_SUCCESS) {
				SKR_VK_CHECK_NRET(vr, "vkCreateFramebuffer");
				continue;
			}
			_skr_cmd_destroy_framebuffer(ctx.destroy_list, framebuffer);
		}

		// Create image view for the previous mip level (source)
		VkImageView src_view = VK_NULL_HANDLE;
		{
			VkImageViewCreateInfo src_view_info = {
				.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image      = ref_tex->image,
				.viewType   = view_type,
				.format     = format,
				.subresourceRange = {
					.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel   = mip - 1,
					.levelCount     = 1,
					.baseArrayLayer = 0,
					.layerCount     = ref_tex->layer_count,
				},
			};
			VkResult vr = vkCreateImageView(device, &src_view_info, NULL, &src_view);
			if (vr != VK_SUCCESS) {
				SKR_VK_CHECK_NRET(vr, "vkCreateImageView");
				continue;
			}
			_skr_cmd_destroy_image_view(ctx.destroy_list, src_view);
		}

		// Transition current mip to color attachment (automatic tracking handles UNDEFINED vs previous layout)
		VkImageMemoryBarrier barrier = {
			.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask       = 0,
			.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image               = ref_tex->image,
			.subresourceRange    = {
				.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = mip,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = ref_tex->layer_count,
			},
		};
		vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		                     0, 0, NULL, 0, NULL, 1, &barrier);

		// Begin render pass
		vkCmdBeginRenderPass(ctx.cmd, &(VkRenderPassBeginInfo){
			.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass      = render_pass,
			.framebuffer     = framebuffer,
			.renderArea      = {{0, 0}, {mip_width, mip_height}},
			.clearValueCount = 0,
		}, VK_SUBPASS_CONTENTS_INLINE);

		// Bind pipeline
		vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		// Set viewport and scissor
		VkViewport viewport = {0, 0, (float)mip_width, (float)mip_height, 0.0f, 1.0f};
		VkRect2D   scissor  = {{0, 0}, {mip_width, mip_height}};
		vkCmdSetViewport(ctx.cmd, 0, 1, &viewport);
		vkCmdSetScissor (ctx.cmd, 0, 1, &scissor);

		// Build descriptor writes using material system
		VkWriteDescriptorSet   writes      [32];
		VkDescriptorBufferInfo buffer_infos[16];
		VkDescriptorImageInfo  image_infos [16];
		uint32_t write_ct  = 0;
		uint32_t buffer_ct = 0;
		uint32_t image_ct  = 0;

		// Handle material parameters if present ($Global buffer with per-mip offset)
		if (skr_buffer_is_valid(&params_buffer)) {
			buffer_infos[buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = params_buffer.buffer,
				.offset = (mip - 1) * material.param_buffer_size,  // Offset for this mip
				.range  = material.param_buffer_size,
			};
			writes[write_ct++] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = SKR_BIND_SHIFT_BUFFER + _skr_vk.bind_settings.material_slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo     = &buffer_infos[buffer_ct++],
			};
		}

		// Manually add source texture binding (since we create it per-mip)
		image_infos[image_ct] = (VkDescriptorImageInfo){
			.sampler     = ref_tex->sampler,
			.imageView   = src_view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
		writes[write_ct++] = (VkWriteDescriptorSet){
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstBinding      = bind_source.slot,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo      = &image_infos[image_ct++],
		};

		// Add any other material bindings (textures/buffers, including globals)
		const int32_t ignore_slots[] = {
			SKR_BIND_SHIFT_BUFFER + _skr_vk.bind_settings.material_slot,  // Already handled above
			bind_source.slot                                               // Source texture (per-mip, handled above)
		};

		_skr_bind_pool_lock();
		int32_t fail_idx = _skr_material_add_writes(
			_skr_bind_pool_get(material.bind_start), material.bind_count,
			ignore_slots, sizeof(ignore_slots)/sizeof(ignore_slots[0]),
			writes,       sizeof(writes      )/sizeof(writes      [0]),
			buffer_infos, sizeof(buffer_infos)/sizeof(buffer_infos[0]),
			image_infos,  sizeof(image_infos )/sizeof(image_infos [0]),
			&write_ct, &buffer_ct, &image_ct
		);
		_skr_bind_pool_unlock();

		if (fail_idx >= 0) {
			const sksc_shader_meta_t* meta = material.key.shader->meta;
			skr_log(skr_log_critical, "Mipmap generation missing binding '%s' in shader '%s'", _skr_material_bind_name(meta, fail_idx), meta->name);
			vkCmdEndRenderPass(ctx.cmd);
			continue;
		}

		// Push descriptors and draw
		_skr_bind_descriptors(
			ctx.cmd, ctx.descriptor_pool, VK_PIPELINE_BIND_POINT_GRAPHICS,
			_skr_pipeline_get_layout           (material.pipeline_material_idx),
			_skr_pipeline_get_descriptor_layout(material.pipeline_material_idx),
			writes, write_ct);

		// Draw fullscreen triangle (with instances for each layer/face)
		vkCmdDraw(ctx.cmd, 3, ref_tex->layer_count, 0, 0);

		// End render pass
		vkCmdEndRenderPass(ctx.cmd);

		// Transition current mip to shader read for next iteration
		barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		vkCmdPipelineBarrier( ctx.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);
	}

	skr_buffer_destroy  (&params_buffer);
	skr_material_destroy(&material);

	_skr_cmd_release(ctx.cmd);
	_skr_pipeline_unlock();
}

///////////////////////////////////////////////////////////////////////////////
// Sampler creation
///////////////////////////////////////////////////////////////////////////////

VkSampler _skr_sampler_create_vk(VkDevice device, skr_tex_sampler_t settings) {
	VkFilter             filter      = _skr_to_vk_filter (settings.sample);
	VkSamplerAddressMode address     = _skr_to_vk_address(settings.address);
	VkCompareOp          compare_op  = _skr_to_vk_compare(settings.sample_compare);
	bool                 anisotropic = settings.sample == skr_tex_sample_anisotropic;

	VkSamplerCreateInfo sampler_info = {
		.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter               = filter,
		.minFilter               = filter,
		.mipmapMode              = filter == VK_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU            = address,
		.addressModeV            = address,
		.addressModeW            = address,
		.anisotropyEnable        = anisotropic ? VK_TRUE : VK_FALSE,
		.maxAnisotropy           = anisotropic ? (float)settings.anisotropy : 1.0f,
		.compareEnable           = settings.sample_compare != skr_compare_none ? VK_TRUE : VK_FALSE,
		.compareOp               = compare_op,
		.minLod                  = 0.0f,
		.maxLod                  = VK_LOD_CLAMP_NONE,
		.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE,
	};

	VkSampler vk_sampler = VK_NULL_HANDLE;
	VkResult  vr         = vkCreateSampler(device, &sampler_info, NULL, &vk_sampler);
	SKR_VK_CHECK_RET(vr, "vkCreateSampler", VK_NULL_HANDLE);

	// Generate debug name based on sampler settings
	const char* filter_str = 
		settings.sample         == skr_tex_sample_linear      ? "linear" :
		settings.sample         == skr_tex_sample_point       ? "point" :
		settings.sample         == skr_tex_sample_anisotropic ? "aniso" : "unk";
	const char* address_str = 
		settings.address        == skr_tex_address_wrap       ? "wrap" :
		settings.address        == skr_tex_address_clamp      ? "clamp" :
		settings.address        == skr_tex_address_mirror     ? "mirror" : "unk";
	const char* compare_str = 
		settings.sample_compare == skr_compare_none           ? "" :
		settings.sample_compare == skr_compare_less           ? "_less" :
		settings.sample_compare == skr_compare_less_or_eq     ? "_lesseq" : "_cmp";

	char name[128];
	snprintf(name, sizeof(name), "sampler_%s_%s_%s", filter_str, address_str, compare_str);
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_SAMPLER, (uint64_t)vk_sampler, name);

	return vk_sampler;
}

///////////////////////////////////////////////////////////////////////////////
// Sampler cache implementation
///////////////////////////////////////////////////////////////////////////////

#define SKR_SAMPLER_CACHE_INITIAL_CAPACITY 16

static bool _skr_sampler_settings_equal(skr_tex_sampler_t a, skr_tex_sampler_t b) {
	return a.sample         == b.sample &&
	       a.address        == b.address &&
	       a.sample_compare == b.sample_compare &&
	       a.anisotropy     == b.anisotropy;
}

void _skr_sampler_cache_init(void) {
	_skr_sampler_cache_t* cache = &_skr_vk.sampler_cache;

	mtx_init(&cache->mutex, mtx_plain);

	cache->capacity = SKR_SAMPLER_CACHE_INITIAL_CAPACITY;
	cache->count    = 0;
	cache->entries  = _skr_calloc(cache->capacity, sizeof(_skr_sampler_entry_t));
}

void _skr_sampler_cache_shutdown(void) {
	_skr_sampler_cache_t* cache = &_skr_vk.sampler_cache;

	// Destroy all cached samplers
	for (uint32_t i = 0; i < cache->count; i++) {
		if (cache->entries[i].sampler != VK_NULL_HANDLE) {
			vkDestroySampler(_skr_vk.device, cache->entries[i].sampler, NULL);
		}
	}

	mtx_destroy(&cache->mutex);

	_skr_free(cache->entries);
	*cache = (_skr_sampler_cache_t){0};
}

VkSampler _skr_sampler_cache_acquire(skr_tex_sampler_t settings) {
	_skr_sampler_cache_t* cache = &_skr_vk.sampler_cache;
	VkSampler result = VK_NULL_HANDLE;

	mtx_lock(&cache->mutex);

	// Search for existing sampler with matching settings
	for (uint32_t i = 0; i < cache->count; i++) {
		if (_skr_sampler_settings_equal(cache->entries[i].settings, settings)) {
			cache->entries[i].ref_count++;
			result = cache->entries[i].sampler;
			goto done;
		}
	}

	// Not found, create new sampler (this can be slow, but happens rarely)
	result = _skr_sampler_create_vk(_skr_vk.device, settings);
	if (result == VK_NULL_HANDLE) {
		goto done;
	}

	// Grow cache if needed
	if (cache->count >= cache->capacity) {
		uint32_t new_capacity = cache->capacity * 2;
		_skr_sampler_entry_t* new_entries = _skr_realloc(cache->entries, new_capacity * sizeof(_skr_sampler_entry_t));
		if (!new_entries) {
			skr_log(skr_log_warning, "Failed to grow sampler cache, sampler will not be cached");
			// Return the sampler anyway - it just won't be cached for reuse
			goto done;
		}
		cache->entries  = new_entries;
		cache->capacity = new_capacity;
	}

	// Add new entry
	cache->entries[cache->count++] = (_skr_sampler_entry_t){
		.settings  = settings,
		.sampler   = result,
		.ref_count = 1,
	};

done:
	mtx_unlock(&cache->mutex);
	return result;
}

void _skr_sampler_cache_release(skr_tex_sampler_t settings) {
	_skr_sampler_cache_t* cache = &_skr_vk.sampler_cache;

	mtx_lock(&cache->mutex);

	// Find the entry and decrement ref count
	// We don't destroy samplers when ref hits zero because the GPU might still
	// be using them. Samplers are tiny (~64 bytes) and reused, so keeping them
	// until shutdown is fine. They'll be destroyed in _skr_sampler_cache_shutdown.
	for (uint32_t i = 0; i < cache->count; i++) {
		if (_skr_sampler_settings_equal(cache->entries[i].settings, settings)) {
			if (cache->entries[i].ref_count > 0) {
				cache->entries[i].ref_count--;
			}
			mtx_unlock(&cache->mutex);
			return;
		}
	}

	mtx_unlock(&cache->mutex);
}

///////////////////////////////////////////////////////////////////////////////

bool skr_tex_fmt_is_supported(skr_tex_fmt_ format, skr_tex_flags_ flags, int32_t multisample) {
	VkFormat vk_format = skr_tex_fmt_to_native(format);
	if (vk_format == VK_FORMAT_UNDEFINED) {
		return false;
	}

	// Check if this is a depth format
	bool is_depth = (
		format == skr_tex_fmt_depth16 ||
		format == skr_tex_fmt_depth32 ||
		format == skr_tex_fmt_depth32s8 ||
		format == skr_tex_fmt_depth24s8 ||
		format == skr_tex_fmt_depth16s8);

	// Build usage flags based on tex_flags
	VkImageUsageFlags usage = 0;

	if (flags & skr_tex_flags_writeable) {
		if (is_depth) { usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; } 
		else          { usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }
	}
	if (flags & skr_tex_flags_readable) { usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; }
	if (flags & skr_tex_flags_dynamic)  { usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT; }
	if (flags & skr_tex_flags_compute)  { usage |= VK_IMAGE_USAGE_STORAGE_BIT; }

	// Default: sampled texture
	if (usage == 0 || !(flags & skr_tex_flags_writeable)) {
		// For MSAA depth textures, don't add SAMPLED_BIT as it's often not supported
		bool is_msaa_depth = (multisample > 1) && is_depth;
		if (!is_msaa_depth) { usage |= VK_IMAGE_USAGE_SAMPLED_BIT; }
	}

	VkSampleCountFlagBits samples = (multisample > 1) ? (VkSampleCountFlagBits)multisample : VK_SAMPLE_COUNT_1_BIT;

	VkImageFormatProperties format_props;
	VkResult result = vkGetPhysicalDeviceImageFormatProperties(
		_skr_vk.physical_device,
		vk_format,
		VK_IMAGE_TYPE_2D,
		VK_IMAGE_TILING_OPTIMAL,
		usage,
		0,
		&format_props);

	if (result != VK_SUCCESS) {
		return false;
	}

	// Check if requested sample count is supported
	if (multisample > 1 && !(format_props.sampleCounts & samples)) {
		return false;
	}

	return true;
}

void skr_tex_fmt_block_info(skr_tex_fmt_ format, uint32_t* opt_out_block_width, uint32_t* opt_out_block_height, uint32_t* opt_out_bytes_per_block) {
	uint32_t block_w = 1, block_h = 1, block_bytes = 0;

	switch (format) {
		// BC formats (4x4 blocks)
		case skr_tex_fmt_bc1_rgb:
		case skr_tex_fmt_bc1_rgb_srgb:
		case skr_tex_fmt_bc4_r:
			block_w = 4; block_h = 4; block_bytes = 8; break;
		case skr_tex_fmt_bc3_rgba:
		case skr_tex_fmt_bc3_rgba_srgb:
		case skr_tex_fmt_bc5_rg:
		case skr_tex_fmt_bc7_rgba:
		case skr_tex_fmt_bc7_rgba_srgb:
			block_w = 4; block_h = 4; block_bytes = 16; break;

		// ETC formats (4x4 blocks)
		case skr_tex_fmt_etc1_rgb:
			block_w = 4; block_h = 4; block_bytes = 8; break;
		case skr_tex_fmt_etc2_rgba:
		case skr_tex_fmt_etc2_rgba_srgb:
			block_w = 4; block_h = 4; block_bytes = 16; break;
		case skr_tex_fmt_etc2_r11:
			block_w = 4; block_h = 4; block_bytes = 8; break;
		case skr_tex_fmt_etc2_rg11:
			block_w = 4; block_h = 4; block_bytes = 16; break;

		// PVRTC formats (variable block sizes, but typically 4x4 for 4bpp)
		case skr_tex_fmt_pvrtc1_rgb:
		case skr_tex_fmt_pvrtc1_rgb_srgb:
			block_w = 8; block_h = 4; block_bytes = 8; break; // 2bpp
		case skr_tex_fmt_pvrtc1_rgba:
		case skr_tex_fmt_pvrtc1_rgba_srgb:
		case skr_tex_fmt_pvrtc2_rgba:
		case skr_tex_fmt_pvrtc2_rgba_srgb:
			block_w = 4; block_h = 4; block_bytes = 8; break; // 4bpp

		// ASTC 4x4 (16 bytes per block)
		case skr_tex_fmt_astc4x4_rgba:
		case skr_tex_fmt_astc4x4_rgba_srgb:
			block_w = 4; block_h = 4; block_bytes = 16; break;

		// Uncompressed formats (1x1 "blocks")
		default:
			block_bytes = _skr_tex_fmt_to_size(format);
			break;
	}

	if (opt_out_block_width)      *opt_out_block_width      = block_w;
	if (opt_out_block_height)     *opt_out_block_height     = block_h;
	if (opt_out_bytes_per_block)  *opt_out_bytes_per_block  = block_bytes;
}

uint32_t skr_tex_calc_mip_count(skr_vec3i_t size) {
	int32_t max_dim = size.x > size.y ? size.x : size.y;
	if (size.z > max_dim) max_dim = size.z;
	if (max_dim < 1) return 0;

	uint32_t count = 1;
	while (max_dim > 1) {
		max_dim >>= 1;
		count++;
	}
	return count;
}

skr_vec3i_t skr_tex_calc_mip_dimensions(skr_vec3i_t base_size, uint32_t mip_level) {
	skr_vec3i_t result = {
		.x = base_size.x >> mip_level,
		.y = base_size.y >> mip_level,
		.z = base_size.z >> mip_level,
	};
	if (result.x < 1) result.x = 1;
	if (result.y < 1) result.y = 1;
	if (result.z < 1) result.z = 1;
	return result;
}

uint64_t skr_tex_calc_mip_size(skr_tex_fmt_ format, skr_vec3i_t base_size, uint32_t mip_level) {
	skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(base_size, mip_level);

	uint32_t block_w, block_h, block_bytes;
	skr_tex_fmt_block_info(format, &block_w, &block_h, &block_bytes);

	uint64_t blocks_x = (mip_size.x + block_w - 1) / block_w;
	uint64_t blocks_y = (mip_size.y + block_h - 1) / block_h;

	return blocks_x * blocks_y * mip_size.z * block_bytes;
}

///////////////////////////////////////////////////////////////////////////////
// External texture support (for wrapping VkImages from FFmpeg, etc.)
///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_tex_create_external(skr_tex_external_info_t info, skr_tex_t* out_tex) {
	if (!out_tex) return skr_err_invalid_parameter;
	if (info.image == VK_NULL_HANDLE) return skr_err_invalid_parameter;

	*out_tex = (skr_tex_t){};

	VkFormat vk_format = skr_tex_fmt_to_native(info.format);
	if (vk_format == VK_FORMAT_UNDEFINED) {
		skr_log(skr_log_warning, "skr_tex_create_external: unsupported format");
		return skr_err_unsupported;
	}

	// Handle MSAA sample count
	VkSampleCountFlagBits vk_samples = VK_SAMPLE_COUNT_1_BIT;
	if (info.multisample > 1) {
		switch (info.multisample) {
		case 2:  vk_samples = VK_SAMPLE_COUNT_2_BIT;  break;
		case 4:  vk_samples = VK_SAMPLE_COUNT_4_BIT;  break;
		case 8:  vk_samples = VK_SAMPLE_COUNT_8_BIT;  break;
		case 16: vk_samples = VK_SAMPLE_COUNT_16_BIT; break;
		default: vk_samples = VK_SAMPLE_COUNT_4_BIT;  break;  // Default to 4x
		}
	}

	// Handle array layers
	int32_t layer_count = info.array_layers > 1 ? info.array_layers : 1;
	bool is_array = layer_count > 1;

	// Store external image reference
	// Normalize size.z to 1 since external textures don't support 3D
	out_tex->image       = info.image;
	out_tex->memory      = info.memory;  // May be VK_NULL_HANDLE for truly external memory
	out_tex->size        = (skr_vec3i_t){ info.size.x, info.size.y, 1 };
	out_tex->format      = info.format;
	out_tex->flags       = is_array ? skr_tex_flags_array : 0;
	out_tex->samples     = vk_samples;
	out_tex->mip_levels  = 1;
	out_tex->layer_count = layer_count;
	out_tex->aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
	out_tex->is_external = !info.owns_image;  // If we don't own it, it's external

	// Layout tracking
	out_tex->current_layout       = info.current_layout;
	out_tex->current_queue_family = _skr_vk.graphics_queue_family;
	out_tex->first_use            = false;
	out_tex->is_transient_discard = false;

	// Create or use provided image view
	if (info.view != VK_NULL_HANDLE) {
		out_tex->view = info.view;
	} else {
		VkImageViewCreateInfo view_info = {
			.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image    = info.image,
			.viewType = is_array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
			.format   = vk_format,
			.subresourceRange = {
				.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = layer_count,
			},
		};

		VkResult vr = vkCreateImageView(_skr_vk.device, &view_info, NULL, &out_tex->view);
		if (vr != VK_SUCCESS) {
			skr_log(skr_log_critical, "skr_tex_create_external: vkCreateImageView failed");
			*out_tex = (skr_tex_t){};
			return skr_err_device_error;
		}
	}

	// Acquire sampler from cache
	out_tex->sampler_settings = info.sampler;
	out_tex->sampler          = _skr_sampler_cache_acquire(info.sampler);

	return skr_err_success;
}

skr_err_ skr_tex_update_external(skr_tex_t* ref_tex, skr_tex_external_update_t update) {
	if (!ref_tex) return skr_err_invalid_parameter;
	if (update.image == VK_NULL_HANDLE) return skr_err_invalid_parameter;

	// Destroy old view if we created it (not provided externally)
	// We always recreate the view for simplicity
	if (ref_tex->view != VK_NULL_HANDLE) {
		_skr_cmd_destroy_image_view(NULL, ref_tex->view);
		ref_tex->view = VK_NULL_HANDLE;
	}

	// Update image reference
	ref_tex->image = update.image;

	// Create or use provided image view
	if (update.view != VK_NULL_HANDLE) {
		ref_tex->view = update.view;
	} else {
		VkFormat vk_format = skr_tex_fmt_to_native(ref_tex->format);

		// Determine view type based on texture flags (match skr_tex_create_external behavior)
		VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
		if      (ref_tex->flags & skr_tex_flags_3d)      view_type = VK_IMAGE_VIEW_TYPE_3D;
		else if (ref_tex->flags & skr_tex_flags_cubemap) view_type = VK_IMAGE_VIEW_TYPE_CUBE;
		else if (ref_tex->flags & skr_tex_flags_array)   view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;

		VkImageViewCreateInfo view_info = {
			.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image    = update.image,
			.viewType = view_type,
			.format   = vk_format,
			.subresourceRange = {
				.aspectMask     = ref_tex->aspect_mask,
				.baseMipLevel   = 0,
				.levelCount     = ref_tex->mip_levels,
				.baseArrayLayer = 0,
				.layerCount     = ref_tex->layer_count,
			},
		};

		VkResult vr = vkCreateImageView(_skr_vk.device, &view_info, NULL, &ref_tex->view);
		if (vr != VK_SUCCESS) {
			skr_log(skr_log_critical, "skr_tex_update_external: vkCreateImageView failed");
			return skr_err_device_error;
		}
	}

	// Update layout tracking
	ref_tex->current_layout = update.current_layout;
	ref_tex->first_use      = false;

	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////
// Texture Copy and Readback
///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_tex_copy(const skr_tex_t* src, skr_tex_t* dst,
                      uint32_t src_mip, uint32_t src_layer,
                      uint32_t dst_mip, uint32_t dst_layer) {
	// Validate inputs
	if (!src || !dst) return skr_err_invalid_parameter;
	if (src->image == VK_NULL_HANDLE || dst->image == VK_NULL_HANDLE) return skr_err_invalid_parameter;

	// Check mip/layer bounds
	if (src_mip >= src->mip_levels || src_layer >= src->layer_count) {
		skr_log(skr_log_critical, "skr_tex_copy: source mip/layer out of bounds");
		return skr_err_invalid_parameter;
	}
	if (dst_mip >= dst->mip_levels || dst_layer >= dst->layer_count) {
		skr_log(skr_log_critical, "skr_tex_copy: destination mip/layer out of bounds");
		return skr_err_invalid_parameter;
	}

	// Calculate mip dimensions
	int32_t src_width  = src->size.x >> src_mip; if (src_width  < 1) src_width  = 1;
	int32_t src_height = src->size.y >> src_mip; if (src_height < 1) src_height = 1;
	int32_t dst_width  = dst->size.x >> dst_mip; if (dst_width  < 1) dst_width  = 1;
	int32_t dst_height = dst->size.y >> dst_mip; if (dst_height < 1) dst_height = 1;

	// For MSAA resolve, dimensions must match
	bool is_resolve = (src->samples > VK_SAMPLE_COUNT_1_BIT && dst->samples == VK_SAMPLE_COUNT_1_BIT);
	if (is_resolve && (src_width != dst_width || src_height != dst_height)) {
		skr_log(skr_log_critical, "skr_tex_copy: MSAA resolve requires matching dimensions");
		return skr_err_invalid_parameter;
	}

	// For regular copy, dimensions must also match (we don't support scaling)
	if (!is_resolve && (src_width != dst_width || src_height != dst_height)) {
		skr_log(skr_log_critical, "skr_tex_copy: dimensions must match (use blit for scaling)");
		return skr_err_invalid_parameter;
	}

	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

	// Transition source to TRANSFER_SRC_OPTIMAL
	_skr_tex_transition(ctx.cmd, (skr_tex_t*)src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	// Transition destination to TRANSFER_DST_OPTIMAL
	_skr_tex_transition(ctx.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

	if (is_resolve) {
		// MSAA resolve
		VkImageResolve resolve_region = {
			.srcSubresource = {
				.aspectMask     = src->aspect_mask,
				.mipLevel       = src_mip,
				.baseArrayLayer = src_layer,
				.layerCount     = 1,
			},
			.srcOffset = {0, 0, 0},
			.dstSubresource = {
				.aspectMask     = dst->aspect_mask,
				.mipLevel       = dst_mip,
				.baseArrayLayer = dst_layer,
				.layerCount     = 1,
			},
			.dstOffset = {0, 0, 0},
			.extent    = {src_width, src_height, 1},
		};

		vkCmdResolveImage(ctx.cmd,
			src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &resolve_region);
	} else {
		// Regular image copy
		VkImageCopy copy_region = {
			.srcSubresource = {
				.aspectMask     = src->aspect_mask,
				.mipLevel       = src_mip,
				.baseArrayLayer = src_layer,
				.layerCount     = 1,
			},
			.srcOffset = {0, 0, 0},
			.dstSubresource = {
				.aspectMask     = dst->aspect_mask,
				.mipLevel       = dst_mip,
				.baseArrayLayer = dst_layer,
				.layerCount     = 1,
			},
			.dstOffset = {0, 0, 0},
			.extent    = {src_width, src_height, 1},
		};

		vkCmdCopyImage(ctx.cmd,
			src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &copy_region);
	}

	// Transition both back to shader-readable layouts
	_skr_tex_transition_for_shader_read(ctx.cmd, (skr_tex_t*)src, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	_skr_tex_transition_for_shader_read(ctx.cmd, dst, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

	_skr_cmd_release(ctx.cmd);

	return skr_err_success;
}

skr_err_ skr_tex_create_copy(const skr_tex_t* src, skr_tex_fmt_ format, skr_tex_flags_ flags, int32_t multisample, skr_tex_t* out_tex) {
	if (!src || !out_tex)             return skr_err_invalid_parameter;
	if (src->image == VK_NULL_HANDLE) return skr_err_invalid_parameter;

	// Resolve parameters
	skr_tex_fmt_ dst_format  = (format      == skr_tex_fmt_none) ? src->format           : format;
	int32_t      dst_samples = (multisample == 0)                ? (int32_t)src->samples : multisample;

	// Create destination texture
	// Use the source's sampler settings for convenience
	skr_tex_sampler_t sampler = src->sampler_settings;

	skr_err_ err = skr_tex_create(dst_format, flags, sampler, src->size, dst_samples, src->mip_levels, NULL, out_tex);
	if (err != skr_err_success) {
		return err;
	}

	// Copy all mip levels and layers
	bool is_resolve = (src->samples > VK_SAMPLE_COUNT_1_BIT && dst_samples == 1);

	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

	// Transition source to TRANSFER_SRC_OPTIMAL
	_skr_tex_transition(ctx.cmd, (skr_tex_t*)src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	// Transition destination to TRANSFER_DST_OPTIMAL
	_skr_tex_transition(ctx.cmd, out_tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

	// For MSAA resolve, we can only resolve mip 0 (MSAA textures typically have 1 mip)
	uint32_t mip_count = is_resolve ? 1 : src->mip_levels;

	for (uint32_t mip = 0; mip < mip_count; mip++) {
		skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(src->size, mip);

		for (uint32_t layer = 0; layer < src->layer_count; layer++) {
			if (is_resolve) {
				VkImageResolve resolve_region = {
					.srcSubresource = {
						.aspectMask     = src->aspect_mask,
						.mipLevel       = mip,
						.baseArrayLayer = layer,
						.layerCount     = 1,
					},
					.srcOffset = {0, 0, 0},
					.dstSubresource = {
						.aspectMask     = out_tex->aspect_mask,
						.mipLevel       = mip,
						.baseArrayLayer = layer,
						.layerCount     = 1,
					},
					.dstOffset = {0, 0, 0},
					.extent    = {mip_size.x, mip_size.y, 1},
				};

				vkCmdResolveImage(ctx.cmd,
					src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					out_tex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1, &resolve_region);
			} else {
				VkImageCopy copy_region = {
					.srcSubresource = {
						.aspectMask     = src->aspect_mask,
						.mipLevel       = mip,
						.baseArrayLayer = layer,
						.layerCount     = 1,
					},
					.srcOffset = {0, 0, 0},
					.dstSubresource = {
						.aspectMask     = out_tex->aspect_mask,
						.mipLevel       = mip,
						.baseArrayLayer = layer,
						.layerCount     = 1,
					},
					.dstOffset = {0, 0, 0},
					.extent    = {mip_size.x, mip_size.y, 1},
				};

				vkCmdCopyImage(ctx.cmd,
					src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					out_tex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1, &copy_region);
			}
		}
	}

	// Transition both back to shader-readable layouts
	_skr_tex_transition_for_shader_read(ctx.cmd, (skr_tex_t*)src, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	_skr_tex_transition_for_shader_read(ctx.cmd, out_tex, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

	_skr_cmd_release(ctx.cmd);

	return skr_err_success;
}

// Internal structure for readback state
typedef struct _skr_tex_readback_internal_t {
	VkBuffer       staging_buffer;
	VkDeviceMemory staging_memory;
} _skr_tex_readback_internal_t;

skr_err_ skr_tex_readback(const skr_tex_t* tex, uint32_t mip_level, uint32_t array_layer, skr_tex_readback_t* out_readback) {
	if (!tex || !out_readback)        return skr_err_invalid_parameter;
	if (tex->image == VK_NULL_HANDLE) return skr_err_invalid_parameter;

	// Check mip/layer bounds
	if (mip_level >= tex->mip_levels || array_layer >= tex->layer_count) {
		skr_log(skr_log_critical, "skr_tex_readback: mip/layer out of bounds");
		return skr_err_invalid_parameter;
	}

	// MSAA textures cannot be directly read back - must resolve first
	if (tex->samples > VK_SAMPLE_COUNT_1_BIT) {
		skr_log(skr_log_critical, "skr_tex_readback: MSAA textures cannot be read back directly. Use skr_tex_copy to resolve first.");
		return skr_err_unsupported;
	}

	// Check readable flag
	if (!(tex->flags & skr_tex_flags_readable)) {
		skr_log(skr_log_critical, "skr_tex_readback: texture was not created with skr_tex_flags_readable");
		return skr_err_unsupported;
	}

	// size.z is always the actual depth (1 for non-3D textures)
	skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(tex->size, mip_level);

	// Calculate data size (handle compressed formats)
	uint32_t block_width, block_height, bytes_per_block;
	skr_tex_fmt_block_info(tex->format, &block_width, &block_height, &bytes_per_block);

	uint32_t blocks_x  = (mip_size.x + block_width  - 1) / block_width;
	uint32_t blocks_y  = (mip_size.y + block_height - 1) / block_height;
	uint32_t data_size = blocks_x * blocks_y * mip_size.z * bytes_per_block;

	if (data_size == 0) {
		skr_log(skr_log_critical, "skr_tex_readback: unsupported format or zero data size");
		return skr_err_unsupported;
	}

	// Create staging buffer with TRANSFER_DST usage
	VkBufferCreateInfo buffer_info = {
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = data_size,
		.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	VkBuffer staging_buffer;
	VkResult vr = vkCreateBuffer(_skr_vk.device, &buffer_info, NULL, &staging_buffer);
	if (vr != VK_SUCCESS) {
		skr_log(skr_log_critical, "skr_tex_readback: vkCreateBuffer failed");
		return skr_err_device_error;
	}

	// Allocate host-visible memory
	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(_skr_vk.device, staging_buffer, &mem_requirements);

	uint32_t memory_type_index = _skr_find_memory_type(_skr_vk.physical_device, mem_requirements,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	if (memory_type_index == UINT32_MAX) {
		vkDestroyBuffer(_skr_vk.device, staging_buffer, NULL);
		skr_log(skr_log_critical, "skr_tex_readback: no suitable memory type found");
		return skr_err_out_of_memory;
	}

	VkMemoryAllocateInfo alloc_info = {
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize  = mem_requirements.size,
		.memoryTypeIndex = memory_type_index,
	};

	VkDeviceMemory staging_memory;
	vr = vkAllocateMemory(_skr_vk.device, &alloc_info, NULL, &staging_memory);
	if (vr != VK_SUCCESS) {
		vkDestroyBuffer(_skr_vk.device, staging_buffer, NULL);
		skr_log(skr_log_critical, "skr_tex_readback: vkAllocateMemory failed");
		return skr_err_out_of_memory;
	}

	vkBindBufferMemory(_skr_vk.device, staging_buffer, staging_memory, 0);

	// Map the staging buffer (persistent mapping)
	void* mapped_data;
	vr = vkMapMemory(_skr_vk.device, staging_memory, 0, data_size, 0, &mapped_data);
	if (vr != VK_SUCCESS) {
		vkFreeMemory   (_skr_vk.device, staging_memory, NULL);
		vkDestroyBuffer(_skr_vk.device, staging_buffer, NULL);
		skr_log(skr_log_critical, "skr_tex_readback: vkMapMemory failed");
		return skr_err_device_error;
	}

	// Acquire command buffer and issue copy
	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

	// Transition texture to TRANSFER_SRC_OPTIMAL
	_skr_tex_transition(ctx.cmd, (skr_tex_t*)tex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	// Copy image to buffer
	VkBufferImageCopy copy_region = {
		.bufferOffset      = 0,
		.bufferRowLength   = 0,  // Tightly packed
		.bufferImageHeight = 0,
		.imageSubresource  = {
			.aspectMask     = tex->aspect_mask,
			.mipLevel       = mip_level,
			.baseArrayLayer = array_layer,
			.layerCount     = 1,
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {mip_size.x, mip_size.y, mip_size.z},
	};

	vkCmdCopyImageToBuffer(ctx.cmd, tex->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer, 1, &copy_region);

	// Transition texture back to shader-readable
	_skr_tex_transition_for_shader_read(ctx.cmd, (skr_tex_t*)tex, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

	// Get future before releasing command buffer
	skr_future_t future = skr_future_get();

	_skr_cmd_release(ctx.cmd);

	// Allocate internal state
	_skr_tex_readback_internal_t* internal = (_skr_tex_readback_internal_t*)_skr_malloc(sizeof(_skr_tex_readback_internal_t));
	if (!internal) {
		vkUnmapMemory  (_skr_vk.device, staging_memory);
		vkFreeMemory   (_skr_vk.device, staging_memory, NULL);
		vkDestroyBuffer(_skr_vk.device, staging_buffer, NULL);
		return skr_err_out_of_memory;
	}

	internal->staging_buffer = staging_buffer;
	internal->staging_memory = staging_memory;

	// Populate output
	out_readback->data      = mapped_data;
	out_readback->size      = data_size;
	out_readback->future    = future;
	out_readback->_internal = internal;

	return skr_err_success;
}

void skr_tex_readback_destroy(skr_tex_readback_t* ref_readback) {
	if (!ref_readback || !ref_readback->_internal) return;

	_skr_tex_readback_internal_t* internal = (_skr_tex_readback_internal_t*)ref_readback->_internal;

	// Wait for GPU to complete before destroying (in case user forgot)
	skr_future_wait(&ref_readback->future);

	// Unmap and free staging resources
	vkUnmapMemory  (_skr_vk.device, internal->staging_memory);
	vkFreeMemory   (_skr_vk.device, internal->staging_memory, NULL);
	vkDestroyBuffer(_skr_vk.device, internal->staging_buffer, NULL);

	_skr_free(internal);

	// Zero out the readback struct
	*ref_readback = (skr_tex_readback_t){0};
}
