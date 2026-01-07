// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"
#include "skr_pipeline.h"

#include <stdio.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Debug naming utilities
///////////////////////////////////////////////////////////////////////////////

void _skr_set_debug_name(VkDevice device, VkObjectType type, uint64_t handle, const char* name) {
	if (name == NULL || handle == 0 || vkSetDebugUtilsObjectNameEXT == NULL) return;

	vkSetDebugUtilsObjectNameEXT(device, &(VkDebugUtilsObjectNameInfoEXT){
		.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
		.objectType   = type,
		.objectHandle = handle,
		.pObjectName  = name,
	});
}

void _skr_append_vertex_format(char* ref_str, size_t str_size, const skr_vert_component_t* components, uint32_t component_count) {
	if (!ref_str || !components || component_count == 0) return;

	size_t pos = strlen(ref_str);

	for (uint32_t i = 0; i < component_count && pos < str_size - 3; i++) {
		char semantic = '?';
		switch (components[i].semantic) {
			case skr_semantic_position:     semantic = 'p'; break;
			case skr_semantic_texcoord:     semantic = 'u'; break;
			case skr_semantic_normal:       semantic = 'n'; break;
			case skr_semantic_color:        semantic = 'c'; break;
			case skr_semantic_tangent:      semantic = 't'; break;
			case skr_semantic_binormal:     semantic = 'b'; break;
			case skr_semantic_psize:        semantic = 's'; break;
			case skr_semantic_blendweight:  semantic = 'w'; break;
			case skr_semantic_blendindices: semantic = 'i'; break;
			default:                        semantic = '?'; break;
		}

		// Limit count to single digit (clamp to 0-9)
		uint8_t count = components[i].count > 9 ? 9 : components[i].count;

		ref_str[pos++] = semantic;
		ref_str[pos++] = '0' + count;
	}

	ref_str[pos] = '\0';
}

void _skr_append_material_config(char* ref_str, size_t str_size, const _skr_pipeline_material_key_t* mat_key) {
	if (!ref_str || !mat_key) return;

	// Cull mode
	const char* cull_str =
		mat_key->cull == skr_cull_back  ? "b" :
		mat_key->cull == skr_cull_front ? "f" :
		mat_key->cull == skr_cull_none  ? "n" : "?";

	// Depth test
	const char* depth_str =
		mat_key->depth_test == skr_compare_none         ? "!" :
		mat_key->depth_test == skr_compare_less         ? "<" :
		mat_key->depth_test == skr_compare_less_or_eq   ? "<=" :
		mat_key->depth_test == skr_compare_greater      ? ">" :
		mat_key->depth_test == skr_compare_greater_or_eq? ">=" :
		mat_key->depth_test == skr_compare_equal        ? "=" :
		mat_key->depth_test == skr_compare_always       ? "==" : "?";

	// Blend mode - check if blending is enabled
	bool blend_enabled = (mat_key->blend_state.src_color_factor != skr_blend_one ||
	                      mat_key->blend_state.dst_color_factor != skr_blend_zero);
	const char* blend_str = blend_enabled ? "b" : "o";
	if (mat_key->alpha_to_coverage) blend_str = "a2c";

	// Write mask (compact: just show what's written)
	char write_str[16];
	int write_pos = 0;
	if (mat_key->write_mask & skr_write_r)       write_str[write_pos++] = 'r';
	if (mat_key->write_mask & skr_write_g)       write_str[write_pos++] = 'g';
	if (mat_key->write_mask & skr_write_b)       write_str[write_pos++] = 'b';
	if (mat_key->write_mask & skr_write_a)       write_str[write_pos++] = 'a';
	if (mat_key->write_mask & skr_write_depth)   write_str[write_pos++] = 'd';
	if (mat_key->write_mask & skr_write_stencil) write_str[write_pos++] = 's';
	write_str[write_pos] = '\0';

	size_t pos = strlen(ref_str);
	snprintf(ref_str + pos, str_size - pos, "%s%s%s-%s", cull_str, depth_str, blend_str, write_str);
}

void _skr_append_renderpass_config(char* ref_str, size_t str_size, const skr_pipeline_renderpass_key_t* rp_key) {
	if (!ref_str || !rp_key) return;

	const char* color_str =
		rp_key->color_format == VK_FORMAT_UNDEFINED               ? "none" :
		rp_key->color_format == VK_FORMAT_B8G8R8A8_SRGB           ? "bgra8_srgb" :
		rp_key->color_format == VK_FORMAT_B8G8R8A8_UNORM          ? "bgra8" :
		rp_key->color_format == VK_FORMAT_R8G8B8A8_SRGB           ? "rgba8_srgb" :
		rp_key->color_format == VK_FORMAT_R8G8B8A8_UNORM          ? "rgba8" :
		rp_key->color_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ? "rg11b10" :
		rp_key->color_format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32  ? "rgb9e5" :
		rp_key->color_format == VK_FORMAT_R16G16B16A16_SFLOAT     ? "rgba16f" :
		rp_key->color_format == VK_FORMAT_R32G32B32A32_SFLOAT     ? "rgba32f" : "?";

	const char* depth_str =
		rp_key->depth_format == VK_FORMAT_UNDEFINED          ? "no-depth" :
		rp_key->depth_format == VK_FORMAT_D16_UNORM          ? "d16" :
		rp_key->depth_format == VK_FORMAT_D16_UNORM_S8_UINT  ? "d16s8" :
		rp_key->depth_format == VK_FORMAT_D24_UNORM_S8_UINT  ? "d24s8" :
		rp_key->depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT ? "d32s8" :
		rp_key->depth_format == VK_FORMAT_D32_SFLOAT         ? "d32" : "?";

	size_t pos = strlen(ref_str);
	snprintf(ref_str + pos, str_size - pos, "%s_%s_x%d", color_str, depth_str, rp_key->samples);
}

static const char* _skr_descriptor_type_name(VkDescriptorType type) {
	switch (type) {
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:         return "UniformBuffer";
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:         return "StorageBuffer";
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "Texture";
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:          return "StorageImage";
		default:                                        return "Unknown";
	}
}

void _skr_log_descriptor_writes(
	const VkWriteDescriptorSet*   writes,
	const VkDescriptorBufferInfo* buffer_infos,
	const VkDescriptorImageInfo*  image_infos,
	uint32_t write_ct,
	uint32_t buffer_ct,
	uint32_t image_ct)
{
	skr_log(skr_log_info, "=== Descriptor Writes ===");
	skr_log(skr_log_info, "Total writes: %u | Buffers: %u | Images: %u", write_ct, buffer_ct, image_ct);

	if (write_ct == 0) {
		skr_log(skr_log_info, "(No descriptors to bind)");
		return;
	}

	skr_log(skr_log_info, "");

	for (uint32_t i = 0; i < write_ct; i++) {
		const VkWriteDescriptorSet* w = &writes[i];
		const char* type_name = _skr_descriptor_type_name(w->descriptorType);

		skr_log(skr_log_info, "  [%2u] Binding %-2u | %-16s | Count: %u",
			i, w->dstBinding, type_name, w->descriptorCount);

		// Show buffer details
		if (w->pBufferInfo) {
			for (uint32_t j = 0; j < w->descriptorCount; j++) {
				const VkDescriptorBufferInfo* buf = &w->pBufferInfo[j];
				skr_log(skr_log_info, "       └─ Buffer: %p | Offset: %llu | Range: %llu",
					(void*)buf->buffer, (unsigned long long)buf->offset, (unsigned long long)buf->range);
			}
		}

		// Show image details
		if (w->pImageInfo) {
			for (uint32_t j = 0; j < w->descriptorCount; j++) {
				const VkDescriptorImageInfo* img = &w->pImageInfo[j];
				const char* layout_name =
					img->imageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? "ReadOnly" :
					img->imageLayout == VK_IMAGE_LAYOUT_GENERAL ? "General" :
					"Other";
				skr_log(skr_log_info, "       └─ Image: %p | Sampler: %p | Layout: %s",
					(void*)img->imageView, (void*)img->sampler, layout_name);
			}
		}
	}

	skr_log(skr_log_info, "=========================");
}