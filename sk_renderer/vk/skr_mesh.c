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

///////////////////////////////////////////////////////////////////////////////
// Vertex type creation
///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_vert_type_create(const skr_vert_component_t* items, int32_t item_count, skr_vert_type_t* out_type) {
	if (!out_type) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_type = (skr_vert_type_t){};

	if (!items || item_count == 0) {
		skr_log(skr_log_warning, "Cannot create vertex type with no components");
		return skr_err_invalid_parameter;
	}

	// Allocate storage
	out_type->component_count = item_count;
	out_type->components      = _skr_malloc(sizeof(skr_vert_component_t) * item_count);
	out_type->attributes      = _skr_malloc(sizeof(VkVertexInputAttributeDescription) * item_count);

	if (!out_type->components || !out_type->attributes) {
		_skr_free(out_type->components);
		_skr_free(out_type->attributes);
		*out_type = (skr_vert_type_t){};
		return skr_err_out_of_memory;
	}

	memcpy(out_type->components, items, sizeof(skr_vert_component_t) * item_count);

	// Find max binding to determine how many bindings we need
	uint8_t max_binding = 0;
	for (int32_t i = 0; i < item_count; i++) {
		if (items[i].binding > max_binding) {
			max_binding = items[i].binding;
		}
	}
	out_type->binding_count = max_binding + 1;

	// Allocate bindings array
	out_type->bindings = _skr_malloc(sizeof(VkVertexInputBindingDescription) * out_type->binding_count);
	if (!out_type->bindings) {
		_skr_free(out_type->components);
		_skr_free(out_type->attributes);
		*out_type = (skr_vert_type_t){};
		return skr_err_out_of_memory;
	}

	// Calculate stride for each binding
	uint32_t binding_strides[16] = {0};  // Max 16 bindings (Vulkan limit)
	uint32_t binding_offsets[16] = {0};  // Track current offset per binding

	for (int32_t i = 0; i < item_count; i++) {
		uint32_t component_size = _skr_vert_fmt_to_size(items[i].format) * items[i].count;
		binding_strides[items[i].binding] += component_size;
	}

	// Create binding descriptions
	for (uint32_t b = 0; b < out_type->binding_count; b++) {
		out_type->bindings[b] = (VkVertexInputBindingDescription){
			.binding   = b,
			.stride    = binding_strides[b],
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};
	}

	// Populate attributes with per-binding offsets
	for (int32_t i = 0; i < item_count; i++) {
		uint8_t  binding        = items[i].binding;
		uint32_t component_size = _skr_vert_fmt_to_size(items[i].format) * items[i].count;

		out_type->attributes[i] = (VkVertexInputAttributeDescription){
			.location = i,
			.binding  = binding,
			.format   = _skr_to_vk_vert_fmt(items[i].format, items[i].count),
			.offset   = binding_offsets[binding],
		};

		binding_offsets[binding] += component_size;
	}

	// Register with pipeline system (vertex format owns this registration)
	out_type->pipeline_idx = _skr_pipeline_register_vertformat(*out_type);

	return skr_err_success;
}

bool skr_vert_type_is_valid(const skr_vert_component_t* type) {
	const skr_vert_type_t* vert_type = (const skr_vert_type_t*)type;
	return vert_type && vert_type->attributes != NULL && vert_type->component_count > 0;
}

void skr_vert_type_destroy(skr_vert_type_t* ref_type) {
	if (!ref_type) return;

	// Unregister from pipeline system
	if (ref_type->pipeline_idx >= 0) {
		_skr_pipeline_unregister_vertformat(ref_type->pipeline_idx);
	}

	_skr_free(ref_type->attributes);
	_skr_free(ref_type->bindings);
	_skr_free(ref_type->components);
	*ref_type = (skr_vert_type_t){};
}

///////////////////////////////////////////////////////////////////////////////
// Mesh creation
///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_mesh_create(const skr_vert_type_t* vert_type, skr_index_fmt_ ind_type, const void* vert_data, uint32_t vert_count, const void* opt_ind_data, uint32_t ind_count, skr_mesh_t* out_mesh) {
	if (!out_mesh) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_mesh = (skr_mesh_t){};

	if (!vert_type || vert_count == 0) {
		return skr_err_invalid_parameter;
	}

	// Set up mesh metadata
	out_mesh->vert_type  = vert_type;
	out_mesh->ind_format = ind_type;

	// Use the set functions to create the buffers
	skr_err_ err = skr_mesh_set_data(out_mesh, vert_data, vert_count, opt_ind_data, ind_count);
	if (err != skr_err_success) {
		*out_mesh = (skr_mesh_t){};
		return err;
	}

	return skr_err_success;
}

bool skr_mesh_is_valid(const skr_mesh_t* mesh) {
	if (!mesh) return false;
	if (mesh->vert_count > 0 || mesh->ind_count > 0) return true;
	for (uint32_t i = 0; i < mesh->vertex_buffer_count; i++) {
		if (skr_buffer_is_valid(&mesh->vertex_buffers[i])) {
			return true;
		}
	}
	return false;
}

void skr_mesh_destroy(skr_mesh_t* ref_mesh) {
	if (!ref_mesh) return;

	// Destroy only owned vertex buffers (externally-referenced buffers are not destroyed)
	for (uint32_t i = 0; i < ref_mesh->vertex_buffer_count; i++) {
		if (ref_mesh->vertex_buffer_owned & (1u << i)) {
			skr_buffer_destroy(&ref_mesh->vertex_buffers[i]);
		}
	}

	skr_buffer_destroy(&ref_mesh->index_buffer);
	*ref_mesh = (skr_mesh_t){};
}

uint32_t skr_mesh_get_vert_count(const skr_mesh_t* mesh) {
	return mesh ? mesh->vert_count : 0;
}

uint32_t skr_mesh_get_ind_count(const skr_mesh_t* mesh) {
	return mesh ? mesh->ind_count : 0;
}

void skr_mesh_set_name(skr_mesh_t* ref_mesh, const char* name) {
	if (!ref_mesh) return;

	// Name the vertex and index buffers with appropriate suffixes
	char buffer_name[256];
	for (uint32_t i = 0; i < ref_mesh->vertex_buffer_count; i++) {
		if (skr_buffer_is_valid(&ref_mesh->vertex_buffers[i])) {
			snprintf(buffer_name, sizeof(buffer_name), "verts%u_%s", i, name);
			skr_buffer_set_name(&ref_mesh->vertex_buffers[i], buffer_name);
		}
	}
	if (skr_buffer_is_valid(&ref_mesh->index_buffer)) {
		snprintf(buffer_name, sizeof(buffer_name), "indices_%s", name);
		skr_buffer_set_name(&ref_mesh->index_buffer, buffer_name);
	}
}

///////////////////////////////////////////////////////////////////////////////
// Mesh data update
///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_mesh_set_verts(skr_mesh_t* ref_mesh, const void* vert_data, uint32_t vert_count) {
	if (!ref_mesh) {
		return skr_err_invalid_parameter;
	}

	// If NULL data or 0 count, destroy buffers and just set count
	if (!vert_data || vert_count == 0) {
		for (uint32_t i = 0; i < ref_mesh->vertex_buffer_count; i++) {
			if (ref_mesh->vertex_buffer_owned & (1u << i)) {
				skr_buffer_destroy(&ref_mesh->vertex_buffers[i]);
			}
		}
		ref_mesh->vertex_buffer_count = 0;
		ref_mesh->vertex_buffer_owned = 0;
		ref_mesh->vert_count = vert_count;
		return skr_err_success;
	}

	// Need vert_type to create buffer
	if (!ref_mesh->vert_type) {
		return skr_err_invalid_parameter;
	}

	// For legacy single-buffer mode, we assume binding 0 contains all data
	// Calculate total stride from first binding (assumes interleaved data at binding 0)
	uint32_t vert_stride = ref_mesh->vert_type->bindings[0].stride;
	uint32_t vert_size   = vert_count * vert_stride;

	// Ensure we have at least one buffer slot in use
	if (ref_mesh->vertex_buffer_count == 0) {
		ref_mesh->vertex_buffer_count = 1;
	}

	// If buffer exists, check if we need to resize or convert to dynamic
	if (skr_buffer_is_valid(&ref_mesh->vertex_buffers[0])) {
		bool needs_resize = vert_size > ref_mesh->vertex_buffers[0].size;
		bool is_static    = ref_mesh->vertex_buffers[0].use & skr_use_static;

		if (is_static || needs_resize) {
			// Either static (convert to dynamic) or too small (resize)
			skr_buffer_destroy(&ref_mesh->vertex_buffers[0]);

			skr_err_ err = skr_buffer_create(vert_data, vert_count, vert_stride, skr_buffer_type_vertex, skr_use_dynamic, &ref_mesh->vertex_buffers[0]);
			if (err != skr_err_success) {
				skr_log(skr_log_critical, "Failed to create dynamic vertex buffer for mesh");
				return err;
			}
			ref_mesh->vertex_buffer_owned |= (1u << 0);  // Mark as owned
		} else {
			// Already dynamic and large enough, just update
			skr_buffer_set(&ref_mesh->vertex_buffers[0], vert_data, vert_size);
		}
	} else {
		// First time setting verts, create static buffer
		skr_err_ err = skr_buffer_create(vert_data, vert_count, vert_stride, skr_buffer_type_vertex, skr_use_static, &ref_mesh->vertex_buffers[0]);
		if (err != skr_err_success) {
			skr_log(skr_log_critical, "Failed to create vertex buffer for mesh");
			return err;
		}
		ref_mesh->vertex_buffer_owned |= (1u << 0);  // Mark as owned
	}

	ref_mesh->vert_count = vert_count;
	return skr_err_success;
}

skr_err_ skr_mesh_set_inds(skr_mesh_t* ref_mesh, const void* ind_data, uint32_t ind_count) {
	if (!ref_mesh) {
		return skr_err_invalid_parameter;
	}

	// If NULL data or 0 count, destroy buffer and just set count
	if (!ind_data || ind_count == 0) {
		if (skr_buffer_is_valid(&ref_mesh->index_buffer)) {
			skr_buffer_destroy(&ref_mesh->index_buffer);
		}
		ref_mesh->ind_count = ind_count;
		return skr_err_success;
	}

	uint32_t ind_stride = _skr_index_fmt_to_size(ref_mesh->ind_format);
	uint32_t ind_size   = ind_count * ind_stride;

	// If buffer exists, check if we need to resize or convert to dynamic
	if (skr_buffer_is_valid(&ref_mesh->index_buffer)) {
		bool needs_resize = ind_size > ref_mesh->index_buffer.size;
		bool is_static    = ref_mesh->index_buffer.use & skr_use_static;

		if (is_static || needs_resize) {
			// Either static (convert to dynamic) or too small (resize)
			skr_buffer_destroy(&ref_mesh->index_buffer);

			skr_err_ err = skr_buffer_create(ind_data, ind_count, ind_stride, skr_buffer_type_index, skr_use_dynamic, &ref_mesh->index_buffer);
			if (err != skr_err_success) {
				skr_log(skr_log_critical, "Failed to create dynamic index buffer for mesh");
				return err;
			}
		} else {
			// Already dynamic and large enough, just update
			skr_buffer_set(&ref_mesh->index_buffer, ind_data, ind_size);
		}
	} else {
		// First time setting inds, create static buffer
		ref_mesh->ind_format_vk = _skr_to_vk_index_fmt(ref_mesh->ind_format);
		skr_err_ err            = skr_buffer_create(ind_data, ind_count, ind_stride, skr_buffer_type_index, skr_use_static, &ref_mesh->index_buffer);
		if (err != skr_err_success) {
			skr_log(skr_log_critical, "Failed to create index buffer for mesh");
			return err;
		}
	}

	ref_mesh->ind_count = ind_count;
	return skr_err_success;
}

skr_err_ skr_mesh_set_data(skr_mesh_t* ref_mesh, const void* vert_data, uint32_t vert_count, const void* ind_data, uint32_t ind_count) {
	if (!ref_mesh) {
		return skr_err_invalid_parameter;
	}

	// Set vertices first
	if (vert_data && vert_count > 0) {
		skr_err_ err = skr_mesh_set_verts(ref_mesh, vert_data, vert_count);
		if (err != skr_err_success) {
			return err;
		}
	}

	// Set indices second
	if (ind_data && ind_count > 0) {
		skr_err_ err = skr_mesh_set_inds(ref_mesh, ind_data, ind_count);
		if (err != skr_err_success) {
			return err;
		}
	}

	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////
// Multi-buffer vertex API
///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_mesh_set_vertex_buffer(skr_mesh_t* ref_mesh, uint32_t binding, const skr_buffer_t* buffer, uint32_t vert_count) {
	if (!ref_mesh || !buffer) {
		return skr_err_invalid_parameter;
	}

	if (!ref_mesh->vert_type) {
		return skr_err_invalid_parameter;
	}

	// Validate binding index against fixed limit
	if (binding >= SKR_MAX_VERTEX_BUFFERS) {
		skr_log(skr_log_warning, "Binding %u exceeds max vertex buffers (%u)", binding, SKR_MAX_VERTEX_BUFFERS);
		return skr_err_invalid_parameter;
	}

	// Validate binding index against vertex type
	if (binding >= ref_mesh->vert_type->binding_count) {
		skr_log(skr_log_warning, "Binding %u exceeds vertex type binding count %u", binding, ref_mesh->vert_type->binding_count);
		return skr_err_invalid_parameter;
	}

	// Validate buffer type (must include vertex usage)
	if (!(buffer->type & skr_buffer_type_vertex)) {
		skr_log(skr_log_warning, "Buffer assigned to binding %u does not have vertex buffer type flag", binding);
		return skr_err_invalid_parameter;
	}

	// Validate buffer stride matches vertex type
	uint32_t expected_stride = ref_mesh->vert_type->bindings[binding].stride;
	if (buffer->size > 0 && vert_count > 0) {
		uint32_t actual_stride = buffer->size / vert_count;
		if (actual_stride != expected_stride) {
			skr_log(skr_log_warning, "Buffer stride mismatch for binding %u: expected %u, got %u",
			        binding, expected_stride, actual_stride);
			return skr_err_invalid_parameter;
		}
	}

	// If we previously owned the buffer at this binding, destroy it first
	if ((ref_mesh->vertex_buffer_owned & (1u << binding)) && skr_buffer_is_valid(&ref_mesh->vertex_buffers[binding])) {
		skr_buffer_destroy(&ref_mesh->vertex_buffers[binding]);
	}

	// Assign buffer reference (externally owned - we do NOT destroy it)
	ref_mesh->vertex_buffers[binding] = *buffer;
	ref_mesh->vertex_buffer_owned &= ~(1u << binding);  // Clear ownership bit (external buffer)

	// Update buffer count to include this binding
	if (binding >= ref_mesh->vertex_buffer_count) {
		ref_mesh->vertex_buffer_count = binding + 1;
	}

	ref_mesh->vert_count = vert_count;

	return skr_err_success;
}

skr_buffer_t* skr_mesh_get_vertex_buffer(const skr_mesh_t* mesh, uint32_t binding) {
	if (!mesh || binding >= SKR_MAX_VERTEX_BUFFERS || binding >= mesh->vertex_buffer_count) {
		return NULL;
	}
	return (skr_buffer_t*)&mesh->vertex_buffers[binding];
}
