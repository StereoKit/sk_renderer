// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "sk_renderer.h"
#include "_sk_renderer.h"

#include "skr_vulkan.h"

#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_render_list_create(skr_render_list_t* out_list) {
	if (!out_list) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_list = (skr_render_list_t){};

	out_list->capacity                       = 16;
	out_list->items                          = _skr_malloc(sizeof(skr_render_item_t) * out_list->capacity);
	out_list->instance_data_capacity         = 1024;
	out_list->instance_data                  = _skr_malloc(out_list->instance_data_capacity);
	out_list->instance_data_sorted_capacity  = 1024;
	out_list->instance_data_sorted           = _skr_malloc(out_list->instance_data_sorted_capacity);
	out_list->material_data_capacity         = 1024;
	out_list->material_data                  = _skr_malloc(out_list->material_data_capacity);

	if (!out_list->items || !out_list->instance_data || !out_list->instance_data_sorted || !out_list->material_data) {
		skr_log(skr_log_critical, "Failed to allocate render list");
		_skr_free(out_list->items);
		_skr_free(out_list->instance_data);
		_skr_free(out_list->instance_data_sorted);
		_skr_free(out_list->material_data);
		*out_list = (skr_render_list_t){};
		return skr_err_out_of_memory;
	}

	return skr_err_success;
}

void skr_render_list_destroy(skr_render_list_t* ref_list) {
	if (!ref_list) return;

	_skr_free(ref_list->instance_data);
	_skr_free(ref_list->instance_data_sorted);
	_skr_free(ref_list->material_data);
	_skr_free(ref_list->items);
	*ref_list = (skr_render_list_t){};
}

void skr_render_list_clear(skr_render_list_t* ref_list) {
	if (!ref_list) return;
	ref_list->count = 0;
	ref_list->instance_data_used = 0;
	ref_list->material_data_used = 0;
	ref_list->needs_sort = false;
}

// Sort key layout (64 bits, ascending sort):
// Bits 63-32 (32 bits): alpha_mode * 10000 + queue_offset (separates opaque/a2c/transparent)
// Bits 31-16 (16 bits): pipeline_material_idx (groups by shader/render state)
// Bits 15-0  (16 bits): mesh pointer hash (groups same mesh for instancing)
static inline uint64_t _skr_render_sort_key(skr_material_t* material, VkBuffer first_vertex_buffer) {
	// Derive alpha mode: 0 = opaque, 1 = alpha-to-coverage, 2 = transparent
	uint32_t alpha_mode = 0;
	if (material->key.alpha_to_coverage) {
		alpha_mode = 1;
	} else if (material->key.blend_state.dst_color_factor != skr_blend_zero) {
		alpha_mode = 2;
	}
	// Combine alpha mode and queue_offset into sections (bias queue to handle negatives)
	uint32_t queue   = alpha_mode * 10000 + (uint32_t)(material->queue_offset + 1000);
	uint16_t mat_idx = (uint16_t)material->pipeline_material_idx;
	// Use VkBuffer handle bits for mesh grouping (shift past alignment, take 16 bits)
	uint16_t mesh_id = (uint16_t)((uintptr_t)first_vertex_buffer >> 4);
	return ((uint64_t)queue << 32) | ((uint64_t)mat_idx << 16) | mesh_id;
}

void skr_render_list_add_indexed(skr_render_list_t* ref_list, skr_mesh_t* mesh, skr_material_t* material, int32_t first_index, int32_t index_count, int32_t vertex_offset, const void* opt_instance_data, uint32_t single_instance_data_size, uint32_t instance_count) {
	if (!ref_list || !mesh || !material) return;

	// Grow if needed
	if (ref_list->count >= ref_list->capacity) {
		uint32_t           new_capacity = ref_list->capacity * 2;
		skr_render_item_t* new_items    = _skr_realloc(ref_list->items, sizeof(skr_render_item_t) * new_capacity);
		if (!new_items) {
			skr_log(skr_log_critical, "Failed to grow render list");
			return;
		}
		ref_list->items    = new_items;
		ref_list->capacity = new_capacity;
	}

	// Add item - copy mesh/material data so originals can be destroyed
	skr_render_item_t* item = &ref_list->items[ref_list->count++];

	// Copy mesh Vulkan handles
	item->vertex_buffer_count = (uint8_t)mesh->vertex_buffer_count;
	for (uint32_t i = 0; i < mesh->vertex_buffer_count && i < SKR_MAX_VERTEX_BUFFERS; i++) {
		item->vertex_buffers[i] = mesh->vertex_buffers[i].buffer;
	}
	item->index_buffer      = mesh->index_buffer.buffer;
	item->index_format      = (uint8_t)mesh->ind_format_vk;
	item->vert_count        = mesh->vert_count;
	item->ind_count         = mesh->ind_count;
	item->pipeline_vert_idx = (uint16_t)mesh->vert_type->pipeline_idx;

	// Copy material data
	item->pipeline_material_idx  = (uint16_t)material->pipeline_material_idx;
	item->param_buffer_size      = (uint16_t)material->param_buffer_size;
	item->has_system_buffer      = material->has_system_buffer ? 1 : 0;
	item->instance_buffer_stride = (uint16_t)material->instance_buffer_stride;
	item->bind_start             = material->bind_start;
	item->bind_count             = (uint8_t)material->bind_count;

	// Copy material param_buffer data (so material can be destroyed after add)
	item->param_data_offset = ref_list->material_data_used;
	if (material->param_buffer && material->param_buffer_size > 0) {
		// Resize material data if needed
		while (ref_list->material_data_used + material->param_buffer_size > ref_list->material_data_capacity) {
			uint32_t new_capacity = ref_list->material_data_capacity * 2;
			uint8_t* new_data     = _skr_realloc(ref_list->material_data, new_capacity);
			if (!new_data) {
				skr_log(skr_log_critical, "Failed to grow render list material data");
				return;
			}
			ref_list->material_data          = new_data;
			ref_list->material_data_capacity = new_capacity;
		}
		memcpy(&ref_list->material_data[ref_list->material_data_used], material->param_buffer, material->param_buffer_size);
		ref_list->material_data_used += material->param_buffer_size;
	}

	// Render item data
	item->sort_key           = _skr_render_sort_key(material, item->vertex_buffers[0]);
	item->instance_offset    = ref_list->instance_data_used;
	item->instance_data_size = (uint16_t)single_instance_data_size;
	item->instance_count     = instance_count;
	item->first_index        = first_index;
	item->index_count        = index_count;
	item->vertex_offset      = vertex_offset;

	// Copy instance data if provided
	uint32_t total_size = single_instance_data_size * instance_count;
	if (opt_instance_data && total_size > 0) {
		// Resize instance data if needed
		while (ref_list->instance_data_used + total_size > ref_list->instance_data_capacity) {
			uint32_t new_capacity = ref_list->instance_data_capacity * 2;
			uint8_t* new_data     = _skr_realloc(ref_list->instance_data, new_capacity);
			if (!new_data) {
				skr_log(skr_log_critical, "Failed to grow render list instance data");
				return;
			}
			ref_list->instance_data          = new_data;
			ref_list->instance_data_capacity = new_capacity;
		}
		memcpy(&ref_list->instance_data[ref_list->instance_data_used], opt_instance_data, total_size);
		ref_list->instance_data_used += total_size;
	}

	// Mark list as needing sort
	ref_list->needs_sort = true;
}

void skr_render_list_add(skr_render_list_t* ref_list, skr_mesh_t* mesh, skr_material_t* material, const void* opt_instance_data, uint32_t single_instance_data_size, uint32_t instance_count) {
	// Call indexed version with default offsets (draw entire mesh)
	skr_render_list_add_indexed(ref_list, mesh, material, 0, 0, 0, opt_instance_data, single_instance_data_size, instance_count);
}

static int _skr_render_item_compare(const void* a, const void* b) {
	const skr_render_item_t* item_a = (const skr_render_item_t*)a;
	const skr_render_item_t* item_b = (const skr_render_item_t*)b;

	// Primary: sort by pre-computed key
	if (item_a->sort_key < item_b->sort_key) return -1;
	if (item_a->sort_key > item_b->sort_key) return  1;

	// Secondary: indexed draw parameters (uncommon, only for sub-mesh draws)
	if (item_a->first_index < item_b->first_index) return -1;
	if (item_a->first_index > item_b->first_index) return  1;

	if (item_a->index_count < item_b->index_count) return -1;
	if (item_a->index_count > item_b->index_count) return  1;

	if (item_a->vertex_offset < item_b->vertex_offset) return -1;
	if (item_a->vertex_offset > item_b->vertex_offset) return  1;

	return 0;
}

void _skr_render_list_sort(skr_render_list_t* ref_list) {
	if (!ref_list || !ref_list->needs_sort || ref_list->count == 0) return;

	qsort(ref_list->items, ref_list->count, sizeof(skr_render_item_t), _skr_render_item_compare);
	ref_list->needs_sort = false;

	// After sorting, instance_offset values no longer match the sorted order
	// Rebuild instance data in sorted order
	if (ref_list->instance_data_used > 0) {
		// Keep sorted buffer same size as instance_data buffer
		if (ref_list->instance_data_sorted_capacity != ref_list->instance_data_capacity) {
			_skr_free(ref_list->instance_data_sorted);
			ref_list->instance_data_sorted          = _skr_malloc(ref_list->instance_data_capacity);
			ref_list->instance_data_sorted_capacity = ref_list->instance_data_capacity;
			if (!ref_list->instance_data_sorted) {
				skr_log(skr_log_critical, "Failed to allocate render list sorted instance data");
				ref_list->instance_data_sorted_capacity = 0;
				return;
			}
		}

		// Copy instance data in sorted order and update offsets
		// Batch consecutive runs to minimize memcpy calls
		uint32_t sorted_offset = 0;
		uint32_t i = 0;
		while (i < ref_list->count) {
			skr_render_item_t* item = &ref_list->items[i];
			uint32_t           size = item->instance_data_size * item->instance_count;

			if (size > 0) {
				// Find run of consecutive items in source buffer
				uint32_t run_start_src = item->instance_offset;
				uint32_t run_start_dst = sorted_offset;
				uint32_t run_size      = size;
				uint32_t run_items     = 1;

				item->instance_offset = sorted_offset;
				sorted_offset += size;

				// Check if next items are consecutive in source
				while (i + run_items < ref_list->count) {
					skr_render_item_t* next_item = &ref_list->items[i + run_items];
					uint32_t           next_size = next_item->instance_data_size * next_item->instance_count;

					if (next_size > 0 && next_item->instance_offset == run_start_src + run_size) {
						next_item->instance_offset = sorted_offset;
						sorted_offset += next_size;
						run_size      += next_size;
						run_items++;
					} else {
						break;
					}
				}

				// Copy the entire run at once
				memcpy(&ref_list->instance_data_sorted[run_start_dst], &ref_list->instance_data[run_start_src], run_size);

				i += run_items;
			} else {
				i++;
			}
		}

		// Swap the buffers
		uint8_t* temp = ref_list->instance_data;
		ref_list->instance_data        = ref_list->instance_data_sorted;
		ref_list->instance_data_sorted = temp;
	}
}
