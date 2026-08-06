// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "skr_pipeline.h"

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_render_list_create(skr_render_list_t* out_list) {
	if (out_list == NULL) return skr_err_invalid_parameter;
	memset(out_list, 0, sizeof(*out_list));

	out_list->capacity               = 64;
	out_list->items                  = (skr_render_item_t*)_skr_malloc(sizeof(skr_render_item_t) * out_list->capacity);
	out_list->instance_data_capacity = 4096;
	out_list->instance_data          = (uint8_t*)_skr_malloc(out_list->instance_data_capacity);
	out_list->material_data_capacity = 4096;
	out_list->material_data          = (uint8_t*)_skr_malloc(out_list->material_data_capacity);
	return skr_err_success;
}

void skr_render_list_destroy(skr_render_list_t* ref_list) {
	if (ref_list == NULL) return;
	_skr_free(ref_list->items);
	_skr_free(ref_list->instance_data);
	_skr_free(ref_list->material_data);
	memset(ref_list, 0, sizeof(*ref_list));
}

void skr_render_list_clear(skr_render_list_t* ref_list) {
	if (ref_list == NULL) return;
	ref_list->count              = 0;
	ref_list->instance_data_used = 0;
	ref_list->material_data_used = 0;
	ref_list->needs_sort         = false;
}

///////////////////////////////////////////////////////////////////////////////

// Sort key: material queue offset dominates, then pipeline state, then mesh —
// same spirit as the Vulkan backend's key
static uint64_t _skr_render_sort_key(const skr_material_t* material, WGPUBuffer first_vertex_buffer) {
	uint64_t queue = (uint64_t)(uint32_t)(material->queue_offset + 0x8000) & 0xFFFF;
	uint64_t pipe  = (uint64_t)(uint16_t)material->pipeline_material_idx;
	uint64_t mesh  = ((uint64_t)(uintptr_t)first_vertex_buffer >> 4) & 0xFFFFFFFF;
	return (queue << 48) | (pipe << 32) | mesh;
}

void skr_render_list_add(skr_render_list_t* ref_list, skr_mesh_t* mesh, skr_material_t* material, const void* opt_instance_data, uint32_t single_instance_data_size, uint32_t instance_count) {
	skr_render_list_add_indexed(ref_list, mesh, material, 0, 0, 0, opt_instance_data, single_instance_data_size, instance_count);
}

void skr_render_list_add_indexed(skr_render_list_t* ref_list, skr_mesh_t* mesh, skr_material_t* material, int32_t first_index, int32_t index_count, int32_t vertex_offset, const void* opt_instance_data, uint32_t single_instance_data_size, uint32_t instance_count) {
	if (ref_list == NULL || mesh == NULL || material == NULL) return;
	if (!skr_material_is_valid(material) || material->pipeline_material_idx < 0) return;

	// Vertex formats register lazily on first use; vertex-pulling meshes
	// (NULL vert_type) draw with no vertex buffers at all
	if (mesh->vert_type != NULL && mesh->vert_type->pipeline_idx < 0)
		((skr_vert_type_t*)mesh->vert_type)->pipeline_idx = _skr_pipeline_register_vertformat(mesh->vert_type);

	if (ref_list->count >= ref_list->capacity) {
		ref_list->capacity *= 2;
		ref_list->items     = (skr_render_item_t*)_skr_realloc(ref_list->items, sizeof(skr_render_item_t) * ref_list->capacity);
	}

	skr_render_item_t* item = &ref_list->items[ref_list->count++];
	memset(item, 0, sizeof(*item));

	for (uint32_t i = 0; i < mesh->vertex_buffer_count && i < SKR_MAX_VERTEX_BUFFERS; i++)
		item->vertex_buffers[i] = mesh->vertex_buffers[i].buffer;
	item->index_buffer      = mesh->index_buffer.buffer;
	item->vert_count        = mesh->vert_count;
	item->pipeline_vert_idx = mesh->vert_type ? (uint16_t)mesh->vert_type->pipeline_idx : (uint16_t)0xFFFF;

	item->pipeline_material_idx = (uint16_t)material->pipeline_material_idx;
	item->param_buffer_size     = (uint16_t)material->param_buffer_size;
	item->bind_start            = material->bind_start;
	item->bind_count            = (uint8_t)material->bind_count;

	item->flags = (mesh->ind_format_wgpu == WGPUIndexFormat_Uint32 ? skr_item_flag_index_32bit : 0)
	            | (mesh->vertex_buffer_count << skr_item_flag_vb_count_shift);

	if (material->instance_buffer_stride > 0 && single_instance_data_size != material->instance_buffer_stride)
		skr_log(skr_log_warning, "Instance data size mismatch: shader expects %u bytes, got %u bytes", material->instance_buffer_stride, single_instance_data_size);

	// Copy material params at an aligned offset so bind offsets stay legal
	uint32_t aligned_mat_offset = (ref_list->material_data_used + _SKR_OFFSET_ALIGN - 1) & ~(_SKR_OFFSET_ALIGN - 1);
	item->param_data_offset = aligned_mat_offset;
	if (material->param_buffer && material->param_buffer_size > 0) {
		uint32_t needed = aligned_mat_offset + material->param_buffer_size;
		while (needed > ref_list->material_data_capacity) {
			ref_list->material_data_capacity *= 2;
			ref_list->material_data = (uint8_t*)_skr_realloc(ref_list->material_data, ref_list->material_data_capacity);
		}
		memcpy(&ref_list->material_data[aligned_mat_offset], material->param_buffer, material->param_buffer_size);
		ref_list->material_data_used = aligned_mat_offset + material->param_buffer_size;
	}

	uint32_t aligned_inst_offset = (ref_list->instance_data_used + _SKR_OFFSET_ALIGN - 1) & ~(_SKR_OFFSET_ALIGN - 1);
	int32_t  resolved_index_count = index_count > 0 ? index_count : (int32_t)mesh->ind_count;
	item->sort_key           = _skr_render_sort_key(material, item->vertex_buffers[0]);
	item->instance_offset    = aligned_inst_offset;
	item->instance_data_size = (uint16_t)single_instance_data_size;
	item->instance_count     = instance_count > 0 ? instance_count : 1;
	item->first_index        = first_index;
	item->index_count        = resolved_index_count;
	item->vertex_offset      = vertex_offset;

	uint32_t total_size = single_instance_data_size * instance_count;
	if (opt_instance_data && total_size > 0) {
		uint32_t needed = aligned_inst_offset + total_size;
		while (needed > ref_list->instance_data_capacity) {
			ref_list->instance_data_capacity *= 2;
			ref_list->instance_data = (uint8_t*)_skr_realloc(ref_list->instance_data, ref_list->instance_data_capacity);
		}
		memcpy(&ref_list->instance_data[aligned_inst_offset], opt_instance_data, total_size);
		ref_list->instance_data_used = aligned_inst_offset + total_size;
	}

	ref_list->needs_sort = true;
}

///////////////////////////////////////////////////////////////////////////////

static int _skr_item_compare(const void* a, const void* b) {
	uint64_t ka = ((const skr_render_item_t*)a)->sort_key;
	uint64_t kb = ((const skr_render_item_t*)b)->sort_key;
	return ka < kb ? -1 : ka > kb ? 1 : 0;
}

// Draw-order sort. Instance/material data offsets are baked into items, so
// items can reorder freely without touching the data blobs.
void _skr_render_list_sort(skr_render_list_t* ref_list) {
	if (ref_list == NULL || !ref_list->needs_sort) return;
	qsort(ref_list->items, ref_list->count, sizeof(skr_render_item_t), _skr_item_compare);
	ref_list->needs_sort = false;
}
