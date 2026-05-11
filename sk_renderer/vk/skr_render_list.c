// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "sk_renderer.h"
#include "_sk_renderer.h"

#include "skr_vulkan.h"

#include <stdlib.h>
#include <string.h>

// Cross-platform prefetch hint. No-op where the compiler doesn't support it.
// Used by the radix scatter loops below to hide the latency of writing into
// random-access destination buckets.
#if defined(__GNUC__) || defined(__clang__)
	#define _SKR_PREFETCH(p) __builtin_prefetch((const void*)(p))
#elif defined(_MSC_VER)
	#include <intrin.h>
	#if defined(_M_ARM) || defined(_M_ARM64)
		#define _SKR_PREFETCH(p) __prefetch((const void*)(p))
	#else
		#define _SKR_PREFETCH(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
	#endif
#else
	#define _SKR_PREFETCH(p) ((void)0)
#endif

_Static_assert(sizeof(skr_render_item_t) <= 80, "skr_render_item_t grew beyond 80 bytes!");

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_render_list_create(skr_render_list_t* out_list) {
	if (!out_list) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_list = (skr_render_list_t){0};

	out_list->capacity                       = 16;
	out_list->items                          = _skr_malloc(sizeof(skr_render_item_t) * out_list->capacity);
	out_list->items_tmp                      = _skr_malloc(sizeof(skr_render_item_t) * out_list->capacity);
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
		*out_list = (skr_render_list_t){0};
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
	_skr_free(ref_list->items_tmp);
	_skr_free(ref_list->sort_scratch_a);
	_skr_free(ref_list->sort_scratch_b);
	*ref_list = (skr_render_list_t){0};
}

void skr_render_list_clear(skr_render_list_t* ref_list) {
	if (!ref_list) return;
	ref_list->count = 0;
	ref_list->instance_data_used = 0;
	ref_list->material_data_used = 0;
	ref_list->needs_sort = false;
}

// Sort key layout (64 bits, ascending sort):
// Bits 63-54 (10): queue          — (alpha_mode << 8) | (queue_offset + 128), queue_offset ∈ [-128, 127]
// Bits 53-40 (14): pipeline_idx   — pipeline cache index (up to 16384)
// Bits 39-22 (18): material_id    — low bits of bind_start; unique per material instance
// Bits 21-10 (12): mesh_id        — hash of first vertex buffer pointer (up to 4096)
// Bits 9-0   (10): sub_hash       — hash(first_index, index_count, vertex_offset) sub-mesh disambiguator
//
// material_id uses bind_start directly rather than a hash. bind_start values come from a sequential
// pool allocator so they're dense small integers — for any realistic scene the low 18 bits are
// already unique per material. Hashing into a narrower field would introduce birthday-paradox
// collisions that interleave unrelated materials in the sorted list, which is exactly the
// spread-out-draws symptom we're trying to avoid. 18 bits covers ~50K materials with 5 binds each.
static inline uint64_t _skr_render_sort_key(skr_material_t* material, VkBuffer first_vertex_buffer, int32_t first_index, int32_t index_count, int32_t vertex_offset) {
	// Derive alpha mode: 0 = opaque, 1 = alpha-to-coverage, 2 = transparent
	uint32_t alpha_mode = 0;
	if (material->key.alpha_to_coverage) {
		alpha_mode = 1;
	} else if (material->key.blend_state.dst_color_factor != skr_blend_zero) {
		alpha_mode = 2;
	}
	// Bit-pack alpha_mode and queue_offset into 10 bits: 2 bits alpha_mode (high) + 8 bits biased offset.
	// queue_offset is biased by +128 so negative values stay positive in a uint8.
	uint64_t queue       = ((uint64_t)alpha_mode << 8) | (uint64_t)((material->queue_offset + 128) & 0xFF);
	uint64_t pipeline_id = (uint64_t)(material->pipeline_material_idx)                    & 0x3FFF;
	uint64_t material_id = (uint64_t)(material->bind_start)                               & 0x3FFFF;

	uint64_t mesh_id = (uint64_t)((uintptr_t)first_vertex_buffer >> 4) & 0xFFF;

	// Hash draw params into 10 bits to disambiguate different draw ranges of the same mesh.
	uint32_t sub = (uint32_t)first_index ^ ((uint32_t)index_count * 2654435761u) ^ ((uint32_t)vertex_offset * 2246822519u);
	uint64_t sub_hash = (uint64_t)((sub ^ (sub >> 10)) & 0x3FF);

	return (queue << 54) | (pipeline_id << 40) | (material_id << 22) | (mesh_id << 10) | sub_hash;
}

void skr_render_list_add_indexed(skr_render_list_t* ref_list, skr_mesh_t* mesh, skr_material_t* material, int32_t first_index, int32_t index_count, int32_t vertex_offset, const void* opt_instance_data, uint32_t single_instance_data_size, uint32_t instance_count) {
	if (!ref_list || !mesh || !material) return;

	// Grow both items and items_tmp together (they get swapped during sort)
	if (ref_list->count >= ref_list->capacity) {
		uint32_t           new_capacity = ref_list->capacity * 2;
		skr_render_item_t* new_items     = _skr_realloc(ref_list->items,     sizeof(skr_render_item_t) * new_capacity);
		skr_render_item_t* new_items_tmp = _skr_realloc(ref_list->items_tmp, sizeof(skr_render_item_t) * new_capacity);
		if (!new_items || !new_items_tmp) {
			skr_log(skr_log_critical, "Failed to grow render list");
			return;
		}
		ref_list->items     = new_items;
		ref_list->items_tmp = new_items_tmp;
		ref_list->capacity  = new_capacity;
	}

	// Add item - copy mesh/material data so originals can be destroyed
	skr_render_item_t* item = &ref_list->items[ref_list->count++];

	// Copy mesh Vulkan handles
	for (uint32_t i = 0; i < mesh->vertex_buffer_count && i < SKR_MAX_VERTEX_BUFFERS; i++) {
		item->vertex_buffers[i] = mesh->vertex_buffers[i].buffer;
	}
	item->index_buffer      = mesh->index_buffer.buffer;
	item->vert_count        = mesh->vert_count;
	item->pipeline_vert_idx = (uint16_t)mesh->vert_type->pipeline_idx;

	// Copy material data
	item->pipeline_material_idx = (uint16_t)material->pipeline_material_idx;
	item->param_buffer_size     = (uint16_t)material->param_buffer_size;
	item->bind_start            = material->bind_start;
	item->bind_count            = (uint8_t)material->bind_count;

	// Pack flags (vertex_buffer_count in bits 2-3)
	item->flags = (material->has_system_buffer       ? skr_item_flag_system_buffer : 0)
	            | (mesh->ind_format_vk               ? skr_item_flag_index_32bit   : 0)
	            | (material->instance_buffer_stride   ? skr_item_flag_instance_buffer  : 0)
	            | (mesh->vertex_buffer_count << skr_item_flag_vb_count_shift);

	// Validate instance_buffer_stride at add-time (removed from item)
	if (material->instance_buffer_stride > 0 && single_instance_data_size != material->instance_buffer_stride) {
		skr_log(skr_log_warning, "Instance data size mismatch: shader expects %u bytes, got %u bytes",
			material->instance_buffer_stride, single_instance_data_size);
	}

	// Copy material param_buffer data (so material can be destroyed after add)
	// Align offset for uniform buffer access (minUniformBufferOffsetAlignment)
	uint32_t ubo_align          = _skr_vk.min_ubo_offset_align;
	uint32_t aligned_mat_offset = (ref_list->material_data_used + ubo_align - 1) & ~(ubo_align - 1);
	item->param_data_offset     = aligned_mat_offset;
	if (material->param_buffer && material->param_buffer_size > 0) {
		uint32_t needed = aligned_mat_offset + material->param_buffer_size;
		// Resize material data if needed
		while (needed > ref_list->material_data_capacity) {
			uint32_t new_capacity = ref_list->material_data_capacity * 2;
			uint8_t* new_data     = _skr_realloc(ref_list->material_data, new_capacity);
			if (!new_data) {
				skr_log(skr_log_critical, "Failed to grow render list material data");
				return;
			}
			ref_list->material_data          = new_data;
			ref_list->material_data_capacity = new_capacity;
		}
		memcpy(&ref_list->material_data[aligned_mat_offset], material->param_buffer, material->param_buffer_size);
		ref_list->material_data_used = aligned_mat_offset + material->param_buffer_size;
	}

	// Render item data
	// Align instance offset for storage buffer access (minStorageBufferOffsetAlignment)
	uint32_t ssbo_align          = _skr_vk.min_ssbo_offset_align;
	uint32_t aligned_inst_offset = (ref_list->instance_data_used + ssbo_align - 1) & ~(ssbo_align - 1);
	// Resolve index_count at add-time: 0 means "use mesh default"
	int32_t resolved_index_count  = index_count > 0 ? index_count : (int32_t)mesh->ind_count;
	item->sort_key               = _skr_render_sort_key(material, item->vertex_buffers[0], first_index, resolved_index_count, vertex_offset);
	item->instance_offset        = aligned_inst_offset;
	item->instance_data_size     = (uint16_t)single_instance_data_size;
	item->instance_count         = instance_count;
	item->first_index            = first_index;
	item->index_count            = resolved_index_count;
	item->vertex_offset          = vertex_offset;

	// Copy instance data if provided
	uint32_t total_size = single_instance_data_size * instance_count;
	if (opt_instance_data && total_size > 0) {
		uint32_t needed = aligned_inst_offset + total_size;
		// Resize instance data if needed
		while (needed > ref_list->instance_data_capacity) {
			uint32_t new_capacity = ref_list->instance_data_capacity * 2;
			uint8_t* new_data     = _skr_realloc(ref_list->instance_data, new_capacity);
			if (!new_data) {
				skr_log(skr_log_critical, "Failed to grow render list instance data");
				return;
			}
			ref_list->instance_data          = new_data;
			ref_list->instance_data_capacity = new_capacity;
		}
		memcpy(&ref_list->instance_data[aligned_inst_offset], opt_instance_data, total_size);
		ref_list->instance_data_used = aligned_inst_offset + total_size;
	}

	// Mark list as needing sort
	ref_list->needs_sort = true;
}

void skr_render_list_add(skr_render_list_t* ref_list, skr_mesh_t* mesh, skr_material_t* material, const void* opt_instance_data, uint32_t single_instance_data_size, uint32_t instance_count) {
	// Call indexed version with default offsets (draw entire mesh)
	skr_render_list_add_indexed(ref_list, mesh, material, 0, 0, 0, opt_instance_data, single_instance_data_size, instance_count);
}

///////////////////////////////////////////////////////////////////////////////
// Radix sort
///////////////////////////////////////////////////////////////////////////////

typedef struct {
	uint64_t key;
	uint32_t idx;
} _skr_sort_pair_t;

// Insertion sort for small lists — sorts items in-place by sort_key
static void _skr_render_list_insertion_sort(skr_render_item_t* items, uint32_t count) {
	for (uint32_t i = 1; i < count; i++) {
		skr_render_item_t tmp = items[i];
		uint32_t j = i;
		while (j > 0 && items[j - 1].sort_key > tmp.sort_key) {
			items[j] = items[j - 1];
			j--;
		}
		items[j] = tmp;
	}
}

// Ensure radix sort scratch buffers are large enough
static bool _skr_render_list_ensure_scratch(skr_render_list_t* list, uint32_t count) {
	if (count <= list->sort_scratch_capacity) return true;

	// Grow to next power of 2, minimum 64
	uint32_t cap = count < 64 ? 64 : count;
	cap--;
	cap |= cap >> 1;
	cap |= cap >> 2;
	cap |= cap >> 4;
	cap |= cap >> 8;
	cap |= cap >> 16;
	cap++;

	_skr_free(list->sort_scratch_a);
	_skr_free(list->sort_scratch_b);
	list->sort_scratch_a        = _skr_malloc(sizeof(_skr_sort_pair_t) * cap);
	list->sort_scratch_b        = _skr_malloc(sizeof(_skr_sort_pair_t) * cap);
	list->sort_scratch_capacity = cap;

	return list->sort_scratch_a && list->sort_scratch_b;
}

void _skr_render_list_sort(skr_render_list_t* ref_list) {
	if (!ref_list || !ref_list->needs_sort || ref_list->count == 0) return;

	uint32_t n = ref_list->count;
	ref_list->needs_sort = false;

	// Small lists: insertion sort directly on items (avoids scratch overhead)
	if (n <= 32) {
		_skr_render_list_insertion_sort(ref_list->items, n);
		goto reorder_instance_data;
	}

	// Ensure scratch buffers are large enough
	if (!_skr_render_list_ensure_scratch(ref_list, n)) {
		skr_log(skr_log_critical, "Failed to allocate radix sort scratch");
		goto reorder_instance_data;
	}

	// Initialize sort pairs from items, and build all 8 histograms in one pass
	_skr_sort_pair_t* cur   = (_skr_sort_pair_t*)ref_list->sort_scratch_a;
	_skr_sort_pair_t* other = (_skr_sort_pair_t*)ref_list->sort_scratch_b;
	uint32_t histograms[8][256];
	memset(histograms, 0, sizeof(histograms));
	for (uint32_t i = 0; i < n; i++) {
		uint64_t key     = ref_list->items[i].sort_key;
		cur[i].key       = key;
		cur[i].idx       = i;
		histograms[0][(uint8_t)(key      )]++;
		histograms[1][(uint8_t)(key >>  8)]++;
		histograms[2][(uint8_t)(key >> 16)]++;
		histograms[3][(uint8_t)(key >> 24)]++;
		histograms[4][(uint8_t)(key >> 32)]++;
		histograms[5][(uint8_t)(key >> 40)]++;
		histograms[6][(uint8_t)(key >> 48)]++;
		histograms[7][(uint8_t)(key >> 56)]++;
	}

	// Scatter passes — skip uniform bytes (histogram has single bucket == n)
	// Find the last non-uniform pass so we can fuse it with the permute
	int32_t last_pass = -1;
	for (int32_t pass = 7; pass >= 0; pass--) {
		bool uniform = false;
		for (int32_t b = 0; b < 256; b++) {
			if (histograms[pass][b] == n) { uniform = true; break; }
		}
		if (!uniform) { last_pass = pass; break; }
	}

	// All passes, except the last non-uniform one (which is fused with permute)
	for (int32_t pass = 0; pass < 8; pass++) {
		if (pass == last_pass) continue;
		int32_t  shift   = pass * 8;
		uint32_t* counts = histograms[pass];

		// Skip uniform passes
		bool uniform = false;
		for (int32_t b = 0; b < 256; b++) {
			if (counts[b] == n) { uniform = true; break; }
		}
		if (uniform) continue;

		// Prefix sum (exclusive)
		uint32_t offsets[256];
		offsets[0] = 0;
		for (int32_t b = 1; b < 256; b++)
			offsets[b] = offsets[b - 1] + counts[b - 1];

		// Scatter pairs. Prefetching the next slot we'll write to in this
		// digit bucket hides the latency of the random destination access —
		// the hardware prefetcher can't see through scatter patterns.
		for (uint32_t i = 0; i < n; i++) {
			uint8_t digit = (uint8_t)(cur[i].key >> shift);
			other[offsets[digit]++] = cur[i];
			_SKR_PREFETCH(&other[offsets[digit]]);
		}

		// Swap buffers
		_skr_sort_pair_t* tmp = cur;
		cur   = other;
		other = tmp;
	}

	// Fused final scatter + permute: scatter pairs by last_pass digit AND
	// gather items into sorted order in one pass (one random read per item)
	{
		skr_render_item_t* items     = ref_list->items;
		skr_render_item_t* items_tmp = ref_list->items_tmp;

		if (last_pass >= 0) {
			int32_t  shift   = last_pass * 8;
			uint32_t* counts = histograms[last_pass];
			uint32_t offsets[256];
			offsets[0] = 0;
			for (int32_t b = 1; b < 256; b++)
				offsets[b] = offsets[b - 1] + counts[b - 1];

			// Final fused scatter+permute. Two prefetch hints: the next slot
			// in the destination digit bucket (scatter target) and the source
			// item for the next iteration (the random gather read).
			for (uint32_t i = 0; i < n; i++) {
				uint8_t  digit = (uint8_t)(cur[i].key >> shift);
				uint32_t dst   = offsets[digit]++;
				items_tmp[dst] = items[cur[i].idx];
				_SKR_PREFETCH(&items_tmp[offsets[digit]]);
				if (i + 1 < n) _SKR_PREFETCH(&items[cur[i + 1].idx]);
			}
		} else {
			// All passes were uniform — items are already sorted, just copy
			memcpy(items_tmp, items, n * sizeof(skr_render_item_t));
		}

		// Swap — both buffers are always 'capacity' elements
		ref_list->items     = items_tmp;
		ref_list->items_tmp = items;
	}

reorder_instance_data:
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
