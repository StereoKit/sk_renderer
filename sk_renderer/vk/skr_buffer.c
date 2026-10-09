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

skr_err_ _skr_buffer_create_vk(VkDeviceSize size, VkBufferUsageFlags usage, _skr_mem_usage_ mem_usage, skr_mem_category_ category, VkBuffer* out_buffer, _skr_mem_t* out_mem, void** opt_out_mapped) {
	*out_mem = (_skr_mem_t){0};
	VkResult vr = vkCreateBuffer(_skr_vk.device, &(VkBufferCreateInfo){
		.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size        = size,
		.usage       = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	}, NULL, out_buffer);
	if (vr != VK_SUCCESS) {
		skr_log(skr_log_critical, "vkCreateBuffer: 0x%X", (uint32_t)vr);
		*out_buffer = VK_NULL_HANDLE;
		return skr_err_device_error;
	}

	skr_err_ err = _skr_mem_bind_buffer(*out_buffer, mem_usage, category, out_mem, opt_out_mapped);
	if (err != skr_err_success) {
		vkDestroyBuffer(_skr_vk.device, *out_buffer, NULL);
		*out_buffer = VK_NULL_HANDLE;
	}
	return err;
}

void _skr_buffer_destroy_vk(VkBuffer buffer, _skr_mem_t mem) {
	_skr_destroy_batch_begin();
	_skr_destroy_shared_mem   (mem);
	_skr_destroy_shared_buffer(buffer);
	_skr_destroy_batch_end();
}

// Creation and rename slots share this, so a slot always matches the buffer it renames
static VkBufferUsageFlags _skr_buffer_usage(skr_buffer_type_ type, skr_use_ use, bool initial_data) {
	VkBufferUsageFlags usage = _skr_to_vk_buffer_usage(type);
	if (initial_data && !(use & skr_use_dynamic)) usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	return usage;
}

static skr_mem_category_ _skr_buffer_category(skr_buffer_type_ type) {
	return (type & (skr_buffer_type_vertex | skr_buffer_type_index)) ? skr_mem_category_geometry : skr_mem_category_buffer;
}

///////////////////////////////////////////////////////////////////////////////
// Buffer creation and destruction
///////////////////////////////////////////////////////////////////////////////

static skr_err_ _skr_buffer_create(const void* opt_data, uint32_t size, skr_buffer_type_ type, skr_use_ use, skr_mem_category_ category, skr_buffer_t* out_buffer) {
	*out_buffer = (skr_buffer_t){ .size = size, .type = type, .use = use };
	bool dynamic = use & skr_use_dynamic;
	bool zero    = opt_data == NULL && !(use & skr_use_uninitialized);

	VkBufferUsageFlags usage = _skr_buffer_usage(type, use, opt_data != NULL || zero);

	_skr_mem_usage_ mem_usage = dynamic ? _skr_mem_usage_stream : opt_data ? _skr_mem_usage_upload : _skr_mem_usage_gpu;
	void*           mapped;
	skr_err_        err       = _skr_buffer_create_vk(size, usage, mem_usage, category, &out_buffer->buffer, &out_buffer->mem, &mapped);
	if (err != skr_err_success) {
		*out_buffer = (skr_buffer_t){0};
		return err;
	}
	if (dynamic) out_buffer->mapped = mapped;
	out_buffer->uid = _skr_uid_new();

	// Zeroed unless asked not to, since pooled memory holds whatever its last owner left
	if (opt_data == NULL && !zero) return skr_err_success;
	// Memory the CPU can write directly needs no staging copy: vkQueueSubmit
	// makes earlier host writes visible to anything it runs
	if (mapped) {
		if (opt_data) memcpy(mapped, opt_data, size);
		else          memset(mapped, 0, size);
		return skr_err_success;
	}

	VkBuffer   staging_buffer = VK_NULL_HANDLE;
	_skr_mem_t staging_mem    = {0};
	if (opt_data) {
		void* staging_mapped;
		err = _skr_buffer_create_vk(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, _skr_mem_usage_staging, skr_mem_category_staging,
			&staging_buffer, &staging_mem, &staging_mapped);
		if (err != skr_err_success) {
			vkDestroyBuffer(_skr_vk.device, out_buffer->buffer, NULL);
			_skr_mem_free(out_buffer->mem);
			*out_buffer = (skr_buffer_t){0};
			return err;
		}
		memcpy(staging_mapped, opt_data, size);
	}

	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

	if (opt_data) vkCmdCopyBuffer(ctx.cmd, staging_buffer, out_buffer->buffer, 1, &(VkBufferCopy){ .size = out_buffer->size });
	else          vkCmdFillBuffer(ctx.cmd, out_buffer->buffer, 0, VK_WHOLE_SIZE, 0);

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

	if (opt_data) _skr_buffer_destroy_vk(staging_buffer, staging_mem);
	_skr_cmd_release(ctx.cmd);
	return skr_err_success;
}

skr_err_ skr_buffer_create(const void* opt_data, uint32_t size_count, uint32_t size_stride,
                            skr_buffer_type_ type, skr_use_ use, skr_buffer_t* out_buffer) {
	if (!out_buffer) return skr_err_invalid_parameter;
	*out_buffer = (skr_buffer_t){0};
	if (size_count == 0 || size_stride == 0) return skr_err_invalid_parameter;

	return _skr_buffer_create(opt_data, size_count * size_stride, type, use, _skr_buffer_category(type), out_buffer);
}

bool skr_buffer_is_valid(const skr_buffer_t* buffer) {
	return buffer && buffer->buffer != VK_NULL_HANDLE;
}

///////////////////////////////////////////////////////////////////////////////
// Rename slots. skr_buffer_set writes into a slot no submission reads, and
// the slot it replaces retires through the shared destroy path, the same proof
// a destroy waits for, before anything writes it again.

typedef enum {
	_skr_slot_current,
	_skr_slot_pending, // replaced, a submission may still read it
	_skr_slot_free,
	_skr_slot_orphan,  // its buffer was destroyed while pending; retiring frees it
} _skr_slot_;

// Slots live until their buffer is destroyed: render list items keep the
// handle they captured, so an idle slot may still be drawn
struct _skr_buffer_slot_t {
	VkBuffer              buffer;
	_skr_mem_t            mem;
	void*                 mapped;
	uint64_t              uid;
	_skr_atomic(uint32_t) state; // _skr_slot_
};

typedef struct _skr_buffer_ring_t {
	_skr_buffer_slot_t** slots;
	uint32_t             count;
	uint32_t             capacity;
	uint32_t             current;
} _skr_buffer_ring_t;

void _skr_buffer_slot_retired(_skr_buffer_slot_t* slot) {
	if (_skr_exchange_u32(&slot->state, _skr_slot_free) != _skr_slot_orphan) return;
	vkDestroyBuffer(_skr_vk.device, slot->buffer, NULL);
	_skr_mem_free(slot->mem);
	_skr_free(slot);
}

static bool _skr_buffer_ring_add(_skr_buffer_ring_t* ref_ring, _skr_buffer_slot_t* slot) {
	if (ref_ring->count == ref_ring->capacity) {
		uint32_t             capacity = ref_ring->capacity == 0 ? 4 : ref_ring->capacity * 2;
		_skr_buffer_slot_t** slots    = _skr_realloc(ref_ring->slots, capacity * sizeof(_skr_buffer_slot_t*));
		if (!slots) return false;
		ref_ring->slots    = slots;
		ref_ring->capacity = capacity;
	}
	ref_ring->slots[ref_ring->count++] = slot;
	return true;
}

// The buffer's own allocation becomes the first slot
static _skr_buffer_ring_t* _skr_buffer_ring_create(const skr_buffer_t* buffer) {
	_skr_buffer_ring_t* ring = _skr_calloc(1, sizeof(_skr_buffer_ring_t));
	_skr_buffer_slot_t* slot = _skr_malloc(sizeof(_skr_buffer_slot_t));
	if (!ring || !slot || !_skr_buffer_ring_add(ring, slot)) {
		_skr_free(ring);
		_skr_free(slot);
		return NULL;
	}
	*slot = (_skr_buffer_slot_t){ .buffer = buffer->buffer, .mem = buffer->mem, .mapped = buffer->mapped, .uid = buffer->uid, .state = _skr_slot_current };
	return ring;
}

// A free slot, or a new one when the set rate outruns the ring
static _skr_buffer_slot_t* _skr_buffer_ring_take(_skr_buffer_ring_t* ref_ring, const skr_buffer_t* buffer) {
	for (uint32_t i = 0; i < ref_ring->count; i++)
		if (_skr_load_u32(&ref_ring->slots[i]->state) == _skr_slot_free) return ref_ring->slots[i];

	_skr_buffer_slot_t* taken = _skr_malloc(sizeof(_skr_buffer_slot_t));
	if (!taken) return NULL;
	*taken = (_skr_buffer_slot_t){ .uid = _skr_uid_new(), .state = _skr_slot_free };
	if (_skr_buffer_create_vk(buffer->size, _skr_buffer_usage(buffer->type, buffer->use, false), _skr_mem_usage_stream, (skr_mem_category_)buffer->mem.category,
	                          &taken->buffer, &taken->mem, &taken->mapped) != skr_err_success
	    || !_skr_buffer_ring_add(ref_ring, taken)) {
		if (taken->buffer != VK_NULL_HANDLE) {
			vkDestroyBuffer(_skr_vk.device, taken->buffer, NULL);
			_skr_mem_free(taken->mem);
		}
		_skr_free(taken);
		return NULL;
	}
	return taken;
}

void skr_buffer_set(skr_buffer_t* ref_buffer, const void* data, uint32_t size_bytes) {
	if (!ref_buffer || !data) return;

	if (!(ref_buffer->use & skr_use_dynamic)) {
		skr_log(skr_log_critical, "skr_buffer_set only supports dynamic buffers");
		return;
	}

	uint32_t copy_size = size_bytes < ref_buffer->size ? size_bytes : ref_buffer->size;

	if (!ref_buffer->_ring) ref_buffer->_ring = _skr_buffer_ring_create(ref_buffer);
	_skr_buffer_ring_t* ring = ref_buffer->_ring;
	_skr_buffer_slot_t* next = ring ? _skr_buffer_ring_take(ring, ref_buffer) : NULL;
	if (!next) {
		skr_log(skr_log_critical, "skr_buffer_set: out of memory for a rename slot, overwriting data that may still be in use");
		memcpy(ref_buffer->mapped, data, copy_size);
		return;
	}

	memcpy(next->mapped, data, copy_size);
	_skr_store_u32(&next->state, _skr_slot_current);

	_skr_buffer_slot_t* prev = ring->slots[ring->current];
	_skr_store_u32(&prev->state, _skr_slot_pending);
	_skr_destroy_shared_buffer_slot(prev);

	for (uint32_t i = 0; i < ring->count; i++)
		if (ring->slots[i] == next) ring->current = i;
	ref_buffer->buffer = next->buffer;
	ref_buffer->mem    = next->mem;
	ref_buffer->mapped = next->mapped;
	ref_buffer->uid    = next->uid;
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
	VkBuffer   staging_buffer;
	_skr_mem_t staging_mem;
} _skr_buffer_readback_internal_t;

skr_err_ skr_buffer_readback(const skr_buffer_t* buffer, skr_buffer_readback_t* out_readback) {
	if (!buffer || !out_readback) return skr_err_invalid_parameter;
	memset(out_readback, 0, sizeof(*out_readback));

	// Storage is the only type both backends can copy out of; see the header
	if (!(buffer->type & skr_buffer_type_storage)) {
		skr_log(skr_log_critical, "skr_buffer_readback needs a storage-type buffer");
		return skr_err_unsupported;
	}

	VkBuffer   staging_buffer;
	_skr_mem_t staging_mem;
	void*      mapped;
	skr_err_   err = _skr_buffer_create_vk(buffer->size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, _skr_mem_usage_readback, skr_mem_category_staging,
		&staging_buffer, &staging_mem, &mapped);
	if (err != skr_err_success) return err;

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
	internal->staging_mem    = staging_mem;

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
	// submit, and blocking on that here deadlocks
	_skr_buffer_destroy_vk(internal->staging_buffer, internal->staging_mem);
	_skr_free(internal);

	*ref_readback = (skr_buffer_readback_t){0};
}

void skr_buffer_set_name(skr_buffer_t* ref_buffer, const char* name) {
	if (!ref_buffer || ref_buffer->buffer == VK_NULL_HANDLE) return;
	_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_BUFFER, (uint64_t)ref_buffer->buffer, name);
}

// Private when only this thread's command buffers have read it (the bump ring)
typedef enum {
	_skr_buffer_free_shared,   // any thread may have bound it
	_skr_buffer_free_private,  // only this thread's command buffers read it (the bump ring)
	_skr_buffer_free_now,      // every fence that could read it has been waited
} _skr_buffer_free_;

static void _skr_buffer_free_pair(_skr_buffer_free_ how, _skr_mem_t mem, VkBuffer buffer) {
	switch (how) {
		case _skr_buffer_free_shared:  _skr_destroy_shared_mem (mem); _skr_destroy_shared_buffer (buffer); break;
		case _skr_buffer_free_private: _skr_destroy_private_mem(mem); _skr_destroy_private_buffer(buffer); break;
		case _skr_buffer_free_now:     vkDestroyBuffer(_skr_vk.device, buffer, NULL); _skr_mem_free(mem); break;
	}
}

static void _skr_buffer_release(skr_buffer_t* ref_buffer, _skr_buffer_free_ how) {
	if (ref_buffer->buffer == VK_NULL_HANDLE) return;

	_skr_destroy_batch_begin();
	_skr_buffer_ring_t* ring = ref_buffer->_ring;
	if (ring) {
		// A pending slot's retire item still holds it, so that frees it instead
		for (uint32_t i = 0; i < ring->count; i++) {
			_skr_buffer_slot_t* slot = ring->slots[i];
			if (_skr_exchange_u32(&slot->state, _skr_slot_orphan) == _skr_slot_pending) continue;
			_skr_buffer_free_pair(how, slot->mem, slot->buffer);
			_skr_free(slot);
		}
		_skr_free(ring->slots);
		_skr_free(ring);
	} else {
		_skr_buffer_free_pair(how, ref_buffer->mem, ref_buffer->buffer);
	}
	_skr_destroy_batch_end();

	*ref_buffer = (skr_buffer_t){0};
}

void skr_buffer_destroy(skr_buffer_t* ref_buffer) {
	if (!ref_buffer) return;
	_skr_buffer_release(ref_buffer, _skr_buffer_free_shared);
}

///////////////////////////////////////////////////////////////////////////////
// Bump ring, see _skr_bump_ring_t

#define _SKR_BUMP_RING_MIN        (64 * 1024)
#define _SKR_BUMP_RING_WINDOW_MIN 4096

void _skr_bump_ring_init(_skr_bump_ring_t* out_ring, skr_buffer_type_ type, uint32_t alignment, const _skr_cmd_ring_slot_t* slots) {
	*out_ring = (_skr_bump_ring_t){
		.buffer_type = type,
		.alignment   = alignment > 0 ? alignment : 1,
		.slots       = slots,
	};
}

// Only called once every fence that could read the ring has been waited
void _skr_bump_ring_destroy(_skr_bump_ring_t* ref_ring) {
	_skr_buffer_release(&ref_ring->buffer, _skr_buffer_free_now);
	*ref_ring = (_skr_bump_ring_t){0};
}

// Widens the range every offset must be able to reach, for callers that bind
// a fixed window from a varying offset.
uint32_t _skr_bump_ring_reserve_window(_skr_bump_ring_t* ref_ring, uint32_t bytes) {
	while (ref_ring->window < bytes)
		ref_ring->window = ref_ring->window == 0 ? _SKR_BUMP_RING_WINDOW_MIN : ref_ring->window * 2;
	return ref_ring->window;
}

void _skr_bump_ring_slot_begin(_skr_bump_ring_t* ref_ring, uint32_t slot) {
	ref_ring->slot        = slot;
	ref_ring->start[slot] = ref_ring->head;
	ref_ring->end  [slot] = ref_ring->head;
}

// Oldest position any slot still reads, or head when nothing is live
static uint64_t _skr_bump_ring_tail(const _skr_bump_ring_t* ring) {
	uint64_t tail = ring->head;
	for (uint32_t i = 0; i < skr_MAX_COMMAND_RING; i++)
		if (ring->start[i] != ring->end[i] && ring->start[i] < tail) tail = ring->start[i];
	return tail;
}

// A slot whose fence has signaled no longer reads its range. Only worth the
// driver calls when the ring is full.
static void _skr_bump_ring_poll(_skr_bump_ring_t* ref_ring) {
	for (uint32_t i = 0; i < skr_MAX_COMMAND_RING; i++) {
		if (i == ref_ring->slot || ref_ring->start[i] == ref_ring->end[i]) continue;
		if (vkGetFenceStatus(_skr_vk.device, ref_ring->slots[i].fence) == VK_SUCCESS)
			ref_ring->end[i] = ref_ring->start[i];
	}
}

// In-flight slots and descriptors written earlier still read the old buffer;
// the private destroy retires it with the recording slot's fence, which
// orders after every earlier submission.
static bool _skr_bump_ring_grow(_skr_bump_ring_t* ref_ring, uint32_t min_capacity) {
	uint32_t capacity = ref_ring->capacity == 0 ? _SKR_BUMP_RING_MIN : ref_ring->capacity;
	while (capacity < min_capacity && capacity != 0) capacity *= 2;
	if (capacity == 0) {
		skr_log(skr_log_critical, "Bump ring can't hold a %u byte write", min_capacity);
		return false;
	}
	// Create first, so a failed grow keeps the old buffer instead of none
	skr_buffer_t grown = {0};
	skr_err_ err = _skr_buffer_create(NULL, capacity, ref_ring->buffer_type, skr_use_dynamic | skr_use_uninitialized, skr_mem_category_frame, &grown);
	if (err != skr_err_success) {
		skr_log(skr_log_critical, "Bump ring couldn't grow to %u bytes: %d", capacity, err);
		return false;
	}
	_skr_buffer_release(&ref_ring->buffer, _skr_buffer_free_private);
	ref_ring->buffer   = grown;
	ref_ring->capacity = capacity;
	ref_ring->head     = 0;
	for (uint32_t i = 0; i < skr_MAX_COMMAND_RING; i++) {
		ref_ring->start[i] = 0;
		ref_ring->end  [i] = 0;
	}
	return true;
}

skr_bump_result_t _skr_bump_ring_write(_skr_bump_ring_t* ref_ring, const void* data, uint32_t size) {
	skr_bump_result_t result = {0};
	if (size == 0) return result;

	// Every offset must be able to reach `window` bytes without running off
	// the end, since dynamic descriptors bind that range from any offset
	uint32_t reach = size > ref_ring->window ? size : ref_ring->window;
	if (ref_ring->capacity < reach && !_skr_bump_ring_grow(ref_ring, reach)) return result;

	for (int32_t attempt = 0; ; attempt++) {
		uint64_t pos    = (ref_ring->head + ref_ring->alignment - 1) & ~(uint64_t)(ref_ring->alignment - 1);
		uint32_t offset = (uint32_t)(pos % ref_ring->capacity);
		if (offset + reach > ref_ring->capacity) { // never straddle the end
			pos   += ref_ring->capacity - offset;
			offset = 0;
		}
		uint64_t end = pos + size;
		if (end - _skr_bump_ring_tail(ref_ring) <= ref_ring->capacity) {
			memcpy((uint8_t*)ref_ring->buffer.mapped + offset, data, size);
			uint32_t slot = ref_ring->slot;
			if (ref_ring->start[slot] == ref_ring->end[slot]) ref_ring->start[slot] = pos;
			ref_ring->end[slot] = end;
			ref_ring->head      = end;
			return (skr_bump_result_t){ .buffer = ref_ring->buffer.buffer, .uid = ref_ring->buffer.uid, .offset = offset };
		}
		if (attempt == 0) { _skr_bump_ring_poll(ref_ring); continue; }
		if (!_skr_bump_ring_grow(ref_ring, ref_ring->capacity * 2)) return result;
	}
}
