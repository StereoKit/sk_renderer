// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Lists

static void _skr_destroy_list_reserve(skr_destroy_list_t* ref_list, uint32_t required) {
	if (ref_list->capacity >= required) return;

	uint32_t new_capacity = ref_list->capacity == 0 ? 8 : ref_list->capacity * 2;
	while (new_capacity < required)
		new_capacity *= 2;

	skr_destroy_item_t* new_items = _skr_realloc(ref_list->items, new_capacity * sizeof(skr_destroy_item_t));
	if (!new_items) {
		skr_log(skr_log_critical, "Failed to resize destroy list");
		return;
	}
	ref_list->items    = new_items;
	ref_list->capacity = new_capacity;
}

static void _skr_destroy_list_add(skr_destroy_list_t* ref_list, skr_destroy_item_t item) {
	_skr_destroy_list_reserve(ref_list, ref_list->count + 1);
	ref_list->items[ref_list->count++] = item;
}

static void _skr_destroy_item(skr_destroy_item_t item, _skr_desc_cache_t* opt_ref_cache) {
	switch (item.type) {
		#define MAKE_CASE(name, vk_type, destroy_func) case skr_destroy_type_##name: destroy_func(_skr_vk.device, (vk_type)item.handle, NULL); break;
		FOREACH_DESTROY_TYPE(MAKE_CASE)
		#undef MAKE_CASE
		case skr_destroy_type_bind_pool_slots: _skr_bind_pool_free((int32_t)item.handle, item.aux); break;
		case skr_destroy_type_desc_set:        _skr_desc_cache_free(opt_ref_cache, (VkDescriptorSet)item.handle, (uint8_t)item.aux); break;
		case skr_destroy_type_mem: {
			_skr_mem_t mem;
			memcpy(&mem, &item.handle, sizeof(mem));
			_skr_mem_free(mem);
		} break;
		case skr_destroy_type_buffer_slot: _skr_buffer_slot_retired((_skr_buffer_slot_t*)(uintptr_t)item.handle); break;
	}
}

// The cache is the list owner's; pending blocks hold no sets and pass NULL
void _skr_destroy_list_execute(skr_destroy_list_t* ref_list, _skr_desc_cache_t* opt_ref_cache) {
	for (int32_t i = ref_list->count - 1; i >= 0; i--)
		_skr_destroy_item(ref_list->items[i], opt_ref_cache);
	ref_list->count = 0;
}

///////////////////////////////////////////////////////////////////////////////
// Pending stack: pushed from any thread, taken whole under the queue mutex.
// Nothing pops one node, so there is no ABA to guard.

// Links a newest-to-oldest chain under the current head in one step
static void _skr_destroy_push_chain(_skr_destroy_block_t* newest, _skr_destroy_block_t* oldest) {
	for (;;) {
		_skr_destroy_block_t* head = _skr_load_ptr(&_skr_vk.pending);
		oldest->next = head;
		if (_skr_cas_ptr(&_skr_vk.pending, head, newest)) return;
	}
}

static _skr_destroy_block_t* _skr_destroy_block_create(skr_destroy_list_t* ref_items, skr_future_t covered_by) {
	_skr_destroy_block_t* block = _skr_calloc(1, sizeof(_skr_destroy_block_t));
	block->list       = *ref_items;
	block->covered_by = covered_by;
	*ref_items = (skr_destroy_list_t){0};
	return block;
}

// Off the queue mutex, so a scene teardown's vkDestroy calls don't stall
// every other thread's submit
void _skr_destroy_execute_blocks(_skr_destroy_block_t* opt_blocks) {
	while (opt_blocks) {
		_skr_destroy_block_t* next = opt_blocks->next;
		_skr_destroy_list_execute(&opt_blocks->list, NULL);
		_skr_free(opt_blocks->list.items);
		_skr_free(opt_blocks);
		opt_blocks = next;
	}
}

// Reads another thread's slot the way skr_future_check does. The owner resets
// or destroys that fence only under the queue mutex, which the caller holds.
static bool _skr_destroy_block_done(const _skr_destroy_block_t* block) {
	const _skr_cmd_ring_slot_t* slot = block->covered_by.slot;
	if (slot == NULL) return false;
	if (slot->generation != block->covered_by.generation) return true;
	return vkGetFenceStatus(_skr_vk.device, slot->fence) == VK_SUCCESS;
}

// Every submit calls this under the queue mutex, right after its vkQueueSubmit,
// so `cover` orders after everything already queued and the stamp is exact.
// Returns the done blocks, newest first, for the caller to run off the mutex.
_skr_destroy_block_t* _skr_destroy_retire(skr_destroy_list_t* ref_shared_open, skr_future_t cover, bool may_stamp) {
	_skr_destroy_block_t* head = _skr_exchange_ptr(&_skr_vk.pending, NULL);
	if (ref_shared_open->count > 0) {
		_skr_destroy_block_t* block = _skr_destroy_block_create(ref_shared_open, (skr_future_t){0});
		block->next = head;
		head = block;
	}
	// Compound destroys are batched into one block, and a block's own stamp
	// covers everything in it, so done blocks free in any order
	_skr_destroy_block_t* done      = NULL;
	_skr_destroy_block_t* done_tail = NULL;
	_skr_destroy_block_t* keep      = NULL;
	_skr_destroy_block_t* keep_tail = NULL;
	for (_skr_destroy_block_t* b = head, *next; b; b = next) {
		next    = b->next;
		b->next = NULL;
		if (may_stamp && b->covered_by.slot == NULL) b->covered_by = cover;
		if (_skr_destroy_block_done(b)) { if (done_tail) done_tail->next = b; else done = b; done_tail = b; }
		else                            { if (keep_tail) keep_tail->next = b; else keep = b; keep_tail = b; }
	}
	if (keep) _skr_destroy_push_chain(keep, keep_tail);
	return done;
}

// Before a thread destroys its fences, under the queue mutex: no block may
// poll a dead fence, so its stamps come off and a later submit re-stamps them
void _skr_destroy_disown(const _skr_cmd_ring_slot_t* slots) {
	_skr_destroy_block_t* head = _skr_exchange_ptr(&_skr_vk.pending, NULL);
	if (head == NULL) return;
	_skr_destroy_block_t* oldest = head;
	for (_skr_destroy_block_t* b = head; b; b = b->next) {
		uintptr_t offset = (uintptr_t)b->covered_by.slot - (uintptr_t)slots;
		if (offset < skr_MAX_COMMAND_RING * sizeof(_skr_cmd_ring_slot_t)) b->covered_by = (skr_future_t){0};
		oldest = b;
	}
	_skr_destroy_push_chain(head, oldest);
}

// Shutdown only, with the device idle. Returns how many items it found.
uint32_t _skr_destroy_drain(void) {
	_skr_destroy_block_t* blocks = _skr_exchange_ptr(&_skr_vk.pending, NULL);
	uint32_t count = 0;
	for (_skr_destroy_block_t* b = blocks; b; b = b->next)
		count += b->list.count;
	_skr_destroy_execute_blocks(blocks);
	return count;
}

///////////////////////////////////////////////////////////////////////////////
// Entry points

// A compound destroy from a thread with no list to batch in (idle, or never
// registered) gathers here and leaves as one block. Separate blocks could be
// split by a retire taking the stack between the pushes, and the push-back
// would then put the older one on top.
static thread_local skr_destroy_list_t _skr_destroy_batch;
static thread_local int32_t            _skr_destroy_batch_depth;

void _skr_destroy_batch_begin(void) {
	_skr_destroy_batch_depth++;
}

void _skr_destroy_batch_end(void) {
	if (--_skr_destroy_batch_depth > 0 || _skr_destroy_batch.count == 0) return;
	_skr_destroy_block_t* block = _skr_destroy_block_create(&_skr_destroy_batch, (skr_future_t){0});
	_skr_destroy_push_chain(block, block);
}

static void _skr_destroy_shared_item(skr_destroy_item_t item) {
	_skr_vk_thread_t* thr = _skr_cmd_get_thread();
	if (thr && (thr->ref_count > 0 || thr->thread_idx == _skr_load_u32(&_skr_vk.main_pool_idx))) {
		_skr_destroy_list_add(&thr->shared_open, item);
		return;
	}
	if (_skr_destroy_batch_depth > 0) {
		_skr_destroy_list_add(&_skr_destroy_batch, item);
		return;
	}
	// Nothing on this thread will submit, so the item waits as its own block
	skr_destroy_list_t one = {0};
	_skr_destroy_list_add(&one, item);
	_skr_destroy_block_t* block = _skr_destroy_block_create(&one, (skr_future_t){0});
	_skr_destroy_push_chain(block, block);
}

static void _skr_destroy_private_item(skr_destroy_item_t item) {
	_skr_vk_thread_t* thr = _skr_cmd_get_thread();
	assert(thr && thr->ref_count > 0 && "Private destroys need a recording command buffer");
	_skr_destroy_list_add(&thr->cmd_ring[thr->cur].destroy_list, item);
}

#define MAKE_SHARED_FUNCTION(name, vk_type, func) \
void _skr_destroy_shared_##name(vk_type handle) { \
	if (handle == VK_NULL_HANDLE) return; \
	_skr_destroy_shared_item((skr_destroy_item_t){ .type = skr_destroy_type_##name, .handle = (uint64_t)handle }); \
}
FOREACH_DESTROY_TYPE(MAKE_SHARED_FUNCTION)
#undef MAKE_SHARED_FUNCTION

void _skr_destroy_shared_bind_pool_slots(int32_t start, uint32_t count) {
	if (start < 0 || count == 0) return;
	_skr_destroy_shared_item((skr_destroy_item_t){ .type = skr_destroy_type_bind_pool_slots, .handle = (uint64_t)start, .aux = count });
}

static skr_destroy_item_t _skr_destroy_mem_item(_skr_mem_t mem) {
	_Static_assert(sizeof(_skr_mem_t) == sizeof(uint64_t), "_skr_mem_t must pack into a destroy item handle");
	skr_destroy_item_t item = { .type = skr_destroy_type_mem };
	memcpy(&item.handle, &mem, sizeof(mem));
	return item;
}
void _skr_destroy_shared_mem (_skr_mem_t mem) { if (mem.block != 0) _skr_destroy_shared_item (_skr_destroy_mem_item(mem)); }

void _skr_destroy_shared_buffer_slot(_skr_buffer_slot_t* slot) {
	_skr_destroy_shared_item((skr_destroy_item_t){ .type = skr_destroy_type_buffer_slot, .handle = (uint64_t)(uintptr_t)slot });
}
void _skr_destroy_private_mem(_skr_mem_t mem) { if (mem.block != 0) _skr_destroy_private_item(_skr_destroy_mem_item(mem)); }

void _skr_destroy_private_buffer     (VkBuffer       handle) { if (handle != VK_NULL_HANDLE) _skr_destroy_private_item((skr_destroy_item_t){ .type = skr_destroy_type_buffer,      .handle = (uint64_t)handle }); }
void _skr_destroy_private_image_view (VkImageView    handle) { if (handle != VK_NULL_HANDLE) _skr_destroy_private_item((skr_destroy_item_t){ .type = skr_destroy_type_image_view,  .handle = (uint64_t)handle }); }
void _skr_destroy_private_framebuffer(VkFramebuffer  handle) { if (handle != VK_NULL_HANDLE) _skr_destroy_private_item((skr_destroy_item_t){ .type = skr_destroy_type_framebuffer, .handle = (uint64_t)handle }); }

void _skr_destroy_private_desc_set(VkDescriptorSet set, uint8_t pool) {
	_skr_destroy_private_item((skr_destroy_item_t){ .type = skr_destroy_type_desc_set, .handle = (uint64_t)set, .aux = pool });
}
