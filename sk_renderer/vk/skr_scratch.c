// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "sk_renderer.h"
#include "_sk_renderer.h"
#include "skr_scratch.h"

#include <threads.h>

#define SKR_SCRATCH_IDLE_FRAMES      30
#define SKR_SCRATCH_INITIAL_CAPACITY  4

typedef struct {
	// Match key — the template shape the caller originally requested. The scratch
	// image itself is smaller than this (mip 0 is omitted to save ~75% memory),
	// so we can't derive the key from `tex` fields at match time.
	skr_tex_fmt_   fmt;
	skr_vec3i_t    template_size;
	uint32_t       template_mip_levels;
	uint32_t       layer_count;
	skr_tex_flags_ flags_subset;   // cubemap | array only

	skr_tex_t      tex;
	uint32_t       last_used_frame;
	bool           in_use;
} _skr_scratch_entry_t;

typedef struct {
	mtx_t                 mutex;
	_skr_scratch_entry_t* entries;
	uint32_t              count;
	uint32_t              capacity;
} _skr_scratch_pool_t;

static _skr_scratch_pool_t _pool;

///////////////////////////////////////////////////////////////////////////////

static bool _skr_scratch_matches(const _skr_scratch_entry_t* e, const skr_tex_t* template_src) {
	const skr_tex_flags_ key_flags = skr_tex_flags_cubemap | skr_tex_flags_array;
	return e->fmt                 == template_src->format
	    && e->template_size.x     == template_src->size.x
	    && e->template_size.y     == template_src->size.y
	    && e->template_size.z     == template_src->size.z
	    && e->template_mip_levels == template_src->mip_levels
	    && e->layer_count         == template_src->layer_count
	    && e->flags_subset        == (template_src->flags & key_flags);
}

///////////////////////////////////////////////////////////////////////////////

void _skr_scratch_pool_init(void) {
	mtx_init(&_pool.mutex, mtx_plain);

	_pool.capacity = SKR_SCRATCH_INITIAL_CAPACITY;
	_pool.count    = 0;
	_pool.entries  = _skr_calloc(_pool.capacity, sizeof(_skr_scratch_entry_t));
}

void _skr_scratch_pool_shutdown(void) {
	// No mutex needed — caller guarantees device is idle and no other threads are active
	for (uint32_t i = 0; i < _pool.count; i++) {
		if (_pool.entries[i].tex.image != VK_NULL_HANDLE) {
			skr_tex_destroy(&_pool.entries[i].tex);
		}
	}

	mtx_destroy(&_pool.mutex);
	_skr_free(_pool.entries);
	_pool = (_skr_scratch_pool_t){0};
}

skr_tex_t* _skr_scratch_acquire(const skr_tex_t* template_src) {
	if (!template_src) return NULL;

	mtx_lock(&_pool.mutex);

	// Look for an unused entry matching the template
	for (uint32_t i = 0; i < _pool.count; i++) {
		_skr_scratch_entry_t* e = &_pool.entries[i];
		if (!e->in_use && _skr_scratch_matches(e, template_src)) {
			e->in_use = true;
			mtx_unlock(&_pool.mutex);
			return &e->tex;
		}
	}

	// Grow if needed
	if (_pool.count >= _pool.capacity) {
		uint32_t new_cap = _pool.capacity * 2;
		_skr_scratch_entry_t* new_entries = _skr_realloc(_pool.entries, new_cap * sizeof(_skr_scratch_entry_t));
		if (!new_entries) {
			skr_log(skr_log_critical, "Scratch pool grow failed");
			mtx_unlock(&_pool.mutex);
			return NULL;
		}
		// Zero-init newly allocated slots
		for (uint32_t i = _pool.capacity; i < new_cap; i++) {
			new_entries[i] = (_skr_scratch_entry_t){0};
		}
		_pool.entries  = new_entries;
		_pool.capacity = new_cap;
	}

	// Create a new scratch entry
	_skr_scratch_entry_t* e = &_pool.entries[_pool.count];
	skr_err_ err = _skr_tex_create_scratch(template_src, &e->tex);
	if (err != skr_err_success) {
		skr_log(skr_log_critical, "Scratch texture creation failed for mipgen");
		mtx_unlock(&_pool.mutex);
		return NULL;
	}
	e->fmt                 = template_src->format;
	e->template_size       = template_src->size;
	e->template_mip_levels = template_src->mip_levels;
	e->layer_count         = template_src->layer_count;
	e->flags_subset        = template_src->flags & (skr_tex_flags_cubemap | skr_tex_flags_array);
	e->in_use              = true;
	e->last_used_frame     = _skr_vk.frame;
	_pool.count++;

	mtx_unlock(&_pool.mutex);
	return &e->tex;
}

void _skr_scratch_release(skr_tex_t* scratch) {
	if (!scratch) return;

	mtx_lock(&_pool.mutex);

	for (uint32_t i = 0; i < _pool.count; i++) {
		_skr_scratch_entry_t* e = &_pool.entries[i];
		if (&e->tex == scratch) {
			e->in_use          = false;
			e->last_used_frame = _skr_vk.frame;
			break;
		}
	}

	mtx_unlock(&_pool.mutex);
}

void _skr_scratch_pool_tick(void) {
	mtx_lock(&_pool.mutex);

	uint32_t frame = _skr_vk.frame;
	for (uint32_t i = 0; i < _pool.count; ) {
		_skr_scratch_entry_t* e = &_pool.entries[i];
		if (!e->in_use && (frame - e->last_used_frame) > SKR_SCRATCH_IDLE_FRAMES) {
			// skr_tex_destroy uses the deferred destroy chain (NULL list), so
			// the VkImage/view/memory are held until fences clear.
			skr_tex_destroy(&e->tex);

			// Swap-remove
			if (i + 1 < _pool.count) {
				_pool.entries[i] = _pool.entries[_pool.count - 1];
			}
			_pool.entries[_pool.count - 1] = (_skr_scratch_entry_t){0};
			_pool.count--;
			continue;
		}
		i++;
	}

	mtx_unlock(&_pool.mutex);
}
