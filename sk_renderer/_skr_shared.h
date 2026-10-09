// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include "include/sk_renderer.h"
#include "skr_atomics.h"

///////////////////////////////////////////////////////////////////////////////
// Internal contract between the backend-independent sources (skr_log.c) and
// the active backend. Each backend implements these allocation wrappers,
// honoring the skr_settings_t allocator callbacks once initialized.
///////////////////////////////////////////////////////////////////////////////

void* _skr_malloc (size_t size);
void* _skr_calloc (size_t count, size_t size);
void* _skr_realloc(void* ptr, size_t size);
void  _skr_free   (void* ptr);

// Presentation timeline ring (skr_present.c), embedded in each backend's surface
skr_present_info_t* _skr_present_begin     (skr_present_ring_t* ref_ring, uint64_t cpu_present_ns);  // Claims the next id
skr_present_info_t* _skr_present_slot      (skr_present_ring_t* ref_ring, uint64_t id);              // NULL once the id has left the ring
void                _skr_present_display   (skr_present_ring_t* ref_ring, uint64_t id, uint64_t display_ns, skr_timing_source_ source);  // Upgrades only
void                _skr_present_vblank    (skr_present_ring_t* ref_ring, uint64_t vblank_ns);  // Places waiting presents on the vblank grid, once per distinct stamp
int32_t             _skr_present_drain     (skr_present_ring_t* ref_ring, uint32_t settle_count, skr_present_info_t* out_infos, int32_t max_count);
uint64_t            _skr_present_refresh_ns(const skr_present_ring_t* ring);

///////////////////////////////////////////////////////////////////////////////
// Futures across shutdown. _skr_future_calls is a refcount of future calls in
// progress, with its top bit as the "accepting calls" flag. Sharing one word
// means a call can't slip in between shutdown clearing the flag and waiting
// for the count to drain. It lives outside each backend's state, since
// resetting that would race the count.
///////////////////////////////////////////////////////////////////////////////

#define _SKR_FUTURE_OPEN 0x80000000u
extern _skr_atomic(uint32_t) _skr_future_calls;

// A true return must be paired with _skr_future_leave
static inline bool _skr_future_enter(void) {
	if (_skr_add_acq_rel(&_skr_future_calls, 1) & _SKR_FUTURE_OPEN) return true;
	_skr_add_acq_rel(&_skr_future_calls, (uint32_t)-1);
	return false;
}
static inline void _skr_future_leave(void) { _skr_add_acq_rel(&_skr_future_calls, (uint32_t)-1); }
static inline void _skr_future_open (void) { _skr_add_acq_rel(&_skr_future_calls, _SKR_FUTURE_OPEN); }
static inline void _skr_future_close(void) {
	_skr_and_acq_rel(&_skr_future_calls, ~_SKR_FUTURE_OPEN);
	while (_skr_load_u32_acquire(&_skr_future_calls) != 0) _skr_yield();
}

// The lowest generation this skr_init hands out. An earlier session's future
// sits below it, and its slot may be freed, so checks turn it away unread.
extern uint64_t _skr_gen_floor;
static inline bool _skr_future_current(const skr_future_t* future) { return future->generation >= _skr_gen_floor; }
