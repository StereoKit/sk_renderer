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

// A Texture2DArray slot, which cubes and plain 2D textures fill through their 2D array view
static inline bool _skr_resource_is_2d_array(const sksc_shader_resource_t* res) {
	return res->bind.register_type == skr_register_texture
	    && (res->shape & (SKSC_SHAPE_DIM_MASK | SKSC_SHAPE_ARRAYED | SKSC_SHAPE_MS)) == (SKSC_SHAPE_DIM_2D | SKSC_SHAPE_ARRAYED);
}

///////////////////////////////////////////////////////////////////////////////
// Default textures, picked by a shader resource's declared value. Each
// backend keeps its defaults in one array and registers its address range,
// so a bind holding a default can be told apart from one the user set.
///////////////////////////////////////////////////////////////////////////////

typedef enum {
	_skr_default_white,
	_skr_default_black,
	_skr_default_gray,
	_skr_default_flat,  // normal maps
	_skr_default_rough, // metal/roughness maps
	_skr_default_count,
} _skr_default_;

extern const uint32_t _skr_default_colors[_skr_default_count]; // rgba32
extern uintptr_t      _skr_default_tex_begin;
extern uintptr_t      _skr_default_tex_end;

_skr_default_ _skr_default_for(const char* value);

static inline bool _skr_tex_is_default(const skr_tex_t* tex) {
	return (uintptr_t)tex >= _skr_default_tex_begin && (uintptr_t)tex < _skr_default_tex_end;
}

///////////////////////////////////////////////////////////////////////////////
// Global binds (skr_globals.c). A bind carries its resource's name hash, or 0
// when it never reads globals. What a bind reads, highest first: a deprecated
// numeric global (its old priority, so existing users keep working), the
// material's own value, a named global, then the shader's default.
///////////////////////////////////////////////////////////////////////////////

#define _SKR_GLOBAL_MAX         32
#define _SKR_GLOBAL_NUMERIC_MAX 16

extern _skr_atomic(uint64_t)            _skr_global_hash      [_SKR_GLOBAL_MAX]; // a slot keeps its name while empty, until a new name claims it
extern _skr_atomic(const skr_tex_t*)    _skr_global_tex       [_SKR_GLOBAL_MAX];
extern _skr_atomic(const skr_buffer_t*) _skr_global_buffer    [_SKR_GLOBAL_MAX];
extern _skr_atomic(uint32_t)            _skr_global_count;     // one past the highest live slot
extern _skr_atomic(uint64_t)            _skr_global_mask;      // a _skr_global_bit per live slot, so most binds skip the scan
extern _skr_atomic(uint32_t)            _skr_global_num_used;  // set once any numeric global is
extern _skr_atomic(const skr_tex_t*)    _skr_global_num_tex   [_SKR_GLOBAL_NUMERIC_MAX];
extern _skr_atomic(const skr_buffer_t*) _skr_global_num_buffer[_SKR_GLOBAL_NUMERIC_MAX];

void _skr_global_set_tex       (const char* name, const skr_tex_t*    opt_tex);
void _skr_global_set_buffer    (const char* name, const skr_buffer_t* opt_buffer);
void _skr_global_set_num_tex   (int32_t     reg,  const skr_tex_t*    opt_tex);
void _skr_global_set_num_buffer(int32_t     reg,  const skr_buffer_t* opt_buffer);
void _skr_globals_reset        (void);
void _skr_binds_init_globals   (const sksc_shader_meta_t* meta, skr_material_bind_t* ref_binds); // fills global_hash/global_bit on binds laid out by _skr_binds_init

// HLSL register number of a bind, or -1 for classes globals never reach
static inline int32_t _skr_bind_register(skr_bind_t bind) {
	switch (bind.register_type) {
	case skr_register_constant:      return bind.slot - SKSC_SLOT_CONSTANT;
	case skr_register_texture:
	case skr_register_read_buffer:   return bind.slot - SKSC_SLOT_TEXTURE;
	case skr_register_readwrite:
	case skr_register_readwrite_tex: return bind.slot - SKSC_SLOT_READWRITE;
	default:                         return -1;
	}
}

// Fibonacci hashing, since FNV's last characters barely reach its high bits
static inline uint8_t _skr_global_bit(uint64_t name_hash) { return (uint8_t)((name_hash * 0x9E3779B97F4A7C15ull) >> 58); }

// Read once per descriptor build, then handed to every find
typedef struct {
	uint64_t mask;
	uint32_t count;
	bool     numeric;
} _skr_globals_snap_t;

static inline _skr_globals_snap_t _skr_globals_snap(void) {
	_skr_globals_snap_t snap;
	snap.count   = _skr_load_u32_acquire(&_skr_global_count);
	snap.mask    = _skr_load_u64(&_skr_global_mask);
	snap.numeric = _skr_load_u32(&_skr_global_num_used) != 0;
	return snap;
}

// The value is acquired before the name is checked again, since a freed slot
// can be renamed under us; the setter stores a new name before its value.
static inline const skr_tex_t* _skr_global_named_tex(_skr_globals_snap_t snap, const skr_material_bind_t* bind) {
	if ((snap.mask & (1ull << bind->global_bit)) == 0) return NULL;
	for (uint32_t i = 0; i < snap.count; i++) {
		if (_skr_load_u64(&_skr_global_hash[i]) != bind->global_hash) continue;
		const skr_tex_t* tex = (const skr_tex_t*)_skr_load_acquire(&_skr_global_tex[i]);
		return _skr_load_u64(&_skr_global_hash[i]) == bind->global_hash ? tex : NULL;
	}
	return NULL;
}

static inline const skr_buffer_t* _skr_global_named_buffer(_skr_globals_snap_t snap, const skr_material_bind_t* bind) {
	if ((snap.mask & (1ull << bind->global_bit)) == 0) return NULL;
	for (uint32_t i = 0; i < snap.count; i++) {
		if (_skr_load_u64(&_skr_global_hash[i]) != bind->global_hash) continue;
		const skr_buffer_t* buffer = (const skr_buffer_t*)_skr_load_acquire(&_skr_global_buffer[i]);
		return _skr_load_u64(&_skr_global_hash[i]) == bind->global_hash ? buffer : NULL;
	}
	return NULL;
}

static inline const skr_tex_t* _skr_bind_tex(_skr_globals_snap_t snap, const skr_material_bind_t* bind) {
	const skr_tex_t* tex = bind->texture;
	if (bind->global_hash == 0) return tex;
	if (snap.numeric) {
		uint32_t reg = (uint32_t)_skr_bind_register(bind->bind);
		const skr_tex_t* numeric = reg < _SKR_GLOBAL_NUMERIC_MAX ? _skr_load_ptr(&_skr_global_num_tex[reg]) : NULL;
		if (numeric) return numeric;
	}
	if (tex && !_skr_tex_is_default(tex)) return tex;
	const skr_tex_t* named = _skr_global_named_tex(snap, bind);
	return named ? named : tex;
}

static inline const skr_buffer_t* _skr_bind_buffer(_skr_globals_snap_t snap, const skr_material_bind_t* bind) {
	if (bind->global_hash == 0) return bind->buffer;
	if (snap.numeric) {
		uint32_t reg = (uint32_t)_skr_bind_register(bind->bind);
		const skr_buffer_t* numeric = reg < _SKR_GLOBAL_NUMERIC_MAX ? _skr_load_ptr(&_skr_global_num_buffer[reg]) : NULL;
		if (numeric) return numeric;
	}
	return bind->buffer ? bind->buffer : _skr_global_named_buffer(snap, bind);
}

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
