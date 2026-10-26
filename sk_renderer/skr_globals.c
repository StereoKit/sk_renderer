// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_skr_shared.h"

#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Global binds and default textures, shared by both backends. Setters run on
// one thread at a time; draws on any thread read through _skr_bind_tex/_buffer.
///////////////////////////////////////////////////////////////////////////////

_skr_atomic(uint64_t)            _skr_global_hash      [_SKR_GLOBAL_MAX];
_skr_atomic(const skr_tex_t*)    _skr_global_tex       [_SKR_GLOBAL_MAX];
_skr_atomic(const skr_buffer_t*) _skr_global_buffer    [_SKR_GLOBAL_MAX];
_skr_atomic(uint32_t)            _skr_global_count;
_skr_atomic(uint64_t)            _skr_global_mask;
_skr_atomic(uint32_t)            _skr_global_num_used;
_skr_atomic(const skr_tex_t*)    _skr_global_num_tex   [_SKR_GLOBAL_NUMERIC_MAX];
_skr_atomic(const skr_buffer_t*) _skr_global_num_buffer[_SKR_GLOBAL_NUMERIC_MAX];

static bool _skr_global_full_logged;

const uint32_t _skr_default_colors[_skr_default_count] = {
	[_skr_default_white] = 0xFFFFFFFF,
	[_skr_default_black] = 0xFF000000,
	[_skr_default_gray ] = 0xFF808080,
	[_skr_default_flat ] = 0xFFFF8080, // 0.5, 0.5, 1
	[_skr_default_rough] = 0xFFFFFF00, // 0, 1, 1
};
uintptr_t _skr_default_tex_begin;
uintptr_t _skr_default_tex_end;

_skr_default_ _skr_default_for(const char* value) {
	if (strcmp(value, "black") == 0) return _skr_default_black;
	if (strcmp(value, "gray" ) == 0 ||
	    strcmp(value, "grey" ) == 0) return _skr_default_gray;
	if (strcmp(value, "flat" ) == 0) return _skr_default_flat;
	if (strcmp(value, "rough") == 0) return _skr_default_rough;
	return _skr_default_white;
}

// skshaderc keeps 31 characters of a name, so hash the same prefix
static uint64_t _skr_global_name_hash(const char* name) {
	char trunc[sizeof(((sksc_shader_resource_t*)0)->name)];
	strncpy(trunc, name, sizeof(trunc) - 1);
	trunc[sizeof(trunc) - 1] = '\0';
	return skr_hash(trunc);
}

static bool _skr_global_live(uint32_t i) {
	return _skr_load_ptr(&_skr_global_tex[i]) != NULL || _skr_load_ptr(&_skr_global_buffer[i]) != NULL;
}

// The slot holding this name, empty or not. Failing that, a claim renames
// the lowest empty slot, so the live range stays short.
static int32_t _skr_global_slot(const char* name, bool claim) {
	uint64_t hash  = _skr_global_name_hash(name);
	int32_t  empty = -1;
	for (uint32_t i = 0; i < _SKR_GLOBAL_MAX; i++) {
		if (_skr_load_u64(&_skr_global_hash[i]) == hash) return (int32_t)i;
		if (empty < 0 && !_skr_global_live(i)) empty = (int32_t)i;
	}
	if (!claim) return -1;
	if (empty < 0) {
		if (!_skr_global_full_logged)
			skr_log(skr_log_critical, "Global '%s' exceeds the maximum of %d named globals set at once, ignoring it", name, _SKR_GLOBAL_MAX);
		_skr_global_full_logged = true;
		return -1;
	}
	_skr_store_u64(&_skr_global_hash[empty], hash);
	return empty;
}

// After every change, so readers skip unset bits and stop past the last live slot
static void _skr_global_refresh(void) {
	uint64_t mask = 0;
	uint32_t end  = 0;
	for (uint32_t i = 0; i < _SKR_GLOBAL_MAX; i++) {
		if (!_skr_global_live(i)) continue;
		mask |= 1ull << _skr_global_bit(_skr_load_u64(&_skr_global_hash[i]));
		end   = i + 1;
	}
	_skr_store_u64        (&_skr_global_mask,  mask);
	_skr_store_u32_release(&_skr_global_count, end);
}

void _skr_global_set_tex(const char* name, const skr_tex_t* opt_tex) {
	int32_t idx = _skr_global_slot(name, opt_tex != NULL);
	if (idx < 0 || _skr_load_ptr(&_skr_global_tex[idx]) == opt_tex) return;
	_skr_store_release(&_skr_global_tex[idx], opt_tex);
	_skr_global_refresh();
}

void _skr_global_set_buffer(const char* name, const skr_buffer_t* opt_buffer) {
	int32_t idx = _skr_global_slot(name, opt_buffer != NULL);
	if (idx < 0 || _skr_load_ptr(&_skr_global_buffer[idx]) == opt_buffer) return;
	_skr_store_release(&_skr_global_buffer[idx], opt_buffer);
	_skr_global_refresh();
}

void _skr_global_set_num_tex(int32_t reg, const skr_tex_t* opt_tex) {
	if (reg < 0) return;
	if (reg >= _SKR_GLOBAL_NUMERIC_MAX) {
		skr_log(skr_log_critical, "Global texture binding %d exceeds maximum of %d slots", reg, _SKR_GLOBAL_NUMERIC_MAX);
		return;
	}
	_skr_store_release(&_skr_global_num_tex[reg], opt_tex);
	if (opt_tex) _skr_store_u32(&_skr_global_num_used, 1);
}

void _skr_global_set_num_buffer(int32_t reg, const skr_buffer_t* opt_buffer) {
	if (reg < 0) return;
	if (reg >= _SKR_GLOBAL_NUMERIC_MAX) {
		skr_log(skr_log_critical, "Global buffer binding %d exceeds maximum of %d slots", reg, _SKR_GLOBAL_NUMERIC_MAX);
		return;
	}
	_skr_store_release(&_skr_global_num_buffer[reg], opt_buffer);
	if (opt_buffer) _skr_store_u32(&_skr_global_num_used, 1);
}

void _skr_globals_reset(void) {
	memset((void*)_skr_global_hash,       0, sizeof(_skr_global_hash));
	memset((void*)_skr_global_tex,        0, sizeof(_skr_global_tex));
	memset((void*)_skr_global_buffer,     0, sizeof(_skr_global_buffer));
	memset((void*)_skr_global_num_tex,    0, sizeof(_skr_global_num_tex));
	memset((void*)_skr_global_num_buffer, 0, sizeof(_skr_global_num_buffer));
	_skr_store_u32(&_skr_global_count,    0);
	_skr_store_u32(&_skr_global_num_used, 0);
	_skr_store_u64(&_skr_global_mask,     0);
	_skr_global_full_logged = false;
}

// The hash a resource's bind matches globals by, 0 for classes globals never reach
static uint64_t _skr_resource_global_hash(const sksc_shader_resource_t* res) {
	switch (res->bind.register_type) {
	case skr_register_texture:
	case skr_register_read_buffer:
	case skr_register_readwrite:
	case skr_register_readwrite_tex: return res->name_hash;
	default:                         return 0;
	}
}

void _skr_binds_init_globals(const sksc_shader_meta_t* meta, skr_material_bind_t* ref_binds) {
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		if ((int32_t)i == meta->global_buffer_id) continue;
		ref_binds[i].global_hash = meta->buffers[i].name_hash;
		ref_binds[i].global_bit  = _skr_global_bit(meta->buffers[i].name_hash);
	}
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		uint64_t hash = _skr_resource_global_hash(&meta->resources[i]);
		ref_binds[meta->buffer_count + i].global_hash = hash;
		ref_binds[meta->buffer_count + i].global_bit  = _skr_global_bit(hash);
	}
}
