// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "skr_pipeline.h"

///////////////////////////////////////////////////////////////////////////////
// Bind pool — a global pool of skr_material_bind_t so render items can
// reference a material's binds by index, matching the Vulkan backend.
// Storage is chunked so entries never move: readers (the per-draw bind-group
// path) take no locks, they follow a release-published chunk pointer. The
// mutex only coordinates writers (material create/destroy from any thread) —
// a rare, already-heavyweight path. A range never crosses a chunk boundary.
// Frees defer to the next frame boundary so render items recorded this frame
// keep reading valid bind data.

#define _SKR_BIND_CHUNK_SHIFT 8
#define _SKR_BIND_CHUNK_SIZE  (1u << _SKR_BIND_CHUNK_SHIFT)
#define _SKR_BIND_CHUNK_MASK  (_SKR_BIND_CHUNK_SIZE - 1)
#define _SKR_BIND_MAX_CHUNKS  1024 // 256k binds; slots recycle, so this is headroom, not a leak budget

typedef struct { uint32_t start, count; } _skr_bind_range_t;

static _skr_atomic(skr_material_bind_t*) _bind_chunks[_SKR_BIND_MAX_CHUNKS];
static uint32_t           _bind_chunk_count;    // writer mutex
static uint32_t           _bind_chunk_used;     // entries used in the last chunk; writer mutex
static _skr_bind_range_t* _bind_free;           // writer mutex
static uint32_t           _bind_free_count;
static uint32_t           _bind_free_capacity;
static _skr_bind_range_t* _bind_deferred;       // writer mutex
static uint32_t           _bind_deferred_count;
static uint32_t           _bind_deferred_capacity;
static _skr_mtx_t         _bind_mutex;
static bool               _bind_mutex_ready;

static void _skr_bind_pool_ensure(void) {
	if (_bind_mutex_ready) return;
	_skr_mtx_init(&_bind_mutex);
	_bind_mutex_ready = true;
}

static void _skr_defaults_create(void);

// Called from skr_init while still single-threaded: mutex setup and eager
// default textures, so neither needs cross-thread once-guards later
void _skr_material_sys_init(void) {
	_skr_bind_pool_ensure();
	_skr_defaults_create();
}

skr_material_bind_t* _skr_bind_pool_get(int32_t start) {
	if (start < 0) return NULL;
	skr_material_bind_t* chunk = _skr_load_acquire(&_bind_chunks[start >> _SKR_BIND_CHUNK_SHIFT]);
	return chunk ? &chunk[start & _SKR_BIND_CHUNK_MASK] : NULL;
}

///////////////////////////////////////////////////////////////////////////////
// Bind group cache — one cached WGPUBindGroup per bind-pool slice (i.e. per
// material instance), keyed on the slice's start index. Dynamic offsets keep
// the group valid across frames; it's rebuilt when the material's own binds
// change (set_tex/set_buffer set epoch 0) or the global bind epoch moves
// (globals/sampler/bump-buffer changes — see _skr_bind_epoch). Storage
// mirrors the pool's chunking: only the draw thread creates entries, worker
// threads at most zero an epoch, and the release-published chunk pointers
// keep both sides safe without locks.

static _skr_atomic(_skr_bind_cache_t*) _bind_cache_chunks[_SKR_BIND_MAX_CHUNKS];

// Draw thread only: the slice's cache entry, allocating its chunk on demand
_skr_bind_cache_t* _skr_bind_cache_slot(int32_t start) {
	if (start < 0) return NULL;
	uint32_t chunk_idx = (uint32_t)start >> _SKR_BIND_CHUNK_SHIFT;
	_skr_bind_cache_t* chunk = _skr_load_acquire(&_bind_cache_chunks[chunk_idx]);
	if (chunk == NULL) {
		chunk = (_skr_bind_cache_t*)_skr_calloc(_SKR_BIND_CHUNK_SIZE, sizeof(_skr_bind_cache_t));
		_skr_store_release(&_bind_cache_chunks[chunk_idx], chunk);
	}
	return &chunk[start & _SKR_BIND_CHUNK_MASK];
}

// Any thread: mark a slice's cached group stale (rebuilt on next draw)
void _skr_bind_cache_invalidate(int32_t start) {
	if (start < 0) return;
	_skr_bind_cache_t* chunk = _skr_load_acquire(&_bind_cache_chunks[(uint32_t)start >> _SKR_BIND_CHUNK_SHIFT]);
	if (chunk) chunk[start & _SKR_BIND_CHUNK_MASK].epoch = 0;
}

// Main thread (drain/shutdown): drop a freed slice's cached group entirely
static void _skr_bind_cache_release(int32_t start) {
	if (start < 0) return;
	_skr_bind_cache_t* chunk = _skr_load_acquire(&_bind_cache_chunks[(uint32_t)start >> _SKR_BIND_CHUNK_SHIFT]);
	if (chunk == NULL) return;
	_skr_bind_cache_t* entry = &chunk[start & _SKR_BIND_CHUNK_MASK];
	if (entry->group) wgpuBindGroupRelease(entry->group);
	entry->group = NULL;
	entry->epoch = 0;
}

static void _skr_bind_range_push(_skr_bind_range_t** arr, uint32_t* count, uint32_t* capacity, _skr_bind_range_t range) {
	if (*count >= *capacity) {
		*capacity = *capacity == 0 ? 16 : *capacity * 2;
		*arr      = (_skr_bind_range_t*)_skr_realloc(*arr, sizeof(_skr_bind_range_t) * *capacity);
	}
	(*arr)[(*count)++] = range;
}

static int32_t _skr_bind_pool_alloc(uint32_t count) {
	if (count == 0) return -1;
	if (count > _SKR_BIND_CHUNK_SIZE) { skr_log(skr_log_critical, "Material needs %u binds, above the pool's %u limit", count, _SKR_BIND_CHUNK_SIZE); return -1; }
	_skr_bind_pool_ensure();
	_skr_mtx_lock(&_bind_mutex);

	int32_t result = -1;
	for (uint32_t i = 0; i < _bind_free_count; i++) {
		if (_bind_free[i].count < count) continue;
		result = (int32_t)_bind_free[i].start;
		if (_bind_free[i].count == count) _bind_free[i] = _bind_free[--_bind_free_count];
		else { _bind_free[i].start += count; _bind_free[i].count -= count; }
		break;
	}
	if (result < 0) {
		// Bump-allocate from the newest chunk; a range that doesn't fit opens
		// a new chunk and the tail is left on the free list
		if (_bind_chunk_count == 0 || _bind_chunk_used + count > _SKR_BIND_CHUNK_SIZE) {
			if (_bind_chunk_count >= _SKR_BIND_MAX_CHUNKS) {
				skr_log(skr_log_critical, "Bind pool is out of chunks");
				_skr_mtx_unlock(&_bind_mutex);
				return -1;
			}
			uint32_t tail = _bind_chunk_count > 0 ? _SKR_BIND_CHUNK_SIZE - _bind_chunk_used : 0;
			if (tail > 0)
				_skr_bind_range_push(&_bind_free, &_bind_free_count, &_bind_free_capacity,
					(_skr_bind_range_t){ .start = ((_bind_chunk_count - 1) << _SKR_BIND_CHUNK_SHIFT) + _bind_chunk_used, .count = tail });
			skr_material_bind_t* chunk = (skr_material_bind_t*)_skr_calloc(_SKR_BIND_CHUNK_SIZE, sizeof(skr_material_bind_t));
			_skr_store_release(&_bind_chunks[_bind_chunk_count], chunk);
			_bind_chunk_count += 1;
			_bind_chunk_used   = 0;
		}
		result = (int32_t)(((_bind_chunk_count - 1) << _SKR_BIND_CHUNK_SHIFT) + _bind_chunk_used);
		_bind_chunk_used += count;
	}
	skr_material_bind_t* chunk = _skr_load_acquire(&_bind_chunks[result >> _SKR_BIND_CHUNK_SHIFT]);
	memset(&chunk[result & _SKR_BIND_CHUNK_MASK], 0, sizeof(skr_material_bind_t) * count);

	_skr_mtx_unlock(&_bind_mutex);
	return result;
}

// Queue a range for reuse at the next frame boundary — render items recorded
// this frame may still index into it
static void _skr_bind_pool_free_deferred(int32_t start, uint32_t count) {
	if (start < 0 || count == 0) return;
	_skr_bind_pool_ensure();
	_skr_mtx_lock(&_bind_mutex);
	_skr_bind_range_push(&_bind_deferred, &_bind_deferred_count, &_bind_deferred_capacity,
		(_skr_bind_range_t){ .start = (uint32_t)start, .count = count });
	_skr_mtx_unlock(&_bind_mutex);
}

void _skr_bind_pool_drain(void) {
	_skr_bind_pool_ensure();
	_skr_mtx_lock(&_bind_mutex);
	for (uint32_t d = 0; d < _bind_deferred_count; d++) {
		uint32_t start = _bind_deferred[d].start;
		uint32_t count = _bind_deferred[d].count;
		uint32_t end   = start + count;
		_skr_bind_cache_release((int32_t)start); // freed slice = dead cached group

		// Coalesce with an adjacent free range when possible — but never
		// across a chunk boundary, ranges must stay within one chunk
		bool merged = false;
		for (uint32_t i = 0; i < _bind_free_count && !merged; i++) {
			if ((_bind_free[i].start >> _SKR_BIND_CHUNK_SHIFT) != (start >> _SKR_BIND_CHUNK_SHIFT)) continue;
			if (_bind_free[i].start + _bind_free[i].count == start) {
				_bind_free[i].count += count;
				for (uint32_t j = 0; j < _bind_free_count; j++)
					if (j != i && _bind_free[j].start == end &&
					    (_bind_free[j].start >> _SKR_BIND_CHUNK_SHIFT) == (start >> _SKR_BIND_CHUNK_SHIFT)) {
						_bind_free[i].count += _bind_free[j].count;
						_bind_free[j] = _bind_free[--_bind_free_count];
						break;
					}
				merged = true;
			} else if (_bind_free[i].start == end) {
				_bind_free[i].start  = start;
				_bind_free[i].count += count;
				merged = true;
			}
		}
		if (!merged)
			_skr_bind_range_push(&_bind_free, &_bind_free_count, &_bind_free_capacity,
				(_skr_bind_range_t){ .start = start, .count = count });
	}
	_bind_deferred_count = 0;
	_skr_mtx_unlock(&_bind_mutex);
}

///////////////////////////////////////////////////////////////////////////////
// Default textures

// Bind group layouts encode each binding's view dimension, so defaults must
// match the shader's declared shape: [color][0 = 2D, 1 = 2D array, 2 = cube,
// 3 = 3D]. Multisampled/1D/cube-array resources get no default (NULL). All
// twelve are tiny, so they're created eagerly at init — no lazy-init locking
// even though materials can be created from worker threads.
static skr_tex_t _defaults[3][4];

static void _skr_defaults_create(void) {
	const uint32_t colors[3] = { 0xFFFFFFFF, 0xFF000000, 0xFF808080 };
	for (int32_t c = 0; c < 3; c++)
	for (int32_t f = 0; f < 4; f++) {
		if (_defaults[c][f].texture != NULL) continue;
		skr_tex_flags_ flags  = f == 1 ? skr_tex_flags_array
		                      : f == 2 ? skr_tex_flags_cubemap
		                      : f == 3 ? skr_tex_flags_3d : skr_tex_flags_none;
		skr_vec3i_t    size   = { 2, 2, (f == 1 || f == 3) ? 2 : 1 };
		uint32_t       layers = f == 2 ? 6 : f == 1 ? 2 : 1;

		uint32_t pixels[2 * 2 * 6];
		for (uint32_t i = 0; i < 2 * 2 * (f == 3 ? 2 : layers); i++) pixels[i] = colors[c];
		skr_tex_data_t data = { .data = pixels, .mip_count = 1, .layer_count = layers };
		skr_tex_sampler_t sampler = {0};
		skr_tex_create(skr_tex_fmt_rgba32, flags, sampler, size, 1, 1, &data, &_defaults[c][f]);
	}
}

static skr_tex_t* _skr_default_tex(const char* value, uint8_t shape) {
	if (shape & SKSC_SHAPE_MS)         return NULL; // no meaningful multisampled default
	if (shape & SKSC_SHAPE_COMPARISON) return NULL; // depth-comparison bindings need a real depth texture

	int32_t color_idx = 0;
	if      (strcmp(value, "black") == 0) color_idx = 1;
	else if (strcmp(value, "gray" ) == 0 ||
	         strcmp(value, "grey" ) == 0) color_idx = 2;

	uint32_t dim     = shape & SKSC_SHAPE_DIM_MASK;
	bool     arrayed = (shape & SKSC_SHAPE_ARRAYED) != 0;
	int32_t  family  = 0;
	if      (dim == SKSC_SHAPE_DIM_CUBE && arrayed) return NULL;
	else if (dim == SKSC_SHAPE_DIM_CUBE)            family = 2;
	else if (dim == SKSC_SHAPE_DIM_3D)              family = 3;
	else if (dim == SKSC_SHAPE_DIM_1D)              return NULL;
	else if (arrayed)                               family = 1;

	skr_tex_t* tex = &_defaults[color_idx][family];
	return tex->texture ? tex : NULL;
}

///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_material_create(skr_material_info_t info, skr_material_t* out_material) {
	if (out_material == NULL) return skr_err_invalid_parameter;
	*out_material = (skr_material_t){0};
	out_material->pipeline_material_idx = -1; // never alias a real registry slot
	out_material->bind_start            = -1;

	if (info.shader == NULL || !skr_shader_is_valid(info.shader)) {
		skr_log(skr_log_warning, "Cannot create material with invalid shader");
		return skr_err_invalid_parameter;
	}

	out_material->key = (_skr_pipeline_material_key_t){
		.shader            = info.shader,
		.cull              = info.cull,
		.write_mask        = info.write_mask ? info.write_mask : skr_write_default,
		.depth_test        = info.depth_test,
		.blend_state       = info.blend_state,
		.alpha_to_coverage = info.alpha_to_coverage,
		.depth_clamp       = info.depth_clamp,
		.wireframe         = info.wireframe,
		.stencil_front     = info.stencil_front,
		.stencil_back      = info.stencil_back,
	};
	_skr_resolve_spec_constants(&info.shader->meta, info.spec_constants, info.spec_constant_count, out_material->key.spec_constant_values);
	out_material->queue_offset = info.queue_offset;

	const sksc_shader_meta_t* meta = &out_material->key.shader->meta;

	if (meta->global_buffer_id >= 0) {
		sksc_shader_buffer_t* global_buffer = &meta->buffers[meta->global_buffer_id];
		out_material->param_buffer_size = global_buffer->size;
		out_material->param_buffer      = _skr_malloc(global_buffer->size);
		if (global_buffer->defaults) memcpy(out_material->param_buffer, global_buffer->defaults, global_buffer->size);
		else                         memset(out_material->param_buffer, 0, global_buffer->size);
	}

	out_material->bind_count = meta->resource_count + meta->buffer_count;
	out_material->bind_start = _skr_bind_pool_alloc(out_material->bind_count);
	skr_material_bind_t* binds = _skr_bind_pool_get(out_material->bind_start);
	if (binds == NULL && out_material->bind_count > 0) { // pool failure already logged
		_skr_free(out_material->param_buffer);
		*out_material = (skr_material_t){ .pipeline_material_idx = -1, .bind_start = -1 };
		return skr_err_out_of_memory;
	}
	for (uint32_t i = 0; i < meta->buffer_count;   i++) binds[i].bind = meta->buffers[i].bind;
	for (uint32_t i = 0; i < meta->resource_count; i++) binds[i + meta->buffer_count].bind = meta->resources[i].bind;

	uint32_t instance_slot = SKSC_SLOT_TEXTURE + (uint32_t)_skr_wgpu.binds.instance_slot; // StructuredBuffers live in t registers
	out_material->instance_buffer_stride = 0;
	for (uint32_t i = 0; i < meta->resource_count; i++)
		if (meta->resources[i].bind.slot == instance_slot && meta->resources[i].bind.stage_bits != 0)
			out_material->instance_buffer_stride = meta->resources[i].element_size;

	out_material->pipeline_material_idx = _skr_pipeline_register_material(&out_material->key);
	if (out_material->pipeline_material_idx < 0) {
		_skr_free(out_material->param_buffer);
		*out_material = (skr_material_t){0};
		out_material->pipeline_material_idx = -1;
		out_material->bind_start            = -1;
		return skr_err_device_error;
	}

	// Default textures for every sampled resource
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		if (meta->resources[i].bind.register_type != skr_register_texture) continue;
		skr_tex_t* def = _skr_default_tex(meta->resources[i].value, meta->resources[i].shape);
		if (def) skr_material_set_tex(out_material, meta->resources[i].name, def);
	}

	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////

void skr_material_set_pipeline(skr_material_t* ref_material, skr_material_info_t info) {
	if (ref_material == NULL) return;

	int32_t old_idx = ref_material->pipeline_material_idx;
	_skr_pipeline_material_key_t new_key = (_skr_pipeline_material_key_t){
		.shader            = info.shader ? info.shader : ref_material->key.shader,
		.cull              = info.cull,
		.write_mask        = info.write_mask ? info.write_mask : skr_write_default,
		.depth_test        = info.depth_test,
		.blend_state       = info.blend_state,
		.alpha_to_coverage = info.alpha_to_coverage,
		.depth_clamp       = info.depth_clamp,
		.wireframe         = info.wireframe,
		.stencil_front     = info.stencil_front,
		.stencil_back      = info.stencil_back,
	};
	_skr_resolve_spec_constants(&new_key.shader->meta, info.spec_constants, info.spec_constant_count, new_key.spec_constant_values);

	ref_material->key          = new_key;
	ref_material->queue_offset = info.queue_offset;
	ref_material->pipeline_material_idx = _skr_pipeline_register_material(&new_key);
	if (old_idx >= 0) _skr_pipeline_unregister_material(old_idx);
	_skr_bind_cache_invalidate(ref_material->bind_start); // layout may have changed
}

bool skr_material_is_valid(const skr_material_t* material) {
	return material != NULL && material->key.shader != NULL;
}

void skr_material_destroy(skr_material_t* ref_material) {
	if (ref_material == NULL) return;
	if (ref_material->pipeline_material_idx >= 0)
		_skr_pipeline_unregister_material(ref_material->pipeline_material_idx);
	_skr_bind_pool_free_deferred(ref_material->bind_start, ref_material->bind_count);
	_skr_free(ref_material->param_buffer);
	memset(ref_material, 0, sizeof(*ref_material));
	ref_material->pipeline_material_idx = -1;
	ref_material->bind_start            = -1;
}

///////////////////////////////////////////////////////////////////////////////

void skr_material_set_tex(skr_material_t* ref_material, const char* name, skr_tex_t* texture) {
	if (ref_material == NULL || name == NULL || ref_material->key.shader == NULL) return;
	skr_bind_t bind = sksc_shader_meta_get_bind(&ref_material->key.shader->meta, name);
	if (bind.stage_bits == 0) return;

	skr_material_bind_t* binds = _skr_bind_pool_get(ref_material->bind_start);
	for (uint32_t i = 0; i < ref_material->bind_count; i++) {
		if (binds[i].bind.slot == bind.slot && binds[i].bind.register_type == bind.register_type) {
			if (binds[i].texture != texture) { // no-op re-sets keep the cached group
				binds[i].texture = texture;
				_skr_bind_cache_invalidate(ref_material->bind_start);
			}
			return;
		}
	}
}

void skr_material_set_buffer(skr_material_t* ref_material, const char* name, skr_buffer_t* buffer) {
	if (ref_material == NULL || name == NULL || ref_material->key.shader == NULL) return;
	skr_bind_t bind = sksc_shader_meta_get_bind(&ref_material->key.shader->meta, name);
	if (bind.stage_bits == 0) return;

	skr_material_bind_t* binds = _skr_bind_pool_get(ref_material->bind_start);
	for (uint32_t i = 0; i < ref_material->bind_count; i++) {
		if (binds[i].bind.slot == bind.slot && binds[i].bind.register_type == bind.register_type) {
			if (binds[i].buffer != buffer) { // no-op re-sets keep the cached group
				binds[i].buffer = buffer;
				_skr_bind_cache_invalidate(ref_material->bind_start);
			}
			return;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

void skr_material_set_params(skr_material_t* ref_material, const void* data, uint32_t size) {
	if (ref_material == NULL || ref_material->param_buffer == NULL || data == NULL) return;
	if (size > ref_material->param_buffer_size) size = ref_material->param_buffer_size;
	memcpy(ref_material->param_buffer, data, size);
}

void skr_material_set_param(skr_material_t* ref_material, const char* name, sksc_shader_var_ type, uint32_t count, const void* data) {
	if (ref_material == NULL || ref_material->param_buffer == NULL || name == NULL || data == NULL) return;
	const sksc_shader_meta_t* meta = &ref_material->key.shader->meta;
	int32_t idx = sksc_shader_meta_get_var_index(meta, name);
	if (idx < 0) return;
	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(meta, idx);
	if (var == NULL || var->type != (uint16_t)type) return;

	uint32_t elem = type == sksc_shader_var_double ? 8 : type == sksc_shader_var_uint8 ? 1 : 4;
	uint32_t size = elem * count;
	if (size > var->size) size = var->size;
	memcpy((uint8_t*)ref_material->param_buffer + var->offset, data, size);
}

void skr_material_get_param(const skr_material_t* material, const char* name, sksc_shader_var_ type, uint32_t count, void* out_data) {
	if (material == NULL || material->param_buffer == NULL || name == NULL || out_data == NULL) return;
	const sksc_shader_meta_t* meta = &material->key.shader->meta;
	int32_t idx = sksc_shader_meta_get_var_index(meta, name);
	if (idx < 0) return;
	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(meta, idx);
	if (var == NULL || var->type != (uint16_t)type) return;

	uint32_t elem = type == sksc_shader_var_double ? 8 : type == sksc_shader_var_uint8 ? 1 : 4;
	uint32_t size = elem * count;
	if (size > var->size) size = var->size;
	memcpy(out_data, (const uint8_t*)material->param_buffer + var->offset, size);
}

bool skr_material_get_param_info(const skr_material_t* material, const char* param_name, skr_shader_param_info_t* opt_out_info) {
	if (material == NULL) return false;
	return skr_shader_get_param_info(material->key.shader, param_name, opt_out_info);
}

bool skr_material_get_tex_info(const skr_material_t* material, const char* tex_name, skr_shader_tex_info_t* opt_out_info) {
	if (material == NULL) return false;
	return skr_shader_get_tex_info(material->key.shader, tex_name, opt_out_info);
}

///////////////////////////////////////////////////////////////////////////////

// Full teardown so skr_shutdown -> skr_init cycles start clean: default
// textures released, bind pool storage freed, locks destroyed for re-init
void _skr_material_sys_shutdown(void) {
	for (int32_t c = 0; c < 3; c++)
	for (int32_t f = 0; f < 4; f++)
		if (_defaults[c][f].texture) skr_tex_destroy(&_defaults[c][f]);
	memset(_defaults, 0, sizeof(_defaults));

	for (uint32_t i = 0; i < _SKR_BIND_MAX_CHUNKS; i++) {
		skr_material_bind_t* chunk = _skr_load_acquire(&_bind_chunks[i]);
		if (chunk) _skr_free(chunk);
		_skr_store_release(&_bind_chunks[i], (skr_material_bind_t*)NULL);

		_skr_bind_cache_t* cache = _skr_load_acquire(&_bind_cache_chunks[i]);
		if (cache) {
			for (uint32_t e = 0; e < _SKR_BIND_CHUNK_SIZE; e++)
				if (cache[e].group) wgpuBindGroupRelease(cache[e].group);
			_skr_free(cache);
		}
		_skr_store_release(&_bind_cache_chunks[i], (_skr_bind_cache_t*)NULL);
	}
	_bind_chunk_count = 0;
	_bind_chunk_used  = 0;
	_skr_free(_bind_free);     _bind_free     = NULL; _bind_free_count     = 0; _bind_free_capacity     = 0;
	_skr_free(_bind_deferred); _bind_deferred = NULL; _bind_deferred_count = 0; _bind_deferred_capacity = 0;
	if (_bind_mutex_ready) { _skr_mtx_destroy(&_bind_mutex); _bind_mutex_ready = false; }
}
