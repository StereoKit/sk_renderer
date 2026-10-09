// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////

static WGPUBufferUsage _skr_buffer_usage(skr_buffer_type_ type, skr_use_ use) {
	WGPUBufferUsage usage = WGPUBufferUsage_CopyDst;
	if (type & skr_buffer_type_vertex  ) usage |= WGPUBufferUsage_Vertex;
	if (type & skr_buffer_type_index   ) usage |= WGPUBufferUsage_Index;
	if (type & skr_buffer_type_constant) usage |= WGPUBufferUsage_Uniform;
	if (type & skr_buffer_type_storage ) usage |= WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Indirect;
	if (use  & skr_use_compute_read    ) usage |= WGPUBufferUsage_Storage;
	if (use  & skr_use_compute_write   ) usage |= WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
	return usage;
}

static skr_mem_category_ _skr_buffer_category(skr_buffer_type_ type) {
	return (type & (skr_buffer_type_vertex | skr_buffer_type_index)) ? skr_mem_category_geometry : skr_mem_category_buffer;
}

skr_err_ skr_buffer_create(const void* opt_data, uint32_t size_count, uint32_t size_stride, skr_buffer_type_ type, skr_use_ use, skr_buffer_t* out_buffer) {
	if (out_buffer == NULL)                  return skr_err_invalid_parameter;
	memset(out_buffer, 0, sizeof(*out_buffer));
	if (size_count == 0 || size_stride == 0) return skr_err_invalid_parameter;

	uint64_t full_size = (uint64_t)size_count * size_stride;
	if (full_size > UINT32_MAX) return skr_err_invalid_parameter;
	uint32_t size = (uint32_t)full_size;

	// WebGPU requires copy sizes in multiples of 4; pad the allocation so
	// writeBuffer of the rounded size is always legal
	uint64_t alloc_size = (size + 3) & ~3ull;

	WGPUBufferDescriptor desc = {
		.usage            = _skr_buffer_usage(type, use),
		.size             = alloc_size,
		.mappedAtCreation = opt_data != NULL,
	};
	WGPUBuffer buffer = wgpuDeviceCreateBuffer(_skr_wgpu.device, &desc);
	if (buffer == NULL) return skr_err_device_error;

	if (opt_data != NULL) {
		void* mapped = wgpuBufferGetMappedRange(buffer, 0, (size_t)alloc_size);
		if (mapped == NULL) { wgpuBufferRelease(buffer); return skr_err_device_error; }
		memcpy(mapped, opt_data, size);
		wgpuBufferUnmap(buffer);
	}

	out_buffer->buffer = buffer;
	out_buffer->uid    = _skr_uid_new();
	out_buffer->size   = size;
	out_buffer->type   = type;
	out_buffer->use    = use;
	_skr_mem_track(_skr_buffer_category(type), (int64_t)size);
	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////
// Rename slots: writeBuffer lands ahead of unsubmitted work, so a set while recording
// writes another buffer. Slots live as long as the buffer, list items keep their handles.

typedef struct {
	WGPUBuffer buffer;
	uint64_t   uid;
	uint64_t   retired; // _skr_cmd_submits() when it stopped being current; free once that moves
} _skr_buffer_slot_t;

typedef struct _skr_buffer_ring_t {
	_skr_buffer_slot_t* slots;
	uint32_t            count;
	uint32_t            capacity;
} _skr_buffer_ring_t;

static void _skr_buffer_slot_release(const skr_buffer_t* buffer, _skr_buffer_slot_t slot) {
	wgpuBufferRelease(slot.buffer);
	_skr_mem_track(_skr_buffer_category(buffer->type), -(int64_t)buffer->size);
}

// A free slot, or a new one when the set rate outruns the ring
static bool _skr_buffer_slot_take(skr_buffer_t* ref_buffer, _skr_buffer_slot_t* out_slot) {
	_skr_buffer_ring_t* ring    = ref_buffer->_ring;
	uint64_t            submits = _skr_cmd_submits();
	for (uint32_t i = 0; ring && i < ring->count; i++) {
		if (ring->slots[i].retired >= submits) continue;
		*out_slot      = ring->slots[i];
		ring->slots[i] = ring->slots[--ring->count];
		return true;
	}

	WGPUBuffer buffer = wgpuDeviceCreateBuffer(_skr_wgpu.device, &(WGPUBufferDescriptor){
		.usage = _skr_buffer_usage(ref_buffer->type, ref_buffer->use),
		.size  = (ref_buffer->size + 3) & ~3ull,
	});
	if (buffer == NULL) return false;
	_skr_mem_track(_skr_buffer_category(ref_buffer->type), (int64_t)ref_buffer->size);
	*out_slot = (_skr_buffer_slot_t){ .buffer = buffer, .uid = _skr_uid_new() };
	return true;
}

static void _skr_buffer_slot_retire(skr_buffer_t* ref_buffer) {
	if (ref_buffer->_ring == NULL)
		ref_buffer->_ring = _skr_calloc(1, sizeof(_skr_buffer_ring_t));
	_skr_buffer_ring_t* ring = ref_buffer->_ring;
	if (ring->count == ring->capacity) {
		ring->capacity = ring->capacity == 0 ? 2 : ring->capacity * 2;
		ring->slots    = _skr_realloc(ring->slots, ring->capacity * sizeof(_skr_buffer_slot_t));
	}
	ring->slots[ring->count++] = (_skr_buffer_slot_t){
		.buffer  = ref_buffer->buffer,
		.uid     = ref_buffer->uid,
		.retired = _skr_cmd_submits(),
	};
}

///////////////////////////////////////////////////////////////////////////////

void skr_buffer_destroy(skr_buffer_t* ref_buffer) {
	if (ref_buffer == NULL || ref_buffer->buffer == NULL) return;
	wgpuBufferRelease(ref_buffer->buffer);
	_skr_mem_track(_skr_buffer_category(ref_buffer->type), -(int64_t)ref_buffer->size);
	if (ref_buffer->_ring) {
		for (uint32_t i = 0; i < ref_buffer->_ring->count; i++)
			_skr_buffer_slot_release(ref_buffer, ref_buffer->_ring->slots[i]);
		_skr_free(ref_buffer->_ring->slots);
		_skr_free(ref_buffer->_ring);
	}
	memset(ref_buffer, 0, sizeof(*ref_buffer));
}

bool skr_buffer_is_valid(const skr_buffer_t* buffer) {
	return buffer != NULL && buffer->buffer != NULL;
}

///////////////////////////////////////////////////////////////////////////////

void skr_buffer_set(skr_buffer_t* ref_buffer, const void* data, uint32_t size_bytes) {
	if (ref_buffer == NULL || ref_buffer->buffer == NULL || data == NULL) return;
	if (size_bytes > ref_buffer->size) size_bytes = ref_buffer->size;

	if (_skr_cmd_recording()) {
		_skr_buffer_slot_t next;
		if (_skr_buffer_slot_take(ref_buffer, &next)) {
			_skr_buffer_slot_retire(ref_buffer);
			ref_buffer->buffer = next.buffer;
			ref_buffer->uid    = next.uid;
		} else {
			skr_log(skr_log_critical, "skr_buffer_set: out of memory for a rename slot, overwriting data recorded work may still read");
		}
	}

	// Copy size must be a multiple of 4; the allocation is padded for this.
	uint32_t write_size = (size_bytes + 3) & ~3u;
	if (write_size == size_bytes) {
		wgpuQueueWriteBuffer(_skr_wgpu.queue, ref_buffer->buffer, 0, data, size_bytes);
	} else {
		uint8_t* padded = (uint8_t*)_skr_malloc(write_size);
		memcpy(padded, data, size_bytes);
		memset(padded + size_bytes, 0, write_size - size_bytes);
		wgpuQueueWriteBuffer(_skr_wgpu.queue, ref_buffer->buffer, 0, padded, write_size);
		_skr_free(padded);
	}
}

///////////////////////////////////////////////////////////////////////////////

typedef struct { bool done; WGPUMapAsyncStatus status; } _skr_map_ctx_t;

static void _skr_on_map(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)message; (void)userdata2;
	_skr_map_ctx_t* ctx = (_skr_map_ctx_t*)userdata1;
	ctx->done   = true;
	ctx->status = status;
}

// Synchronous GPU->CPU read. Fine on native for tools/tests; on web this
// blocks and is a hard error by contract — use skr_tex_readback-style
// pollable paths instead.
void skr_buffer_get(const skr_buffer_t* buffer, void* ref_buffer, uint32_t buffer_size) {
	if (buffer == NULL || buffer->buffer == NULL || ref_buffer == NULL) return;
	if (buffer_size > buffer->size) buffer_size = buffer->size;
	uint64_t copy_size = (buffer_size + 3) & ~3ull;

	WGPUBufferDescriptor staging_desc = {
		.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
		.size  = copy_size,
	};
	WGPUBuffer staging = wgpuDeviceCreateBuffer(_skr_wgpu.device, &staging_desc);
	if (staging == NULL) return;

	WGPUCommandEncoder encoder = _skr_cmd_get();
	wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer->buffer, 0, staging, 0, copy_size);
	skr_future_t submitted = _skr_cmd_submit();
	skr_future_wait(&submitted);

	_skr_map_ctx_t ctx = {0};
	WGPUFuture f = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, (size_t)copy_size, (WGPUBufferMapCallbackInfo){
		.mode      = WGPUCallbackMode_WaitAnyOnly,
		.callback  = _skr_on_map,
		.userdata1 = &ctx });
	_skr_wait_future(f);

	if (ctx.status == WGPUMapAsyncStatus_Success) {
		const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, (size_t)copy_size);
		if (mapped) memcpy(ref_buffer, mapped, buffer_size);
		wgpuBufferUnmap(staging);
	} else {
		skr_log(skr_log_warning, "skr_buffer_get: map failed (%d)", (int)ctx.status);
	}
	wgpuBufferRelease(staging);
}

///////////////////////////////////////////////////////////////////////////////

static void _skr_on_buffer_readback_map(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)message; (void)userdata2;
	_skr_readback_base_t* ctx = (_skr_readback_base_t*)userdata1;
	// settled != 0 means destroy already gave up on the data; skip the copy
	if (status == WGPUMapAsyncStatus_Success && _skr_load_acquire(&ctx->settled) == 0) {
		const void* mapped = wgpuBufferGetConstMappedRange(ctx->staging, 0, (size_t)ctx->map_bytes);
		if (mapped) memcpy(ctx->dest, mapped, ctx->dest_size);
		wgpuBufferUnmap(ctx->staging);
	} else if (status != WGPUMapAsyncStatus_Success) {
		skr_log(skr_log_warning, "Buffer readback map failed (%d)", (int)status);
	}
	_skr_readback_finish(ctx);
}

skr_err_ skr_buffer_readback(const skr_buffer_t* buffer, skr_buffer_readback_t* out_readback) {
	if (buffer == NULL || buffer->buffer == NULL || out_readback == NULL) return skr_err_invalid_parameter;
	memset(out_readback, 0, sizeof(*out_readback));

	// Storage is the only type both backends can copy out of; see the header
	if (!(buffer->type & skr_buffer_type_storage)) {
		skr_log(skr_log_critical, "skr_buffer_readback needs a storage-type buffer");
		return skr_err_unsupported;
	}

	_skr_readback_base_t* ctx = (_skr_readback_base_t*)_skr_calloc(1, sizeof(_skr_readback_base_t));
	ctx->dest_size = buffer->size;
	ctx->map_bytes = (buffer->size + 3) & ~3u; // in-bounds: skr_buffer_create pads the allocation to 4
	ctx->dest      = _skr_malloc(ctx->dest_size);

	WGPUBufferDescriptor staging_desc = {
		.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
		.size  = ctx->map_bytes,
	};
	ctx->staging = wgpuDeviceCreateBuffer(_skr_wgpu.device, &staging_desc);
	if (ctx->staging == NULL) { _skr_free(ctx->dest); _skr_free(ctx); return skr_err_device_error; }

	wgpuCommandEncoderCopyBufferToBuffer(_skr_cmd_get(), buffer->buffer, 0, ctx->staging, 0, ctx->map_bytes);

	// mapAsync has to follow the submit, which flushes this thread's whole encoder
	_skr_cmd_submit();
	out_readback->future    = _skr_readback_map(ctx, _skr_on_buffer_readback_map);
	out_readback->data      = ctx->dest;
	out_readback->size      = ctx->dest_size;
	out_readback->_internal = ctx;
	return skr_err_success;
}

void skr_buffer_readback_destroy(skr_buffer_readback_t* ref_readback) {
	if (ref_readback == NULL) return;
	_skr_readback_destroy((_skr_readback_base_t*)ref_readback->_internal);
	memset(ref_readback, 0, sizeof(*ref_readback));
}

///////////////////////////////////////////////////////////////////////////////

uint32_t skr_buffer_get_size(const skr_buffer_t* buffer) {
	return buffer ? buffer->size : 0;
}

void skr_buffer_set_name(skr_buffer_t* ref_buffer, const char* name) {
	if (ref_buffer == NULL || ref_buffer->buffer == NULL || name == NULL) return;
	wgpuBufferSetLabel(ref_buffer->buffer, (WGPUStringView){ name, strlen(name) });
}
