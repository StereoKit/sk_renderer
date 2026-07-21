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
	out_buffer->size   = size;
	out_buffer->type   = type;
	out_buffer->use    = use;
	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////

void skr_buffer_destroy(skr_buffer_t* ref_buffer) {
	if (ref_buffer == NULL || ref_buffer->buffer == NULL) return;
	wgpuBufferRelease(ref_buffer->buffer);
	memset(ref_buffer, 0, sizeof(*ref_buffer));
}

bool skr_buffer_is_valid(const skr_buffer_t* buffer) {
	return buffer != NULL && buffer->buffer != NULL;
}

///////////////////////////////////////////////////////////////////////////////

void skr_buffer_set(skr_buffer_t* ref_buffer, const void* data, uint32_t size_bytes) {
	if (ref_buffer == NULL || ref_buffer->buffer == NULL || data == NULL) return;
	if (size_bytes > ref_buffer->size) size_bytes = ref_buffer->size;

	// writeBuffer stages internally and lands in queue order, so in-flight
	// frames keep the data they were submitted with — no ring needed here.
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

uint32_t skr_buffer_get_size(const skr_buffer_t* buffer) {
	return buffer ? buffer->size : 0;
}

void skr_buffer_set_name(skr_buffer_t* ref_buffer, const char* name) {
	if (ref_buffer == NULL || ref_buffer->buffer == NULL || name == NULL) return;
	wgpuBufferSetLabel(ref_buffer->buffer, (WGPUStringView){ name, strlen(name) });
}
