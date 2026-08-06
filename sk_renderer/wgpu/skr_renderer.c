// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "skr_pipeline.h"

#include <time.h>

///////////////////////////////////////////////////////////////////////////////
// Pass + frame state.
//
// WebGPU has no multiview: a layered pass runs one render pass per set bit of
// view_mask, each with its own command encoder so the immediate-mode draw API
// can record into all of them simultaneously (an encoder allows only one open
// pass at a time, but encoders are cheap). Pipelines differ per pass only by
// the sk_view_index override value.

#define _SKR_MAX_PASS_VIEWS 6

typedef struct _skr_pass_state_t {
	bool                    active;
	uint32_t                view_count;
	uint32_t                view_indices[_SKR_MAX_PASS_VIEWS];
	WGPUCommandEncoder      encoders    [_SKR_MAX_PASS_VIEWS];
	WGPURenderPassEncoder   passes      [_SKR_MAX_PASS_VIEWS];
	skr_pipeline_pass_key_t base_key;
	skr_vec2i_t             size;
} _skr_pass_state_t;

// Per-frame bump allocation into persistent, grow-on-demand buffers.
// writeBuffer stages data in queue order, so one buffer per usage works.
typedef struct _skr_bump_t {
	WGPUBuffer buffer;
	uint64_t   capacity;
	uint64_t   used;
} _skr_bump_t;

typedef struct _skr_global_bind_t {
	int32_t             slot;
	const skr_buffer_t* buffer;
	const skr_tex_t*    tex;
} _skr_global_bind_t;

static _skr_pass_state_t  _pass;
static _skr_bump_t        _bump_uniform;
static _skr_bump_t        _bump_storage;
static _skr_global_bind_t _globals[16];
static uint32_t           _global_count;

// Most recent system-data allocation, reused by immediate draws
static uint64_t _system_offset;
static uint32_t _system_size;

// Per-pass inputs for lowered postfx/resolve stages: on WebGPU, subpass
// inputs become plain textures, bound from these views. The postfx runner
// sets them (per layer) around each fullscreen stage. Attachments named
// "depth" read the depth view, everything else the color/previous stage —
// the same name convention the Vulkan backend keys postfx depth reads on.
static WGPUTextureView _pass_input_color;
static WGPUTextureView _pass_input_depth;
static WGPUSampler     _fallback_sampler; // plain linear, for unpaired shader samplers

void _skr_pass_inputs_set(WGPUTextureView color, WGPUTextureView depth) {
	_pass_input_color = color;
	_pass_input_depth = depth;
}

// Bind epoch: cached bind groups are valid only while this counter holds
// still. Anything that changes what a rebuilt group would contain without
// going through a material's own set_tex/set_buffer bumps it: global binding
// changes, bump-buffer reallocation, and sampler swaps on live textures.
static uint64_t _bind_epoch = 1;

void     _skr_bind_epoch_bump(void) { _bind_epoch++; }
uint64_t _skr_bind_epoch     (void) { return _bind_epoch; }

///////////////////////////////////////////////////////////////////////////////

static uint64_t _skr_bump_write(_skr_bump_t* bump, WGPUBufferUsage usage, const void* data, uint32_t size) {
	uint64_t offset = (bump->used + _SKR_OFFSET_ALIGN - 1) & ~(uint64_t)(_SKR_OFFSET_ALIGN - 1);
	uint64_t needed = offset + ((size + 3) & ~3u);
	if (needed + _SKR_BUMP_SLACK > bump->capacity || bump->buffer == NULL) {
		uint64_t new_cap = bump->capacity == 0 ? 256 * 1024 : bump->capacity;
		while (needed + _SKR_BUMP_SLACK > new_cap) new_cap *= 2;
		// Old buffer stays alive for this frame's earlier bind groups via
		// their own references; our release here is safe. Cached groups
		// reference the old buffer too — the epoch bump retires them.
		if (bump->buffer) wgpuBufferRelease(bump->buffer);
		WGPUBufferDescriptor desc = { .usage = usage | WGPUBufferUsage_CopyDst, .size = new_cap };
		bump->buffer   = wgpuDeviceCreateBuffer(_skr_wgpu.device, &desc);
		bump->capacity = new_cap;
		bump->used     = 0;
		offset         = 0;
		needed         = (size + 3) & ~3u;
		_skr_bind_epoch_bump();
	}
	uint32_t write_size = (size + 3) & ~3u;
	if (write_size == size) {
		wgpuQueueWriteBuffer(_skr_wgpu.queue, bump->buffer, offset, data, size);
	} else {
		uint8_t padded[4] = {0}; // only the tail needs padding
		wgpuQueueWriteBuffer(_skr_wgpu.queue, bump->buffer, offset, data, size & ~3u);
		memcpy(padded, (const uint8_t*)data + (size & ~3u), size & 3u);
		wgpuQueueWriteBuffer(_skr_wgpu.queue, bump->buffer, offset + (size & ~3u), padded, 4);
	}
	bump->used = needed;
	return offset;
}

///////////////////////////////////////////////////////////////////////////////

// Uniform-bump access for compute dispatch and mipgen parameter uploads
WGPUBuffer _skr_bump_uniform_write(const void* data, uint32_t size, uint64_t* out_offset) {
	*out_offset = _skr_bump_write(&_bump_uniform, WGPUBufferUsage_Uniform, data, size);
	return _bump_uniform.buffer;
}

///////////////////////////////////////////////////////////////////////////////
// Frame timing. GPU time sums per-pass timestamp pairs (core WebGPU has no
// encoder-level timestamps, only pass timestampWrites), harvested a few
// frames later through an async readback ring. CPU time is the frame_begin ->
// frame_end wall clock. Both report the most recently completed frame, like
// the Vulkan backend.

#define _SKR_TIMER_MAX_PAIRS 64
#define _SKR_TIMER_RING      4

typedef struct _skr_timer_frame_t {
	WGPUBuffer        readback;
	uint32_t          pair_count;
	volatile bool     pending;      // submitted, waiting on the map callback
	volatile bool     map_done;
	WGPUMapAsyncStatus map_status;
} _skr_timer_frame_t;

static WGPUQuerySet       _timer_set;
static WGPUBuffer         _timer_resolve;
static _skr_timer_frame_t _timer_ring[_SKR_TIMER_RING];
static uint32_t           _timer_ring_idx;
static uint32_t           _timer_pair_next;
static bool               _timer_init_tried;
static uint64_t           _gpu_time_us;
static uint64_t           _cpu_time_us;
static uint64_t           _cpu_frame_start_ns;

static uint64_t _skr_time_now_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// Hands out a begin/end timestamp index pair for a render or compute pass;
// false when timing is off (no feature) or the frame's budget is spent
static bool _skr_timer_alloc_pair(uint32_t* out_begin, uint32_t* out_end) {
	if (!_skr_wgpu.feat_timestamp || _skr_wgpu.device == NULL) return false;
	if (!_timer_init_tried) {
		_timer_init_tried = true;
		_timer_set = wgpuDeviceCreateQuerySet(_skr_wgpu.device, &(WGPUQuerySetDescriptor){
			.type  = WGPUQueryType_Timestamp,
			.count = _SKR_TIMER_MAX_PAIRS * 2,
		});
		_timer_resolve = wgpuDeviceCreateBuffer(_skr_wgpu.device, &(WGPUBufferDescriptor){
			.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc,
			.size  = _SKR_TIMER_MAX_PAIRS * 2 * sizeof(uint64_t),
		});
	}
	if (_timer_set == NULL || _timer_pair_next >= _SKR_TIMER_MAX_PAIRS) return false;

	*out_begin = _timer_pair_next * 2;
	*out_end   = _timer_pair_next * 2 + 1;
	_timer_pair_next++;
	return true;
}

static WGPUQuerySet _skr_timer_query_set(void) { return _timer_set; }

const WGPUPassTimestampWrites* _skr_timer_pass_writes(WGPUPassTimestampWrites* ts) {
	uint32_t ts_begin, ts_end;
	if (!_skr_timer_alloc_pair(&ts_begin, &ts_end)) return NULL;
	*ts = (WGPUPassTimestampWrites){ .querySet = _skr_timer_query_set(), .beginningOfPassWriteIndex = ts_begin, .endOfPassWriteIndex = ts_end };
	return ts;
}

static void _skr_on_timer_map(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)message; (void)userdata2;
	_skr_timer_frame_t* frame = (_skr_timer_frame_t*)userdata1;
	frame->map_status = status;
	frame->map_done   = true;
}

// Harvest finished readbacks (frame_begin) — sums each pass's duration.
// If several ring entries completed, the last one visited wins; for a stats
// readout that's fine, so no need to order them by frame age.
static void _skr_timer_harvest(void) {
	for (uint32_t i = 0; i < _SKR_TIMER_RING; i++) {
		_skr_timer_frame_t* frame = &_timer_ring[i];
		if (!frame->pending || !frame->map_done) continue;
		if (frame->map_status == WGPUMapAsyncStatus_Success) {
			const uint64_t* stamps = (const uint64_t*)wgpuBufferGetConstMappedRange(frame->readback, 0, frame->pair_count * 2 * sizeof(uint64_t));
			if (stamps) {
				uint64_t total_ns = 0;
				for (uint32_t p = 0; p < frame->pair_count; p++)
					if (stamps[p * 2 + 1] > stamps[p * 2]) total_ns += stamps[p * 2 + 1] - stamps[p * 2];
				_gpu_time_us = total_ns / 1000;
			}
			wgpuBufferUnmap(frame->readback);
		}
		frame->pending  = false;
		frame->map_done = false;
	}
}

// Kick off this frame's resolve + readback (frame_end)
static void _skr_timer_flush(void) {
	uint32_t pairs = _timer_pair_next;
	_timer_pair_next = 0;
	if (pairs == 0 || _timer_set == NULL) return;

	_skr_timer_frame_t* frame = &_timer_ring[_timer_ring_idx];
	if (frame->pending) return; // ring full: skip this frame's numbers
	_timer_ring_idx = (_timer_ring_idx + 1) % _SKR_TIMER_RING;

	uint64_t size = (uint64_t)pairs * 2 * sizeof(uint64_t);
	if (frame->readback == NULL)
		frame->readback = wgpuDeviceCreateBuffer(_skr_wgpu.device, &(WGPUBufferDescriptor){
			.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
			.size  = _SKR_TIMER_MAX_PAIRS * 2 * sizeof(uint64_t),
		});

	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_skr_wgpu.device, NULL);
	wgpuCommandEncoderResolveQuerySet(encoder, _timer_set, 0, pairs * 2, _timer_resolve, 0);
	wgpuCommandEncoderCopyBufferToBuffer(encoder, _timer_resolve, 0, frame->readback, 0, size);
	WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
	wgpuCommandEncoderRelease(encoder);
	wgpuQueueSubmit(_skr_wgpu.queue, 1, &cmd);
	wgpuCommandBufferRelease(cmd);

	frame->pair_count = pairs;
	frame->map_done   = false;
	frame->pending    = true;
	wgpuBufferMapAsync(frame->readback, WGPUMapMode_Read, 0, (size_t)size, (WGPUBufferMapCallbackInfo){
		.mode      = _SKR_CB_MODE_ASYNC,
		.callback  = _skr_on_timer_map,
		.userdata1 = frame,
	});
}

void skr_renderer_frame_begin(void) {
	wgpuInstanceProcessEvents(_skr_wgpu.instance);
	_bump_uniform.used = 0;
	_bump_storage.used = 0;
	_system_size       = 0;
	// Bind ranges and material slots freed last frame are safe to recycle
	// now — the items that referenced them were encoded before the frame ended
	_skr_bind_pool_drain();
	_skr_pipeline_registry_drain();
	_skr_transient_tick();
	_skr_timer_harvest();
	_cpu_frame_start_ns = _skr_time_now_ns();
}

void skr_renderer_frame_end(skr_surface_t** opt_surfaces, uint32_t count) {
	(void)opt_surfaces; (void)count;
	_skr_cmd_submit();
	_skr_timer_flush();
	_cpu_time_us = (_skr_time_now_ns() - _cpu_frame_start_ns) / 1000;
	wgpuInstanceProcessEvents(_skr_wgpu.instance);
}

uint64_t skr_renderer_get_gpu_time_us(void) { return _gpu_time_us; }
uint64_t skr_renderer_get_cpu_time_us(void) { return _cpu_time_us; }

///////////////////////////////////////////////////////////////////////////////

void skr_renderer_begin_pass(skr_tex_t* color, skr_tex_t* depth, skr_tex_t* opt_resolve, skr_clear_ clear, skr_vec4_t clear_color, float clear_depth, uint32_t clear_stencil, uint32_t view_mask, uint32_t correlation_mask) {
	(void)correlation_mask; // meaningful only for real multiview
	if (_pass.active) { skr_log(skr_log_warning, "skr_renderer_begin_pass while a pass is active"); return; }
	if (color == NULL && depth == NULL) return;

	// Uploads recorded so far must land before the pass' submissions
	_skr_cmd_submit();

	if (view_mask == 0) view_mask = 1;
	memset(&_pass, 0, sizeof(_pass));

	skr_tex_t* size_src = color ? color : depth;
	_pass.size     = (skr_vec2i_t){ size_src->size.x, size_src->size.y };
	_pass.base_key = (skr_pipeline_pass_key_t){
		.color_format = color ? _skr_tex_fmt_to_wgpu(color->format) : WGPUTextureFormat_Undefined,
		.depth_format = depth ? _skr_tex_fmt_to_wgpu(depth->format) : WGPUTextureFormat_Undefined,
		.sample_count = color ? color->samples : (depth ? depth->samples : 1),
	};

	for (uint32_t bit = 0; bit < 32 && _pass.view_count < _SKR_MAX_PASS_VIEWS; bit++) {
		if (!(view_mask & (1u << bit))) continue;
		uint32_t v = _pass.view_count++;
		_pass.view_indices[v] = bit;

		_pass.encoders[v] = wgpuDeviceCreateCommandEncoder(_skr_wgpu.device, NULL);

		WGPURenderPassColorAttachment color_attach = {
			.view          = color ? _skr_tex_layer_view(color, bit) : NULL,
			.depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED,
			.resolveTarget = opt_resolve ? _skr_tex_layer_view(opt_resolve, bit) : NULL,
			// WebGPU has no discard load-op, and Load is the expensive one here
			// just as on Vulkan, so discard maps to Clear, not Load.
			.loadOp        = (clear & (skr_clear_color | skr_clear_color_discard)) ? WGPULoadOp_Clear : WGPULoadOp_Load,
			.storeOp       = (color && opt_resolve && !(color->flags & skr_tex_flags_readable)) ? WGPUStoreOp_Discard : WGPUStoreOp_Store,
			.clearValue    = { clear_color.x, clear_color.y, clear_color.z, clear_color.w },
		};
		bool has_stencil = _pass.base_key.depth_format == WGPUTextureFormat_Depth24PlusStencil8 ||
		                   _pass.base_key.depth_format == WGPUTextureFormat_Depth32FloatStencil8;
		WGPURenderPassDepthStencilAttachment depth_attach = {
			.view              = depth ? _skr_tex_layer_view(depth, bit) : NULL,
			.depthLoadOp       = (clear & skr_clear_depth) ? WGPULoadOp_Clear : WGPULoadOp_Load,
			.depthStoreOp      = WGPUStoreOp_Store,
			.depthClearValue   = clear_depth,
			.stencilLoadOp     = !has_stencil ? WGPULoadOp_Undefined : (clear & skr_clear_stencil) ? WGPULoadOp_Clear : WGPULoadOp_Load,
			.stencilStoreOp    = !has_stencil ? WGPUStoreOp_Undefined : WGPUStoreOp_Store,
			.stencilClearValue = clear_stencil,
		};

		WGPURenderPassDescriptor desc = {
			.colorAttachmentCount   = color ? 1u : 0u,
			.colorAttachments       = color ? &color_attach : NULL,
			.depthStencilAttachment = depth ? &depth_attach : NULL,
		};
		WGPUPassTimestampWrites ts;
		desc.timestampWrites = _skr_timer_pass_writes(&ts);
		_pass.passes[v] = wgpuCommandEncoderBeginRenderPass(_pass.encoders[v], &desc);
	}
	_pass.active = true;
}

void skr_renderer_end_pass(void) {
	if (!_pass.active) return;

	WGPUCommandBuffer buffers[_SKR_MAX_PASS_VIEWS];
	for (uint32_t v = 0; v < _pass.view_count; v++) {
		wgpuRenderPassEncoderEnd(_pass.passes[v]);
		wgpuRenderPassEncoderRelease(_pass.passes[v]);
		buffers[v] = wgpuCommandEncoderFinish(_pass.encoders[v], NULL);
		wgpuCommandEncoderRelease(_pass.encoders[v]);
	}
	wgpuQueueSubmit(_skr_wgpu.queue, _pass.view_count, buffers);
	for (uint32_t v = 0; v < _pass.view_count; v++)
		wgpuCommandBufferRelease(buffers[v]);
	memset(&_pass, 0, sizeof(_pass));
}

///////////////////////////////////////////////////////////////////////////////

void skr_renderer_set_viewport(skr_rect_t viewport) {
	if (!_pass.active) return;
	for (uint32_t v = 0; v < _pass.view_count; v++)
		wgpuRenderPassEncoderSetViewport(_pass.passes[v], viewport.x, viewport.y, viewport.w, viewport.h, 0.0f, 1.0f);
}

void skr_renderer_set_scissor(skr_recti_t scissor) {
	if (!_pass.active) return;
	for (uint32_t v = 0; v < _pass.view_count; v++)
		wgpuRenderPassEncoderSetScissorRect(_pass.passes[v], (uint32_t)scissor.x, (uint32_t)scissor.y, (uint32_t)scissor.w, (uint32_t)scissor.h);
}

// Both setters bump the bind epoch only on real changes — scenes re-assert
// the same globals every frame, and a no-op set must not retire every
// cached bind group.
void skr_renderer_set_global_constants(int32_t bind, const skr_buffer_t* buffer) {
	for (uint32_t i = 0; i < _global_count; i++)
		if (_globals[i].slot == bind) {
			if (_globals[i].buffer != buffer) { _globals[i].buffer = buffer; _skr_bind_epoch_bump(); }
			return;
		}
	if (_global_count < 16) {
		_globals[_global_count++] = (_skr_global_bind_t){ .slot = bind, .buffer = buffer };
		_skr_bind_epoch_bump();
	}
}

void skr_renderer_set_global_texture(int32_t bind, const skr_tex_t* tex) {
	for (uint32_t i = 0; i < _global_count; i++)
		if (_globals[i].slot == bind) {
			if (_globals[i].tex != tex) { _globals[i].tex = tex; _skr_bind_epoch_bump(); }
			return;
		}
	if (_global_count < 16) {
		_globals[_global_count++] = (_skr_global_bind_t){ .slot = bind, .tex = tex };
		_skr_bind_epoch_bump();
	}
}

///////////////////////////////////////////////////////////////////////////////
// Bind group assembly. The reserved material/system/instance slots reference
// the bump buffers at offset 0 with the shader-declared sizes; each draw's
// actual position arrives as dynamic offsets at SetBindGroup time (see
// _skr_dynamic_offsets), which is what makes bind groups cacheable.

static uint32_t _skr_meta_buffer_size(const sksc_shader_meta_t* meta, uint32_t slot) {
	for (uint32_t i = 0; i < meta->buffer_count; i++)
		if (meta->buffers[i].bind.slot == slot) return meta->buffers[i].size;
	return 0;
}

static void _skr_bump_ensure(_skr_bump_t* bump, WGPUBufferUsage usage) {
	if (bump->buffer != NULL) return;
	uint64_t cap = 256 * 1024;
	while (cap < 2 * (uint64_t)_SKR_BUMP_SLACK) cap *= 2;
	WGPUBufferDescriptor desc = { .usage = usage | WGPUBufferUsage_CopyDst, .size = cap };
	bump->buffer   = wgpuDeviceCreateBuffer(_skr_wgpu.device, &desc);
	bump->capacity = cap;
	bump->used     = 0;
	_skr_bind_epoch_bump();
}

// Fills the dynamic-offset array for a draw, ordered by binding number the
// way SetBindGroup expects. Presence must mirror _skr_bind_layout_create's
// dynamic entries exactly: declared in meta and visible to stage_mask.
uint32_t _skr_dynamic_offsets(const sksc_shader_meta_t* meta, uint8_t stage_mask, const _skr_draw_buffers_t* db, uint32_t out_offsets[3]) {
	uint32_t material_slot = (uint32_t)_skr_wgpu.binds.material_slot;
	uint32_t system_slot   = (uint32_t)_skr_wgpu.binds.system_slot;
	uint32_t instance_slot = SKSC_SLOT_TEXTURE + (uint32_t)_skr_wgpu.binds.instance_slot; // StructuredBuffers live in t registers

	struct { uint32_t binding, offset; } d[3];
	uint32_t n = 0;
	for (uint32_t i = 0; i < meta->buffer_count && n < 3; i++) {
		if (!(meta->buffers[i].bind.stage_bits & stage_mask)) continue;
		uint32_t slot = meta->buffers[i].bind.slot;
		if      (slot == material_slot) { d[n].binding = slot; d[n].offset = (uint32_t)db->material_offset; n++; }
		else if (slot == system_slot)   { d[n].binding = slot; d[n].offset = (uint32_t)db->system_offset;   n++; }
	}
	for (uint32_t i = 0; i < meta->resource_count && n < 3; i++) {
		if (!(meta->resources[i].bind.stage_bits & stage_mask)) continue;
		if (meta->resources[i].bind.register_type == skr_register_read_buffer && meta->resources[i].bind.slot == instance_slot) {
			d[n].binding = instance_slot; d[n].offset = (uint32_t)db->instance_offset; n++;
		}
	}
	// Insertion sort by binding — SetBindGroup consumes offsets in
	// binding-number order of the layout's dynamic entries
	for (uint32_t i = 1; i < n; i++)
		for (uint32_t j = i; j > 0 && d[j-1].binding > d[j].binding; j--) {
			uint32_t tb = d[j].binding, to = d[j].offset;
			d[j].binding = d[j-1].binding; d[j].offset = d[j-1].offset;
			d[j-1].binding = tb; d[j-1].offset = to;
		}
	for (uint32_t i = 0; i < n; i++) out_offsets[i] = d[i].offset;
	return n;
}

WGPUBindGroup _skr_build_bind_group_meta(const sksc_shader_meta_t* meta, WGPUBindGroupLayout layout, const skr_material_bind_t* binds, uint32_t bind_count) {
	uint32_t material_slot = (uint32_t)_skr_wgpu.binds.material_slot;
	uint32_t system_slot   = (uint32_t)_skr_wgpu.binds.system_slot;
	uint32_t instance_slot = SKSC_SLOT_TEXTURE + (uint32_t)_skr_wgpu.binds.instance_slot; // StructuredBuffers live in t registers

	WGPUBindGroupEntry entries[64];
	uint32_t           entry_count = 0;

	// Buffers + resources from the material's bind list
	for (uint32_t i = 0; i < bind_count && entry_count < 60; i++) {
		const skr_material_bind_t* bind = &binds[i];
		uint32_t slot = bind->bind.slot;

		if ((slot == material_slot || slot == system_slot) && bind->bind.register_type == skr_register_constant) {
			_skr_bump_ensure(&_bump_uniform, WGPUBufferUsage_Uniform);
			uint32_t size = _skr_meta_buffer_size(meta, slot);
			if (size > 0)
				entries[entry_count++] = (WGPUBindGroupEntry){ .binding = slot, .buffer = _bump_uniform.buffer, .offset = 0, .size = size };
			continue;
		}
		if (slot == instance_slot && bind->bind.register_type == skr_register_read_buffer) {
			_skr_bump_ensure(&_bump_storage, WGPUBufferUsage_Storage);
			entries[entry_count++] = (WGPUBindGroupEntry){ .binding = slot, .buffer = _bump_storage.buffer, .offset = 0, .size = _SKR_BUMP_SLACK };
			continue;
		}

		// Lowered subpass inputs bind the current pass-input views directly
		if (bind->bind.register_type == skr_register_input_attachment) {
			WGPUTextureView view = _pass_input_color;
			for (uint32_t r = 0; r < meta->resource_count; r++)
				if (meta->resources[r].bind.slot == slot &&
				    meta->resources[r].bind.register_type == skr_register_input_attachment &&
				    strcmp(meta->resources[r].name, "depth") == 0) { view = _pass_input_depth; break; }
			if (view == NULL) {
				skr_log(skr_log_warning, "Material reads a subpass input, but no pass input is bound (postfx materials only draw inside pass submission)");
				return NULL;
			}
			entries[entry_count++] = (WGPUBindGroupEntry){ .binding = slot, .textureView = view };
			continue;
		}

		// Globals take priority over the material's own binds (which include
		// baked-in defaults), matching the Vulkan backend — texture-type
		// bindings only see texture globals, buffer-type only buffer globals.
		// Global binds use raw register indices (t5, b1, ...) while material
		// slots carry the WGSL register shift, so unshift before matching.
		bool is_buffer_bind = bind->bind.register_type == skr_register_constant
		                   || bind->bind.register_type == skr_register_read_buffer
		                   || bind->bind.register_type == skr_register_readwrite;
		int32_t global_slot = (int32_t)slot;
		switch (bind->bind.register_type) {
			case skr_register_texture:
			case skr_register_read_buffer:
			case skr_register_input_attachment: global_slot -= SKSC_SLOT_TEXTURE;   break;
			case skr_register_readwrite:
			case skr_register_readwrite_tex:    global_slot -= SKSC_SLOT_READWRITE; break;
			default: break;
		}

		const skr_tex_t*    tex = NULL;
		const skr_buffer_t* buf = NULL;
		for (uint32_t g = 0; g < _global_count; g++) {
			if (_globals[g].slot != global_slot) continue;
			if (is_buffer_bind) buf = _globals[g].buffer;
			else                tex = _globals[g].tex;
			break;
		}
		if (tex == NULL && buf == NULL) {
			if (is_buffer_bind) buf = bind->buffer;
			else                tex = bind->texture;
		}

		if (buf && buf->buffer) {
			entries[entry_count++] = (WGPUBindGroupEntry){ .binding = slot, .buffer = buf->buffer, .offset = 0, .size = (buf->size + 3) & ~3u };
		} else if (tex && tex->view) {
			entries[entry_count++] = (WGPUBindGroupEntry){ .binding = slot, .textureView = tex->view };
		} else {
			skr_log(skr_log_warning, "Material is missing a binding for slot %u", slot);
			return NULL;
		}
	}

	// Standalone samplers: each takes its paired texture's sampler. The
	// shader's usage decides filtering vs comparison (via the paired
	// texture's meta shape), and the matching variant gets bound.
	for (uint32_t s = 0; s < meta->sampler_count && entry_count < 64; s++) {
		const sksc_shader_sampler_t* samp       = &meta->samplers[s];
		bool                         comparison = _skr_sampler_is_comparison(meta, samp->paired_slot);

		// Globals first, same as the texture binds above (raw t-register
		// indices; paired_slot carries the +100 shift)
		const skr_tex_t* paired = NULL;
		for (uint32_t g = 0; g < _global_count; g++)
			if (_globals[g].slot + SKSC_SLOT_TEXTURE == (int32_t)samp->paired_slot && _globals[g].tex) { paired = _globals[g].tex; break; }
		if (paired == NULL) {
			for (uint32_t i = 0; i < bind_count; i++)
				if (binds[i].bind.slot == samp->paired_slot && binds[i].texture) { paired = binds[i].texture; break; }
		}
		WGPUSampler sampler = NULL;
		if (paired)
			sampler = comparison
				? (paired->sampler_compare ? paired->sampler_compare : paired->sampler)
				: paired->sampler;
		if (sampler == NULL) {
			// Unpaired or texture-less: fall back to a plain linear sampler
			if (_fallback_sampler == NULL) {
				skr_tex_sampler_t settings = {0};
				_fallback_sampler = _skr_sampler_create(settings, 0);
			}
			sampler = _fallback_sampler;
		}
		entries[entry_count++] = (WGPUBindGroupEntry){ .binding = samp->slot, .sampler = sampler };
	}

	WGPUBindGroupDescriptor desc = {
		.layout     = layout,
		.entryCount = entry_count,
		.entries    = entries,
	};
	return wgpuDeviceCreateBindGroup(_skr_wgpu.device, &desc);
}

static WGPUBindGroup _skr_build_bind_group(int32_t material_idx, const skr_material_bind_t* binds, uint32_t bind_count) {
	const _skr_pipeline_material_key_t* key = _skr_pipeline_get_key(material_idx);
	if (key == NULL || key->shader == NULL) return NULL;
	return _skr_build_bind_group_meta(&key->shader->meta, _skr_pipeline_get_bind_layout(material_idx), binds, bind_count);
}

// One cached bind group per material slice: dynamic offsets carry the
// per-draw buffer positions, so a group survives across draws and frames
// until the slice's binds change or the global bind epoch moves. Slices that
// can't cache (uncacheable = pass-input readers whose views change per stage)
// get a fresh group the caller must release (*out_owned).
static WGPUBindGroup _skr_bind_group_get(int32_t material_idx, int32_t bind_start, uint32_t bind_count, bool cacheable, bool* out_owned) {
	*out_owned = false;
	uint64_t           epoch = _skr_bind_epoch();
	_skr_bind_cache_t* cache = cacheable ? _skr_bind_cache_slot(bind_start) : NULL;
	if (cache && cache->group && cache->epoch == epoch)
		return cache->group;

	WGPUBindGroup group = _skr_build_bind_group(material_idx, _skr_bind_pool_get(bind_start), bind_count);
	if (group == NULL) return NULL;
	if (cache) {
		if (cache->group) wgpuBindGroupRelease(cache->group);
		cache->group = group;
		cache->epoch = epoch;
	} else {
		*out_owned = true;
	}
	return group;
}

// Encode one fullscreen-triangle pass: the pipeline's vertex stage generates
// the triangle from vertex indices, so there's no mesh or vertex buffer
void _skr_fullscreen_pass(WGPUCommandEncoder encoder, WGPUTextureView target, WGPULoadOp load_op, const skr_recti_t* opt_bounds, WGPURenderPipeline pipeline, WGPUBindGroup bind_group, const uint32_t* dyn_offsets, uint32_t dyn_count) {
	WGPURenderPassColorAttachment color_attach = {
		.view       = target,
		.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
		.loadOp     = load_op,
		.storeOp    = WGPUStoreOp_Store,
	};
	WGPURenderPassDescriptor desc = {
		.colorAttachmentCount = 1,
		.colorAttachments     = &color_attach,
	};
	WGPUPassTimestampWrites ts;
	desc.timestampWrites = _skr_timer_pass_writes(&ts);

	WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &desc);
	if (opt_bounds) {
		wgpuRenderPassEncoderSetViewport   (pass, (float)opt_bounds->x, (float)opt_bounds->y, (float)opt_bounds->w, (float)opt_bounds->h, 0.0f, 1.0f);
		wgpuRenderPassEncoderSetScissorRect(pass, (uint32_t)opt_bounds->x, (uint32_t)opt_bounds->y, (uint32_t)opt_bounds->w, (uint32_t)opt_bounds->h);
	}
	wgpuRenderPassEncoderSetPipeline (pass, pipeline);
	wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, dyn_count, dyn_offsets);
	wgpuRenderPassEncoderDraw        (pass, 3, 1, 0, 0);
	wgpuRenderPassEncoderEnd(pass);
	wgpuRenderPassEncoderRelease(pass);
}

///////////////////////////////////////////////////////////////////////////////

static void _skr_draw_item(const skr_render_item_t* item, const _skr_draw_buffers_t* db) {
	const _skr_pipeline_material_key_t* key = _skr_pipeline_get_key(item->pipeline_material_idx);
	if (key == NULL || key->shader == NULL) return;

	bool          owned;
	WGPUBindGroup bind_group = _skr_bind_group_get(item->pipeline_material_idx, item->bind_start, item->bind_count, true, &owned);
	if (bind_group == NULL) return;

	uint32_t dyn_offsets[3];
	uint32_t dyn_count = _skr_dynamic_offsets(&key->shader->meta, (uint8_t)(skr_stage_vertex | skr_stage_pixel), db, dyn_offsets);

	uint32_t vb_count = (item->flags & skr_item_flag_vb_count_mask) >> skr_item_flag_vb_count_shift;

	for (uint32_t v = 0; v < _pass.view_count; v++) {
		skr_pipeline_pass_key_t pass_key = _pass.base_key;
		pass_key.view_index = _pass.view_indices[v];

		WGPURenderPipeline pipeline = _skr_pipeline_get(item->pipeline_material_idx, &pass_key, item->pipeline_vert_idx);
		if (pipeline == NULL) continue;

		WGPURenderPassEncoder pass = _pass.passes[v];
		wgpuRenderPassEncoderSetPipeline (pass, pipeline);
		wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, dyn_count, dyn_offsets);
		for (uint32_t b = 0; b < vb_count; b++)
			if (item->vertex_buffers[b])
				wgpuRenderPassEncoderSetVertexBuffer(pass, b, item->vertex_buffers[b], 0, WGPU_WHOLE_SIZE);

		if (item->index_buffer && item->index_count > 0) {
			wgpuRenderPassEncoderSetIndexBuffer(pass, item->index_buffer,
				(item->flags & skr_item_flag_index_32bit) ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
			wgpuRenderPassEncoderDrawIndexed(pass, (uint32_t)item->index_count, item->instance_count, (uint32_t)item->first_index, item->vertex_offset, 0);
		} else {
			wgpuRenderPassEncoderDraw(pass, item->vert_count, item->instance_count, 0, 0);
		}
	}
	if (owned) wgpuBindGroupRelease(bind_group);
}

void skr_renderer_draw(skr_render_list_t* list, const void* system_data, uint32_t system_data_size) {
	if (list == NULL || list->count == 0 || !_pass.active) return;

	_skr_render_list_sort(list);

	uint64_t material_base = 0, system_base = 0, instance_base = 0;
	if (list->material_data_used > 0)
		material_base = _skr_bump_write(&_bump_uniform, WGPUBufferUsage_Uniform, list->material_data, list->material_data_used);
	if (system_data != NULL && system_data_size > 0) {
		system_base    = _skr_bump_write(&_bump_uniform, WGPUBufferUsage_Uniform, system_data, system_data_size);
		_system_offset = system_base;
		_system_size   = system_data_size;
	} else {
		system_base      = _system_offset;
		system_data_size = _system_size;
	}
	if (list->instance_data_used > 0)
		instance_base = _skr_bump_write(&_bump_storage, WGPUBufferUsage_Storage, list->instance_data, list->instance_data_used);

	for (uint32_t i = 0; i < list->count; i++) {
		const skr_render_item_t* item = &list->items[i];
		_skr_draw_buffers_t db = {
			.material_offset = material_base + item->param_data_offset,
			.material_size   = item->param_buffer_size,
			.system_offset   = system_base,
			.system_size     = system_data_size,
			.instance_offset = instance_base + item->instance_offset,
			.instance_size   = item->instance_data_size * item->instance_count,
		};
		// The instance binding is a fixed _SKR_BUMP_SLACK window; a bigger
		// slice would fail device validation far from the cause, so skip it
		// here with a pointed message instead
		if (db.instance_size > _SKR_BUMP_SLACK) {
			skr_log(skr_log_critical, "Draw skipped: %u bytes of instance data exceeds the %u byte per-draw limit", (uint32_t)db.instance_size, (uint32_t)_SKR_BUMP_SLACK);
			continue;
		}
		_skr_draw_item(item, &db);
	}
}

void skr_renderer_draw_mesh_immediate(skr_mesh_t* mesh, skr_material_t* material, int32_t first_index, int32_t index_count, int32_t vertex_offset, int32_t instance_count) {
	if (mesh == NULL || material == NULL || !_pass.active) return;
	if (!skr_material_is_valid(material) || material->pipeline_material_idx < 0) return;

	if (mesh->vert_type != NULL && mesh->vert_type->pipeline_idx < 0)
		((skr_vert_type_t*)mesh->vert_type)->pipeline_idx = _skr_pipeline_register_vertformat(mesh->vert_type);

	_skr_draw_buffers_t db = {
		.system_offset = _system_offset,
		.system_size   = _system_size,
	};
	if (material->param_buffer && material->param_buffer_size > 0) {
		db.material_offset = _skr_bump_write(&_bump_uniform, WGPUBufferUsage_Uniform, material->param_buffer, material->param_buffer_size);
		db.material_size   = material->param_buffer_size;
	}

	skr_render_item_t item = {0};
	for (uint32_t i = 0; i < mesh->vertex_buffer_count && i < SKR_MAX_VERTEX_BUFFERS; i++)
		item.vertex_buffers[i] = mesh->vertex_buffers[i].buffer;
	item.index_buffer          = mesh->index_buffer.buffer;
	item.vert_count            = mesh->vert_count;
	item.pipeline_vert_idx     = mesh->vert_type ? (uint16_t)mesh->vert_type->pipeline_idx : (uint16_t)0xFFFF;
	item.pipeline_material_idx = (uint16_t)material->pipeline_material_idx;
	item.param_buffer_size     = (uint16_t)material->param_buffer_size;
	item.bind_start            = material->bind_start;
	item.bind_count            = (uint8_t)material->bind_count;
	item.first_index           = first_index;
	item.index_count           = index_count > 0 ? index_count : (int32_t)mesh->ind_count;
	item.vertex_offset         = vertex_offset;
	item.instance_count        = instance_count > 0 ? (uint32_t)instance_count : 1;
	item.flags = ((mesh->ind_format_wgpu == WGPUIndexFormat_Uint32) ? skr_item_flag_index_32bit : 0)
	           | (mesh->vertex_buffer_count << skr_item_flag_vb_count_shift);

	_skr_draw_item(&item, &db);
}

///////////////////////////////////////////////////////////////////////////////

// Fullscreen-triangle blit: the material's vertex stage generates the
// triangle from vertex indices, so there's no mesh. Layered targets get one
// pass per layer with the layer's index as sk_view_index.
void skr_renderer_blit(skr_material_t* material, skr_tex_t* to, skr_recti_t bounds_px) {
	if (material == NULL || to == NULL || to->texture == NULL) return;
	if (!skr_material_is_valid(material) || material->pipeline_material_idx < 0) return;
	if (_pass.active) { skr_log(skr_log_warning, "skr_renderer_blit inside an active pass"); return; }

	_skr_cmd_submit(); // uploads recorded so far land first

	_skr_draw_buffers_t db = {
		.system_offset = _system_offset,
		.system_size   = _system_size,
	};
	if (material->param_buffer && material->param_buffer_size > 0) {
		db.material_offset = _skr_bump_write(&_bump_uniform, WGPUBufferUsage_Uniform, material->param_buffer, material->param_buffer_size);
		db.material_size   = material->param_buffer_size;
	}

	// Cache like the draw path — but not for materials reading pass inputs,
	// whose views change per stage/layer
	const sksc_shader_meta_t* meta       = &material->key.shader->meta;
	bool                      has_inputs = false;
	for (uint32_t r = 0; r < meta->resource_count; r++)
		if (meta->resources[r].bind.register_type == skr_register_input_attachment) { has_inputs = true; break; }

	bool          owned;
	WGPUBindGroup bind_group = _skr_bind_group_get(material->pipeline_material_idx, material->bind_start, material->bind_count, !has_inputs, &owned);
	if (bind_group == NULL) return;

	uint32_t dyn_offsets[3];
	uint32_t dyn_count = _skr_dynamic_offsets(meta, (uint8_t)(skr_stage_vertex | skr_stage_pixel), &db, dyn_offsets);

	if (bounds_px.w <= 0) bounds_px = (skr_recti_t){ 0, 0, to->size.x, to->size.y };

	skr_pipeline_pass_key_t pass_key = {
		.color_format = _skr_tex_fmt_to_wgpu(to->format),
		.depth_format = WGPUTextureFormat_Undefined,
		.sample_count = to->samples,
	};

	for (uint32_t layer = 0; layer < to->layer_count; layer++) {
		pass_key.view_index = layer;
		WGPURenderPipeline pipeline = _skr_pipeline_get(material->pipeline_material_idx, &pass_key, -1);
		if (pipeline == NULL) break;

		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_skr_wgpu.device, NULL);
		_skr_fullscreen_pass(encoder, _skr_tex_layer_view(to, layer), WGPULoadOp_Load, &bounds_px, pipeline, bind_group, dyn_offsets, dyn_count);
		WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
		wgpuCommandEncoderRelease(encoder);
		wgpuQueueSubmit(_skr_wgpu.queue, 1, &cmd);
		wgpuCommandBufferRelease(cmd);
	}
	if (owned) wgpuBindGroupRelease(bind_group);
}

///////////////////////////////////////////////////////////////////////////////
// Transient texture pool — plain pooled textures for postfx intermediates
// (the Vulkan backend's lazy/tile-memory transients lower to this on WebGPU)

typedef struct _skr_transient_t {
	skr_tex_t tex;
	bool      in_use;
	uint32_t  last_used_frame;
} _skr_transient_t;

#define _SKR_TRANSIENT_MAX        16
#define _SKR_TRANSIENT_IDLE_EVICT 120

static _skr_transient_t _transients[_SKR_TRANSIENT_MAX];
static uint32_t         _frame_index;

static skr_tex_t* _skr_transient_acquire(skr_tex_fmt_ format, int32_t width, int32_t height, int32_t layers) {
	for (uint32_t i = 0; i < _SKR_TRANSIENT_MAX; i++) {
		_skr_transient_t* t = &_transients[i];
		if (t->in_use || t->tex.texture == NULL) continue;
		if (t->tex.format != format || t->tex.size.x != width || t->tex.size.y != height || (int32_t)t->tex.layer_count != layers) continue;
		t->in_use          = true;
		t->last_used_frame = _frame_index;
		return &t->tex;
	}
	for (uint32_t i = 0; i < _SKR_TRANSIENT_MAX; i++) {
		_skr_transient_t* t = &_transients[i];
		if (t->in_use || t->tex.texture != NULL) continue;
		skr_tex_flags_ flags = skr_tex_flags_writeable | skr_tex_flags_readable | (layers > 1 ? skr_tex_flags_array : skr_tex_flags_none);
		skr_tex_sampler_t sampler = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };
		if (skr_tex_create(format, flags, sampler, (skr_vec3i_t){ width, height, layers }, 1, 1, NULL, &t->tex) != skr_err_success)
			return NULL;
		skr_tex_set_name(&t->tex, "skr_transient");
		t->in_use          = true;
		t->last_used_frame = _frame_index;
		return &t->tex;
	}
	skr_log(skr_log_critical, "Transient texture pool exhausted (%d entries)", _SKR_TRANSIENT_MAX);
	return NULL;
}

// Frame boundary: transients never span frames, so every entry returns to
// the pool here — pass submission doesn't manage their lifetimes at all.
// (Content safety: submitted command buffers hold their own texture refs.)
// Entries idle long enough release their GPU memory.
void _skr_transient_tick(void) {
	_frame_index++;
	for (uint32_t i = 0; i < _SKR_TRANSIENT_MAX; i++) {
		_skr_transient_t* t = &_transients[i];
		t->in_use = false;
		if (t->tex.texture != NULL && _frame_index - t->last_used_frame > _SKR_TRANSIENT_IDLE_EVICT)
			skr_tex_destroy(&t->tex);
	}
}

///////////////////////////////////////////////////////////////////////////////
// Deferred pass assembly — the pass struct is assembled by the app, then
// submitted through the immediate pass API above

void skr_pass_add_draw(skr_pass_t* pass, skr_render_list_t* list, const void* system_data, uint32_t system_data_size) {
	if (pass == NULL || pass->draw_count >= SKR_PASS_MAX_DRAWS) return;
	pass->draws[pass->draw_count++] = (skr_pass_draw_t){ .list = list, .system_data = system_data, .system_data_size = system_data_size };
}

void skr_pass_add_resolve(skr_pass_t* pass, skr_material_t* resolve_material) {
	if (pass == NULL) return;
	pass->resolve_material = resolve_material;
}

void skr_pass_add_postfx(skr_pass_t* pass, skr_material_t* postfx_material) {
	if (pass == NULL || pass->postfx_count >= SKR_PASS_MAX_POSTFX) return;
	pass->postfx[pass->postfx_count++] = postfx_material;
}

// Builtin MSAA depth -> r32f resolve material, created lazily on first use
#include "skr_depth_resolve.hlsl.h"

static skr_shader_t   _depth_resolve_shader;
static skr_material_t _depth_resolve_material;
static bool           _depth_resolve_tried;

static skr_material_t* _skr_depth_resolve_material(void) {
	if (!_depth_resolve_tried) {
		_depth_resolve_tried = true;
		if (skr_shader_create(sks_skr_depth_resolve_hlsl, sizeof(sks_skr_depth_resolve_hlsl), &_depth_resolve_shader) == skr_err_success) {
			skr_material_create((skr_material_info_t){
				.shader     = &_depth_resolve_shader,
				.cull       = skr_cull_none,
				.depth_test = skr_compare_always,
				.write_mask = skr_write_rgba,
			}, &_depth_resolve_material);
		} else {
			skr_log(skr_log_warning, "Builtin depth resolve shader failed to load; postfx depth reads under MSAA are disabled");
		}
	}
	return skr_material_is_valid(&_depth_resolve_material) ? &_depth_resolve_material : NULL;
}

// One lowered fullscreen stage: draw `material`'s fullscreen triangle into
// every layer of `target`, with the previous stage's output bound as the
// stage's color input and (optionally) `input_depth` as its depth input.
// Inputs bind per layer so stereo/layered postfx reads its own layer.
// Returns false when the stage couldn't draw, so the caller can pass the
// chain input through instead of leaving the target unwritten.
static bool _skr_lowered_stage(skr_tex_t* target, skr_material_t* material, skr_tex_t* input_color, skr_tex_t* input_depth, const void* system_data, uint32_t system_data_size) {
	if (!skr_material_is_valid(material) || material->pipeline_material_idx < 0) return false;
	bool drew = false;

	_skr_draw_buffers_t db = {0};
	if (material->param_buffer && material->param_buffer_size > 0) {
		db.material_offset = _skr_bump_write(&_bump_uniform, WGPUBufferUsage_Uniform, material->param_buffer, material->param_buffer_size);
		db.material_size   = material->param_buffer_size;
	}
	if (system_data != NULL && system_data_size > 0) {
		db.system_offset = _skr_bump_write(&_bump_uniform, WGPUBufferUsage_Uniform, system_data, system_data_size);
		db.system_size   = system_data_size;
	}

	skr_pipeline_pass_key_t pass_key = {
		.color_format = _skr_tex_fmt_to_wgpu(target->format),
		.depth_format = WGPUTextureFormat_Undefined,
		.sample_count = target->samples,
	};

	uint32_t dyn_offsets[3];
	uint32_t dyn_count = _skr_dynamic_offsets(&material->key.shader->meta, (uint8_t)(skr_stage_vertex | skr_stage_pixel), &db, dyn_offsets);

	for (uint32_t layer = 0; layer < target->layer_count; layer++) {
		uint32_t in_layer = input_color->layer_count > layer ? layer : 0;
		_skr_pass_inputs_set(
			_skr_tex_layer_view(input_color, in_layer),
			input_depth ? _skr_tex_layer_view(input_depth, input_depth->layer_count > layer ? layer : 0) : NULL);

		WGPUBindGroup bind_group = _skr_build_bind_group(material->pipeline_material_idx, _skr_bind_pool_get(material->bind_start), material->bind_count);
		if (bind_group == NULL) break;

		pass_key.view_index = layer;
		WGPURenderPipeline pipeline = _skr_pipeline_get(material->pipeline_material_idx, &pass_key, -1);
		if (pipeline == NULL) { wgpuBindGroupRelease(bind_group); break; }

		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(_skr_wgpu.device, NULL);
		// Clear load op: the fullscreen triangle covers everything anyway
		_skr_fullscreen_pass(encoder, _skr_tex_layer_view(target, layer), WGPULoadOp_Clear, NULL, pipeline, bind_group, dyn_offsets, dyn_count);
		WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
		wgpuCommandEncoderRelease(encoder);
		wgpuQueueSubmit(_skr_wgpu.queue, 1, &cmd);
		wgpuCommandBufferRelease(cmd);
		wgpuBindGroupRelease(bind_group);
		drew = true;
	}
	_skr_pass_inputs_set(NULL, NULL);
	return drew;
}

void skr_pass_submit(skr_pass_t* pass) {
	if (pass == NULL) return;

	uint32_t view_mask   = pass->view_count > 1 ? (1u << pass->view_count) - 1 : 1;
	uint32_t correlation = pass->views_correlated ? view_mask : 0;
	int32_t  layers      = pass->view_count > 0 ? pass->view_count : 1;

	bool            has_resolve_mat = pass->resolve_material && skr_material_is_valid(pass->resolve_material);
	skr_material_t* postfx[SKR_PASS_MAX_POSTFX];
	uint32_t        postfx_count = 0;
	for (uint32_t i = 0; i < pass->postfx_count; i++)
		if (pass->postfx[i] && skr_material_is_valid(pass->postfx[i])) postfx[postfx_count++] = pass->postfx[i];

	// --- Single-stage path ---
	if (postfx_count == 0 && !has_resolve_mat) {
		skr_renderer_begin_pass(pass->color, pass->depth, pass->resolve, pass->clear, pass->clear_color, pass->clear_depth, pass->clear_stencil, view_mask, correlation);
		if (pass->viewport.w > 0 || pass->viewport.h > 0) skr_renderer_set_viewport(pass->viewport);
		if (pass->scissor.w  > 0 || pass->scissor.h  > 0) skr_renderer_set_scissor(pass->scissor);
		for (uint32_t i = 0; i < pass->draw_count; i++)
			skr_renderer_draw(pass->draws[i].list, pass->draws[i].system_data, pass->draws[i].system_data_size);
		skr_renderer_end_pass();
		return;
	}

	// --- Multi-stage path: geometry -> [resolve] -> postfx... as sequential
	// render passes, the previous stage bound as a plain texture ---
	skr_tex_t* color   = pass->color;
	skr_tex_t* depth   = pass->depth;
	skr_tex_t* resolve = pass->resolve;
	bool       use_msaa = resolve && color && color->samples > 1;

	skr_tex_t* final_output = pass->postfx_output;
	if (!final_output) final_output = use_msaa ? resolve : color;
	if (!final_output || !color) { skr_log(skr_log_critical, "skr_pass_submit: no postfx output target"); return; }

	// Depth becomes a postfx input when a postfx shader declares an input
	// attachment named "depth" — same name convention the Vulkan backend uses.
	// Under MSAA, depth resolves to an r32f transient after the geometry pass
	// (WebGPU render passes can't resolve depth attachments themselves).
	skr_tex_t* postfx_depth      = NULL;
	skr_tex_t* depth_resolve_out = NULL; // transient; returns to the pool at the frame boundary
	for (uint32_t p = 0; p < postfx_count && postfx_depth == NULL; p++) {
		const sksc_shader_meta_t* meta = &postfx[p]->key.shader->meta;
		for (uint32_t r = 0; r < meta->resource_count; r++) {
			if (meta->resources[r].bind.register_type == skr_register_input_attachment &&
			    strcmp(meta->resources[r].name, "depth") == 0) { postfx_depth = depth; break; }
		}
	}

	const void* system_data      = pass->draw_count > 0 ? pass->draws[0].system_data      : NULL;
	uint32_t    system_data_size = pass->draw_count > 0 ? pass->draws[0].system_data_size : 0;
	skr_tex_fmt_ chain_fmt = final_output->format;

	// Geometry stage. Without a manual resolve material, MSAA hardware-resolves
	// straight into the first chain input; with one, the MSAA color itself is
	// the resolve stage's (multisampled) input.
	skr_tex_t* geometry_out = NULL; // transient the geometry stage resolved into
	skr_tex_t* cur          = color;
	if (use_msaa && !has_resolve_mat) {
		// A hardware resolve target must match the color attachment's format
		// exactly; the postfx chain converts to chain_fmt on its last write
		geometry_out = _skr_transient_acquire(color->format, color->size.x, color->size.y, layers);
		if (!geometry_out) return;
		cur = geometry_out;
	}
	skr_renderer_begin_pass(color, depth, geometry_out, pass->clear, pass->clear_color, pass->clear_depth, pass->clear_stencil, view_mask, correlation);
	if (pass->viewport.w > 0 || pass->viewport.h > 0) skr_renderer_set_viewport(pass->viewport);
	if (pass->scissor.w  > 0 || pass->scissor.h  > 0) skr_renderer_set_scissor(pass->scissor);
	for (uint32_t i = 0; i < pass->draw_count; i++)
		skr_renderer_draw(pass->draws[i].list, pass->draws[i].system_data, pass->draws[i].system_data_size);
	skr_renderer_end_pass();

	// Multisampled depth resolves to an r32f copy postfx can read
	if (postfx_depth && depth && depth->samples > 1) {
		skr_material_t* resolve_mat = _skr_depth_resolve_material();
		depth_resolve_out   = resolve_mat ? _skr_transient_acquire(skr_tex_fmt_r32f, depth->size.x, depth->size.y, layers) : NULL;
		if (depth_resolve_out && _skr_lowered_stage(depth_resolve_out, resolve_mat, depth, NULL, NULL, 0)) {
			postfx_depth = depth_resolve_out;
		} else {
			static bool warned = false;
			if (!warned) { skr_log(skr_log_warning, "PostFX depth read under MSAA couldn't resolve; the depth-reading stage will be skipped"); warned = true; }
			postfx_depth = NULL;
		}
	}

	// Manual resolve stage reads the multisampled color directly
	if (has_resolve_mat) {
		skr_tex_t* target = postfx_count > 0 ? _skr_transient_acquire(chain_fmt, color->size.x, color->size.y, layers) : final_output;
		if (!target) return;
		_skr_lowered_stage(target, pass->resolve_material, cur, NULL, system_data, system_data_size);
		cur = target;
	}

	// Postfx chain. A stage that can't draw (e.g. its depth read is skipped
	// under MSAA) passes its input through so the frame still resolves.
	for (uint32_t p = 0; p < postfx_count; p++) {
		skr_tex_t* target = p + 1 == postfx_count ? final_output : _skr_transient_acquire(chain_fmt, color->size.x, color->size.y, layers);
		if (!target) break;
		if (target == cur) {
			skr_log(skr_log_critical, "skr_pass_submit: the postfx chain would read and write the same texture; set postfx_output (or use MSAA + resolve)");
			break;
		}
		if (!_skr_lowered_stage(target, postfx[p], cur, postfx_depth, system_data, system_data_size) &&
		    cur->format == target->format && cur->samples == target->samples) {
			skr_tex_copy(cur, target, 0, 0, 0, 0, cur->layer_count < target->layer_count ? cur->layer_count : target->layer_count);
			_skr_cmd_submit(); // later stages read `target` through their own submissions
		}
		cur = target;
	}
	// Transients acquired above (geometry resolve, depth resolve, chain
	// intermediates) all return to the pool at the next frame boundary
}

///////////////////////////////////////////////////////////////////////////////

// Full teardown of this module's state so skr_shutdown -> skr_init cycles
// start clean: releases every GPU handle and resets the lazy-init flags that
// would otherwise skip recreation against the new device.
void _skr_renderer_sys_shutdown(void) {
	// A pass left active is an app bug, but don't leak its encoders
	if (_pass.active) {
		for (uint32_t v = 0; v < _pass.view_count; v++) {
			if (_pass.passes[v])   { wgpuRenderPassEncoderEnd(_pass.passes[v]); wgpuRenderPassEncoderRelease(_pass.passes[v]); }
			if (_pass.encoders[v]) wgpuCommandEncoderRelease(_pass.encoders[v]);
		}
	}
	memset(&_pass, 0, sizeof(_pass));

	if (_bump_uniform.buffer) wgpuBufferRelease(_bump_uniform.buffer);
	if (_bump_storage.buffer) wgpuBufferRelease(_bump_storage.buffer);
	memset(&_bump_uniform, 0, sizeof(_bump_uniform));
	memset(&_bump_storage, 0, sizeof(_bump_storage));
	_system_offset = 0;
	_system_size   = 0;

	memset(_globals, 0, sizeof(_globals));
	_global_count     = 0;
	_pass_input_color = NULL;
	_pass_input_depth = NULL;
	if (_fallback_sampler) { wgpuSamplerRelease(_fallback_sampler); _fallback_sampler = NULL; }

	// Frame timers — pending readback maps just get their buffers released;
	// the callbacks write into ring entries we zero right after
	if (_timer_set)     { wgpuQuerySetDestroy(_timer_set); wgpuQuerySetRelease(_timer_set); _timer_set = NULL; }
	if (_timer_resolve) { wgpuBufferRelease(_timer_resolve); _timer_resolve = NULL; }
	for (uint32_t i = 0; i < _SKR_TIMER_RING; i++) {
		if (_timer_ring[i].readback) wgpuBufferRelease(_timer_ring[i].readback);
		memset(&_timer_ring[i], 0, sizeof(_timer_ring[i]));
	}
	_timer_ring_idx     = 0;
	_timer_pair_next    = 0;
	_timer_init_tried   = false;
	_gpu_time_us        = 0;
	_cpu_time_us        = 0;
	_cpu_frame_start_ns = 0;

	for (uint32_t i = 0; i < _SKR_TRANSIENT_MAX; i++) {
		if (_transients[i].tex.texture) skr_tex_destroy(&_transients[i].tex);
		memset(&_transients[i], 0, sizeof(_transients[i]));
	}
	_frame_index = 0;

	if (skr_material_is_valid(&_depth_resolve_material)) skr_material_destroy(&_depth_resolve_material);
	if (skr_shader_is_valid(&_depth_resolve_shader))     skr_shader_destroy(&_depth_resolve_shader);
	memset(&_depth_resolve_material, 0, sizeof(_depth_resolve_material));
	memset(&_depth_resolve_shader,   0, sizeof(_depth_resolve_shader));
	_depth_resolve_tried = false;
}
