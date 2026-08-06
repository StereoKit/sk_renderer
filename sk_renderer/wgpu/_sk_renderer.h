// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include "../include/sk_renderer.h"
#include "../_skr_shared.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Threading shims: apps may create resources from worker threads on native
// (Dawn's API is thread-safe; our registries coordinate writers with one
// mutex and publish to lock-free readers with release/acquire pointer
// stores), while the web build is single-threaded and compiles both away.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
	typedef int _skr_mtx_t;
	#define _skr_mtx_init(m)    (void)(m)
	#define _skr_mtx_destroy(m) (void)(m)
	#define _skr_mtx_lock(m)    (void)(m)
	#define _skr_mtx_unlock(m)  (void)(m)
	#define _skr_atomic(T)      T
	#define _skr_load_acquire(p)      (*(p))
	#define _skr_store_release(p, v)  (*(p) = (v))
#else
	#include <threads.h>
	#include <stdatomic.h>
	typedef mtx_t _skr_mtx_t;
	#define _skr_mtx_init(m)    mtx_init((m), mtx_plain)
	#define _skr_mtx_destroy(m) mtx_destroy(m)
	#define _skr_mtx_lock(m)    mtx_lock(m)
	#define _skr_mtx_unlock(m)  mtx_unlock(m)
	#define _skr_atomic(T)      _Atomic(T)
	#define _skr_load_acquire(p)      atomic_load_explicit((p), memory_order_acquire)
	#define _skr_store_release(p, v)  atomic_store_explicit((p), (v), memory_order_release)
#endif

///////////////////////////////////////////////////////////////////////////////
// Threading model
//
// Resource create/destroy (materials, textures, shaders, buffers) may happen
// on any thread: registries and pools coordinate writers with a mutex and
// publish pointer-stable storage to lock-free readers via release/acquire
// stores. Frees defer to the next frame boundary so draws recorded this frame
// keep reading valid data.
//
// Draw-side work — pass begin/end, draws, blits, pass submission, pipeline
// lookup (_skr_pipeline_get appends to a slot's pipeline cache), bind-group
// caching, and lazy layer views — is DRAW THREAD ONLY by contract. These
// paths read registries without locks and mutate draw-owned caches without
// synchronization.
///////////////////////////////////////////////////////////////////////////////

// WebGPU's minimum uniform/storage dynamic-offset alignment (both default 256)
#define _SKR_OFFSET_ALIGN 256

// Cached bind groups reference the bump buffers at offset 0 with fixed
// binding sizes; per-draw positions arrive as dynamic offsets. Validation
// needs dynamicOffset + bindingSize <= bufferSize for EVERY offset handed
// out, so the bump buffers always keep this much slack past the last
// allocation — and it is also the fixed window size the instance binding
// declares, capping any single draw's instance data.
#define _SKR_BUMP_SLACK (1024 * 1024)

///////////////////////////////////////////////////////////////////////////////
// Internal state
///////////////////////////////////////////////////////////////////////////////

// Async callback delivery: on the web, the device is imported from JS with no
// parent instance, so its events land in emdawnwebgpu's null-instance bucket
// where wgpuInstanceProcessEvents/WaitAny (which filter by instance) never
// see them. AllowSpontaneous events complete straight from the JS promise
// instead. Native keeps AllowProcessEvents, pumped from frame_begin.
#ifdef __EMSCRIPTEN__
	#define _SKR_CB_MODE_ASYNC WGPUCallbackMode_AllowSpontaneous
#else
	#define _SKR_CB_MODE_ASYNC WGPUCallbackMode_AllowProcessEvents
#endif

// Submission tracking slots behind skr_future_t. A slot holds the WGPUFuture
// of a queue submission's OnSubmittedWorkDone; generation counters detect
// recycled slots so stale skr_future_t handles read as completed. On the web,
// completion arrives via the spontaneous callback (`completed`); on native it
// is polled through wgpuInstanceWaitAny.
#define _SKR_CMD_SLOTS 64
typedef struct _skr_cmd_slot_t {
	WGPUFuture    future;
	uint64_t      generation;
	bool          in_flight;
	volatile bool completed;
} _skr_cmd_slot_t;

typedef struct _skr_wgpu_state_t {
	WGPUInstance instance;
	WGPUAdapter  adapter;
	WGPUDevice   device;
	WGPUQueue    queue;

	WGPULimits   limits;
	bool feat_timestamp;
	bool feat_bc;
	bool feat_etc2;
	bool feat_astc;
	bool feat_depth_clip;         // DepthClipControl -> depth_clamp support
	bool feat_depth32s8;
	bool feat_float32_filterable;
	bool feat_rg11b10_renderable;
	bool feat_indirect_first_instance;

	_skr_cmd_slot_t cmd_slots[_SKR_CMD_SLOTS];
	uint32_t        cmd_slot_next;
	uint64_t        generation_next;

	skr_bind_settings_t binds;    // resolved bind slots (material/system/instance)
	bool                initialized;
} _skr_wgpu_state_t;

extern _skr_wgpu_state_t _skr_wgpu;

///////////////////////////////////////////////////////////////////////////////
// Internal helpers
///////////////////////////////////////////////////////////////////////////////

// WGPUStringView is not null-terminated; copy into buf for printf-style use
const char*  _skr_sv(WGPUStringView sv, char* buf, size_t buf_size);

// Ensure a command encoder exists and return it (implicit skr_cmd_begin)
WGPUCommandEncoder _skr_cmd_get(void);

// Finish + submit the active encoder (no-op future if none active)
skr_future_t _skr_cmd_submit(void);

// Wrap any WGPUFuture (mapAsync etc.) in a pollable skr_future_t slot
skr_future_t _skr_future_from_wgpu(WGPUFuture future);

// Samplers from sk_renderer sampler settings (skr_texture.c). The plain
// version strips comparison state; _ex keeps it.
WGPUSampler  _skr_sampler_create   (skr_tex_sampler_t settings, uint32_t mip_count);
WGPUSampler  _skr_sampler_create_ex(skr_tex_sampler_t settings, uint32_t mip_count);

// Block until a WGPUFuture completes. Native-only convenience for init and
// shutdown paths; browsers can never block, so runtime code polls instead.
bool _skr_wait_future(WGPUFuture future);

// Format conversion (skr_conversions.c)
WGPUTextureFormat _skr_tex_fmt_to_wgpu  (skr_tex_fmt_ format);
skr_tex_fmt_      _skr_tex_fmt_from_wgpu(WGPUTextureFormat format);

// Vertex format for a component (skr_mesh.c); pipeline creation uses this
// when grouping attributes per binding with shader-matched locations
WGPUVertexFormat  _skr_vert_format(const skr_vert_component_t* c);

// Material bind pool (skr_material.c)
skr_material_bind_t* _skr_bind_pool_get(int32_t start); // lock-free; chunked storage never moves
void _skr_pass_inputs_set(WGPUTextureView color, WGPUTextureView depth); // views lowered postfx stages read as subpass inputs
void _skr_transient_tick (void); // advance the transient pool's frame clock + evict idle entries; call at frame begin
const WGPUPassTimestampWrites* _skr_timer_pass_writes(WGPUPassTimestampWrites* ts); // fill ts with a per-pass timestamp pair; NULL = timing off

// Encode one fullscreen-triangle pass into `encoder`: single color target,
// optional viewport+scissor bounds, GPU timer pair when timing is on
// (skr_renderer.c; used by blits, lowered postfx stages, and mipgen)
void _skr_fullscreen_pass(WGPUCommandEncoder encoder, WGPUTextureView target, WGPULoadOp load_op, const skr_recti_t* opt_bounds, WGPURenderPipeline pipeline, WGPUBindGroup bind_group, const uint32_t* dyn_offsets, uint32_t dyn_count);
void _skr_bind_pool_drain   (void); // reclaim ranges freed last frame; call at frame begin
void _skr_material_sys_init (void); // bind-pool writer mutex + default textures; call from skr_init

// Subsystem teardown for skr_shutdown -> skr_init cycles: every GPU handle
// released, every lazy-init flag reset, locks destroyed for re-creation
void _skr_renderer_sys_shutdown(void);
void _skr_material_sys_shutdown(void);
void _skr_texture_sys_shutdown (void);
void _skr_command_sys_shutdown (void);

// Per-draw buffer regions in the frame's bump allocations (skr_renderer.c)
typedef struct _skr_draw_buffers_t {
	uint64_t material_offset; uint32_t material_size;
	uint64_t system_offset;   uint32_t system_size;
	uint64_t instance_offset; uint32_t instance_size;
} _skr_draw_buffers_t;

// Bind group assembly from shader meta + a material-style bind list; used by
// draws, compute dispatch, and mipgen (skr_renderer.c)
WGPUBindGroup _skr_build_bind_group_meta(const sksc_shader_meta_t* meta, WGPUBindGroupLayout layout, const skr_material_bind_t* binds, uint32_t bind_count);

// Dynamic offsets for the reserved material/system/instance slots, ordered
// the way SetBindGroup expects; returns the count (skr_renderer.c)
uint32_t _skr_dynamic_offsets(const sksc_shader_meta_t* meta, uint8_t stage_mask, const _skr_draw_buffers_t* db, uint32_t out_offsets[3]);

// Bind epoch: cached bind groups are valid only while this holds still
// (skr_renderer.c). Bumped on global-bind changes, bump-buffer reallocation,
// and sampler swaps on live textures.
uint64_t _skr_bind_epoch     (void);
void     _skr_bind_epoch_bump(void);

// Per-slice bind group cache (skr_material.c) — dynamic offsets make groups
// frame-stable, so one group per material slice serves every draw
typedef struct _skr_bind_cache_t {
	WGPUBindGroup group;
	uint64_t      epoch; // matches _skr_bind_epoch() when valid; 0 = dirty
} _skr_bind_cache_t;
_skr_bind_cache_t* _skr_bind_cache_slot      (int32_t start); // draw thread; creates storage on demand
void               _skr_bind_cache_invalidate(int32_t start); // any thread; marks stale

// Write into the frame's uniform bump allocation (skr_renderer.c)
WGPUBuffer _skr_bump_uniform_write(const void* data, uint32_t size, uint64_t* out_offset);

// Draw-order sort (skr_render_list.c)
void _skr_render_list_sort(skr_render_list_t* ref_list);

// Lazily-created single-layer render view for layered targets (skr_texture.c)
WGPUTextureView _skr_tex_layer_view(skr_tex_t* tex, uint32_t layer);
