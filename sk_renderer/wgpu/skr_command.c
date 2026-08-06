// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////
// Threads: WebGPU objects are single-threaded on web (and the web build runs
// without pthreads), so thread init is a no-op. Multithreaded recording can
// revisit this on native later if profiling ever demands it.

void skr_thread_init         (void) {}
void skr_thread_shutdown     (void) {}
bool skr_thread_is_initialized(void) { return _skr_wgpu.initialized; }

///////////////////////////////////////////////////////////////////////////////
// Futures: each queue submission grabs a ring slot holding the WGPUFuture of
// its OnSubmittedWorkDone. Generations detect slot reuse — a stale future
// whose slot moved on is, by construction, already complete. The ring is
// shared across threads (worker-thread texture uploads submit too), so slot
// state is guarded by a mutex; Dawn calls inside the lock are cheap queries.

static _skr_mtx_t _slot_mutex;
static bool       _slot_mutex_ready;

static void _skr_slot_lock(void) {
	if (!_slot_mutex_ready) { _skr_mtx_init(&_slot_mutex); _slot_mutex_ready = true; }
	_skr_mtx_lock(&_slot_mutex);
}
static void _skr_slot_unlock(void) { _skr_mtx_unlock(&_slot_mutex); }

static void _skr_on_work_done(WGPUQueueWorkDoneStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)status; (void)message;
	// Web path: mark the slot completed (see _SKR_CB_MODE_ASYNC). Generation
	// guards against a late callback landing on a recycled slot.
	_skr_cmd_slot_t* slot = (_skr_cmd_slot_t*)userdata1;
	if (slot && slot->generation == (uint64_t)(uintptr_t)userdata2)
		slot->completed = true;
}

// Caller holds the slot lock
static _skr_cmd_slot_t* _skr_cmd_slot_alloc(uint64_t* out_generation) {
	// Find a free slot; poll for completions first so long sessions recycle
	for (uint32_t attempt = 0; attempt < _SKR_CMD_SLOTS; attempt++) {
		_skr_cmd_slot_t* slot = &_skr_wgpu.cmd_slots[_skr_wgpu.cmd_slot_next % _SKR_CMD_SLOTS];
		_skr_wgpu.cmd_slot_next++;
		if (slot->in_flight && slot->completed)
			slot->in_flight = false;
#ifndef __EMSCRIPTEN__
		if (slot->in_flight) {
			WGPUFutureWaitInfo info = { .future = slot->future };
			if (wgpuInstanceWaitAny(_skr_wgpu.instance, 1, &info, 0) == WGPUWaitStatus_Success && info.completed)
				slot->in_flight = false;
		}
#endif
		if (!slot->in_flight) {
			slot->generation = ++_skr_wgpu.generation_next;
			slot->completed  = false;
			*out_generation  = slot->generation;
			return slot;
		}
	}
	// All slots busy: reuse the oldest anyway; its old handles read complete
	// via generation mismatch, which errs on the safe side for new waiters.
	skr_log(skr_log_warning, "skr future ring exhausted; recycling oldest slot");
	_skr_cmd_slot_t* slot = &_skr_wgpu.cmd_slots[_skr_wgpu.cmd_slot_next++ % _SKR_CMD_SLOTS];
	slot->generation = ++_skr_wgpu.generation_next;
	*out_generation  = slot->generation;
	return slot;
}

skr_future_t _skr_future_from_wgpu(WGPUFuture future) {
	_skr_slot_lock();
	uint64_t         generation = 0;
	_skr_cmd_slot_t* slot       = _skr_cmd_slot_alloc(&generation);
	slot->future    = future;
	slot->in_flight = true;
	_skr_slot_unlock();
	skr_future_t result = { .slot = slot, .generation = generation };
	return result;
}

skr_future_t skr_future_get(void) {
	// A future for "everything submitted so far": submit any active encoder
	return _skr_cmd_submit();
}

bool skr_future_check(const skr_future_t* future) {
	if (future == NULL || future->slot == NULL) return true;
	_skr_cmd_slot_t* slot = (_skr_cmd_slot_t*)future->slot;

	wgpuInstanceProcessEvents(_skr_wgpu.instance);

	_skr_slot_lock();
	bool done = true;
	if (slot->generation == future->generation) { // else: slot recycled, long done
		done = slot->completed;
#ifndef __EMSCRIPTEN__
		if (!done) {
			WGPUFutureWaitInfo info = { .future = slot->future };
			done = wgpuInstanceWaitAny(_skr_wgpu.instance, 1, &info, 0) == WGPUWaitStatus_Success && info.completed;
		}
#endif
		if (done) slot->in_flight = false;
	}
	_skr_slot_unlock();
	return done;
}

void skr_future_wait(const skr_future_t* future) {
	if (future == NULL || future->slot == NULL) return;
	_skr_cmd_slot_t* slot = (_skr_cmd_slot_t*)future->slot;

	_skr_slot_lock();
	bool     ours   = slot->generation == future->generation;
	WGPUFuture wgpu_future = slot->future;
	_skr_slot_unlock();
	if (!ours) return;

	// Blocking wait — valid on native, a hard error on web by design; web
	// code must poll skr_future_check from the frame loop instead. Waiting
	// happens outside the slot lock so other threads can submit meanwhile.
	WGPUFutureWaitInfo info = { .future = wgpu_future };
	if (wgpuInstanceWaitAny(_skr_wgpu.instance, 1, &info, UINT64_MAX) == WGPUWaitStatus_Success && info.completed) {
		_skr_slot_lock();
		if (slot->generation == future->generation) slot->in_flight = false;
		_skr_slot_unlock();
	}
}

///////////////////////////////////////////////////////////////////////////////
// Command encoder lifecycle. skr_cmd_begin/end bracket work explicitly; other
// code paths use _skr_cmd_get for lazy implicit begin, matching how the
// Vulkan backend hands out its shared command buffer. The encoder is
// thread-local, mirroring the Vulkan backend's per-thread command buffers —
// worker threads (asset loaders generating mips, uploading textures) record
// and submit their own work without touching the main thread's encoder.

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
static WGPUCommandEncoder _thread_encoder;
#else
static _Thread_local WGPUCommandEncoder _thread_encoder;
#endif

WGPUCommandEncoder _skr_cmd_get(void) {
	if (_thread_encoder == NULL)
		_thread_encoder = wgpuDeviceCreateCommandEncoder(_skr_wgpu.device, NULL);
	return _thread_encoder;
}

skr_future_t _skr_cmd_submit(void) {
	skr_future_t result = {0};
	if (_thread_encoder == NULL) return result;

	WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(_thread_encoder, NULL);
	wgpuCommandEncoderRelease(_thread_encoder);
	_thread_encoder = NULL;
	if (cmd == NULL) return result;

	wgpuQueueSubmit(_skr_wgpu.queue, 1, &cmd);
	wgpuCommandBufferRelease(cmd);

	_skr_slot_lock();
	uint64_t         generation = 0;
	_skr_cmd_slot_t* slot       = _skr_cmd_slot_alloc(&generation);
	slot->in_flight = true;
	slot->future = wgpuQueueOnSubmittedWorkDone(_skr_wgpu.queue, (WGPUQueueWorkDoneCallbackInfo){
		.mode      = _SKR_CB_MODE_ASYNC,
		.callback  = _skr_on_work_done,
		.userdata1 = slot,
		.userdata2 = (void*)(uintptr_t)generation });
	_skr_slot_unlock();

	result.slot       = slot;
	result.generation = generation;
	return result;
}

void skr_cmd_begin(void) {
	_skr_cmd_get();
}

skr_future_t skr_cmd_end(void) {
	return _skr_cmd_submit();
}

skr_future_t skr_cmd_flush(void) {
	skr_future_t f = _skr_cmd_submit();
	_skr_cmd_get();
	return f;
}

bool skr_cmd_is_active(void) {
	return _thread_encoder != NULL;
}

///////////////////////////////////////////////////////////////////////////////

// Drops this thread's encoder and resets the slot-ring lock so
// skr_shutdown -> skr_init cycles start clean. Worker threads should have
// submitted their work before shutdown; their thread-local encoders can't be
// reached from here.
void _skr_command_sys_shutdown(void) {
	if (_thread_encoder) { wgpuCommandEncoderRelease(_thread_encoder); _thread_encoder = NULL; }
	if (_slot_mutex_ready) { _skr_mtx_destroy(&_slot_mutex); _slot_mutex_ready = false; }
}
