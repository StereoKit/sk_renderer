// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"
#include "skr_pipeline.h"

_skr_wgpu_state_t _skr_wgpu = {0};

// User-overridable allocators from skr_settings_t (see _skr_shared.h)
static void* (*_skr_malloc_fn) (size_t)         = NULL;
static void* (*_skr_calloc_fn) (size_t, size_t) = NULL;
static void* (*_skr_realloc_fn)(void*, size_t)  = NULL;
static void  (*_skr_free_fn)   (void*)          = NULL;

void* _skr_malloc (size_t size)               { return _skr_malloc_fn  ? _skr_malloc_fn (size)        : malloc (size); }
void* _skr_calloc (size_t count, size_t size) { return _skr_calloc_fn  ? _skr_calloc_fn (count, size) : calloc (count, size); }
void* _skr_realloc(void* ptr, size_t size)    { return _skr_realloc_fn ? _skr_realloc_fn(ptr, size)   : realloc(ptr, size); }
void  _skr_free   (void* ptr)                 { if (_skr_free_fn) _skr_free_fn(ptr); else free(ptr); }

///////////////////////////////////////////////////////////////////////////////

const char* _skr_sv(WGPUStringView sv, char* buf, size_t buf_size) {
	if (sv.data == NULL || sv.length == 0) { buf[0] = '\0'; return buf; }
	size_t len = sv.length < buf_size - 1 ? sv.length : buf_size - 1;
	memcpy(buf, sv.data, len);
	buf[len] = '\0';
	return buf;
}

bool _skr_wait_future(WGPUFuture future) {
	WGPUFutureWaitInfo info = { .future = future };
	return wgpuInstanceWaitAny(_skr_wgpu.instance, 1, &info, UINT64_MAX) == WGPUWaitStatus_Success;
}

///////////////////////////////////////////////////////////////////////////////

static void _skr_on_uncaptured_error(WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)device; (void)userdata1; (void)userdata2;
	char buf[2048];
	skr_log(skr_log_critical, "WebGPU error (%d): %s", (int)type, _skr_sv(message, buf, sizeof(buf)));
}

static void _skr_on_device_lost(WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)device; (void)userdata1; (void)userdata2;
	if (reason == WGPUDeviceLostReason_Destroyed) return; // normal shutdown
	char buf[2048];
	skr_log(skr_log_critical, "WebGPU device lost (%d): %s", (int)reason, _skr_sv(message, buf, sizeof(buf)));
}

static void _skr_on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)userdata2;
	if (status == WGPURequestAdapterStatus_Success) {
		*(WGPUAdapter*)userdata1 = adapter;
	} else {
		char buf[512];
		skr_log(skr_log_critical, "WebGPU adapter request failed: %s", _skr_sv(message, buf, sizeof(buf)));
	}
}

static void _skr_on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)userdata2;
	if (status == WGPURequestDeviceStatus_Success) {
		*(WGPUDevice*)userdata1 = device;
	} else {
		char buf[512];
		skr_log(skr_log_critical, "WebGPU device request failed: %s", _skr_sv(message, buf, sizeof(buf)));
	}
}

///////////////////////////////////////////////////////////////////////////////

bool skr_init(skr_settings_t settings) {
	if (_skr_wgpu.initialized) {
		skr_log(skr_log_warning, "skr_init called while already initialized");
		return true;
	}

	_skr_malloc_fn  = settings.malloc_func;
	_skr_calloc_fn  = settings.calloc_func;
	_skr_realloc_fn = settings.realloc_func;
	_skr_free_fn    = settings.free_func;

	_skr_wgpu.binds = settings.bind_settings
		? *settings.bind_settings
		: (skr_bind_settings_t){ .material_slot = 0, .system_slot = 1, .instance_slot = 2 };

	if (settings.wgpu_device != NULL) {
		// Pre-initialized path: JS (web) or the host app supplies the device.
		// Mandatory on web, where main() can never block to request one.
		if (settings.wgpu_instance == NULL) {
			skr_log(skr_log_critical, "skr_init: wgpu_device requires wgpu_instance to be provided too");
			return false;
		}
		_skr_wgpu.instance = (WGPUInstance)settings.wgpu_instance;
		_skr_wgpu.adapter  = (WGPUAdapter )settings.wgpu_adapter;
		_skr_wgpu.device   = (WGPUDevice  )settings.wgpu_device;
		wgpuInstanceAddRef(_skr_wgpu.instance);
		if (_skr_wgpu.adapter) wgpuAdapterAddRef(_skr_wgpu.adapter);
		wgpuDeviceAddRef(_skr_wgpu.device);
	} else {
		// Native request path — the single sanctioned blocking branch, used
		// only for development/debugging against native WebGPU.
		WGPUInstanceFeatureName instance_features[] = { WGPUInstanceFeatureName_TimedWaitAny };
		WGPUInstanceDescriptor  instance_desc = {
			.requiredFeatureCount = 1,
			.requiredFeatures     = instance_features,
		};
		_skr_wgpu.instance = wgpuCreateInstance(&instance_desc);
		if (_skr_wgpu.instance == NULL) {
			skr_log(skr_log_critical, "wgpuCreateInstance failed");
			return false;
		}

		WGPURequestAdapterOptions adapter_opts = {
			.featureLevel    = WGPUFeatureLevel_Core,
			.powerPreference = (settings.gpu_prefer & skr_gpu_integrated)
				? WGPUPowerPreference_LowPower
				: WGPUPowerPreference_HighPerformance,
		};
		WGPUFuture f = wgpuInstanceRequestAdapter(_skr_wgpu.instance, &adapter_opts, (WGPURequestAdapterCallbackInfo){
			.mode      = WGPUCallbackMode_WaitAnyOnly,
			.callback  = _skr_on_adapter,
			.userdata1 = &_skr_wgpu.adapter });
		_skr_wait_future(f);
		if (_skr_wgpu.adapter == NULL) return false;

		// Optional features: take what the adapter offers of the set we use
		WGPUFeatureName wanted[16];
		size_t          wanted_count = 0;
		const WGPUFeatureName maybe[] = {
			WGPUFeatureName_TimestampQuery,
			WGPUFeatureName_TextureCompressionBC,
			WGPUFeatureName_TextureCompressionETC2,
			WGPUFeatureName_TextureCompressionASTC,
			WGPUFeatureName_DepthClipControl,
			WGPUFeatureName_Depth32FloatStencil8,
			WGPUFeatureName_Float32Filterable,
			WGPUFeatureName_RG11B10UfloatRenderable,
			WGPUFeatureName_IndirectFirstInstance,
#ifndef __EMSCRIPTEN__
			// Dawn native is NOT thread-safe without this (a device-level
			// mutex); apps create textures/materials from worker threads. On
			// web everything is single-threaded, and the enum is Dawn-specific.
			WGPUFeatureName_ImplicitDeviceSynchronization,
#endif
		};
		for (size_t i = 0; i < sizeof(maybe)/sizeof(maybe[0]); i++)
			if (wgpuAdapterHasFeature(_skr_wgpu.adapter, maybe[i]))
				wanted[wanted_count++] = maybe[i];

		WGPUDeviceDescriptor device_desc = {
			.label                = { settings.app_name, settings.app_name ? strlen(settings.app_name) : 0 },
			.requiredFeatureCount = wanted_count,
			.requiredFeatures     = wanted,
			.deviceLostCallbackInfo = {
				.mode     = WGPUCallbackMode_AllowProcessEvents,
				.callback = _skr_on_device_lost },
			.uncapturedErrorCallbackInfo = {
				.callback = _skr_on_uncaptured_error },
		};
		f = wgpuAdapterRequestDevice(_skr_wgpu.adapter, &device_desc, (WGPURequestDeviceCallbackInfo){
			.mode      = WGPUCallbackMode_WaitAnyOnly,
			.callback  = _skr_on_device,
			.userdata1 = &_skr_wgpu.device });
		_skr_wait_future(f);
		if (_skr_wgpu.device == NULL) return false;
	}

	_skr_wgpu.queue = wgpuDeviceGetQueue(_skr_wgpu.device);

	_skr_wgpu.limits = (WGPULimits){0};
	wgpuDeviceGetLimits(_skr_wgpu.device, &_skr_wgpu.limits);

	_skr_wgpu.feat_timestamp              = wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_TimestampQuery);
	_skr_wgpu.feat_bc                     = wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_TextureCompressionBC);
	_skr_wgpu.feat_etc2                   = wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_TextureCompressionETC2);
	_skr_wgpu.feat_astc                   = wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_TextureCompressionASTC);
	_skr_wgpu.feat_depth_clip             = wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_DepthClipControl);
	_skr_wgpu.feat_depth32s8              = wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_Depth32FloatStencil8);
	_skr_wgpu.feat_float32_filterable     = wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_Float32Filterable);
	_skr_wgpu.feat_rg11b10_renderable     = wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_RG11B10UfloatRenderable);
	_skr_wgpu.feat_indirect_first_instance= wgpuDeviceHasFeature(_skr_wgpu.device, WGPUFeatureName_IndirectFirstInstance);

	WGPUAdapterInfo info = {0};
	if (_skr_wgpu.adapter && wgpuAdapterGetInfo(_skr_wgpu.adapter, &info) == WGPUStatus_Success) {
		char dev[256], desc[256];
		skr_log(skr_log_info, "WebGPU device: %s (%s)", _skr_sv(info.device, dev, sizeof(dev)), _skr_sv(info.description, desc, sizeof(desc)));
		wgpuAdapterInfoFreeMembers(info);
	}
	skr_log(skr_log_info, "WebGPU features: timestamps %s, BC %s, ETC2 %s, ASTC %s, depth-clip %s, d32s8 %s, f32-filter %s, rg11b10-rt %s",
		_skr_wgpu.feat_timestamp ? "yes" : "no",  _skr_wgpu.feat_bc        ? "yes" : "no",
		_skr_wgpu.feat_etc2      ? "yes" : "no",  _skr_wgpu.feat_astc      ? "yes" : "no",
		_skr_wgpu.feat_depth_clip? "yes" : "no",  _skr_wgpu.feat_depth32s8 ? "yes" : "no",
		_skr_wgpu.feat_float32_filterable ? "yes" : "no", _skr_wgpu.feat_rg11b10_renderable ? "yes" : "no");

	// Set up registry/pool locks while still single-threaded
	_skr_pipeline_init();
	_skr_material_sys_init();

	_skr_wgpu.initialized = true;
	return true;
}

///////////////////////////////////////////////////////////////////////////////

void skr_shutdown(void) {
	if (!_skr_wgpu.initialized) return;

	// Drain before destruction; on web page teardown does it, since we can't block
#ifndef __EMSCRIPTEN__
	skr_future_t pending = _skr_cmd_submit();
	skr_future_wait(&pending);
#else
	_skr_cmd_submit();
#endif

	// Subsystems before the device: internal materials unregister from the
	// pipeline registry, so the registry tears down after them
	_skr_renderer_sys_shutdown();
	_skr_texture_sys_shutdown();
	_skr_material_sys_shutdown();
	_skr_pipeline_shutdown();
	_skr_command_sys_shutdown();

	if (_skr_wgpu.queue)    wgpuQueueRelease   (_skr_wgpu.queue);
	if (_skr_wgpu.device)   wgpuDeviceRelease  (_skr_wgpu.device);
	if (_skr_wgpu.adapter)  wgpuAdapterRelease (_skr_wgpu.adapter);
	if (_skr_wgpu.instance) wgpuInstanceRelease(_skr_wgpu.instance);
	memset(&_skr_wgpu, 0, sizeof(_skr_wgpu));
}

///////////////////////////////////////////////////////////////////////////////

bool skr_is_capable(skr_capability_ capability) {
	switch (capability) {
		case skr_capability_presentation: return true;
		// Vulkan-specific interop concepts, never available here
		default:                          return false;
	}
}

///////////////////////////////////////////////////////////////////////////////

int32_t skr_get_max_msaa_samples(void) {
	return 4; // WebGPU guarantees sample counts 1 and 4, and only those
}

///////////////////////////////////////////////////////////////////////////////

WGPUInstance skr_get_wgpu_instance(void) { return _skr_wgpu.instance; }
WGPUAdapter  skr_get_wgpu_adapter (void) { return _skr_wgpu.adapter; }
WGPUDevice   skr_get_wgpu_device  (void) { return _skr_wgpu.device; }
WGPUQueue    skr_get_wgpu_queue   (void) { return _skr_wgpu.queue; }
