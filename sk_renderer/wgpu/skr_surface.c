// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////

static skr_present_mode_ _skr_present_mode_from_wgpu(WGPUPresentMode mode) {
	switch (mode) {
	case WGPUPresentMode_Fifo:        return skr_present_mode_fifo;
	case WGPUPresentMode_FifoRelaxed: return skr_present_mode_fifo_relaxed;
	case WGPUPresentMode_Mailbox:     return skr_present_mode_mailbox;
	case WGPUPresentMode_Immediate:   return skr_present_mode_immediate;
	default:                          return skr_present_mode_max;
	}
}

static WGPUPresentMode _skr_present_mode_to_wgpu(skr_present_mode_ mode) {
	switch (mode) {
	case skr_present_mode_fifo_relaxed: return WGPUPresentMode_FifoRelaxed;
	case skr_present_mode_mailbox:      return WGPUPresentMode_Mailbox;
	case skr_present_mode_immediate:    return WGPUPresentMode_Immediate;
	default:                            return WGPUPresentMode_Fifo;
	}
}

skr_err_ skr_surface_create(skr_surface_info_t info, skr_surface_t* out_surface) {
	if (out_surface == NULL) return skr_err_invalid_parameter;
	memset(out_surface, 0, sizeof(*out_surface));
	if (info.native_surface == NULL) return skr_err_invalid_parameter;

	// Takes ownership of the caller's reference (matching the Vulkan
	// backend's VkSurfaceKHR convention); released in skr_surface_destroy
	out_surface->surface = (WGPUSurface)info.native_surface;

	WGPUSurfaceCapabilities caps = {0};
	if (wgpuSurfaceGetCapabilities(out_surface->surface, _skr_wgpu.adapter, &caps) != WGPUStatus_Success || caps.formatCount == 0) {
		// The caller's reference is only consumed on success, matching Vulkan.
		// Releasing here left the caller holding a dangling handle.
		skr_log(skr_log_critical, "Failed to query surface capabilities");
		memset(out_surface, 0, sizeof(*out_surface));
		return skr_err_device_error;
	}

	// Prefer an sRGB format so the display applies the linear->sRGB encode.
	// Native surfaces usually offer one directly; web canvases never do — they
	// configure the base format and take the sRGB variant as a view format.
	out_surface->format      = (uint32_t)caps.formats[0]; // preferred format first
	out_surface->view_format = out_surface->format;
	for (size_t i = 0; i < caps.formatCount; i++) {
		if (caps.formats[i] == WGPUTextureFormat_BGRA8UnormSrgb || caps.formats[i] == WGPUTextureFormat_RGBA8UnormSrgb) {
			out_surface->format      = (uint32_t)caps.formats[i];
			out_surface->view_format = out_surface->format;
			break;
		}
	}
	if      (out_surface->format == WGPUTextureFormat_BGRA8Unorm) out_surface->view_format = WGPUTextureFormat_BGRA8UnormSrgb;
	else if (out_surface->format == WGPUTextureFormat_RGBA8Unorm) out_surface->view_format = WGPUTextureFormat_RGBA8UnormSrgb;

	out_surface->present_mode_mask = 0;
	for (size_t i = 0; i < caps.presentModeCount; i++) {
		skr_present_mode_ mode = _skr_present_mode_from_wgpu(caps.presentModes[i]);
		if (mode != skr_present_mode_max) out_surface->present_mode_mask |= 1u << mode;
	}
	// Fifo is always offered; the Vulkan backend's relaxed default doesn't
	// exist on the web, so Fifo is the default here as well
	out_surface->present_mode         = skr_present_mode_fifo;
	out_surface->present_mode_request = info.present_mode;
	wgpuSurfaceCapabilitiesFreeMembers(caps);

	// A caller that gave no size still needs a valid configuration
	skr_surface_resize(out_surface, (info.size.x > 0 && info.size.y > 0) ? info.size : (skr_vec2i_t){ 1280, 720 });
	return skr_err_success;
}

bool skr_surface_is_valid(const skr_surface_t* surface) {
	return surface != NULL && surface->surface != NULL;
}

void skr_surface_destroy(skr_surface_t* ref_surface) {
	if (ref_surface == NULL || ref_surface->surface == NULL) return;
	if (ref_surface->current.view)    wgpuTextureViewRelease(ref_surface->current.view);
	if (ref_surface->current.texture) wgpuTextureRelease(ref_surface->current.texture);
	wgpuSurfaceUnconfigure(ref_surface->surface);
	wgpuSurfaceRelease(ref_surface->surface);
	_skr_frame_forget_surface(ref_surface);
	memset(ref_surface, 0, sizeof(*ref_surface));
}

///////////////////////////////////////////////////////////////////////////////

void skr_surface_resize(skr_surface_t* ref_surface, skr_vec2i_t size) {
	if (ref_surface == NULL || ref_surface->surface == NULL) return;
	if (size.x <= 0 || size.y <= 0) return;

	// WebGPU never reports its own size, so this is the only source there is
	ref_surface->size = size;

	skr_present_mode_ wanted = ref_surface->present_mode_request;
	if (wanted == skr_present_mode_default || !(ref_surface->present_mode_mask & (1u << wanted)))
		wanted = skr_present_mode_fifo;
	ref_surface->present_mode = wanted;

	WGPUTextureFormat view_format = (WGPUTextureFormat)ref_surface->view_format;
	WGPUSurfaceConfiguration config = {
		.device      = _skr_wgpu.device,
		.format      = (WGPUTextureFormat)ref_surface->format,
		.usage       = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst,
		.width       = (uint32_t)ref_surface->size.x,
		.height      = (uint32_t)ref_surface->size.y,
		.alphaMode   = WGPUCompositeAlphaMode_Auto,
		.presentMode = _skr_present_mode_to_wgpu(wanted),
	};
	if (ref_surface->view_format != ref_surface->format) {
		config.viewFormatCount = 1;
		config.viewFormats     = &view_format;
	}
	wgpuSurfaceConfigure(ref_surface->surface, &config);
	ref_surface->configured = true;
}

///////////////////////////////////////////////////////////////////////////////

skr_acquire_ skr_surface_next_tex(skr_surface_t* ref_surface, skr_vec2i_t size, skr_tex_t** out_tex) {
	if (out_tex) *out_tex = NULL;
	if (ref_surface == NULL || ref_surface->surface == NULL || !ref_surface->configured) return skr_acquire_error;

	// A zero size is a minimized window, with nothing to acquire
	if (size.x <= 0 || size.y <= 0) return skr_acquire_not_ready;

	// WebGPU surfaces never report an extent of their own, so the caller's
	// size is the only resize signal there is.
	if (size.x != ref_surface->size.x || size.y != ref_surface->size.y) return skr_acquire_needs_resize;

	// Release the previous frame's wrapper if present didn't run
	if (ref_surface->current.view)    { wgpuTextureViewRelease(ref_surface->current.view);    ref_surface->current.view    = NULL; }
	if (ref_surface->current.texture) { wgpuTextureRelease(ref_surface->current.texture);     ref_surface->current.texture = NULL; }

	// Where vsync lands: Dawn blocks here until a swapchain image retires
	uint64_t wait_start = _skr_time_now_ns();

	WGPUSurfaceTexture surface_tex = {0};
	wgpuSurfaceGetCurrentTexture(ref_surface->surface, &surface_tex);

	uint64_t wait_end = _skr_time_now_ns();
	_skr_cpu_wait_add(wait_start);

	// Dawn hides its images: a blocking acquire means the present two back (of
	// its assumed three images) is on screen now. The browser never blocks here.
#ifndef __EMSCRIPTEN__
	bool fifo = ref_surface->present_mode == skr_present_mode_fifo || ref_surface->present_mode == skr_present_mode_fifo_relaxed;
	if (fifo && wait_end - wait_start > SKR_PRESENT_ACQUIRE_BLOCKED_NS && ref_surface->ring.count >= 2)
		_skr_present_display(&ref_surface->ring, ref_surface->ring.count - 1, wait_end, skr_timing_source_coarse);
#endif

	switch (surface_tex.status) {
		case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
		case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
			break;
		case WGPUSurfaceGetCurrentTextureStatus_Timeout:
			return skr_acquire_not_ready;
		case WGPUSurfaceGetCurrentTextureStatus_Outdated:
			if (surface_tex.texture) wgpuTextureRelease(surface_tex.texture);
			return skr_acquire_needs_resize;
		case WGPUSurfaceGetCurrentTextureStatus_Lost:
			if (surface_tex.texture) wgpuTextureRelease(surface_tex.texture);
			return skr_acquire_surface_lost;
		default:
			return skr_acquire_error;
	}

	skr_tex_t* tex = &ref_surface->current;
	memset(tex, 0, sizeof(*tex));
	tex->texture     = surface_tex.texture;
	tex->view        = wgpuTextureCreateView(surface_tex.texture, &(WGPUTextureViewDescriptor){
		.format          = (WGPUTextureFormat)ref_surface->view_format,
		.dimension       = WGPUTextureViewDimension_2D,
		.mipLevelCount   = 1,
		.arrayLayerCount = 1,
		.aspect          = WGPUTextureAspect_All,
	});
	tex->size        = (skr_vec3i_t){ (int32_t)wgpuTextureGetWidth(surface_tex.texture), (int32_t)wgpuTextureGetHeight(surface_tex.texture), 1 };
	tex->format      = _skr_tex_fmt_from_wgpu((WGPUTextureFormat)ref_surface->view_format);
	tex->flags       = skr_tex_flags_writeable;
	tex->samples     = 1;
	tex->mip_levels  = 1;
	tex->layer_count = 1;
	tex->is_external = true; // released here, not by skr_tex_destroy

	if (out_tex) *out_tex = tex;
	return skr_acquire_success;
}

skr_acquire_ skr_surface_present(skr_surface_t* ref_surface) {
	if (ref_surface == NULL || ref_surface->surface == NULL) return skr_acquire_error;

	// Anything still recorded must reach the queue before present; when there
	// was something, its completion is the later signal for this slot
	skr_future_t tail = _skr_cmd_submit();
	uint32_t     slot = ref_surface->frame_idx % SKR_MAX_FRAMES_IN_FLIGHT;
	if (tail.slot) ref_surface->frame_future[slot] = tail;

	uint64_t id = _skr_present_begin(&ref_surface->ring, _skr_time_now_ns())->id;
	ref_surface->slot_present_id[slot] = id;
	_skr_frame_note_present(ref_surface, id);

#ifdef __EMSCRIPTEN__
	// The browser presents the canvas implicitly when the frame callback
	// returns to the event loop; wgpuSurfacePresent traps on the web
	WGPUStatus status = WGPUStatus_Success;
#else
	// Present can block for the same reason acquire can, so it's excluded too
	uint64_t   present_start = _skr_time_now_ns();
	WGPUStatus status        = wgpuSurfacePresent(ref_surface->surface);
	_skr_cpu_wait_add(present_start);
#endif

	if (ref_surface->current.view)    { wgpuTextureViewRelease(ref_surface->current.view);    ref_surface->current.view    = NULL; }
	if (ref_surface->current.texture) { wgpuTextureRelease(ref_surface->current.texture);     ref_surface->current.texture = NULL; }
	ref_surface->frame_idx++;

	return status == WGPUStatus_Success ? skr_acquire_success : skr_acquire_error;
}

skr_vec2i_t skr_surface_get_size(const skr_surface_t* surface) {
	skr_vec2i_t zero = {0};
	return surface ? surface->size : zero;
}

skr_present_mode_ skr_surface_get_present_mode(const skr_surface_t* surface) {
	return surface->present_mode;
}

///////////////////////////////////////////////////////////////////////////////

uint64_t skr_surface_get_refresh_ns(const skr_surface_t* surface) {
	return _skr_present_refresh_ns(&surface->ring);
}

uint64_t skr_surface_get_next_id(const skr_surface_t* surface) {
	return surface->ring.count + 1;
}

int32_t skr_surface_present_history(skr_surface_t* ref_surface, skr_present_info_t* out_infos, int32_t max_count) {
	// The acquire signal for a present arrives two presents later, see next_tex
	return _skr_present_drain(&ref_surface->ring, 3, out_infos, max_count);
}

void skr_surface_set_refresh(skr_surface_t* ref_surface, uint64_t refresh_ns) {
	ref_surface->ring.refresh_hint_ns = refresh_ns;
}

void skr_surface_set_vblank(skr_surface_t* ref_surface, uint64_t vblank_ns) {
	_skr_present_vblank(&ref_surface->ring, vblank_ns);
}

bool skr_surface_wait_present(skr_surface_t* ref_surface, uint64_t id, uint64_t timeout_ns) {
	(void)timeout_ns;
	if (id == 0 || id > ref_surface->ring.count) return false;
#ifdef __EMSCRIPTEN__
	// The browser can't block, and paces the canvas itself
	return true;
#else
	// No present wait in WebGPU; the frame that fed the present having
	// finished on the GPU is the closest signal there is
	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		if (ref_surface->slot_present_id[i] != id) continue;
		skr_future_wait(&ref_surface->frame_future[i]);
		return true;
	}
	return true;  // slot recycled: the present retired long ago
#endif
}
