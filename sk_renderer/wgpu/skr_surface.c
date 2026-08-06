// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_surface_create(void* native_surface, skr_surface_t* out_surface) {
	if (out_surface == NULL) return skr_err_invalid_parameter;
	memset(out_surface, 0, sizeof(*out_surface));
	if (native_surface == NULL) return skr_err_invalid_parameter;

	// Takes ownership of the caller's reference (matching the Vulkan
	// backend's VkSurfaceKHR convention); released in skr_surface_destroy
	out_surface->surface = (WGPUSurface)native_surface;

	WGPUSurfaceCapabilities caps = {0};
	if (wgpuSurfaceGetCapabilities(out_surface->surface, _skr_wgpu.adapter, &caps) != WGPUStatus_Success || caps.formatCount == 0) {
		skr_log(skr_log_critical, "Failed to query surface capabilities");
		wgpuSurfaceRelease(out_surface->surface);
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
	wgpuSurfaceCapabilitiesFreeMembers(caps);

	// The app sets .size and calls skr_surface_resize for the real window
	// size; this default just guarantees a valid configuration exists
	out_surface->size = (skr_vec2i_t){ 1280, 720 };
	skr_surface_resize(out_surface);
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
	memset(ref_surface, 0, sizeof(*ref_surface));
}

///////////////////////////////////////////////////////////////////////////////

void skr_surface_resize(skr_surface_t* ref_surface) {
	if (ref_surface == NULL || ref_surface->surface == NULL) return;
	if (ref_surface->size.x <= 0 || ref_surface->size.y <= 0) return;

	WGPUTextureFormat view_format = (WGPUTextureFormat)ref_surface->view_format;
	WGPUSurfaceConfiguration config = {
		.device      = _skr_wgpu.device,
		.format      = (WGPUTextureFormat)ref_surface->format,
		.usage       = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst,
		.width       = (uint32_t)ref_surface->size.x,
		.height      = (uint32_t)ref_surface->size.y,
		.alphaMode   = WGPUCompositeAlphaMode_Auto,
		.presentMode = WGPUPresentMode_Fifo,
	};
	if (ref_surface->view_format != ref_surface->format) {
		config.viewFormatCount = 1;
		config.viewFormats     = &view_format;
	}
	wgpuSurfaceConfigure(ref_surface->surface, &config);
	ref_surface->configured = true;
}

///////////////////////////////////////////////////////////////////////////////

skr_acquire_ skr_surface_next_tex(skr_surface_t* ref_surface, skr_tex_t** out_tex) {
	if (out_tex) *out_tex = NULL;
	if (ref_surface == NULL || ref_surface->surface == NULL || !ref_surface->configured) return skr_acquire_error;

	// Release the previous frame's wrapper if present didn't run
	if (ref_surface->current.view)    { wgpuTextureViewRelease(ref_surface->current.view);    ref_surface->current.view    = NULL; }
	if (ref_surface->current.texture) { wgpuTextureRelease(ref_surface->current.texture);     ref_surface->current.texture = NULL; }

	WGPUSurfaceTexture surface_tex = {0};
	wgpuSurfaceGetCurrentTexture(ref_surface->surface, &surface_tex);
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

	// Anything still recorded must reach the queue before present
	_skr_cmd_submit();

#ifdef __EMSCRIPTEN__
	// The browser presents the canvas implicitly when the frame callback
	// returns to the event loop; wgpuSurfacePresent traps on the web
	WGPUStatus status = WGPUStatus_Success;
#else
	WGPUStatus status = wgpuSurfacePresent(ref_surface->surface);
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
