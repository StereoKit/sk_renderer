// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"
#include "skr_conversions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Surface
///////////////////////////////////////////////////////////////////////////////

static VkSurfaceFormatKHR _skr_find_surface_format(const VkSurfaceFormatKHR* formats, uint32_t format_count, const VkFormat* preferred, uint32_t preferred_count) {
	for (uint32_t j = 0; j < preferred_count; j++) {
		for (uint32_t i = 0; i < format_count; i++) {
			if (formats[i].format     == preferred[j] &&
			    formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return formats[i];
			}
		}
	}
	return formats[0];
}

// Helper to create/recreate swapchain and allocate resources
static bool _skr_surface_create_swapchain(VkDevice device, VkPhysicalDevice phys_device, uint32_t graphics_queue_family, skr_surface_t* ref_surface, VkSwapchainKHR old_swapchain) {
	// Get surface capabilities. An unchecked failure here leaves the whole
	// struct uninitialized, and every value below is read from it.
	VkSurfaceCapabilitiesKHR capabilities = {0};
	VkResult caps_result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_device, ref_surface->surface, &capabilities);
	SKR_VK_CHECK_RET(caps_result, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR", false);

	// Get surface formats. The count from the first call is the driver's total,
	// so it has to be clamped to the array before the second call fills it.
	uint32_t            format_count = 0;
	VkSurfaceFormatKHR  formats[64];
	vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device, ref_surface->surface, &format_count, NULL);
	if (format_count > sizeof(formats) / sizeof(formats[0]))
		format_count = sizeof(formats) / sizeof(formats[0]);
	vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device, ref_surface->surface, &format_count, formats);
	if (format_count == 0) {
		skr_log(skr_log_critical, "Surface reports no formats");
		return false;
	}

	// Choose format based on platform preference
	// Android/mobile: prefer RGBA for native GPU ordering
	// Desktop: prefer BGRA for Windows/D3D compositor compatibility
#ifdef __ANDROID__
	VkFormat preferred_formats[] = {
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_FORMAT_B8G8R8A8_SRGB,
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_FORMAT_B8G8R8A8_UNORM,
	};
#else
	VkFormat preferred_formats[] = {
		VK_FORMAT_B8G8R8A8_SRGB,
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_FORMAT_B8G8R8A8_UNORM,
		VK_FORMAT_R8G8B8A8_UNORM,
	};
#endif

	VkSurfaceFormatKHR surface_format = _skr_find_surface_format(formats, format_count, preferred_formats, sizeof(preferred_formats) / sizeof(preferred_formats[0]));

	// Get present modes, clamped to the array as above
	uint32_t         present_mode_count = 0;
	VkPresentModeKHR present_modes[16];
	vkGetPhysicalDeviceSurfacePresentModesKHR(phys_device, ref_surface->surface, &present_mode_count, NULL);
	if (present_mode_count > sizeof(present_modes) / sizeof(present_modes[0]))
		present_mode_count = sizeof(present_modes) / sizeof(present_modes[0]);
	vkGetPhysicalDeviceSurfacePresentModesKHR(phys_device, ref_surface->surface, &present_mode_count, present_modes);

	// Choose present mode: prefer FIFO_RELAXED (vsync but tolerant of missed
	// deadlines), fall back to FIFO. MAILBOX doesn't vsync on many Linux
	// compositors, and FIFO cascades missed frames.
	VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
	for (uint32_t i = 0; i < present_mode_count; i++) {
		if (present_modes[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
			present_mode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
			break;
		}
	}

	// Determine extent. A currentExtent of UINT32_MAX means the surface cannot
	// report a size and the client picks one; Wayland always answers this way,
	// the same as WebGPU. The app pushes the window size into ref_surface->size
	// for exactly this case, so prefer that over any invented default.
	VkExtent2D extent = capabilities.currentExtent;
	if (extent.width == UINT32_MAX) {
		if (ref_surface->size.x > 0 && ref_surface->size.y > 0) {
			extent.width  = (uint32_t)ref_surface->size.x;
			extent.height = (uint32_t)ref_surface->size.y;
		} else {
			extent.width  = 1280;
			extent.height = 720;
		}
		// The chosen size still has to satisfy the surface's own limits
		if (extent.width  < capabilities.minImageExtent.width ) extent.width  = capabilities.minImageExtent.width;
		if (extent.height < capabilities.minImageExtent.height) extent.height = capabilities.minImageExtent.height;
		if (extent.width  > capabilities.maxImageExtent.width ) extent.width  = capabilities.maxImageExtent.width;
		if (extent.height > capabilities.maxImageExtent.height) extent.height = capabilities.maxImageExtent.height;
	}

	// Handle minimized window (0x0 extent)
	if (extent.width == 0 || extent.height == 0) {
		return false;
	}

	// Determine image count based on buffering preference
	uint32_t desired      = (uint32_t)_skr_vk.buffering;
	uint32_t image_count  = capabilities.minImageCount > desired ? capabilities.minImageCount : desired;
	if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount)
		image_count = capabilities.maxImageCount;

	// Create swapchain
	VkSwapchainCreateInfoKHR swapchain_info = {
		.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface          = ref_surface->surface,
		.minImageCount    = image_count,
		.imageFormat      = surface_format.format,
		.imageColorSpace  = surface_format.colorSpace,
		.imageExtent      = extent,
		.imageArrayLayers = 1,
		.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.preTransform     = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode      = present_mode,
		.clipped          = VK_TRUE,
		.oldSwapchain     = old_swapchain,
	};

	VkSwapchainKHR swapchain;
	VkResult vr = vkCreateSwapchainKHR(device, &swapchain_info, NULL, &swapchain);
	SKR_VK_CHECK_RET(vr, "vkCreateSwapchainKHR", false);

	// The caller waited the surface quiet (present fences, or device idle on
	// the fallback path), so the retired old swapchain can go right away.
	if (old_swapchain != VK_NULL_HANDLE) {
		vkDestroySwapchainKHR(device, old_swapchain, NULL);
	}
	ref_surface->swapchain = swapchain;

	// Get swapchain images
	VkImage vk_images[16];
	vkGetSwapchainImagesKHR(device, swapchain, &image_count, NULL);
	vkGetSwapchainImagesKHR(device, swapchain, &image_count, vk_images);

	// Reallocate images array and per-image semaphores if count changed
	if (image_count != ref_surface->image_count) {
		// Destroy old per-image submit semaphores
		if (ref_surface->semaphore_submit) {
			for (uint32_t i = 0; i < ref_surface->image_count; i++) {
				if (ref_surface->semaphore_submit[i]) vkDestroySemaphore(device, ref_surface->semaphore_submit[i], NULL);
			}
			_skr_free(ref_surface->semaphore_submit);
		}

		// Allocate new per-image submit semaphores for new image count
		ref_surface->semaphore_submit = (VkSemaphore*)_skr_calloc(image_count, sizeof(VkSemaphore));

		VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		for (uint32_t i = 0; i < image_count; i++) {
			vkCreateSemaphore(device, &semaphore_info, NULL, &ref_surface->semaphore_submit[i]);
		}

		// Reallocate images array
		if (ref_surface->images) _skr_free(ref_surface->images);
		ref_surface->images      = (skr_tex_t*)_skr_calloc(image_count, sizeof(skr_tex_t));
		ref_surface->image_count = image_count;
	}

	// Update size
	ref_surface->size = (skr_vec2i_t){extent.width, extent.height};

	// Create image views and initialize layout tracking
	for (uint32_t i = 0; i < image_count; i++) {
		// Basic properties
		ref_surface->images[i].image             = vk_images[i];
		ref_surface->images[i].size              = (skr_vec3i_t){extent.width, extent.height, 1};
		ref_surface->images[i].format            = skr_tex_fmt_from_native(surface_format.format);
		ref_surface->images[i].samples           = VK_SAMPLE_COUNT_1_BIT;
		ref_surface->images[i].mip_levels        = 1;
		ref_surface->images[i].layer_count       = 1;
		ref_surface->images[i].aspect_mask       = VK_IMAGE_ASPECT_COLOR_BIT;  // CRITICAL: Must be set!
		ref_surface->images[i].usage             = swapchain_info.imageUsage;
		ref_surface->images[i].framebuffer       = VK_NULL_HANDLE;
		ref_surface->images[i].framebuffer_depth      = VK_NULL_HANDLE;
		ref_surface->images[i].framebuffer_pass       = VK_NULL_HANDLE;
		ref_surface->images[i].framebuffer_depth_pass = VK_NULL_HANDLE;
		ref_surface->images[i].sampler           = VK_NULL_HANDLE;
		ref_surface->images[i].memory            = VK_NULL_HANDLE;  // Swapchain owns memory

		// Initialize layout tracking for swapchain images
		// Swapchain images start in UNDEFINED, render pass will transition them
		ref_surface->images[i].current_layout       = VK_IMAGE_LAYOUT_UNDEFINED;
		ref_surface->images[i].current_queue_family = graphics_queue_family;
		ref_surface->images[i].first_use            = true;
		ref_surface->images[i].is_transient_discard = false;  // Swapchain images are not transient

		VkImageViewCreateInfo view_info = {
			.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image      = vk_images[i],
			.viewType   = VK_IMAGE_VIEW_TYPE_2D,
			.format     = surface_format.format,
			.components = {0},  // Defaults to IDENTITY
			.subresourceRange = {
				.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		};

		vr = vkCreateImageView(device, &view_info, NULL, &ref_surface->images[i].view);
		SKR_VK_CHECK_NRET(vr, "vkCreateImageView");
	}

	return true;
}

skr_err_ skr_surface_create(void* vk_surface_khr, skr_vec2i_t size, skr_surface_t* out_surface) {
	if (!out_surface) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_surface = (skr_surface_t){0};

	// Only consulted when the surface cannot report its own extent, which is
	// Wayland. The swapchain overwrites this with the extent it settled on.
	out_surface->size = size;

	if (!skr_is_capable(skr_capability_presentation)) {
		skr_log(skr_log_critical, "skr_surface_create: VK_KHR_surface/VK_KHR_swapchain not available (headless?)");
		return skr_err_unsupported;
	}

	VkSurfaceKHR vk_surface = (VkSurfaceKHR)vk_surface_khr;
	if (!vk_surface) return skr_err_invalid_parameter;

	// Check present support
	VkBool32 present_support = VK_FALSE;
	vkGetPhysicalDeviceSurfaceSupportKHR(_skr_vk.physical_device, _skr_vk.present_queue_family, vk_surface, &present_support);
	if (!present_support) {
		// The caller created this VkSurfaceKHR and still owns it on failure.
		// Destroying it here left the caller holding a dangling handle, which
		// became a double free once it ran its own cleanup.
		skr_log(skr_log_critical, "Surface doesn't support presentation");
		return skr_err_unsupported;
	}

	out_surface->surface = vk_surface;

	// Create swapchain using helper
	if (!_skr_surface_create_swapchain(_skr_vk.device, _skr_vk.physical_device, _skr_vk.graphics_queue_family, out_surface, VK_NULL_HANDLE)) {
		*out_surface = (skr_surface_t){0};
		return skr_err_device_error;
	}

	// Create per-frame synchronization objects (fences and acquire semaphores)
	VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		vkCreateSemaphore(_skr_vk.device, &semaphore_info, NULL, &out_surface->semaphore_acquire[i]);
	}

	// Present fences, when the driver has them; created signaled so the
	// wait-before-reuse in skr_surface_present passes on the first frames.
	if (_skr_vk.has_present_fence) {
		VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
		for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
			vkCreateFence(_skr_vk.device, &fence_info, NULL, &out_surface->present_fence[i]);
		}
	}

	return skr_err_success;
}

// Blocks until every present this surface has issued is fully retired, which
// is what makes destroying its swapchain or reusing its semaphores legal. The
// fences observe that exactly; without them a device idle is the only option,
// heavy and, for the presentation engine's semaphore use, only approximate.
static void _skr_surface_wait_presents(skr_surface_t* ref_surface) {
	if (!_skr_vk.has_present_fence) {
		_skr_device_wait_idle();
		return;
	}
	VkFence  fences[SKR_MAX_FRAMES_IN_FLIGHT];
	uint32_t count = 0;
	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		if (ref_surface->present_fence[i] != VK_NULL_HANDLE) fences[count++] = ref_surface->present_fence[i];
	}
	if (count > 0) vkWaitForFences(_skr_vk.device, count, fences, VK_TRUE, UINT64_MAX);
}

void skr_surface_destroy(skr_surface_t* ref_surface) {
	if (!ref_surface) return;

	// Callers destroy the native window as soon as this returns, so the
	// swapchain and surface can't go on a command ring to be destroyed later.
	_skr_device_wait_idle();
	_skr_surface_wait_presents(ref_surface);
	skr_destroy_list_t list = _skr_destroy_list_create();

	// Destroy per-frame synchronization objects
	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		_skr_cmd_destroy_semaphore(&list, ref_surface->semaphore_acquire[i]);
		_skr_cmd_destroy_fence    (&list, ref_surface->present_fence   [i]);
	}

	// Destroy per-image synchronization objects
	if (ref_surface->semaphore_submit) {
		for (uint32_t i = 0; i < ref_surface->image_count; i++)
			_skr_cmd_destroy_semaphore(&list, ref_surface->semaphore_submit[i]);
		_skr_free(ref_surface->semaphore_submit);
	}

	// Destroy image views and cached framebuffers
	if (ref_surface->images) {
		for (uint32_t i = 0; i < ref_surface->image_count; i++) {
			_skr_cmd_destroy_framebuffer(&list, ref_surface->images[i].framebuffer);
			_skr_cmd_destroy_framebuffer(&list, ref_surface->images[i].framebuffer_depth);
			_skr_cmd_destroy_image_view (&list, ref_surface->images[i].view);
		}
		_skr_free(ref_surface->images);
	}

	// Executes LIFO, so the surface outlives the swapchain
	_skr_cmd_destroy_surface  (&list, ref_surface->surface  );
	_skr_cmd_destroy_swapchain(&list, ref_surface->swapchain);

	_skr_destroy_list_execute(&list);
	_skr_destroy_list_free   (&list);

	*ref_surface = (skr_surface_t){0};
}

void skr_surface_resize(skr_surface_t* ref_surface, skr_vec2i_t size) {
	if (!ref_surface) return;

	// Only consulted when the surface cannot report its own extent. The
	// swapchain overwrites this with the extent it settled on.
	ref_surface->size = size;

	// Only this surface's in-flight frames reference the views destroyed
	// below, so they are all that needs waiting on. A device-wide idle here
	// turned every step of an interactive resize into a full pipeline drain.
	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++)
		skr_future_wait(&ref_surface->frame_future[i]);

	// The old swapchain's presents must also be retired before it and its
	// semaphores can be recycled; the newest is a frame old, so in practice
	// its fence has long signaled and this does not block.
	_skr_surface_wait_presents(ref_surface);

	// Destroy old image views and framebuffers
	for (uint32_t i = 0; i < ref_surface->image_count; i++) {
		skr_tex_t* tex = &ref_surface->images[i];
		if (tex->framebuffer      ) { vkDestroyFramebuffer(_skr_vk.device, tex->framebuffer,       NULL); tex->framebuffer       = VK_NULL_HANDLE; }
		if (tex->framebuffer_depth) { vkDestroyFramebuffer(_skr_vk.device, tex->framebuffer_depth, NULL); tex->framebuffer_depth = VK_NULL_HANDLE; }
		if (tex->view             ) { vkDestroyImageView  (_skr_vk.device, tex->view,              NULL); tex->view              = VK_NULL_HANDLE; }
	}

	// Recreate swapchain using helper (old swapchain will be destroyed by helper)
	if (!_skr_surface_create_swapchain(_skr_vk.device, _skr_vk.physical_device, _skr_vk.graphics_queue_family, ref_surface, ref_surface->swapchain)) {
		// The views above are already gone, so anything still counting on
		// image_count would be handing out null handles.
		skr_log(skr_log_critical, "skr_surface_resize: failed to rebuild the swapchain");
		ref_surface->image_count = 0;
	}
}

skr_acquire_ skr_surface_next_tex(skr_surface_t* ref_surface, skr_vec2i_t size, skr_tex_t** out_tex) {
	if (!ref_surface || !out_tex) return skr_acquire_error;

	*out_tex = NULL;

	// A zero size is a minimized window, with nothing to acquire
	if (size.x <= 0 || size.y <= 0) return skr_acquire_not_ready;

	// The caller's size is the only resize signal for surfaces that report no
	// extent of their own (Wayland): acquire succeeds forever at the stale
	// size there, while the compositor scales the result.
	if (size.x != ref_surface->size.x || size.y != ref_surface->size.y)
		return skr_acquire_needs_resize;

	// Check if the surface needs to be recreated before touching any per-frame
	// state. Some drivers (e.g. Adreno) do not return VK_SUBOPTIMAL_KHR on
	// dimension mismatch — they silently scale in the compositor instead.
	// Polling capabilities is the only reliable cross-driver way to detect
	// this.
	{
		VkSurfaceCapabilitiesKHR caps = {0};
		if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_skr_vk.physical_device, ref_surface->surface, &caps) == VK_SUCCESS &&
		     caps.currentExtent.width  != UINT32_MAX &&
		    (caps.currentExtent.width  != (uint32_t)ref_surface->size.x ||
		     caps.currentExtent.height != (uint32_t)ref_surface->size.y))
			return skr_acquire_needs_resize;
	}

	// Track wait time for CPU timing (excluded from CPU busy time)
	// Both skr_future_wait and vkAcquireNextImageKHR can block
	uint64_t wait_start = _skr_time_get_ns();

	// Wait on the future from N-frames-ago to ensure this frame slot is available
	skr_future_wait(&ref_surface->frame_future[ref_surface->frame_idx]);

	// Acquire next image using per-frame acquire semaphore
	// Frame fence ensures this semaphore is not in use from previous frames
	VkResult result = vkAcquireNextImageKHR(
		_skr_vk.device, ref_surface->swapchain, UINT64_MAX,
		ref_surface->semaphore_acquire[ref_surface->frame_idx],
		VK_NULL_HANDLE, &ref_surface->current_image
	);

	uint64_t wait_end = _skr_time_get_ns();
	if (_skr_vk.in_frame) {
		_skr_vk.cpu_frame_wait_ns[_skr_vk.flight_idx] += (wait_end - wait_start);
	}

	// Handle surface lost - cannot recover here, caller must recreate surface
	if (result == VK_ERROR_SURFACE_LOST_KHR) {
		skr_log(skr_log_critical, "Surface lost - full surface recreation needed");
		// Advance frame index since we won't call present() for this frame
		//surface->frame_idx = (surface->frame_idx + 1) % SKR_MAX_FRAMES_IN_FLIGHT;
		return skr_acquire_surface_lost;
	}

	// Handle swapchain out-of-date or suboptimal
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		// If VK_SUBOPTIMAL_KHR, the semaphore was signaled even though we won't use the image
		// We need to consume the semaphore with a dummy submit to unsignal it
		if (result == VK_SUBOPTIMAL_KHR) {
			VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

			mtx_lock(_skr_vk.graphics_queue_mutex);
			vkQueueSubmit(_skr_vk.graphics_queue, 1, &(VkSubmitInfo){
				.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.waitSemaphoreCount   = 1,
				.pWaitSemaphores      = &ref_surface->semaphore_acquire[ref_surface->frame_idx],
				.pWaitDstStageMask    = &wait_stage,
				.commandBufferCount   = 0,  // No commands, just consume the semaphore
			}, VK_NULL_HANDLE);

			// Wait for the dummy submit to complete so semaphore is unsignaled
			vkQueueWaitIdle(_skr_vk.graphics_queue);
			mtx_unlock(_skr_vk.graphics_queue_mutex);
		}

		// Don't advance frame index - we can reuse the same (now unsignaled) semaphore
		return skr_acquire_needs_resize;
	}

	// Handle other errors
	if (result != VK_SUCCESS) {
		skr_log(skr_log_critical, "Failed to acquire swapchain image: 0x%X", result);
		return skr_acquire_error;
	}

	*out_tex = &ref_surface->images[ref_surface->current_image];
	return skr_acquire_success;
}

skr_acquire_ skr_surface_present(skr_surface_t* ref_surface) {
	if (!ref_surface) return skr_acquire_error;

	// This slot's fence is from SKR_MAX_FRAMES_IN_FLIGHT presents ago, so the
	// wait is a formality; once attached, the fence pins down exactly when the
	// presentation engine is done with the image and semaphore.
	VkSwapchainPresentFenceInfoEXT fence_info = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT };
	VkFence fence = ref_surface->present_fence[ref_surface->frame_idx];
	if (fence != VK_NULL_HANDLE) {
		vkWaitForFences(_skr_vk.device, 1, &fence, VK_TRUE, UINT64_MAX);
		vkResetFences  (_skr_vk.device, 1, &fence);
		fence_info.swapchainCount = 1;
		fence_info.pFences        = &fence;
	}

	// Just present - all command buffer work happened before frame_end!
	mtx_lock(_skr_vk.present_queue_mutex);
	VkResult result = vkQueuePresentKHR(_skr_vk.present_queue, &(VkPresentInfoKHR){
		.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext              = fence != VK_NULL_HANDLE ? &fence_info : NULL,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores    = &ref_surface->semaphore_submit[ref_surface->current_image],
		.swapchainCount     = 1,
		.pSwapchains        = &ref_surface->swapchain,
		.pImageIndices      = &ref_surface->current_image,
	});
	mtx_unlock(_skr_vk.present_queue_mutex);

	ref_surface->frame_idx = (ref_surface->frame_idx + 1) % SKR_MAX_FRAMES_IN_FLIGHT;

	if (result == VK_ERROR_SURFACE_LOST_KHR)                                    return skr_acquire_surface_lost;
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)      return skr_acquire_needs_resize;
	if (result != VK_SUCCESS)              { skr_log(skr_log_critical, "vkQueuePresentKHR failed: 0x%X", result); return skr_acquire_error; }
	return skr_acquire_success;
}

bool skr_surface_is_valid(const skr_surface_t* surface) {
	return surface && surface->surface != VK_NULL_HANDLE;
}

skr_vec2i_t skr_surface_get_size(const skr_surface_t* surface) {
	return surface ? surface->size : (skr_vec2i_t){0, 0};
}
