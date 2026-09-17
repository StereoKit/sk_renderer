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

static VkPresentModeKHR _skr_present_mode_to_vk(skr_present_mode_ mode) {
	switch (mode) {
	case skr_present_mode_fifo_relaxed: return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
	case skr_present_mode_mailbox:      return VK_PRESENT_MODE_MAILBOX_KHR;
	case skr_present_mode_immediate:    return VK_PRESENT_MODE_IMMEDIATE_KHR;
	default:                            return VK_PRESENT_MODE_FIFO_KHR;
	}
}

static skr_present_mode_ _skr_present_mode_from_vk(VkPresentModeKHR mode) {
	switch (mode) {
	case VK_PRESENT_MODE_FIFO_KHR:         return skr_present_mode_fifo;
	case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return skr_present_mode_fifo_relaxed;
	case VK_PRESENT_MODE_MAILBOX_KHR:      return skr_present_mode_mailbox;
	case VK_PRESENT_MODE_IMMEDIATE_KHR:    return skr_present_mode_immediate;
	default:                               return skr_present_mode_max;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Presentation timing

// A present timing report can trail its present by several frames
#define SKR_TIMING_SETTLE_PRESENTS       8
// A reported stamp further than this from its present is not a time at all;
// seen on presents mailbox dropped
#define SKR_TIMING_STAMP_SANITY_NS       1000000000ull
// refreshDuration reads 0 until feedback has arrived, VRR can change it, and
// the clock calibration drifts
#define SKR_TIMING_REFRESH_REREAD_FRAMES 64

// Timing extensions each declare per-surface support, and a swapchain may only
// ask for the ones its surface confirmed
static void _skr_surface_query_present_caps(const skr_surface_t* surface, bool* out_timing, uint32_t* out_timing_stages, bool* out_present_id2) {
	*out_timing = false; *out_timing_stages = 0; *out_present_id2 = false;
	if (vkGetPhysicalDeviceSurfaceCapabilities2KHR == NULL) return;
	if (!_skr_vk.has_present_timing && !_skr_vk.has_present_wait2) return;

	VkPresentTimingSurfaceCapabilitiesEXT timing_caps = { .sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT };
	VkSurfaceCapabilitiesPresentId2KHR    id2_caps    = { .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR };
	VkSurfaceCapabilitiesPresentWait2KHR  wait2_caps  = { .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_WAIT_2_KHR };
	VkSurfaceCapabilities2KHR             caps2       = { .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR };
	if (_skr_vk.has_present_timing) { timing_caps.pNext = caps2.pNext; caps2.pNext = &timing_caps; }
	if (_skr_vk.has_present_wait2)  { id2_caps.pNext   = caps2.pNext; caps2.pNext = &id2_caps;
	                                  wait2_caps.pNext = caps2.pNext; caps2.pNext = &wait2_caps; }
	VkResult vr = vkGetPhysicalDeviceSurfaceCapabilities2KHR(_skr_vk.physical_device, &(VkPhysicalDeviceSurfaceInfo2KHR){
		.sType   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
		.surface = surface->surface,
	}, &caps2);
	if (vr != VK_SUCCESS) return;

	*out_timing        = _skr_vk.has_present_timing && timing_caps.presentTimingSupported;
	*out_timing_stages = timing_caps.presentStageQueries;
	*out_present_id2   = _skr_vk.has_present_wait2 && id2_caps.presentId2Supported && wait2_caps.presentWait2Supported;
}

static void _skr_surface_timing_read_refresh(skr_surface_t* ref_surface) {
	if (ref_surface->timing) {
		VkSwapchainTimingPropertiesEXT props   = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT };
		uint64_t                       counter = 0;
		if (vkGetSwapchainTimingPropertiesEXT(_skr_vk.device, ref_surface->swapchain, &props, &counter) == VK_SUCCESS)
			ref_surface->ring.refresh_reported_ns = props.refreshDuration;
	} else if (_skr_vk.has_display_timing_google) {
		VkRefreshCycleDurationGOOGLE cycle = {0};
		if (vkGetRefreshCycleDurationGOOGLE(_skr_vk.device, ref_surface->swapchain, &cycle) == VK_SUCCESS)
			ref_surface->ring.refresh_reported_ns = cycle.refreshDuration;
	}
}

static void _skr_surface_timing_calibrate(skr_surface_t* ref_surface);

static void _skr_surface_timing_setup(skr_surface_t* ref_surface, bool timing, uint32_t stages) {
	ref_surface->timing_stages           = stages & (VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT | VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT
	                                               | VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT | VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT);
	// A surface with no queryable stage has nothing to report; the acquire tiers do better
	ref_surface->timing                  = timing && ref_surface->timing_stages != 0 && vkSetSwapchainPresentTimingQueueSizeEXT != NULL;
	ref_surface->timing_domain           = 0;
	ref_surface->timing_domain_id        = 0;
	ref_surface->timing_frames           = 0;
	memset(ref_surface->timing_stage_offset_ns, 0, sizeof(ref_surface->timing_stage_offset_ns));
	ref_surface->ring.refresh_reported_ns = 0;
	_skr_surface_timing_read_refresh(ref_surface);
	if (!ref_surface->timing) return;

	vkSetSwapchainPresentTimingQueueSizeEXT(_skr_vk.device, ref_surface->swapchain, SKR_PRESENT_RING);

	// The host clock when the driver offers it directly, else a swapchain
	// local domain that gets calibrated against it each frame
	VkSwapchainTimeDomainPropertiesEXT domains = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT };
	uint64_t        counter = 0;
	VkTimeDomainKHR types[8];
	uint64_t        ids  [8];
	vkGetSwapchainTimeDomainPropertiesEXT(_skr_vk.device, ref_surface->swapchain, &domains, &counter);
	if (domains.timeDomainCount > 8) domains.timeDomainCount = 8;
	domains.pTimeDomains   = types;
	domains.pTimeDomainIds = ids;
	vkGetSwapchainTimeDomainPropertiesEXT(_skr_vk.device, ref_surface->swapchain, &domains, &counter);
	int32_t pick = -1;
	for (uint32_t i = 0; i < domains.timeDomainCount && pick < 0; i++) if (types[i] == _skr_vk.host_time_domain)                 pick = (int32_t)i;
	for (uint32_t i = 0; i < domains.timeDomainCount && pick < 0; i++) if (types[i] == VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT)         pick = (int32_t)i;
	for (uint32_t i = 0; i < domains.timeDomainCount && pick < 0; i++) if (types[i] == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT)     pick = (int32_t)i;
	if (pick < 0 && domains.timeDomainCount > 0) pick = 0;
	if (pick >= 0) {
		ref_surface->timing_domain    = types[pick];
		ref_surface->timing_domain_id = ids  [pick];
	}
	_skr_surface_timing_calibrate(ref_surface);
}

static int32_t _skr_stage_index(VkPresentStageFlagsEXT stage) {
	int32_t index = 0;
	while (stage > 1) { stage >>= 1; index++; }
	return index < 4 ? index : 3;
}

static uint64_t _skr_surface_domain_to_host(const skr_surface_t* surface, VkPresentStageFlagsEXT stage, uint64_t stamp) {
	if (surface->timing_domain == _skr_vk.host_time_domain) return _skr_time_from_host_domain(stamp);
	return (uint64_t)((int64_t)stamp + surface->timing_stage_offset_ns[_skr_stage_index(stage)]);
}

// One calibration per clock: a swapchain-local domain is a single clock,
// a stage-local domain is one clock per stage the surface reports
static void _skr_surface_timing_calibrate(skr_surface_t* ref_surface) {
	if (ref_surface->timing_domain == _skr_vk.host_time_domain) return;
	PFN_vkGetCalibratedTimestampsKHR get = _skr_vk.get_calibrated_timestamps;
	if (get == NULL) return;

	bool     per_stage = ref_surface->timing_domain == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT;
	uint32_t stages    = per_stage ? ref_surface->timing_stages : 1;
	for (uint32_t stage = 1; stage <= 8; stage <<= 1) {
		if (!(stages & stage)) continue;
		VkSwapchainCalibratedTimestampInfoEXT swapchain_info = {
			.sType        = VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT,
			.swapchain    = ref_surface->swapchain,
			.presentStage = per_stage ? stage : 0,
			.timeDomainId = ref_surface->timing_domain_id,
		};
		VkCalibratedTimestampInfoKHR infos[2] = {
			{ .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .pNext = &swapchain_info, .timeDomain = ref_surface->timing_domain },
			{ .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = _skr_vk.host_time_domain },
		};
		uint64_t stamps[2], deviation;
		if (get(_skr_vk.device, 2, infos, stamps, &deviation) != VK_SUCCESS) continue;
		int64_t offset = (int64_t)_skr_time_from_host_domain(stamps[1]) - (int64_t)stamps[0];
		if (per_stage) ref_surface->timing_stage_offset_ns[_skr_stage_index(stage)] = offset;
		else for (int32_t i = 0; i < 4; i++) ref_surface->timing_stage_offset_ns[i] = offset;
	}
}

static void _skr_surface_poll_timing(skr_surface_t* ref_surface) {
	if (!ref_surface->timing) return;
	if (++ref_surface->timing_frames >= SKR_TIMING_REFRESH_REREAD_FRAMES) {
		ref_surface->timing_frames = 0;
		_skr_surface_timing_read_refresh(ref_surface);
		_skr_surface_timing_calibrate   (ref_surface);
	}

	VkPastPresentationTimingInfoEXT       info  = { .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT, .swapchain = ref_surface->swapchain };
	VkPastPresentationTimingPropertiesEXT props = { .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT };
	VkResult vr = vkGetPastPresentationTimingEXT(_skr_vk.device, &info, &props);
	if ((vr != VK_SUCCESS && vr != VK_INCOMPLETE) || props.presentationTimingCount == 0) return;

	VkPresentStageTimeEXT       stages [16][4];
	VkPastPresentationTimingEXT results[16];
	uint32_t count = props.presentationTimingCount < 16 ? props.presentationTimingCount : 16;
	for (uint32_t i = 0; i < count; i++)
		results[i] = (VkPastPresentationTimingEXT){ .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT, .presentStageCount = 4, .pPresentStages = stages[i] };
	props.presentationTimingCount = count;
	props.pPresentationTimings    = results;
	vr = vkGetPastPresentationTimingEXT(_skr_vk.device, &info, &props);
	if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) return;

	for (uint32_t i = 0; i < props.presentationTimingCount; i++) {
		// Without present ids chained the results carry 0 and arrive in order
		uint64_t id = results[i].presentId ? results[i].presentId : ref_surface->timing_seq_id++;
		skr_present_info_t* slot = _skr_present_slot(&ref_surface->ring, id);
		if (slot == NULL) continue;

		uint64_t visible = 0, out = 0;
		for (uint32_t s = 0; s < results[i].presentStageCount; s++) {
			// An uncalibrated local domain is anchored at its first stamp:
			// intervals stay exact, absolute times land near now
			int64_t* offset = &ref_surface->timing_stage_offset_ns[_skr_stage_index(stages[i][s].stage)];
			if (ref_surface->timing_domain != _skr_vk.host_time_domain && *offset == 0)
				*offset = (int64_t)_skr_time_get_ns() - (int64_t)stages[i][s].time;

			uint64_t t = _skr_surface_domain_to_host(ref_surface, stages[i][s].stage, stages[i][s].time);
			uint64_t now = _skr_time_get_ns();
			if (t + SKR_TIMING_STAMP_SANITY_NS < slot->cpu_present_ns || t > now + SKR_TIMING_STAMP_SANITY_NS) continue;
			switch (stages[i][s].stage) {
			case VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT:       slot->gpu_done_ns = t; break;
			case VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT:           slot->queued_ns   = t; break;
			case VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT:      out     = t; break;
			case VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT:  visible = t; break;
			default: break;
			}
		}
		if (visible || out) _skr_present_display(&ref_surface->ring, id, visible ? visible : out, skr_timing_source_reported);
	}
}

static void _skr_surface_poll_google(skr_surface_t* ref_surface) {
	if (!_skr_vk.has_display_timing_google || ref_surface->timing) return;
	if (++ref_surface->timing_frames >= SKR_TIMING_REFRESH_REREAD_FRAMES) {
		ref_surface->timing_frames = 0;
		_skr_surface_timing_read_refresh(ref_surface);
	}
	uint32_t count = 0;
	if (vkGetPastPresentationTimingGOOGLE(_skr_vk.device, ref_surface->swapchain, &count, NULL) != VK_SUCCESS || count == 0) return;
	VkPastPresentationTimingGOOGLE results[16];
	if (count > 16) count = 16;
	VkResult vr = vkGetPastPresentationTimingGOOGLE(_skr_vk.device, ref_surface->swapchain, &count, results);
	if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) return;
	for (uint32_t i = 0; i < count; i++)
		_skr_present_display(&ref_surface->ring, results[i].presentID, results[i].actualPresentTime, skr_timing_source_reported);
}

// The image handed back was replaced on screen by the present after its own:
// at about now if the acquire blocked, some time before if it didn't
static void _skr_surface_note_release(skr_surface_t* ref_surface, uint32_t image, uint64_t now_ns, bool blocked) {
	uint64_t prev = ref_surface->image_present_id[image];
	bool     fifo = ref_surface->present_mode == skr_present_mode_fifo || ref_surface->present_mode == skr_present_mode_fifo_relaxed;
	if (prev == 0 || !fifo) return;
	_skr_present_display(&ref_surface->ring, prev + 1, now_ns, blocked ? skr_timing_source_estimate : skr_timing_source_coarse);
}

///////////////////////////////////////////////////////////////////////////////

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

	// The default prefers FIFO_RELAXED (vsync but tolerant of missed
	// deadlines), falling back to FIFO. MAILBOX doesn't vsync on many Linux
	// compositors, and FIFO cascades missed frames.
	ref_surface->present_mode_mask = 0;
	for (uint32_t i = 0; i < present_mode_count; i++) {
		skr_present_mode_ mode = _skr_present_mode_from_vk(present_modes[i]);
		if (mode != skr_present_mode_max) ref_surface->present_mode_mask |= 1u << mode;
	}
	skr_present_mode_ wanted = ref_surface->present_mode_request;
	if (wanted == skr_present_mode_default || !(ref_surface->present_mode_mask & (1u << wanted)))
		wanted = (ref_surface->present_mode_mask & (1u << skr_present_mode_fifo_relaxed)) ? skr_present_mode_fifo_relaxed : skr_present_mode_fifo;
	bool     mode_changed     = wanted != ref_surface->present_mode;
	uint32_t prev_image_count = ref_surface->image_count;
	ref_surface->present_mode = wanted;
	VkPresentModeKHR present_mode = _skr_present_mode_to_vk(wanted);

	bool     timing        = false;
	bool     present_id2   = false;
	uint32_t timing_stages = 0;
	_skr_surface_query_present_caps(ref_surface, &timing, &timing_stages, &present_id2);
	VkSwapchainCreateFlagsKHR create_flags = 0;
	if (timing)      create_flags |= VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT;
	if (present_id2) create_flags |= VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR | VK_SWAPCHAIN_CREATE_PRESENT_WAIT_2_BIT_KHR;

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
		.flags            = create_flags,
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
		if (ref_surface->images)           _skr_free(ref_surface->images);
		if (ref_surface->image_present_id) _skr_free(ref_surface->image_present_id);
		ref_surface->images           = (skr_tex_t*)_skr_calloc(image_count, sizeof(skr_tex_t));
		ref_surface->image_present_id = (uint64_t*) _skr_calloc(image_count, sizeof(uint64_t));
		ref_surface->image_count      = image_count;
	}

	// New images, and a new results queue: nothing from the old swapchain can
	// be matched to them
	memset(ref_surface->image_present_id, 0, sizeof(uint64_t) * image_count);
	ref_surface->present_id2        = present_id2;
	ref_surface->swapchain_first_id = ref_surface->ring.count + 1;
	ref_surface->timing_seq_id      = ref_surface->swapchain_first_id;
	_skr_surface_timing_setup(ref_surface, timing, timing_stages);

	// Resizes rebuild too, and have nothing new to say
	static const char* mode_names[] = { "default", "fifo", "fifo_relaxed", "mailbox", "immediate" };
	if (mode_changed || image_count != prev_image_count)
		skr_log(skr_log_info, "Swapchain: %u images, %s", image_count, mode_names[ref_surface->present_mode]);

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
		ref_surface->images[i].external_prior_use   = true;  // The presentation engine

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

skr_err_ skr_surface_create(skr_surface_info_t info, skr_surface_t* out_surface) {
	if (!out_surface) return skr_err_invalid_parameter;

	// Zero out immediately
	*out_surface = (skr_surface_t){0};

	// Only consulted when the surface cannot report its own extent, which is
	// Wayland. The swapchain overwrites this with the extent it settled on.
	out_surface->size                 = info.size;
	out_surface->present_mode_request = info.present_mode;

	if (!skr_is_capable(skr_capability_presentation)) {
		skr_log(skr_log_critical, "skr_surface_create: VK_KHR_surface/VK_KHR_swapchain not available (headless?)");
		return skr_err_unsupported;
	}

	VkSurfaceKHR vk_surface = (VkSurfaceKHR)info.native_surface;
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
	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++)
		if (_skr_vk.frame_surface[i] == ref_surface) _skr_vk.frame_surface[i] = NULL;
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
	if (ref_surface->image_present_id) _skr_free(ref_surface->image_present_id);

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

	_skr_surface_poll_timing(ref_surface);
	_skr_surface_poll_google(ref_surface);

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
	uint64_t acquire_start = _skr_time_get_ns();

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

	// Out of date acquired nothing; the semaphore stays unsignaled for a retry
	if (result == VK_ERROR_OUT_OF_DATE_KHR) return skr_acquire_needs_resize;

	// Suboptimal still delivered a usable image, and a real extent change is
	// caught by the capabilities poll above. XWayland reports it every few
	// frames for a buffer modifier copy, which no rebuild ever satisfies.
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		skr_log(skr_log_critical, "Failed to acquire swapchain image: 0x%X", result);
		return skr_acquire_error;
	}

	_skr_surface_note_release(ref_surface, ref_surface->current_image, wait_end, wait_end - acquire_start > SKR_PRESENT_ACQUIRE_BLOCKED_NS);

	*out_tex = &ref_surface->images[ref_surface->current_image];
	return skr_acquire_success;
}

skr_acquire_ skr_surface_present(skr_surface_t* ref_surface) {
	if (!ref_surface) return skr_acquire_error;

	uint64_t id = _skr_present_begin(&ref_surface->ring, _skr_time_get_ns())->id;
	ref_surface->image_present_id[ref_surface->current_image] = id;
	ref_surface->slot_present_id [ref_surface->frame_idx]     = id;
	_skr_frame_note_present(ref_surface, id);

	// pNext chain, built innermost first
	const void* chain = NULL;

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
		fence_info.pNext          = chain;
		chain                     = &fence_info;
	}

	VkPresentIdKHR  id_info  = { .sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,   .swapchainCount = 1, .pPresentIds = &id };
	VkPresentId2KHR id2_info = { .sType = VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR, .swapchainCount = 1, .pPresentIds = &id };
	if      (ref_surface->present_id2)  { id2_info.pNext = chain; chain = &id2_info; }
	else if (_skr_vk.has_present_wait)  { id_info.pNext  = chain; chain = &id_info;  }

	VkPresentTimingInfoEXT timing_info = {
		.sType               = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT,
		.timeDomainId        = ref_surface->timing_domain_id,
		.presentStageQueries = ref_surface->timing_stages,
	};
	VkPresentTimingsInfoEXT timings_info = { .sType = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT, .swapchainCount = 1, .pTimingInfos = &timing_info };
	if (ref_surface->timing) {
		timings_info.pNext = chain;
		chain              = &timings_info;
	}

	VkPresentTimeGOOGLE      google_time  = { .presentID = (uint32_t)id };
	VkPresentTimesInfoGOOGLE google_times = { .sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, .swapchainCount = 1, .pTimes = &google_time };
	if (_skr_vk.has_display_timing_google && !ref_surface->timing) {
		google_times.pNext = chain;
		chain              = &google_times;
	}

	// Just present - all command buffer work happened before frame_end!
	mtx_lock(_skr_vk.present_queue_mutex);
	VkResult result = vkQueuePresentKHR(_skr_vk.present_queue, &(VkPresentInfoKHR){
		.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext              = chain,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores    = &ref_surface->semaphore_submit[ref_surface->current_image],
		.swapchainCount     = 1,
		.pSwapchains        = &ref_surface->swapchain,
		.pImageIndices      = &ref_surface->current_image,
	});
	mtx_unlock(_skr_vk.present_queue_mutex);

	ref_surface->frame_idx = (ref_surface->frame_idx + 1) % SKR_MAX_FRAMES_IN_FLIGHT;

	if (result == VK_ERROR_SURFACE_LOST_KHR)                                    return skr_acquire_surface_lost;
	if (result == VK_ERROR_OUT_OF_DATE_KHR)                                     return skr_acquire_needs_resize;
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) { skr_log(skr_log_critical, "vkQueuePresentKHR failed: 0x%X", result); return skr_acquire_error; }
	return skr_acquire_success;
}

bool skr_surface_is_valid(const skr_surface_t* surface) {
	return surface && surface->surface != VK_NULL_HANDLE;
}

skr_vec2i_t skr_surface_get_size(const skr_surface_t* surface) {
	return surface ? surface->size : (skr_vec2i_t){0, 0};
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
	// Reports trail their present by a few frames; without them a slot's last
	// signal is its successor's image coming back, a full lap of the swapchain
	bool     reported = ref_surface->timing || _skr_vk.has_display_timing_google;
	uint32_t settle   = reported ? SKR_TIMING_SETTLE_PRESENTS : ref_surface->image_count + 1;
	return _skr_present_drain(&ref_surface->ring, settle, out_infos, max_count);
}

void skr_surface_set_refresh(skr_surface_t* ref_surface, uint64_t refresh_ns) {
	ref_surface->ring.refresh_hint_ns = refresh_ns;
}

void skr_surface_set_vblank(skr_surface_t* ref_surface, uint64_t vblank_ns) {
	_skr_present_vblank(&ref_surface->ring, vblank_ns);
}

bool skr_surface_wait_present(skr_surface_t* ref_surface, uint64_t id, uint64_t timeout_ns) {
	if (id == 0 || id > ref_surface->ring.count) return false;
	// Ids from before a rebuild belong to a swapchain that no longer exists,
	// and everything it showed has long since been replaced
	if (id < ref_surface->swapchain_first_id) return true;

	VkResult vr = VK_ERROR_EXTENSION_NOT_PRESENT;
	if (ref_surface->present_id2 && vkWaitForPresent2KHR) {
		vr = vkWaitForPresent2KHR(_skr_vk.device, ref_surface->swapchain, &(VkPresentWait2InfoKHR){
			.sType = VK_STRUCTURE_TYPE_PRESENT_WAIT_2_INFO_KHR, .presentId = id, .timeout = timeout_ns });
	} else if (_skr_vk.has_present_wait && vkWaitForPresentKHR) {
		vr = vkWaitForPresentKHR(_skr_vk.device, ref_surface->swapchain, id, timeout_ns);
	}
	if (vr == VK_SUCCESS || vr == VK_SUBOPTIMAL_KHR) return true;
	if (vr == VK_TIMEOUT)                            return false;

	// No present wait: the closest thing is the frame that fed the present
	// having finished on the GPU, plus its present fence where there is one
	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		if (ref_surface->slot_present_id[i] != id) continue;
		skr_future_wait(&ref_surface->frame_future[i]);
		if (ref_surface->present_fence[i] != VK_NULL_HANDLE)
			return vkWaitForFences(_skr_vk.device, 1, &ref_surface->present_fence[i], VK_TRUE, timeout_ns) == VK_SUCCESS;
		return true;
	}
	return true;  // slot recycled: the present retired long ago
}
