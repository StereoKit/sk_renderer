// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"
#include "skr_pipeline.h"
#include "skr_conversions.h"

#define VOLK_IMPLEMENTATION
#include <volk.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Global state
///////////////////////////////////////////////////////////////////////////////

_skr_vk_t _skr_vk;

///////////////////////////////////////////////////////////////////////////////
// Memory allocation wrappers
///////////////////////////////////////////////////////////////////////////////

void* _skr_malloc(size_t size) {
	return _skr_vk.malloc_func(size);
}

void* _skr_calloc(size_t count, size_t size) {
	return _skr_vk.calloc_func(count, size);
}

void* _skr_realloc(void* ptr, size_t size) {
	return _skr_vk.realloc_func(ptr, size);
}

void _skr_free(void* ptr) {
	_skr_vk.free_func(ptr);
}

///////////////////////////////////////////////////////////////////////////////
// Validation layers
///////////////////////////////////////////////////////////////////////////////

static VKAPI_ATTR VkBool32 VKAPI_CALL _skr_vk_debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
	VkDebugUtilsMessageTypeFlagsEXT             type,
	const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
	void*                                       user_data) {

	if (callback_data->messageIdNumber == 0)           return VK_FALSE; // A lot of noise?
	if (callback_data->messageIdNumber == -1744492148) return VK_FALSE; // vkCreateGraphicsPipelines: pCreateInfos[] Inside the fragment shader, it writes to output Location X but there is no VkSubpassDescription::pColorAttachments[X] and this write is unused. Spec information at https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-fragmentoutput
	if (callback_data->messageIdNumber == -937765618 ) return VK_FALSE; // vkCreateGraphicsPipelines: pCreateInfos[].pVertexInputState Vertex attribute at location X not consumed by shader.
	if (callback_data->messageIdNumber == -60244330  ) return VK_FALSE;
	if (callback_data->messageIdNumber ==  533026821 ) return VK_FALSE; // gl_Layer ?
	if (callback_data->messageIdNumber ==  115483881 ) return VK_FALSE; // Geometry shader req, might need attention

	const char *severity_str = severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT ? "VERBOSE" :
							   severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT    ? "INFO"    :
							   severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? "WARNING" :
							   severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT   ? "ERROR"   : "UNKNOWN";

	printf("[Vulkan:%s:%d] %s\n", severity_str, callback_data->messageIdNumber, callback_data->pMessage);

	if (severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
		severity_str = severity_str;
	}
	return VK_FALSE;
}

static bool _skr_vk_create_debug_messenger(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT callback, VkDebugUtilsMessengerEXT* out_messenger) {
	VkDebugUtilsMessengerCreateInfoEXT create_info = {
		.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		.pfnUserCallback = callback,
	};

	VkResult vr = vkCreateDebugUtilsMessengerEXT(instance, &create_info, NULL, out_messenger);
	SKR_VK_CHECK_RET(vr, "vkCreateDebugUtilsMessengerEXT", false);
	return true;
}

///////////////////////////////////////////////////////////////////////////////
// Initialization
///////////////////////////////////////////////////////////////////////////////

bool skr_init(skr_settings_t settings) {
	if (_skr_vk.initialized) {
		skr_log(skr_log_warning, "sk_renderer already initialized");
		return false;
	}

	// Validate memory allocators - either all provided or none provided
	int32_t allocator_count = (settings.malloc_func  != NULL) + (settings.calloc_func  != NULL) +
	                          (settings.realloc_func != NULL) + (settings.free_func    != NULL);
	if (allocator_count != 0 && allocator_count != 4) {
		skr_log(skr_log_critical, "sk_renderer: Memory allocators must be all provided or all NULL");
		return false;
	}

	_skr_vk = (_skr_vk_t){};
	_skr_vk.validation_enabled        = settings.enable_validation;
	_skr_vk.current_renderpass_idx    = -1;
	_skr_vk.main_thread_id            = thrd_current();
	_skr_vk.destroy_list              = _skr_destroy_list_create();

	// Set up memory allocators (use stdlib if none provided)
	_skr_vk.malloc_func  = settings.malloc_func  ? settings.malloc_func  : malloc;
	_skr_vk.calloc_func  = settings.calloc_func  ? settings.calloc_func  : calloc;
	_skr_vk.realloc_func = settings.realloc_func ? settings.realloc_func : realloc;
	_skr_vk.free_func    = settings.free_func    ? settings.free_func    : free;

	_skr_bind_pool_init();
	_skr_sampler_cache_init();

	// Set up bind slot configuration (use defaults if not provided)
	if (settings.bind_settings) {
		_skr_vk.bind_settings = *settings.bind_settings;
	} else {
		_skr_vk.bind_settings = (skr_bind_settings_t){
			.material_slot = 0,
			.system_slot   = 1,
			.instance_slot = 2,
		};
	}

	// Initialize volk
	VkResult vr = volkInitialize();
	SKR_VK_CHECK_RET(vr, volkInitialize, false);

	///////////////////////////////////////////////////////////////////////////
	// Extension definitions
	///////////////////////////////////////////////////////////////////////////

	// Instance extensions
	const char* required_instance_exts[] = {
		VK_KHR_SURFACE_EXTENSION_NAME,
	};
	const char* optional_instance_exts[] = {
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
		"VK_EXT_present_mode_fifo_latest_ready",
	};
	const uint32_t required_instance_ext_count = sizeof(required_instance_exts) / sizeof(required_instance_exts[0]);
	const uint32_t optional_instance_ext_count = sizeof(optional_instance_exts) / sizeof(optional_instance_exts[0]);

	// Device extensions
	const char* required_device_exts[] = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};
	const char* optional_device_exts[] = {
		VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
	};
	const uint32_t required_device_ext_count = sizeof(required_device_exts) / sizeof(required_device_exts[0]);
	const uint32_t optional_device_ext_count = sizeof(optional_device_exts) / sizeof(optional_device_exts[0]);

	// Video decode extensions (all required together for video support)
	const char* video_device_exts[] = {
		"VK_KHR_synchronization2",
		"VK_KHR_video_queue",
		"VK_KHR_video_decode_queue",
		"VK_KHR_video_decode_h264",
	};
	const uint32_t video_device_ext_count = sizeof(video_device_exts) / sizeof(video_device_exts[0]);

	///////////////////////////////////////////////////////////////////////////
	// Instance creation
	///////////////////////////////////////////////////////////////////////////

	VkApplicationInfo app_info = {
		.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName   = settings.app_name ? settings.app_name : "sk_renderer_app",
		.applicationVersion = settings.app_version,
		.pEngineName        = "sk_renderer",
		.engineVersion      = VK_MAKE_VERSION(0, 1, 0),
		.apiVersion         = VK_API_VERSION_1_1,
	};

	// Get available instance extensions
	uint32_t available_inst_ext_count = 0;
	vkEnumerateInstanceExtensionProperties(NULL, &available_inst_ext_count, NULL);
	VkExtensionProperties* available_inst_exts = _skr_malloc(available_inst_ext_count * sizeof(VkExtensionProperties));
	vkEnumerateInstanceExtensionProperties(NULL, &available_inst_ext_count, available_inst_exts);

	// Helper to check if an extension is available
	#define EXT_AVAILABLE(name, available, count) ({ \
		bool _found = false; \
		for (uint32_t _i = 0; _i < (count); _i++) { \
			if (strcmp((name), (available)[_i].extensionName) == 0) { _found = true; break; } \
		} \
		_found; \
	})

	// Build final instance extension list
	const char* instance_exts[64];
	uint32_t    instance_ext_count = 0;

	// Add application-required extensions first
	for (uint32_t i = 0; i < settings.required_extension_count && instance_ext_count < 64; i++) {
		if (EXT_AVAILABLE(settings.required_extensions[i], available_inst_exts, available_inst_ext_count)) {
			instance_exts[instance_ext_count++] = settings.required_extensions[i];
		} else {
			skr_log(skr_log_critical, "Required instance extension '%s' not available", settings.required_extensions[i]);
			_skr_free(available_inst_exts);
			return false;
		}
	}

	// Add sk_renderer required extensions
	for (uint32_t i = 0; i < required_instance_ext_count && instance_ext_count < 64; i++) {
		if (EXT_AVAILABLE(required_instance_exts[i], available_inst_exts, available_inst_ext_count)) {
			instance_exts[instance_ext_count++] = required_instance_exts[i];
		} else {
			skr_log(skr_log_critical, "Required instance extension '%s' not available", required_instance_exts[i]);
			_skr_free(available_inst_exts);
			return false;
		}
	}

	// Add optional extensions if available
	for (uint32_t i = 0; i < optional_instance_ext_count && instance_ext_count < 64; i++) {
		// Skip debug utils if validation not enabled
		if (strcmp(optional_instance_exts[i], VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0 && !_skr_vk.validation_enabled) {
			continue;
		}
		if (EXT_AVAILABLE(optional_instance_exts[i], available_inst_exts, available_inst_ext_count)) {
			instance_exts[instance_ext_count++] = optional_instance_exts[i];
		}
	}

	_skr_free(available_inst_exts);

	// Build list of desired layers
	const char* desired_layers[8];
	uint32_t    desired_layer_count = 0;
	if (_skr_vk.validation_enabled) {
		desired_layers[desired_layer_count++] = "VK_LAYER_KHRONOS_validation";
	}

	// Get available layers
	uint32_t available_layer_count = 0;
	vkEnumerateInstanceLayerProperties(&available_layer_count, NULL);
	VkLayerProperties* available_layers = _skr_malloc(available_layer_count * sizeof(VkLayerProperties));
	vkEnumerateInstanceLayerProperties(&available_layer_count, available_layers);

	// Filter layers to only those available
	const char* layers[8];
	uint32_t    layer_count = 0;
	for (uint32_t i = 0; i < desired_layer_count; i++) {
		bool found = false;
		for (uint32_t j = 0; j < available_layer_count; j++) {
			if (strcmp(desired_layers[i], available_layers[j].layerName) == 0) {
				found = true;
				break;
			}
		}
		if (found) {
			layers[layer_count++] = desired_layers[i];
		} else {
			skr_log(skr_log_warning, "Layer '%s' not available, skipping", desired_layers[i]);
			if (strcmp(desired_layers[i], "VK_LAYER_KHRONOS_validation") == 0) _skr_vk.validation_enabled = false;
		}
	}
	_skr_free(available_layers);

	VkInstanceCreateInfo instance_info = {
		.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo        = &app_info,
		.enabledExtensionCount   = instance_ext_count,
		.ppEnabledExtensionNames = instance_exts,
		.enabledLayerCount       = layer_count,
		.ppEnabledLayerNames     = layers,
	};

	VkResult result = vkCreateInstance(&instance_info, NULL, &_skr_vk.instance);
	if (result != VK_SUCCESS) {
		skr_log(skr_log_critical, "Failed to create Vulkan instance: 0x%X", result);
		skr_log(skr_log_info,     "  Enabled extensions (%u):", instance_ext_count);
		for (uint32_t i = 0; i < instance_ext_count; i++)
			skr_log(skr_log_info, "    - %s", instance_exts[i]);
		if (layer_count > 0) {
			skr_log(skr_log_info, "  Enabled layers (%u):", layer_count);
			for (uint32_t i = 0; i < layer_count; i++) {
				skr_log(skr_log_info, "    - %s", layers[i]);
			}
		}
		skr_log(skr_log_info, "  Tip: If using RenderDoc, ensure it's launched with Vulkan support enabled");
		return false;
	}

	volkLoadInstance(_skr_vk.instance);

	if (_skr_vk.validation_enabled) {
		if (!_skr_vk_create_debug_messenger(_skr_vk.instance, _skr_vk_debug_callback, &_skr_vk.debug_messenger )) {
			skr_log(skr_log_warning, "Failed to create debug messenger");
		} else {
			_skr_cmd_destroy_debug_messenger(&_skr_vk.destroy_list, _skr_vk.debug_messenger);
		}
	}

	// Call device initialization callback if provided (e.g., for OpenXR integration)
	// This allows external systems to query for physical device and device extensions
	// after VkInstance is available but before VkDevice is created.
	skr_device_request_t device_request = {0};
	if (settings.device_init_callback) {
		device_request = settings.device_init_callback(_skr_vk.instance, settings.device_init_user_data);
	}

	// Pick physical device - callback's choice takes precedence over settings
	VkPhysicalDevice requested_physical_device = device_request.physical_device
		? (VkPhysicalDevice)device_request.physical_device
		: (VkPhysicalDevice)settings.physical_device;

	if (requested_physical_device) {
		// Use the device specified by the application (e.g., from OpenXR)
		_skr_vk.physical_device = requested_physical_device;
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(_skr_vk.physical_device, &props);
		skr_log(skr_log_info, "Using application-specified GPU: %s", props.deviceName);
	} else {
		// Enumerate and select GPU based on require/prefer flags
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(_skr_vk.instance, &device_count, NULL);
		if (device_count == 0) {
			skr_log(skr_log_critical, "No Vulkan-compatible GPUs found");
			return false;
		}

		VkPhysicalDevice devices[32];
		vkEnumeratePhysicalDevices(_skr_vk.instance, &device_count, devices);

		// Score each device and find best match
		int32_t best_score = -1;
		int32_t best_idx   = -1;

		for (uint32_t i = 0; i < device_count; i++) {
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(devices[i], &props);

			// Determine device capabilities
			bool is_discrete   = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
			bool is_integrated = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
			bool has_video     = false;

			// Check for video decode support
			uint32_t ext_count = 0;
			vkEnumerateDeviceExtensionProperties(devices[i], NULL, &ext_count, NULL);
			if (ext_count > 0) {
				VkExtensionProperties* exts = _skr_malloc(sizeof(VkExtensionProperties) * ext_count);
				vkEnumerateDeviceExtensionProperties(devices[i], NULL, &ext_count, exts);

				uint32_t video_found = 0;
				for (uint32_t v = 0; v < video_device_ext_count; v++) {
					for (uint32_t e = 0; e < ext_count; e++) {
						if (strcmp(exts[e].extensionName, video_device_exts[v]) == 0) {
							video_found++;
							break;
						}
					}
				}
				_skr_free(exts);
				has_video = (video_found == video_device_ext_count);
			}

			// Check required flags - skip device if any required flag is missing
			skr_gpu_ req = settings.gpu_require;
			if ((req & skr_gpu_discrete)   && !is_discrete)   continue;
			if ((req & skr_gpu_integrated) && !is_integrated) continue;
			if ((req & skr_gpu_video)      && !has_video)     continue;

			// Calculate score based on preferred flags and device type
			int32_t score = 0;
			skr_gpu_ pref = settings.gpu_prefer;

			if (pref == skr_gpu_none) {
				// No preference: default to discrete > integrated
				if (is_discrete)   score += 1000;
				if (is_integrated) score += 100;
			} else {
				// Score based on matching preferred flags
				if ((pref & skr_gpu_discrete)   && is_discrete)   score += 1000;
				if ((pref & skr_gpu_integrated) && is_integrated) score += 1000;
				if ((pref & skr_gpu_video)      && has_video)     score += 500;
			}

			if (score > best_score) {
				best_score = score;
				best_idx   = (int32_t)i;
			}
		}

		if (best_idx < 0) {
			skr_log(skr_log_critical, "No GPU found matching required features (require=0x%X)", settings.gpu_require);
			return false;
		}

		_skr_vk.physical_device = devices[best_idx];
	}

	// Get device properties for timing and logging
	VkPhysicalDeviceProperties device_props;
	vkGetPhysicalDeviceProperties(_skr_vk.physical_device, &device_props);

	// Print selected device if we didn't find discrete GPU
	skr_log(skr_log_info, "Using GPU: %s", device_props.deviceName);

	// Store timestamp period for GPU timing
	_skr_vk.timestamp_period = device_props.limits.timestampPeriod;

	// Find queue families
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(_skr_vk.physical_device, &queue_family_count, NULL);

	VkQueueFamilyProperties queue_families[32];
	vkGetPhysicalDeviceQueueFamilyProperties(_skr_vk.physical_device, &queue_family_count, queue_families);

	// Find graphics queue family
	_skr_vk.graphics_queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < queue_family_count; i++) {
		if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			_skr_vk.graphics_queue_family = i;
			_skr_vk.present_queue_family  = i; // Assume same for now
			break;
		}
	}

	if (_skr_vk.graphics_queue_family == UINT32_MAX) {
		skr_log(skr_log_critical, "Failed to find graphics queue family");
		return false;
	}

	// Find dedicated transfer queue (TRANSFER but not GRAPHICS)
	_skr_vk.transfer_queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < queue_family_count; i++) {
		if ((queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			!(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			_skr_vk.transfer_queue_family = i;
			_skr_vk.has_dedicated_transfer = true;
			break;
		}
	}

	// Fall back to graphics queue for transfers
	if (_skr_vk.transfer_queue_family == UINT32_MAX) {
		_skr_vk.transfer_queue_family = _skr_vk.graphics_queue_family;
		_skr_vk.has_dedicated_transfer = false;
	}

	// Find video decode queue family (requires VK_QUEUE_VIDEO_DECODE_BIT_KHR = 0x20)
	_skr_vk.video_decode_queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < queue_family_count; i++) {
		if (queue_families[i].queueFlags & 0x00000020) {  // VK_QUEUE_VIDEO_DECODE_BIT_KHR
			_skr_vk.video_decode_queue_family = i;
			break;
		}
	}

	// Create queue create infos
	float queue_priority = 1.0f;
	VkDeviceQueueCreateInfo queue_infos[3];
	uint32_t queue_info_count = 0;

	// Always create graphics queue
	queue_infos[queue_info_count++] = (VkDeviceQueueCreateInfo){
		.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = _skr_vk.graphics_queue_family,
		.queueCount       = 1,
		.pQueuePriorities = &queue_priority,
	};

	// Create dedicated transfer queue if available
	if (_skr_vk.has_dedicated_transfer) {
		queue_infos[queue_info_count++] = (VkDeviceQueueCreateInfo){
			.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = _skr_vk.transfer_queue_family,
			.queueCount       = 1,
			.pQueuePriorities = &queue_priority,
		};
	}

	// Create video decode queue if available and different from existing queues
	bool need_video_decode_queue = _skr_vk.video_decode_queue_family != UINT32_MAX &&
	                               _skr_vk.video_decode_queue_family != _skr_vk.graphics_queue_family &&
	                               _skr_vk.video_decode_queue_family != _skr_vk.transfer_queue_family;
	if (need_video_decode_queue) {
		queue_infos[queue_info_count++] = (VkDeviceQueueCreateInfo){
			.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = _skr_vk.video_decode_queue_family,
			.queueCount       = 1,
			.pQueuePriorities = &queue_priority,
		};
	}

	///////////////////////////////////////////////////////////////////////////
	// Device creation
	///////////////////////////////////////////////////////////////////////////

	// Get available device extensions
	uint32_t available_device_ext_count = 0;
	vkEnumerateDeviceExtensionProperties(_skr_vk.physical_device, NULL, &available_device_ext_count, NULL);
	VkExtensionProperties* available_device_exts = _skr_malloc(available_device_ext_count * sizeof(VkExtensionProperties));
	vkEnumerateDeviceExtensionProperties(_skr_vk.physical_device, NULL, &available_device_ext_count, available_device_exts);

	// Build final device extension list
	const char* device_exts[64];
	uint32_t    device_ext_count = 0;

	// Add required device extensions
	for (uint32_t i = 0; i < required_device_ext_count && device_ext_count < 64; i++) {
		if (EXT_AVAILABLE(required_device_exts[i], available_device_exts, available_device_ext_count)) {
			device_exts[device_ext_count++] = required_device_exts[i];
		} else {
			skr_log(skr_log_critical, "Required device extension '%s' not available", required_device_exts[i]);
			_skr_free(available_device_exts);
			return false;
		}
	}

	// Add device extensions from callback (e.g., from OpenXR)
	for (uint32_t i = 0; i < device_request.required_device_extension_count && device_ext_count < 64; i++) {
		if (EXT_AVAILABLE(device_request.required_device_extensions[i], available_device_exts, available_device_ext_count)) {
			device_exts[device_ext_count++] = device_request.required_device_extensions[i];
		} else {
			skr_log(skr_log_critical, "Required device extension '%s' not available", device_request.required_device_extensions[i]);
			_skr_free(available_device_exts);
			return false;
		}
	}

	// Add optional device extensions if available
	_skr_vk.has_push_descriptors = false;
	bool has_viewport_layer      = false;
	for (uint32_t i = 0; i < optional_device_ext_count && device_ext_count < 64; i++) {
		if (EXT_AVAILABLE(optional_device_exts[i], available_device_exts, available_device_ext_count)) {
			device_exts[device_ext_count++] = optional_device_exts[i];
			if (strcmp(optional_device_exts[i], VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME            ) == 0) _skr_vk.has_push_descriptors = true;
			if (strcmp(optional_device_exts[i], VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME) == 0) has_viewport_layer           = true;
		}
	}

	_skr_free(available_device_exts);

	// Log optional extension status
	if (!_skr_vk.has_push_descriptors) {
		skr_log(skr_log_info, "Device extension '%s' not available, using descriptor set fallback", VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
	}
	if (!has_viewport_layer) {
		skr_log(skr_log_warning, "Device extension '%s' not available, multi-view rendering will not work", VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME);
	}

	// Query available device features
	VkPhysicalDeviceFeatures available_features;
	vkGetPhysicalDeviceFeatures(_skr_vk.physical_device, &available_features);

	// Track feature availability
	_skr_vk.has_depth_clamp = available_features.depthClamp;

	// Enable features we need (only if available)
	VkPhysicalDeviceFeatures device_features = {
		.samplerAnisotropy = available_features.samplerAnisotropy,
		.sampleRateShading = VK_FALSE, // Not using sample shading yet
		.fillModeNonSolid  = VK_FALSE, // Not using wireframe
		.depthClamp        = available_features.depthClamp,
	};

	VkDeviceCreateInfo device_info = {
		.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext                   = NULL,
		.queueCreateInfoCount    = queue_info_count,
		.pQueueCreateInfos       = queue_infos,
		.enabledExtensionCount   = device_ext_count,
		.ppEnabledExtensionNames = device_exts,
		.pEnabledFeatures        = &device_features,
	};

	vr = vkCreateDevice(_skr_vk.physical_device, &device_info, NULL, &_skr_vk.device);
	SKR_VK_CHECK_RET(vr, "vkCreateDevice", false);

	volkLoadDevice(_skr_vk.device);

	// Get graphics queue
	vkGetDeviceQueue(_skr_vk.device, _skr_vk.graphics_queue_family, 0, &_skr_vk.graphics_queue);
	_skr_vk.present_queue = _skr_vk.graphics_queue;

	// Get transfer queue (same as graphics if no dedicated queue)
	if (_skr_vk.has_dedicated_transfer) {
		vkGetDeviceQueue(_skr_vk.device, _skr_vk.transfer_queue_family, 0, &_skr_vk.transfer_queue);
	} else {
		_skr_vk.transfer_queue = _skr_vk.graphics_queue;
	}

	// Initialize queue mutexes for thread-safe queue submission
	// We use 3 slots but may only need 1 or 2 if queues are aliased
	mtx_init(&_skr_vk.queue_mutexes[0], mtx_plain);  // Graphics queue
	mtx_init(&_skr_vk.queue_mutexes[1], mtx_plain);  // Present queue (may alias graphics)
	mtx_init(&_skr_vk.queue_mutexes[2], mtx_plain);  // Transfer queue (may alias graphics)

	// Set up mutex pointers based on queue aliasing
	_skr_vk.graphics_queue_mutex = &_skr_vk.queue_mutexes[0];

	// Present always aliases graphics
	_skr_vk.present_queue_mutex = &_skr_vk.queue_mutexes[0];

	// Transfer uses dedicated mutex if it has a dedicated queue, otherwise aliases graphics
	if (_skr_vk.has_dedicated_transfer) {
		_skr_vk.transfer_queue_mutex = &_skr_vk.queue_mutexes[2];
	} else {
		_skr_vk.transfer_queue_mutex = &_skr_vk.queue_mutexes[0];
	}

	// Create command pool
	VkCommandPoolCreateInfo pool_info = {
		.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = _skr_vk.graphics_queue_family,
	};

	vr = vkCreateCommandPool(_skr_vk.device, &pool_info, NULL, &_skr_vk.command_pool);
	SKR_VK_CHECK_RET(vr, "vkCreateCommandPool", false);
	_skr_cmd_destroy_command_pool(&_skr_vk.destroy_list, _skr_vk.command_pool);

	// Allocate command buffers (one per frame in flight)
	VkCommandBufferAllocateInfo alloc_info = {
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool        = _skr_vk.command_pool,
		.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = SKR_MAX_FRAMES_IN_FLIGHT,
	};

	vr = vkAllocateCommandBuffers(_skr_vk.device, &alloc_info, _skr_vk.command_buffers);
	SKR_VK_CHECK_RET(vr, "vkAllocateCommandBuffers", false);

	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		vr = vkCreateFence(_skr_vk.device, &(VkFenceCreateInfo){
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT, // Start signaled so first frame doesn't wait
		}, NULL, &_skr_vk.frame_fences[i]);
		SKR_VK_CHECK_RET(vr, "vkCreateFence", false);
		_skr_cmd_destroy_fence(&_skr_vk.destroy_list, _skr_vk.frame_fences[i]);
	}

	vr = vkCreateQueryPool(_skr_vk.device, &(VkQueryPoolCreateInfo){
		.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType  = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = 2 * SKR_MAX_FRAMES_IN_FLIGHT,
	}, NULL, &_skr_vk.timestamp_pool);
	SKR_VK_CHECK_RET(vr, "vkCreateQueryPool", false);
	_skr_cmd_destroy_query_pool(&_skr_vk.destroy_list, _skr_vk.timestamp_pool);

	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		_skr_vk.timestamps_valid[i] = false;
	}

	vr = vkCreatePipelineCache(_skr_vk.device, &(VkPipelineCacheCreateInfo){
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
	}, NULL, &_skr_vk.pipeline_cache);
	SKR_VK_CHECK_RET(vr, "vkCreatePipelineCache", false);
	_skr_cmd_destroy_pipeline_cache(&_skr_vk.destroy_list, _skr_vk.pipeline_cache);

	// Create descriptor pool for compute shaders
	VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount = 1000 },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          .descriptorCount = 1000 },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1000 },
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         .descriptorCount = 1000 },
	};

	VkDescriptorPoolCreateInfo desc_pool_info = {
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets       = 1000,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes    = pool_sizes,
	};

	vr = vkCreateDescriptorPool(_skr_vk.device, &desc_pool_info, NULL, &_skr_vk.descriptor_pool);
	SKR_VK_CHECK_RET(vr, "vkCreateDescriptorPool", false);
	_skr_cmd_destroy_descriptor_pool(&_skr_vk.destroy_list, _skr_vk.descriptor_pool);

	_skr_pipeline_init();

	if (!_skr_cmd_init()) {
		skr_log(skr_log_critical, "Failed to initialize upload system");
		return false;
	}

	// Initialize main thread
	skr_thread_init();

	const skr_tex_sampler_t sampler = {
		.sample  = skr_tex_sample_linear,
		.address = skr_tex_address_clamp
	};
	uint32_t color = 0xFFFFFFFF;
	skr_tex_create( skr_tex_fmt_rgba32_linear, skr_tex_flags_readable, sampler, (skr_vec3i_t){1, 1, 1}, 1, 1, &(skr_tex_data_t){.data = &color, .mip_count = 1, .layer_count = 1}, &_skr_vk.default_tex_white);
	color = 0xFF808080;
	skr_tex_create( skr_tex_fmt_rgba32_linear, skr_tex_flags_readable, sampler, (skr_vec3i_t){1, 1, 1}, 1, 1, &(skr_tex_data_t){.data = &color, .mip_count = 1, .layer_count = 1}, &_skr_vk.default_tex_gray);
	color = 0xFF000000;
	skr_tex_create( skr_tex_fmt_rgba32_linear, skr_tex_flags_readable, sampler, (skr_vec3i_t){1, 1, 1}, 1, 1, &(skr_tex_data_t){.data = &color, .mip_count = 1, .layer_count = 1}, &_skr_vk.default_tex_black);

	_skr_vk.initialized = true;
	return true;
}

void skr_shutdown(void) {
	if (!_skr_vk.initialized) return;

	vkDeviceWaitIdle(_skr_vk.device);

	skr_tex_destroy(&_skr_vk.default_tex_white);
	skr_tex_destroy(&_skr_vk.default_tex_gray);
	skr_tex_destroy(&_skr_vk.default_tex_black);

	_skr_cmd_shutdown      ();  // Executes per-command destroy lists (may free bind pool slots)
	_skr_pipeline_shutdown ();

	_skr_destroy_list_execute(&_skr_vk.destroy_list);  // Execute global destroy list
	_skr_destroy_list_free   (&_skr_vk.destroy_list);

	_skr_bind_pool_shutdown();     // Free bind pool after all deferred destroys are done
	_skr_sampler_cache_shutdown(); // Destroy cached samplers after GPU is idle

	// Free dynamic arrays
	if (_skr_vk.pending_transitions)      _skr_free(_skr_vk.pending_transitions);
	if (_skr_vk.pending_transition_types) _skr_free(_skr_vk.pending_transition_types);

	// Destroy queue mutexes
	for (int32_t i = 0; i < 3; i++) {
		mtx_destroy(&_skr_vk.queue_mutexes[i]);
	}

	// Destroy device and instance directly (special cases not in destroy list)
	if (_skr_vk.device   != VK_NULL_HANDLE) { vkDestroyDevice  (_skr_vk.device,   NULL); }
	if (_skr_vk.instance != VK_NULL_HANDLE) { vkDestroyInstance(_skr_vk.instance, NULL); }

	_skr_vk = (_skr_vk_t){};
}

VkInstance skr_get_vk_instance(void) {
	return _skr_vk.instance;
}

VkDevice skr_get_vk_device(void) {
	return _skr_vk.device;
}

VkPhysicalDevice skr_get_vk_physical_device(void) {
	return _skr_vk.physical_device;
}

VkQueue skr_get_vk_graphics_queue(void) {
	return _skr_vk.graphics_queue;
}

uint32_t skr_get_vk_graphics_queue_family(void) {
	return _skr_vk.graphics_queue_family;
}

void skr_get_vk_device_uuid(uint8_t out_uuid[VK_UUID_SIZE]) {
	VkPhysicalDeviceIDProperties id_props = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
	};
	VkPhysicalDeviceProperties2 props2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &id_props,
	};
	vkGetPhysicalDeviceProperties2(_skr_vk.physical_device, &props2);
	memcpy(out_uuid, id_props.deviceUUID, VK_UUID_SIZE);
}

