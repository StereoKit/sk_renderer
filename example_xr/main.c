// OpenXR + sk_renderer Example
// A minimal VR application demonstrating sk_renderer with OpenXR.
//
// Two graphics bindings, matching sk_renderer's own backend split:
//   native  Vulkan (XR_KHR_vulkan_enable2) + the system OpenXR loader
//   web     WebGPU (XR_SKW_webgpu_enable)  + WebOpenXR, on top of WebXR
//
// Everything above the binding - scenes, app_xr.c, the whole render path - is
// shared. See openxr_util.h.

#include "openxr_util.h"
#include "app_xr.h"

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <sk_app.h>

#ifdef __EMSCRIPTEN__
	// wgpuCreateInstance and emscripten_webgpu_get_device both come from
	// emdawnwebgpu's <webgpu/webgpu.h>, which openxr_util.h already pulls in
	// via openxr_webxr.h - there is no separate emscripten/html5_webgpu.h here.
	#include <emscripten.h>
#endif

///////////////////////////////////////////
// OpenXR Error Checking
///////////////////////////////////////////

#include <openxr/openxr_reflection.h>

static const char *xr_result_to_string(XrResult result) {
	switch (result) {
#define ENTRY(NAME, VALUE) case VALUE: return #NAME;
	XR_LIST_ENUM_XrResult(ENTRY)
#undef ENTRY
	default: return "<UNKNOWN>";
	}
}

// Check XrResult and log on failure - continues execution
#define XR_CHECK(call) do { \
	XrResult _result = (call); \
	if (XR_FAILED(_result)) \
		ska_log(ska_log_error, "[OpenXR] %s:%d %s returned %s (%d)", __FILE__, __LINE__, #call, xr_result_to_string(_result), _result); \
} while(0)

///////////////////////////////////////////
// Globals
///////////////////////////////////////////

const XrPosef xr_pose_identity = { {0, 0, 0, 1}, {0, 0, 0} };

XrInstance               xr_instance      = XR_NULL_HANDLE;
XrSession                xr_session       = XR_NULL_HANDLE;
XrSessionState           xr_session_state = XR_SESSION_STATE_UNKNOWN;
bool                     xr_running       = false;
XrSpace                  xr_app_space     = XR_NULL_HANDLE;
XrSystemId               xr_system_id     = XR_NULL_SYSTEM_ID;
xr_input_state_t         xr_input         = {0};
XrEnvironmentBlendMode   xr_blend         = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
XrDebugUtilsMessengerEXT xr_debug         = XR_NULL_HANDLE;

XrView*                  xr_views         = NULL;
XrViewConfigurationView* xr_config_views  = NULL;
xr_swapchain_t           xr_swapchain     = {0};  // Single stereo swapchain
uint32_t                 xr_view_count    = 0;

#ifdef __EMSCRIPTEN__
// Kept so the per-frame transient wrap can name the format the swapchain was
// created with - the compositor's texture is new each frame, its format is not.
static skr_tex_fmt_      xr_swapchain_tex_format = skr_tex_fmt_rgba32;
#endif

skr_tex_t                xr_depth_texture = {0};  // Single array depth texture

// MSAA is native only: WebGPU forbids a multisampled texture with more than one
// array layer, and a stereo projection layer is two by definition. The web path
// renders straight into the compositor's texture.
#ifdef __EMSCRIPTEN__
static const int32_t     xr_msaa_samples  = 1;
#else
skr_tex_t                xr_color_msaa    = {0};  // MSAA array color texture (render target)
static const int32_t     xr_msaa_samples  = 4;    // MSAA sample count
#endif

XrFormFactor             xr_config_form   = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
XrViewConfigurationType  xr_config_view   = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

#ifndef __EMSCRIPTEN__
// Function pointers for OpenXR extensions (vulkan_enable2)
PFN_xrCreateVulkanInstanceKHR          ext_xrCreateVulkanInstanceKHR          = NULL;
PFN_xrCreateVulkanDeviceKHR            ext_xrCreateVulkanDeviceKHR            = NULL;
PFN_xrGetVulkanGraphicsDevice2KHR      ext_xrGetVulkanGraphicsDevice2KHR      = NULL;
PFN_xrGetVulkanGraphicsRequirements2KHR ext_xrGetVulkanGraphicsRequirements2KHR = NULL;
PFN_xrCreateDebugUtilsMessengerEXT     ext_xrCreateDebugUtilsMessengerEXT     = NULL;
PFN_xrDestroyDebugUtilsMessengerEXT    ext_xrDestroyDebugUtilsMessengerEXT    = NULL;
#endif

///////////////////////////////////////////
// Forward declarations
///////////////////////////////////////////

static bool openxr_render_layer(XrTime predicted_time, XrCompositionLayerProjectionView* views, uint32_t view_count, XrCompositionLayerProjection* layer);
static void xr_swapchain_destroy(xr_swapchain_t* swapchain);
#ifdef __EMSCRIPTEN__
static bool openxr_create_swapchain(void);
#endif

///////////////////////////////////////////
// Debug callback
//
// XR_EXT_debug_utils is a loader/runtime facility; WebOpenXR doesn't implement
// it (the browser console and sk_app's log carry the same information), so the
// whole messenger is native-only.
///////////////////////////////////////////

#ifndef __EMSCRIPTEN__
static XrBool32 XRAPI_CALL xr_debug_callback(XrDebugUtilsMessageSeverityFlagsEXT severity, XrDebugUtilsMessageTypeFlagsEXT types, const XrDebugUtilsMessengerCallbackDataEXT* data, void* user_data) {
	(void)user_data;
	const char* type_str = "";
	if      (types & XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT ) type_str = "VALIDATION ";
	else if (types & XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) type_str = "PERFORMANCE ";
	else if (types & XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT) type_str = "CONFORMANCE ";

	ska_log_ level = ska_log_info;
	if      (severity & XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT  ) level = ska_log_error;
	else if (severity & XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) level = ska_log_warn;

	ska_log(level, "[XR %s] %s: %s", type_str, data->functionName, data->message);
	return XR_FALSE;
}

///////////////////////////////////////////
// sk_renderer callbacks for XR_KHR_vulkan_enable2
///////////////////////////////////////////

// HTC Wave runtime workaround: it calls vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateDevice")
// which is invalid per the Vulkan spec. We wrap the proc addr to retry with the real instance.
static VkInstance                s_vk_instance          = VK_NULL_HANDLE;
static PFN_vkGetInstanceProcAddr s_real_get_proc_addr   = NULL;

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL xr_vkGetInstanceProcAddr(VkInstance instance, const char* name) {
	return s_real_get_proc_addr(s_vk_instance != VK_NULL_HANDLE ? s_vk_instance : instance, name);
}

// Instance creation callback: lets OpenXR create the VkInstance
static void* xr_instance_create_callback(skr_instance_create_info_t* info, void* user_data) {
	(void)user_data;
	s_real_get_proc_addr = (PFN_vkGetInstanceProcAddr)info->get_instance_proc_addr;

	VkInstance instance  = VK_NULL_HANDLE;
	VkResult   vk_result = VK_SUCCESS;
	XrResult   xr_result = ext_xrCreateVulkanInstanceKHR(xr_instance, &(XrVulkanInstanceCreateInfoKHR){
		.type               = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
		.systemId           = xr_system_id,
		.pfnGetInstanceProcAddr = xr_vkGetInstanceProcAddr,
		.vulkanCreateInfo   = info->instance_create_info,
	}, &instance, &vk_result);

	if (XR_FAILED(xr_result) || vk_result != VK_SUCCESS) {
		ska_log(ska_log_error, "xrCreateVulkanInstanceKHR failed (xr=%d, vk=%d)", xr_result, vk_result);
		return NULL;
	}
	s_vk_instance = instance;
	return instance;
}

// Device init callback: gets the physical device OpenXR wants (no extension querying needed with enable2)
static skr_device_request_t xr_device_init_callback(void* vk_instance, void* user_data) {
	(void)user_data;

	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	ext_xrGetVulkanGraphicsDevice2KHR(xr_instance, &(XrVulkanGraphicsDeviceGetInfoKHR){
		.type           = XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR,
		.systemId       = xr_system_id,
		.vulkanInstance = (VkInstance)vk_instance,
	}, &physical_device);

	return (skr_device_request_t){
		.physical_device = physical_device,
	};
}

// Device creation callback: lets OpenXR create the VkDevice
static void* xr_device_create_callback(skr_device_create_info_t* info, void* user_data) {
	(void)user_data;

	VkDevice device    = VK_NULL_HANDLE;
	VkResult vk_result = VK_SUCCESS;
	XrResult xr_result = ext_xrCreateVulkanDeviceKHR(xr_instance, &(XrVulkanDeviceCreateInfoKHR){
		.type               = XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR,
		.systemId           = xr_system_id,
		.pfnGetInstanceProcAddr = xr_vkGetInstanceProcAddr,
		.vulkanPhysicalDevice = (VkPhysicalDevice)info->vk_physical_device,
		.vulkanCreateInfo   = info->device_create_info,
	}, &device, &vk_result);

	if (XR_FAILED(xr_result) || vk_result != VK_SUCCESS) {
		ska_log(ska_log_error, "xrCreateVulkanDeviceKHR failed (xr=%d, vk=%d)", xr_result, vk_result);
		return NULL;
	}
	return device;
}
#endif // !__EMSCRIPTEN__

///////////////////////////////////////////
// OpenXR Initialization
///////////////////////////////////////////

bool openxr_init(const char* app_name) {
#ifdef __ANDROID__
	// On Android, must initialize the OpenXR loader with JNI context before any other calls
	PFN_xrInitializeLoaderKHR xrInitializeLoaderKHR = NULL;
	XrResult loader_result = xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", (PFN_xrVoidFunction*)&xrInitializeLoaderKHR);

	if (loader_result != XR_SUCCESS || !xrInitializeLoaderKHR) {
		ska_log(ska_log_warn, "xrGetInstanceProcAddr for xrInitializeLoaderKHR failed (result=%d, fn=%p)", loader_result, (void*)xrInitializeLoaderKHR);
	} else {
		JavaVM* vm       = (JavaVM*)ska_android_get_vm();
		jobject activity = (jobject)ska_android_get_activity();
		ska_log(ska_log_info, "Android loader init: VM=%p, Activity=%p", (void*)vm, (void*)activity);

		if (!vm || !activity) {
			ska_log(ska_log_error, "sk_app Android context not available");
			return false;
		}

		XrResult init_result = xrInitializeLoaderKHR((XrLoaderInitInfoBaseHeaderKHR*)&(XrLoaderInitInfoAndroidKHR){
			.type               = XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR,
			.applicationVM      = vm,
			.applicationContext = activity });

		if (XR_FAILED(init_result)) {
			ska_log(ska_log_error, "xrInitializeLoaderKHR failed (result=%d)", init_result);
			return false;
		}
		ska_log(ska_log_info, "OpenXR Android loader initialized successfully");
	}
#endif

	// Check available extensions
	uint32_t ext_count = 0;
	xrEnumerateInstanceExtensionProperties(NULL, 0, &ext_count, NULL);

	XrExtensionProperties* xr_exts = calloc(ext_count, sizeof(XrExtensionProperties));
	for (uint32_t i = 0; i < ext_count; i++) {
		xr_exts[i].type = XR_TYPE_EXTENSION_PROPERTIES;
	}
	xrEnumerateInstanceExtensionProperties(NULL, ext_count, &ext_count, xr_exts);

	// Build list of extensions to use
	const char* ask_extensions[] = {
#ifdef __EMSCRIPTEN__
		XR_SKW_WEBGPU_ENABLE_EXTENSION_NAME,
		// Optional. With it the compositor lends us its own texture each frame
		// and we render straight into it; without it the runtime allocates a
		// swapchain and copies. See the transient handling in the frame loop.
		XR_SKW_TRANSIENT_SWAPCHAIN_IMAGES_EXTENSION_NAME,
#else
		XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
		XR_EXT_DEBUG_UTILS_EXTENSION_NAME,
	#ifdef __ANDROID__
		XR_KHR_LOADER_INIT_ANDROID_EXTENSION_NAME,
		XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME,
	#endif
#endif
	};
	const char* use_extensions[16];
	uint32_t    use_count = 0;

	ska_log(ska_log_info, "OpenXR extensions available:");
	bool has_gfx = false;
	for (uint32_t i = 0; i < ext_count; i++) {
		ska_log(ska_log_info, "- %s", xr_exts[i].extensionName);

		for (uint32_t ask = 0; ask < sizeof(ask_extensions)/sizeof(ask_extensions[0]); ask++) {
			if (strcmp(ask_extensions[ask], xr_exts[i].extensionName) == 0) {
				use_extensions[use_count++] = ask_extensions[ask];
#ifdef __EMSCRIPTEN__
				if (strcmp(ask_extensions[ask], XR_SKW_WEBGPU_ENABLE_EXTENSION_NAME) == 0)
					has_gfx = true;
				if (strcmp(ask_extensions[ask], XR_SKW_TRANSIENT_SWAPCHAIN_IMAGES_EXTENSION_NAME) == 0)
					xr_swapchain.transient = true;
#else
				if (strcmp(ask_extensions[ask], XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME) == 0)
					has_gfx = true;
#endif
				break;
			}
		}
	}
	free(xr_exts);

	if (!has_gfx) {
#ifdef __EMSCRIPTEN__
		// Almost always means the browser has no XRGPUBinding: Quest Browser 149
		// lacks it entirely, and desktop Chrome needs
		// --enable-features=WebXRWebGPUBinding.
		ska_log(ska_log_error, "OpenXR runtime does not support %s - does this browser have XRGPUBinding?",
			XR_SKW_WEBGPU_ENABLE_EXTENSION_NAME);
#else
		ska_log(ska_log_error, "OpenXR runtime does not support Vulkan");
#endif
		return false;
	}

	// Create OpenXR instance
	XrInstanceCreateInfo create_info = {
		.type                       = XR_TYPE_INSTANCE_CREATE_INFO,
		.enabledExtensionCount      = use_count,
		.enabledExtensionNames      = use_extensions,
		.applicationInfo.apiVersion = XR_CURRENT_API_VERSION  };
	strncpy(create_info.applicationInfo.applicationName, app_name, XR_MAX_APPLICATION_NAME_SIZE - 1);

#ifdef __ANDROID__
	XrInstanceCreateInfoAndroidKHR android_create_info = {
		.type                = XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR,
		.applicationVM       = ska_android_get_vm(),
		.applicationActivity = ska_android_get_activity() };
	create_info.next = &android_create_info;
#endif

	XrResult result = xrCreateInstance(&create_info, &xr_instance);
	if (XR_FAILED(result) || xr_instance == XR_NULL_HANDLE) {
		ska_log(ska_log_error, "Failed to create OpenXR instance (result=%d)", result);
		ska_log(ska_log_error, "Make sure an OpenXR runtime is installed and active.");
		return false;
	}

#ifndef __EMSCRIPTEN__
	// Load extension function pointers (vulkan_enable2)
	xrGetInstanceProcAddr(xr_instance, "xrCreateVulkanInstanceKHR",           (PFN_xrVoidFunction*)&ext_xrCreateVulkanInstanceKHR);
	xrGetInstanceProcAddr(xr_instance, "xrCreateVulkanDeviceKHR",             (PFN_xrVoidFunction*)&ext_xrCreateVulkanDeviceKHR);
	xrGetInstanceProcAddr(xr_instance, "xrGetVulkanGraphicsDevice2KHR",       (PFN_xrVoidFunction*)&ext_xrGetVulkanGraphicsDevice2KHR);
	xrGetInstanceProcAddr(xr_instance, "xrGetVulkanGraphicsRequirements2KHR", (PFN_xrVoidFunction*)&ext_xrGetVulkanGraphicsRequirements2KHR);
	xrGetInstanceProcAddr(xr_instance, "xrCreateDebugUtilsMessengerEXT",      (PFN_xrVoidFunction*)&ext_xrCreateDebugUtilsMessengerEXT);
	xrGetInstanceProcAddr(xr_instance, "xrDestroyDebugUtilsMessengerEXT",     (PFN_xrVoidFunction*)&ext_xrDestroyDebugUtilsMessengerEXT);

	// Set up debug messenger (optional)
	if (ext_xrCreateDebugUtilsMessengerEXT) {
		XrDebugUtilsMessengerCreateInfoEXT debug_info = { XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
		debug_info.messageTypes =
			XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT     |
			XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT  |
			XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT;
		debug_info.messageSeverities =
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debug_info.userCallback = xr_debug_callback;
		debug_info.userData     = NULL;
		ext_xrCreateDebugUtilsMessengerEXT(xr_instance, &debug_info, &xr_debug);
	}
#endif // !__EMSCRIPTEN__

	// Get system (HMD)
	result = xrGetSystem(xr_instance, &(XrSystemGetInfo){
			.type       = XR_TYPE_SYSTEM_GET_INFO,
			.formFactor = xr_config_form },
		&xr_system_id);

	if (XR_FAILED(result)) {
		ska_log(ska_log_error, "Failed to get OpenXR system (result=%d)", result);
		ska_log(ska_log_error, "Make sure a VR headset is connected.");
		return false;
	}

	// Get blend mode - prefer ALPHA_BLEND for AR passthrough, fall back to OPAQUE
	uint32_t blend_count = 0;
	XrEnvironmentBlendMode blend_modes[8];
	xrEnumerateEnvironmentBlendModes(xr_instance, xr_system_id, xr_config_view, 8, &blend_count, blend_modes);

	xr_blend = blend_modes[0];
	for (uint32_t i = 0; i < blend_count; i++) {
		if (blend_modes[i] == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) {
			xr_blend = XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND;
			break;
		}
	}

	const char* blend_name = (xr_blend == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) ? "ALPHA_BLEND" :
	                         (xr_blend == XR_ENVIRONMENT_BLEND_MODE_ADDITIVE)    ? "ADDITIVE" : "OPAQUE";
	ska_log(ska_log_info, "Using blend mode: %s", blend_name);

	// Get graphics requirements. Mandatory before xrCreateSession on both paths -
	// Monado enforces it, and so does WebOpenXR.
#ifdef __EMSCRIPTEN__
	// Not in the dispatch table as a named symbol; it's an extension entry point.
	PFN_xrVoidFunction gfx_req_fn = NULL;
	if (XR_FAILED(xrGetInstanceProcAddr(xr_instance, "xrGetWebGPUGraphicsRequirementsSKW", &gfx_req_fn)) || gfx_req_fn == NULL) {
		ska_log(ska_log_error, "xrGetWebGPUGraphicsRequirementsSKW missing");
		return false;
	}
	XrGraphicsRequirementsWebGPUSKW gfx_requirements = { XR_TYPE_GRAPHICS_REQUIREMENTS_WEBGPU_SKW };
	((PFN_xrGetWebGPUGraphicsRequirementsSKW)gfx_req_fn)(xr_instance, xr_system_id, &gfx_requirements);
	ska_log(ska_log_info, "OpenXR WebGPU requirements: xrCompatible adapter required = %d",
		(int)gfx_requirements.xrCompatibleAdapterRequired);

	// The adapter must be requested {xrCompatible:true} before wasm starts, so
	// web/pre.js parks the device on Module.preinitializedWebGPUDevice. Instance
	// and device are mandatory together: without the device skr_init takes the
	// native path and dies with "wgpuCreateInstance failed".
	if (!EM_ASM_INT({ return Module['preinitializedWebGPUDevice'] ? 1 : 0; })) {
		ska_log(ska_log_error, "No pre-initialized WebGPU device - does this browser have WebGPU?");
		return false;
	}
	if (!skr_init((skr_settings_t){
		.app_name          = app_name,
		.app_version       = 1,
		.enable_validation = true,
		.wgpu_instance     = wgpuCreateInstance(NULL),
		.wgpu_device       = emscripten_webgpu_get_device(),
	})) {
		ska_log(ska_log_error, "Failed to initialize sk_renderer");
		return false;
	}

	XrGraphicsBindingWebGPUSKW binding = {
		.type     = XR_TYPE_GRAPHICS_BINDING_WEBGPU_SKW,
		.instance = skr_get_wgpu_instance(),
		.adapter  = skr_get_wgpu_adapter (),
		.device   = skr_get_wgpu_device  () };
#else
	XrGraphicsRequirementsVulkan2KHR gfx_requirements = { XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR };
	ext_xrGetVulkanGraphicsRequirements2KHR(xr_instance, xr_system_id, &gfx_requirements);
	ska_log(ska_log_info, "OpenXR Vulkan requirements: API %u.%u.%u - %u.%u.%u",
		XR_VERSION_MAJOR(gfx_requirements.minApiVersionSupported),
		XR_VERSION_MINOR(gfx_requirements.minApiVersionSupported),
		XR_VERSION_PATCH(gfx_requirements.minApiVersionSupported),
		XR_VERSION_MAJOR(gfx_requirements.maxApiVersionSupported),
		XR_VERSION_MINOR(gfx_requirements.maxApiVersionSupported),
		XR_VERSION_PATCH(gfx_requirements.maxApiVersionSupported));

	// Initialize sk_renderer with enable2 callbacks.
	// OpenXR creates VkInstance and VkDevice on our behalf, handling extensions internally.
	if (!skr_init((skr_settings_t){
		.app_name                  = app_name,
		.app_version               = 1,
		.enable_validation         = true,
		.instance_create_callback  = xr_instance_create_callback,
		.device_init_callback      = xr_device_init_callback,
		.device_create_callback    = xr_device_create_callback,
	})) {
		ska_log(ska_log_error, "Failed to initialize sk_renderer");
		return false;
	}

	// Create OpenXR session with our Vulkan device
	XrGraphicsBindingVulkan2KHR binding = {
		.type             = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR,
		.instance         = skr_get_vk_instance             (),
		.physicalDevice   = skr_get_vk_physical_device      (),
		.device           = skr_get_vk_device               (),
		.queueFamilyIndex = skr_get_vk_graphics_queue_family(),
		.queueIndex       = 0 };
#endif

	result = xrCreateSession(xr_instance, &(XrSessionCreateInfo){
		.type     = XR_TYPE_SESSION_CREATE_INFO,
		.next     = &binding,
		.systemId = xr_system_id
	}, &xr_session);
	if (XR_FAILED(result) || xr_session == XR_NULL_HANDLE) {
		ska_log(ska_log_error, "Failed to create OpenXR session (result=%d)", result);
		return false;
	}

	// Create reference space
	xrCreateReferenceSpace(xr_session, &(XrReferenceSpaceCreateInfo){
		.type                 = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
		.poseInReferenceSpace = xr_pose_identity,
		.referenceSpaceType   = XR_REFERENCE_SPACE_TYPE_LOCAL
	}, &xr_app_space);

	// Enumerate views
	xrEnumerateViewConfigurationViews(xr_instance, xr_system_id, xr_config_view, 0, &xr_view_count, NULL);
	xr_config_views = calloc(xr_view_count, sizeof(XrViewConfigurationView));
	xr_views        = calloc(xr_view_count, sizeof(XrView));
	for (uint32_t i = 0; i < xr_view_count; i++) {
		xr_config_views[i].type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
		xr_views[i].type        = XR_TYPE_VIEW;
	}
	xrEnumerateViewConfigurationViews(xr_instance, xr_system_id, xr_config_view, xr_view_count, &xr_view_count, xr_config_views);

	ska_log(ska_log_info, "OpenXR views: %u", xr_view_count);
	for (uint32_t i = 0; i < xr_view_count; i++) {
		ska_log(ska_log_info, "  View %u: %ux%u, %u samples", i,
			xr_config_views[i].recommendedImageRectWidth,
			xr_config_views[i].recommendedImageRectHeight,
			xr_config_views[i].recommendedSwapchainSampleCount);
	}

#ifdef __EMSCRIPTEN__
	// Web: deferred to the first frame. The projection layer does not exist
	// until an immersive session does, and the view size reported now is a
	// guess. Native runtimes answer both at xrCreateSession.
	return true;
}

// Split out of openxr_init so the web can call it later. Native calls it inline
// and nothing about its behaviour changed.
static bool openxr_create_swapchain(void) {
	// Re-read the view configuration: on the web the size at xrCreateSession is
	// provisional and only becomes exact once the session is running.
	xrEnumerateViewConfigurationViews(xr_instance, xr_system_id, xr_config_view,
		xr_view_count, &xr_view_count, xr_config_views);
	ska_log(ska_log_info, "OpenXR views (final): %ux%u",
		xr_config_views[0].recommendedImageRectWidth,
		xr_config_views[0].recommendedImageRectHeight);
#endif

	// The runtime lists only formats it supports, in its own preference order,
	// so take the first one we recognise rather than second-guessing it. Values
	// are backend-native: VkFormat natively, WGPUTextureFormat on the web.
	static const struct { int64_t api_format; skr_tex_fmt_ tex_format; } known_formats[] = {
#ifdef __EMSCRIPTEN__
		{ 18, skr_tex_fmt_rgba32      },   // WGPUTextureFormat_RGBA8Unorm
		{ 19, skr_tex_fmt_rgba32_srgb },   // WGPUTextureFormat_RGBA8UnormSrgb
		{ 23, skr_tex_fmt_bgra32      },   // WGPUTextureFormat_BGRA8Unorm
		{ 24, skr_tex_fmt_bgra32_srgb },   // WGPUTextureFormat_BGRA8UnormSrgb
#else
		{ VK_FORMAT_R8G8B8A8_SRGB,  skr_tex_fmt_rgba32_srgb },
		{ VK_FORMAT_R8G8B8A8_UNORM, skr_tex_fmt_rgba32      },
		{ VK_FORMAT_B8G8R8A8_SRGB,  skr_tex_fmt_bgra32_srgb },
		{ VK_FORMAT_B8G8R8A8_UNORM, skr_tex_fmt_bgra32      },
#endif
	};

	uint32_t format_count = 0;
	xrEnumerateSwapchainFormats(xr_session, 0, &format_count, NULL);
	int64_t* formats = malloc(format_count * sizeof(int64_t));
	xrEnumerateSwapchainFormats(xr_session, format_count, &format_count, formats);

	int64_t      swapchain_format = 0;
	skr_tex_fmt_ tex_format       = skr_tex_fmt_none;
	for (uint32_t i = 0; i < format_count && tex_format == skr_tex_fmt_none; i++) {
		for (uint32_t k = 0; k < sizeof(known_formats)/sizeof(known_formats[0]); k++) {
			if (formats[i] != known_formats[k].api_format) continue;
			swapchain_format = formats[i];
			tex_format       = known_formats[k].tex_format;
			break;
		}
	}
	free(formats);

	if (tex_format == skr_tex_fmt_none) {
		ska_log(ska_log_error, "None of the runtime's %u swapchain formats are ones we can render into", format_count);
		return false;
	}
	ska_log(ska_log_info, "Using swapchain format: %lld", (long long)swapchain_format);

#ifdef __EMSCRIPTEN__
	xr_swapchain_tex_format = tex_format;
#endif

	// Create a single stereo swapchain (without MSAA - we'll render to separate MSAA texture and resolve)
	// Use recommended settings from first view (both eyes should match for stereo)
	XrViewConfigurationView* view = &xr_config_views[0];

	ska_log(ska_log_info, "Creating stereo swapchain: %ux%u, %d layers (MSAA %dx to separate texture)",
		view->recommendedImageRectWidth,
		view->recommendedImageRectHeight,
		xr_view_count,
		xr_msaa_samples);

	XrSwapchainCreateInfo swapchain_info = {
		.type        = XR_TYPE_SWAPCHAIN_CREATE_INFO,
		.arraySize   = xr_view_count,
		.mipCount    = 1,
		.faceCount   = 1,
		.format      = swapchain_format,
		.width       = view->recommendedImageRectWidth,
		.height      = view->recommendedImageRectHeight,
		.sampleCount = 1,  // No MSAA in swapchain - we resolve from separate MSAA texture
		.usageFlags  = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT,
	};

	XrSwapchain handle;
	XrResult    sc_result = xrCreateSwapchain(xr_session, &swapchain_info, &handle);
	if (XR_FAILED(sc_result)) {
		ska_log(ska_log_error, "Failed to create swapchain (result=%d)", sc_result);
		return false;
	}

	// Store swapchain info
	xr_swapchain.handle       = handle;
	xr_swapchain.width        = swapchain_info.width;
	xr_swapchain.height       = swapchain_info.height;
	xr_swapchain.array_size   = xr_view_count;
	xr_swapchain.sample_count = 1;  // Swapchain has no MSAA - we resolve to it

#ifdef __EMSCRIPTEN__
	if (xr_swapchain.transient) {
		// The compositor lends out a different texture every frame, so there is
		// nothing stable to enumerate here. One slot, re-pointed each frame in
		// openxr_render_layer - the extension's contract is that the app
		// re-reads and caches nothing.
		xr_swapchain.image_count    = 1;
		xr_swapchain.color_textures = calloc(1, sizeof(skr_tex_t));
	} else {
		uint32_t image_count = 0;
		xrEnumerateSwapchainImages(handle, 0, &image_count, NULL);

		XrSwapchainImageWebGPUSKW* images = calloc(image_count, sizeof(XrSwapchainImageWebGPUSKW));
		for (uint32_t j = 0; j < image_count; j++)
			images[j].type = XR_TYPE_SWAPCHAIN_IMAGE_WEBGPU_SKW;
		xrEnumerateSwapchainImages(handle, image_count, &image_count, (XrSwapchainImageBaseHeader*)images);

		xr_swapchain.image_count    = image_count;
		xr_swapchain.color_textures = calloc(image_count, sizeof(skr_tex_t));
		for (uint32_t j = 0; j < image_count; j++) {
			skr_tex_create_external_wgpu((skr_tex_external_wgpu_info_t){
				.texture      = images[j].texture,
				.format       = tex_format,
				.sampler      = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp },
				.size         = { (int32_t)swapchain_info.width, (int32_t)swapchain_info.height, (int32_t)xr_view_count },
				.multisample  = 1,
				.owns_texture = false,
			}, &xr_swapchain.color_textures[j]);
		}
		free(images);
	}
#else
	// Enumerate swapchain images
	uint32_t image_count = 0;
	xrEnumerateSwapchainImages(handle, 0, &image_count, NULL);

	XrSwapchainImageVulkanKHR* images = calloc(image_count, sizeof(XrSwapchainImageVulkanKHR));
	for (uint32_t j = 0; j < image_count; j++) {
		images[j].type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR;
	}
	xrEnumerateSwapchainImages(handle, image_count, &image_count, (XrSwapchainImageBaseHeader*)images);

	xr_swapchain.image_count  = image_count;
	xr_swapchain.color_textures = calloc(image_count, sizeof(skr_tex_t));

	// Wrap each VkImage in sk_renderer texture (as array texture, no MSAA)
	for (uint32_t j = 0; j < image_count; j++) {
		skr_tex_create_external_vk((skr_tex_external_info_t){
			.image          = images[j].image,
			.view           = VK_NULL_HANDLE,
			.memory         = VK_NULL_HANDLE,
			.format         = tex_format,
			.size           = { (int32_t)swapchain_info.width, (int32_t)swapchain_info.height, 1 },
			.current_layout = VK_IMAGE_LAYOUT_UNDEFINED,
			.sampler        = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp },
			.multisample    = 1,  // No MSAA - this is the resolve target
			.array_layers   = xr_view_count,
			.owns_image     = false,
		}, &xr_swapchain.color_textures[j]);
	}

	free(images);

	// Create MSAA color array texture for rendering (resolve target is swapchain)
	skr_tex_create(
		tex_format,
		skr_tex_flags_writeable | skr_tex_flags_array,
		(skr_tex_sampler_t){0},
		(skr_vec3i_t){ (int32_t)swapchain_info.width, (int32_t)swapchain_info.height, (int32_t)xr_view_count },
		xr_msaa_samples, 1, NULL,
		&xr_color_msaa );
	skr_tex_set_name(&xr_color_msaa, "XR Color MSAA (Array)");
#endif

	// Depth. Multisampled on the Vulkan path to match the MSAA colour target;
	// single-sampled on the web, where there is no MSAA target to match.
	skr_tex_create(
		skr_tex_fmt_depth16,
		skr_tex_flags_writeable | skr_tex_flags_array,
		(skr_tex_sampler_t){0},
		(skr_vec3i_t){ (int32_t)swapchain_info.width, (int32_t)swapchain_info.height, (int32_t)xr_view_count },
		xr_msaa_samples, 1, NULL,
		&xr_depth_texture );
	skr_tex_set_name(&xr_depth_texture, "XR Depth");

	return true;
}

///////////////////////////////////////////
// Input Actions
///////////////////////////////////////////

void openxr_make_actions(void) {
	XrActionSetCreateInfo actionset_info = { XR_TYPE_ACTION_SET_CREATE_INFO };
	strncpy(actionset_info.actionSetName,          "gameplay", XR_MAX_ACTION_SET_NAME_SIZE);
	strncpy(actionset_info.localizedActionSetName, "Gameplay", XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE);
	xrCreateActionSet(xr_instance, &actionset_info, &xr_input.actionSet);

	xrStringToPath(xr_instance, "/user/hand/left",  &xr_input.handSubactionPath[0]);
	xrStringToPath(xr_instance, "/user/hand/right", &xr_input.handSubactionPath[1]);

	// Hand pose action
	XrActionCreateInfo action_info = { XR_TYPE_ACTION_CREATE_INFO };
	action_info.countSubactionPaths = 2;
	action_info.subactionPaths      = xr_input.handSubactionPath;
	action_info.actionType          = XR_ACTION_TYPE_POSE_INPUT;
	strncpy(action_info.actionName,          "hand_pose", XR_MAX_ACTION_NAME_SIZE);
	strncpy(action_info.localizedActionName, "Hand Pose", XR_MAX_LOCALIZED_ACTION_NAME_SIZE);
	xrCreateAction(xr_input.actionSet, &action_info, &xr_input.poseAction);

	// Select action
	action_info.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
	strncpy(action_info.actionName,          "select", XR_MAX_ACTION_NAME_SIZE);
	strncpy(action_info.localizedActionName, "Select", XR_MAX_LOCALIZED_ACTION_NAME_SIZE);
	xrCreateAction(xr_input.actionSet, &action_info, &xr_input.selectAction);

	// Bind to simple_controller profile
	XrPath profile_path;
	XrPath pose_path[2];
	XrPath select_path[2];
	xrStringToPath(xr_instance, "/user/hand/left/input/grip/pose",     &pose_path[0]);
	xrStringToPath(xr_instance, "/user/hand/right/input/grip/pose",    &pose_path[1]);
	xrStringToPath(xr_instance, "/user/hand/left/input/select/click",  &select_path[0]);
	xrStringToPath(xr_instance, "/user/hand/right/input/select/click", &select_path[1]);
	xrStringToPath(xr_instance, "/interaction_profiles/khr/simple_controller", &profile_path);

	XrActionSuggestedBinding bindings[] = {
		{ xr_input.poseAction,   pose_path[0]   },
		{ xr_input.poseAction,   pose_path[1]   },
		{ xr_input.selectAction, select_path[0] },
		{ xr_input.selectAction, select_path[1] },
	};

	xrSuggestInteractionProfileBindings(xr_instance, &(XrInteractionProfileSuggestedBinding){
		.type                   = XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING,
		.interactionProfile     = profile_path,
		.suggestedBindings      = bindings,
		.countSuggestedBindings = sizeof(bindings) / sizeof(bindings[0]) });

	// Create hand spaces
	for (int32_t i = 0; i < 2; i++) {
		xrCreateActionSpace(xr_session, &(XrActionSpaceCreateInfo){
			.type             = XR_TYPE_ACTION_SPACE_CREATE_INFO,
			.action           = xr_input.poseAction,
			.poseInActionSpace = xr_pose_identity,
			.subactionPath    = xr_input.handSubactionPath[i] },
			&xr_input.handSpace[i]);
	}

	// Attach action set to session
	xrAttachSessionActionSets(xr_session, &(XrSessionActionSetsAttachInfo){
		.type            = XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO,
		.countActionSets = 1,
		.actionSets      = &xr_input.actionSet });
}

///////////////////////////////////////////
// Shutdown
///////////////////////////////////////////

void openxr_shutdown(void) {
	xrDestroySwapchain  (xr_swapchain.handle);
	xr_swapchain_destroy(&xr_swapchain);
#ifndef __EMSCRIPTEN__
	skr_tex_destroy     (&xr_color_msaa);
#endif
	skr_tex_destroy     (&xr_depth_texture);
	free                ( xr_config_views);
	free                ( xr_views);

	// Destroy input resources
	if (xr_input.actionSet != XR_NULL_HANDLE) {
		if (xr_input.handSpace[0] != XR_NULL_HANDLE) xrDestroySpace(xr_input.handSpace[0]);
		if (xr_input.handSpace[1] != XR_NULL_HANDLE) xrDestroySpace(xr_input.handSpace[1]);
		xrDestroyActionSet(xr_input.actionSet);
	}

	if (xr_app_space != XR_NULL_HANDLE) xrDestroySpace(xr_app_space);
	if (xr_session   != XR_NULL_HANDLE) xrDestroySession(xr_session);
#ifndef __EMSCRIPTEN__
	if (xr_debug     != XR_NULL_HANDLE && ext_xrDestroyDebugUtilsMessengerEXT) {
		ext_xrDestroyDebugUtilsMessengerEXT(xr_debug);
	}
#endif
	if (xr_instance  != XR_NULL_HANDLE) xrDestroyInstance(xr_instance);
}

static void xr_swapchain_destroy(xr_swapchain_t* swapchain) {
	for (uint32_t i = 0; i < swapchain->image_count; i++) {
		if (skr_tex_is_valid(&swapchain->color_textures[i])) {
			skr_tex_destroy(&swapchain->color_textures[i]);
		}
	}
	free(swapchain->color_textures);
}

///////////////////////////////////////////
// Event Polling
///////////////////////////////////////////

void openxr_poll_events(bool* out_exit) {
	*out_exit = false;

	XrEventDataBuffer event_buffer = { XR_TYPE_EVENT_DATA_BUFFER };
	while (xrPollEvent(xr_instance, &event_buffer) == XR_SUCCESS) {
		switch (event_buffer.type) {
		case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
			XrEventDataSessionStateChanged* changed = (XrEventDataSessionStateChanged*)&event_buffer;
			xr_session_state = changed->state;

			switch (xr_session_state) {
			case XR_SESSION_STATE_READY:
				xrBeginSession(xr_session, &(XrSessionBeginInfo){
					.type                       = XR_TYPE_SESSION_BEGIN_INFO,
					.primaryViewConfigurationType = xr_config_view });
				xr_running = true;
				break;
			case XR_SESSION_STATE_STOPPING: { xr_running = false; xrEndSession(xr_session); } break;
			case XR_SESSION_STATE_EXITING:
			case XR_SESSION_STATE_LOSS_PENDING: *out_exit = true; break;
			default: break;
			}
		} break;
		case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: *out_exit = true; return;
		default: break;
		}
		event_buffer = (XrEventDataBuffer){ XR_TYPE_EVENT_DATA_BUFFER };
	}
}

///////////////////////////////////////////
// Action Polling
///////////////////////////////////////////

void openxr_poll_actions(void) {
	if (xr_session_state != XR_SESSION_STATE_FOCUSED)
		return;

	XrActiveActionSet action_set = { .actionSet = xr_input.actionSet, .subactionPath = XR_NULL_PATH };
	xrSyncActions(xr_session, &(XrActionsSyncInfo){
		.type                 = XR_TYPE_ACTIONS_SYNC_INFO,
		.countActiveActionSets = 1,
		.activeActionSets      = &action_set });

	for (uint32_t hand = 0; hand < 2; hand++) {
		XrActionStateGetInfo get_info = { XR_TYPE_ACTION_STATE_GET_INFO };
		get_info.subactionPath = xr_input.handSubactionPath[hand];

		// Pose state
		XrActionStatePose pose_state = { XR_TYPE_ACTION_STATE_POSE };
		get_info.action = xr_input.poseAction;
		xrGetActionStatePose(xr_session, &get_info, &pose_state);
		xr_input.renderHand[hand] = pose_state.isActive;

		// Select state
		XrActionStateBoolean select_state = { XR_TYPE_ACTION_STATE_BOOLEAN };
		get_info.action = xr_input.selectAction;
		xrGetActionStateBoolean(xr_session, &get_info, &select_state);
		xr_input.handSelect[hand] = select_state.currentState && select_state.changedSinceLastSync;

		// Update hand pose on select
		if (xr_input.handSelect[hand]) {
			XrSpaceLocation space_location = { XR_TYPE_SPACE_LOCATION };
			XrResult res = xrLocateSpace(xr_input.handSpace[hand], xr_app_space, select_state.lastChangeTime, &space_location);
			if (XR_SUCCEEDED(res) &&
				(space_location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
				(space_location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0) {
				xr_input.handPose[hand] = space_location.pose;
			}
		}
	}
}

void openxr_poll_predicted(XrTime predicted_time) {
	if (xr_session_state != XR_SESSION_STATE_FOCUSED)
		return;

	for (uint32_t i = 0; i < 2; i++) {
		if (!xr_input.renderHand[i])
			continue;
		XrSpaceLocation space_relation = { XR_TYPE_SPACE_LOCATION };
		XrResult res = xrLocateSpace(xr_input.handSpace[i], xr_app_space, predicted_time, &space_relation);
		if (XR_SUCCEEDED(res) &&
			(space_relation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0 &&
			(space_relation.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0) {
			xr_input.handPose[i] = space_relation.pose;
		}
	}
}

///////////////////////////////////////////
// Frame Rendering
///////////////////////////////////////////

void openxr_render_frame(void) {
#ifdef __EMSCRIPTEN__
	// Built on the first frame rather than at init - see openxr_create_swapchain
	// for why the web can't answer either question any earlier.
	if (xr_swapchain.handle == XR_NULL_HANDLE && !openxr_create_swapchain()) {
		ska_log(ska_log_error, "[OpenXR] swapchain creation failed");
		return;
	}
#endif

	XrFrameState frame_state = { XR_TYPE_FRAME_STATE };
	XrResult wait_result = xrWaitFrame(xr_session, &(XrFrameWaitInfo){ .type = XR_TYPE_FRAME_WAIT_INFO }, &frame_state);
	if (XR_FAILED(wait_result)) {
		ska_log(ska_log_error, "[OpenXR] xrWaitFrame failed: %s (%d)", xr_result_to_string(wait_result), wait_result);
		return;
	}

	XrResult begin_result = xrBeginFrame(xr_session, &(XrFrameBeginInfo){ .type = XR_TYPE_FRAME_BEGIN_INFO });
	if (XR_FAILED(begin_result)) {
		ska_log(ska_log_error, "[OpenXR] xrBeginFrame failed: %s (%d)", xr_result_to_string(begin_result), begin_result);
		return;
	}

	openxr_poll_predicted(frame_state.predictedDisplayTime);
	app_xr_update_predicted();

	XrCompositionLayerBaseHeader*     layer      = NULL;
	XrCompositionLayerProjection      layer_proj = { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
	XrCompositionLayerProjectionView* views      = calloc(xr_view_count, sizeof(XrCompositionLayerProjectionView));

	bool should_render = frame_state.shouldRender &&
	                     (xr_session_state == XR_SESSION_STATE_VISIBLE ||
	                      xr_session_state == XR_SESSION_STATE_FOCUSED);

	if (should_render && openxr_render_layer(frame_state.predictedDisplayTime, views, xr_view_count, &layer_proj)) {
		layer = (XrCompositionLayerBaseHeader*)&layer_proj;
	}

	XR_CHECK(xrEndFrame(xr_session, &(XrFrameEndInfo){
		.type                 = XR_TYPE_FRAME_END_INFO,
		.displayTime          = frame_state.predictedDisplayTime,
		.environmentBlendMode = xr_blend,
		.layerCount           = layer ? 1 : 0,
		.layers               = (const XrCompositionLayerBaseHeader* const*)&layer }));

	free(views);
}

static bool openxr_render_layer(XrTime predicted_time, XrCompositionLayerProjectionView* views, uint32_t view_count, XrCompositionLayerProjection* layer) {
	// Locate views
	XrViewState  view_state    = { XR_TYPE_VIEW_STATE };
	uint32_t     located_count = 0;
	XR_CHECK(xrLocateViews(xr_session, &(XrViewLocateInfo){
		.type                  = XR_TYPE_VIEW_LOCATE_INFO,
		.viewConfigurationType = xr_config_view,
		.displayTime           = predicted_time,
		.space                 = xr_app_space },
		&view_state, xr_view_count, &located_count, xr_views));

	// Begin sk_renderer frame
	skr_renderer_frame_begin();

	// Acquire single stereo swapchain image (array texture with both views)
	uint32_t img_idx;
	XR_CHECK(xrAcquireSwapchainImage(xr_swapchain.handle, &(XrSwapchainImageAcquireInfo){ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO }, &img_idx));
	XR_CHECK(xrWaitSwapchainImage   (xr_swapchain.handle, &(XrSwapchainImageWaitInfo){ .type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, .timeout = XR_INFINITE_DURATION }));

	// Set up composition layer views - each view references a different array layer
	for (uint32_t i = 0; i < view_count; i++) {
		views[i].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
		views[i].pose = xr_views[i].pose;
		views[i].fov  = xr_views[i].fov;
		views[i].subImage.swapchain        = xr_swapchain.handle;
		views[i].subImage.imageRect.offset = (XrOffset2Di){ 0, 0 };
		views[i].subImage.imageRect.extent = (XrExtent2Di){ xr_swapchain.width, xr_swapchain.height };
		views[i].subImage.imageArrayIndex  = i;  // Each view uses a different array layer
	}

#ifdef __EMSCRIPTEN__
	// No MSAA here (see xr_msaa_samples), so we render straight into the
	// swapchain texture and there is nothing to resolve into.
	if (xr_swapchain.transient) {
		// Re-read every frame: under XR_SKW_transient_swapchain_images the
		// compositor hands out a *different* texture each frame, and caching the
		// previous one renders into a texture that no longer exists.
		XrSwapchainImageWebGPUSKW image = { XR_TYPE_SWAPCHAIN_IMAGE_WEBGPU_SKW };
		uint32_t got = 0;
		XR_CHECK(xrEnumerateSwapchainImages(xr_swapchain.handle, 1, &got, (XrSwapchainImageBaseHeader*)&image));
		if (got == 0 || image.texture == NULL) {
			ska_log(ska_log_error, "[OpenXR] no transient swapchain texture this frame");
			skr_renderer_frame_end(NULL, 0);
			XR_CHECK(xrReleaseSwapchainImage(xr_swapchain.handle, &(XrSwapchainImageReleaseInfo){ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO }));
			return false;
		}

		skr_tex_t* swap = &xr_swapchain.color_textures[0];
		skr_err_   err  = skr_tex_is_valid(swap)
			? skr_tex_update_external_wgpu(swap, image.texture)
			: skr_tex_create_external_wgpu((skr_tex_external_wgpu_info_t){
				.texture      = image.texture,
				.format       = xr_swapchain_tex_format,
				.sampler      = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp },
				.size         = { xr_swapchain.width, xr_swapchain.height, xr_swapchain.array_size },
				.multisample  = 1,
				.owns_texture = false,
			}, swap);
		if (err != skr_err_success) {
			ska_log(ska_log_error, "[OpenXR] failed to wrap the transient swapchain texture");
			skr_renderer_frame_end(NULL, 0);
			XR_CHECK(xrReleaseSwapchainImage(xr_swapchain.handle, &(XrSwapchainImageReleaseInfo){ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO }));
			return false;
		}
		img_idx = 0;
	}

	app_xr_render_stereo(&xr_swapchain.color_textures[img_idx], NULL, &xr_depth_texture,
		xr_views, view_count, xr_swapchain.width, xr_swapchain.height);
#else
	// Single-pass stereo rendering - render to MSAA, resolve to swapchain
	skr_tex_t* color_target   = &xr_color_msaa;  // MSAA render target
	skr_tex_t* resolve_target = &xr_swapchain.color_textures[img_idx];  // Swapchain for resolve
	skr_tex_t* depth_target   = &xr_depth_texture;  // MSAA depth
	app_xr_render_stereo(color_target, resolve_target, depth_target, xr_views, view_count, xr_swapchain.width, xr_swapchain.height);
#endif

	// End sk_renderer frame - must happen BEFORE releasing swapchain images
	skr_renderer_frame_end(NULL, 0);

	// Release swapchain image
	XR_CHECK(xrReleaseSwapchainImage(xr_swapchain.handle, &(XrSwapchainImageReleaseInfo){ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO }));

	layer->space     = xr_app_space;
	layer->viewCount = view_count;
	layer->views     = views;
	return true;
}

///////////////////////////////////////////
// Main
///////////////////////////////////////////

// Both targets run this exact function; only who calls it differs. It must
// return void, because emscripten_set_main_loop takes a void(*)(void) and wasm
// traps on an indirect call whose signature doesn't match - so the "should we
// keep going" answer leaves through s_running natively, and through
// emscripten_cancel_main_loop on the web, rather than a return value.
#ifndef __EMSCRIPTEN__
static bool s_running = true;
#endif

static void frame(void) {
	bool quit = false;
	openxr_poll_events(&quit);

	if (quit) {
		// Teardown lives here rather than after the loop: on the web
		// emscripten_set_main_loop's simulate_infinite_loop unwinds the stack,
		// so main() never gets to run its own.
#ifndef __EMSCRIPTEN__
		s_running = false;
		vkDeviceWaitIdle(skr_get_vk_device());
#endif
		app_xr_shutdown();
		openxr_shutdown();
		skr_shutdown();
		ska_log(ska_log_info, "Cleanly shut down.");
#ifdef __EMSCRIPTEN__
		emscripten_cancel_main_loop();
#endif
		return;
	}
	if (!xr_running) return;

	openxr_poll_actions();
	app_xr_update();
	openxr_render_frame();

#ifndef __EMSCRIPTEN__
	// Sleep if not visible. Native only - on the web this function returns to
	// the browser every tick, so the page idles politely by construction, and
	// blocking here would just freeze the tab.
	if (xr_session_state != XR_SESSION_STATE_VISIBLE &&
	    xr_session_state != XR_SESSION_STATE_FOCUSED) {
		ska_time_sleep(250);
	}
#endif
}

int main(int argc, char* argv[]) {
	(void)argc;
	(void)argv;

	ska_init(NULL);
	ska_set_cwd(NULL);

	ska_log(ska_log_info, "sk_renderer OpenXR Example");
	ska_log(ska_log_info, "==========================");

	if (!openxr_init("sk_renderer XR Test")) {
		ska_log(ska_log_error, "Failed to initialize OpenXR.");
		ska_log(ska_log_error, "Make sure:");
		ska_log(ska_log_error, "  1. An OpenXR runtime is installed (e.g., Monado, SteamVR)");
		ska_log(ska_log_error, "  2. The runtime is set as active");
		ska_log(ska_log_error, "  3. A VR headset is connected");
		skr_shutdown();
		return 1;
	}

	openxr_make_actions();
	app_xr_init();

#ifdef __EMSCRIPTEN__
	// window.requestAnimationFrame stops firing once a session goes immersive, so
	// an app-owned loop would stop dead in VR. WebOpenXR redirects emscripten's
	// loop onto the session rAF, which needs this call and fps 0; any other value
	// selects setTimeout scheduling, which cannot be redirected.
	emscripten_set_main_loop(frame, 0, 1);   // does not return
#else
	while (s_running) frame();
#endif
	return 0;
}




