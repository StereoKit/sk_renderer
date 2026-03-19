#pragma once

// sk_renderer uses Volk for Vulkan - must include first before any Vulkan headers
#include <sk_renderer.h>

// Tell OpenXR we're using Vulkan
#define XR_USE_GRAPHICS_API_VULKAN

// Android platform support
#ifdef __ANDROID__
#include <jni.h>
#define XR_USE_PLATFORM_ANDROID
#endif

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <stdbool.h>
#include <stdint.h>

///////////////////////////////////////////
// Types
///////////////////////////////////////////

typedef struct xr_swapchain_t {
	XrSwapchain  handle;
	int32_t      width;
	int32_t      height;
	int32_t      array_size;      // Number of array layers (2 for stereo)
	int32_t      sample_count;    // MSAA sample count
	skr_tex_t*   color_textures;  // Array of wrapped VkImages (each is an array texture for stereo)
	uint32_t     image_count;
} xr_swapchain_t;

typedef struct xr_input_state_t {
	XrActionSet actionSet;
	XrAction    poseAction;
	XrAction    selectAction;
	XrPath      handSubactionPath[2];
	XrSpace     handSpace[2];
	XrPosef     handPose[2];
	XrBool32    renderHand[2];
	XrBool32    handSelect[2];
} xr_input_state_t;

///////////////////////////////////////////
// Globals (defined in main.c)
///////////////////////////////////////////

extern const XrPosef         xr_pose_identity;
extern XrInstance            xr_instance;
extern XrSession             xr_session;
extern XrSessionState        xr_session_state;
extern bool                  xr_running;
extern XrSpace               xr_app_space;
extern XrSystemId            xr_system_id;
extern xr_input_state_t      xr_input;
extern XrEnvironmentBlendMode xr_blend;
extern XrDebugUtilsMessengerEXT xr_debug;

extern XrView*               xr_views;
extern XrViewConfigurationView* xr_config_views;
extern xr_swapchain_t        xr_swapchain;       // Single stereo swapchain (array texture)
extern uint32_t              xr_view_count;

extern skr_tex_t             xr_depth_texture;   // Single array depth texture with MSAA

///////////////////////////////////////////
// Configuration
///////////////////////////////////////////

extern XrFormFactor            xr_config_form;
extern XrViewConfigurationType xr_config_view;

///////////////////////////////////////////
// Function pointers for OpenXR extensions
///////////////////////////////////////////

extern PFN_xrCreateVulkanInstanceKHR           ext_xrCreateVulkanInstanceKHR;
extern PFN_xrCreateVulkanDeviceKHR             ext_xrCreateVulkanDeviceKHR;
extern PFN_xrGetVulkanGraphicsDevice2KHR       ext_xrGetVulkanGraphicsDevice2KHR;
extern PFN_xrGetVulkanGraphicsRequirements2KHR ext_xrGetVulkanGraphicsRequirements2KHR;
extern PFN_xrCreateDebugUtilsMessengerEXT      ext_xrCreateDebugUtilsMessengerEXT;
extern PFN_xrDestroyDebugUtilsMessengerEXT     ext_xrDestroyDebugUtilsMessengerEXT;

///////////////////////////////////////////
// OpenXR Functions
///////////////////////////////////////////

bool openxr_init          (const char* app_name);
void openxr_make_actions  (void);
void openxr_shutdown      (void);
void openxr_poll_events   (bool* out_exit);
void openxr_poll_actions  (void);
void openxr_poll_predicted(XrTime predicted_time);
void openxr_render_frame  (void);
