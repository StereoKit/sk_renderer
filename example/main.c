// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "app.h"
#include "tools/scene_util.h"
#include "imgui_backend/imgui_impl_sk_renderer.h"
#include "imgui_impl_sk_app.h"

#include <sk_app.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// sk_app-based file reader for su_initialize (handles Android assets via ska_asset_read)
static bool _ska_file_read(const char* filename, void** out_data, size_t* out_size, void* user_data) {
	(void)user_data;

	// Try loading as an asset first (handles Android APK assets)
	if (ska_asset_read(filename, out_data, out_size)) {
		return true;
	}

	// Fall back to regular file read
	if (ska_file_read(filename, out_data, out_size)) {
		return true;
	}

	su_log(su_log_critical, "Failed to open file '%s'", filename);
	return false;
}

int main(int argc, char* argv[]) {
	// Parse command line arguments
	int  test_frames   = 0;   // 0 = run normally, >0 = exit after N frames
	int  start_scene   = -1;  // -1 = use default, >= 0 = start with this scene
	bool test_all      = false;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-test") == 0) {
			test_frames = 10;  // Default test mode: 10 frames
		} else if (strcmp(argv[i], "-testall") == 0) {
			test_all    = true;
			test_frames = test_frames > 0 ? test_frames : 10;
		} else if (strcmp(argv[i], "-frames") == 0 && i + 1 < argc) {
			test_frames = atoi(argv[++i]);
		} else if (strcmp(argv[i], "-scene") == 0 && i + 1 < argc) {
			start_scene = atoi(argv[++i]);
		}
	}

	// Configuration
	const bool enable_validation = true;
	const bool enable_offscreen  = false;
	const bool enable_bloom      = true && enable_offscreen;

	// Initialize sk_app
	if (!ska_init(NULL)) {
		su_log(su_log_critical, "sk_app initialization failed: %s\n", ska_error_get());
		return 1;
	}

	// Set working directory to executable's path for asset loading
	ska_set_cwd(NULL);

	// Load saved window geometry (kvpstore so the window starts in the right
	// place instead of centering first and jumping after ImGui loads its ini)
	ska_kvpstore_set_app_name("sk_renderer_test");
	int32_t win_x = SKA_WINDOWPOS_CENTERED;
	int32_t win_y = SKA_WINDOWPOS_CENTERED;
	int32_t win_w = 2560;
	int32_t win_h = 1440;

	skr_recti_t saved_geom = {0};
	size_t geom_size = 0;
	if (ska_kvpstore_load("window_geometry", &saved_geom, sizeof(saved_geom), &geom_size)
		&& geom_size == sizeof(saved_geom)
		&& saved_geom.w > 0 && saved_geom.h > 0) {
		win_x = saved_geom.x;
		win_y = saved_geom.y;
		win_w = saved_geom.w;
		win_h = saved_geom.h;
	}

	// Create window
	ska_window_t* window = NULL;
#ifdef __ANDROID__
	window = ska_window_create("sk_renderer_test",
		SKA_WINDOWPOS_UNDEFINED, SKA_WINDOWPOS_UNDEFINED,
		0, 0,
		ska_window_fullscreen);
#else
	window = ska_window_create("sk_renderer_test",
		win_x, win_y,
		win_w, win_h,
		ska_window_resizable);
#endif
	if (!window) {
		su_log(su_log_critical, "Failed to create window: %s", ska_error_get());
		ska_shutdown();
		return 1;
	}

	// Get required Vulkan extensions from sk_app
	uint32_t     extension_count = 0;
	const char** extensions      = ska_vk_get_instance_extensions(&extension_count);

	// Initialize sk_renderer
	skr_settings_t settings = {
		.app_name                 = "sk_renderer_test",
		.app_version              = 1,
		.enable_validation        = enable_validation,
		.required_extensions      = extensions,
		.required_extension_count = extension_count,
	};

	if (!skr_init(settings)) {
		su_log(su_log_critical, "Failed to initialize sk_renderer!");
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	// Create Vulkan surface
	VkSurfaceKHR vk_surface;
	if (!ska_vk_create_surface(window, skr_get_vk_instance(), &vk_surface)) {
		su_log(su_log_critical, "Failed to create Vulkan surface: %s", ska_error_get());
		skr_shutdown();
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	// Create sk_renderer surface
	skr_surface_t surface;
	skr_surface_create(vk_surface, &surface);
	if (surface.surface == VK_NULL_HANDLE) {
		su_log(su_log_critical, "Failed to create surface!");
		skr_shutdown();
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	su_log(su_log_info, "sk_renderer initialized successfully!");

	// Initialize scene utilities with sk_app file reader (handles Android assets)
	su_initialize(_ska_file_read, NULL);

	// Initialize ImGui
	igCreateContext(NULL);
	ImGuiIO* io = igGetIO();
	io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls

	// Build font atlas at larger size for crisp rendering
	ImFontConfig* font_cfg = ImFontConfig_ImFontConfig();
	font_cfg->SizePixels = 16.0f;  // Larger default font (13px default -> 20px)
	ImFontAtlas_AddFontDefault(io->Fonts, font_cfg);
	ImFontConfig_destroy(font_cfg);

	#if defined(ANDROID)
	ImGuiStyle* style = igGetStyle();
	ImGuiStyle_ScaleAllSizes(style, 2.0f);
	io->FontGlobalScale = 2.0f;
	#endif

	// Initialize ImGui sk_app backend
	ImGui_ImplSkApp_Init(window);

	// Initialize ImGui sk_renderer backend
	if (!ImGui_ImplSkRenderer_Init()) {
		su_log(su_log_critical, "Failed to initialize ImGui sk_renderer backend!");
		ImGui_ImplSkApp_Shutdown();
		igDestroyContext(NULL);
		skr_surface_destroy(&surface);
		skr_shutdown();
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	su_log(su_log_info, "ImGui initialized successfully!");

	// Create application
	app_t* app = app_create(start_scene);
	if (!app) {
		su_log(su_log_critical, "Failed to create application!");
		ImGui_ImplSkRenderer_Shutdown();
		ImGui_ImplSkApp_Shutdown();
		igDestroyContext(NULL);
		skr_surface_destroy(&surface);
		skr_shutdown();
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	// Main loop
	int      frame_count    = 0;
	bool     running        = true;
	bool     suspended      = false;
	double   last_time      = ska_time_get_elapsed_s();
	uint64_t last_frame_ns  = ska_time_get_elapsed_ns();

	while (running) {
		frame_count++;

		// Exit after N frames in test mode, or advance to next scene in testall mode
		if (test_frames > 0 && frame_count >= test_frames) {
			if (test_all) {
				int32_t next = app_scene_index(app) + 1;
				if (next < app_scene_count(app)) {
					app_set_scene(app, next);
					frame_count = 0;
				} else {
					running = false;
					break;
				}
			} else {
				running = false;
				break;
			}
		}

		// Handle events
		ska_event_t event;
		while (ska_event_poll(&event)) {
			// Pass event to ImGui first
			ImGui_ImplSkApp_ProcessEvent(&event);

			switch (event.type) {
				case ska_event_quit:
				case ska_event_window_close:
					running = false;
					break;

				case ska_event_window_minimized:
					suspended = true;
					break;

				case ska_event_window_restored:
					suspended = false;
					break;

				case ska_event_app_background:
					su_log(su_log_info, "App entering background - suspending rendering");
					suspended = true;
					break;

				case ska_event_app_foreground:
					su_log(su_log_info, "App entering foreground - resuming rendering");
					suspended = false;
					break;

				case ska_event_window_hidden:
					// Native window destroyed (screen off, backgrounded).
					// The VkSurfaceKHR is now invalid — tear down immediately.
					su_log(su_log_info, "Window hidden — destroying surface");
					if (skr_get_vk_device()) vkDeviceWaitIdle(skr_get_vk_device());
					skr_surface_destroy(&surface);
					break;

				case ska_event_window_shown:
					// New native window available — recreate Vulkan surface.
					// Skip the initial shown event: the surface was already
					// created during startup.
					if (skr_surface_is_valid(&surface)) break;
					su_log(su_log_info, "Window shown — recreating surface");
					VkSurfaceKHR new_vk_surface;
					if (!ska_vk_create_surface(window, skr_get_vk_instance(), &new_vk_surface)) {
						su_log(su_log_critical, "Failed to recreate Vulkan surface: %s", ska_error_get());
						running = false;
						break;
					}
					skr_surface_create(new_vk_surface, &surface);
					if (!skr_surface_is_valid(&surface)) {
						su_log(su_log_critical, "Failed to recreate sk_renderer surface");
						running = false;
						break;
					}
					break;

				case ska_event_window_resized:
					skr_surface_resize(&surface);
					break;

				default:
					break;
			}
		}

		// Skip rendering and updates while suspended (backgrounded/minimized)
		if (suspended) {
			ska_time_sleep(100);  // Reduce CPU usage while suspended
			continue;
		}

		// Skip rendering without a valid native window (screen off)
		if (!skr_surface_is_valid(&surface)) {
			ska_time_sleep(16);
			continue;
		}

		// Calculate delta time
		double current_time = ska_time_get_elapsed_s();
		float  delta_time   = (float)(current_time - last_time);
		last_time = current_time;

		// Start ImGui frame
		ImGui_ImplSkRenderer_NewFrame();
		ImGui_ImplSkApp_NewFrame();
		igNewFrame();

		skr_renderer_frame_begin();

		app_update      (app, delta_time);
		app_render_imgui(app, NULL, surface.size.x, surface.size.y);

		// Finalize ImGui rendering to get draw data
		igRender();

		// Get next swapchain image (vsync blocking happens here via vkAcquireNextImageKHR)
		skr_tex_t*   target         = NULL;
		skr_acquire_ acquire_result = skr_surface_next_tex(&surface, &target);

		// Frame time measured after surface_next_tex (the vsync sync point when GPU-fast)
		uint64_t now_ns     = ska_time_get_elapsed_ns();
		float    frame_time = (float)(now_ns - last_frame_ns) / 1000000.0f;
		last_frame_ns       = now_ns;
		app_set_frame_time(app, frame_time);

		skr_acquire_ surface_result = acquire_result;
		if (acquire_result == skr_acquire_success && target) {
			// Render (ImGui is rendered inside app_render, in the same pass)
			app_render(app, target, surface.size.x, surface.size.y);

			// End frame with surface synchronization
			skr_surface_t* surfaces[] = {&surface};
			skr_renderer_frame_end(surfaces, 1);

			// Present
			surface_result = skr_surface_present(&surface);
		} else {
			skr_renderer_frame_end(NULL, 0);
		}

		// Handle surface issues from either acquire or present
		if (surface_result != skr_acquire_success) {
			if (!running) {
				su_log(su_log_info, "Surface issue during shutdown - exiting gracefully");
				break;
			}
			if (surface_result == skr_acquire_needs_resize) {
				skr_surface_resize(&surface);
			}
		}
	}

	su_log(su_log_info, "Completed %d frames, shutting down...", frame_count);

	// Wait for GPU
	vkDeviceWaitIdle(skr_get_vk_device());

	// Cleanup ImGui
	ImGui_ImplSkRenderer_Shutdown();
	ImGui_ImplSkApp_Shutdown();
	igDestroyContext(NULL);

	// Save window geometry for next session
	{
		skr_recti_t geom;
		ska_window_get_frame_position(window, &geom.x, &geom.y);
		ska_window_get_frame_size    (window, &geom.w, &geom.h);
		ska_kvpstore_save("window_geometry", &geom, sizeof(geom));
	}

	// Cleanup
	app_destroy(app);
	skr_surface_destroy(&surface);
	skr_shutdown();
	ska_window_destroy(window);
	ska_shutdown();

	return 0;
}
