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
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

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

///////////////////////////////////////////////////////////////////////////////
// Backend-specific surface glue. Vulkan surfaces come from
// ska_vk_create_surface, WebGPU surfaces from ska_wgpu_create_surface. Neither
// Wayland nor WebGPU can report a surface's size, so the drawable size goes to
// skr_surface_create directly.

static bool app_surface_create(ska_window_t* window, skr_surface_t* out_surface) {
	// Read this before creating the surface: on the web, configuring a surface
	// resizes the canvas backing store, which would overwrite what we read
	int32_t w = 0, h = 0;
	ska_window_get_drawable_size(window, &w, &h);

#if defined(SKR_VK)
	VkSurfaceKHR vk_surface;
	if (!ska_vk_create_surface(window, skr_get_vk_instance(), &vk_surface)) {
		su_log(su_log_critical, "Failed to create Vulkan surface: %s", ska_error_get());
		return false;
	}
	if (skr_surface_create(vk_surface, (skr_vec2i_t){ w, h }, out_surface) != skr_err_success) {
		vkDestroySurfaceKHR(skr_get_vk_instance(), vk_surface, NULL);
		return false;
	}
#elif defined(SKR_WEBGPU)
	void* wgpu_surface = NULL;
	if (!ska_wgpu_create_surface(window, skr_get_wgpu_instance(), &wgpu_surface)) {
		su_log(su_log_critical, "Failed to create WebGPU surface: %s", ska_error_get());
		return false;
	}
	skr_surface_create(wgpu_surface, (skr_vec2i_t){ w, h }, out_surface);
#endif
	return skr_surface_is_valid(out_surface);
}

static void app_surface_resize(ska_window_t* window, skr_surface_t* surface) {
	int32_t w = 0, h = 0;
	ska_window_get_drawable_size(window, &w, &h);
	skr_surface_resize(surface, (skr_vec2i_t){ w, h });
}

static void app_device_wait_idle(void) {
#ifdef SKR_VK
	if (skr_get_vk_device()) vkDeviceWaitIdle(skr_get_vk_device());
#else
	skr_future_t f = skr_future_get();
	skr_future_wait(&f);
#endif
}

// Everything the frame callback needs. Static so it outlives main() on the
// web, where ska_run returns immediately and the browser drives the frames.
typedef struct main_loop_t {
	ska_window_t* window;
	app_t*        app;
	skr_surface_t surface;
	int           frame_count;
	bool          running;
	bool          suspended;
	double        last_time;
	uint64_t      last_frame_ns;
	int           test_frames;
	bool          test_all;
	bool          cycle_resolve;
} main_loop_t;

static main_loop_t _loop;

#ifdef __EMSCRIPTEN__
static void _web_exit(void* arg) { (void)arg; exit(0); }
#endif

static void main_shutdown(main_loop_t* s) {
	su_log(su_log_info, "Completed %d frames, shutting down...", s->frame_count);

	// Wait for GPU (native only — the web can't block, and the page teardown
	// collects everything anyway)
#ifndef __EMSCRIPTEN__
	app_device_wait_idle();
#endif

	// Cleanup ImGui
	ImGui_ImplSkRenderer_Shutdown();
	ImGui_ImplSkApp_Shutdown();
	igDestroyContext(NULL);

	// Save window geometry for next session
	{
		skr_recti_t geom;
		ska_window_get_frame_position(s->window, &geom.x, &geom.y);
		ska_window_get_frame_size    (s->window, &geom.w, &geom.h);
		ska_kvpstore_save("window_geometry", &geom, sizeof(geom));
	}

	// Cleanup
	app_destroy(s->app);
	skr_surface_destroy(&s->surface);
	skr_shutdown();
	ska_window_destroy(s->window);
	ska_shutdown();

#ifdef __EMSCRIPTEN__
	// main() returned long ago on the web; report the exit code explicitly so
	// harnesses (emrun --test runs) see a real result. Deferred a tick:
	// exit() unwinds via an exception, which would keep the frame callback
	// from returning false and cancelling the browser's frame loop.
	emscripten_async_call(_web_exit, NULL, 0);
#endif
}

static bool main_frame(void* user_data) {
	main_loop_t* s = (main_loop_t*)user_data;
	s->frame_count++;

	// Alternate normal ↔ each other resolve mode every 4 frames, covering
	// every mode transition — regression for stale cached framebuffers.
	if (s->cycle_resolve) {
		int32_t step = s->frame_count / 4;
		app_set_resolve_mode(s->app, (step & 1) ? (step / 2) % app_resolve_mode_count() : 0);
	}

	// Exit after N frames in test mode, or advance to next scene in testall mode
	if (s->test_frames > 0 && s->frame_count >= s->test_frames) {
		if (s->test_all) {
			int32_t next = app_scene_index(s->app) + 1;
			if (next < app_scene_count(s->app)) {
				app_set_scene(s->app, next);
				s->frame_count = 0;
			} else {
				s->running = false;
			}
		} else {
			s->running = false;
		}
		if (!s->running) { main_shutdown(s); return false; }
	}

	// Handle events
	ska_event_t event;
	while (ska_event_poll(&event)) {
		// Pass event to ImGui first
		ImGui_ImplSkApp_ProcessEvent(&event);

		switch (event.type) {
			case ska_event_quit:
			case ska_event_window_close:
				s->running = false;
				break;

			case ska_event_window_minimized:
				s->suspended = true;
				break;

			case ska_event_window_restored:
				s->suspended = false;
				break;

			case ska_event_app_background:
				su_log(su_log_info, "App entering background - suspending rendering");
				s->suspended = true;
				break;

			case ska_event_app_foreground:
				su_log(su_log_info, "App entering foreground - resuming rendering");
				s->suspended = false;
				break;

			case ska_event_window_hidden:
				// Native window destroyed (screen off, backgrounded).
				// The native surface is now invalid — tear down immediately.
				su_log(su_log_info, "Window hidden — destroying surface");
				app_device_wait_idle();
				skr_surface_destroy(&s->surface);
				break;

			case ska_event_window_shown:
				// New native window available — recreate the surface.
				// Skip the initial shown event: the surface was already
				// created during startup.
				if (skr_surface_is_valid(&s->surface)) break;
				su_log(su_log_info, "Window shown — recreating surface");
				if (!app_surface_create(s->window, &s->surface)) {
					su_log(su_log_critical, "Failed to recreate sk_renderer surface");
					s->running = false;
					break;
				}
				break;

			case ska_event_window_resized:
				app_surface_resize(s->window, &s->surface);
				break;

			default:
				break;
		}
	}
	if (!s->running) { main_shutdown(s); return false; }

	// Skip rendering and updates while suspended (backgrounded/minimized)
	if (s->suspended) {
#ifndef __EMSCRIPTEN__
		ska_time_sleep(100);  // Reduce CPU usage while suspended
#endif
		return true;
	}

	// Skip rendering without a valid native window (screen off)
	if (!skr_surface_is_valid(&s->surface)) {
#ifndef __EMSCRIPTEN__
		ska_time_sleep(16);
#endif
		return true;
	}

	// Calculate delta time
	double current_time = ska_time_get_elapsed_s();
	float  delta_time   = (float)(current_time - s->last_time);
	s->last_time = current_time;

	// Start ImGui frame
	ImGui_ImplSkRenderer_NewFrame();
	ImGui_ImplSkApp_NewFrame();
	igNewFrame();

	skr_renderer_frame_begin();

	app_update      (s->app, delta_time);
	app_render_imgui(s->app, NULL, s->surface.size.x, s->surface.size.y);

	// Finalize ImGui rendering to get draw data
	igRender();

	// Get next swapchain image (vsync blocking happens here via vkAcquireNextImageKHR)
	skr_vec2i_t drawable = {0};
	ska_window_get_drawable_size(s->window, &drawable.x, &drawable.y);
	skr_tex_t*   target         = NULL;
	skr_acquire_ acquire_result = skr_surface_next_tex(&s->surface, drawable, &target);

	// Frame time measured after surface_next_tex (the vsync sync point when GPU-fast)
	uint64_t now_ns     = ska_time_get_elapsed_ns();
	float    frame_time = (float)(now_ns - s->last_frame_ns) / 1000000.0f;
	s->last_frame_ns    = now_ns;
	app_set_frame_time(s->app, frame_time);

	skr_acquire_ surface_result = acquire_result;
	if (acquire_result == skr_acquire_success && target) {
		// Render (ImGui is rendered inside app_render, in the same pass)
		app_render(s->app, target, s->surface.size.x, s->surface.size.y);

		// End frame with surface synchronization
		skr_surface_t* surfaces[] = {&s->surface};
		skr_renderer_frame_end(surfaces, 1);

		// Present
		surface_result = skr_surface_present(&s->surface);
	} else {
		skr_renderer_frame_end(NULL, 0);
	}

	// Handle surface issues from either acquire or present
	if (surface_result != skr_acquire_success) {
		if (!s->running) {
			su_log(su_log_info, "Surface issue during shutdown - exiting gracefully");
			main_shutdown(s);
			return false;
		}
		if (surface_result == skr_acquire_needs_resize) {
			app_surface_resize(s->window, &s->surface);
		}
	}
	return true;
}

int main(int argc, char* argv[]) {
	// Parse command line arguments
	int  test_frames   = 0;   // 0 = run normally, >0 = exit after N frames
	int  start_scene   = -1;  // -1 = use default, >= 0 = start with this scene
	int  resolve_mode  = -1;  // -1 = use default, >= 0 = enum resolve_mode_ index
	int  msaa          = -1;  // -1 = use default, 1 = off, 2/4/8 = sample count
	bool test_all      = false;
	bool cycle_resolve = false;  // Step through every resolve mode transition (stale-cache regression)
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
		} else if (strcmp(argv[i], "-resolve") == 0 && i + 1 < argc) {
			resolve_mode = atoi(argv[++i]);
		} else if (strcmp(argv[i], "-msaa") == 0 && i + 1 < argc) {
			msaa = atoi(argv[++i]);
		} else if (strcmp(argv[i], "-cycleresolve") == 0) {
			cycle_resolve = true;
		}
	}

	// Configuration
	const bool enable_validation = true;

	// Initialize sk_app
	if (!ska_init(NULL)) {
		su_log(su_log_critical, "sk_app initialization failed: %s\n", ska_error_get());
		return 1;
	}

	// Set working directory to executable's path for asset loading
#ifndef __EMSCRIPTEN__
	// Web builds skip this: assets are preloaded into MEMFS at /Assets, which
	// is already relative to the default working directory
	ska_set_cwd(NULL);
#endif

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
	su_log(su_log_info, "Window created");
	if (!window) {
		su_log(su_log_critical, "Failed to create window: %s", ska_error_get());
		ska_shutdown();
		return 1;
	}

	// Get required Vulkan extensions from sk_app (WebGPU needs none)
	uint32_t     extension_count = 0;
	const char** extensions      = NULL;
#ifdef SKR_VK
	extensions = ska_vk_get_instance_extensions(&extension_count);
#endif

	// Initialize sk_renderer
	skr_settings_t settings = {
		.app_name                 = "sk_renderer_test",
		.app_version              = 1,
		.enable_validation        = enable_validation,
		.required_extensions      = extensions,
		.required_extension_count = extension_count,
	};

#if defined(SKR_WEBGPU) && defined(__EMSCRIPTEN__)
	// Web: main() can never block to request a device, so the page's pre-init
	// JS did it before starting the runtime (see web/pre.js); wrap the JS
	// device for the C API and hand both to skr_init's pre-provided path.
	// Probe the JS state first — wrapping a missing device traps in JS.
	int web_has_gpu    = EM_ASM_INT({ return (typeof navigator !== 'undefined' && navigator.gpu) ? 1 : 0; });
	int web_has_device = EM_ASM_INT({ return Module['preinitializedWebGPUDevice'] ? 1 : 0; });
	su_log(su_log_info, "WebGPU: navigator.gpu=%s, pre-initialized device=%s",
		web_has_gpu ? "yes" : "no", web_has_device ? "yes" : "no");
	if (!web_has_device) {
		su_log(su_log_critical, "No pre-initialized WebGPU device — does this browser support WebGPU?");
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}
	settings.wgpu_instance = wgpuCreateInstance(NULL);
	settings.wgpu_device   = emscripten_webgpu_get_device();
#endif

	if (!skr_init(settings)) {
		su_log(su_log_critical, "Failed to initialize sk_renderer!");
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	// Create the rendering surface for the active backend
	if (!app_surface_create(window, &_loop.surface)) {
		su_log(su_log_critical, "Failed to create surface!");
		skr_shutdown();
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	su_log(su_log_info, "sk_renderer initialized successfully!");

#ifdef __EMSCRIPTEN__
	// Drop the shell's loading overlay — rendering is about to start. The
	// shell also hides it via onRuntimeInitialized; this covers runtimes
	// where that Module hook doesn't fire.
	EM_ASM({ var o = document.getElementById('sk-overlay'); if (o) o.classList.add('hidden'); });
#endif

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
		skr_surface_destroy(&_loop.surface);
		skr_shutdown();
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	su_log(su_log_info, "ImGui initialized successfully!");

	// Create application
	app_t* app = app_create(start_scene);
	if (app && resolve_mode >= 0)
		app_set_resolve_mode(app, resolve_mode);
	if (app && msaa >= 1)
		app_set_msaa(app, msaa);
	if (!app) {
		su_log(su_log_critical, "Failed to create application!");
		ImGui_ImplSkRenderer_Shutdown();
		ImGui_ImplSkApp_Shutdown();
		igDestroyContext(NULL);
		skr_surface_destroy(&_loop.surface);
		skr_shutdown();
		ska_window_destroy(window);
		ska_shutdown();
		return 1;
	}

	// Main loop, driven through ska_run: a hand-written `while` on native, but
	// on the web the browser owns the event loop and drives the frame callback
	// via requestAnimationFrame — ska_run doesn't return there, so all loop
	// state lives in _loop and cleanup happens inside the callback.
	_loop.window        = window;
	_loop.app           = app;
	_loop.frame_count   = 0;
	_loop.running       = true;
	_loop.suspended     = false;
	_loop.last_time     = ska_time_get_elapsed_s();
	_loop.last_frame_ns = ska_time_get_elapsed_ns();
	_loop.test_frames   = test_frames;
	_loop.test_all      = test_all;
	_loop.cycle_resolve = cycle_resolve;

	ska_run(main_frame, &_loop);

	return 0;
}
