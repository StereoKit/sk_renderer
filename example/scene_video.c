// SPDX-License-Identifier: MIT
// Video playback scene

#include "app.h"
#include "scene.h"
#include "tools/scene_util.h"
#include "tools/video.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <threads.h>
#include <stdatomic.h>

#include <vulkan/vulkan.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

///////////////////////////////////////////////////////////////////////////////
// Video worker thread - handles all FFmpeg operations
///////////////////////////////////////////////////////////////////////////////

typedef struct {
	// Thread control
	thrd_t       thread;
	atomic_bool  running;
	atomic_bool  shutdown;

	// Commands from main thread (atomics for lock-free communication)
	atomic_bool  cmd_open;        // Request to open new video
	atomic_bool  cmd_seek;        // Request to seek
	atomic_bool  playing;         // Playback state

	// Command parameters (written by main, read by worker)
	char*        open_path;       // Path to open (main thread sets, worker reads and frees)
	mtx_t        open_path_mutex; // Protects open_path
	double       seek_target;     // Target seek time
	double       playback_time;   // Current playback position (shared)

	// Video state (owned by worker thread)
	video_t*     video;

	// Status (written by worker, read by main)
	atomic_bool  loading;         // Currently loading
	atomic_bool  seeking;         // Currently seeking
	atomic_bool  video_ready;     // Video is loaded and ready
} video_worker_t;

static int _video_worker_thread(void* arg) {
	video_worker_t* w = arg;

	// Initialize sk_renderer's per-thread command pool
	skr_thread_init();

	while (!atomic_load(&w->shutdown)) {
		// Check for open command
		if (atomic_load(&w->cmd_open)) {
			atomic_store(&w->cmd_open, false);
			atomic_store(&w->loading, true);
			atomic_store(&w->video_ready, false);

			// Get path (protected by mutex)
			mtx_lock(&w->open_path_mutex);
			char* path = w->open_path;
			w->open_path = NULL;
			mtx_unlock(&w->open_path_mutex);

			if (path) {
				// Destroy old video
				if (w->video) {
					video_destroy(w->video);
					w->video = NULL;
				}

				// Open new video
				w->video = video_open(path);
				free(path);

				if (w->video && video_is_valid(w->video)) {
					atomic_store(&w->video_ready, true);
					w->playback_time = 0.0;
					printf("[video] Loaded successfully\n");
				} else {
					printf("[video] Failed to load\n");
				}
			}
			atomic_store(&w->loading, false);
		}

		// Check for seek command
		if (atomic_load(&w->cmd_seek)) {
			atomic_store(&w->cmd_seek, false);
			atomic_store(&w->seeking, true);

			if (w->video && video_is_valid(w->video)) {
				double target = w->seek_target;
				video_seek(w->video, target);
				w->playback_time = target;
			}

			atomic_store(&w->seeking, false);
		}

		// Decode frames if playing
		if (atomic_load(&w->playing) && w->video && video_is_valid(w->video) &&
		    !atomic_load(&w->loading) && !atomic_load(&w->seeking)) {

			bool is_live = video_is_live(w->video);

			if (is_live) {
				// Live streams: just decode whatever frame is available
				video_decode_next_frame(w->video);
			} else {
				// VOD: decode frames to catch up to playback time
				double video_time = video_get_current_time(w->video);
				double target_time = w->playback_time;

				// Decode up to 3 frames per iteration to catch up
				for (int i = 0; i < 3 && video_time < target_time; i++) {
					if (!video_decode_next_frame(w->video)) {
						// EOF
						break;
					}
					video_time = video_get_current_time(w->video);
				}
			}
		}

		// Small sleep to avoid spinning (1ms)
		struct timespec ts = {0, 1000000};
		thrd_sleep(&ts, NULL);
	}

	// Note: Video is destroyed by scene_video_destroy after GPU idle
	skr_thread_shutdown();
	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Scene
///////////////////////////////////////////////////////////////////////////////

typedef struct {
	scene_t        base;

	video_worker_t worker;
	char*          video_path;  // Display path for UI

	skr_mesh_t     quad;
	skr_shader_t   shader;
	skr_material_t material;

	bool           loop;
	float          seek_slider;
	double         last_duration;  // Cached duration for EOF detection
} scene_video_t;

static void _worker_open(video_worker_t* w, const char* path) {
	mtx_lock(&w->open_path_mutex);
	if (w->open_path) free(w->open_path);
	w->open_path = strdup(path);
	mtx_unlock(&w->open_path_mutex);
	atomic_store(&w->cmd_open, true);
}

static void _worker_seek(video_worker_t* w, double time) {
	w->seek_target = time;
	atomic_store(&w->cmd_seek, true);
}

static scene_t* _scene_video_create(void) {
	scene_video_t* scene = calloc(1, sizeof(scene_video_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_video_t);
	scene->loop      = true;

	// Initialize worker
	video_worker_t* w = &scene->worker;
	atomic_store(&w->running, false);
	atomic_store(&w->shutdown, false);
	atomic_store(&w->cmd_open, false);
	atomic_store(&w->cmd_seek, false);
	atomic_store(&w->playing, true);
	atomic_store(&w->loading, false);
	atomic_store(&w->seeking, false);
	atomic_store(&w->video_ready, false);
	mtx_init(&w->open_path_mutex, mtx_plain);

	// Start worker thread
	if (thrd_create(&w->thread, _video_worker_thread, w) == thrd_success) {
		atomic_store(&w->running, true);
	}

	// Create fullscreen quad
	scene->quad = su_mesh_create_fullscreen_quad();
	skr_mesh_set_name(&scene->quad, "video_quad");

	// Load video shader
	scene->shader = su_shader_load("shaders/video.hlsl.sks", "video");

	// Create material
	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.cull       = skr_cull_none,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_always,  // Fullscreen, no depth
	}, &scene->material);

	// Start loading default video
	const char* default_video = "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4";
	scene->video_path = strdup(default_video);
	_worker_open(&scene->worker, default_video);

	return (scene_t*)scene;
}

static void _scene_video_destroy(scene_t* base) {
	scene_video_t* scene = (scene_video_t*)base;
	video_worker_t* w = &scene->worker;

	// Signal shutdown and wait for worker to stop submitting new work
	atomic_store(&w->shutdown, true);
	if (atomic_load(&w->running)) {
		thrd_join(w->thread, NULL);
	}

	// Now safe to destroy video (worker has exited, GPU is idle)
	if (w->video) {
		video_destroy(w->video);
		w->video = NULL;
	}

	// Cleanup worker resources
	mtx_destroy(&w->open_path_mutex);
	if (w->open_path) free(w->open_path);

	if (scene->video_path) free(scene->video_path);

	skr_material_destroy(&scene->material);
	skr_shader_destroy(&scene->shader);
	skr_mesh_destroy(&scene->quad);

	free(scene);
}

static void _scene_video_update(scene_t* base, float delta_time) {
	scene_video_t* scene = (scene_video_t*)base;
	video_worker_t* w = &scene->worker;

	if (!atomic_load(&w->video_ready)) return;
	if (atomic_load(&w->loading) || atomic_load(&w->seeking)) return;

	// Update playback time (main thread advances, worker reads)
	if (atomic_load(&w->playing) && w->video && !video_is_live(w->video)) {
		w->playback_time += delta_time;

		// Update seek slider
		double duration = video_get_duration(w->video);
		if (duration > 0) {
			scene->seek_slider = (float)(w->playback_time / duration);
			scene->last_duration = duration;

			// Check for EOF and loop
			if (w->playback_time >= duration) {
				if (scene->loop && video_is_seekable(w->video)) {
					_worker_seek(w, 0.0);
					scene->seek_slider = 0.0f;
				} else {
					atomic_store(&w->playing, false);
				}
			}
		}
	}
}

static void _scene_video_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_video_t*  scene = (scene_video_t*)base;
	video_worker_t* w     = &scene->worker;
	(void)ref_system_buffer;

	if (!atomic_load(&w->video_ready) || !w->video) return;

	// Bind video textures to material
	skr_tex_t* tex_y  = video_get_tex_y(w->video);
	skr_tex_t* tex_uv = video_get_tex_uv(w->video);

	if (tex_y && tex_uv && skr_tex_is_valid(tex_y) && skr_tex_is_valid(tex_uv)) {
		skr_material_set_tex(&scene->material, "tex_y",  tex_y);
		skr_material_set_tex(&scene->material, "tex_uv", tex_uv);
	} else {
		return;  // Textures not ready yet
	}

	// Calculate aspect ratio scaling to fit video in screen
	float video_w = (float)video_get_width(w->video);
	float video_h = (float)video_get_height(w->video);
	float screen_w = (float)width;
	float screen_h = (float)height;

	float video_aspect  = video_w / video_h;
	float screen_aspect = screen_w / screen_h;

	float scale_x = 1.0f;
	float scale_y = 1.0f;

	if (video_aspect > screen_aspect) {
		// Video is wider than screen - fit to width, letterbox top/bottom
		scale_y = screen_aspect / video_aspect;
	} else {
		// Video is taller than screen - fit to height, pillarbox left/right
		scale_x = video_aspect / screen_aspect;
	}

	// Scale the quad (NDC space: -1 to 1)
	float4x4 world = float4x4_s((float3){scale_x, scale_y, 1.0f});

	skr_render_list_add(ref_render_list, &scene->quad, &scene->material, &world, sizeof(float4x4), 1);
}

static void _scene_video_render_ui(scene_t* base) {
	scene_video_t* scene = (scene_video_t*)base;
	video_worker_t* w = &scene->worker;

	igText("Video Playback");
	igSeparator();

	bool is_loading = atomic_load(&w->loading);
	bool is_seeking = atomic_load(&w->seeking);
	bool is_ready   = atomic_load(&w->video_ready);
	bool is_playing = atomic_load(&w->playing);

	// File loading
	if (su_file_dialog_supported()) {
		if (is_loading) igBeginDisabled(true);
		if (igButton("Open Video...", (ImVec2){0, 0})) {
			char* path = su_file_dialog_open("Open Video", "Video Files", "mp4;mkv;webm;avi;mov");
			if (path) {
				if (scene->video_path) free(scene->video_path);
				scene->video_path = path;
				_worker_open(w, path);
			}
		}
		if (is_loading) igEndDisabled();
	}

	// Show loading state
	if (is_loading) {
		igText("Loading: %s", scene->video_path ? scene->video_path : "...");
		igText("Please wait...");
		return;
	}

	if (is_ready && w->video) {
		bool is_live     = video_is_live(w->video);
		bool is_seekable = video_is_seekable(w->video);

		// Video info
		igText("File: %s", scene->video_path ? scene->video_path : "(none)");
		igText("Resolution: %dx%d", video_get_width(w->video), video_get_height(w->video));
		if (is_live) {
			igText("Duration: Live");
		} else {
			igText("Duration: %.1fs", video_get_duration(w->video));
		}
		igText("FPS: %.2f", video_get_framerate(w->video));
		igText("HW Accel: %s", video_is_hw_accelerated(w->video) ? "Yes (Vulkan)" : "No (Software)");
		if (is_seeking) {
			igText("Seeking...");
		}

		igSeparator();

		// Playback controls
		if (igButton(is_playing ? "Pause" : "Play", (ImVec2){0, 0})) {
			atomic_store(&w->playing, !is_playing);
		}

		// Restart only makes sense for seekable streams
		if (is_seekable) {
			igSameLine(0, -1);
			if (igButton("Restart", (ImVec2){0, 0})) {
				_worker_seek(w, 0.0);
				scene->seek_slider = 0.0f;
			}
		}

		// Loop only makes sense for non-live streams
		if (!is_live) {
			igCheckbox("Loop", &scene->loop);
		}

		// Seek slider (only for seekable streams)
		if (is_seekable) {
			float old_seek = scene->seek_slider;
			if (igSliderFloat("Position", &scene->seek_slider, 0.0f, 1.0f, "%.2f", 0)) {
				if (scene->seek_slider != old_seek) {
					double new_time = scene->seek_slider * video_get_duration(w->video);
					_worker_seek(w, new_time);
				}
			}
		}

		// Current time display
		if (is_live) {
			igText("Time: %.2f (live)", video_get_current_time(w->video));
		} else {
			igText("Time: %.2f / %.2f", w->playback_time, video_get_duration(w->video));
		}
	} else {
		igText("No video loaded");
		igText("Use 'Open Video...' to load a video file");
	}
}

const scene_vtable_t scene_video_vtable = {
	.name       = "Video Player",
	.create     = _scene_video_create,
	.destroy    = _scene_video_destroy,
	.update     = _scene_video_update,
	.render     = _scene_video_render,
	.get_camera = NULL,
	.render_ui  = _scene_video_render_ui,
};
