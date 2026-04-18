// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include <sk_renderer.h>
#include <stdbool.h>
#include <stdint.h>

// Platform-agnostic application interface
// The app works entirely with sk_renderer abstractions and knows nothing about SDL, surfaces, or windowing

// Application state (opaque to platform layer)
typedef struct app_t app_t;

// Keyboard input
typedef enum {
	app_key_left,
	app_key_right,
} app_key_;

// Lifecycle functions
app_t* app_create (int32_t start_scene);  // start_scene: scene index to start with (-1 for default)
void   app_destroy(app_t* app);

// Scene management
void    app_set_scene  (app_t* app, int32_t scene_index);
int32_t app_scene_index(app_t* app);
int32_t app_scene_count(app_t* app);

// Input handling
void app_key_press     (app_t* app, app_key_ key);
void app_set_frame_time(app_t* app, float frame_time_ms);

// Per-frame functions
void app_update        (app_t* app, float delta_time);
void app_render        (app_t* app, skr_tex_t* render_target, int32_t width, int32_t height);

// ImGui UI building (builds the UI, does NOT render)
void app_render_imgui(app_t* app, skr_tex_t* render_target, int32_t width, int32_t height);
