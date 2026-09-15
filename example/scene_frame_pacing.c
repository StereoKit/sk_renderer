// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <sk_app.h>
#include <stdlib.h>
#include <math.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// Frame pacing scene: a scrolling picket fence, a moving marker, and a per
// frame color strip and counter. A repeated or skipped display frame shows as
// a hitch in the fence, a hop in the marker, or a gap in a video capture.
// The unpaced toggle swaps the display driven step the app hands in for the
// wall clock gap between updates, which is what the loop used to feed.

// Matches Inst in frame_pacing.hlsl
typedef struct {
	float    scroll_px;
	float    period_px;
	float    bar_px;
	float    marker_px;
	uint32_t frame_index;
	uint32_t _pad[3];
} pacing_inst_t;

#define MARKER_WIDTH_PX 32

typedef struct {
	scene_t        base;
	skr_mesh_t     quad;
	skr_shader_t   shader;
	skr_material_t material;

	float    scroll_px;
	float    marker_px;
	uint32_t frame_index;
	uint64_t last_update_ns;  // for the wall clock step; 0 before the first update
	float    paced_step_px;   // both steps are computed every frame, so the panel can show them side by side
	float    wall_step_px;

	float   px_per_frame;  // speed at the display's refresh rate
	bool    unpaced;       // advance by the wall clock instead of the display
	int32_t period_px;
	int32_t bar_px;

	// Deliberate CPU stalls, to see what a real hitch looks like on this display
	int32_t hitch_every;   // frames, 0 = off
	float   hitch_ms;
} scene_frame_pacing_t;

static scene_t* _scene_frame_pacing_create(void) {
	scene_frame_pacing_t* scene = calloc(1, sizeof(scene_frame_pacing_t));
	if (!scene) return NULL;
	scene->base.size = sizeof(scene_frame_pacing_t);

	scene->px_per_frame = 8.0f;
	scene->period_px    = 64;
	scene->bar_px       = 32;
	scene->hitch_ms     = 30.0f;

	scene->quad   = su_mesh_create_fullscreen_quad();
	scene->shader = su_shader_load("shaders/frame_pacing.hlsl.sks", "frame_pacing");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.cull       = skr_cull_none,
		.depth_test = skr_compare_always,
		.write_mask = skr_write_default,
	}, &scene->material);

	return (scene_t*)scene;
}

static void _scene_frame_pacing_destroy(scene_t* base) {
	scene_frame_pacing_t* scene = (scene_frame_pacing_t*)base;
	skr_material_destroy(&scene->material);
	skr_shader_destroy  (&scene->shader);
	skr_mesh_destroy    (&scene->quad);
	free(scene);
}

static void _spin_ms(float ms) {
	uint64_t end = ska_time_get_elapsed_ns() + (uint64_t)(ms * 1000000.0f);
	while (ska_time_get_elapsed_ns() < end) {}
}

// Turning pixels per frame into pixels per second needs a rate; 60 stands in
// where the platform can't say.
static float _refresh_hz(const scene_t* base) {
	return base->refresh_hz > 0.0f ? base->refresh_hz : 60.0f;
}

static void _scene_frame_pacing_update(scene_t* base, float delta_time) {
	scene_frame_pacing_t* scene = (scene_frame_pacing_t*)base;
	scene->frame_index++;

	uint64_t now     = ska_time_get_elapsed_ns();
	float    wall_dt = scene->last_update_ns ? (float)((double)(now - scene->last_update_ns) / 1e9) : delta_time;
	float    px_s    = scene->px_per_frame * _refresh_hz(base);
	scene->last_update_ns = now;
	scene->paced_step_px  = px_s * delta_time;
	scene->wall_step_px   = px_s * wall_dt;

	float step = scene->unpaced ? scene->wall_step_px : scene->paced_step_px;
	scene->scroll_px  = fmodf(scene->scroll_px + step, (float)scene->period_px);
	scene->marker_px += step;

	if (scene->hitch_every > 0 && scene->frame_index % (uint32_t)scene->hitch_every == 0)
		_spin_ms(scene->hitch_ms);
}

static void _scene_frame_pacing_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_frame_pacing_t* scene = (scene_frame_pacing_t*)base;
	(void)height; (void)ref_system_buffer;

	float track = (float)(width - 24 - MARKER_WIDTH_PX);
	if (track > 0.0f) scene->marker_px = fmodf(scene->marker_px, track);

	pacing_inst_t inst = {
		.scroll_px   = scene->scroll_px,
		.period_px   = (float)scene->period_px,
		.bar_px      = (float)scene->bar_px,
		.marker_px   = 24.0f + scene->marker_px,
		.frame_index = scene->frame_index,
	};
	skr_render_list_add(ref_render_list, &scene->quad, &scene->material, &inst, sizeof(inst), 1);
}

static void _scene_frame_pacing_render_ui(scene_t* base) {
	scene_frame_pacing_t* scene = (scene_frame_pacing_t*)base;

	igText("Frame %u, step paced %.2f px, wall clock %.2f px", scene->frame_index, scene->paced_step_px, scene->wall_step_px);
	igText("Fence: %.0f px/s at %.2f Hz%s", scene->px_per_frame * _refresh_hz(base), _refresh_hz(base),
		base->refresh_hz > 0.0f ? "" : " (assumed, window reports none)");

	igCheckbox("Unpaced (wall clock step, as before the pacer)", &scene->unpaced);
	igSliderFloat("Pixels / frame", &scene->px_per_frame, 1.0f, 32.0f, "%.0f", 0);
	igSliderInt  ("Bar period",     &scene->period_px,    8, 256, "%d px", 0);
	igSliderInt  ("Bar width",      &scene->bar_px,       1, scene->period_px, "%d px", 0);

	igSeparator();
	igText("Inject CPU hitch");
	igSliderInt  ("Every N frames", &scene->hitch_every, 0, 240, scene->hitch_every ? "%d" : "off", 0);
	igSliderFloat("Stall",          &scene->hitch_ms,    1.0f, 100.0f, "%.0f ms", 0);
}

const scene_vtable_t scene_frame_pacing_vtable = {
	.name       = "Frame Pacing (Picket Fence)",
	.create     = _scene_frame_pacing_create,
	.destroy    = _scene_frame_pacing_destroy,
	.update     = _scene_frame_pacing_update,
	.render     = _scene_frame_pacing_render,
	.get_camera = NULL,
	.render_ui  = _scene_frame_pacing_render_ui,
};
