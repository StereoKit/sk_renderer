// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include <sk_renderer.h>
#include <stdbool.h>
#include <stdint.h>

// app.h includes this for pacer_run_ahead_ and must stay free of sk_app
typedef struct ska_window_t ska_window_t;

// Frame pacing for a window surface: a simulation clock driven by the
// display's vblanks, and a cap on how far the CPU may run ahead of the
// screen. The library reports and waits; this is the policy on top.

typedef enum {
	pacer_run_ahead_full,  // as far as the swapchain allows: most slack, most latency
	pacer_run_ahead_two,   // wait for the present before last: one frame of slack
	pacer_run_ahead_one,   // wait for the last present: lowest latency, no slack
	pacer_run_ahead_max,
} pacer_run_ahead_;

#define PACER_LOG_RING 64

// Indexed by skr_present_mode_ and skr_timing_source_, for the panel, the
// log, and the command line
extern const char* const pacer_present_mode_names[skr_present_mode_max];
extern const char* const pacer_source_names      [skr_timing_source_reported + 1];

typedef struct {
	pacer_run_ahead_ run_ahead;
	bool             log;         // print every drained present record

	// The newest trustworthy display time and its id anchor a prediction for
	// every present after it. Tiers sit on different clocks (a vblank grid
	// placement reads early by the queue depth), so the anchor never drops a tier
	uint64_t           anchor_display_ns;
	uint64_t           anchor_id;
	skr_timing_source_ anchor_source;
	uint64_t predicted_ns;  // display time predicted for the previous frame
	uint64_t refresh_ns;    // as of the last pacer_frame_begin

	// Log bookkeeping, by present id
	uint64_t wait_return_ns[PACER_LOG_RING];  // when the run ahead wait on that id came back
	uint64_t gpu_time_ns   [PACER_LOG_RING];  // GPU time of the frame that fed it
	uint64_t last_display_ns;
} pacer_t;

// Top of the frame, before skr_renderer_frame_begin. Feeds the window's
// refresh rate and last vblank to the surface, blocks per run_ahead, and
// returns the simulation step in seconds. wall_delta_s is only the fallback
// before any display time is known.
float pacer_frame_begin(pacer_t* ref_pacer, skr_surface_t* ref_surface, float wall_delta_s, const ska_window_t* window);

// After skr_surface_present. Drains the surface's present history into
// out_presents, re-anchors the clock from it, and returns the count.
// opt_frame is this frame's skr_renderer_get_frame_timing, for the log.
int32_t pacer_frame_end(pacer_t* ref_pacer, skr_surface_t* ref_surface, const skr_frame_timing_t* opt_frame, skr_present_info_t* out_presents, int32_t max_count);
