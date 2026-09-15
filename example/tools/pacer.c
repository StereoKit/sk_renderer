// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "pacer.h"
#include "scene_util.h"

#include <sk_app.h>
#include <math.h>

// A present that hasn't shown after this long is not coming; don't hang the loop on it
#define PACER_WAIT_TIMEOUT_NS 100000000ull

// A re-anchored prediction can move back by a fraction of a refresh; the step
// never goes below this so animation doesn't freeze. There is no upper bound:
// a stall advances time by however long the display went without a new frame.
#define PACER_STEP_MIN_REFRESHES 0.5

const char* const pacer_present_mode_names[skr_present_mode_max]           = { "default", "fifo", "fifo_relaxed", "mailbox", "immediate" };
const char* const pacer_source_names      [skr_timing_source_reported + 1] = { "none", "coarse", "vblank", "estimate", "reported" };

static uint32_t _pacer_ahead(pacer_run_ahead_ run_ahead) {
	switch (run_ahead) {
	case pacer_run_ahead_one: return 1;
	case pacer_run_ahead_two: return 2;
	default:                  return 0;
	}
}

float pacer_frame_begin(pacer_t* ref_pacer, skr_surface_t* ref_surface, float wall_delta_s, const ska_window_t* window) {
	// Re-sent every frame since the window can move to a monitor with a
	// different rate; the vblank crosses from sk_app's clock as an age
	float    refresh_hz = ska_window_get_refresh_rate(window);
	uint64_t vblank_ns  = ska_window_get_vblank_ns   (window);
	skr_surface_set_refresh(ref_surface, refresh_hz > 0.0f ? (uint64_t)(1e9 / refresh_hz) : 0);
	skr_surface_set_vblank (ref_surface, vblank_ns ? skr_time_now_ns() - (ska_time_get_elapsed_ns() - vblank_ns) : 0);
	uint64_t refresh_ns   = skr_surface_get_refresh_ns(ref_surface);
	uint64_t next_id      = skr_surface_get_next_id   (ref_surface);
	ref_pacer->refresh_ns = refresh_ns;

	// The step is the gap between this frame's predicted vblank and the last
	// one's, so animation moves exactly as far as the display will show it
	// moving. Whole refreshes of wall clock is the fallback before any display
	// time is known, and what the coarse tier gets.
	uint64_t predicted = ref_pacer->anchor_id
		? ref_pacer->anchor_display_ns + (next_id - ref_pacer->anchor_id) * refresh_ns
		: 0;
	double step_ns;
	if (predicted && ref_pacer->predicted_ns && predicted > ref_pacer->predicted_ns) {
		step_ns = (double)(predicted - ref_pacer->predicted_ns);
		if (step_ns < PACER_STEP_MIN_REFRESHES * refresh_ns) step_ns = PACER_STEP_MIN_REFRESHES * refresh_ns;
	} else {
		double steps = round((double)wall_delta_s * 1e9 / (double)refresh_ns);
		step_ns = (steps < 1.0 ? 1.0 : steps) * (double)refresh_ns;
	}
	ref_pacer->predicted_ns = predicted;

	// Run ahead: how many presents may still be waiting to show when this
	// frame starts. Waiting on the one before that caps the queue depth.
	uint32_t ahead = _pacer_ahead(ref_pacer->run_ahead);
	if (ahead != 0 && next_id > ahead) {
		skr_surface_wait_present(ref_surface, next_id - ahead, PACER_WAIT_TIMEOUT_NS);
		ref_pacer->wait_return_ns[(next_id - ahead) % PACER_LOG_RING] = skr_time_now_ns();
	}
	return (float)(step_ns / 1e9);
}

static void _pacer_log(pacer_t* ref_pacer, const skr_present_info_t* presents, int32_t count) {
	for (int32_t i = 0; i < count; i++) {
		const skr_present_info_t* p    = &presents[i];
		uint64_t                  wait = ref_pacer->wait_return_ns[p->id % PACER_LOG_RING];
		su_log(su_log_info, "present %llu %-8s interval %6.2f ms  latency %6.2f ms  gpu_done %+6.2f ms  gpu %5.2f ms  wait_lag %+6.2f ms",
			(unsigned long long)p->id, pacer_source_names[p->source],
			p->display_ns && ref_pacer->last_display_ns ? (p->display_ns - ref_pacer->last_display_ns) / 1e6 : 0.0,
			p->display_ns  ? ((double)p->display_ns  - (double)p->cpu_present_ns) / 1e6 : 0.0,
			p->gpu_done_ns ? ((double)p->gpu_done_ns - (double)p->cpu_present_ns) / 1e6 : 0.0,
			ref_pacer->gpu_time_ns[p->id % PACER_LOG_RING] / 1e6,
			wait && p->display_ns ? ((double)wait - (double)p->display_ns) / 1e6 : 0.0);
		if (p->display_ns) ref_pacer->last_display_ns = p->display_ns;
	}
}

int32_t pacer_frame_end(pacer_t* ref_pacer, skr_surface_t* ref_surface, const skr_frame_timing_t* opt_frame, skr_present_info_t* out_presents, int32_t max_count) {
	if (opt_frame && opt_frame->present_id)
		ref_pacer->gpu_time_ns[opt_frame->present_id % PACER_LOG_RING] = opt_frame->gpu_time_ns;

	int32_t count = skr_surface_present_history(ref_surface, out_presents, max_count);
	for (int32_t i = 0; i < count; i++) {
		const skr_present_info_t* p = &out_presents[i];
		if (p->source < skr_timing_source_vblank || p->source < ref_pacer->anchor_source || p->id <= ref_pacer->anchor_id) continue;
		ref_pacer->anchor_id         = p->id;
		ref_pacer->anchor_display_ns = p->display_ns;
		ref_pacer->anchor_source     = p->source;
	}
	if (ref_pacer->log) _pacer_log(ref_pacer, out_presents, count);
	return count;
}
