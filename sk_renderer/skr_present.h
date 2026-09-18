// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

// Presentation timeline ring embedded in each backend's skr_surface_t; the
// logic is in skr_present.c.

#include "include/sk_renderer.h"
#include <stdint.h>

#define SKR_PRESENT_RING               32
#define SKR_PRESENT_INTERVALS          5       // recent display deltas kept for the measured refresh
#define SKR_PRESENT_ACQUIRE_BLOCKED_NS 200000  // an acquire back faster than this never waited on a vblank

typedef struct skr_present_ring_t {
	skr_present_info_t slots[SKR_PRESENT_RING];       // indexed by id % SKR_PRESENT_RING
	uint64_t           count;                         // presents so far; the next carries count + 1
	uint64_t           drained_id;                    // every id at or below this has been handed out
	uint64_t           last_display_ns;               // of the newest drained slot that had one
	uint64_t           last_vblank_ns;                // skr_surface_set_vblank, so a stamp read twice places nothing
	uint64_t           last_placed_ns;                // where the vblank grid last put a present, the next goes after it
	uint64_t           refresh_hint_ns;               // skr_surface_set_refresh
	uint64_t           refresh_reported_ns;           // from the driver, 0 without present timing
	uint64_t           interval_ns[SKR_PRESENT_INTERVALS];
	int32_t            interval_idx;
	int32_t            interval_count;
} skr_present_ring_t;
