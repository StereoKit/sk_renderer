// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include "include/sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////
// Internal contract between the backend-independent sources (skr_log.c) and
// the active backend. Each backend implements these allocation wrappers,
// honoring the skr_settings_t allocator callbacks once initialized.
///////////////////////////////////////////////////////////////////////////////

void* _skr_malloc (size_t size);
void* _skr_calloc (size_t count, size_t size);
void* _skr_realloc(void* ptr, size_t size);
void  _skr_free   (void* ptr);

// Presentation timeline ring (skr_present.c), embedded in each backend's surface
skr_present_info_t* _skr_present_begin     (skr_present_ring_t* ref_ring, uint64_t cpu_present_ns);  // Claims the next id
skr_present_info_t* _skr_present_slot      (skr_present_ring_t* ref_ring, uint64_t id);              // NULL once the id has left the ring
void                _skr_present_display   (skr_present_ring_t* ref_ring, uint64_t id, uint64_t display_ns, skr_timing_source_ source);  // Upgrades only
void                _skr_present_vblank    (skr_present_ring_t* ref_ring, uint64_t vblank_ns);  // Places waiting presents on the vblank grid, once per distinct stamp
int32_t             _skr_present_drain     (skr_present_ring_t* ref_ring, uint32_t settle_count, skr_present_info_t* out_infos, int32_t max_count);
uint64_t            _skr_present_refresh_ns(const skr_present_ring_t* ring);
