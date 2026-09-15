// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_skr_shared.h"

#define SKR_PRESENT_INTERVALS_MIN      4        // a vote of fewer is noise
#define SKR_PRESENT_REFRESH_DEFAULT_NS 16666667 // 60 Hz, until anything is known
#define SKR_PRESENT_VBLANK_DEADLINE    4        // a present within a refresh/4 of a vblank misses it

///////////////////////////////////////////////////////////////////////////////
// Presentation timeline ring, shared by both backends. Each backend claims a
// slot per present and feeds display times in from whatever signals it has;
// this file decides which signal wins and when a slot is finished.
///////////////////////////////////////////////////////////////////////////////

skr_present_info_t* _skr_present_slot(skr_present_ring_t* ref_ring, uint64_t id) {
	if (id == 0 || id > ref_ring->count) return NULL;
	skr_present_info_t* slot = &ref_ring->slots[id % SKR_PRESENT_RING];
	return slot->id == id ? slot : NULL;
}

skr_present_info_t* _skr_present_begin(skr_present_ring_t* ref_ring, uint64_t cpu_present_ns) {
	uint64_t            id   = ++ref_ring->count;
	skr_present_info_t* slot = &ref_ring->slots[id % SKR_PRESENT_RING];
	*slot = (skr_present_info_t){
		.id             = id,
		.cpu_present_ns = cpu_present_ns,
	};
	// An undrained slot this overwrote is gone; don't let the drain stall on it
	if (ref_ring->drained_id + SKR_PRESENT_RING < id)
		ref_ring->drained_id = id - SKR_PRESENT_RING;
	return slot;
}

void _skr_present_display(skr_present_ring_t* ref_ring, uint64_t id, uint64_t display_ns, skr_timing_source_ source) {
	skr_present_info_t* slot = _skr_present_slot(ref_ring, id);
	if (slot == NULL || slot->source >= source) return;
	slot->display_ns = display_ns;
	slot->source     = source;
}

// FIFO shows one present per vblank, in order: each waiting present goes to the
// first grid vblank after both its submission and the previous placement
void _skr_present_vblank(skr_present_ring_t* ref_ring, uint64_t vblank_ns) {
	// The same vblank comes back every frame, and a stamp crossed from another
	// clock never repeats exactly, so anything short of half a refresh newer
	// than the last is that same vblank
	uint64_t refresh_ns = _skr_present_refresh_ns(ref_ring);
	if (vblank_ns == 0 || vblank_ns < ref_ring->last_vblank_ns + refresh_ns / 2) return;
	ref_ring->last_vblank_ns = vblank_ns;
	for (uint64_t id = ref_ring->drained_id + 1; id <= ref_ring->count; id++) {
		skr_present_info_t* slot = _skr_present_slot(ref_ring, id);
		if (slot == NULL || slot->source >= skr_timing_source_vblank) continue;
		uint64_t ready_ns = slot->cpu_present_ns + refresh_ns / SKR_PRESENT_VBLANK_DEADLINE;
		if (ready_ns >= vblank_ns) return;
		uint64_t at = vblank_ns - ((vblank_ns - ready_ns) / refresh_ns) * refresh_ns;
		if (at <= ready_ns)                 at += refresh_ns;
		if (at <= ref_ring->last_placed_ns) at  = ref_ring->last_placed_ns + refresh_ns;
		if (at > vblank_ns) return;
		_skr_present_display(ref_ring, id, at, skr_timing_source_vblank);
		ref_ring->last_placed_ns = at;
	}
}

// A slot is finished once the engine has reported on it, or once enough later
// presents have gone by that no more signals can arrive for it.
int32_t _skr_present_drain(skr_present_ring_t* ref_ring, uint32_t settle_count, skr_present_info_t* out_infos, int32_t max_count) {
	int32_t count = 0;
	while (count < max_count && ref_ring->drained_id < ref_ring->count) {
		uint64_t            id   = ref_ring->drained_id + 1;
		skr_present_info_t* slot = _skr_present_slot(ref_ring, id);
		if (slot == NULL) { ref_ring->drained_id = id; continue; }
		if (slot->source < skr_timing_source_reported && id + settle_count > ref_ring->count) break;

		if (slot->display_ns != 0) {
			if (ref_ring->last_display_ns != 0 && slot->display_ns > ref_ring->last_display_ns) {
				ref_ring->interval_ns[ref_ring->interval_idx] = slot->display_ns - ref_ring->last_display_ns;
				ref_ring->interval_idx = (ref_ring->interval_idx + 1) % SKR_PRESENT_INTERVALS;
				if (ref_ring->interval_count < SKR_PRESENT_INTERVALS) ref_ring->interval_count++;
			}
			ref_ring->last_display_ns = slot->display_ns;
		}
		out_infos[count++]   = *slot;
		ref_ring->drained_id = id;
	}
	return count;
}

// The recent display interval with the most neighbours within a percent: a
// duplicated frame or a short coarse sample is outvoted, and ties go to the
// shortest so equals don't flicker. Matches sk_app's web refresh estimate.
static uint64_t _skr_present_measured_ns(const skr_present_ring_t* ring) {
	if (ring->interval_count < SKR_PRESENT_INTERVALS_MIN) return 0;
	uint64_t best  = 0;
	int32_t  votes = 0;
	for (int32_t i = 0; i < ring->interval_count; i++) {
		uint64_t v   = ring->interval_ns[i];
		uint64_t tol = v / 100;
		int32_t  n   = 0;
		for (int32_t j = 0; j < ring->interval_count; j++) {
			uint64_t w = ring->interval_ns[j];
			if (w + tol >= v && w <= v + tol) n++;
		}
		if (n > votes || (n == votes && v < best)) { votes = n; best = v; }
	}
	return best;
}

uint64_t _skr_present_refresh_ns(const skr_present_ring_t* ring) {
	if (ring->refresh_reported_ns) return ring->refresh_reported_ns;
	if (ring->refresh_hint_ns)     return ring->refresh_hint_ns;
	uint64_t measured = _skr_present_measured_ns(ring);
	return measured ? measured : SKR_PRESENT_REFRESH_DEFAULT_NS;
}
