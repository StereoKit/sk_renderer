// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// Host fuzz test for the device memory suballocator (skr_tlsf.c). Random
// alloc/free traffic checked against an ownership map, plus structural
// invariants after every step. No GPU, no Vulkan.

#include "../sk_renderer/vk/skr_tlsf.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void* _skr_realloc(void* ptr, size_t size) { return realloc(ptr, size); }
void  _skr_free   (void* ptr)              { free(ptr); }

static int32_t _failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { _failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); return false; } } while (0)

static uint64_t _rng_state;
static uint32_t _rand(void) {
	_rng_state ^= _rng_state << 13;
	_rng_state ^= _rng_state >> 7;
	_rng_state ^= _rng_state << 17;
	return (uint32_t)_rng_state;
}

typedef struct {
	uint32_t node;
	uint32_t offset;
	uint32_t size; // as requested
} alloc_t;

// Every free node is filed in the bin its size maps to, never touches another
// free node, and the free sizes sum to free_bytes
static bool _check_structure(const _skr_tlsf_t* tlsf) {
	uint64_t free_sum = 0;
	for (uint32_t bin = 0; bin < 256; bin++) {
		bool leaf_bit = (tlsf->leaf_masks[bin >> 3] >> (bin & 7)) & 1;
		CHECK(leaf_bit == (tlsf->bins[bin] != _SKR_TLSF_NONE), "bin %u mask disagrees with its list", bin);
		uint32_t prev = _SKR_TLSF_NONE;
		for (uint32_t i = tlsf->bins[bin]; i != _SKR_TLSF_NONE; i = tlsf->nodes[i].bin_next) {
			const _skr_tlsf_node_t* n = &tlsf->nodes[i];
			CHECK(!n->used,                 "used node %u in bin %u", i, bin);
			CHECK(n->bin_prev == prev,      "node %u bin_prev broken", i);
			CHECK(n->size >= _SKR_TLSF_GRANULARITY && n->size % _SKR_TLSF_GRANULARITY == 0, "free node %u size %u", i, n->size);
			CHECK(n->phys_prev == _SKR_TLSF_NONE || tlsf->nodes[n->phys_prev].used, "free node %u has a free left neighbor", i);
			CHECK(n->phys_next == _SKR_TLSF_NONE || tlsf->nodes[n->phys_next].used, "free node %u has a free right neighbor", i);
			CHECK(n->phys_prev == _SKR_TLSF_NONE || tlsf->nodes[n->phys_prev].offset + tlsf->nodes[n->phys_prev].size == n->offset, "node %u not contiguous with left", i);
			free_sum += n->size;
			prev = i;
		}
	}
	for (uint32_t top = 0; top < 32; top++)
		CHECK(((tlsf->top_mask >> top) & 1) == (tlsf->leaf_masks[top] != 0), "top bit %u disagrees", top);
	CHECK(free_sum == tlsf->free_bytes, "free sum %" PRIu64 " != free_bytes %u", free_sum, tlsf->free_bytes);
	return true;
}

static bool _check_empty(const _skr_tlsf_t* tlsf) {
	CHECK(tlsf->free_bytes == tlsf->size, "free_bytes %u != size %u after freeing everything", tlsf->free_bytes, tlsf->size);
	uint32_t count = 0;
	for (uint32_t bin = 0; bin < 256; bin++)
		for (uint32_t i = tlsf->bins[bin]; i != _SKR_TLSF_NONE; i = tlsf->nodes[i].bin_next) {
			CHECK(tlsf->nodes[i].offset == 0 && tlsf->nodes[i].size == tlsf->size, "leftover free node %u", i);
			count++;
		}
	CHECK(count == 1, "%u free nodes after freeing everything", count);
	return true;
}

// Sizes skewed small, the way buffers and small textures are
static uint32_t _rand_size(uint32_t range) {
	uint32_t r = _rand() % 100;
	uint32_t max = r < 60 ? 4096 : r < 90 ? 256 * 1024 : r < 99 ? 4 * 1024 * 1024 : range / 2;
	return 1 + _rand() % max;
}

static uint32_t _rand_align(void) {
	uint32_t r = _rand() % 100;
	return r < 50 ? 16 : r < 70 ? 64 : r < 85 ? 256 : r < 95 ? 4096 : 65536;
}

static bool _fuzz(uint64_t seed, uint32_t range, uint32_t steps, bool check_every_step) {
	_rng_state = seed;
	_skr_tlsf_t tlsf;
	CHECK(_skr_tlsf_init(&tlsf, range), "init %u", range);

	uint32_t  units = range / _SKR_TLSF_GRANULARITY;
	uint8_t*  owned = calloc(units, 1);
	alloc_t*  live  = malloc(sizeof(alloc_t) * steps);
	uint32_t  live_count = 0, fails = 0;
	uint64_t  live_bytes = 0, peak_fill = 0;

	for (uint32_t step = 0; step < steps; step++) {
		// Drift between filling and draining so both full and sparse states get coverage
		uint32_t phase = (step / 4096) % 2;
		bool     do_alloc = live_count == 0 || (_rand() % 100) < (phase ? 70u : 40u);

		if (do_alloc) {
			uint32_t size  = _rand_size(range);
			uint32_t align = _rand_align();
			uint32_t offset;
			uint32_t node = _skr_tlsf_alloc(&tlsf, size, align, &offset);
			if (node == _SKR_TLSF_NONE) { fails++; continue; }

			uint32_t rounded = _skr_tlsf_size(&tlsf, node);
			CHECK(offset % align == 0 && offset % _SKR_TLSF_GRANULARITY == 0, "offset %u misaligned for %u", offset, align);
			CHECK(rounded >= size && rounded - size < _SKR_TLSF_GRANULARITY, "size %u rounded to %u", size, rounded);
			CHECK((uint64_t)offset + rounded <= range, "allocation [%u, +%u) past range", offset, rounded);
			for (uint32_t u = offset / _SKR_TLSF_GRANULARITY; u < (offset + rounded) / _SKR_TLSF_GRANULARITY; u++) {
				CHECK(!owned[u], "overlap at %u", u * _SKR_TLSF_GRANULARITY);
				owned[u] = 1;
			}
			live[live_count++] = (alloc_t){ node, offset, rounded };
			live_bytes += rounded;
			if (live_bytes > peak_fill) peak_fill = live_bytes;
		} else {
			uint32_t idx = _rand() % live_count;
			alloc_t  a   = live[idx];
			live[idx] = live[--live_count];
			for (uint32_t u = a.offset / _SKR_TLSF_GRANULARITY; u < (a.offset + a.size) / _SKR_TLSF_GRANULARITY; u++)
				owned[u] = 0;
			_skr_tlsf_free(&tlsf, a.node);
			live_bytes -= a.size;
		}

		CHECK(tlsf.size - tlsf.free_bytes == live_bytes, "used %u != live %" PRIu64, tlsf.size - tlsf.free_bytes, live_bytes);
		if (check_every_step && !_check_structure(&tlsf)) return false;
	}
	if (!_check_structure(&tlsf)) return false;

	while (live_count > 0) {
		_skr_tlsf_free(&tlsf, live[--live_count].node);
		if (check_every_step && !_check_structure(&tlsf)) return false;
	}
	if (!_check_empty(&tlsf)) return false;

	printf("  seed %-4" PRIu64 " range %5u KB  steps %7u  failed allocs %6u  peak fill %5.1f%%\n",
		seed, range / 1024, steps, fails, 100.0 * (double)peak_fill / range);

	_skr_tlsf_destroy(&tlsf);
	free(owned);
	free(live);
	return true;
}

// Fill with equal pieces, free every other one, then the rest: exercises both
// coalescing directions and the exact fit bins
static bool _checkerboard(void) {
	const uint32_t range = 1 << 20, piece = 4096, count = range / piece;
	_skr_tlsf_t tlsf;
	CHECK(_skr_tlsf_init(&tlsf, range), "init");
	uint32_t nodes[256];
	for (uint32_t i = 0; i < count; i++) {
		uint32_t offset;
		nodes[i] = _skr_tlsf_alloc(&tlsf, piece, 16, &offset);
		CHECK(nodes[i] != _SKR_TLSF_NONE && offset == i * piece, "piece %u at %u", i, offset);
	}
	uint32_t offset;
	CHECK(_skr_tlsf_alloc(&tlsf, 64, 16, &offset) == _SKR_TLSF_NONE, "allocated past a full range");
	for (uint32_t i = 0; i < count; i += 2) _skr_tlsf_free(&tlsf, nodes[i]);
	if (!_check_structure(&tlsf)) return false;
	CHECK(_skr_tlsf_alloc(&tlsf, piece + 64, 16, &offset) == _SKR_TLSF_NONE, "allocated across a used piece");
	for (uint32_t i = 1; i < count; i += 2) _skr_tlsf_free(&tlsf, nodes[i]);
	if (!_check_empty(&tlsf)) return false;

	// Whole range, then a large alignment against an unaligned hole
	uint32_t whole = _skr_tlsf_alloc(&tlsf, range, 65536, &offset);
	CHECK(whole != _SKR_TLSF_NONE && offset == 0, "whole range");
	_skr_tlsf_free(&tlsf, whole);
	uint32_t a = _skr_tlsf_alloc(&tlsf, 64, 64, &offset);
	uint32_t b = _skr_tlsf_alloc(&tlsf, 65536, 65536, &offset);
	CHECK(b != _SKR_TLSF_NONE && offset == 65536, "64 KB aligned piece at %u", offset);
	if (!_check_structure(&tlsf)) return false;
	_skr_tlsf_free(&tlsf, a);
	_skr_tlsf_free(&tlsf, b);
	if (!_check_empty(&tlsf)) return false;

	_skr_tlsf_destroy(&tlsf);
	printf("  checkerboard ok\n");
	return true;
}

static void _bench(void) {
	const uint32_t range = 32u << 20, live_max = 4096, ops = 2000000;
	_rng_state = 1234;
	_skr_tlsf_t tlsf;
	_skr_tlsf_init(&tlsf, range);
	uint32_t* live = malloc(sizeof(uint32_t) * live_max);
	uint32_t  live_count = 0;

	struct timespec t0, t1;
	timespec_get(&t0, TIME_UTC);
	for (uint32_t i = 0; i < ops; i++) {
		if (live_count < live_max && (live_count == 0 || (_rand() & 1))) {
			uint32_t offset;
			uint32_t node = _skr_tlsf_alloc(&tlsf, 1 + _rand() % 16384, 64u << (_rand() % 4), &offset);
			if (node != _SKR_TLSF_NONE) live[live_count++] = node;
		} else {
			uint32_t idx = _rand() % live_count;
			_skr_tlsf_free(&tlsf, live[idx]);
			live[idx] = live[--live_count];
		}
	}
	timespec_get(&t1, TIME_UTC);
	double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec);
	printf("  bench: %.1f ns per op (%u ops, up to %u live, node capacity %u)\n", ns / ops, ops, live_max, tlsf.node_capacity);
	free(live);
	_skr_tlsf_destroy(&tlsf);
}

int main(int argc, char** argv) {
	bool long_run = argc > 1 && strcmp(argv[1], "-long") == 0;
	printf("skr_tlsf tests\n");

	_checkerboard();
	for (uint64_t seed = 1; seed <= 16; seed++)
		_fuzz(seed, (1u << 20) << (seed % 6), long_run ? 200000 : 20000, true);
	for (uint64_t seed = 100; seed < (long_run ? 116u : 102u); seed++)
		_fuzz(seed, 32u << 20, 1000000, false);
	_bench();

	printf(_failures ? "FAILED (%d)\n" : "passed\n", _failures);
	return _failures ? 1 : 0;
}
