// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "skr_tlsf.h"

#include <assert.h>
#include <string.h>

#if defined(_MSC_VER)
	#include <intrin.h>
	static inline uint32_t _ctz(uint32_t v) { unsigned long i; _BitScanForward(&i, v); return (uint32_t)i; }
	static inline uint32_t _msb(uint32_t v) { unsigned long i; _BitScanReverse(&i, v); return (uint32_t)i; }
#else
	static inline uint32_t _ctz(uint32_t v) { return (uint32_t)__builtin_ctz(v); }
	static inline uint32_t _msb(uint32_t v) { return 31u - (uint32_t)__builtin_clz(v); }
#endif

///////////////////////////////////////////////////////////////////////////////
// Bins

// Where a free node of this size is filed: the largest bin whose lower bound
// is <= size
static uint32_t _bin_down(uint32_t size) {
	if (size < 8) return size;
	uint32_t shift = _msb(size) - 3;
	return ((shift + 1) << 3) | ((size >> shift) & 7);
}

// Where a search for this size starts: the smallest bin whose lower bound is
// >= size, so every node in it fits
static uint32_t _bin_up(uint32_t size) {
	if (size < 8) return size;
	uint32_t shift = _msb(size) - 3;
	uint32_t bin   = ((shift + 1) << 3) | ((size >> shift) & 7);
	return (size & ((1u << shift) - 1)) ? bin + 1 : bin;
}

// First non-empty bin at or above `bin`
static uint32_t _bin_find(const _skr_tlsf_t* tlsf, uint32_t bin) {
	uint32_t top = bin >> 3;
	if (top >= 32) return _SKR_TLSF_NONE;

	uint32_t leaf = tlsf->leaf_masks[top] & (0xFFu << (bin & 7));
	if (leaf) return (top << 3) | _ctz(leaf);

	uint32_t tops = top < 31 ? tlsf->top_mask & (0xFFFFFFFFu << (top + 1)) : 0;
	if (!tops) return _SKR_TLSF_NONE;
	top = _ctz(tops);
	return (top << 3) | _ctz(tlsf->leaf_masks[top]);
}

static void _bin_insert(_skr_tlsf_t* ref_tlsf, uint32_t node) {
	_skr_tlsf_node_t* n   = &ref_tlsf->nodes[node];
	uint32_t          bin = _bin_down(n->size);
	n->used     = false;
	n->bin_prev = _SKR_TLSF_NONE;
	n->bin_next = ref_tlsf->bins[bin];
	if (n->bin_next != _SKR_TLSF_NONE) ref_tlsf->nodes[n->bin_next].bin_prev = node;
	ref_tlsf->bins[bin]              = node;
	ref_tlsf->leaf_masks[bin >> 3] |= (uint8_t)(1u << (bin & 7));
	ref_tlsf->top_mask             |= 1u << (bin >> 3);
	ref_tlsf->free_bytes           += n->size;
}

static void _bin_remove(_skr_tlsf_t* ref_tlsf, uint32_t node) {
	_skr_tlsf_node_t* n = &ref_tlsf->nodes[node];
	if (n->bin_prev != _SKR_TLSF_NONE) {
		ref_tlsf->nodes[n->bin_prev].bin_next = n->bin_next;
	} else {
		uint32_t bin = _bin_down(n->size);
		ref_tlsf->bins[bin] = n->bin_next;
		if (n->bin_next == _SKR_TLSF_NONE) {
			ref_tlsf->leaf_masks[bin >> 3] &= (uint8_t)~(1u << (bin & 7));
			if (ref_tlsf->leaf_masks[bin >> 3] == 0)
				ref_tlsf->top_mask &= ~(1u << (bin >> 3));
		}
	}
	if (n->bin_next != _SKR_TLSF_NONE) ref_tlsf->nodes[n->bin_next].bin_prev = n->bin_prev;
	ref_tlsf->free_bytes -= n->size;
}

///////////////////////////////////////////////////////////////////////////////
// Nodes

static uint32_t _node_new(_skr_tlsf_t* ref_tlsf) {
	if (ref_tlsf->node_unused == _SKR_TLSF_NONE) {
		uint32_t          capacity = ref_tlsf->node_capacity == 0 ? 64 : ref_tlsf->node_capacity * 2;
		_skr_tlsf_node_t* nodes    = _skr_realloc(ref_tlsf->nodes, capacity * sizeof(_skr_tlsf_node_t));
		if (!nodes) return _SKR_TLSF_NONE;
		for (uint32_t i = ref_tlsf->node_capacity; i < capacity; i++)
			nodes[i] = (_skr_tlsf_node_t){ .bin_next = i + 1 < capacity ? i + 1 : _SKR_TLSF_NONE };
		ref_tlsf->nodes         = nodes;
		ref_tlsf->node_unused   = ref_tlsf->node_capacity;
		ref_tlsf->node_capacity = capacity;
	}
	uint32_t node = ref_tlsf->node_unused;
	ref_tlsf->node_unused = ref_tlsf->nodes[node].bin_next;
	return node;
}

static void _node_release(_skr_tlsf_t* ref_tlsf, uint32_t node) {
	ref_tlsf->nodes[node] = (_skr_tlsf_node_t){ .bin_next = ref_tlsf->node_unused };
	ref_tlsf->node_unused = node;
}

///////////////////////////////////////////////////////////////////////////////

bool _skr_tlsf_init(_skr_tlsf_t* out_tlsf, uint32_t size) {
	*out_tlsf = (_skr_tlsf_t){ .node_unused = _SKR_TLSF_NONE };
	memset(out_tlsf->bins, 0xFF, sizeof(out_tlsf->bins));

	size &= ~(uint32_t)(_SKR_TLSF_GRANULARITY - 1);
	if (size == 0 || size >= _SKR_TLSF_MAX_SIZE) return false;
	out_tlsf->size = size;

	uint32_t node = _node_new(out_tlsf);
	if (node == _SKR_TLSF_NONE) return false;
	out_tlsf->nodes[node] = (_skr_tlsf_node_t){
		.size      = size,
		.phys_prev = _SKR_TLSF_NONE,
		.phys_next = _SKR_TLSF_NONE,
	};
	_bin_insert(out_tlsf, node);
	return true;
}

void _skr_tlsf_destroy(_skr_tlsf_t* ref_tlsf) {
	_skr_free(ref_tlsf->nodes);
	*ref_tlsf = (_skr_tlsf_t){0};
}

uint32_t _skr_tlsf_alloc(_skr_tlsf_t* ref_tlsf, uint32_t size, uint32_t align, uint32_t* out_offset) {
	// Both bounds keep size + pad_max below 2^32
	if (size == 0 || size > ref_tlsf->size || align > ref_tlsf->size) return _SKR_TLSF_NONE;
	size = (size + _SKR_TLSF_GRANULARITY - 1) & ~(uint32_t)(_SKR_TLSF_GRANULARITY - 1);
	if (align < _SKR_TLSF_GRANULARITY) align = _SKR_TLSF_GRANULARITY;
	assert((align & (align - 1)) == 0);

	// Searching for the worst case padding guarantees a fit. When that finds
	// nothing, an exact size node may still happen to sit aligned.
	uint32_t pad_max = align - _SKR_TLSF_GRANULARITY;
	uint32_t bin     = _bin_find(ref_tlsf, _bin_up(size + pad_max));
	uint32_t node    = bin == _SKR_TLSF_NONE ? _SKR_TLSF_NONE : ref_tlsf->bins[bin];
	if (node == _SKR_TLSF_NONE && pad_max > 0) {
		bin = _bin_find(ref_tlsf, _bin_up(size));
		if (bin == _SKR_TLSF_NONE) return _SKR_TLSF_NONE;
		const _skr_tlsf_node_t* candidate = &ref_tlsf->nodes[ref_tlsf->bins[bin]];
		uint32_t pad = ((candidate->offset + align - 1) & ~(align - 1)) - candidate->offset;
		if (candidate->size - size >= pad) node = ref_tlsf->bins[bin];
	}
	if (node == _SKR_TLSF_NONE) return _SKR_TLSF_NONE;

	uint32_t offset = ref_tlsf->nodes[node].offset;
	uint32_t pad    = ((offset + align - 1) & ~(align - 1)) - offset;
	uint32_t rest   = ref_tlsf->nodes[node].size - pad - size;

	// Split nodes come first: they can grow the node array
	uint32_t front = pad  > 0 ? _node_new(ref_tlsf) : _SKR_TLSF_NONE;
	uint32_t back  = rest > 0 ? _node_new(ref_tlsf) : _SKR_TLSF_NONE;
	if ((pad > 0 && front == _SKR_TLSF_NONE) || (rest > 0 && back == _SKR_TLSF_NONE)) {
		if (front != _SKR_TLSF_NONE) _node_release(ref_tlsf, front);
		return _SKR_TLSF_NONE;
	}

	_bin_remove(ref_tlsf, node);
	_skr_tlsf_node_t* nodes = ref_tlsf->nodes;

	// The split pieces never need coalescing: the node was free, so its
	// neighbors are allocated
	if (front != _SKR_TLSF_NONE) {
		nodes[front] = (_skr_tlsf_node_t){
			.offset    = offset,
			.size      = pad,
			.phys_prev = nodes[node].phys_prev,
			.phys_next = node,
		};
		if (nodes[node].phys_prev != _SKR_TLSF_NONE) nodes[nodes[node].phys_prev].phys_next = front;
		nodes[node].phys_prev = front;
		_bin_insert(ref_tlsf, front);
	}
	if (back != _SKR_TLSF_NONE) {
		nodes[back] = (_skr_tlsf_node_t){
			.offset    = offset + pad + size,
			.size      = rest,
			.phys_prev = node,
			.phys_next = nodes[node].phys_next,
		};
		if (nodes[node].phys_next != _SKR_TLSF_NONE) nodes[nodes[node].phys_next].phys_prev = back;
		nodes[node].phys_next = back;
		_bin_insert(ref_tlsf, back);
	}

	nodes[node].offset = offset + pad;
	nodes[node].size   = size;
	nodes[node].used   = true;
	*out_offset = offset + pad;
	return node;
}

void _skr_tlsf_free(_skr_tlsf_t* ref_tlsf, uint32_t node) {
	_skr_tlsf_node_t* nodes = ref_tlsf->nodes;
	assert(nodes[node].used);

	uint32_t prev = nodes[node].phys_prev;
	if (prev != _SKR_TLSF_NONE && !nodes[prev].used) {
		_bin_remove(ref_tlsf, prev);
		nodes[node].offset     = nodes[prev].offset;
		nodes[node].size      += nodes[prev].size;
		nodes[node].phys_prev  = nodes[prev].phys_prev;
		if (nodes[node].phys_prev != _SKR_TLSF_NONE) nodes[nodes[node].phys_prev].phys_next = node;
		_node_release(ref_tlsf, prev);
	}

	uint32_t next = nodes[node].phys_next;
	if (next != _SKR_TLSF_NONE && !nodes[next].used) {
		_bin_remove(ref_tlsf, next);
		nodes[node].size      += nodes[next].size;
		nodes[node].phys_next  = nodes[next].phys_next;
		if (nodes[node].phys_next != _SKR_TLSF_NONE) nodes[nodes[node].phys_next].phys_prev = node;
		_node_release(ref_tlsf, next);
	}

	_bin_insert(ref_tlsf, node);
}

uint32_t _skr_tlsf_size(const _skr_tlsf_t* tlsf, uint32_t node) {
	return tlsf->nodes[node].size;
}
