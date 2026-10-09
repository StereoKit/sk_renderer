// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// O(1) two level segregated fit offset allocator, the suballocator inside
// each device memory block. Metadata lives in a flat node array, never in the
// managed range, since device memory can't hold headers.

#define _SKR_TLSF_NONE        0xFFFFFFFFu
#define _SKR_TLSF_GRANULARITY 64          // every offset and size is a multiple of this
#define _SKR_TLSF_MAX_SIZE    0x80000000u // range sizes stay below this

typedef struct {
	uint32_t offset;
	uint32_t size;
	uint32_t bin_prev;  // free list within a bin, unused while allocated
	uint32_t bin_next;  // also links the stack of unused nodes
	uint32_t phys_prev; // neighbors in offset order
	uint32_t phys_next;
	bool     used;
} _skr_tlsf_node_t;

typedef struct {
	_skr_tlsf_node_t* nodes;
	uint32_t          node_capacity;
	uint32_t          node_unused;  // head of the unused node stack
	uint32_t          size;
	uint32_t          free_bytes;
	uint32_t          top_mask;     // bit per exponent with any non-empty bin
	uint8_t           leaf_masks[32];
	uint32_t          bins[256];    // free list heads
} _skr_tlsf_t;

// The backend's allocation wrappers, see _skr_shared.h
void* _skr_realloc(void* ptr, size_t size);
void  _skr_free   (void* ptr);

bool     _skr_tlsf_init   (_skr_tlsf_t* out_tlsf, uint32_t size);
void     _skr_tlsf_destroy(_skr_tlsf_t* ref_tlsf);
uint32_t _skr_tlsf_alloc  (_skr_tlsf_t* ref_tlsf, uint32_t size, uint32_t align, uint32_t* out_offset); // node, or _SKR_TLSF_NONE
void     _skr_tlsf_free   (_skr_tlsf_t* ref_tlsf, uint32_t node);
uint32_t _skr_tlsf_size   (const _skr_tlsf_t* tlsf, uint32_t node); // allocated bytes, after granularity rounding
