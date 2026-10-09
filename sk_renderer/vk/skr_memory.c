// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"
#include "skr_tlsf.h"

#include <assert.h>
#include <string.h>

// Blocks start small and grow with their pool, up to the cap. Pools hold only
// small items, so the cap never limits a resource's size.
#define _SKR_MEM_BLOCK_MIN   (256u << 10)
#define _SKR_MEM_BLOCK_MAX   (32u  << 20)
#define _SKR_MEM_POOLED_MAX           (1u   << 20) // larger items go dedicated
#define _SKR_MEM_TRANSIENT_POOLED_MAX (512u << 10) // the same, for staging and readback
#define _SKR_MEM_EMPTY_FRAMES 120 // an empty block outliving this goes back to the driver
#define _SKR_MEM_CHUNK_SHIFT 6
#define _SKR_MEM_CHUNK       (1u << _SKR_MEM_CHUNK_SHIFT)
#define _SKR_MEM_MAX_CHUNKS  1024
#define _SKR_MEM_MAX_POOLS   (VK_MAX_MEMORY_TYPES * 2)
#define _SKR_MEM_NEVER       (VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD)

typedef enum {
	_skr_mem_block_pooled,
	_skr_mem_block_dedicated,
	_skr_mem_block_lazy,     // dedicated, in a lazily allocated type
	_skr_mem_block_external, // imported, owned but not ours to report as reserved
} _skr_mem_block_;

typedef struct {
	VkDeviceMemory memory;    // VK_NULL_HANDLE while the slot is unused
	uint8_t*       mapped;    // persistent mapping, NULL unless host visible
	VkDeviceSize   size;
	_skr_tlsf_t*   tlsf;      // pooled blocks only
	uint32_t       next_free; // unused slot stack
	uint16_t       pool;
	uint8_t        kind;      // _skr_mem_block_
} _skr_mem_block_t;

typedef struct {
	mtx_t                 mutex;
	uint32_t*             blocks;      // block indices, oldest first
	uint32_t              block_count;
	uint32_t              block_capacity;
	uint32_t              type;
	_skr_atomic(uint32_t) empty_block; // the one empty block kept against churn, 0 = none; the tick peeks unlocked
	uint32_t              empty_frame; // when it emptied
} _skr_mem_pool_t;

typedef struct {
	VkMemoryPropertyFlags required;
	VkMemoryPropertyFlags preferred;
	VkMemoryPropertyFlags avoided;
} _skr_mem_usage_flags_t;

static const _skr_mem_usage_flags_t _skr_mem_usage_flags[_skr_mem_usage_max] = {
	[_skr_mem_usage_gpu]      = { 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT },
	[_skr_mem_usage_stream]   = { VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT },
	// Staging in system RAM leaves BAR and VRAM for what the GPU reads every frame
	[_skr_mem_usage_staging]  = { VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT },
	[_skr_mem_usage_readback] = { VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0 },
	[_skr_mem_usage_lazy]     = { 0, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT },
	[_skr_mem_usage_upload]   = { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0 },
};

typedef struct {
	VkPhysicalDeviceMemoryProperties props;
	uint8_t          order      [_skr_mem_usage_max][VK_MAX_MEMORY_TYPES]; // usable types, best first
	uint32_t         order_count[_skr_mem_usage_max];
	uint32_t         block_max  [VK_MAX_MEMORY_TYPES];
	bool             split_linear; // bufferImageGranularity > 1: buffers and images never share a block
	bool             has_budget;

	_skr_mem_pool_t  pools[_SKR_MEM_MAX_POOLS];

	mtx_t             table_mutex;
	_skr_mem_block_t* chunks[_SKR_MEM_MAX_CHUNKS];
	uint32_t          chunk_count;
	uint32_t          free_slot;   // unused slot stack, 0 = empty

	_skr_atomic(uint64_t) reserved;
	_skr_atomic(uint64_t) used;
	_skr_atomic(uint64_t) peak;
	_skr_atomic(uint64_t) lazy;
	_skr_atomic(uint64_t) external;
	_skr_atomic(uint64_t) category[skr_mem_category_max];
	_skr_atomic(uint32_t) block_count;
	_skr_atomic(uint32_t) dedicated_count;
	_skr_atomic(uint32_t) allocation_count;
} _skr_mem_state_t;
static _skr_mem_state_t _mem;

///////////////////////////////////////////////////////////////////////////////
// Block table

static _skr_mem_block_t* _block(uint32_t idx) {
	return &_mem.chunks[(idx - 1) >> _SKR_MEM_CHUNK_SHIFT][(idx - 1) & (_SKR_MEM_CHUNK - 1)];
}

static uint32_t _block_new(void) {
	mtx_lock(&_mem.table_mutex);
	uint32_t idx = _mem.free_slot;
	if (idx != 0) {
		_mem.free_slot = _block(idx)->next_free;
	} else if (_mem.chunk_count < _SKR_MEM_MAX_CHUNKS) {
		_skr_mem_block_t* chunk = _skr_calloc(_SKR_MEM_CHUNK, sizeof(_skr_mem_block_t));
		if (chunk) {
			uint32_t base = _mem.chunk_count * _SKR_MEM_CHUNK;
			_mem.chunks[_mem.chunk_count++] = chunk;
			for (uint32_t e = _SKR_MEM_CHUNK - 1; e > 0; e--) {
				chunk[e].next_free = _mem.free_slot;
				_mem.free_slot     = base + e + 1;
			}
			idx = base + 1;
		}
	}
	mtx_unlock(&_mem.table_mutex);
	return idx;
}

static void _block_release(uint32_t idx) {
	mtx_lock(&_mem.table_mutex);
	*_block(idx)   = (_skr_mem_block_t){ .next_free = _mem.free_slot };
	_mem.free_slot = idx;
	mtx_unlock(&_mem.table_mutex);
}

///////////////////////////////////////////////////////////////////////////////
// Stats

static void _stat_sub(_skr_atomic(uint64_t)* ref_stat, uint64_t bytes) {
	_skr_add_u64(ref_stat, (uint64_t)0 - bytes);
}

static void _stat_used(skr_mem_category_ category, uint64_t bytes) {
	_skr_add_u64(&_mem.category[category], bytes);
	uint64_t used = _skr_add_u64(&_mem.used, bytes) + bytes;
	uint64_t peak = _skr_load_u64(&_mem.peak);
	while (used > peak && !_skr_cas_u64(&_mem.peak, peak, used))
		peak = _skr_load_u64(&_mem.peak);
}

static void _stat_unused(skr_mem_category_ category, uint64_t bytes) {
	_stat_sub(&_mem.category[category], bytes);
	_stat_sub(&_mem.used, bytes);
}

///////////////////////////////////////////////////////////////////////////////
// Device memory

static uint32_t _bits(uint32_t v) {
	uint32_t count = 0;
	for (; v; v &= v - 1) count++;
	return count;
}

// Coherent host visible memory is mapped once, for the block's whole life.
// Nothing here flushes, so a non-coherent type is never handed out mapped.
static bool _device_alloc(uint32_t type, VkDeviceSize size, VkBuffer opt_buffer, VkImage opt_image, VkDeviceMemory* out_memory, uint8_t** out_mapped) {
	VkMemoryDedicatedAllocateInfo dedicated = {
		.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
		.image  = opt_image,
		.buffer = opt_buffer,
	};
	VkResult vr = vkAllocateMemory(_skr_vk.device, &(VkMemoryAllocateInfo){
		.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext           = (opt_buffer || opt_image) ? &dedicated : NULL,
		.allocationSize  = size,
		.memoryTypeIndex = type,
	}, NULL, out_memory);
	if (vr != VK_SUCCESS) return false;

	*out_mapped = NULL;
	VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	if ((_mem.props.memoryTypes[type].propertyFlags & host) == host) {
		if (vkMapMemory(_skr_vk.device, *out_memory, 0, VK_WHOLE_SIZE, 0, (void**)out_mapped) != VK_SUCCESS) {
			vkFreeMemory(_skr_vk.device, *out_memory, NULL);
			return false;
		}
	}
	return true;
}

static bool _alloc_dedicated(uint32_t type, VkDeviceSize size, VkBuffer opt_buffer, VkImage opt_image, skr_mem_category_ category, _skr_mem_t* out_mem) {
	VkDeviceMemory memory;
	uint8_t*       mapped;
	if (!_device_alloc(type, size, opt_buffer, opt_image, &memory, &mapped)) return false;

	uint32_t idx = _block_new();
	if (idx == 0) {
		vkFreeMemory(_skr_vk.device, memory, NULL);
		return false;
	}
	bool lazy = _mem.props.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
	*_block(idx) = (_skr_mem_block_t){
		.memory = memory,
		.mapped = mapped,
		.size   = size,
		.kind   = lazy ? _skr_mem_block_lazy : _skr_mem_block_dedicated,
	};
	*out_mem = (_skr_mem_t){ .block = idx, .category = category };

	if (lazy) {
		_skr_add_u64(&_mem.lazy, size);
	} else {
		_skr_add_u64(&_mem.reserved, size);
		_skr_add_u32(&_mem.dedicated_count, 1);
		_skr_add_u32(&_mem.allocation_count, 1);
		_stat_used(category, size);
	}
	return true;
}

static bool _pool_try(_skr_mem_pool_t* ref_pool, uint32_t size, uint32_t align, _skr_mem_t* out_mem, VkDeviceSize* out_offset) {
	for (uint32_t i = 0; i < ref_pool->block_count; i++) {
		_skr_mem_block_t* block = _block(ref_pool->blocks[i]);
		uint32_t          offset;
		uint32_t          node = _skr_tlsf_alloc(block->tlsf, size, align, &offset);
		if (node == _SKR_TLSF_NONE) continue;
		assert(node < (1u << 24));
		if (ref_pool->empty_block == ref_pool->blocks[i]) _skr_store_u32(&ref_pool->empty_block, 0);
		out_mem->block = ref_pool->blocks[i];
		out_mem->node  = node;
		*out_offset    = offset;
		return true;
	}
	return false;
}

static void _pool_remove(_skr_mem_pool_t* ref_pool, uint32_t block) {
	uint32_t i = 0;
	while (ref_pool->blocks[i] != block) i++;
	for (i++; i < ref_pool->block_count; i++) ref_pool->blocks[i - 1] = ref_pool->blocks[i];
	ref_pool->block_count--;
}

// Only once the block is out of its pool, so no other thread can reach it
static void _block_destroy_pooled(uint32_t idx) {
	_skr_mem_block_t* block = _block(idx);
	_stat_sub   (&_mem.reserved, block->size);
	_skr_add_u32(&_mem.block_count, (uint32_t)-1);
	_skr_tlsf_destroy(block->tlsf);
	_skr_free(block->tlsf);
	vkFreeMemory(_skr_vk.device, block->memory, NULL);
	_block_release(idx);
}

static bool _alloc_pooled(uint32_t pool_idx, const VkMemoryRequirements* reqs, skr_mem_category_ category, _skr_mem_t* out_mem, VkDeviceSize* out_offset) {
	_skr_mem_pool_t* pool  = &_mem.pools[pool_idx];
	uint32_t         size  = (uint32_t)reqs->size;
	uint32_t         align = (uint32_t)reqs->alignment;
	*out_mem = (_skr_mem_t){ .category = category };

	// About an eighth of the pool, so growth overshoots ~12.5% at most; cheap,
	// since vkAllocateMemory costs by the byte rather than by the call
	mtx_lock(&pool->mutex);
	bool     found = _pool_try(pool, size, align, out_mem, out_offset);
	uint64_t total = 0;
	for (uint32_t i = 0; !found && i < pool->block_count; i++)
		total += _block(pool->blocks[i])->size;
	mtx_unlock(&pool->mutex);
	if (found) goto done;
	uint32_t block_size = _SKR_MEM_BLOCK_MIN;
	while ((uint64_t)block_size * 2 <= total / 8) block_size *= 2;
	if (block_size > _mem.block_max[pool->type]) block_size = _mem.block_max[pool->type];
	while (block_size < size) block_size *= 2;

	// The new block's vkAllocateMemory runs unlocked, so other threads keep
	// suballocating meanwhile. A failed block retries at just the size needed.
	VkDeviceMemory memory;
	uint8_t*       mapped;
	if (!_device_alloc(pool->type, block_size, VK_NULL_HANDLE, VK_NULL_HANDLE, &memory, &mapped)) {
		block_size = (size + 0xFFFFu) & ~0xFFFFu;
		if (!_device_alloc(pool->type, block_size, VK_NULL_HANDLE, VK_NULL_HANDLE, &memory, &mapped)) return false;
	}
	_skr_tlsf_t* tlsf = _skr_malloc(sizeof(_skr_tlsf_t));
	uint32_t     idx  = tlsf && _skr_tlsf_init(tlsf, block_size) ? _block_new() : 0;
	if (idx == 0) {
		if (tlsf) _skr_tlsf_destroy(tlsf);
		_skr_free(tlsf);
		vkFreeMemory(_skr_vk.device, memory, NULL);
		return false;
	}
	*_block(idx) = (_skr_mem_block_t){
		.memory = memory,
		.mapped = mapped,
		.size   = block_size,
		.tlsf   = tlsf,
		.pool   = (uint16_t)pool_idx,
		.kind   = _skr_mem_block_pooled,
	};
	mtx_lock(&pool->mutex);
	if (pool->block_count == pool->block_capacity) {
		uint32_t  capacity = pool->block_capacity == 0 ? 8 : pool->block_capacity * 2;
		uint32_t* blocks   = _skr_realloc(pool->blocks, capacity * sizeof(uint32_t));
		if (!blocks) {
			mtx_unlock(&pool->mutex);
			_skr_tlsf_destroy(tlsf);
			_skr_free(tlsf);
			vkFreeMemory(_skr_vk.device, memory, NULL);
			_block_release(idx);
			return false;
		}
		pool->blocks         = blocks;
		pool->block_capacity = capacity;
	}
	pool->blocks[pool->block_count++] = idx;
	uint32_t offset;
	uint32_t node = _skr_tlsf_alloc(tlsf, size, align, &offset);
	mtx_unlock(&pool->mutex);
	_skr_add_u64(&_mem.reserved, block_size);
	_skr_add_u32(&_mem.block_count, 1);

	assert(node != _SKR_TLSF_NONE);
	out_mem->block = idx;
	out_mem->node  = node;
	*out_offset    = offset;

done:
	_skr_add_u32(&_mem.allocation_count, 1);
	_stat_used(category, _skr_tlsf_size(_block(out_mem->block)->tlsf, out_mem->node));
	return true;
}

static skr_err_ _skr_mem_alloc(const VkMemoryRequirements* reqs, bool prefer_dedicated, bool optimal, VkBuffer opt_buffer, VkImage opt_image, _skr_mem_usage_ usage, skr_mem_category_ category, _skr_mem_t* out_mem, VkDeviceSize* out_offset) {
	*out_mem    = (_skr_mem_t){0};
	*out_offset = 0;
	bool dedicated = prefer_dedicated || usage == _skr_mem_usage_lazy;

	for (uint32_t i = 0; i < _mem.order_count[usage]; i++) {
		uint32_t type = _mem.order[usage][i];
		if (!(reqs->memoryTypeBits & (1u << type))) continue;

		// Pools are for small items that would each waste a page; big ones only
		// leave holes, and short-lived staging would strand a block
		bool     transient  = usage == _skr_mem_usage_staging || usage == _skr_mem_usage_readback;
		uint32_t pooled_max = transient ? _SKR_MEM_TRANSIENT_POOLED_MAX : _SKR_MEM_POOLED_MAX;
		if (pooled_max > _mem.block_max[type] / 8) pooled_max = _mem.block_max[type] / 8;
		if (dedicated || reqs->size >= pooled_max || reqs->alignment > pooled_max) {
			if (_alloc_dedicated(type, reqs->size, opt_buffer, opt_image, category, out_mem)) return skr_err_success;
		} else {
			uint32_t pool = type * 2 + (_mem.split_linear && optimal ? 1 : 0);
			if (_alloc_pooled(pool, reqs, category, out_mem, out_offset)) return skr_err_success;
		}
	}
	skr_log(skr_log_critical, "Out of device memory: %llu bytes, usage %d, type bits 0x%x",
		(unsigned long long)reqs->size, (int32_t)usage, reqs->memoryTypeBits);
	return skr_err_out_of_memory;
}

///////////////////////////////////////////////////////////////////////////////

void _skr_mem_init(void) {
	_mem = (_skr_mem_state_t){0};
	mtx_init(&_mem.table_mutex, mtx_plain);

	VkPhysicalDeviceMaintenance3Properties maintenance3 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES };
	VkPhysicalDeviceProperties2            props2       = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &maintenance3 };
	vkGetPhysicalDeviceProperties2(_skr_vk.physical_device, &props2);
	vkGetPhysicalDeviceMemoryProperties(_skr_vk.physical_device, &_mem.props);
	_mem.split_linear = props2.properties.limits.bufferImageGranularity > 1;
	_mem.has_budget   = skr_vk_request_enabled(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

	for (uint32_t t = 0; t < _mem.props.memoryTypeCount; t++) {
		VkDeviceSize cap = _mem.props.memoryHeaps[_mem.props.memoryTypes[t].heapIndex].size / 16;
		if (cap > _SKR_MEM_BLOCK_MAX)                    cap = _SKR_MEM_BLOCK_MAX;
		if (cap > maintenance3.maxMemoryAllocationSize)  cap = maintenance3.maxMemoryAllocationSize;
		if (cap < _SKR_MEM_BLOCK_MIN)                    cap = _SKR_MEM_BLOCK_MIN;
		_mem.block_max[t] = (uint32_t)cap;
	}
	for (uint32_t p = 0; p < _SKR_MEM_MAX_POOLS; p++) {
		mtx_init(&_mem.pools[p].mutex, mtx_plain);
		_mem.pools[p].type = p / 2;
	}

	// Uploads only map memory in the main device heap; a separate small one
	// is a discrete GPU's BAR window, too scarce for static data
	uint32_t main_heap = UINT32_MAX;
	for (uint32_t h = 0; h < _mem.props.memoryHeapCount; h++) {
		if (!(_mem.props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
		if (main_heap == UINT32_MAX || _mem.props.memoryHeaps[h].size > _mem.props.memoryHeaps[main_heap].size) main_heap = h;
	}

	// Rank the usable types per usage. Ties keep the driver's order, which
	// lists its preferred types first.
	for (int32_t u = 0; u < _skr_mem_usage_max; u++) {
		const _skr_mem_usage_flags_t* want   = &_skr_mem_usage_flags[u];
		VkMemoryPropertyFlags         never  = _SKR_MEM_NEVER | (u == _skr_mem_usage_lazy ? 0 : VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
		int32_t                       scores[VK_MAX_MEMORY_TYPES];
		uint32_t                      count  = 0;
		for (uint32_t t = 0; t < _mem.props.memoryTypeCount; t++) {
			VkMemoryPropertyFlags flags = _mem.props.memoryTypes[t].propertyFlags;
			if ((flags & want->required) != want->required || (flags & never)) continue;
			if (u == _skr_mem_usage_upload && _mem.props.memoryTypes[t].heapIndex != main_heap) continue;
			int32_t score = 2 * (int32_t)_bits(flags & want->preferred) - (int32_t)_bits(flags & want->avoided);
			uint32_t at = count++;
			while (at > 0 && scores[at - 1] < score) {
				scores    [at]    = scores    [at - 1];
				_mem.order[u][at] = _mem.order[u][at - 1];
				at--;
			}
			scores    [at]    = score;
			_mem.order[u][at] = (uint8_t)t;
		}
		_mem.order_count[u] = count;
	}
}

void _skr_mem_shutdown(void) {
	uint64_t leaked_bytes = 0;
	uint32_t leaked_count = 0;
	for (uint32_t c = 0; c < _mem.chunk_count; c++) {
		for (uint32_t e = 0; e < _SKR_MEM_CHUNK; e++) {
			_skr_mem_block_t* block = &_mem.chunks[c][e];
			if (block->memory == VK_NULL_HANDLE) continue;
			if (block->kind == _skr_mem_block_pooled) {
				uint32_t in_use = block->tlsf->size - block->tlsf->free_bytes;
				if (in_use > 0) { leaked_bytes += in_use; leaked_count++; }
				_skr_tlsf_destroy(block->tlsf);
				_skr_free(block->tlsf);
			} else {
				leaked_bytes += block->size;
				leaked_count++;
			}
			vkFreeMemory(_skr_vk.device, block->memory, NULL);
		}
		_skr_free(_mem.chunks[c]);
	}
	if (leaked_count > 0)
		skr_log(skr_log_warning, "Device memory leak: %llu bytes still allocated in %u blocks at shutdown", (unsigned long long)leaked_bytes, leaked_count);

	for (uint32_t p = 0; p < _SKR_MEM_MAX_POOLS; p++) {
		_skr_free(_mem.pools[p].blocks);
		mtx_destroy(&_mem.pools[p].mutex);
	}
	mtx_destroy(&_mem.table_mutex);
	_mem = (_skr_mem_state_t){0};
}

skr_err_ _skr_mem_bind_buffer(VkBuffer buffer, _skr_mem_usage_ usage, skr_mem_category_ category, _skr_mem_t* out_mem, void** opt_out_mapped) {
	VkMemoryDedicatedRequirements dedicated = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
	VkMemoryRequirements2         reqs      = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated };
	vkGetBufferMemoryRequirements2(_skr_vk.device, &(VkBufferMemoryRequirementsInfo2){
		.sType  = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
		.buffer = buffer,
	}, &reqs);

	VkDeviceSize offset;
	skr_err_     err = _skr_mem_alloc(&reqs.memoryRequirements, dedicated.prefersDedicatedAllocation || dedicated.requiresDedicatedAllocation,
		false, buffer, VK_NULL_HANDLE, usage, category, out_mem, &offset);
	if (err != skr_err_success) return err;

	const _skr_mem_block_t* block = _block(out_mem->block);
	VkResult vr = vkBindBufferMemory(_skr_vk.device, buffer, block->memory, offset);
	if (vr != VK_SUCCESS) {
		skr_log(skr_log_critical, "vkBindBufferMemory: 0x%X", (uint32_t)vr);
		_skr_mem_free(*out_mem);
		*out_mem = (_skr_mem_t){0};
		return skr_err_device_error;
	}
	if (opt_out_mapped) *opt_out_mapped = block->mapped ? block->mapped + offset : NULL;
	return skr_err_success;
}

skr_err_ _skr_mem_bind_image(VkImage image, VkImageUsageFlags image_usage, _skr_mem_usage_ usage, skr_mem_category_ category, _skr_mem_t* out_mem) {
	VkMemoryDedicatedRequirements dedicated = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
	VkMemoryRequirements2         reqs      = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated };
	vkGetImageMemoryRequirements2(_skr_vk.device, &(VkImageMemoryRequirementsInfo2){
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
		.image = image,
	}, &reqs);

	// Render targets get their own allocation whether or not the driver asks:
	// vendors advise it (NVIDIA), and a resize then frees exactly
	bool target = image_usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	VkDeviceSize offset;
	skr_err_     err = _skr_mem_alloc(&reqs.memoryRequirements, target || dedicated.prefersDedicatedAllocation || dedicated.requiresDedicatedAllocation,
		true, VK_NULL_HANDLE, image, usage, category, out_mem, &offset);
	if (err != skr_err_success) return err;

	VkResult vr = vkBindImageMemory(_skr_vk.device, image, _block(out_mem->block)->memory, offset);
	if (vr != VK_SUCCESS) {
		skr_log(skr_log_critical, "vkBindImageMemory: 0x%X", (uint32_t)vr);
		_skr_mem_free(*out_mem);
		*out_mem = (_skr_mem_t){0};
		return skr_err_device_error;
	}
	return skr_err_success;
}

uint32_t _skr_mem_find_type(uint32_t type_bits, _skr_mem_usage_ usage) {
	for (uint32_t i = 0; i < _mem.order_count[usage]; i++)
		if (type_bits & (1u << _mem.order[usage][i])) return _mem.order[usage][i];
	// External memory decides its own type, a protected one included, so the
	// ranking's exclusions don't apply
	for (uint32_t t = 0; t < _mem.props.memoryTypeCount; t++)
		if (type_bits & (1u << t)) return t;
	return UINT32_MAX;
}

bool _skr_mem_has_lazy(void) {
	return _mem.order_count[_skr_mem_usage_lazy] > 0 &&
		(_mem.props.memoryTypes[_mem.order[_skr_mem_usage_lazy][0]].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
}

_skr_mem_t _skr_mem_import(VkDeviceMemory memory, VkDeviceSize size) {
	uint32_t idx = _block_new();
	if (idx == 0) return (_skr_mem_t){0};
	*_block(idx) = (_skr_mem_block_t){
		.memory = memory,
		.size   = size,
		.kind   = _skr_mem_block_external,
	};
	_skr_add_u64(&_mem.external, size);
	return (_skr_mem_t){ .block = idx };
}

void _skr_mem_free(_skr_mem_t mem) {
	if (mem.block == 0) return;
	_skr_mem_block_t*  block    = _block(mem.block);
	skr_mem_category_  category = (skr_mem_category_)mem.category;

	if (block->kind != _skr_mem_block_pooled) {
		switch (block->kind) {
		case _skr_mem_block_external: _stat_sub(&_mem.external, block->size); break;
		case _skr_mem_block_lazy:     _stat_sub(&_mem.lazy,     block->size); break;
		default:
			_stat_sub   (&_mem.reserved, block->size);
			_stat_unused(category,       block->size);
			_skr_add_u32(&_mem.dedicated_count,  (uint32_t)-1);
			_skr_add_u32(&_mem.allocation_count, (uint32_t)-1);
			break;
		}
		VkDeviceMemory memory = block->memory;
		_block_release(mem.block);
		vkFreeMemory(_skr_vk.device, memory, NULL);
		return;
	}

	// A pool keeps one empty block for a while, so churn at a block boundary
	// doesn't reach vkAllocateMemory. Any other block that empties goes back.
	_skr_mem_pool_t* pool = &_mem.pools[block->pool];
	mtx_lock(&pool->mutex);
	uint32_t size = _skr_tlsf_size(block->tlsf, mem.node);
	_skr_tlsf_free(block->tlsf, mem.node);
	bool release = false;
	if (block->tlsf->free_bytes == block->tlsf->size) {
		if (pool->empty_block == 0) {
			_skr_store_u32(&pool->empty_block, mem.block);
			pool->empty_frame = _skr_vk.frame;
		} else {
			_pool_remove(pool, mem.block);
			release = true;
		}
	}
	mtx_unlock(&pool->mutex);

	_stat_unused(category, size);
	_skr_add_u32(&_mem.allocation_count, (uint32_t)-1);
	if (release) _block_destroy_pooled(mem.block);
}

void _skr_mem_tick(uint32_t frame) {
	for (uint32_t p = 0; p < _mem.props.memoryTypeCount * 2; p++) {
		_skr_mem_pool_t* pool = &_mem.pools[p];
		if (_skr_load_u32(&pool->empty_block) == 0) continue;
		mtx_lock(&pool->mutex);
		uint32_t idx = 0;
		if (pool->empty_block != 0 && frame - pool->empty_frame > _SKR_MEM_EMPTY_FRAMES) {
			idx = pool->empty_block;
			_skr_store_u32(&pool->empty_block, 0);
			_pool_remove(pool, idx);
		}
		mtx_unlock(&pool->mutex);
		if (idx != 0) _block_destroy_pooled(idx);
	}
}

///////////////////////////////////////////////////////////////////////////////

void skr_mem_get_stats(skr_mem_stats_t* out_stats) {
	*out_stats = (skr_mem_stats_t){
		.reserved_bytes   = _skr_load_u64(&_mem.reserved),
		.used_bytes       = _skr_load_u64(&_mem.used),
		.peak_used_bytes  = _skr_load_u64(&_mem.peak),
		.lazy_bytes       = _skr_load_u64(&_mem.lazy),
		.external_bytes   = _skr_load_u64(&_mem.external),
		.block_count      = _skr_load_u32(&_mem.block_count),
		.dedicated_count  = _skr_load_u32(&_mem.dedicated_count),
		.allocation_count = _skr_load_u32(&_mem.allocation_count),
	};
	for (int32_t c = 0; c < skr_mem_category_max; c++)
		out_stats->category_bytes[c] = _skr_load_u64(&_mem.category[c]);

	if (!_mem.has_budget) return;
	VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
	VkPhysicalDeviceMemoryProperties2         props  = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &budget };
	vkGetPhysicalDeviceMemoryProperties2(_skr_vk.physical_device, &props);
	for (uint32_t h = 0; h < props.memoryProperties.memoryHeapCount; h++) {
		if (!(props.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
		out_stats->budget_bytes       += budget.heapBudget[h];
		out_stats->device_usage_bytes += budget.heapUsage [h];
	}
}
