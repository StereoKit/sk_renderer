// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "sk_renderer.h"
#include "_sk_renderer.h"
#include "skr_transient.h"

#include <threads.h>

#define SKR_TRANSIENT_IDLE_FRAMES      30
#define SKR_TRANSIENT_INITIAL_CAPACITY  4

typedef struct {
	VkFormat   vk_format;
	int32_t    width, height, layers;
	bool       depth;
	skr_tex_t  tex;
	uint32_t   last_used_frame;
	bool       in_use;
} _skr_transient_entry_t;

typedef struct {
	mtx_t                     mutex;
	// Entries are individually allocated so acquired skr_tex_t pointers stay
	// valid while the array grows.
	_skr_transient_entry_t**  entries;
	uint32_t                  count;
	uint32_t                  capacity;
} _skr_transient_pool_t;

static _skr_transient_pool_t _pool;

///////////////////////////////////////////////////////////////////////////////

static void _skr_transient_entry_destroy(_skr_transient_entry_t* e) {
	// Not skr_tex_destroy — these textures never acquired a sampler, and the
	// destroy list executes LIFO, so push memory, image, then view for
	// dependency-ordered retirement (view → image → memory).
	_skr_cmd_destroy_memory    (NULL, e->tex.memory);
	_skr_cmd_destroy_image     (NULL, e->tex.image);
	_skr_cmd_destroy_image_view(NULL, e->tex.view);
	_skr_free(e);
}

static bool _skr_transient_entry_create(_skr_transient_entry_t* e, VkFormat format, int32_t width, int32_t height, int32_t layers, bool depth) {
	VkImageUsageFlags usage = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT
		| (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

	VkResult vr = vkCreateImage(_skr_vk.device, &(VkImageCreateInfo){
		.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType     = VK_IMAGE_TYPE_2D,
		.format        = format,
		.extent        = { (uint32_t)width, (uint32_t)height, 1 },
		.mipLevels     = 1,
		.arrayLayers   = (uint32_t)layers,
		.samples       = VK_SAMPLE_COUNT_1_BIT,
		.tiling        = VK_IMAGE_TILING_OPTIMAL,
		.usage         = usage,
		.sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	}, NULL, &e->tex.image);
	if (vr != VK_SUCCESS) {
		skr_log(skr_log_critical, "Transient attachment vkCreateImage: 0x%X", (uint32_t)vr);
		return false;
	}
	if (_skr_allocate_image_memory(_skr_vk.device, _skr_vk.physical_device, e->tex.image, true, &e->tex.memory) == VK_NULL_HANDLE) {
		skr_log(skr_log_critical, "Transient attachment memory allocation failed");
		vkDestroyImage(_skr_vk.device, e->tex.image, NULL);
		return false;
	}
	vkBindImageMemory(_skr_vk.device, e->tex.image, e->tex.memory, 0);

	// Depth views cover only the depth aspect — that's all a depth resolve
	// writes and all a SubpassInput descriptor may reference.
	VkImageAspectFlags aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	vr = vkCreateImageView(_skr_vk.device, &(VkImageViewCreateInfo){
		.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image      = e->tex.image,
		.viewType   = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
		.format     = format,
		.subresourceRange = {
			.aspectMask     = aspect,
			.baseMipLevel   = 0,
			.levelCount     = 1,
			.baseArrayLayer = 0,
			.layerCount     = (uint32_t)layers,
		},
	}, NULL, &e->tex.view);
	if (vr != VK_SUCCESS) {
		skr_log(skr_log_critical, "Transient attachment vkCreateImageView: 0x%X", (uint32_t)vr);
		vkDestroyImage(_skr_vk.device, e->tex.image,  NULL);
		vkFreeMemory  (_skr_vk.device, e->tex.memory, NULL);
		return false;
	}

	e->tex.format              = skr_tex_fmt_from_native(format);
	e->tex.size                = (skr_vec3i_t){ width, height, 1 };
	e->tex.samples             = VK_SAMPLE_COUNT_1_BIT;
	e->tex.mip_levels          = 1;
	e->tex.layer_count         = (uint32_t)layers;
	e->tex.aspect_mask         = aspect;
	e->tex.current_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
	e->tex.is_transient_discard = true;

	e->vk_format = format;
	e->width     = width;
	e->height    = height;
	e->layers    = layers;
	e->depth     = depth;
	return true;
}

///////////////////////////////////////////////////////////////////////////////

void _skr_transient_pool_init(void) {
	mtx_init(&_pool.mutex, mtx_plain);

	_pool.capacity = SKR_TRANSIENT_INITIAL_CAPACITY;
	_pool.count    = 0;
	_pool.entries  = _skr_calloc(_pool.capacity, sizeof(_skr_transient_entry_t*));
}

void _skr_transient_pool_shutdown(void) {
	// No mutex needed — caller guarantees device is idle and no other threads are active
	for (uint32_t i = 0; i < _pool.count; i++)
		_skr_transient_entry_destroy(_pool.entries[i]);

	mtx_destroy(&_pool.mutex);
	_skr_free(_pool.entries);
	_pool = (_skr_transient_pool_t){0};
}

skr_tex_t* _skr_transient_acquire(VkFormat format, int32_t width, int32_t height, int32_t layers, bool depth) {
	mtx_lock(&_pool.mutex);

	// Reuse a matching idle entry. Prior-pass GPU work may still be reading
	// it — the multisubpass renderpass's EXTERNAL dependencies order this
	// pass's UNDEFINED transition after that work.
	for (uint32_t i = 0; i < _pool.count; i++) {
		_skr_transient_entry_t* e = _pool.entries[i];
		if (!e->in_use
			&& e->vk_format == format
			&& e->width     == width
			&& e->height    == height
			&& e->layers    == layers
			&& e->depth     == depth) {
			e->in_use          = true;
			e->last_used_frame = _skr_vk.frame;
			mtx_unlock(&_pool.mutex);
			return &e->tex;
		}
	}

	// Grow if needed
	if (_pool.count >= _pool.capacity) {
		uint32_t new_cap = _pool.capacity * 2;
		_skr_transient_entry_t** new_entries = _skr_realloc(_pool.entries, new_cap * sizeof(_skr_transient_entry_t*));
		if (!new_entries) {
			skr_log(skr_log_critical, "Transient pool grow failed");
			mtx_unlock(&_pool.mutex);
			return NULL;
		}
		_pool.entries  = new_entries;
		_pool.capacity = new_cap;
	}

	_skr_transient_entry_t* e = _skr_calloc(1, sizeof(_skr_transient_entry_t));
	if (!e || !_skr_transient_entry_create(e, format, width, height, layers, depth)) {
		_skr_free(e);
		mtx_unlock(&_pool.mutex);
		return NULL;
	}
	e->in_use          = true;
	e->last_used_frame = _skr_vk.frame;
	_pool.entries[_pool.count++] = e;

	mtx_unlock(&_pool.mutex);
	return &e->tex;
}

void _skr_transient_release(skr_tex_t* transient) {
	if (!transient) return;

	mtx_lock(&_pool.mutex);

	for (uint32_t i = 0; i < _pool.count; i++) {
		_skr_transient_entry_t* e = _pool.entries[i];
		if (&e->tex == transient) {
			e->in_use          = false;
			e->last_used_frame = _skr_vk.frame;
			break;
		}
	}

	mtx_unlock(&_pool.mutex);
}

void _skr_transient_pool_tick(void) {
	mtx_lock(&_pool.mutex);

	uint32_t frame = _skr_vk.frame;
	for (uint32_t i = 0; i < _pool.count; ) {
		_skr_transient_entry_t* e = _pool.entries[i];
		if (!e->in_use && (frame - e->last_used_frame) > SKR_TRANSIENT_IDLE_FRAMES) {
			// Deferred destroy chain holds the resources until fences clear
			_skr_transient_entry_destroy(e);

			// Swap-remove
			_pool.entries[i] = _pool.entries[_pool.count - 1];
			_pool.entries[_pool.count - 1] = NULL;
			_pool.count--;
			continue;
		}
		i++;
	}

	mtx_unlock(&_pool.mutex);
}
