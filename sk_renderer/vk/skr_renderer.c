// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"
#include "skr_pipeline.h"
#include "skr_conversions.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

// Query pool has 2 queries per frame (start/end timestamps)
#define SKR_QUERIES_PER_FRAME 2

// Maximum global buffer/texture binding slots
#define SKR_MAX_GLOBAL_BINDINGS 16

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

static VkFramebuffer _skr_get_or_create_framebuffer(VkDevice device, skr_tex_t* cache_target, VkRenderPass render_pass, skr_tex_t* color, skr_tex_t* depth, skr_tex_t* opt_resolve, bool has_depth) {
	VkFramebuffer* cached_fb = has_depth
		? &cache_target->framebuffer_depth
		: &cache_target->framebuffer;

	// Check if we have a cached framebuffer for this render pass
	if (*cached_fb != VK_NULL_HANDLE && cache_target->framebuffer_pass == render_pass) {
		return *cached_fb;
	}

	// Destroy old cached framebuffer if render pass changed
	if (*cached_fb != VK_NULL_HANDLE) {
		_skr_cmd_destroy_framebuffer(NULL, *cached_fb);
	}

	// Create and cache new framebuffer
	*cached_fb = _skr_create_framebuffer(device, render_pass, color, depth, opt_resolve);
	cache_target->framebuffer_pass = render_pass;
	return *cached_fb;
}

///////////////////////////////////////////////////////////////////////////////
// Deferred Texture Transition System
///////////////////////////////////////////////////////////////////////////////

// Queue a texture for transition (will be flushed before next render pass)
void _skr_tex_transition_enqueue(skr_tex_t* ref_tex, uint8_t type) {
	if (!ref_tex || !ref_tex->image) return;

	// Check if already queued (avoid duplicates)
	for (uint32_t i = 0; i < _skr_vk.pending_transition_count; i++) {
		if (_skr_vk.pending_transitions[i] == ref_tex) {
			// Update type if needed (storage takes priority over shader_read)
			if (type > _skr_vk.pending_transition_types[i]) {
				_skr_vk.pending_transition_types[i] = type;
			}
			return;
		}
	}

	// Grow array if needed
	if (_skr_vk.pending_transition_count >= _skr_vk.pending_transition_capacity) {
		uint32_t new_capacity = _skr_vk.pending_transition_capacity == 0 ? 16 : _skr_vk.pending_transition_capacity * 2;
		_skr_vk.pending_transitions      = _skr_realloc(_skr_vk.pending_transitions, new_capacity * sizeof(skr_tex_t*));
		_skr_vk.pending_transition_types = _skr_realloc(_skr_vk.pending_transition_types, new_capacity * sizeof(uint8_t));
		_skr_vk.pending_transition_capacity = new_capacity;
	}

	// Add to queue
	_skr_vk.pending_transitions     [_skr_vk.pending_transition_count] = ref_tex;
	_skr_vk.pending_transition_types[_skr_vk.pending_transition_count] = type;
	_skr_vk.pending_transition_count++;
}

// Flush all pending texture transitions (called before render pass begins)
static void _skr_flush_texture_transitions(VkCommandBuffer cmd) {
	for (uint32_t i = 0; i < _skr_vk.pending_transition_count; i++) {
		skr_tex_t* tex  = _skr_vk.pending_transitions[i];
		uint8_t    type = _skr_vk.pending_transition_types[i];

		if (type == 1) {  // storage
			_skr_tex_transition_for_storage(cmd, tex);
		} else {  // shader_read
			_skr_tex_transition_for_shader_read(cmd, tex, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
		}
	}

	// Clear the queue
	_skr_vk.pending_transition_count = 0;
}

///////////////////////////////////////////////////////////////////////////////
// Rendering
///////////////////////////////////////////////////////////////////////////////

void skr_renderer_frame_begin() {
	_skr_vk.in_frame = true;

	// Start a command buffer batch for this frame
	VkCommandBuffer cmd = _skr_cmd_begin().cmd;

	// Reset and write start timestamp
	uint32_t query_start = _skr_vk.flight_idx * SKR_QUERIES_PER_FRAME;
	vkCmdResetQueryPool(cmd, _skr_vk.timestamp_pool, query_start, SKR_QUERIES_PER_FRAME);
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, _skr_vk.timestamp_pool, query_start);
}

void skr_renderer_frame_end(skr_surface_t** opt_surfaces, uint32_t count) {
	if (!_skr_vk.in_frame) {
		skr_log(skr_log_warning, "skr_renderer_frame_end called outside frame");
		return;
	}

	assert(count <= SKR_MAX_SURFACES && "Maximum surfaces supported for VR stereo");

	// Write end timestamp
	VkCommandBuffer cmd = _skr_cmd_acquire().cmd;
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, _skr_vk.timestamp_pool, _skr_vk.flight_idx * 2 + 1);
	_skr_cmd_release(cmd);

	// Gather semaphores and transition surfaces
	VkSemaphore wait_semaphores  [SKR_MAX_SURFACES];
	VkSemaphore signal_semaphores[SKR_MAX_SURFACES];

	for (uint32_t i = 0; i < count; i++) {
		skr_surface_t* surface = opt_surfaces[i];

		// Transition swapchain image to PRESENT_SRC_KHR
		cmd = _skr_cmd_acquire().cmd;
		skr_tex_t* swapchain_image = &surface->images[surface->current_image];
		_skr_tex_transition(cmd, swapchain_image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
		_skr_cmd_release   (cmd);

		wait_semaphores  [i] = surface->semaphore_acquire[surface->frame_idx];
		signal_semaphores[i] = surface->semaphore_submit [surface->current_image];
	}

	// Submit and get future
	skr_future_t future = _skr_cmd_end_submit(
		count > 0 ? wait_semaphores   : NULL, count,
		count > 0 ? signal_semaphores : NULL, count
	);

	// Record future in all surfaces for their current frame_idx
	for (uint32_t i = 0; i < count; i++) {
		opt_surfaces[i]->frame_future[opt_surfaces[i]->frame_idx] = future;
	}

	// Read timestamps from N-frames-ago (triple buffering delay)
	if (_skr_vk.frame >= SKR_MAX_FRAMES_IN_FLIGHT) {
		uint32_t prev_flight = (_skr_vk.flight_idx + 1) % SKR_MAX_FRAMES_IN_FLIGHT;
		uint32_t query_start = prev_flight * SKR_QUERIES_PER_FRAME;

		VkResult result = vkGetQueryPoolResults(
			_skr_vk.device, _skr_vk.timestamp_pool, query_start, SKR_QUERIES_PER_FRAME,
			sizeof(uint64_t) * SKR_QUERIES_PER_FRAME, _skr_vk.frame_timestamps[prev_flight],
			sizeof(uint64_t), VK_QUERY_RESULT_64_BIT
		);
		_skr_vk.timestamps_valid[prev_flight] = (result == VK_SUCCESS);
	}

	_skr_vk.in_frame = false;
	_skr_vk.frame++;
	_skr_vk.flight_idx = _skr_vk.frame % SKR_MAX_FRAMES_IN_FLIGHT;
}

void skr_renderer_begin_pass(skr_tex_t* color, skr_tex_t* depth, skr_tex_t* opt_resolve, skr_clear_ clear, skr_vec4_t clear_color, float clear_depth, uint32_t clear_stencil) {
	// Require at least one attachment (color or depth)
	if (!color && !depth) return;

	// Lock pipeline cache for the duration of this render pass.
	// This protects all pipeline get operations during drawing.
	// Unlocked in skr_renderer_end_pass.
	_skr_pipeline_lock();

	VkCommandBuffer cmd = _skr_cmd_acquire().cmd;

	// Flush all pending texture transitions BEFORE starting render pass
	// This prevents barriers inside render pass which require self-dependencies
	_skr_flush_texture_transitions(cmd);

	// Register render pass format with pipeline system
	skr_pipeline_renderpass_key_t rp_key = {
		.color_format    = color                                           ? skr_tex_fmt_to_native(color->format)         : VK_FORMAT_UNDEFINED,
		.depth_format    = depth                                           ? skr_tex_fmt_to_native(depth->format)         : VK_FORMAT_UNDEFINED,
		.resolve_format  = (opt_resolve && color && color->samples > VK_SAMPLE_COUNT_1_BIT) ? skr_tex_fmt_to_native(opt_resolve->format) : VK_FORMAT_UNDEFINED,
		.samples         = color ? color->samples : (depth ? depth->samples : VK_SAMPLE_COUNT_1_BIT),
		.depth_store_op  = (depth && (depth->flags & skr_tex_flags_readable)) ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.color_load_op   = (clear & skr_clear_color) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
	};
	_skr_vk.current_renderpass_idx = _skr_pipeline_register_renderpass(&rp_key);

	// Get render pass from pipeline system
	VkRenderPass render_pass = _skr_pipeline_get_renderpass(_skr_vk.current_renderpass_idx);
	if (render_pass == VK_NULL_HANDLE) { _skr_pipeline_unlock(); return; }

	// Determine which texture to use for framebuffer caching
	// Priority: resolve target (for MSAA) > color > depth
	skr_tex_t* fb_cache_target = color;
	if (opt_resolve && rp_key.samples > VK_SAMPLE_COUNT_1_BIT) {
		fb_cache_target = opt_resolve;  // Use resolve target for MSAA
	} else if (!color) {
		fb_cache_target = depth;  // Depth-only pass
	}

	// Get or create cached framebuffer
	VkFramebuffer framebuffer = _skr_get_or_create_framebuffer(_skr_vk.device, fb_cache_target, render_pass, color, depth, opt_resolve, depth != NULL);

	if (framebuffer == VK_NULL_HANDLE) { _skr_pipeline_unlock(); return; }

	// Transition depth texture to attachment layout if needed
	// Automatic system handles the optimization:
	// - Non-readable depth (transient_discard=true): Uses UNDEFINED oldLayout (tile GPU optimization)
	// - Readable depth: Properly tracks previous layout
	if (depth && (depth->flags & skr_tex_flags_writeable)) {
		_skr_tex_transition(cmd, depth,
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
	}

	// Note: Color attachments use render pass implicit transitions (initialLayout/finalLayout)
	// We'll notify the system after vkCmdBeginRenderPass about the layout change

	// Setup clear values
	// Need to match attachment count: [color], [resolve], [depth]
	VkClearValue clear_values[3];
	uint32_t     clear_value_count = 0;

	if (color) {
		if (clear & skr_clear_color) {
			clear_values[clear_value_count] = (VkClearValue){ .color = {.float32 = {clear_color.x, clear_color.y, clear_color.z, clear_color.w}} };
		}
		clear_value_count++; // Color attachment needs an entry

		if (opt_resolve && rp_key.samples > VK_SAMPLE_COUNT_1_BIT) {
			// Resolve has loadOp = DONT_CARE, but still needs an entry
			clear_value_count++;
		}
	}

	if (depth) {
		if (clear & (skr_clear_depth | skr_clear_stencil)) {
			clear_values[clear_value_count] = (VkClearValue){ .depthStencil = {.depth = clear_depth, .stencil = clear_stencil} };
		}
		clear_value_count++;
	}

	// Determine render area from whichever attachment is available
	uint32_t render_width  = color ? color->size.x : depth->size.x;
	uint32_t render_height = color ? color->size.y : depth->size.y;

	// Begin render pass
	vkCmdBeginRenderPass(cmd, &(VkRenderPassBeginInfo){
		.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass      = render_pass,
		.framebuffer     = framebuffer,
		.clearValueCount = clear_value_count,
		.pClearValues    = clear_values,
		.renderArea      = {
			.extent = {render_width, render_height}
		},
	}, VK_SUBPASS_CONTENTS_INLINE);

	// Notify automatic system about render pass implicit layout transitions
	// Render pass transitions color to COLOR_ATTACHMENT_OPTIMAL
	if (color) {
		_skr_tex_transition_notify_layout(color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	}
	// Resolve target (if used) goes to COLOR_ATTACHMENT_OPTIMAL
	if (opt_resolve && rp_key.samples > VK_SAMPLE_COUNT_1_BIT) {
		_skr_tex_transition_notify_layout(opt_resolve, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	}
	// Depth remains in DEPTH_STENCIL_ATTACHMENT_OPTIMAL (render pass preserves it)
	if (depth) {
		_skr_tex_transition_notify_layout(depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	// Store current textures for end_pass layout transitions
	_skr_vk.current_color_texture = color;
	_skr_vk.current_depth_texture = depth;

	_skr_cmd_release(cmd);
}

void skr_renderer_end_pass() {
	VkCommandBuffer cmd = _skr_cmd_acquire().cmd;
	vkCmdEndRenderPass(cmd);

	// Transition readable color attachments to shader-read layout for next use
	// Automatic system handles this - tracks that color is currently in COLOR_ATTACHMENT_OPTIMAL
	if (_skr_vk.current_color_texture && (_skr_vk.current_color_texture->flags & skr_tex_flags_readable)) {
		_skr_tex_transition_for_shader_read(cmd, _skr_vk.current_color_texture,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	}

	// Transition readable depth texture to shader-read layout for next use (e.g., shadow maps)
	// Automatic system handles this - tracks that depth is currently in DEPTH_STENCIL_ATTACHMENT_OPTIMAL
	if (_skr_vk.current_depth_texture && (_skr_vk.current_depth_texture->flags & skr_tex_flags_readable)) {
		_skr_tex_transition_for_shader_read(cmd, _skr_vk.current_depth_texture,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	}

	_skr_vk.current_color_texture = NULL;
	_skr_vk.current_depth_texture = NULL;
	_skr_cmd_release(cmd);

	// Unlock pipeline cache (locked in skr_renderer_begin_pass)
	_skr_pipeline_unlock();
}

void skr_renderer_set_global_constants(int32_t bind, const skr_buffer_t* buffer) {
	if (bind < 0 || bind >= SKR_MAX_GLOBAL_BINDINGS) {
		if (bind >= SKR_MAX_GLOBAL_BINDINGS) {
			skr_log(skr_log_critical, "Global buffer binding %d exceeds maximum of %d slots", bind, SKR_MAX_GLOBAL_BINDINGS);
		}
		return;
	}
	_skr_vk.global_buffers[bind] = (skr_buffer_t*)buffer;
}

void skr_renderer_set_global_texture(int32_t bind, const skr_tex_t* tex) {
	if (bind < 0 || bind >= SKR_MAX_GLOBAL_BINDINGS) {
		if (bind >= SKR_MAX_GLOBAL_BINDINGS) {
			skr_log(skr_log_critical, "Global texture binding %d exceeds maximum of %d slots", bind, SKR_MAX_GLOBAL_BINDINGS);
		}
		return;
	}
	_skr_vk.global_textures[bind] = (skr_tex_t*)tex;

	// Queue transition for this global texture (only if needed)
	// It will be flushed before the next render pass begins
	if (tex) {
		uint8_t type = (tex->flags & skr_tex_flags_compute) ? 1 : 0;  // storage : shader_read
		if (_skr_tex_needs_transition(tex, type)) {
			_skr_tex_transition_enqueue((skr_tex_t*)tex, type);
		}
	}
}

void skr_renderer_set_viewport(skr_rect_t viewport) {
	VkCommandBuffer cmd = _skr_cmd_acquire().cmd;
	// Negative height flips Y to match DirectX/OpenGL conventions (VK_KHR_maintenance1, core in 1.1)
	vkCmdSetViewport(cmd, 0, 1, &(VkViewport){
		.x        = viewport.x,
		.y        = viewport.y + viewport.h,
		.width    = viewport.w,
		.height   = -viewport.h,
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	});
	_skr_cmd_release(cmd);
}

void skr_renderer_set_scissor(skr_recti_t scissor) {
	VkCommandBuffer cmd = _skr_cmd_acquire().cmd;
	vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){
		.offset = {scissor.x, scissor.y},
		.extent = {(uint32_t)scissor.w, (uint32_t)scissor.h},
	});
	_skr_cmd_release(cmd);
}

void skr_renderer_blit(skr_material_t* material, skr_tex_t* to, skr_recti_t bounds_px) {
	if (!material || !to) return;
	if (!skr_material_is_valid(material) || !skr_tex_is_valid(to)) return;

	// Determine if this is a cubemap, array, or regular 2D texture
	bool     is_cubemap  = (to->flags & skr_tex_flags_cubemap) != 0;
	bool     is_array    = (to->flags & skr_tex_flags_array  ) != 0;
	uint32_t layer_count = to->layer_count;

	// Determine if this is a full-image blit or partial
	bool is_full_blit = 
		(bounds_px.w <= 0 || bounds_px.h <= 0) ||
		(bounds_px.x == 0 && bounds_px.y == 0  &&
		 bounds_px.w == to->size.x && bounds_px.h == to->size.y);

	uint32_t width  = bounds_px.w > 0 ? bounds_px.w : to->size.x;
	uint32_t height = bounds_px.h > 0 ? bounds_px.h : to->size.y;

	// Lock pipeline cache for this blit operation
	_skr_pipeline_lock();

	// Register render pass format with pipeline system
	// Use DONT_CARE for full blit (discard previous contents), LOAD for partial (preserve)
	skr_pipeline_renderpass_key_t rp_key = {
		.color_format   = skr_tex_fmt_to_native(to->format),
		.depth_format   = VK_FORMAT_UNDEFINED,
		.resolve_format = VK_FORMAT_UNDEFINED,
		.samples        = to->samples,
		.depth_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE,  // No depth in blit
		.color_load_op  = is_full_blit ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD,
	};
	int32_t renderpass_idx = _skr_pipeline_register_renderpass(&rp_key);
	int32_t vert_idx       = _skr_pipeline_register_vertformat((skr_vert_type_t){});

	// Get render pass from pipeline system
	VkRenderPass render_pass = _skr_pipeline_get_renderpass(renderpass_idx);
	if (render_pass == VK_NULL_HANDLE) {
		_skr_pipeline_unlock();
		return;
	}

	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();

	// Build per-draw descriptor writes
	VkWriteDescriptorSet   writes      [32];
	VkDescriptorBufferInfo buffer_infos[16];
	VkDescriptorImageInfo  image_infos [16];
	uint32_t write_ct  = 0;
	uint32_t buffer_ct = 0;
	uint32_t image_ct  = 0;

	skr_bump_result_t param_bump = {0};
	if (material->param_buffer_size > 0) {
		param_bump = _skr_bump_alloc_write(ctx.const_bump, material->param_buffer, material->param_buffer_size);
		if (param_bump.buffer) {
			buffer_infos[buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = param_bump.buffer->buffer,
				.offset = param_bump.offset,
				.range  = material->param_buffer_size,
			};
			writes[write_ct++] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = SKR_BIND_SHIFT_BUFFER + _skr_vk.bind_settings.material_slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo     = &buffer_infos[buffer_ct++],
			};
		}
	}

	// Material texture and buffer binds
	const sksc_shader_meta_t* meta = material->key.shader->meta;
	const int32_t ignore_slots[] = { SKR_BIND_SHIFT_BUFFER + _skr_vk.bind_settings.material_slot };

	_skr_bind_pool_lock();
	skr_material_bind_t* mat_binds = _skr_bind_pool_get(material->bind_start);
	int32_t fail_idx = _skr_material_add_writes(mat_binds, material->bind_count, ignore_slots, sizeof(ignore_slots)/sizeof(ignore_slots[0]),
		writes,       sizeof(writes      )/sizeof(writes      [0]),
		buffer_infos, sizeof(buffer_infos)/sizeof(buffer_infos[0]),
		image_infos,  sizeof(image_infos )/sizeof(image_infos [0]),
		&write_ct, &buffer_ct, &image_ct);
	if (fail_idx >= 0) {
		_skr_bind_pool_unlock();
		skr_log(skr_log_critical, "Blit missing binding '%s' in shader '%s'", _skr_material_bind_name(meta, fail_idx), meta->name);
		return;
	}

	// Transition any source textures in material to shader-read layout
	for (uint32_t i=0; i<meta->resource_count; i++) {
		skr_material_bind_t* res = &mat_binds[meta->buffer_count + i];
		if (res->texture)
			_skr_tex_transition_for_shader_read(ctx.cmd, res->texture, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	}
	_skr_bind_pool_unlock();

	// Transition target texture to color attachment layout
	_skr_tex_transition(ctx.cmd, to, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

	// Create framebuffer - layered for cubemaps/arrays, cached for 2D
	VkFramebuffer framebuffer   = VK_NULL_HANDLE;
	VkImageView   temp_view     = VK_NULL_HANDLE;
	uint32_t      draw_instances = 1;

	if (is_cubemap || is_array) {
		// Layered rendering: create multi-layer view and framebuffer for single instanced draw
		// IMPORTANT: For framebuffer attachments with SV_RenderTargetArrayIndex, we must use
		// VK_IMAGE_VIEW_TYPE_2D_ARRAY even for cubemaps. Cube views are for sampling, but for
		// rendering to individual layers we treat the cubemap as a 6-layer 2D array.
		VkImageViewCreateInfo view_info = {
			.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image      = to->image,
			.viewType   = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
			.format     = skr_tex_fmt_to_native(to->format),
			.subresourceRange = {
				.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = layer_count,
			},
		};
		VkResult vr = vkCreateImageView(_skr_vk.device, &view_info, NULL, &temp_view);
		if (vr != VK_SUCCESS) {
			SKR_VK_CHECK_NRET(vr, "vkCreateImageView");
			_skr_cmd_release(ctx.cmd);
			return;
		}

		VkFramebufferCreateInfo fb_info = {
			.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass      = render_pass,
			.attachmentCount = 1,
			.pAttachments    = &temp_view,
			.width           = width,
			.height          = height,
			.layers          = layer_count,
		};
		vr = vkCreateFramebuffer(_skr_vk.device, &fb_info, NULL, &framebuffer);
		if (vr != VK_SUCCESS) {
			SKR_VK_CHECK_NRET(vr, "vkCreateFramebuffer");
			vkDestroyImageView(_skr_vk.device, temp_view, NULL);
			_skr_cmd_release(ctx.cmd);
			return;
		}

		draw_instances = layer_count;  // One instance per layer
	} else {
		// Regular 2D: use cached framebuffer
		framebuffer = _skr_get_or_create_framebuffer(_skr_vk.device, to, render_pass, to, NULL, NULL, false);
		if (framebuffer == VK_NULL_HANDLE) {
			_skr_cmd_release(ctx.cmd);
			return;
		}
	}

	// Common rendering path for all texture types
	vkCmdBeginRenderPass(ctx.cmd, &(VkRenderPassBeginInfo){
		.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass  = render_pass,
		.framebuffer = framebuffer,
		.renderArea  = {{bounds_px.x, bounds_px.y}, {width, height}},
	}, VK_SUBPASS_CONTENTS_INLINE);

	VkPipeline pipeline = _skr_pipeline_get(material->pipeline_material_idx, renderpass_idx, vert_idx);
	if (pipeline != VK_NULL_HANDLE) {
		vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vkCmdSetViewport (ctx.cmd, 0, 1, &(VkViewport){(float)bounds_px.x, (float)bounds_px.y, (float)width, (float)height, 0.0f, 1.0f});
		vkCmdSetScissor  (ctx.cmd, 0, 1, &(VkRect2D  ){{bounds_px.x, bounds_px.y}, {width, height}});

		_skr_bind_descriptors(ctx.cmd, ctx.descriptor_pool, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                      _skr_pipeline_get_layout(material->pipeline_material_idx),
		                      _skr_pipeline_get_descriptor_layout(material->pipeline_material_idx),
		                      writes, write_ct);

		// Draw fullscreen triangle - instanced for cubemaps/arrays, single for 2D
		vkCmdDraw(ctx.cmd, 3, draw_instances, 0, 0);
	}

	vkCmdEndRenderPass(ctx.cmd);

	// Clean up temporary resources (layered view/framebuffer)
	if (temp_view != VK_NULL_HANDLE) {
		_skr_cmd_destroy_framebuffer(ctx.destroy_list, framebuffer);
		_skr_cmd_destroy_image_view (ctx.destroy_list, temp_view);
	}

	// Transition target texture back to shader read layout
	// Automatic system tracks that it's currently in COLOR_ATTACHMENT_OPTIMAL
	_skr_tex_transition_for_shader_read(ctx.cmd, to, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	_skr_cmd_release(ctx.cmd);

	_skr_pipeline_unlock();
}

void skr_renderer_draw(skr_render_list_t* list, const void* system_data, uint32_t system_data_size, int32_t instance_multiplier) {
	if (!list || list->count == 0) return;
	instance_multiplier = (instance_multiplier < 1) ? 1 : instance_multiplier;

	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();
	VkCommandBuffer cmd = ctx.cmd;

	_skr_render_list_sort(list);
	// Material param data is already copied at add-time into list->material_data

	// Upload data to bump allocators from command context
	skr_bump_result_t system_bump   = {0};
	skr_bump_result_t material_bump = {0};
	skr_bump_result_t instance_bump = {0};

	if (system_data && system_data_size > 0) {
		system_bump = _skr_bump_alloc_write(ctx.const_bump, system_data, system_data_size);
	}
	if (list->material_data_used > 0) {
		material_bump = _skr_bump_alloc_write(ctx.const_bump, list->material_data, list->material_data_used);
	}
	if (list->instance_data_used > 0) {
		instance_bump = _skr_bump_alloc_write(ctx.storage_bump, list->instance_data, list->instance_data_used);
	}

	// Draw items with batching
	VkPipeline bound_pipeline = VK_NULL_HANDLE;
	for (uint32_t i = 0; i < list->count; ) {
		const skr_render_item_t* item = &list->items[i];

		// Get pipeline from the cache (using inlined indices)
		VkPipeline pipeline = _skr_pipeline_get(item->pipeline_material_idx, _skr_vk.current_renderpass_idx, item->pipeline_vert_idx);
		assert(pipeline != VK_NULL_HANDLE && "Is the Vertex format out of scope?");

		// Find consecutive items with same mesh/material/draw-params for batching
		// Compare inlined data instead of pointers
		uint32_t batch_count     = 1;
		uint32_t total_instances = item->instance_count;
		uint32_t total_inst_data = item->instance_data_size * item->instance_count;
		while (i + batch_count < list->count) {
			const skr_render_item_t* next = &list->items[i + batch_count];
			// Can only batch if mesh, material, AND draw parameters all match
			if (next->vertex_buffers[0]      != item->vertex_buffers[0]      ||
			    next->pipeline_material_idx  != item->pipeline_material_idx  ||
			    next->bind_start             != item->bind_start             ||
			    next->first_index            != item->first_index            ||
			    next->index_count            != item->index_count            ||
			    next->vertex_offset          != item->vertex_offset)
				break;
			total_instances += next->instance_count;
			total_inst_data += next->instance_data_size * next->instance_count;
			batch_count++;
		}

		// Bind pipeline if changed
		if (pipeline != bound_pipeline) {
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
			bound_pipeline = pipeline;
		}

		// Build per-draw descriptor writes
		VkWriteDescriptorSet   writes      [32];
		VkDescriptorBufferInfo buffer_infos[16];
		VkDescriptorImageInfo  image_infos [16];
		uint32_t write_ct  = 0;
		uint32_t buffer_ct = 0;
		uint32_t image_ct  = 0;

		// Material parameter buffer (using inlined param_buffer_size and param_data_offset)
		if (item->param_buffer_size > 0 && material_bump.buffer) {
			buffer_infos[buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = material_bump.buffer->buffer,
				.offset = material_bump.offset + item->param_data_offset,
				.range  = item->param_buffer_size,
			};
			writes[write_ct++] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = SKR_BIND_SHIFT_BUFFER + _skr_vk.bind_settings.material_slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo     = &buffer_infos[buffer_ct++],
			};
		}

		// System data buffer (using inlined has_system_buffer)
		if (item->has_system_buffer && system_bump.buffer) {
			buffer_infos[buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = system_bump.buffer->buffer,
				.offset = system_bump.offset,
				.range  = system_data_size,
			};
			writes[write_ct++] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = SKR_BIND_SHIFT_BUFFER + _skr_vk.bind_settings.system_slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo     = &buffer_infos[buffer_ct++],
			};
		}

		// Instance data buffer (using inlined instance_buffer_stride)
		if (item->instance_buffer_stride > 0 && instance_bump.buffer) {
			if (item->instance_data_size != item->instance_buffer_stride) {
				skr_log(skr_log_warning, "Instance data size mismatch: shader expects %u bytes, got %u bytes",
					item->instance_buffer_stride, item->instance_data_size);
			}
			buffer_infos[buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = instance_bump.buffer->buffer,
				.offset = instance_bump.offset + item->instance_offset,
				.range  = total_inst_data,
			};
			writes[write_ct++] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = SKR_BIND_SHIFT_TEXTURE + _skr_vk.bind_settings.instance_slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo     = &buffer_infos[buffer_ct++],
			};
		}

		const int32_t ignore_slots[] = {
			SKR_BIND_SHIFT_TEXTURE + _skr_vk.bind_settings.instance_slot,
			SKR_BIND_SHIFT_BUFFER  + _skr_vk.bind_settings.material_slot,
			SKR_BIND_SHIFT_BUFFER  + _skr_vk.bind_settings.system_slot };

		// Material texture and buffer binds (using inlined bind_start/bind_count)
		_skr_bind_pool_lock();
		const skr_material_bind_t* binds = _skr_bind_pool_get(item->bind_start);
		int32_t fail_idx = _skr_material_add_writes(binds, item->bind_count, ignore_slots, sizeof(ignore_slots)/sizeof(ignore_slots[0]),
			writes,       sizeof(writes      )/sizeof(writes      [0]),
			buffer_infos, sizeof(buffer_infos)/sizeof(buffer_infos[0]),
			image_infos,  sizeof(image_infos )/sizeof(image_infos [0]),
			&write_ct, &buffer_ct, &image_ct);

		if (fail_idx >= 0) {
			int32_t       slot = binds[fail_idx].bind.slot;
			skr_register_ type = (skr_register_)binds[fail_idx].bind.register_type;
			char          reg_char;
			int32_t       reg_num;
			switch (type) {
			case skr_register_constant:      reg_char = 'b'; reg_num = slot - SKR_BIND_SHIFT_BUFFER;  break;
			case skr_register_texture:
			case skr_register_read_buffer:   reg_char = 't'; reg_num = slot - SKR_BIND_SHIFT_TEXTURE; break;
			case skr_register_readwrite:
			case skr_register_readwrite_tex: reg_char = 'u'; reg_num = slot - SKR_BIND_SHIFT_UAV;     break;
			default:                         reg_char = '?'; reg_num = slot;                          break;
			}
			skr_log(skr_log_critical, "Draw call missing binding for register(%c%d)", reg_char, reg_num);
			_skr_bind_pool_unlock();
			i += batch_count;
			continue;
		}
		_skr_bind_pool_unlock();

		// Push all descriptors at once (using inlined pipeline_material_idx)
		_skr_bind_descriptors(cmd, ctx.descriptor_pool, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                      _skr_pipeline_get_layout(item->pipeline_material_idx),
		                      _skr_pipeline_get_descriptor_layout(item->pipeline_material_idx),
		                      writes, write_ct);

		// Bind vertex buffers (using inlined VkBuffer handles)
		if (item->vertex_buffer_count > 0) {
			VkBuffer     buffers[SKR_MAX_VERTEX_BUFFERS];
			VkDeviceSize offsets[SKR_MAX_VERTEX_BUFFERS];
			uint32_t     bind_count = 0;

			for (uint32_t j = 0; j < item->vertex_buffer_count; j++) {
				if (item->vertex_buffers[j] != VK_NULL_HANDLE) {
					buffers[bind_count] = item->vertex_buffers[j];
					offsets[bind_count] = 0;
					bind_count++;
				}
			}

			if (bind_count > 0) {
				vkCmdBindVertexBuffers(cmd, 0, bind_count, buffers, offsets);
			}
		}

		// Draw with instancing (using inlined mesh data)
		uint32_t draw_instances = total_instances * instance_multiplier;
		if (item->index_buffer != VK_NULL_HANDLE) {
			vkCmdBindIndexBuffer(cmd, item->index_buffer, 0, (VkIndexType)item->index_format);
			uint32_t draw_index_count = item->index_count > 0 ? (uint32_t)item->index_count : item->ind_count;
			vkCmdDrawIndexed(cmd, draw_index_count, draw_instances, item->first_index, item->vertex_offset, 0);
		} else {
			vkCmdDraw(cmd, item->vert_count, draw_instances, 0, 0);
		}

		i += batch_count;
	}
	_skr_cmd_release(cmd);
}

void skr_renderer_draw_mesh_immediate(skr_mesh_t* mesh, skr_material_t* material,
                                       int32_t first_index, int32_t index_count, int32_t vertex_offset,
                                       int32_t instance_count) {
	if (!mesh || !material) return;
	if (instance_count < 1) instance_count = 1;

	_skr_cmd_ctx_t ctx = _skr_cmd_acquire();
	VkCommandBuffer cmd = ctx.cmd;

	// Get pipeline
	VkPipeline pipeline = _skr_pipeline_get(material->pipeline_material_idx, _skr_vk.current_renderpass_idx, mesh->vert_type->pipeline_idx);
	assert(pipeline != VK_NULL_HANDLE && "Is the Vertex format out of scope?");

	// Bind pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	// Build descriptor writes
	VkWriteDescriptorSet   writes      [32];
	VkDescriptorBufferInfo buffer_infos[16];
	VkDescriptorImageInfo  image_infos [16];
	uint32_t write_ct  = 0;
	uint32_t buffer_ct = 0;
	uint32_t image_ct  = 0;

	// Upload material parameters to bump allocator if needed
	skr_bump_result_t material_bump = {0};
	if (material->param_buffer_size > 0) {
		material_bump = _skr_bump_alloc_write(ctx.const_bump, material->param_buffer, material->param_buffer_size);
		if (material_bump.buffer) {
			buffer_infos[buffer_ct] = (VkDescriptorBufferInfo){
				.buffer = material_bump.buffer->buffer,
				.offset = material_bump.offset,
				.range  = material->param_buffer_size,
			};
			writes[write_ct++] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding      = SKR_BIND_SHIFT_BUFFER + _skr_vk.bind_settings.material_slot,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo     = &buffer_infos[buffer_ct++],
			};
		}
	}

	// No system buffer or instance buffer for immediate draws
	const int32_t ignore_slots[] = {
		SKR_BIND_SHIFT_TEXTURE + _skr_vk.bind_settings.instance_slot,
		SKR_BIND_SHIFT_BUFFER  + _skr_vk.bind_settings.material_slot,
		SKR_BIND_SHIFT_BUFFER  + _skr_vk.bind_settings.system_slot };

	// Add material texture and buffer bindings
	const sksc_shader_meta_t* meta = material->key.shader->meta;

	_skr_bind_pool_lock();
	int32_t fail_idx = _skr_material_add_writes(_skr_bind_pool_get(material->bind_start), material->bind_count, ignore_slots, sizeof(ignore_slots)/sizeof(ignore_slots[0]),
		writes,       sizeof(writes      )/sizeof(writes      [0]),
		buffer_infos, sizeof(buffer_infos)/sizeof(buffer_infos[0]),
		image_infos,  sizeof(image_infos )/sizeof(image_infos [0]),
		&write_ct, &buffer_ct, &image_ct);
	_skr_bind_pool_unlock();

	if (fail_idx >= 0) {
		skr_log(skr_log_critical, "Immediate draw missing binding '%s' in shader '%s'", _skr_material_bind_name(meta, fail_idx), meta->name);
		_skr_cmd_release(cmd);
		return;
	}

	// Bind descriptors
	_skr_bind_descriptors(cmd, ctx.descriptor_pool, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                      _skr_pipeline_get_layout(material->pipeline_material_idx),
	                      _skr_pipeline_get_descriptor_layout(material->pipeline_material_idx),
	                      writes, write_ct);

	// Bind vertex buffers
	if (mesh->vertex_buffer_count > 0) {
		VkBuffer     buffers[16];
		VkDeviceSize offsets[16];
		uint32_t     bind_count = 0;

		for (uint32_t i = 0; i < mesh->vertex_buffer_count; i++) {
			if (skr_buffer_is_valid(&mesh->vertex_buffers[i])) {
				buffers[bind_count] = mesh->vertex_buffers[i].buffer;
				offsets[bind_count] = 0;
				bind_count++;
			}
		}

		if (bind_count > 0) {
			vkCmdBindVertexBuffers(cmd, 0, bind_count, buffers, offsets);
		}
	}

	// Draw
	if (skr_buffer_is_valid(&mesh->index_buffer)) {
		vkCmdBindIndexBuffer(cmd, mesh->index_buffer.buffer, 0, mesh->ind_format_vk);
		uint32_t draw_index_count = index_count > 0 ? index_count : mesh->ind_count;
		vkCmdDrawIndexed(cmd, draw_index_count, instance_count, first_index, vertex_offset, 0);
	} else {
		vkCmdDraw(cmd, mesh->vert_count, instance_count, 0, 0);
	}

	_skr_cmd_release(cmd);
}

float skr_renderer_get_gpu_time_ms() {
	// Return timing from most recently completed frame
	uint32_t read_flight = (_skr_vk.flight_idx + 1) % SKR_MAX_FRAMES_IN_FLIGHT;

	if (!_skr_vk.timestamps_valid[read_flight]) {
		return 0.0f;
	}

	uint64_t start = _skr_vk.frame_timestamps[read_flight][0];
	uint64_t end   = _skr_vk.frame_timestamps[read_flight][1];

	// Convert ticks to milliseconds: (ticks * ns_per_tick) / 1,000,000
	float time_ns = (float)(end - start) * _skr_vk.timestamp_period;
	return time_ns / 1000000.0f;
}
