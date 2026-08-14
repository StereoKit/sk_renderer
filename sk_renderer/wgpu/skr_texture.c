// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////
// Format block info — backend-independent data, mirrors the Vulkan table

void skr_tex_fmt_block_info(skr_tex_fmt_ format, uint32_t* opt_out_block_width, uint32_t* opt_out_block_height, uint32_t* opt_out_bytes_per_block) {
	uint32_t w = 1, h = 1, bytes = 4;
	switch (format) {
		case skr_tex_fmt_rgba32_srgb: case skr_tex_fmt_rgba32:
		case skr_tex_fmt_bgra32_srgb: case skr_tex_fmt_bgra32:
		case skr_tex_fmt_rg11b10uf:   case skr_tex_fmt_rgb10a2: case skr_tex_fmt_rgb9e5uf:
		case skr_tex_fmt_r32ui:       case skr_tex_fmt_r32si:   case skr_tex_fmt_r32f:
		case skr_tex_fmt_depth32:     case skr_tex_fmt_depth24s8:
			bytes = 4; break;

		case skr_tex_fmt_rgba64un: case skr_tex_fmt_rgba64sn: case skr_tex_fmt_rgba64ui:
		case skr_tex_fmt_rgba64si: case skr_tex_fmt_rgba64f:
			bytes = 8; break;

		case skr_tex_fmt_rgba128f:  bytes = 16; break;

		case skr_tex_fmt_r8: case skr_tex_fmt_r8sn: case skr_tex_fmt_r8ui:
		case skr_tex_fmt_r8si: case skr_tex_fmt_r8_srgb:
			bytes = 1; break;

		case skr_tex_fmt_r8g8: case skr_tex_fmt_r16: case skr_tex_fmt_r16sn:
		case skr_tex_fmt_r16ui: case skr_tex_fmt_r16si: case skr_tex_fmt_r16f:
		case skr_tex_fmt_depth16:
			bytes = 2; break;

		case skr_tex_fmt_depth16s8: bytes = 4; break; // stored as depth24s8 here
		case skr_tex_fmt_depth32s8: bytes = 5; break;

		case skr_tex_fmt_bc1_rgb_srgb: case skr_tex_fmt_bc1_rgb:
		case skr_tex_fmt_bc1_rgba_srgb: case skr_tex_fmt_bc1_rgba:
		case skr_tex_fmt_bc4_r: case skr_tex_fmt_bc4_rsn:
		case skr_tex_fmt_etc1_rgb: case skr_tex_fmt_etc1_rgb_srgb:
		case skr_tex_fmt_etc2_r11:
			w = 4; h = 4; bytes = 8; break;

		case skr_tex_fmt_bc2_rgba_srgb: case skr_tex_fmt_bc2_rgba:
		case skr_tex_fmt_bc3_rgba_srgb: case skr_tex_fmt_bc3_rgba:
		case skr_tex_fmt_bc5_rg: case skr_tex_fmt_bc5_rgsn:
		case skr_tex_fmt_bc6h_rgbuf: case skr_tex_fmt_bc6h_rgbf:
		case skr_tex_fmt_bc7_rgba_srgb: case skr_tex_fmt_bc7_rgba:
		case skr_tex_fmt_etc2_rgba_srgb: case skr_tex_fmt_etc2_rgba:
		case skr_tex_fmt_etc2_rg11:
		case skr_tex_fmt_astc4x4_rgba_srgb: case skr_tex_fmt_astc4x4_rgba:
			w = 4; h = 4; bytes = 16; break;

		case skr_tex_fmt_astc6x6_rgba_srgb: case skr_tex_fmt_astc6x6_rgba:
			w = 6; h = 6; bytes = 16; break;

		case skr_tex_fmt_astc8x8_rgba_hdr:
			w = 8; h = 8; bytes = 16; break;

		// YUV 4:2:0 footprint per 2x2 region, mirroring the Vulkan table; WebGPU
		// has no YUV format so creation still refuses these.
		case skr_tex_fmt_nv12:
		case skr_tex_fmt_yuv420p: w = 2; h = 2; bytes = 6;  break;
		case skr_tex_fmt_p010:    w = 2; h = 2; bytes = 12; break;

		default: break; // unsupported formats report 1x1x4; creation refuses them anyway
	}
	if (opt_out_block_width)     *opt_out_block_width     = w;
	if (opt_out_block_height)    *opt_out_block_height    = h;
	if (opt_out_bytes_per_block) *opt_out_bytes_per_block = bytes;
}

///////////////////////////////////////////////////////////////////////////////
// Samplers

static WGPUAddressMode _skr_address_mode(skr_tex_address_ address) {
	switch (address) {
		case skr_tex_address_clamp:  return WGPUAddressMode_ClampToEdge;
		case skr_tex_address_mirror: return WGPUAddressMode_MirrorRepeat;
		default:                     return WGPUAddressMode_Repeat;
	}
}

static WGPUCompareFunction _skr_compare(skr_compare_ compare) {
	switch (compare) {
		case skr_compare_less:          return WGPUCompareFunction_Less;
		case skr_compare_less_or_eq:    return WGPUCompareFunction_LessEqual;
		case skr_compare_greater:       return WGPUCompareFunction_Greater;
		case skr_compare_greater_or_eq: return WGPUCompareFunction_GreaterEqual;
		case skr_compare_equal:         return WGPUCompareFunction_Equal;
		case skr_compare_not_equal:     return WGPUCompareFunction_NotEqual;
		case skr_compare_always:        return WGPUCompareFunction_Always;
		case skr_compare_never:         return WGPUCompareFunction_Never;
		default:                        return WGPUCompareFunction_Undefined;
	}
}

WGPUSampler _skr_sampler_create(skr_tex_sampler_t settings, uint32_t mip_count) {
	// The comparison variant is built separately (see skr_tex_t); this one is
	// always a plain filtering sampler
	settings.sample_compare = skr_compare_none;
	return _skr_sampler_create_ex(settings, mip_count);
}

WGPUSampler _skr_sampler_create_ex(skr_tex_sampler_t settings, uint32_t mip_count) {
	bool point = settings.sample == skr_tex_sample_point;
	WGPUSamplerDescriptor desc = {
		.addressModeU  = _skr_address_mode(settings.address),
		.addressModeV  = _skr_address_mode(settings.address),
		.addressModeW  = _skr_address_mode(settings.address),
		.magFilter     = point ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear,
		.minFilter     = point ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear,
		.mipmapFilter  = point ? WGPUMipmapFilterMode_Nearest : WGPUMipmapFilterMode_Linear,
		.lodMinClamp   = 0.0f,
		.lodMaxClamp   = mip_count > 0 ? (float)mip_count : 32.0f,
		.compare       = _skr_compare(settings.sample_compare),
		.maxAnisotropy = (uint16_t)(settings.sample == skr_tex_sample_anisotropic
			? (settings.anisotropy > 0 ? settings.anisotropy : 4) : 1),
	};
	return wgpuDeviceCreateSampler(_skr_wgpu.device, &desc);
}

///////////////////////////////////////////////////////////////////////////////
// Texture creation

static WGPUTextureViewDimension _skr_view_dimension(const skr_tex_t* tex) {
	if (tex->flags & skr_tex_flags_3d)      return WGPUTextureViewDimension_3D;
	if (tex->flags & skr_tex_flags_cubemap) return WGPUTextureViewDimension_Cube;
	if (tex->flags & skr_tex_flags_array)   return WGPUTextureViewDimension_2DArray;
	return WGPUTextureViewDimension_2D;
}

skr_err_ skr_tex_create(skr_tex_fmt_ format, skr_tex_flags_ flags, skr_tex_sampler_t sampler, skr_vec3i_t size, int32_t multisample, int32_t mip_count, const skr_tex_data_t* opt_data, skr_tex_t* out_tex) {
	if (out_tex == NULL) return skr_err_invalid_parameter;
	memset(out_tex, 0, sizeof(*out_tex));
	if (size.x <= 0 || size.y <= 0) return skr_err_invalid_parameter;

	WGPUTextureFormat wgpu_fmt = _skr_tex_fmt_to_wgpu(format);
	if (wgpu_fmt == WGPUTextureFormat_Undefined) {
		skr_log(skr_log_warning, "skr_tex_create: format %d has no WebGPU equivalent", format);
		return skr_err_unsupported;
	}

	// WebGPU (following D3D12) requires compressed textures to be
	// block-aligned; Vulkan just clips partial edge blocks. Round the
	// physical size up so arbitrary assets still load — the padding texels
	// come from the source data's own edge blocks (compressed data always
	// stores whole blocks), but UVs map across the padded size, so a 30px
	// texture samples as 32px. Authoring block-aligned content avoids the
	// shift; data_size keeps the authored layout for uploads.
	uint32_t block_w, block_h;
	skr_tex_fmt_block_info(format, &block_w, &block_h, NULL);
	skr_vec3i_t data_size = size;
	if (block_w > 1 || block_h > 1) {
		size.x = (int32_t)(((uint32_t)size.x + block_w - 1) / block_w * block_w);
		size.y = (int32_t)(((uint32_t)size.y + block_h - 1) / block_h * block_h);
		if (size.x != data_size.x || size.y != data_size.y)
			skr_log(skr_log_info, "skr_tex_create: %dx%d compressed texture rounded up to block-aligned %dx%d — WebGPU requires it, and UVs cover the padded size", data_size.x, data_size.y, size.x, size.y);
	}

	bool is_3d       = (flags & skr_tex_flags_3d) != 0;
	bool is_depth    = format >= skr_tex_fmt_depth16 && format <= skr_tex_fmt_depth32s8;
	uint32_t layers  = 1;
	if (flags & skr_tex_flags_cubemap) layers = 6;
	else if ((flags & skr_tex_flags_array) && size.z > 0) layers = (uint32_t)size.z;

	if (multisample < 1) multisample = 1;
	if (multisample > 1 && multisample != 4) multisample = 4; // WebGPU supports 1 and 4 only

	uint32_t mips = mip_count > 0 ? (uint32_t)mip_count
	              : (flags & skr_tex_flags_gen_mips) ? skr_tex_calc_mip_count(size) : 1;
	if (multisample > 1) mips = 1;

	// Block-compressed formats can't be render attachments, so render-based
	// mip generation is off the table for them (pre-compressed mip chains
	// upload fine via skr_tex_set_data) — matching the Vulkan backend's
	// polite refusal rather than failing texture creation on usage validation
	bool can_gen_mips = block_w == 1 && block_h == 1;

	WGPUTextureUsage usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
	if (flags & skr_tex_flags_readable)  usage |= WGPUTextureUsage_CopySrc;
	if (flags & skr_tex_flags_writeable) usage |= WGPUTextureUsage_RenderAttachment;
	if (flags & skr_tex_flags_compute)   usage |= WGPUTextureUsage_StorageBinding;
	if ((flags & skr_tex_flags_gen_mips) && can_gen_mips)
		usage |= WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
	if (is_depth)                        usage |= WGPUTextureUsage_RenderAttachment;
	if (multisample > 1)                 usage |= WGPUTextureUsage_RenderAttachment;

	WGPUTextureDescriptor desc = {
		.usage         = usage,
		.dimension     = is_3d ? WGPUTextureDimension_3D : WGPUTextureDimension_2D,
		.size          = { (uint32_t)size.x, (uint32_t)size.y, is_3d ? (uint32_t)(size.z > 0 ? size.z : 1) : layers },
		.format        = wgpu_fmt,
		.mipLevelCount = mips,
		.sampleCount   = (uint32_t)multisample,
	};
	out_tex->texture = wgpuDeviceCreateTexture(_skr_wgpu.device, &desc);
	if (out_tex->texture == NULL) return skr_err_device_error;

	out_tex->size             = size;
	out_tex->data_size        = data_size;
	out_tex->format           = format;
	out_tex->flags            = flags;
	out_tex->samples          = (uint32_t)multisample;
	out_tex->mip_levels       = mips;
	out_tex->layer_count      = is_3d ? 1 : layers;
	out_tex->sampler_settings = sampler;

	WGPUTextureViewDescriptor view_desc = {
		.format          = wgpu_fmt,
		.dimension       = _skr_view_dimension(out_tex),
		.mipLevelCount   = mips,
		.arrayLayerCount = is_3d ? 1 : layers,
		.aspect          = WGPUTextureAspect_All,
	};
	out_tex->view    = wgpuTextureCreateView(out_tex->texture, &view_desc);
	out_tex->sampler = _skr_sampler_create(sampler, mips);
	if (sampler.sample_compare != skr_compare_none)
		out_tex->sampler_compare = _skr_sampler_create_ex(sampler, mips);

	if (opt_data != NULL) {
		skr_err_ err = skr_tex_set_data(out_tex, opt_data);
		if (err != skr_err_success) { skr_tex_destroy(out_tex); return err; }
	}
	return skr_err_success;
}


///////////////////////////////////////////
// External textures
//
// A compositor-owned WGPUTexture wrapped as an skr_tex_t. WebXR hands out a
// fresh texture every frame, so the update path has to be allocation-free in
// steady state - it releases the old views and lets them rebuild lazily rather
// than rebuilding eagerly.
///////////////////////////////////////////

// Drops the cached views without touching the texture or sampler. Shared by
// create (nothing to drop) and update (everything view-shaped is stale).
static void _skr_tex_release_views(skr_tex_t* ref_tex) {
	if (ref_tex->layer_views) {
		for (uint32_t i = 0; i < ref_tex->layer_count; i++) {
			if (ref_tex->layer_views[i]) {
				wgpuTextureViewRelease(ref_tex->layer_views[i]);
				ref_tex->layer_views[i] = NULL;
			}
		}
	}
	if (ref_tex->view) { wgpuTextureViewRelease(ref_tex->view); ref_tex->view = NULL; }
}

skr_err_ skr_tex_create_external_wgpu(skr_tex_external_wgpu_info_t info, skr_tex_t* out_tex) {
	if (out_tex == NULL || info.texture == NULL) return skr_err_invalid_parameter;

	memset(out_tex, 0, sizeof(*out_tex));
	uint32_t layers = info.size.z > 0 ? (uint32_t)info.size.z : 1;

	out_tex->texture          = (WGPUTexture)info.texture;
	out_tex->size             = info.size;
	out_tex->data_size        = info.size;
	out_tex->format           = info.format;
	// The array flag drives _skr_view_dimension. Without it a multi-layer
	// texture gets a 2D view with arrayLayerCount > 1, which WebGPU rejects:
	// "The dimension (TextureViewDimension::e2D) of the texture view is not
	// compatible with the layer count (2)". Stereo XR swapchain images are
	// always arrays, so this is the common case rather than an edge one.
	out_tex->flags            = skr_tex_flags_writeable | (layers > 1 ? skr_tex_flags_array : 0);
	out_tex->samples          = info.multisample > 0 ? (uint32_t)info.multisample : 1;
	out_tex->mip_levels       = 1;
	out_tex->layer_count      = layers;
	out_tex->sampler_settings = info.sampler;
	out_tex->is_external      = !info.owns_texture;

	out_tex->view = wgpuTextureCreateView(out_tex->texture, &(WGPUTextureViewDescriptor){
		.format          = _skr_tex_fmt_to_wgpu(info.format),
		.dimension       = _skr_view_dimension(out_tex),
		.mipLevelCount   = 1,
		.arrayLayerCount = layers,
		.aspect          = WGPUTextureAspect_All,
	});
	if (out_tex->view == NULL) return skr_err_device_error;

	out_tex->sampler = _skr_sampler_create(info.sampler, 1);
	if (info.sampler.sample_compare != skr_compare_none)
		out_tex->sampler_compare = _skr_sampler_create_ex(info.sampler, 1);
	return skr_err_success;
}

skr_err_ skr_tex_update_external_wgpu(skr_tex_t* ref_tex, void* texture) {
	if (ref_tex == NULL || texture == NULL) return skr_err_invalid_parameter;
	if (ref_tex->texture == (WGPUTexture)texture) return skr_err_success;   // nothing changed

	_skr_tex_release_views(ref_tex);
	if (ref_tex->texture && !ref_tex->is_external) wgpuTextureRelease(ref_tex->texture);
	ref_tex->texture = (WGPUTexture)texture;

	// The whole-resource view is needed every frame; layer views are rebuilt on
	// demand by whoever renders to them, so the layer_views array is kept
	// allocated and merely emptied.
	ref_tex->view = wgpuTextureCreateView(ref_tex->texture, &(WGPUTextureViewDescriptor){
		.format          = _skr_tex_fmt_to_wgpu(ref_tex->format),
		.dimension       = _skr_view_dimension(ref_tex),
		.mipLevelCount   = 1,
		.arrayLayerCount = ref_tex->layer_count,
		.aspect          = WGPUTextureAspect_All,
	});
	return ref_tex->view != NULL ? skr_err_success : skr_err_device_error;
}

skr_err_ skr_tex_create_copy(const skr_tex_t* src, skr_tex_fmt_ format, skr_tex_flags_ flags, int32_t multisample, skr_tex_t* out_tex) {
	if (src == NULL || out_tex == NULL || src->texture == NULL) return skr_err_invalid_parameter;

	// Match the Vulkan backend's conventions: fmt_none / multisample 0 = same
	// as the source
	skr_tex_fmt_ dst_format  = format      == skr_tex_fmt_none ? src->format           : format;
	int32_t      dst_samples = multisample == 0                ? (int32_t)src->samples : multisample;
	bool         is_resolve  = src->samples > 1 && dst_samples == 1;

	// A resolve target needs RenderAttachment usage
	skr_tex_flags_ dst_flags = is_resolve ? (flags | skr_tex_flags_writeable) : flags;
	skr_err_ err = skr_tex_create(dst_format, dst_flags, src->sampler_settings, src->size, dst_samples, (int32_t)src->mip_levels, NULL, out_tex);
	if (err != skr_err_success) return err;

	if (is_resolve) {
		if (dst_format != src->format)
			return skr_err_success; // cross-format resolve needs a blit; caller renders into it

		// WebGPU has no vkCmdResolveImage; an empty load/store render pass
		// with a resolveTarget does the job. MSAA sources only have mip 0.
		WGPUCommandEncoder encoder = _skr_cmd_get();
		for (uint32_t layer = 0; layer < src->layer_count; layer++) {
			WGPUTextureViewDescriptor view_desc = {
				.dimension       = WGPUTextureViewDimension_2D,
				.baseMipLevel    = 0, .mipLevelCount   = 1,
				.baseArrayLayer  = layer, .arrayLayerCount = 1,
				.aspect          = WGPUTextureAspect_All,
			};
			WGPUTextureView src_view = wgpuTextureCreateView(src->texture,     &view_desc);
			WGPUTextureView dst_view = wgpuTextureCreateView(out_tex->texture, &view_desc);
			WGPURenderPassColorAttachment color_attach = {
				.view          = src_view,
				.resolveTarget = dst_view,
				.depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED,
				.loadOp        = WGPULoadOp_Load,
				.storeOp       = WGPUStoreOp_Store,
			};
			WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &(WGPURenderPassDescriptor){
				.colorAttachmentCount = 1, .colorAttachments = &color_attach });
			wgpuRenderPassEncoderEnd(pass);
			wgpuRenderPassEncoderRelease(pass);
			wgpuTextureViewRelease(src_view);
			wgpuTextureViewRelease(dst_view);
		}
		return skr_err_success;
	}

	if (dst_format == src->format && dst_samples == (int32_t)src->samples) {
		for (uint32_t mip = 0; mip < src->mip_levels; mip++) {
			err = skr_tex_copy(src, out_tex, mip, 0, mip, 0, src->layer_count);
			if (err != skr_err_success) return err;
		}
		return skr_err_success;
	}
	return skr_err_success; // format conversion copies need a blit; caller renders into it
}

///////////////////////////////////////////////////////////////////////////////

bool skr_tex_is_valid(const skr_tex_t* tex) {
	return tex != NULL && tex->texture != NULL;
}

void skr_tex_destroy(skr_tex_t* ref_tex) {
	if (ref_tex == NULL) return;
	if (ref_tex->layer_views) {
		for (uint32_t i = 0; i < ref_tex->layer_count; i++)
			if (ref_tex->layer_views[i]) wgpuTextureViewRelease(ref_tex->layer_views[i]);
		_skr_free(ref_tex->layer_views);
	}
	if (ref_tex->sampler)         wgpuSamplerRelease(ref_tex->sampler);
	if (ref_tex->sampler_compare) wgpuSamplerRelease(ref_tex->sampler_compare);
	if (ref_tex->view)    wgpuTextureViewRelease(ref_tex->view);
	if (ref_tex->texture && !ref_tex->is_external) wgpuTextureRelease(ref_tex->texture);
	memset(ref_tex, 0, sizeof(*ref_tex));
}

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_tex_copy(const skr_tex_t* src, skr_tex_t* dst, uint32_t src_mip, uint32_t src_layer, uint32_t dst_mip, uint32_t dst_layer, uint32_t layer_count) {
	if (src == NULL || dst == NULL || src->texture == NULL || dst->texture == NULL) return skr_err_invalid_parameter;
	if (layer_count == 0) layer_count = 1;

	skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(src->size, src_mip);
	WGPUTexelCopyTextureInfo from = { .texture = src->texture, .mipLevel = src_mip, .origin = { 0, 0, src_layer } };
	WGPUTexelCopyTextureInfo to   = { .texture = dst->texture, .mipLevel = dst_mip, .origin = { 0, 0, dst_layer } };
	WGPUExtent3D extent = { (uint32_t)mip_size.x, (uint32_t)mip_size.y, layer_count };
	wgpuCommandEncoderCopyTextureToTexture(_skr_cmd_get(), &from, &to, &extent);
	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////
// Upload. Data layout is mip-major (all layers per mip), matching KTX2.

skr_err_ skr_tex_set_data(skr_tex_t* ref_tex, const skr_tex_data_t* data) {
	if (ref_tex == NULL || ref_tex->texture == NULL || data == NULL || data->data == NULL) return skr_err_invalid_parameter;

	uint32_t block_w, block_h, block_bytes;
	skr_tex_fmt_block_info(ref_tex->format, &block_w, &block_h, &block_bytes);

	bool     is_3d       = (ref_tex->flags & skr_tex_flags_3d) != 0;
	uint32_t layer_count = data->layer_count > 0 ? data->layer_count : 1;
	uint32_t mip_count   = data->mip_count   > 0 ? data->mip_count   : 1;
	const uint8_t* src   = (const uint8_t*)data->data;

	// Layout comes from the authored dimensions (data_size): on padded
	// compressed textures the physical mip chain can be a block wider than
	// the data's rows partway down (e.g. 25px -> 28px base: mip 1 is 12px of
	// data in a 14px mip), so pitches must follow the data, not the texture.
	// For everything else data_size == size and this is a no-op distinction.
	skr_vec3i_t base = ref_tex->data_size.x > 0 ? ref_tex->data_size : ref_tex->size;
	for (uint32_t m = 0; m < mip_count; m++) {
		uint32_t    mip       = data->base_mip + m;
		skr_vec3i_t mip_size  = skr_tex_calc_mip_dimensions(base, mip);
		uint32_t    blocks_x  = ((uint32_t)mip_size.x + block_w - 1) / block_w;
		uint32_t    blocks_y  = ((uint32_t)mip_size.y + block_h - 1) / block_h;
		uint32_t    row_pitch  = (mip_count == 1 && data->row_pitch > 0) ? (uint32_t)data->row_pitch : blocks_x * block_bytes;
		uint32_t    depth      = is_3d ? (uint32_t)mip_size.z : layer_count;
		uint64_t    slice_size = (uint64_t)row_pitch * blocks_y;

		// Copy extents for compressed formats must cover whole blocks —
		// including a mip's trailing partial block, which is written at its
		// block-rounded size, never its logical size. The data's whole-block
		// coverage is exactly that (and never exceeds the mip's rounded
		// extent, since data dims <= texture dims). Uncompressed formats have
		// 1x1 blocks, so this stays the plain mip size for them.
		uint32_t extent_w = blocks_x * block_w;
		uint32_t extent_h = blocks_y * block_h;

		WGPUTexelCopyTextureInfo dest = {
			.texture  = ref_tex->texture,
			.mipLevel = mip,
			.origin   = { 0, 0, is_3d ? 0 : data->base_layer },
		};
		WGPUTexelCopyBufferLayout layout = {
			.offset       = 0,
			.bytesPerRow  = row_pitch,
			.rowsPerImage = blocks_y,
		};
		WGPUExtent3D extent = { extent_w, extent_h, depth };
		wgpuQueueWriteTexture(_skr_wgpu.queue, &dest, src, (size_t)(slice_size * depth), &layout, &extent);
		src += slice_size * depth;
	}
	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////
// GPU-side upload of a tightly packed mip chain, same contract as the Vulkan
// backend: whole blocks, mip-major, layers consecutive within each mip.
// Copies record on the shared encoder, after any compute that filled the
// buffer.
//
// Buffer-to-texture copies need a 256-multiple bytesPerRow, which the packed
// tail of a mip chain rarely is. Single-block-row copies are exempt from the
// rule, so unaligned mips copy one block row at a time. Only mips narrower
// than 256/block_bytes blocks take that path, and those have few rows.

skr_err_ skr_tex_set_buffer(skr_tex_t* ref_tex, const skr_buffer_t* buffer, uint32_t base_mip, uint32_t mip_count) {
	if (ref_tex == NULL || ref_tex->texture == NULL || buffer == NULL || buffer->buffer == NULL) return skr_err_invalid_parameter;
	if (mip_count == 0) return skr_err_invalid_parameter;
	if (base_mip + mip_count > ref_tex->mip_levels) {
		skr_log(skr_log_warning, "skr_tex_set_buffer: mip range [%u, %u) exceeds texture mip count %u",
			base_mip, base_mip + mip_count, ref_tex->mip_levels);
		return skr_err_invalid_parameter;
	}

	uint32_t block_w, block_h, block_bytes;
	skr_tex_fmt_block_info(ref_tex->format, &block_w, &block_h, &block_bytes);

	// Like skr_tex_set_data, layout follows the authored dimensions
	// (data_size, not the block-rounded physical size); extents cover whole blocks
	skr_vec3i_t        base    = ref_tex->data_size.x > 0 ? ref_tex->data_size : ref_tex->size;
	bool               is_3d   = (ref_tex->flags & skr_tex_flags_3d) != 0;
	WGPUCommandEncoder encoder = _skr_cmd_get();
	uint64_t           offset  = 0;
	for (uint32_t m = 0; m < mip_count; m++) {
		uint32_t    mip       = base_mip + m;
		skr_vec3i_t mip_size  = skr_tex_calc_mip_dimensions(base, mip);
		uint32_t    blocks_x  = ((uint32_t)mip_size.x + block_w - 1) / block_w;
		uint32_t    blocks_y  = ((uint32_t)mip_size.y + block_h - 1) / block_h;
		uint32_t    row_bytes = blocks_x * block_bytes;
		uint32_t    depth     = is_3d ? (uint32_t)mip_size.z : ref_tex->layer_count;
		uint64_t    slice     = (uint64_t)row_bytes * blocks_y;

		if (row_bytes % 256 == 0) {
			WGPUTexelCopyBufferInfo from = {
				.layout = { .offset = offset, .bytesPerRow = row_bytes, .rowsPerImage = blocks_y },
				.buffer = buffer->buffer };
			WGPUTexelCopyTextureInfo to = { .texture = ref_tex->texture, .mipLevel = mip, .origin = { 0, 0, 0 } };
			WGPUExtent3D extent = { blocks_x * block_w, blocks_y * block_h, depth };
			wgpuCommandEncoderCopyBufferToTexture(encoder, &from, &to, &extent);
		} else {
			for (uint32_t d = 0; d < depth; d++)
			for (uint32_t row = 0; row < blocks_y; row++) {
				WGPUTexelCopyBufferInfo from = {
					.layout = { .offset       = offset + (uint64_t)d * slice + (uint64_t)row * row_bytes,
					            .bytesPerRow  = WGPU_COPY_STRIDE_UNDEFINED,
					            .rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED },
					.buffer = buffer->buffer };
				WGPUTexelCopyTextureInfo to = { .texture = ref_tex->texture, .mipLevel = mip, .origin = { 0, row * block_h, d } };
				WGPUExtent3D extent = { blocks_x * block_w, block_h, 1 };
				wgpuCommandEncoderCopyBufferToTexture(encoder, &from, &to, &extent);
			}
		}
		offset += slice * depth;
	}
	return skr_err_success;
}

///////////////////////////////////////////////////////////////////////////////
// Readback: copy into a mappable staging buffer, then a pollable map. The
// caller's data pointer is heap memory filled by the map callback, so it's
// stable from creation and valid once the future completes.

typedef struct _skr_readback_ctx_t {
	WGPUBuffer staging;
	void*      dest;
	uint32_t   dest_size;
	uint32_t   row_bytes;     // tight row size
	uint32_t   padded_row;    // 256-aligned staging row pitch
	uint32_t   rows;
	// Completion marker for the wrapped future — on the web the map future
	// can't be observed via WaitAny (see _SKR_CB_MODE_ASYNC), so the callback
	// marks the slot directly
	_skr_cmd_slot_t* done_slot;
	uint64_t         done_generation;
} _skr_readback_ctx_t;

static void _skr_on_readback_map(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
	(void)message; (void)userdata2;
	_skr_readback_ctx_t* ctx = (_skr_readback_ctx_t*)userdata1;
	if (status == WGPUMapAsyncStatus_Success) {
		const uint8_t* mapped = (const uint8_t*)wgpuBufferGetConstMappedRange(ctx->staging, 0, (size_t)ctx->padded_row * ctx->rows);
		if (mapped) {
			uint8_t* dst = (uint8_t*)ctx->dest;
			for (uint32_t r = 0; r < ctx->rows; r++)
				memcpy(dst + (size_t)r * ctx->row_bytes, mapped + (size_t)r * ctx->padded_row, ctx->row_bytes);
		}
		wgpuBufferUnmap(ctx->staging);
	} else {
		skr_log(skr_log_warning, "Texture readback map failed (%d)", (int)status);
	}
	wgpuBufferRelease(ctx->staging);
	ctx->staging = NULL;
	if (ctx->done_slot && ctx->done_slot->generation == ctx->done_generation)
		ctx->done_slot->completed = true;
}

skr_err_ skr_tex_readback(const skr_tex_t* tex, uint32_t mip_level, uint32_t array_layer, skr_tex_readback_t* out_readback) {
	if (tex == NULL || tex->texture == NULL || out_readback == NULL) return skr_err_invalid_parameter;
	memset(out_readback, 0, sizeof(*out_readback));

	uint32_t block_w, block_h, block_bytes;
	skr_tex_fmt_block_info(tex->format, &block_w, &block_h, &block_bytes);
	skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(tex->size, mip_level);
	uint32_t blocks_x  = ((uint32_t)mip_size.x + block_w - 1) / block_w;
	uint32_t blocks_y  = ((uint32_t)mip_size.y + block_h - 1) / block_h;
	uint32_t row_bytes = blocks_x * block_bytes;
	uint32_t padded    = (row_bytes + 255) & ~255u; // buffer copies need 256-aligned rows

	_skr_readback_ctx_t* ctx = (_skr_readback_ctx_t*)_skr_calloc(1, sizeof(_skr_readback_ctx_t));
	ctx->row_bytes  = row_bytes;
	ctx->padded_row = padded;
	ctx->rows       = blocks_y;
	ctx->dest_size  = row_bytes * blocks_y;
	ctx->dest       = _skr_malloc(ctx->dest_size);

	WGPUBufferDescriptor staging_desc = {
		.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
		.size  = (uint64_t)padded * blocks_y,
	};
	ctx->staging = wgpuDeviceCreateBuffer(_skr_wgpu.device, &staging_desc);
	if (ctx->staging == NULL) { _skr_free(ctx->dest); _skr_free(ctx); return skr_err_device_error; }

	WGPUTexelCopyTextureInfo from = { .texture = tex->texture, .mipLevel = mip_level, .origin = { 0, 0, array_layer } };
	WGPUTexelCopyBufferInfo  to   = { .layout = { .offset = 0, .bytesPerRow = padded, .rowsPerImage = blocks_y }, .buffer = ctx->staging };
	WGPUExtent3D extent = { (uint32_t)mip_size.x, (uint32_t)mip_size.y, 1 };
	wgpuCommandEncoderCopyTextureToBuffer(_skr_cmd_get(), &from, &to, &extent);

	// Submit, then chain the map; its future is what the caller polls
	_skr_cmd_submit();
	WGPUFuture map_future = wgpuBufferMapAsync(ctx->staging, WGPUMapMode_Read, 0, (size_t)padded * blocks_y, (WGPUBufferMapCallbackInfo){
		.mode      = _SKR_CB_MODE_ASYNC,
		.callback  = _skr_on_readback_map,
		.userdata1 = ctx });

	out_readback->future    = _skr_future_from_wgpu(map_future);
	ctx->done_slot          = (_skr_cmd_slot_t*)out_readback->future.slot;
	ctx->done_generation    = out_readback->future.generation;
	out_readback->data      = ctx->dest;
	out_readback->size      = ctx->dest_size;
	out_readback->_internal = ctx;
	return skr_err_success;
}

void skr_tex_readback_destroy(skr_tex_readback_t* ref_readback) {
	if (ref_readback == NULL) return;
	_skr_readback_ctx_t* ctx = (_skr_readback_ctx_t*)ref_readback->_internal;
	if (ctx) {
		if (ctx->staging != NULL) // map still pending; let it finish (native)
			skr_future_wait(&ref_readback->future);
		_skr_free(ctx->dest);
		_skr_free(ctx);
	}
	memset(ref_readback, 0, sizeof(*ref_readback));
}

///////////////////////////////////////////////////////////////////////////////

skr_vec3i_t skr_tex_get_size(const skr_tex_t* tex) {
	skr_vec3i_t zero = {0};
	return tex ? tex->size : zero;
}

uint32_t       skr_tex_get_array_count(const skr_tex_t* tex) { return tex ? tex->layer_count : 0; }
skr_tex_fmt_   skr_tex_get_format     (const skr_tex_t* tex) { return tex ? tex->format : skr_tex_fmt_none; }
skr_tex_flags_ skr_tex_get_flags      (const skr_tex_t* tex) { return tex ? tex->flags : skr_tex_flags_none; }
int32_t        skr_tex_get_multisample(const skr_tex_t* tex) { return tex ? (int32_t)tex->samples : 1; }

void skr_tex_set_sampler(skr_tex_t* ref_tex, skr_tex_sampler_t sampler) {
	if (ref_tex == NULL) return;
	ref_tex->sampler_settings = sampler;
	if (ref_tex->sampler)         wgpuSamplerRelease(ref_tex->sampler);
	if (ref_tex->sampler_compare) wgpuSamplerRelease(ref_tex->sampler_compare);
	ref_tex->sampler         = _skr_sampler_create(sampler, ref_tex->mip_levels);
	ref_tex->sampler_compare = sampler.sample_compare != skr_compare_none
		? _skr_sampler_create_ex(sampler, ref_tex->mip_levels) : NULL;
	// Cached bind groups hold the old sampler; retire them
	_skr_bind_epoch_bump();
}

skr_tex_sampler_t skr_tex_get_sampler(const skr_tex_t* tex) {
	skr_tex_sampler_t zero = {0};
	return tex ? tex->sampler_settings : zero;
}

///////////////////////////////////////////////////////////////////////////////
// Render-based mipgen using the builtin box-filter shaders: each mip renders
// from a single-mip view of the previous level (disjoint subresources, so
// sampling and rendering the same texture in one pass is legal). Cubemaps use
// sk_view_index to pick the face direction.

#include "skr_pipeline.h"
#include "skr_mipgen_2d.hlsl.h"
#include "skr_mipgen_cube.hlsl.h"

typedef struct _skr_mipgen_t {
	skr_shader_t   shader;
	skr_material_t material;
	bool           tried;
} _skr_mipgen_t;

static _skr_mipgen_t _mipgen_2d;
static _skr_mipgen_t _mipgen_cube;

// One shared linear-clamp sampler for every mipgen path (builtin or custom
// filter shader) — independent of the texture's own sampler settings
static WGPUSampler _mipgen_sampler;

static WGPUSampler _skr_mipgen_sampler(void) {
	if (_mipgen_sampler == NULL) {
		skr_tex_sampler_t settings = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };
		_mipgen_sampler = _skr_sampler_create(settings, 0);
	}
	return _mipgen_sampler;
}

static skr_material_t* _skr_mipgen_material(_skr_mipgen_t* gen, const unsigned char* sks, size_t sks_size) {
	if (!gen->tried) {
		gen->tried = true;
		if (skr_shader_create(sks, (uint32_t)sks_size, &gen->shader) == skr_err_success) {
			skr_material_create((skr_material_info_t){
				.shader     = &gen->shader,
				.cull       = skr_cull_none,
				.depth_test = skr_compare_always,
				.write_mask = skr_write_rgba,
			}, &gen->material);
		} else {
			skr_log(skr_log_warning, "Builtin mipgen shader failed to load; mip generation disabled");
		}
	}
	return skr_material_is_valid(&gen->material) ? &gen->material : NULL;
}

void skr_tex_generate_mips(skr_tex_t* ref_tex, const skr_shader_t* opt_filter_shader) {
	if (ref_tex == NULL || ref_tex->texture == NULL || ref_tex->mip_levels <= 1) return;

	uint32_t block_w, block_h;
	skr_tex_fmt_block_info(ref_tex->format, &block_w, &block_h, NULL);
	if (block_w > 1 || block_h > 1) {
		skr_log(skr_log_warning, "Texture format doesn't support render-based mipgen (block-compressed); upload pre-compressed mips via skr_tex_set_data instead");
		return;
	}

	// A caller-supplied filter shader replaces the builtin material — it's a
	// render (fullscreen) shader following the same convention (src_size/
	// dst_size/src_mip_level/mip_max params, texture + sampler), exactly like
	// the Vulkan backend's custom render path.
	bool           is_cube  = (ref_tex->flags & skr_tex_flags_cubemap) != 0;
	skr_material_t custom   = {0};
	skr_material_t* material = NULL;
	if (opt_filter_shader != NULL && skr_shader_is_valid(opt_filter_shader) && opt_filter_shader->pixel_stage.shader != NULL) {
		if (skr_material_create((skr_material_info_t){
				.shader     = (skr_shader_t*)opt_filter_shader,
				.cull       = skr_cull_none,
				.depth_test = skr_compare_always,
				.write_mask = skr_write_rgba,
			}, &custom) == skr_err_success)
			material = &custom;
	}
	if (material == NULL) {
		if (opt_filter_shader != NULL)
			skr_log(skr_log_warning, "Custom mipgen shader isn't usable here; using the builtin box filter");
		material = is_cube
			? _skr_mipgen_material(&_mipgen_cube, sks_skr_mipgen_cube_hlsl, sizeof(sks_skr_mipgen_cube_hlsl))
			: _skr_mipgen_material(&_mipgen_2d,   sks_skr_mipgen_2d_hlsl,   sizeof(sks_skr_mipgen_2d_hlsl));
	}
	if (material == NULL) return;

	const sksc_shader_meta_t* meta = &material->key.shader->meta;
	if (meta->global_buffer_id < 0 || meta->resource_count < 1 || meta->sampler_count < 1) {
		if (material == &custom) skr_material_destroy(&custom);
		return;
	}
	uint32_t param_slot   = meta->buffers[meta->global_buffer_id].bind.slot;
	uint32_t tex_slot     = meta->resources[0].bind.slot;
	uint32_t sampler_slot = meta->samplers[0].slot;
	uint32_t param_size   = meta->buffers[meta->global_buffer_id].size;

	// Prior uploads (the base mip's data) must land before these passes read it
	_skr_cmd_submit();

	WGPUTextureFormat   fmt    = _skr_tex_fmt_to_wgpu(ref_tex->format);
	WGPUBindGroupLayout layout = _skr_pipeline_get_bind_layout(material->pipeline_material_idx);
	WGPUCommandEncoder  encoder = wgpuDeviceCreateCommandEncoder(_skr_wgpu.device, NULL);

	for (uint32_t mip = 1; mip < ref_tex->mip_levels; mip++) {
		skr_vec3i_t src_size = skr_tex_calc_mip_dimensions(ref_tex->size, mip - 1);
		skr_vec3i_t dst_size = skr_tex_calc_mip_dimensions(ref_tex->size, mip);

		uint32_t src_wh[2] = { (uint32_t)src_size.x, (uint32_t)src_size.y };
		uint32_t dst_wh[2] = { (uint32_t)dst_size.x, (uint32_t)dst_size.y };
		uint32_t src_mip   = mip - 1;
		uint32_t mip_max   = ref_tex->mip_levels - 1;
		skr_material_set_param(material, "src_size",      sksc_shader_var_uint, 2, src_wh);
		skr_material_set_param(material, "dst_size",      sksc_shader_var_uint, 2, dst_wh);
		skr_material_set_param(material, "src_mip_level", sksc_shader_var_uint, 1, &src_mip);
		skr_material_set_param(material, "mip_max",       sksc_shader_var_uint, 1, &mip_max);

		uint64_t   param_offset = 0;
		WGPUBuffer param_buffer = _skr_bump_uniform_write(material->param_buffer, param_size, &param_offset);

		// Source view restricted to the previous mip, keeping it disjoint
		// from the render attachment
		WGPUTextureViewDescriptor src_desc = {
			.format          = fmt,
			.dimension       = is_cube ? WGPUTextureViewDimension_Cube : WGPUTextureViewDimension_2D,
			.baseMipLevel    = mip - 1,
			.mipLevelCount   = 1,
			.baseArrayLayer  = 0,
			.arrayLayerCount = is_cube ? 6u : 1u,
			.aspect          = WGPUTextureAspect_All,
		};
		WGPUTextureView src_view = wgpuTextureCreateView(ref_tex->texture, &src_desc);

		// The material slot is dynamic-offset in the layout (see
		// _skr_bind_layout_create), so the entry binds base 0 and the param
		// position rides in at SetBindGroup time
		WGPUBindGroupEntry entries[3] = {
			{ .binding = param_slot,   .buffer      = param_buffer, .offset = 0, .size = param_size },
			{ .binding = tex_slot,     .textureView = src_view },
			{ .binding = sampler_slot, .sampler     = _skr_mipgen_sampler() },
		};
		WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(_skr_wgpu.device, &(WGPUBindGroupDescriptor){
			.layout = layout, .entryCount = 3, .entries = entries });
		_skr_draw_buffers_t mip_db = { .material_offset = param_offset, .material_size = param_size };
		uint32_t dyn_offsets[3];
		uint32_t dyn_count = _skr_dynamic_offsets(meta, (uint8_t)(skr_stage_vertex | skr_stage_pixel), &mip_db, dyn_offsets);

		for (uint32_t layer = 0; layer < ref_tex->layer_count; layer++) {
			skr_pipeline_pass_key_t pass_key = {
				.color_format = fmt,
				.depth_format = WGPUTextureFormat_Undefined,
				.sample_count = 1,
				.view_index   = layer,
			};
			WGPURenderPipeline pipeline = _skr_pipeline_get(material->pipeline_material_idx, &pass_key, -1);
			if (pipeline == NULL) break;

			WGPUTextureViewDescriptor dst_desc = {
				.format          = fmt,
				.dimension       = WGPUTextureViewDimension_2D,
				.baseMipLevel    = mip,
				.mipLevelCount   = 1,
				.baseArrayLayer  = layer,
				.arrayLayerCount = 1,
				.aspect          = WGPUTextureAspect_All,
			};
			WGPUTextureView dst_view = wgpuTextureCreateView(ref_tex->texture, &dst_desc);
			_skr_fullscreen_pass(encoder, dst_view, WGPULoadOp_Clear, NULL, pipeline, bind_group, dyn_offsets, dyn_count);
			wgpuTextureViewRelease(dst_view);
		}
		wgpuBindGroupRelease(bind_group);
		wgpuTextureViewRelease(src_view);
	}

	WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
	wgpuCommandEncoderRelease(encoder);
	wgpuQueueSubmit(_skr_wgpu.queue, 1, &cmd);
	wgpuCommandBufferRelease(cmd);

	// A caller-shader material is transient; its registry slot and bind slice
	// recycle at the next frame boundary as usual
	if (material == &custom) skr_material_destroy(&custom);
}

// Single-layer render view, lazily created and cached per layer
WGPUTextureView _skr_tex_layer_view(skr_tex_t* tex, uint32_t layer) {
	if (tex == NULL || tex->texture == NULL || layer >= tex->layer_count) return NULL;
	if (tex->layer_count == 1 && !(tex->flags & (skr_tex_flags_array | skr_tex_flags_cubemap)))
		return tex->view;

	if (tex->layer_views == NULL)
		tex->layer_views = (WGPUTextureView*)_skr_calloc(tex->layer_count, sizeof(WGPUTextureView));
	if (tex->layer_views[layer] == NULL) {
		WGPUTextureViewDescriptor desc = {
			.format          = _skr_tex_fmt_to_wgpu(tex->format),
			.dimension       = WGPUTextureViewDimension_2D,
			.baseMipLevel    = 0,
			.mipLevelCount   = 1,
			.baseArrayLayer  = layer,
			.arrayLayerCount = 1,
			.aspect          = WGPUTextureAspect_All,
		};
		tex->layer_views[layer] = wgpuTextureCreateView(tex->texture, &desc);
	}
	return tex->layer_views[layer];
}

void skr_tex_set_name(skr_tex_t* ref_tex, const char* name) {
	if (ref_tex == NULL || ref_tex->texture == NULL || name == NULL) return;
	wgpuTextureSetLabel(ref_tex->texture, (WGPUStringView){ name, strlen(name) });
}

///////////////////////////////////////////////////////////////////////////////

bool skr_tex_fmt_is_supported(skr_tex_fmt_ format, skr_tex_flags_ flags, int32_t multisample) {
	WGPUTextureFormat wgpu_fmt = _skr_tex_fmt_to_wgpu(format);
	if (wgpu_fmt == WGPUTextureFormat_Undefined) return false;

	bool is_bc   = format >= skr_tex_fmt_bc1_rgb_srgb    && format <= skr_tex_fmt_bc7_rgba;
	bool is_etc  = format >= skr_tex_fmt_etc1_rgb        && format <= skr_tex_fmt_etc2_rg11;
	bool is_astc = format >= skr_tex_fmt_astc4x4_rgba_srgb && format <= skr_tex_fmt_astc8x8_rgba_hdr;
	if (is_bc   && !_skr_wgpu.feat_bc)   return false;
	if (is_etc  && !_skr_wgpu.feat_etc2) return false;
	if (is_astc && !_skr_wgpu.feat_astc) return false;
	if ((is_bc || is_etc || is_astc) && (flags & (skr_tex_flags_writeable | skr_tex_flags_compute))) return false;

	if (multisample > 1 && multisample != 4) return false;
	if (flags & skr_tex_flags_compute) {
		// storage-capable formats only; conservative core set
		switch (wgpu_fmt) {
			case WGPUTextureFormat_RGBA8Unorm: case WGPUTextureFormat_RGBA8Snorm:
			case WGPUTextureFormat_RGBA8Uint:  case WGPUTextureFormat_RGBA8Sint:
			case WGPUTextureFormat_RGBA16Uint: case WGPUTextureFormat_RGBA16Sint: case WGPUTextureFormat_RGBA16Float:
			case WGPUTextureFormat_R32Uint:    case WGPUTextureFormat_R32Sint:    case WGPUTextureFormat_R32Float:
			case WGPUTextureFormat_RG32Uint:   case WGPUTextureFormat_RG32Sint:   case WGPUTextureFormat_RG32Float:
			case WGPUTextureFormat_RGBA32Uint: case WGPUTextureFormat_RGBA32Sint: case WGPUTextureFormat_RGBA32Float:
				break;
			default: return false;
		}
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////
// Mip math — backend-independent, mirrors the Vulkan implementations

uint32_t skr_tex_calc_mip_count(skr_vec3i_t size) {
	int32_t max_dim = size.x > size.y ? size.x : size.y;
	if (size.z > max_dim) max_dim = size.z;
	if (max_dim < 1) return 0;

	uint32_t count = 1;
	while (max_dim > 1) {
		max_dim >>= 1;
		count++;
	}
	return count;
}

skr_vec3i_t skr_tex_calc_mip_dimensions(skr_vec3i_t base_size, uint32_t mip_level) {
	skr_vec3i_t result = {
		.x = base_size.x >> mip_level,
		.y = base_size.y >> mip_level,
		.z = base_size.z >> mip_level,
	};
	if (result.x < 1) result.x = 1;
	if (result.y < 1) result.y = 1;
	if (result.z < 1) result.z = 1;
	return result;
}

uint64_t skr_tex_calc_mip_size(skr_tex_fmt_ format, skr_vec3i_t base_size, uint32_t mip_level) {
	skr_vec3i_t mip_size = skr_tex_calc_mip_dimensions(base_size, mip_level);

	uint32_t block_w, block_h, block_bytes;
	skr_tex_fmt_block_info(format, &block_w, &block_h, &block_bytes);
	if (block_w == 0 || block_h == 0) return 0;

	uint64_t blocks_x = (mip_size.x + block_w - 1) / block_w;
	uint64_t blocks_y = (mip_size.y + block_h - 1) / block_h;

	return blocks_x * blocks_y * mip_size.z * block_bytes;
}

///////////////////////////////////////////////////////////////////////////////

// Releases the lazy mipgen materials so skr_shutdown -> skr_init cycles
// rebuild them against the new device
void _skr_texture_sys_shutdown(void) {
	_skr_mipgen_t* gens[] = { &_mipgen_2d, &_mipgen_cube };
	for (uint32_t i = 0; i < 2; i++) {
		if (skr_material_is_valid(&gens[i]->material)) skr_material_destroy(&gens[i]->material);
		if (skr_shader_is_valid(&gens[i]->shader))     skr_shader_destroy(&gens[i]->shader);
		memset(gens[i], 0, sizeof(*gens[i]));
	}
	if (_mipgen_sampler) { wgpuSamplerRelease(_mipgen_sampler); _mipgen_sampler = NULL; }
}
