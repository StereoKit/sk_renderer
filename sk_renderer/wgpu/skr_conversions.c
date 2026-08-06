// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////
// skr_tex_fmt_ <-> WGPUTextureFormat. Formats WebGPU has no equivalent for
// (PVRTC, ATC, HDR ASTC, YUV multi-plane, packed depth16s8/24s8 variants
// beyond Depth24PlusStencil8) map to Undefined and report unsupported.

WGPUTextureFormat _skr_tex_fmt_to_wgpu(skr_tex_fmt_ format) {
	switch (format) {
		case skr_tex_fmt_rgba32_srgb:   return WGPUTextureFormat_RGBA8UnormSrgb;
		case skr_tex_fmt_rgba32:        return WGPUTextureFormat_RGBA8Unorm;
		case skr_tex_fmt_bgra32_srgb:   return WGPUTextureFormat_BGRA8UnormSrgb;
		case skr_tex_fmt_bgra32:        return WGPUTextureFormat_BGRA8Unorm;

		case skr_tex_fmt_rgba64un:      return WGPUTextureFormat_Undefined; // no 16-bit unorm in core WebGPU
		case skr_tex_fmt_rgba64sn:      return WGPUTextureFormat_Undefined;
		case skr_tex_fmt_rgba64ui:      return WGPUTextureFormat_RGBA16Uint;
		case skr_tex_fmt_rgba64si:      return WGPUTextureFormat_RGBA16Sint;
		case skr_tex_fmt_rgba64f:       return WGPUTextureFormat_RGBA16Float;

		case skr_tex_fmt_rgba128f:      return WGPUTextureFormat_RGBA32Float;

		case skr_tex_fmt_rg11b10uf:     return WGPUTextureFormat_RG11B10Ufloat;
		case skr_tex_fmt_rgb10a2:       return WGPUTextureFormat_RGB10A2Unorm;
		case skr_tex_fmt_rgb9e5uf:      return WGPUTextureFormat_RGB9E5Ufloat;

		case skr_tex_fmt_r8:            return WGPUTextureFormat_R8Unorm;
		case skr_tex_fmt_r8sn:          return WGPUTextureFormat_R8Snorm;
		case skr_tex_fmt_r8ui:          return WGPUTextureFormat_R8Uint;
		case skr_tex_fmt_r8si:          return WGPUTextureFormat_R8Sint;
		case skr_tex_fmt_r8_srgb:       return WGPUTextureFormat_Undefined;

		case skr_tex_fmt_r8g8:          return WGPUTextureFormat_RG8Unorm;

		case skr_tex_fmt_r16:           return WGPUTextureFormat_Undefined; // R16Unorm is non-core
		case skr_tex_fmt_r16sn:         return WGPUTextureFormat_Undefined;
		case skr_tex_fmt_r16ui:         return WGPUTextureFormat_R16Uint;
		case skr_tex_fmt_r16si:         return WGPUTextureFormat_R16Sint;
		case skr_tex_fmt_r16f:          return WGPUTextureFormat_R16Float;

		case skr_tex_fmt_r32ui:         return WGPUTextureFormat_R32Uint;
		case skr_tex_fmt_r32si:         return WGPUTextureFormat_R32Sint;
		case skr_tex_fmt_r32f:          return WGPUTextureFormat_R32Float;

		case skr_tex_fmt_depth16:       return WGPUTextureFormat_Depth16Unorm;
		case skr_tex_fmt_depth16s8:     return WGPUTextureFormat_Depth24PlusStencil8; // closest with stencil
		case skr_tex_fmt_depth24s8:     return WGPUTextureFormat_Depth24PlusStencil8;
		case skr_tex_fmt_depth32:       return WGPUTextureFormat_Depth32Float;
		case skr_tex_fmt_depth32s8:     return _skr_wgpu.feat_depth32s8 ? WGPUTextureFormat_Depth32FloatStencil8 : WGPUTextureFormat_Depth24PlusStencil8;

		case skr_tex_fmt_bc1_rgb_srgb:  return WGPUTextureFormat_BC1RGBAUnormSrgb;
		case skr_tex_fmt_bc1_rgb:       return WGPUTextureFormat_BC1RGBAUnorm;
		case skr_tex_fmt_bc1_rgba_srgb: return WGPUTextureFormat_BC1RGBAUnormSrgb;
		case skr_tex_fmt_bc1_rgba:      return WGPUTextureFormat_BC1RGBAUnorm;
		case skr_tex_fmt_bc2_rgba_srgb: return WGPUTextureFormat_BC2RGBAUnormSrgb;
		case skr_tex_fmt_bc2_rgba:      return WGPUTextureFormat_BC2RGBAUnorm;
		case skr_tex_fmt_bc3_rgba_srgb: return WGPUTextureFormat_BC3RGBAUnormSrgb;
		case skr_tex_fmt_bc3_rgba:      return WGPUTextureFormat_BC3RGBAUnorm;
		case skr_tex_fmt_bc4_r:         return WGPUTextureFormat_BC4RUnorm;
		case skr_tex_fmt_bc4_rsn:       return WGPUTextureFormat_BC4RSnorm;
		case skr_tex_fmt_bc5_rg:        return WGPUTextureFormat_BC5RGUnorm;
		case skr_tex_fmt_bc5_rgsn:      return WGPUTextureFormat_BC5RGSnorm;
		case skr_tex_fmt_bc6h_rgbuf:    return WGPUTextureFormat_BC6HRGBUfloat;
		case skr_tex_fmt_bc6h_rgbf:     return WGPUTextureFormat_BC6HRGBFloat;
		case skr_tex_fmt_bc7_rgba_srgb: return WGPUTextureFormat_BC7RGBAUnormSrgb;
		case skr_tex_fmt_bc7_rgba:      return WGPUTextureFormat_BC7RGBAUnorm;

		// ETC1 data is a bit-exact subset of ETC2 RGB8, same as the Vulkan
		// backend's ETC2_R8G8B8 mapping
		case skr_tex_fmt_etc1_rgb:      return WGPUTextureFormat_ETC2RGB8Unorm;
		case skr_tex_fmt_etc1_rgb_srgb: return WGPUTextureFormat_ETC2RGB8UnormSrgb;
		case skr_tex_fmt_etc2_rgba_srgb:return WGPUTextureFormat_ETC2RGBA8UnormSrgb;
		case skr_tex_fmt_etc2_rgba:     return WGPUTextureFormat_ETC2RGBA8Unorm;
		case skr_tex_fmt_etc2_r11:      return WGPUTextureFormat_EACR11Unorm;
		case skr_tex_fmt_etc2_rg11:     return WGPUTextureFormat_EACRG11Unorm;

		case skr_tex_fmt_astc4x4_rgba_srgb: return WGPUTextureFormat_ASTC4x4UnormSrgb;
		case skr_tex_fmt_astc4x4_rgba:      return WGPUTextureFormat_ASTC4x4Unorm;
		case skr_tex_fmt_astc6x6_rgba_srgb: return WGPUTextureFormat_ASTC6x6UnormSrgb;
		case skr_tex_fmt_astc6x6_rgba:      return WGPUTextureFormat_ASTC6x6Unorm;
		// astc8x8_rgba_hdr stays Undefined: WebGPU's texture-compression-astc
		// feature is LDR-only; the spec has no HDR ASTC at all

		default:                        return WGPUTextureFormat_Undefined;
	}
}

skr_tex_fmt_ _skr_tex_fmt_from_wgpu(WGPUTextureFormat format) {
	switch (format) {
		case WGPUTextureFormat_RGBA8UnormSrgb:        return skr_tex_fmt_rgba32_srgb;
		case WGPUTextureFormat_RGBA8Unorm:            return skr_tex_fmt_rgba32;
		case WGPUTextureFormat_BGRA8UnormSrgb:        return skr_tex_fmt_bgra32_srgb;
		case WGPUTextureFormat_BGRA8Unorm:            return skr_tex_fmt_bgra32;
		case WGPUTextureFormat_RGBA16Uint:            return skr_tex_fmt_rgba64ui;
		case WGPUTextureFormat_RGBA16Sint:            return skr_tex_fmt_rgba64si;
		case WGPUTextureFormat_RGBA16Float:           return skr_tex_fmt_rgba64f;
		case WGPUTextureFormat_RGBA32Float:           return skr_tex_fmt_rgba128f;
		case WGPUTextureFormat_RG11B10Ufloat:         return skr_tex_fmt_rg11b10uf;
		case WGPUTextureFormat_RGB10A2Unorm:          return skr_tex_fmt_rgb10a2;
		case WGPUTextureFormat_RGB9E5Ufloat:          return skr_tex_fmt_rgb9e5uf;
		case WGPUTextureFormat_R8Unorm:               return skr_tex_fmt_r8;
		case WGPUTextureFormat_R8Snorm:               return skr_tex_fmt_r8sn;
		case WGPUTextureFormat_R8Uint:                return skr_tex_fmt_r8ui;
		case WGPUTextureFormat_R8Sint:                return skr_tex_fmt_r8si;
		case WGPUTextureFormat_RG8Unorm:              return skr_tex_fmt_r8g8;
		case WGPUTextureFormat_R16Uint:               return skr_tex_fmt_r16ui;
		case WGPUTextureFormat_R16Sint:               return skr_tex_fmt_r16si;
		case WGPUTextureFormat_R16Float:              return skr_tex_fmt_r16f;
		case WGPUTextureFormat_R32Uint:               return skr_tex_fmt_r32ui;
		case WGPUTextureFormat_R32Sint:               return skr_tex_fmt_r32si;
		case WGPUTextureFormat_R32Float:              return skr_tex_fmt_r32f;
		case WGPUTextureFormat_Depth16Unorm:          return skr_tex_fmt_depth16;
		case WGPUTextureFormat_Depth24PlusStencil8:   return skr_tex_fmt_depth24s8;
		case WGPUTextureFormat_Depth32Float:          return skr_tex_fmt_depth32;
		case WGPUTextureFormat_Depth32FloatStencil8:  return skr_tex_fmt_depth32s8;
		case WGPUTextureFormat_BC1RGBAUnormSrgb:      return skr_tex_fmt_bc1_rgba_srgb;
		case WGPUTextureFormat_BC1RGBAUnorm:          return skr_tex_fmt_bc1_rgba;
		case WGPUTextureFormat_BC2RGBAUnormSrgb:      return skr_tex_fmt_bc2_rgba_srgb;
		case WGPUTextureFormat_BC2RGBAUnorm:          return skr_tex_fmt_bc2_rgba;
		case WGPUTextureFormat_BC3RGBAUnormSrgb:      return skr_tex_fmt_bc3_rgba_srgb;
		case WGPUTextureFormat_BC3RGBAUnorm:          return skr_tex_fmt_bc3_rgba;
		case WGPUTextureFormat_BC4RUnorm:             return skr_tex_fmt_bc4_r;
		case WGPUTextureFormat_BC4RSnorm:             return skr_tex_fmt_bc4_rsn;
		case WGPUTextureFormat_BC5RGUnorm:            return skr_tex_fmt_bc5_rg;
		case WGPUTextureFormat_BC5RGSnorm:            return skr_tex_fmt_bc5_rgsn;
		case WGPUTextureFormat_BC6HRGBUfloat:         return skr_tex_fmt_bc6h_rgbuf;
		case WGPUTextureFormat_BC6HRGBFloat:          return skr_tex_fmt_bc6h_rgbf;
		case WGPUTextureFormat_BC7RGBAUnormSrgb:      return skr_tex_fmt_bc7_rgba_srgb;
		case WGPUTextureFormat_BC7RGBAUnorm:          return skr_tex_fmt_bc7_rgba;
		case WGPUTextureFormat_ETC2RGB8Unorm:         return skr_tex_fmt_etc1_rgb;
		case WGPUTextureFormat_ETC2RGB8UnormSrgb:     return skr_tex_fmt_etc1_rgb_srgb;
		case WGPUTextureFormat_ETC2RGBA8UnormSrgb:    return skr_tex_fmt_etc2_rgba_srgb;
		case WGPUTextureFormat_ETC2RGBA8Unorm:        return skr_tex_fmt_etc2_rgba;
		case WGPUTextureFormat_EACR11Unorm:           return skr_tex_fmt_etc2_r11;
		case WGPUTextureFormat_EACRG11Unorm:          return skr_tex_fmt_etc2_rg11;
		case WGPUTextureFormat_ASTC4x4UnormSrgb:      return skr_tex_fmt_astc4x4_rgba_srgb;
		case WGPUTextureFormat_ASTC4x4Unorm:          return skr_tex_fmt_astc4x4_rgba;
		case WGPUTextureFormat_ASTC6x6UnormSrgb:      return skr_tex_fmt_astc6x6_rgba_srgb;
		case WGPUTextureFormat_ASTC6x6Unorm:          return skr_tex_fmt_astc6x6_rgba;
		default:                                      return skr_tex_fmt_none;
	}
}

uint32_t skr_tex_fmt_to_native(skr_tex_fmt_ format) {
	return (uint32_t)_skr_tex_fmt_to_wgpu(format);
}

skr_tex_fmt_ skr_tex_fmt_from_native(uint32_t native_format) {
	return _skr_tex_fmt_from_wgpu((WGPUTextureFormat)native_format);
}
