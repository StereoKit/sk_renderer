// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "skr_conversions.h"

///////////////////////////////////////////////////////////////////////////////
// Texture format conversions
///////////////////////////////////////////////////////////////////////////////

uint32_t skr_tex_fmt_to_native(skr_tex_fmt_ format) {
	switch (format) {
		case skr_tex_fmt_rgba32_srgb:   return VK_FORMAT_R8G8B8A8_SRGB;
		case skr_tex_fmt_rgba32_linear: return VK_FORMAT_R8G8B8A8_UNORM;
		case skr_tex_fmt_bgra32_srgb:   return VK_FORMAT_B8G8R8A8_SRGB;
		case skr_tex_fmt_bgra32_linear: return VK_FORMAT_B8G8R8A8_UNORM;
		case skr_tex_fmt_rg11b10:       return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
		case skr_tex_fmt_rgb10a2:       return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
		case skr_tex_fmt_rgba64u:       return VK_FORMAT_R16G16B16A16_UINT;
		case skr_tex_fmt_rgba64s:       return VK_FORMAT_R16G16B16A16_SINT;
		case skr_tex_fmt_rgba64f:       return VK_FORMAT_R16G16B16A16_SFLOAT;
		case skr_tex_fmt_rgba128:       return VK_FORMAT_R32G32B32A32_SFLOAT;
		case skr_tex_fmt_r8:            return VK_FORMAT_R8_UNORM;
		case skr_tex_fmt_r16un:         return VK_FORMAT_R16_UNORM;
		case skr_tex_fmt_r16sn:         return VK_FORMAT_R16_SNORM;
		case skr_tex_fmt_r16ui:         return VK_FORMAT_R16_UINT;
		case skr_tex_fmt_r16si:         return VK_FORMAT_R16_SINT;
		case skr_tex_fmt_r16f:          return VK_FORMAT_R16_SFLOAT;
		case skr_tex_fmt_r32f:          return VK_FORMAT_R32_SFLOAT;
		case skr_tex_fmt_depth32s8:     return VK_FORMAT_D32_SFLOAT_S8_UINT;
		case skr_tex_fmt_depth24s8:     return VK_FORMAT_D24_UNORM_S8_UINT;
		case skr_tex_fmt_depth16s8:     return VK_FORMAT_D16_UNORM_S8_UINT;
		case skr_tex_fmt_depth32:       return VK_FORMAT_D32_SFLOAT;
		case skr_tex_fmt_depth16:       return VK_FORMAT_D16_UNORM;
		case skr_tex_fmt_r8g8:          return VK_FORMAT_R8G8_UNORM;
		case skr_tex_fmt_rgb9e5:        return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
		// Compressed formats
		case skr_tex_fmt_bc1_rgb_srgb:  return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
		case skr_tex_fmt_bc1_rgb:       return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
		case skr_tex_fmt_bc1_rgba_srgb: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
		case skr_tex_fmt_bc1_rgba:      return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
		case skr_tex_fmt_bc3_rgba_srgb: return VK_FORMAT_BC3_SRGB_BLOCK;
		case skr_tex_fmt_bc3_rgba:      return VK_FORMAT_BC3_UNORM_BLOCK;
		case skr_tex_fmt_bc4_r:         return VK_FORMAT_BC4_UNORM_BLOCK;
		case skr_tex_fmt_bc5_rg:        return VK_FORMAT_BC5_UNORM_BLOCK;
		case skr_tex_fmt_bc7_rgba_srgb: return VK_FORMAT_BC7_SRGB_BLOCK;
		case skr_tex_fmt_bc7_rgba:      return VK_FORMAT_BC7_UNORM_BLOCK;
		// Mobile compressed formats
		case skr_tex_fmt_etc1_rgb:           return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
		case skr_tex_fmt_etc1_rgb_srgb:      return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
		case skr_tex_fmt_etc2_rgba_srgb:     return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
		case skr_tex_fmt_etc2_rgba:          return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
		case skr_tex_fmt_etc2_r11:           return VK_FORMAT_EAC_R11_UNORM_BLOCK;
		case skr_tex_fmt_etc2_rg11:          return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
		case skr_tex_fmt_pvrtc1_rgb_srgb:    return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;
		case skr_tex_fmt_pvrtc1_rgb:         return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
		case skr_tex_fmt_pvrtc1_rgba_srgb:   return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;
		case skr_tex_fmt_pvrtc1_rgba:        return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;
		case skr_tex_fmt_pvrtc2_rgba_srgb:   return VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG;
		case skr_tex_fmt_pvrtc2_rgba:        return VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG;
		case skr_tex_fmt_astc4x4_rgba_srgb:  return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
		case skr_tex_fmt_astc4x4_rgba:       return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
		case skr_tex_fmt_astc6x6_rgba_srgb:  return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
		case skr_tex_fmt_astc6x6_rgba:       return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
		case skr_tex_fmt_atc_rgb:            return VK_FORMAT_UNDEFINED; // No Vulkan equivalent
		case skr_tex_fmt_atc_rgba:           return VK_FORMAT_UNDEFINED; // No Vulkan equivalent
		// YUV / multi-plane formats
		case skr_tex_fmt_nv12:               return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		case skr_tex_fmt_p010:               return VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
		case skr_tex_fmt_yuv420p:            return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
		default:                             return VK_FORMAT_UNDEFINED;
	}
}

skr_tex_fmt_ skr_tex_fmt_from_native(uint32_t format) {
	switch (format) {
		case VK_FORMAT_R8G8B8A8_SRGB:              return skr_tex_fmt_rgba32_srgb;
		case VK_FORMAT_R8G8B8A8_UNORM:             return skr_tex_fmt_rgba32_linear;
		case VK_FORMAT_B8G8R8A8_SRGB:              return skr_tex_fmt_bgra32_srgb;
		case VK_FORMAT_B8G8R8A8_UNORM:             return skr_tex_fmt_bgra32_linear;
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32:    return skr_tex_fmt_rg11b10;
		case VK_FORMAT_A2B10G10R10_UNORM_PACK32:   return skr_tex_fmt_rgb10a2;
		case VK_FORMAT_R16G16B16A16_UINT:          return skr_tex_fmt_rgba64u;
		case VK_FORMAT_R16G16B16A16_SINT:          return skr_tex_fmt_rgba64s;
		case VK_FORMAT_R16G16B16A16_SFLOAT:        return skr_tex_fmt_rgba64f;
		case VK_FORMAT_R32G32B32A32_SFLOAT:        return skr_tex_fmt_rgba128;
		case VK_FORMAT_R8_UNORM:                   return skr_tex_fmt_r8;
		case VK_FORMAT_R16_UNORM:                  return skr_tex_fmt_r16un;
		case VK_FORMAT_R16_SNORM:                  return skr_tex_fmt_r16sn;
		case VK_FORMAT_R16_UINT:                   return skr_tex_fmt_r16ui;
		case VK_FORMAT_R16_SINT:                   return skr_tex_fmt_r16si;
		case VK_FORMAT_R16_SFLOAT:                 return skr_tex_fmt_r16f;
		case VK_FORMAT_R32_SFLOAT:                 return skr_tex_fmt_r32f;
		case VK_FORMAT_D32_SFLOAT_S8_UINT:         return skr_tex_fmt_depth32s8;
		case VK_FORMAT_D24_UNORM_S8_UINT:          return skr_tex_fmt_depth24s8;
		case VK_FORMAT_D16_UNORM_S8_UINT:          return skr_tex_fmt_depth16s8;
		case VK_FORMAT_D32_SFLOAT:                 return skr_tex_fmt_depth32;
		case VK_FORMAT_D16_UNORM:                  return skr_tex_fmt_depth16;
		case VK_FORMAT_R8G8_UNORM:                 return skr_tex_fmt_r8g8;
		case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:     return skr_tex_fmt_rgb9e5;
		// Compressed formats
		case VK_FORMAT_BC1_RGB_SRGB_BLOCK:         return skr_tex_fmt_bc1_rgb_srgb;
		case VK_FORMAT_BC1_RGB_UNORM_BLOCK:        return skr_tex_fmt_bc1_rgb;
		case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:        return skr_tex_fmt_bc1_rgba_srgb;
		case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:       return skr_tex_fmt_bc1_rgba;
		case VK_FORMAT_BC3_SRGB_BLOCK:             return skr_tex_fmt_bc3_rgba_srgb;
		case VK_FORMAT_BC3_UNORM_BLOCK:            return skr_tex_fmt_bc3_rgba;
		case VK_FORMAT_BC4_UNORM_BLOCK:            return skr_tex_fmt_bc4_r;
		case VK_FORMAT_BC5_UNORM_BLOCK:            return skr_tex_fmt_bc5_rg;
		case VK_FORMAT_BC7_SRGB_BLOCK:             return skr_tex_fmt_bc7_rgba_srgb;
		case VK_FORMAT_BC7_UNORM_BLOCK:            return skr_tex_fmt_bc7_rgba;
		// Mobile compressed formats
		case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:    return skr_tex_fmt_etc1_rgb;
		case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:     return skr_tex_fmt_etc1_rgb_srgb;
		case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:   return skr_tex_fmt_etc2_rgba_srgb;
		case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:  return skr_tex_fmt_etc2_rgba;
		case VK_FORMAT_EAC_R11_UNORM_BLOCK:        return skr_tex_fmt_etc2_r11;
		case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:     return skr_tex_fmt_etc2_rg11;
		case VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG: return skr_tex_fmt_pvrtc1_rgb_srgb;
		case VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG:return skr_tex_fmt_pvrtc1_rgb;
		case VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG: return skr_tex_fmt_pvrtc1_rgba_srgb;
		case VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG:return skr_tex_fmt_pvrtc1_rgba;
		case VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG: return skr_tex_fmt_pvrtc2_rgba_srgb;
		case VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG:return skr_tex_fmt_pvrtc2_rgba;
		case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:        return skr_tex_fmt_astc4x4_rgba_srgb;
		case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:       return skr_tex_fmt_astc4x4_rgba;
		case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:        return skr_tex_fmt_astc6x6_rgba_srgb;
		case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:       return skr_tex_fmt_astc6x6_rgba;
		// YUV / multi-plane formats
		case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:                       return skr_tex_fmt_nv12;
		case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:     return skr_tex_fmt_p010;
		case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:                      return skr_tex_fmt_yuv420p;
		default:                                   return skr_tex_fmt_none;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Format size queries (API-independent)
///////////////////////////////////////////////////////////////////////////////

uint32_t _skr_tex_fmt_to_size(skr_tex_fmt_ format) {
	switch (format) {
		case skr_tex_fmt_rgba32_srgb:
		case skr_tex_fmt_rgba32_linear:
		case skr_tex_fmt_bgra32_srgb:
		case skr_tex_fmt_bgra32_linear:
		case skr_tex_fmt_rg11b10:
		case skr_tex_fmt_rgb10a2:       return 4;
		case skr_tex_fmt_rgba64u:
		case skr_tex_fmt_rgba64s:
		case skr_tex_fmt_rgba64f:       return 8;
		case skr_tex_fmt_rgba128:       return 16;
		case skr_tex_fmt_r8:            return 1;
		case skr_tex_fmt_r16un:
		case skr_tex_fmt_r16sn:
		case skr_tex_fmt_r16ui:
		case skr_tex_fmt_r16si:
		case skr_tex_fmt_r16f:          return 2;
		case skr_tex_fmt_r32f:          return 4;
		case skr_tex_fmt_depth32s8:     return 5;
		case skr_tex_fmt_depth24s8:     return 4;
		case skr_tex_fmt_depth16s8:     return 3;
		case skr_tex_fmt_depth32:       return 4;
		case skr_tex_fmt_depth16:       return 2;
		case skr_tex_fmt_r8g8:          return 2;
		case skr_tex_fmt_rgb9e5:        return 4;
		default:                        return 0;
	}
}

uint32_t _skr_vert_fmt_to_size(skr_vertex_fmt_ format) {
	switch (format) {
		case skr_vertex_fmt_f64:              return 8;
		case skr_vertex_fmt_f32:              return 4;
		case skr_vertex_fmt_f16:              return 2;
		case skr_vertex_fmt_i32:              return 4;
		case skr_vertex_fmt_i16:              return 2;
		case skr_vertex_fmt_i8:               return 1;
		case skr_vertex_fmt_i32_normalized:   return 4;
		case skr_vertex_fmt_i16_normalized:   return 2;
		case skr_vertex_fmt_i8_normalized:    return 1;
		case skr_vertex_fmt_ui32:             return 4;
		case skr_vertex_fmt_ui16:             return 2;
		case skr_vertex_fmt_ui8:              return 1;
		case skr_vertex_fmt_ui32_normalized:  return 4;
		case skr_vertex_fmt_ui16_normalized:  return 2;
		case skr_vertex_fmt_ui8_normalized:   return 1;
		default:                              return 0;
	}
}

uint32_t _skr_index_fmt_to_size(skr_index_fmt_ format) {
	switch (format) {
		case skr_index_fmt_u32: return 4;
		case skr_index_fmt_u16: return 2;
		case skr_index_fmt_u8:  return 1;
		default:                return 2;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Vertex format conversions
///////////////////////////////////////////////////////////////////////////////

VkFormat _skr_to_vk_vert_fmt(skr_vertex_fmt_ format, uint8_t count) {
	// Note: This function handles all vertex format conversions
	// Simplified formats (u8, u8n, etc.) are legacy and map to the full enum values
	switch (format) {
		case skr_vertex_fmt_f64:
			if      (count == 1) return VK_FORMAT_R64_SFLOAT;
			else if (count == 2) return VK_FORMAT_R64G64_SFLOAT;
			else if (count == 3) return VK_FORMAT_R64G64B64_SFLOAT;
			else if (count == 4) return VK_FORMAT_R64G64B64A64_SFLOAT;
			break;
		case skr_vertex_fmt_f32:
			if      (count == 1) return VK_FORMAT_R32_SFLOAT;
			else if (count == 2) return VK_FORMAT_R32G32_SFLOAT;
			else if (count == 3) return VK_FORMAT_R32G32B32_SFLOAT;
			else if (count == 4) return VK_FORMAT_R32G32B32A32_SFLOAT;
			break;
		case skr_vertex_fmt_f16:
			if      (count == 1) return VK_FORMAT_R16_SFLOAT;
			else if (count == 2) return VK_FORMAT_R16G16_SFLOAT;
			else if (count == 3) return VK_FORMAT_R16G16B16_SFLOAT;
			else if (count == 4) return VK_FORMAT_R16G16B16A16_SFLOAT;
			break;
		case skr_vertex_fmt_i32:
			if      (count == 1) return VK_FORMAT_R32_SINT;
			else if (count == 2) return VK_FORMAT_R32G32_SINT;
			else if (count == 3) return VK_FORMAT_R32G32B32_SINT;
			else if (count == 4) return VK_FORMAT_R32G32B32A32_SINT;
			break;
		case skr_vertex_fmt_i16:
			if      (count == 1) return VK_FORMAT_R16_SINT;
			else if (count == 2) return VK_FORMAT_R16G16_SINT;
			else if (count == 3) return VK_FORMAT_R16G16B16_SINT;
			else if (count == 4) return VK_FORMAT_R16G16B16A16_SINT;
			break;
		case skr_vertex_fmt_i8:
			if      (count == 1) return VK_FORMAT_R8_SINT;
			else if (count == 2) return VK_FORMAT_R8G8_SINT;
			else if (count == 3) return VK_FORMAT_R8G8B8_SINT;
			else if (count == 4) return VK_FORMAT_R8G8B8A8_SINT;
			break;
		case skr_vertex_fmt_i16_normalized:
			if      (count == 1) return VK_FORMAT_R16_SNORM;
			else if (count == 2) return VK_FORMAT_R16G16_SNORM;
			else if (count == 3) return VK_FORMAT_R16G16B16_SNORM;
			else if (count == 4) return VK_FORMAT_R16G16B16A16_SNORM;
			break;
		case skr_vertex_fmt_i8_normalized:
			if      (count == 1) return VK_FORMAT_R8_SNORM;
			else if (count == 2) return VK_FORMAT_R8G8_SNORM;
			else if (count == 3) return VK_FORMAT_R8G8B8_SNORM;
			else if (count == 4) return VK_FORMAT_R8G8B8A8_SNORM;
			break;
		case skr_vertex_fmt_ui32:
			if      (count == 1) return VK_FORMAT_R32_UINT;
			else if (count == 2) return VK_FORMAT_R32G32_UINT;
			else if (count == 3) return VK_FORMAT_R32G32B32_UINT;
			else if (count == 4) return VK_FORMAT_R32G32B32A32_UINT;
			break;
		case skr_vertex_fmt_ui16:
			if      (count == 1) return VK_FORMAT_R16_UINT;
			else if (count == 2) return VK_FORMAT_R16G16_UINT;
			else if (count == 3) return VK_FORMAT_R16G16B16_UINT;
			else if (count == 4) return VK_FORMAT_R16G16B16A16_UINT;
			break;
		case skr_vertex_fmt_ui8:
			if      (count == 1) return VK_FORMAT_R8_UINT;
			else if (count == 2) return VK_FORMAT_R8G8_UINT;
			else if (count == 3) return VK_FORMAT_R8G8B8_UINT;
			else if (count == 4) return VK_FORMAT_R8G8B8A8_UINT;
			break;
		case skr_vertex_fmt_ui16_normalized:
			if      (count == 1) return VK_FORMAT_R16_UNORM;
			else if (count == 2) return VK_FORMAT_R16G16_UNORM;
			else if (count == 3) return VK_FORMAT_R16G16B16_UNORM;
			else if (count == 4) return VK_FORMAT_R16G16B16A16_UNORM;
			break;
		case skr_vertex_fmt_ui8_normalized:
			if      (count == 1) return VK_FORMAT_R8_UNORM;
			else if (count == 2) return VK_FORMAT_R8G8_UNORM;
			else if (count == 3) return VK_FORMAT_R8G8B8_UNORM;
			else if (count == 4) return VK_FORMAT_R8G8B8A8_UNORM;
			break;
		default: break;
	}
	return VK_FORMAT_UNDEFINED;
}

///////////////////////////////////////////////////////////////////////////////
// Material state conversions
///////////////////////////////////////////////////////////////////////////////

VkCullModeFlags _skr_to_vk_cull(skr_cull_ cull) {
	switch (cull) {
		case skr_cull_back:  return VK_CULL_MODE_BACK_BIT;
		case skr_cull_front: return VK_CULL_MODE_FRONT_BIT;
		case skr_cull_none:  return VK_CULL_MODE_NONE;
		default:             return VK_CULL_MODE_BACK_BIT;
	}
}

VkCompareOp _skr_to_vk_compare(skr_compare_ compare) {
	switch (compare) {
		case skr_compare_none:           return VK_COMPARE_OP_ALWAYS;
		case skr_compare_less:           return VK_COMPARE_OP_LESS;
		case skr_compare_less_or_eq:     return VK_COMPARE_OP_LESS_OR_EQUAL;
		case skr_compare_greater:        return VK_COMPARE_OP_GREATER;
		case skr_compare_greater_or_eq:  return VK_COMPARE_OP_GREATER_OR_EQUAL;
		case skr_compare_equal:          return VK_COMPARE_OP_EQUAL;
		case skr_compare_not_equal:      return VK_COMPARE_OP_NOT_EQUAL;
		case skr_compare_always:         return VK_COMPARE_OP_ALWAYS;
		case skr_compare_never:          return VK_COMPARE_OP_NEVER;
		default:                         return VK_COMPARE_OP_LESS;
	}
}

VkStencilOp _skr_to_vk_stencil_op(skr_stencil_op_ op) {
	switch (op) {
		case skr_stencil_op_keep:              return VK_STENCIL_OP_KEEP;
		case skr_stencil_op_zero:              return VK_STENCIL_OP_ZERO;
		case skr_stencil_op_replace:           return VK_STENCIL_OP_REPLACE;
		case skr_stencil_op_increment_clamp:   return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
		case skr_stencil_op_decrement_clamp:   return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
		case skr_stencil_op_invert:            return VK_STENCIL_OP_INVERT;
		case skr_stencil_op_increment_wrap:    return VK_STENCIL_OP_INCREMENT_AND_WRAP;
		case skr_stencil_op_decrement_wrap:    return VK_STENCIL_OP_DECREMENT_AND_WRAP;
		default:                               return VK_STENCIL_OP_KEEP;
	}
}

VkBlendFactor _skr_to_vk_blend_factor(skr_blend_factor_ factor) {
	switch (factor) {
		case skr_blend_zero:                      return VK_BLEND_FACTOR_ZERO;
		case skr_blend_one:                       return VK_BLEND_FACTOR_ONE;
		case skr_blend_src_color:                 return VK_BLEND_FACTOR_SRC_COLOR;
		case skr_blend_one_minus_src_color:       return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
		case skr_blend_dst_color:                 return VK_BLEND_FACTOR_DST_COLOR;
		case skr_blend_one_minus_dst_color:       return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
		case skr_blend_src_alpha:                 return VK_BLEND_FACTOR_SRC_ALPHA;
		case skr_blend_one_minus_src_alpha:       return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		case skr_blend_dst_alpha:                 return VK_BLEND_FACTOR_DST_ALPHA;
		case skr_blend_one_minus_dst_alpha:       return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
		case skr_blend_constant_color:            return VK_BLEND_FACTOR_CONSTANT_COLOR;
		case skr_blend_one_minus_constant_color:  return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
		case skr_blend_constant_alpha:            return VK_BLEND_FACTOR_CONSTANT_ALPHA;
		case skr_blend_one_minus_constant_alpha:  return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
		case skr_blend_src_alpha_saturate:        return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
		case skr_blend_src1_color:                return VK_BLEND_FACTOR_SRC1_COLOR;
		case skr_blend_one_minus_src1_color:      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
		case skr_blend_src1_alpha:                return VK_BLEND_FACTOR_SRC1_ALPHA;
		case skr_blend_one_minus_src1_alpha:      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
		default:                                  return VK_BLEND_FACTOR_ZERO;
	}
}

VkBlendOp _skr_to_vk_blend_op(skr_blend_op_ op) {
	switch (op) {
		case skr_blend_op_add:              return VK_BLEND_OP_ADD;
		case skr_blend_op_subtract:         return VK_BLEND_OP_SUBTRACT;
		case skr_blend_op_reverse_subtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
		case skr_blend_op_min:              return VK_BLEND_OP_MIN;
		case skr_blend_op_max:              return VK_BLEND_OP_MAX;
		default:                            return VK_BLEND_OP_ADD;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Sampler state conversions
///////////////////////////////////////////////////////////////////////////////

VkSamplerAddressMode _skr_to_vk_address(skr_tex_address_ address) {
	switch (address) {
		case skr_tex_address_wrap:   return VK_SAMPLER_ADDRESS_MODE_REPEAT;
		case skr_tex_address_clamp:  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		case skr_tex_address_mirror: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
		default:                     return VK_SAMPLER_ADDRESS_MODE_REPEAT;
	}
}

VkFilter _skr_to_vk_filter(skr_tex_sample_ sample) {
	switch (sample) {
		case skr_tex_sample_linear:      return VK_FILTER_LINEAR;
		case skr_tex_sample_point:       return VK_FILTER_NEAREST;
		case skr_tex_sample_anisotropic: return VK_FILTER_LINEAR;
		default:                         return VK_FILTER_LINEAR;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Index format conversions
///////////////////////////////////////////////////////////////////////////////

VkIndexType _skr_to_vk_index_fmt(skr_index_fmt_ format) {
	switch (format) {
		case skr_index_fmt_u32: return VK_INDEX_TYPE_UINT32;
		case skr_index_fmt_u16: return VK_INDEX_TYPE_UINT16;
		case skr_index_fmt_u8:  return VK_INDEX_TYPE_UINT8_EXT;
		default:                return VK_INDEX_TYPE_UINT16;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Buffer type conversions
///////////////////////////////////////////////////////////////////////////////

VkBufferUsageFlags _skr_to_vk_buffer_usage(skr_buffer_type_ type) {
	VkBufferUsageFlags flags = 0;
	if (type & skr_buffer_type_vertex)   flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	if (type & skr_buffer_type_index)    flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	if (type & skr_buffer_type_constant) flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	if (type & skr_buffer_type_storage)  flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	return flags;
}

///////////////////////////////////////////////////////////////////////////////
// Format helpers
///////////////////////////////////////////////////////////////////////////////

bool _skr_format_has_stencil(VkFormat format) {
	return format == VK_FORMAT_D24_UNORM_S8_UINT ||
	       format == VK_FORMAT_D16_UNORM_S8_UINT ||
	       format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

bool _skr_tex_fmt_is_yuv(skr_tex_fmt_ format) {
	return format == skr_tex_fmt_nv12    ||
	       format == skr_tex_fmt_p010    ||
	       format == skr_tex_fmt_yuv420p;
}

bool _skr_format_is_depth(VkFormat format) {
	return format == VK_FORMAT_D16_UNORM ||
	       format == VK_FORMAT_D16_UNORM_S8_UINT ||
	       format == VK_FORMAT_D24_UNORM_S8_UINT ||
	       format == VK_FORMAT_D32_SFLOAT ||
	       format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
	       format == VK_FORMAT_X8_D24_UNORM_PACK32;
}
