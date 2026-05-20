// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include "sk_renderer.h"
#include <volk.h>

// Vulkan conversions
VkFormat             _skr_to_vk_vert_fmt    (skr_vertex_fmt_   format, uint8_t count);
VkCullModeFlags      _skr_to_vk_cull        (skr_cull_         cull);
VkCompareOp          _skr_to_vk_compare     (skr_compare_      compare);
VkBlendFactor        _skr_to_vk_blend_factor(skr_blend_factor_ factor);
VkBlendOp            _skr_to_vk_blend_op    (skr_blend_op_     op);
VkSamplerAddressMode _skr_to_vk_address     (skr_tex_address_  address);
VkFilter             _skr_to_vk_filter      (skr_tex_sample_   sample);
VkIndexType          _skr_to_vk_index_fmt   (skr_index_fmt_    format);
VkBufferUsageFlags   _skr_to_vk_buffer_usage(skr_buffer_type_  type);
VkStencilOp          _skr_to_vk_stencil_op  (skr_stencil_op_   op);

// Format queries
bool                 _skr_tex_fmt_is_yuv(skr_tex_fmt_ format);

// Format size queries (API-independent)
uint32_t             _skr_vert_fmt_to_size (skr_vertex_fmt_ format);
uint32_t             _skr_index_fmt_to_size(skr_index_fmt_  format);

