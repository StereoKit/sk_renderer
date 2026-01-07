// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include "sk_renderer.h"
#include "_sk_renderer.h"

#include "skr_vulkan.h"

#include <volk.h>
#include <stdint.h>

///////////////////////////////////////////////////////////////////////////////
// Pipeline management system
//
// Pipelines are determined by three dimensions:
// 1. Material dimension - shader, cull, depth test, blend, etc.
// 2. Render pass dimension - color format, depth format, MSAA samples
// 3. Vertex format dimension - vertex layout (position, normal, uv, etc.)
//
// Each dimension can be registered to get an integer index, which allows
// for fast 3D lookup of pipelines.
///////////////////////////////////////////////////////////////////////////////


// Initialize/shutdown the pipeline system
void                  _skr_pipeline_init                 (void);
void                  _skr_pipeline_shutdown             (void);

// Register/unregister dimensions - returns index for fast lookup
int32_t               _skr_pipeline_register_material    (const _skr_pipeline_material_key_t*  key);
int32_t               _skr_pipeline_register_renderpass  (const skr_pipeline_renderpass_key_t* key);
int32_t               _skr_pipeline_register_vertformat  (const skr_vert_type_t                vert_type);
void                  _skr_pipeline_unregister_material  (int32_t material_idx  );
void                  _skr_pipeline_unregister_renderpass(int32_t renderpass_idx);
void                  _skr_pipeline_unregister_vertformat(int32_t vertformat_idx);

// Get or create pipeline for a material/renderpass/vertformat triplet
// NOTE: These get functions do NOT lock internally for performance. The caller
// must ensure thread safety by either:
// 1. Calling from within a locked region (_skr_pipeline_lock/_skr_pipeline_unlock)
// 2. Ensuring no concurrent modifications (single-threaded use)
VkPipeline            _skr_pipeline_get                  (int32_t material_idx, int32_t renderpass_idx, int32_t vertformat_idx);
VkPipelineLayout      _skr_pipeline_get_layout           (int32_t material_idx  );
VkDescriptorSetLayout _skr_pipeline_get_descriptor_layout(int32_t material_idx  );
VkRenderPass          _skr_pipeline_get_renderpass       (int32_t renderpass_idx);

// Thread safety: Lock the pipeline cache for a region of operations.
// Use these to protect multiple get calls during rendering.
// Registration functions lock internally, so they can be called without
// explicitly locking (and will block if another thread holds the lock).
void                  _skr_pipeline_lock                 (void);
void                  _skr_pipeline_unlock               (void);
