// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////
// Pipeline management, mirroring the Vulkan backend's three dimensions:
// 1. Material   - shader, cull, depth test, blend, spec constants
// 2. Pass       - color/depth formats + sample count (render passes are
//                 implicit in WebGPU) + view index (sk_view_index override,
//                 since layered targets render one pass per layer)
// 3. Vert format- vertex layout
//
// Materials register for an index (deduped by key memcmp); pipelines are
// created lazily per (material, pass, vertformat) triplet. Register/unregister
// may run on any thread (writer mutex); lookups are lock-free reads, and
// _skr_pipeline_get is draw-thread-only (see the threading model note in
// _sk_renderer.h).
///////////////////////////////////////////////////////////////////////////////

typedef struct skr_pipeline_pass_key_t {
	WGPUTextureFormat color_format;   // Undefined = depth-only pass
	WGPUTextureFormat depth_format;   // Undefined = no depth
	uint32_t          sample_count;
	uint32_t          view_index;     // value for the sk_view_index override
} skr_pipeline_pass_key_t;

void    _skr_pipeline_init          (void);
void    _skr_pipeline_shutdown      (void);
void    _skr_pipeline_registry_drain(void); // release handles of materials destroyed last frame; call at frame begin

int32_t _skr_pipeline_register_material  (const _skr_pipeline_material_key_t* key);
void    _skr_pipeline_unregister_material(int32_t material_idx);
int32_t _skr_pipeline_register_vertformat(const skr_vert_type_t* vert_type);

// Lazily create or fetch the pipeline for a triplet
WGPURenderPipeline  _skr_pipeline_get            (int32_t material_idx, const skr_pipeline_pass_key_t* pass, int32_t vertformat_idx);
WGPUBindGroupLayout _skr_pipeline_get_bind_layout(int32_t material_idx);

// Bind group layout straight from shader meta — also used by compute
WGPUBindGroupLayout _skr_bind_layout_create(const sksc_shader_meta_t* meta, uint8_t stage_mask);

// The registered material's key (shader/meta access for draw-time binding)
const _skr_pipeline_material_key_t* _skr_pipeline_get_key(int32_t material_idx);

// Resolve user spec-constant overrides against shader meta into the bit
// patterns a pipeline key stores (defaults where not overridden)
void _skr_resolve_spec_constants(const sksc_shader_meta_t* meta, const skr_spec_constant_t* specs, uint32_t spec_count, uint32_t out_values[SKR_MAX_SPEC_CONSTANTS]);

// WGPUConstantEntry list for one stage from resolved spec values; keys are
// decimal @id(N) strings backed by `keys` (caller keeps it alive). Adds the
// sk_view_index override when has_view_index is set.
typedef char _skr_const_key_t[12];
uint32_t _skr_stage_constants(const sksc_shader_meta_t* meta, const uint32_t* values, skr_stage_ stage, bool has_view_index, uint32_t view_index, WGPUConstantEntry* out, _skr_const_key_t* keys, uint32_t out_max);

// True when the resource paired with a sampler is depth-comparison sampled
bool _skr_sampler_is_comparison(const sksc_shader_meta_t* meta, uint32_t paired_slot);
