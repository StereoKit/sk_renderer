// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include <webgpu/webgpu.h>

#define SKR_MAX_FRAMES_IN_FLIGHT 3

// Future type for tracking GPU work completion (must be before skr_surface_t).
// On WebGPU this wraps queue-done tracking; skr_future_check polls the
// instance, skr_future_wait is a hard error on web (browsers can't block).
typedef struct skr_future_t {
	void*    slot;          // Pointer to internal submission-slot state
	uint64_t generation;    // Generation counter to detect slot reuse (must match slot's generation)
} skr_future_t;

// Texture readback handle for async GPU->CPU texture data transfer
typedef struct skr_tex_readback_t {
	void*        data;      // CPU-accessible data pointer (valid after future completes)
	uint32_t     size;      // Data size in bytes
	skr_future_t future;    // Poll with skr_future_check() — never block on web
	void*        _internal; // Internal state (staging buffer/map) - do not access directly
} skr_tex_readback_t;

// No update ring here: wgpuQueueWriteBuffer stages data internally and lands
// in queue order, so in-flight frames keep the contents they were submitted
// with (see skr_buffer_set)
typedef struct skr_buffer_t {
	WGPUBuffer          buffer;
	uint32_t            size;
	skr_buffer_type_    type;
	skr_use_            use;
} skr_buffer_t;

typedef struct skr_vert_type_t {
	WGPUVertexAttribute*    attributes;
	WGPUVertexBufferLayout* bindings;         // Array of buffer layouts (one per vertex buffer)
	uint32_t                binding_count;    // Number of vertex buffer layouts
	skr_vert_component_t*   components;
	uint32_t                component_count;
	int32_t                 pipeline_idx;     // Cached pipeline vertex format index
} skr_vert_type_t;

#define SKR_MAX_VERTEX_BUFFERS 2

typedef struct skr_mesh_t {
	skr_buffer_t           vertex_buffers[SKR_MAX_VERTEX_BUFFERS];
	uint32_t               vertex_buffer_count;   // Number of vertex buffers in use
	uint32_t               vertex_buffer_owned;   // Bitmask: which buffers are owned (vs externally referenced)
	skr_buffer_t           index_buffer;
	const skr_vert_type_t* vert_type;
	skr_index_fmt_         ind_format;
	WGPUIndexFormat        ind_format_wgpu;
	uint32_t               vert_count;
	uint32_t               ind_count;
} skr_mesh_t;

typedef struct skr_tex_t {
	WGPUTexture            texture;
	WGPUTextureView        view;             // Whole-resource view for sampling
	WGPUTextureView*       layer_views;      // Lazily-created single-layer render views (layer_count
	                                         // entries, NULL until first layered render) — WebGPU has no
	                                         // multiview, so layered targets render one pass per layer
	WGPUSampler            sampler;          // Filtering sampler (compare stripped)
	WGPUSampler            sampler_compare;  // Comparison variant, NULL unless sample_compare is set.
	                                         // WebGPU bind layouts are strict about comparison vs
	                                         // filtering samplers, so both variants exist and draws
	                                         // bind whichever the shader's usage expects.
	skr_tex_sampler_t      sampler_settings; // Sampler settings
	skr_vec3i_t            size;
	// Authored dimensions, pre-padding. WebGPU (following D3D12) requires
	// block-aligned compressed textures, so `size` rounds up to the block
	// grid; caller data (and its mip chain) is still laid out for these
	// dimensions, and skr_tex_set_data derives upload pitches from them.
	// Equal to `size` for everything else.
	skr_vec3i_t            data_size;
	skr_tex_fmt_           format;
	skr_tex_flags_         flags;
	uint32_t               samples;          // Sample count for MSAA (1 or 4 on WebGPU)
	uint32_t               mip_levels;       // Number of mip levels
	uint32_t               layer_count;      // Number of array layers (1 for regular, N for arrays, 6 for cubemaps)
	bool                   is_external;      // Texture is externally owned (surface frame etc., don't destroy)
} skr_tex_t;

typedef struct skr_surface_t {
	WGPUSurface    surface;
	skr_tex_t      current;       // Wraps the surface's current frame texture between next_tex/present
	uint32_t       frame_idx;
	// WebGPU surfaces can't query their window's size, so the app sets `size`
	// before calling skr_surface_resize (initial size is set at create time
	// from the first configure default)
	skr_vec2i_t    size;
	uint32_t       format;        // WGPUTextureFormat the surface is configured with
	// Format frame views are created with. Differs from `format` on the web:
	// canvases only configure base (non-sRGB) formats, and the sRGB encode
	// comes from an explicit viewFormats entry + sRGB frame views.
	uint32_t       view_format;
	bool           configured;
} skr_surface_t;

typedef struct skr_shader_stage_t {
	WGPUShaderModule shader;
	skr_stage_       type;
	bool             has_view_index; // WGSL declares the sk_view_index override; pipelines must set it
} skr_shader_stage_t;

typedef struct skr_shader_t {
	sksc_shader_meta_t  meta;
	skr_shader_stage_t  vertex_stage;
	skr_shader_stage_t  pixel_stage;
	skr_shader_stage_t  compute_stage;
} skr_shader_t;

typedef struct  {
	union {
		skr_tex_t*    texture;
		skr_buffer_t* buffer;
	};
	skr_bind_t bind;
} skr_material_bind_t;

// Internal key struct for pipeline-affecting material parameters only.
// Excludes queue_offset which affects render list sorting but not pipeline
// state. Per-pass state (color/depth formats, sample count, view index) is
// appended at pipeline lookup, not stored here.
//
// Layout is hand-tuned so every byte corresponds to a named field — there are
// no implicit alignment holes on either 64-bit native or 32-bit WASM builds.
// _pad0/_pad1/_pad2 are explicit so designated initializers zero them via the
// C99 "unspecified members → zero" rule, and memcmp-based dedup stays
// byte-deterministic regardless of compiler or C standard version.
#define SKR_MAX_SPEC_CONSTANTS     4
typedef struct {
	// Pointer-aligned block
	const skr_shader_t*  shader;

	// 4-byte aligned sub-structs (no internal padding: all 4-byte fields)
	skr_blend_state_t    blend_state;
	skr_stencil_state_t  stencil_front;
	skr_stencil_state_t  stencil_back;

	// 4-byte aligned scalars
	skr_cull_            cull;
	skr_write_           write_mask;
	skr_compare_         depth_test;
	uint32_t             spec_constant_values[SKR_MAX_SPEC_CONSTANTS]; // Bit patterns for the shader's spec constants, in shader meta order (defaults where not overridden)

	// 1-byte fields packed at the end with explicit padding to fill the
	// alignment tail. _pad* must remain zero — initializer rules above keep
	// this true without any extra code at the call sites.
	bool                 alpha_to_coverage;
	bool                 depth_clamp;
	bool                 wireframe;
	uint8_t              _pad0;   // must be 0
	uint32_t             _pad1;   // must be 0
	uint32_t             _pad2;   // must be 0 (fills tail to pointer alignment on 64-bit)
} _skr_pipeline_material_key_t;

#ifdef __cplusplus
static_assert(sizeof(_skr_pipeline_material_key_t) == (sizeof(void*) == 8 ? 128 : 124),
	"_skr_pipeline_material_key_t layout drifted; explicit padding and memcmp dedup may be broken");
#else
_Static_assert(sizeof(_skr_pipeline_material_key_t) == (sizeof(void*) == 8 ? 128 : 124),
	"_skr_pipeline_material_key_t layout drifted; explicit padding and memcmp dedup may be broken");
#endif

typedef struct skr_material_t {
	int32_t                      pipeline_material_idx; // Index into pipeline cache
	_skr_pipeline_material_key_t key;                   // Pipeline-affecting state
	int32_t                      queue_offset;          // Render queue offset (not pipeline-affecting)

	int32_t                bind_start;            // Index into global bind pool (-1 if none)
	uint32_t               bind_count;
	// Material parameters
	void*                  param_buffer;          // CPU-side parameter data
	uint32_t               param_buffer_size;     // Size of parameter buffer in bytes

	uint32_t               instance_buffer_stride; // Element size of instance buffer (0 = no instance buffer)
} skr_material_t;

typedef struct skr_compute_t {
	const skr_shader_t*    shader;  // Reference to shader (not owned)
	WGPUPipelineLayout     layout;
	WGPUBindGroupLayout    bind_group_layout;
	WGPUComputePipeline    pipeline;

	skr_material_bind_t*   binds;
	uint32_t               bind_count;

	// CPU-side parameter staging
	void*                  param_buffer;
	uint32_t               param_buffer_size;
} skr_compute_t;

// Flags for skr_render_item_t::flags (vertex_buffer_count stored in bits 2-3)
typedef enum skr_item_flag_ {
	skr_item_flag_index_32bit    = 1 << 1, // Index format is uint32 (vs uint16)
	skr_item_flag_vb_count_shift = 2,      // Vertex buffer count (0-2) in bits 2-3
	skr_item_flag_vb_count_mask  = 3 << 2,
} skr_item_flag_;

// Render item with inlined mesh/material data - mesh/material can be destroyed after add.
// Fields are packed by size to minimize padding.
typedef struct skr_render_item_t {
	// Pointer-aligned (WGPUBuffer = pointer)
	WGPUBuffer  vertex_buffers[SKR_MAX_VERTEX_BUFFERS]; // From mesh->vertex_buffers[].buffer
	WGPUBuffer  index_buffer;                           // From mesh->index_buffer.buffer
	uint64_t    sort_key;                               // Pre-computed sort key for fast sorting

	// 4-byte aligned
	uint32_t    vert_count;           // From mesh->vert_count (for non-indexed draws)
	uint32_t    param_data_offset;    // Offset into render_list->material_data (bytes)
	uint32_t    instance_offset;      // Offset into render_list->instance_data (bytes)
	uint32_t    instance_count;       // Number of instances to draw
	int32_t     first_index;          // Index buffer offset (0 = use mesh defaults)
	int32_t     index_count;          // Number of indices to draw (resolved at add-time, never 0)
	int32_t     vertex_offset;        // Base vertex offset
	int32_t     bind_start;           // Index into bind pool (bind pool uses deferred destruction)

	// 2-byte aligned
	uint16_t    pipeline_vert_idx;      // From mesh->vert_type->pipeline_idx
	uint16_t    pipeline_material_idx;  // From material->pipeline_material_idx
	uint16_t    param_buffer_size;      // From material->param_buffer_size
	uint16_t    instance_data_size;     // Size per instance (bytes)

	// 1-byte aligned
	uint8_t     bind_count;           // From material->bind_count (textures+buffers, rarely >32)
	uint8_t     flags;                // skr_item_flag_ bits
} skr_render_item_t;

// Item offsets into the data blobs are baked at add-time, so the sort just
// reorders items — the blobs never move (see _skr_render_list_sort)
typedef struct skr_render_list_t {
	skr_render_item_t* items;
	uint32_t           count;
	uint32_t           capacity;
	uint8_t*           instance_data;
	uint32_t           instance_data_used;
	uint32_t           instance_data_capacity;
	uint8_t*           material_data;
	uint32_t           material_data_used;
	uint32_t           material_data_capacity;
	bool               needs_sort;              // Dirty flag for sorting
} skr_render_list_t;
