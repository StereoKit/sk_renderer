// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include <volk.h>

#define SKR_MAX_FRAMES_IN_FLIGHT 3
#define SKR_MAX_SURFACES 2  // Maximum surfaces for VR stereo rendering

// Number of copies in a dynamic buffer's flipbook ring. This is one more than
// the in-flight frame count: callers write the new frame's data before the
// oldest in-flight frame is retired (the per-frame fence wait happens at
// present/acquire, after dynamic buffers are updated), so up to
// SKR_MAX_FRAMES_IN_FLIGHT frames can still be reading older slots when we
// write. We need a free slot beyond those to avoid stomping in-flight data.
#define SKR_DYNAMIC_BUFFER_COPIES (SKR_MAX_FRAMES_IN_FLIGHT + 1)

// Future type for tracking command buffer completion (must be before skr_surface_t)
typedef struct skr_future_t {
	void*    slot;          // Pointer to _skr_cmd_ring_slot_t
	uint64_t generation;    // Generation counter to detect fence reuse (must match slot's generation)
} skr_future_t;

// Texture readback handle for async GPU->CPU texture data transfer
typedef struct skr_tex_readback_t {
	void*        data;      // CPU-accessible data pointer (valid after future completes)
	uint32_t     size;      // Data size in bytes
	skr_future_t future;    // Poll with skr_future_check(); see future notes in sk_renderer.h
	void*        _internal; // Internal state (staging buffer/memory) - do not access directly
} skr_tex_readback_t;

// Async GPU->CPU snapshot of a storage-type buffer (the only type both
// backends can copy out of). `data` is memory the readback owns, unaffected
// by later writes to the buffer. The future covers work recorded on the
// calling thread so far: dispatch first, then read back, never inside an open
// pass. A skr_buffer_set between the dispatch and the readback snapshots the
// newly set contents instead of the dispatch results. Creation may submit the
// thread's pending commands (WebGPU always does; mapAsync must follow submit).
// Create, poll, and destroy on one thread; destroying mid-flight is safe and
// never blocks.
typedef struct skr_buffer_readback_t {
	void*        data;      // CPU-accessible data pointer (valid after future completes)
	uint32_t     size;      // Data size in bytes
	skr_future_t future;    // Poll with skr_future_check(); see future notes in sk_renderer.h
	void*        _internal; // Internal state (staging buffer/memory) - do not access directly
} skr_buffer_readback_t;

typedef struct skr_buffer_t {
	VkBuffer            buffer;  // Current buffer for binding (= _ring[_ring_index] if ring active)
	VkDeviceMemory      memory;  // Current memory
	void*               mapped;  // Current mapped pointer (for dynamic buffers)
	uint32_t            size;
	skr_buffer_type_    type;
	skr_use_            use;

	// Ring buffer for safe dynamic updates (lazy allocated slots)
	// Allows updates without stomping data still in use by in-flight frames
	struct {
		VkBuffer       buffer;
		VkDeviceMemory memory;
		void*          mapped;
	}                   _ring[SKR_DYNAMIC_BUFFER_COPIES];
	uint8_t             _ring_count;  // Slots allocated so far (0 = no ring, use top-level fields)
	uint8_t             _ring_index;  // Current active slot for reading
} skr_buffer_t;

typedef struct skr_vert_type_t {
	VkVertexInputAttributeDescription* attributes;
	VkVertexInputBindingDescription*   bindings;         // Array of bindings (one per vertex buffer)
	uint32_t                           binding_count;    // Number of bindings
	skr_vert_component_t*              components;
	uint32_t                           component_count;
	int32_t                            pipeline_idx; // Cached pipeline vertex format index
} skr_vert_type_t;

#define SKR_MAX_VERTEX_BUFFERS 2

typedef struct skr_mesh_t {
	skr_buffer_t           vertex_buffers[SKR_MAX_VERTEX_BUFFERS];
	uint32_t               vertex_buffer_count;   // Number of vertex buffers in use
	uint32_t               vertex_buffer_owned;   // Bitmask: which buffers are owned (vs externally referenced)
	skr_buffer_t           index_buffer;
	const skr_vert_type_t* vert_type;
	skr_index_fmt_         ind_format;
	VkIndexType            ind_format_vk;
	uint32_t               vert_count;
	uint32_t               ind_count;
} skr_mesh_t;

typedef struct skr_tex_t {
	VkImage                image;
	VkDeviceMemory         memory;
	VkImageView            view;
	VkFramebuffer          framebuffer;             // Cached framebuffer (color only, no depth)
	VkFramebuffer          framebuffer_depth;       // Cached framebuffer (color + depth, if last used with depth)
	VkRenderPass           framebuffer_pass;        // Render pass the color-only framebuffer was created for
	VkRenderPass           framebuffer_depth_pass;  // Render pass the depth framebuffer was created for
	uint64_t               framebuffer_views;       // Fingerprint of the views the cached framebuffer was built
	uint64_t               framebuffer_depth_views; // from — other attachments can be destroyed while the cache
	                                                // target survives, so the render pass alone under-keys the cache
	VkSampler              sampler;          // Vulkan sampler handle
	skr_tex_sampler_t      sampler_settings; // Sampler settings
	skr_vec3i_t            size;
	skr_tex_fmt_           format;
	skr_tex_flags_         flags;
	VkSampleCountFlagBits  samples;          // Sample count for MSAA (VK_SAMPLE_COUNT_1_BIT, 2, 4, 8, etc.)
	uint32_t               mip_levels;       // Number of mip levels
	uint32_t               layer_count;      // Number of array layers (1 for regular, N for arrays, 6 for cubemaps)
	VkImageAspectFlags     aspect_mask;      // Depth bit for depth textures, color bit for color textures
	VkImageUsageFlags      usage;            // Adopted images carry only the bits their flags promise

	// Automatic layout transition tracking. current_layout is the actual GPU
	// layout right now — used as oldLayout for the next barrier and to skip
	// no-op transitions. Not consulted for descriptor writes (those derive
	// from _skr_tex_sample_layout) or renderpass attachment layouts.
	VkImageLayout          current_layout;       // Current image layout (tracked automatically)
	uint32_t               current_queue_family; // Current queue family owner
	bool                   first_use;            // True until first transition (allows UNDEFINED optimization)
	bool                   is_transient_discard; // True for non-readable depth/MSAA (always use UNDEFINED)
	bool                   is_external;          // True if image/memory are externally owned (don't destroy)

	// YCbCr conversion (Vulkan 1.1) for opaque YUV textures (e.g. AHB video frames)
	VkSamplerYcbcrConversion ycbcr_conversion;   // VK_NULL_HANDLE if unused
	VkSampler                ycbcr_sampler;       // Immutable sampler with YCbCr conversion baked in (VK_NULL_HANDLE if unused)

#ifdef __ANDROID__
	void*                  ahb_handle;           // AHardwareBuffer* if imported via AHB (NULL otherwise)
	bool                   owns_ahb;             // If true, release AHB on destroy
#endif
} skr_tex_t;

// External texture creation info (for wrapping VkImages from external sources like FFmpeg)
typedef struct skr_tex_external_info_t {
	VkImage           image;          // External VkImage (not owned unless owns_image=true)
	VkImageView       view;           // Optional - will create if VK_NULL_HANDLE
	VkDeviceMemory    memory;         // Optional - VK_NULL_HANDLE for external memory
	skr_tex_fmt_      format;         // Texture format
	skr_tex_flags_    flags;          // Usage flags (readable/writeable/etc.) - 0 = infer from format
	skr_vec3i_t       size;           // Dimensions (for array textures, z = layer count)
	VkImageLayout     current_layout; // Layout the image is in *right now* — used as
	                                  // the oldLayout of sk_renderer's next transition.
	                                  // sk_renderer will move the image to its canonical
	                                  // sample layout before sampling (SHADER_READ_ONLY,
	                                  // DEPTH_STENCIL_READ_ONLY, or GENERAL based on flags).
	skr_tex_sampler_t sampler;        // Sampler settings
	int32_t           multisample;    // MSAA sample count (1, 2, 4, 8, etc.), 0 or 1 = no MSAA
	int32_t           array_layers;   // Array layer count (0 or 1 = single texture, >1 = array texture)
	bool              owns_image;     // If true, sk_renderer destroys image on tex_destroy
} skr_tex_external_info_t;

// External texture update info (for video frame cycling)
typedef struct skr_tex_external_update_t {
	VkImage       image;          // New VkImage to reference
	VkImageView   view;           // Optional new view (VK_NULL_HANDLE = recreate from image)
	VkImageLayout current_layout; // Layout the new image is in right now — see
	                              // skr_tex_external_info_t::current_layout for details.
} skr_tex_external_update_t;

// GL external texture import via external memory (FD on Linux/Android, Win32 HANDLE on Windows)
typedef struct skr_tex_external_gl_info_t {
	int32_t           fd;              // FD from glExportMemoryFdEXT (-1 if using handle)
	void*             handle;          // Win32 HANDLE from glExportMemoryWin32HandleNV (NULL if using fd)
	skr_tex_fmt_      format;          // Texture format
	skr_vec3i_t       size;            // Dimensions (z=1 for 2D, z=depth for 3D/array)
	uint64_t          allocation_size; // Total memory allocation size in bytes
	uint64_t          memory_offset;   // Binding offset within the allocation (0 = start)
	uint32_t          mip_levels;      // Number of mip levels (1 = no mipmaps)
	uint32_t          array_layers;    // Array layer count (1 = non-array, 6 = cubemap)
	skr_tex_flags_    flags;           // Texture flags (array, cubemap, 3d, etc.)
	skr_tex_sampler_t sampler;         // Sampler settings
	bool              dedicated;       // Use dedicated allocation (VK_KHR_dedicated_allocation)
	bool              linear_tiling;   // Use LINEAR tiling (for cross-device memory sharing)
} skr_tex_external_gl_info_t;

// DMA-BUF external texture import via VK_EXT_external_memory_dma_buf
typedef struct skr_tex_external_dma_info_t {
	int32_t           fd;              // DMA-BUF file descriptor (consumed on success - caller must not close)
	uint64_t          drm_modifier;    // DRM format modifier (DRM_FORMAT_MOD_LINEAR=0, or driver-specific)
	skr_tex_fmt_      format;          // Texture format
	skr_vec3i_t       size;            // Dimensions (z=1 for 2D)
	uint64_t          allocation_size; // Total memory allocation size in bytes
	uint64_t          offset;          // Offset of image data within the DMA-BUF (typically 0)
	uint32_t          row_pitch;       // Row pitch in bytes (required for modifier layout)
	skr_tex_sampler_t sampler;         // Sampler settings
} skr_tex_external_dma_info_t;

// Android Hardware Buffer external texture import
typedef struct skr_tex_external_ahb_info_t {
	void*             hardware_buffer; // AHardwareBuffer* (void* for C compatibility)
	skr_tex_fmt_      format;          // Texture format (skr_tex_fmt_none = auto-detect from AHB)
	skr_tex_sampler_t sampler;         // Sampler settings
	bool              owns_buffer;     // If true, sk_renderer releases AHB on destroy
} skr_tex_external_ahb_info_t;

typedef struct skr_surface_t {
	VkSurfaceKHR   surface;
	VkSwapchainKHR swapchain;
	VkFence        present_fence[SKR_MAX_FRAMES_IN_FLIGHT]; // Signals when that slot's present retires; all null without maintenance1
	skr_tex_t*     images;
	uint32_t       image_count;
	uint32_t       current_image;
	uint32_t       frame_idx;
	skr_future_t   frame_future     [SKR_MAX_FRAMES_IN_FLIGHT];  // Track command submission for each frame-in-flight
	VkSemaphore    semaphore_acquire[SKR_MAX_FRAMES_IN_FLIGHT];
	VkSemaphore*   semaphore_submit;
	skr_vec2i_t    size;
} skr_surface_t;

typedef struct skr_shader_stage_t {
	VkShaderModule shader;
	skr_stage_     type;
} skr_shader_stage_t;

typedef struct skr_shader_t {
	sksc_shader_meta_t  meta;
	sksc_pass_inputs_t  pass_inputs; // "color"/"depth" convention, resolved at creation
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
	uint32_t   buffer_offset; // Offset within buffer (for bump-allocated buffers)
	uint32_t   buffer_range;  // Range to bind (0 = use buffer->size)
	// The shader samples this texture with QCOM image-processing ops
	// (BoxFilterQCOM etc., meta shape bit 6) — descriptor writes must bind
	// _skr_vk.sampler_image_proc instead of the texture's own sampler.
	bool       image_proc_sampler;
} skr_material_bind_t;

// Internal key struct for pipeline-affecting material parameters only.
// Excludes queue_offset which affects render list sorting but not pipeline state.
//
// Layout is hand-tuned so every byte corresponds to a named field — there are
// no implicit alignment holes. _pad0/_pad1 are explicit so designated
// initializers zero them via the C99 "unspecified members → zero" rule, and
// memcmp-based dedup in _skr_pipeline_register_material is byte-deterministic
// regardless of compiler or C standard version.
#define SKR_MAX_IMMUTABLE_SAMPLERS 2
#define SKR_MAX_SPEC_CONSTANTS     4
typedef struct {
	// 8-byte aligned block
	const skr_shader_t*  shader;                                              // @0   (8)
	VkSampler            immutable_samplers[SKR_MAX_IMMUTABLE_SAMPLERS];      // @8   (16) Immutable samplers for YCbCr textures (VK_NULL_HANDLE = unused)

	// 4-byte aligned sub-structs (no internal padding: all 4-byte fields)
	skr_blend_state_t    blend_state;                                         // @24  (24)
	skr_stencil_state_t  stencil_front;                                       // @48  (28)
	skr_stencil_state_t  stencil_back;                                        // @76  (28)

	// 4-byte aligned scalars
	skr_cull_            cull;                                                // @104 (4)
	skr_write_           write_mask;                                          // @108 (4)
	skr_compare_         depth_test;                                          // @112 (4)
	int32_t              immutable_sampler_count;                             // @116 (4)  Number of active immutable samplers
	int32_t              immutable_sampler_slots[SKR_MAX_IMMUTABLE_SAMPLERS]; // @120 (8)  Descriptor binding slots (sorted by slot for deterministic memcmp)
	uint32_t             spec_constant_values[SKR_MAX_SPEC_CONSTANTS];        // @128 (16) Bit patterns for the shader's spec constants, in shader meta order (defaults where not overridden)

	// 1-byte fields packed at the end with explicit padding to fill the
	// alignment tail. _pad0/_pad1 must remain zero — initializer rules above
	// keep this true without any extra code at the call sites.
	bool                 alpha_to_coverage;                                   // @144 (1)
	bool                 depth_clamp;                                         // @145 (1)
	bool                 wireframe;                                           // @146 (1)
	uint8_t              _pad0;                                               // @147 (1)  must be 0
	uint32_t             _pad1;                                               // @148 (4)  must be 0
} _skr_pipeline_material_key_t;

#ifdef __cplusplus
static_assert(sizeof(_skr_pipeline_material_key_t) == 152,
	"_skr_pipeline_material_key_t layout drifted; explicit padding and memcmp dedup may be broken");
#else
_Static_assert(sizeof(_skr_pipeline_material_key_t) == 152,
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

	bool                   has_system_buffer;
	uint32_t               instance_buffer_stride; // Element size of instance buffer (0 = no instance buffer)
} skr_material_t;

// Pooled GPU buffer with future for tracking completion
typedef struct skr_param_buffer_slot_t {
	skr_buffer_t buffer;
	skr_future_t future;
	uint64_t     hash;  // Content hash for reuse matching
} skr_param_buffer_slot_t;

typedef struct skr_compute_t {
	const skr_shader_t*    shader;  // Reference to shader (not owned)
	VkPipelineLayout       layout;
	VkDescriptorSetLayout  descriptor_layout;
	VkPipeline             pipeline;

	skr_material_bind_t*   binds;
	uint32_t               bind_count;

	// CPU-side parameter staging
	void*                  param_buffer;
	uint32_t               param_buffer_size;
	bool                   param_dirty;
} skr_compute_t;

// Flags for skr_render_item_t::flags (vertex_buffer_count stored in bits 2-3)
typedef enum skr_item_flag_ {
	skr_item_flag_system_buffer  = 1 << 0, // Material has a system buffer binding
	skr_item_flag_index_32bit    = 1 << 1, // Index format is uint32 (vs uint16)
	skr_item_flag_vb_count_shift = 2,      // Vertex buffer count (0-2) in bits 2-3
	skr_item_flag_vb_count_mask  = 3 << 2,
	skr_item_flag_instance_buffer= 1 << 4, // Shader declares an instance structured buffer
} skr_item_flag_;

// Render item with inlined mesh/material data - mesh/material can be destroyed after add.
// Fields are packed by size to minimize padding (80 bytes).
typedef struct skr_render_item_t {
	// 8-byte aligned (VkBuffer = pointer = 8 bytes)
	VkBuffer    vertex_buffers[SKR_MAX_VERTEX_BUFFERS]; // From mesh->vertex_buffers[].buffer
	VkBuffer    index_buffer;                           // From mesh->index_buffer.buffer
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

typedef struct skr_render_list_t {
	skr_render_item_t* items;
	skr_render_item_t* items_tmp;               // Parallel buffer for sort permute (swapped with items)
	uint32_t           count;
	uint32_t           capacity;                // Both items and items_tmp have this capacity
	uint8_t*           instance_data;
	uint32_t           instance_data_used;
	uint32_t           instance_data_capacity;
	uint8_t*           instance_data_sorted;          // Reordered instance data after sort
	uint32_t           instance_data_sorted_capacity;
	uint8_t*           material_data;
	uint32_t           material_data_used;
	uint32_t           material_data_capacity;
	bool               needs_sort;              // Dirty flag for sorting

	// Radix sort scratch (persistent, grown as needed)
	void*              sort_scratch_a;           // Sort pair buffer A
	void*              sort_scratch_b;           // Sort pair buffer B (ping-pong)
	uint32_t           sort_scratch_capacity;    // Capacity in number of pair elements
} skr_render_list_t;
