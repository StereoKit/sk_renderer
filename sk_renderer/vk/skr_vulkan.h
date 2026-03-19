// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include <volk.h>

#define SKR_MAX_FRAMES_IN_FLIGHT 3
#define SKR_MAX_SURFACES 2  // Maximum surfaces for VR stereo rendering

// Future type for tracking command buffer completion (must be before skr_surface_t)
typedef struct skr_future_t {
	void*    slot;          // Pointer to _skr_cmd_ring_slot_t
	uint64_t generation;    // Generation counter to detect fence reuse (must match slot's generation)
} skr_future_t;

// Texture readback handle for async GPU->CPU texture data transfer
typedef struct skr_tex_readback_t {
	void*        data;      // CPU-accessible data pointer (valid after future completes)
	uint32_t     size;      // Data size in bytes
	skr_future_t future;    // Poll with skr_future_check(), block with skr_future_wait()
	void*        _internal; // Internal state (staging buffer/memory) - do not access directly
} skr_tex_readback_t;

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
	}                   _ring[SKR_MAX_FRAMES_IN_FLIGHT];
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
	VkFramebuffer          framebuffer;            // Cached framebuffer (color only, no depth)
	VkFramebuffer          framebuffer_depth;      // Cached framebuffer (color + depth, if last used with depth)
	VkRenderPass           framebuffer_pass;       // Render pass the framebuffer was created for
	VkSampler              sampler;          // Vulkan sampler handle
	skr_tex_sampler_t      sampler_settings; // Sampler settings
	skr_vec3i_t            size;
	skr_tex_fmt_           format;
	skr_tex_flags_         flags;
	VkSampleCountFlagBits  samples;          // Sample count for MSAA (VK_SAMPLE_COUNT_1_BIT, 2, 4, 8, etc.)
	uint32_t               mip_levels;       // Number of mip levels
	uint32_t               layer_count;      // Number of array layers (1 for regular, N for arrays, 6 for cubemaps)
	VkImageAspectFlags     aspect_mask;      // Depth bit for depth textures, color bit for color textures

	// Automatic layout transition tracking
	VkImageLayout          current_layout;       // Current image layout (tracked automatically)
	uint32_t               current_queue_family; // Current queue family owner
	bool                   first_use;            // True until first transition (allows UNDEFINED optimization)
	bool                   is_transient_discard; // True for MSAA attachments (always use UNDEFINED)
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
	skr_vec3i_t       size;           // Dimensions (for array textures, z = layer count)
	VkImageLayout     current_layout; // Current layout of the image
	skr_tex_sampler_t sampler;        // Sampler settings
	int32_t           multisample;    // MSAA sample count (1, 2, 4, 8, etc.), 0 or 1 = no MSAA
	int32_t           array_layers;   // Array layer count (0 or 1 = single texture, >1 = array texture)
	bool              owns_image;     // If true, sk_renderer destroys image on tex_destroy
} skr_tex_external_info_t;

// External texture update info (for video frame cycling)
typedef struct skr_tex_external_update_t {
	VkImage       image;          // New VkImage to reference
	VkImageView   view;           // Optional new view (VK_NULL_HANDLE = recreate from image)
	VkImageLayout current_layout; // Current layout of new image
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
	sksc_shader_meta_t*  meta;
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
} skr_material_bind_t;

// Internal key struct for pipeline-affecting material parameters only.
// Excludes queue_offset which affects render list sorting but not pipeline state.
typedef struct {
	const skr_shader_t*  shader;
	skr_cull_            cull;
	skr_write_           write_mask;
	skr_compare_         depth_test;
	skr_blend_state_t    blend_state;
	bool                 alpha_to_coverage;
	bool                 depth_clamp;
	bool                 wireframe;
	skr_stencil_state_t  stencil_front;
	skr_stencil_state_t  stencil_back;
#define SKR_MAX_IMMUTABLE_SAMPLERS 2
	VkSampler            immutable_samplers[SKR_MAX_IMMUTABLE_SAMPLERS];      // Immutable samplers for YCbCr textures (VK_NULL_HANDLE = unused)
	int32_t              immutable_sampler_slots[SKR_MAX_IMMUTABLE_SAMPLERS];  // Descriptor binding slots (sorted by slot for deterministic memcmp)
	int32_t              immutable_sampler_count;                              // Number of active immutable samplers
} _skr_pipeline_material_key_t;

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

// Render item with inlined mesh/material data - mesh/material can be destroyed after add.
// Fields are packed by size to minimize padding (~80 bytes vs ~104 bytes unpacked).
typedef struct skr_render_item_t {
	// 8-byte aligned (VkBuffer = pointer = 8 bytes)
	VkBuffer    vertex_buffers[SKR_MAX_VERTEX_BUFFERS]; // From mesh->vertex_buffers[].buffer
	VkBuffer    index_buffer;                           // From mesh->index_buffer.buffer
	uint64_t    sort_key;                               // Pre-computed sort key for fast sorting

	// 4-byte aligned
	uint32_t    vert_count;           // From mesh->vert_count
	uint32_t    ind_count;            // From mesh->ind_count
	uint32_t    param_data_offset;    // Offset into render_list->material_data (bytes)
	uint32_t    instance_offset;      // Offset into render_list->instance_data (bytes)
	uint32_t    instance_count;       // Number of instances to draw
	int32_t     first_index;          // Index buffer offset (0 = use mesh defaults)
	int32_t     index_count;          // Number of indices (0 = use mesh ind_count)
	int32_t     vertex_offset;        // Base vertex offset
	int32_t     bind_start;           // Index into bind pool (bind pool uses deferred destruction)

	// 2-byte aligned (max 65535 is plenty for these)
	uint16_t    pipeline_vert_idx;      // From mesh->vert_type->pipeline_idx
	uint16_t    pipeline_material_idx;  // From material->pipeline_material_idx
	uint16_t    param_buffer_size;      // From material->param_buffer_size
	uint16_t    instance_buffer_stride; // From material->instance_buffer_stride
	uint16_t    instance_data_size;     // Size per instance (bytes)

	// 1-byte aligned (small values)
	uint8_t     vertex_buffer_count;  // From mesh->vertex_buffer_count (max SKR_MAX_VERTEX_BUFFERS=2)
	uint8_t     bind_count;           // From material->bind_count (textures+buffers, rarely >32)
	uint8_t     index_format;         // From mesh->ind_format_vk (VkIndexType: 0=uint16, 1=uint32)
	uint8_t     has_system_buffer;    // From material->has_system_buffer (bool)
} skr_render_item_t;

typedef struct skr_render_list_t {
	skr_render_item_t* items;
	uint32_t           count;
	uint32_t           capacity;
	uint8_t*           instance_data;
	uint32_t           instance_data_used;
	uint32_t           instance_data_capacity;
	uint8_t*           instance_data_sorted;          // Reordered instance data after sort
	uint32_t           instance_data_sorted_capacity;
	uint8_t*           material_data;
	uint32_t           material_data_used;
	uint32_t           material_data_capacity;
	bool               needs_sort;  // Dirty flag for sorting
} skr_render_list_t;
