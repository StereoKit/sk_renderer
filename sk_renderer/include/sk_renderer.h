// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "sksc_file.h"

// Symbol visibility macros
#if defined(_WIN32) || defined(_WIN64)
	#ifdef SKR_BUILD_SHARED
		#define SKR_API __declspec(dllexport)
	#else
		#define SKR_API
	#endif
#elif defined(__GNUC__) || defined(__clang__)
	#define SKR_API __attribute__((visibility("default")))
#else
	#define SKR_API
#endif

///////////////////////////////////////////////////////////////////////////////

typedef struct skr_vec2_t {
	float x;
	float y;
} skr_vec2_t;

typedef struct skr_vec3_t {
	float x;
	float y;
	float z;
} skr_vec3_t;

typedef struct skr_vec4_t {
	float x;
	float y;
	float z;
	float w;
} skr_vec4_t;

typedef struct skr_vec2i_t {
	int32_t x;
	int32_t y;
} skr_vec2i_t;

typedef struct skr_vec3i_t {
	int32_t x;
	int32_t y;
	int32_t z;
} skr_vec3i_t;

typedef struct skr_vec4i_t {
	int32_t x;
	int32_t y;
	int32_t z;
	int32_t w;
} skr_vec4i_t;

typedef struct skr_quat_t {
	float x;
	float y;
	float z;
	float w;
} skr_quat_t;

typedef struct skr_matrix {
    // row-major
	float m[16];
} skr_matrix_t;

typedef struct skr_rect_t {
	float x;
	float y;
	float w;
	float h;
} skr_rect_t;

typedef struct skr_recti_t {
	int32_t x;
	int32_t y;
	int32_t w;
	int32_t h;
} skr_recti_t;

///////////////////////////////////////////////////////////////////////////////

typedef enum skr_err_ {
	// Success code (positive)
	skr_err_success = 1,

	// General failure (zero)
	skr_err_failure = 0,

	// Specific errors (negative)
	skr_err_out_of_memory     = -1,  // malloc/calloc/realloc failure
	skr_err_invalid_parameter = -2,  // NULL pointer, zero size, invalid arguments
	skr_err_unsupported       = -3,  // Unsupported format, feature, or operation
	skr_err_device_error      = -4,  // GPU/Vulkan error
} skr_err_;

///////////////////////////////////////////////////////////////////////////////

typedef enum skr_buffer_type_ {
	skr_buffer_type_vertex   = 1 << 0,  // Vertex buffer (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
	skr_buffer_type_index    = 1 << 1,  // Index buffer (VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
	skr_buffer_type_constant = 1 << 2,  // Constant/uniform buffer (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
	skr_buffer_type_storage  = 1 << 3,  // Storage buffer (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) - compute, instance data, etc.
} skr_buffer_type_;

typedef enum skr_use_ {
	skr_use_static        = 1 << 1,
	skr_use_dynamic       = 1 << 2,
	skr_use_compute_read  = 1 << 3,
	skr_use_compute_write = 1 << 4,
	skr_use_compute_readwrite = skr_use_compute_read | skr_use_compute_write
} skr_use_;

typedef enum skr_tex_fmt_ {
	skr_tex_fmt_none = 0,
	skr_tex_fmt_rgba32_srgb = 1,
	skr_tex_fmt_rgba32_linear = 2,
	skr_tex_fmt_bgra32_srgb = 3,
	skr_tex_fmt_bgra32_linear = 4,
	skr_tex_fmt_rg11b10 = 5,
	skr_tex_fmt_rgb10a2 = 6,
	skr_tex_fmt_rgba64u = 7,
	skr_tex_fmt_rgba64s = 8,
	skr_tex_fmt_rgba64f = 9,
	skr_tex_fmt_rgba128 = 10,
	skr_tex_fmt_r8 = 11,
	skr_tex_fmt_r16u = 12,
	skr_tex_fmt_r16s = 13,
	skr_tex_fmt_r16f = 14,
	skr_tex_fmt_r32f = 15,
	skr_tex_fmt_depth24s8 = 16,
	skr_tex_fmt_depth32 = 17,
	skr_tex_fmt_depth16 = 18,
	skr_tex_fmt_r8g8 = 19,
	skr_tex_fmt_rgb9e5 = 20,

	skr_tex_fmt_depth32s8,
	skr_tex_fmt_depth16s8,

	skr_tex_fmt_bc1_rgb_srgb,
	skr_tex_fmt_bc1_rgb,
	skr_tex_fmt_bc3_rgba_srgb,
	skr_tex_fmt_bc3_rgba,
	skr_tex_fmt_bc4_r,
	skr_tex_fmt_bc5_rg,
	skr_tex_fmt_bc7_rgba_srgb,
	skr_tex_fmt_bc7_rgba,

	skr_tex_fmt_etc1_rgb,
	skr_tex_fmt_etc2_rgba_srgb,
	skr_tex_fmt_etc2_rgba,
	skr_tex_fmt_etc2_r11,
	skr_tex_fmt_etc2_rg11,
	skr_tex_fmt_pvrtc1_rgb_srgb,
	skr_tex_fmt_pvrtc1_rgb,
	skr_tex_fmt_pvrtc1_rgba_srgb,
	skr_tex_fmt_pvrtc1_rgba,
	skr_tex_fmt_pvrtc2_rgba_srgb,
	skr_tex_fmt_pvrtc2_rgba,
	skr_tex_fmt_astc4x4_rgba_srgb,
	skr_tex_fmt_astc4x4_rgba,
	skr_tex_fmt_atc_rgb,
	skr_tex_fmt_atc_rgba,
} skr_tex_fmt_;

typedef enum skr_tex_flags_ {
	skr_tex_flags_readable  = 1 << 0,
	skr_tex_flags_writeable = 1 << 1,
	skr_tex_flags_dynamic   = 1 << 2,
	skr_tex_flags_gen_mips  = 1 << 3,
	skr_tex_flags_array     = 1 << 4,
	skr_tex_flags_3d        = 1 << 5,
	skr_tex_flags_in_tile_msaa = 1 << 6,
	skr_tex_flags_compute   = 1 << 7,  // For compute shader RWTexture (storage image)
	skr_tex_flags_cubemap   = 1 << 8,  // Cubemap texture (requires 6 array layers)
} skr_tex_flags_;

typedef enum skr_tex_sample_ {
	skr_tex_sample_linear = 0,
	skr_tex_sample_point,
	skr_tex_sample_anisotropic
} skr_tex_sample_;

typedef enum skr_tex_address_ {
	skr_tex_address_wrap = 0,
	skr_tex_address_clamp,
	skr_tex_address_mirror,
} skr_tex_address_;

typedef enum skr_mip_filter_ {
	skr_mip_filter_default = 0,     // vkCmdBlitImage with linear filtering
	skr_mip_filter_alpha_weighted,  // Compute shader with alpha-weighted averaging
	skr_mip_filter_sdf,             // Compute shader optimized for SDF data
} skr_mip_filter_;

typedef enum skr_compare_ {
	skr_compare_none = 0,
	skr_compare_less,
	skr_compare_less_or_eq,
	skr_compare_greater,
	skr_compare_greater_or_eq,
	skr_compare_equal,
	skr_compare_not_equal,
	skr_compare_always,
	skr_compare_never,
} skr_compare_;

typedef enum skr_blend_factor_ {
	skr_blend_zero = 0,
	skr_blend_one,
	skr_blend_src_color,
	skr_blend_one_minus_src_color,
	skr_blend_dst_color,
	skr_blend_one_minus_dst_color,
	skr_blend_src_alpha,
	skr_blend_one_minus_src_alpha,
	skr_blend_dst_alpha,
	skr_blend_one_minus_dst_alpha,
	skr_blend_constant_color,
	skr_blend_one_minus_constant_color,
	skr_blend_constant_alpha,
	skr_blend_one_minus_constant_alpha,
	skr_blend_src_alpha_saturate,
	skr_blend_src1_color,
	skr_blend_one_minus_src1_color,
	skr_blend_src1_alpha,
	skr_blend_one_minus_src1_alpha,
} skr_blend_factor_;

typedef enum skr_blend_op_ {
	skr_blend_op_add = 0,
	skr_blend_op_subtract,
	skr_blend_op_reverse_subtract,
	skr_blend_op_min,
	skr_blend_op_max,
} skr_blend_op_;

typedef enum skr_stencil_op_ {
	skr_stencil_op_keep = 0,
	skr_stencil_op_zero,
	skr_stencil_op_replace,
	skr_stencil_op_increment_clamp,
	skr_stencil_op_decrement_clamp,
	skr_stencil_op_invert,
	skr_stencil_op_increment_wrap,
	skr_stencil_op_decrement_wrap,
} skr_stencil_op_;

typedef enum skr_write_ {
	skr_write_r       = 1 << 0,
	skr_write_g       = 1 << 1,
	skr_write_b       = 1 << 2,
	skr_write_a       = 1 << 3,
	skr_write_depth   = 1 << 4,
	skr_write_stencil = 1 << 5,
	skr_write_none    = 1 << 6,
	skr_write_rgba    = skr_write_r | skr_write_g | skr_write_b | skr_write_a,
	skr_write_default = skr_write_r | skr_write_g | skr_write_b | skr_write_a | skr_write_depth
} skr_write_;

typedef enum skr_cull_ {
	skr_cull_back = 0,
	skr_cull_front,
	skr_cull_none,
} skr_cull_;

typedef enum skr_index_fmt_ {
	skr_index_fmt_u8,
	skr_index_fmt_u16,
	skr_index_fmt_u32,
} skr_index_fmt_;

typedef enum skr_log_ {
	skr_log_info,
	skr_log_warning,
	skr_log_critical,
} skr_log_;

typedef enum skr_clear_ {
	skr_clear_none    = 0,
	skr_clear_color   = 1 << 0,
	skr_clear_depth   = 1 << 1,
	skr_clear_stencil = 1 << 2,
	skr_clear_all     = skr_clear_color | skr_clear_depth | skr_clear_stencil,
} skr_clear_;

typedef enum skr_acquire_ {
	skr_acquire_success      = 1,   // Successfully acquired image
	skr_acquire_not_ready    = 0,   // Swapchain minimized/not ready (skip frame)
	skr_acquire_needs_resize = -1,  // Swapchain out of date, needs resize
	skr_acquire_surface_lost = -2,  // Surface lost, needs recreation
	skr_acquire_error        = -3,  // General error
} skr_acquire_;

typedef struct skr_tex_sampler_t {
	skr_tex_sample_      sample;
	skr_tex_address_     address;
	skr_compare_         sample_compare;
	int32_t              anisotropy;
} skr_tex_sampler_t;

// Texture data descriptor for uploading texture data with multiple mips/layers.
// Data layout is mip-major: all layers for mip0, then all layers for mip1, etc.
// This matches KTX2 file layout.
typedef struct skr_tex_data_t {
	const void* data;        // Contiguous data pointer
	uint32_t    mip_count;   // Exact number of mips present in data
	uint32_t    layer_count; // Exact number of layers present in data (1 for 3D textures)
	uint32_t    base_mip;    // Target mip level offset in texture
	uint32_t    base_layer;  // Target layer offset in texture
	int32_t     row_pitch;   // Source row pitch in bytes (0 = tightly packed, only valid when mip_count == 1)
} skr_tex_data_t;

typedef struct skr_blend_state_t {
	skr_blend_factor_ src_color_factor;
	skr_blend_factor_ dst_color_factor;
	skr_blend_op_     color_op;
	skr_blend_factor_ src_alpha_factor;
	skr_blend_factor_ dst_alpha_factor;
	skr_blend_op_     alpha_op;
} skr_blend_state_t;

// Common blend state presets
static const skr_blend_state_t skr_blend_off = {
	.color_op         = skr_blend_op_add,
	.alpha_op         = skr_blend_op_add,
};

static const skr_blend_state_t skr_blend_alpha = {
	.src_color_factor = skr_blend_src_alpha,
	.dst_color_factor = skr_blend_one_minus_src_alpha,
	.color_op         = skr_blend_op_add,
	.src_alpha_factor = skr_blend_one,
	.dst_alpha_factor = skr_blend_one_minus_src_alpha,
	.alpha_op         = skr_blend_op_add,
};

static const skr_blend_state_t skr_blend_additive = {
	.src_color_factor = skr_blend_src_alpha,
	.dst_color_factor = skr_blend_one,
	.color_op         = skr_blend_op_add,
	.src_alpha_factor = skr_blend_one,
	.dst_alpha_factor = skr_blend_one,
	.alpha_op         = skr_blend_op_add,
};

static const skr_blend_state_t skr_blend_premultiplied = {
	.src_color_factor = skr_blend_one,
	.dst_color_factor = skr_blend_one_minus_src_alpha,
	.color_op         = skr_blend_op_add,
	.src_alpha_factor = skr_blend_one,
	.dst_alpha_factor = skr_blend_one_minus_src_alpha,
	.alpha_op         = skr_blend_op_add,
};

typedef struct skr_stencil_state_t {
	skr_compare_    compare;
	skr_stencil_op_ fail_op;
	skr_stencil_op_ pass_op;
	skr_stencil_op_ depth_fail_op;
	uint32_t        compare_mask;
	uint32_t        write_mask;
	uint32_t        reference;
} skr_stencil_state_t;

// GPU selection flags - can be combined with bitwise OR
typedef enum skr_gpu_ {
	skr_gpu_none       = 0,
	skr_gpu_discrete   = 1 << 0,  // Discrete/dedicated GPU (not integrated)
	skr_gpu_integrated = 1 << 1,  // Integrated GPU (typically lower power)
	skr_gpu_video      = 1 << 2,  // GPU with hardware video decode support
} skr_gpu_;

// Callback info returned from device_init_callback
// Allows external systems (e.g., OpenXR) to specify device requirements
// after VkInstance is created but before VkDevice is created.
typedef struct skr_device_request_t {
	void*        physical_device;               // VkPhysicalDevice to use (NULL = auto-select)
	const char** required_device_extensions;    // Device extensions to enable
	uint32_t     required_device_extension_count;
} skr_device_request_t;

// Callback type for device initialization
// Called after VkInstance creation, before VkDevice creation.
// vk_instance: The created VkInstance (for querying physical devices, etc.)
// user_data: User-provided context pointer
// Returns: Device requirements (physical device, extensions)
typedef skr_device_request_t (*skr_device_init_callback_t)(void* vk_instance, void* user_data);

// Bind slot configuration for shader/renderer coordination.
// These values must match between skshaderc and sk_renderer.
// Default values (if all zeros): material=0, system=1, instance=2
typedef struct skr_bind_settings_t {
	int32_t material_slot;   // Slot for material cbuffer (default: 0)
	int32_t system_slot;     // Slot for system buffer (default: 1)
	int32_t instance_slot;   // Slot for instance buffer (default: 2)
} skr_bind_settings_t;

typedef struct skr_settings_t {
	const char*  app_name;
	int32_t      app_version;
	bool         enable_validation;

	// GPU Selection:
	// - physical_device: If non-NULL, use this device directly (e.g., from OpenXR)
	// - gpu_require: Fail initialization if no GPU has these flags
	// - gpu_prefer: Prefer GPU with these flags, fall back if not found
	// If no flags set, selects most powerful GPU (discrete preferred).
	void*        physical_device;     // Backend-specific (VkPhysicalDevice for Vulkan)
	skr_gpu_     gpu_require;         // Required GPU features (fail if not found)
	skr_gpu_     gpu_prefer;          // Preferred GPU features (fallback if not found)

	const char** required_extensions;
	uint32_t     required_extension_count;

	// Device initialization callback (optional, for OpenXR integration etc.)
	// If provided, called after VkInstance creation to get device requirements.
	// The callback can query external systems for required physical device and extensions.
	skr_device_init_callback_t device_init_callback;
	void*                      device_init_user_data;

	void*      (*malloc_func) (size_t size);
	void*      (*calloc_func) (size_t count, size_t size);
	void*      (*realloc_func)(void* ptr, size_t size);
	void       (*free_func)   (void* ptr);

	// Bind slot configuration (NULL = use defaults: material=0, system=1, instance=2)
	const skr_bind_settings_t* bind_settings;
} skr_settings_t;

typedef struct skr_shader_t skr_shader_t;

typedef struct skr_material_info_t {
	const skr_shader_t*  shader;
	skr_cull_            cull;
	skr_write_           write_mask;
	skr_compare_         depth_test;
	skr_blend_state_t    blend_state;
	bool                 alpha_to_coverage;
	bool                 depth_clamp;   // Clamp depth to [0,1] instead of clipping (useful for shadow mapping)
	skr_stencil_state_t  stencil_front;
	skr_stencil_state_t  stencil_back;
	int32_t              queue_offset;  // Render queue offset for sorting (lower draws first)
} skr_material_info_t;

// While this project is primarily Vulkan, the option to add backends in the
// future would be nice. WebGPU or D3D12 could be targets. However, we don't
// want to introduce pointer indirection to core graphics assets! We risk a bit
// of API bleed by letting each backend define its own structs, but gain the
// flexibility to manage the memory our structures use.
#define SKR_VK
#ifdef SKR_VK
#include "../vk/skr_vulkan.h"
SKR_API VkInstance        skr_get_vk_instance              (void);
SKR_API VkDevice          skr_get_vk_device                (void);
SKR_API VkPhysicalDevice  skr_get_vk_physical_device       (void);
SKR_API VkQueue           skr_get_vk_graphics_queue        (void);
SKR_API uint32_t          skr_get_vk_graphics_queue_family (void);
SKR_API void              skr_get_vk_device_uuid           (uint8_t out_uuid[VK_UUID_SIZE]);
#endif

///////////////////////////////////////////////////////////////////////////////

SKR_API bool              skr_init                         (skr_settings_t settings);
SKR_API void              skr_shutdown                     (void);
SKR_API void              skr_thread_init                  (void);
SKR_API void              skr_thread_shutdown              (void);

SKR_API skr_future_t      skr_future_get                   (void);
SKR_API bool              skr_future_check                 (const skr_future_t* future);
SKR_API void              skr_future_wait                  (const skr_future_t* future);

SKR_API void              skr_cmd_begin                    (void);
SKR_API skr_future_t      skr_cmd_end                      (void);

SKR_API void              skr_callback_log                 (void (*callback)(skr_log_ level, const char* text));
SKR_API void              skr_log                          (skr_log_ level, const char* text, ...);
SKR_API uint64_t          skr_hash                         (const char *string);

SKR_API skr_err_          skr_buffer_create                (const void *opt_data, uint32_t size_count, uint32_t size_stride, skr_buffer_type_ type, skr_use_ use, skr_buffer_t *out_buffer);
SKR_API void              skr_buffer_destroy               (      skr_buffer_t* ref_buffer);
SKR_API bool              skr_buffer_is_valid              (const skr_buffer_t*     buffer);
SKR_API void              skr_buffer_set                   (      skr_buffer_t* ref_buffer, const void *data, uint32_t size_bytes);
SKR_API void              skr_buffer_get                   (const skr_buffer_t*     buffer, void *ref_buffer, uint32_t buffer_size);
SKR_API uint32_t          skr_buffer_get_size              (const skr_buffer_t*     buffer);
SKR_API void              skr_buffer_set_name              (      skr_buffer_t* ref_buffer, const char* name);

SKR_API skr_err_          skr_vert_type_create             (const skr_vert_component_t* items, int32_t item_count, skr_vert_type_t* out_type);
SKR_API bool              skr_vert_type_is_valid           (const skr_vert_component_t* type);
SKR_API void              skr_vert_type_destroy            (      skr_vert_type_t* ref_type);

SKR_API skr_err_          skr_mesh_create                  (const skr_vert_type_t* vert_type, skr_index_fmt_ ind_type, const void* vert_data, uint32_t vert_count, const void* opt_ind_data, uint32_t ind_count, skr_mesh_t* out_mesh);
SKR_API bool              skr_mesh_is_valid                (const skr_mesh_t*     mesh);
SKR_API void              skr_mesh_destroy                 (      skr_mesh_t* ref_mesh);
SKR_API uint32_t          skr_mesh_get_vert_count          (const skr_mesh_t*     mesh);
SKR_API uint32_t          skr_mesh_get_ind_count           (const skr_mesh_t*     mesh);
SKR_API void              skr_mesh_set_name                (      skr_mesh_t* ref_mesh, const char* name);
SKR_API skr_err_          skr_mesh_set_verts               (      skr_mesh_t* ref_mesh, const void* vert_data, uint32_t vert_count);
SKR_API skr_err_          skr_mesh_set_inds                (      skr_mesh_t* ref_mesh, const void* ind_data,  uint32_t ind_count);
SKR_API skr_err_          skr_mesh_set_data                (      skr_mesh_t* ref_mesh, const void* vert_data, uint32_t vert_count, const void* ind_data, uint32_t ind_count);
SKR_API skr_err_          skr_mesh_set_vertex_buffer       (      skr_mesh_t* ref_mesh, uint32_t binding, const skr_buffer_t* buffer, uint32_t vert_count);
SKR_API skr_buffer_t*     skr_mesh_get_vertex_buffer       (const skr_mesh_t*     mesh, uint32_t binding);

SKR_API skr_err_          skr_tex_create                   (skr_tex_fmt_ format, skr_tex_flags_ flags, skr_tex_sampler_t sampler, skr_vec3i_t size, int32_t multisample, int32_t mip_count, const skr_tex_data_t* opt_data, skr_tex_t* out_tex);
SKR_API skr_err_          skr_tex_create_copy              (const skr_tex_t*     src, skr_tex_fmt_ format, skr_tex_flags_ flags, int32_t multisample, skr_tex_t* out_tex);
SKR_API skr_err_          skr_tex_create_external          (skr_tex_external_info_t info, skr_tex_t* out_tex);
SKR_API skr_err_          skr_tex_update_external          (      skr_tex_t* ref_tex, skr_tex_external_update_t update);
SKR_API bool              skr_tex_is_valid                 (const skr_tex_t*     tex);
SKR_API void              skr_tex_destroy                  (      skr_tex_t* ref_tex);
SKR_API skr_err_          skr_tex_copy                     (const skr_tex_t*     src, skr_tex_t* dst, uint32_t src_mip, uint32_t src_layer, uint32_t dst_mip, uint32_t dst_layer);
SKR_API skr_err_          skr_tex_readback                 (const skr_tex_t*     tex, uint32_t mip_level, uint32_t array_layer, skr_tex_readback_t* out_readback);
SKR_API void              skr_tex_readback_destroy         (      skr_tex_readback_t* ref_readback);
SKR_API skr_vec3i_t       skr_tex_get_size                 (const skr_tex_t*     tex);
SKR_API uint32_t          skr_tex_get_array_count          (const skr_tex_t*     tex);
SKR_API skr_tex_fmt_      skr_tex_get_format               (const skr_tex_t*     tex);
SKR_API skr_tex_flags_    skr_tex_get_flags                (const skr_tex_t*     tex);
SKR_API int32_t           skr_tex_get_multisample          (const skr_tex_t*     tex);
SKR_API void              skr_tex_set_sampler              (      skr_tex_t* ref_tex, skr_tex_sampler_t sampler);
SKR_API skr_tex_sampler_t skr_tex_get_sampler              (const skr_tex_t*     tex);
SKR_API skr_err_          skr_tex_set_data                 (      skr_tex_t* ref_tex, const skr_tex_data_t* data);
SKR_API void              skr_tex_generate_mips            (      skr_tex_t* ref_tex, const skr_shader_t* opt_compute_shader);
SKR_API void              skr_tex_set_name                 (      skr_tex_t* ref_tex, const char* name);
SKR_API bool              skr_tex_fmt_is_supported         (skr_tex_fmt_ format, skr_tex_flags_ flags, int32_t multisample);
SKR_API void              skr_tex_fmt_block_info           (skr_tex_fmt_ format, uint32_t* opt_out_block_width, uint32_t* opt_out_block_height, uint32_t* opt_out_bytes_per_block);
SKR_API uint32_t          skr_tex_fmt_to_native            (skr_tex_fmt_ format);
SKR_API skr_tex_fmt_      skr_tex_fmt_from_native          (uint32_t native_format);
SKR_API uint32_t          skr_tex_calc_mip_count           (skr_vec3i_t size);
SKR_API skr_vec3i_t       skr_tex_calc_mip_dimensions      (skr_vec3i_t base_size, uint32_t mip_level);
SKR_API uint64_t          skr_tex_calc_mip_size            (skr_tex_fmt_ format, skr_vec3i_t base_size, uint32_t mip_level);

SKR_API skr_err_          skr_surface_create               (void* vk_surface_khr, skr_surface_t* out_surface);
SKR_API void              skr_surface_destroy              (      skr_surface_t* ref_surface);
SKR_API void              skr_surface_resize               (      skr_surface_t* ref_surface);
SKR_API skr_acquire_      skr_surface_next_tex             (      skr_surface_t* ref_surface, skr_tex_t** out_tex);
SKR_API void              skr_surface_present              (      skr_surface_t* ref_surface);
SKR_API skr_vec2i_t       skr_surface_get_size             (const skr_surface_t*     surface);

SKR_API skr_err_          skr_shader_create                (const void *shader_data, uint32_t data_size, skr_shader_t* out_shader);
SKR_API bool              skr_shader_is_valid              (const skr_shader_t*     shader);
SKR_API void              skr_shader_destroy               (      skr_shader_t* ref_shader);
SKR_API skr_bind_t        skr_shader_get_bind              (const skr_shader_t*     shader, const char* bind_name);
SKR_API void              skr_shader_set_name              (      skr_shader_t* ref_shader, const char* name);

SKR_API skr_err_          skr_compute_create               (const skr_shader_t* shader, skr_compute_t* out_compute);
SKR_API bool              skr_compute_is_valid             (const skr_compute_t*     compute);
SKR_API void              skr_compute_destroy              (      skr_compute_t* ref_compute);
SKR_API skr_bind_t        skr_compute_get_bind             (const skr_compute_t*     compute, const char* bind_name);
SKR_API void              skr_compute_execute              (      skr_compute_t* ref_compute, uint32_t x, uint32_t y, uint32_t z);
SKR_API void              skr_compute_execute_indirect     (      skr_compute_t* ref_compute, skr_buffer_t* indirect_args);
SKR_API void              skr_compute_set_tex              (      skr_compute_t* ref_compute, const char* name, skr_tex_t*    texture);
SKR_API void              skr_compute_set_buffer           (      skr_compute_t* ref_compute, const char* name, skr_buffer_t* buffer);
SKR_API void              skr_compute_set_params           (      skr_compute_t* ref_compute, const void* data, uint32_t size);
SKR_API void              skr_compute_set_param            (      skr_compute_t*     compute, const char* name, sksc_shader_var_ type, uint32_t count, const void* data);
SKR_API void              skr_compute_get_param            (const skr_compute_t*     compute, const char* name, sksc_shader_var_ type, uint32_t count, void* out_data);

SKR_API skr_err_          skr_material_create              (skr_material_info_t info, skr_material_t* out_material);
SKR_API bool              skr_material_is_valid            (const skr_material_t*     material);
SKR_API void              skr_material_set_tex             (      skr_material_t* ref_material, const char* name, skr_tex_t*    texture);
SKR_API void              skr_material_set_buffer          (      skr_material_t* ref_material, const char* name, skr_buffer_t* buffer);
SKR_API void              skr_material_set_params          (      skr_material_t* ref_material, const void* data, uint32_t size);
SKR_API void              skr_material_destroy             (      skr_material_t* ref_material);
SKR_API void              skr_material_set_param           (      skr_material_t* ref_material, const char* name, sksc_shader_var_ type, uint32_t count, const void* data);
SKR_API void              skr_material_get_param           (const skr_material_t*     material, const char* name, sksc_shader_var_ type, uint32_t count, void* out_data);

SKR_API skr_err_          skr_render_list_create           (skr_render_list_t* out_list);
SKR_API void              skr_render_list_destroy          (skr_render_list_t* ref_list);
SKR_API void              skr_render_list_clear            (skr_render_list_t* ref_list);
SKR_API void              skr_render_list_add              (skr_render_list_t* ref_list, skr_mesh_t* mesh, skr_material_t* material, const void* opt_instance_data, uint32_t single_instance_data_size, uint32_t instance_count);
SKR_API void              skr_render_list_add_indexed      (skr_render_list_t* ref_list, skr_mesh_t* mesh, skr_material_t* material, int32_t first_index, int32_t index_count, int32_t vertex_offset, const void* opt_instance_data, uint32_t single_instance_data_size, uint32_t instance_count);

SKR_API void              skr_renderer_frame_begin         (void);
SKR_API void              skr_renderer_frame_end           (skr_surface_t** opt_surfaces, uint32_t count);  // Submit frame with surface synchronization
SKR_API void              skr_renderer_begin_pass          (skr_tex_t* color, skr_tex_t* depth, skr_tex_t* opt_resolve, skr_clear_ clear, skr_vec4_t clear_color, float clear_depth, uint32_t clear_stencil);
SKR_API void              skr_renderer_end_pass            (void);
SKR_API void              skr_renderer_set_global_constants(int32_t bind, const skr_buffer_t* buffer);
SKR_API void              skr_renderer_set_global_texture  (int32_t bind, const skr_tex_t* tex);
SKR_API void              skr_renderer_set_viewport        (skr_rect_t viewport);
SKR_API void              skr_renderer_set_scissor         (skr_recti_t scissor);
SKR_API void              skr_renderer_blit                (skr_material_t* material, skr_tex_t* to, skr_recti_t bounds_px);

SKR_API void              skr_renderer_draw                (skr_render_list_t* list, const void* system_data, uint32_t system_data_size, int32_t instance_multiplier);
SKR_API void              skr_renderer_draw_mesh_immediate (skr_mesh_t* mesh, skr_material_t* material, int32_t first_index, int32_t index_count, int32_t vertex_offset, int32_t instance_count);
SKR_API float             skr_renderer_get_gpu_time_ms     (void);

#ifdef __cplusplus
}
#endif