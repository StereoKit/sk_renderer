// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Symbol visibility macros
#if defined(_WIN32) || defined(_WIN64)
	#ifdef SKR_BUILD_SHARED
		#define SKSC_API __declspec(dllexport)
	#else
		#define SKSC_API
	#endif
#elif defined(__GNUC__) || defined(__clang__)
	#define SKSC_API __attribute__((visibility("default")))
#else
	#define SKSC_API
#endif

///////////////////////////////////////////////////////////////////////////////

typedef enum {
	skr_vertex_fmt_none,
	skr_vertex_fmt_f64,
	skr_vertex_fmt_f32,
	skr_vertex_fmt_f16,
	skr_vertex_fmt_i32,
	skr_vertex_fmt_i16,
	skr_vertex_fmt_i8,
	skr_vertex_fmt_i32_normalized,
	skr_vertex_fmt_i16_normalized,
	skr_vertex_fmt_i8_normalized,
	skr_vertex_fmt_ui32,
	skr_vertex_fmt_ui16,
	skr_vertex_fmt_ui8,
	skr_vertex_fmt_ui32_normalized,
	skr_vertex_fmt_ui16_normalized,
	skr_vertex_fmt_ui8_normalized,
} skr_vertex_fmt_;

typedef enum {
	sksc_shader_var_none,
	sksc_shader_var_int,
	sksc_shader_var_uint,
	sksc_shader_var_uint8,
	sksc_shader_var_float,
	sksc_shader_var_double,
} sksc_shader_var_;

typedef enum {
	skr_semantic_none,
	skr_semantic_position,
	skr_semantic_texcoord,
	skr_semantic_normal,
	skr_semantic_binormal,
	skr_semantic_tangent,
	skr_semantic_color,
	skr_semantic_psize,
	skr_semantic_blendweight,
	skr_semantic_blendindices,
} skr_semantic_;

typedef enum {
	skr_stage_vertex  = 1 << 0,
	skr_stage_pixel   = 1 << 1,
	skr_stage_compute = 1 << 2,
} skr_stage_;

typedef enum {
	skr_register_default,
	skr_register_vertex,
	skr_register_index,
	skr_register_constant,
	skr_register_texture,
	skr_register_read_buffer,
	skr_register_readwrite,
	skr_register_readwrite_tex,
	skr_register_input_attachment,
} skr_register_;

typedef enum {
	skr_shader_lang_hlsl,
	skr_shader_lang_spirv,
	skr_shader_lang_glsl,
	skr_shader_lang_glsl_es,
	skr_shader_lang_glsl_web,
} skr_shader_lang_;

typedef enum {
	sksc_result_unknown       =  0,
	sksc_result_success       =  1,
	sksc_result_out_of_memory = -1,
	sksc_result_bad_format    = -2,
	sksc_result_old_version   = -3,
	sksc_result_corrupt_data  = -4,
} sksc_result_;

// Bit indexes into sksc_shader_meta_t.features. Each set bit names a Vulkan
// device feature (or format/property query) the runtime must confirm — and
// where applicable enable at device creation — before this shader's pipeline
// can be created. Comments give the exact struct member to check; extension
// names are listed for devices below the Vulkan version that made them core.
// Anything the SPIR-V declares that maps to no bit sets `unknown`, so a
// runtime never silently under-checks a newer compiler's output. The SPIR-V
// blob's own OpCapability/OpExtension lists remain the exhaustive ground truth.
typedef enum {
	// VkPhysicalDeviceShaderFloat16Int8Features.shaderFloat16
	// (VK_KHR_shader_float16_int8; core in 1.2)
	sksc_feature_bit_float16          = 0,
	// VkPhysicalDevice16BitStorageFeatures: storageBuffer16BitAccess,
	// uniformAndStorageBuffer16BitAccess, storagePushConstant16
	// (VK_KHR_16bit_storage; core in 1.1)
	sksc_feature_bit_storage16        = 1,
	// VkPhysicalDevice8BitStorageFeatures: storageBuffer8BitAccess,
	// uniformAndStorageBuffer8BitAccess, storagePushConstant8
	// (VK_KHR_8bit_storage; core in 1.2)
	sksc_feature_bit_storage8         = 2,
	// VkPhysicalDeviceFeatures.shaderStorageImageExtendedFormats
	sksc_feature_bit_extended_formats = 3,
	// not a device feature: the bound storage image's VkFormat must report
	// VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT (vkGetPhysicalDeviceFormatProperties)
	sksc_feature_bit_image_atomics    = 4,
	// VkPhysicalDeviceSubgroupProperties (core in 1.1): supportedStages must
	// include the shader's stages and supportedOperations the used op classes
	// (vote/ballot/arithmetic/shuffle/clustered/quad)
	sksc_feature_bit_subgroups        = 5,
	// VkPhysicalDeviceSubgroupSizeControlFeatures.subgroupSizeControl, with the
	// pinned size inside min/maxSubgroupSize of the matching properties struct
	// (VK_EXT_subgroup_size_control; core in 1.3)
	sksc_feature_bit_wave_size        = 6,
	// VkPhysicalDeviceMultiviewFeatures.multiview (VK_KHR_multiview; core in 1.1)
	sksc_feature_bit_multiview        = 7,
	// VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures.shaderDemoteToHelperInvocation
	// (VK_EXT_shader_demote_to_helper_invocation; core in 1.3)
	sksc_feature_bit_demote           = 8,
	// VkPhysicalDeviceFeatures.shaderInt64
	sksc_feature_bit_int64            = 9,
	// VkPhysicalDeviceFeatures.shaderFloat64
	sksc_feature_bit_float64          = 10,
	// VkPhysicalDeviceFeatures.shaderInt16
	sksc_feature_bit_int16            = 11,
	// VkPhysicalDeviceShaderFloat16Int8Features.shaderInt8
	// (VK_KHR_shader_float16_int8; core in 1.2)
	sksc_feature_bit_int8             = 12,
	// VkPhysicalDeviceFeatures.shaderStorageImageReadWithoutFormat and/or
	// shaderStorageImageWriteWithoutFormat
	sksc_feature_bit_formatless       = 13,
	// VkPhysicalDeviceShaderTileImageFeaturesEXT: shaderTileImageColorReadAccess,
	// plus DepthReadAccess/StencilReadAccess when those are read
	// (VK_EXT_shader_tile_image)
	sksc_feature_bit_tile_image       = 14,
	// VkPhysicalDeviceShaderAtomicFloatFeaturesEXT: shaderBufferFloat32Atomics
	// and shaderSharedFloat32Atomics for exchange, plus the ...AtomicAdd members
	// for add (VK_EXT_shader_atomic_float); float min/max additionally needs
	// VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT ...Float32AtomicMinMax
	// (VK_EXT_shader_atomic_float2)
	sksc_feature_bit_float_atomics    = 15,
	// a capability/extension with no assigned bit: fall back to parsing the
	// SPIR-V's OpCapability/OpExtension lists before trusting this mask
	sksc_feature_bit_unknown          = 63,
} sksc_feature_bit_;

typedef struct {
	skr_vertex_fmt_ format;
	uint8_t         count;
	skr_semantic_   semantic;
	uint8_t         semantic_slot;
	uint8_t         binding;
} skr_vert_component_t;

typedef struct {
	uint16_t slot;
	// of type skr_stage_
	uint8_t  stage_bits; 
	// of type skr_register_
	uint8_t  register_type;
} skr_bind_t;

typedef struct {
	char     name [32];
	uint64_t name_hash;
	char     extra[64];
	char     type_name[32]; // For struct types, the name of the struct (e.g., "circle_mask_t")
	uint32_t offset;
	uint32_t size;
	// of type sksc_shader_var_
	uint16_t type;
	uint16_t type_count;
} sksc_shader_var_t;

typedef struct {
	char              name[32];
	uint64_t          name_hash;
	skr_bind_t        bind;
	uint8_t           space;
	uint32_t          size;
	void             *defaults;
	uint32_t          var_count;
	sksc_shader_var_t *vars;
} sksc_shader_buffer_t;

typedef struct {
	char       name [32];
	uint64_t   name_hash;
	char       value[64];
	char       tags [64];
	skr_bind_t bind;
	uint32_t   element_size; // For StructuredBuffer<T>, the size of T in bytes
	// Texture shape: bits 0-2 dimension (0 = unreported, 1 = 2D, 2 = 3D,
	// 3 = cube, 4 = 1D), bit 3 arrayed, bit 4 multisampled, bit 5 paired with
	// a comparison sampler
	uint8_t    shape;
	// SpvImageFormat of a storage image binding (0 = Unknown / not a storage
	// image); Vulkan requires the bound view's format to match when declared
	uint8_t    image_format;
} sksc_shader_resource_t;

// A specialization constant: [[vk::constant_id(1)]] const int LIGHT_COUNT = 4;
// Values are 32-bit scalars; bools are reflected as int holding VkBool32.
typedef struct {
	char     name[32];
	uint64_t name_hash;
	uint32_t constant_id;   // The [[vk::constant_id(N)]] value
	uint32_t default_value; // Bit pattern of the default; interpret via `type`
	// of type sksc_shader_var_
	uint16_t type;
	// of type skr_stage_
	uint8_t  stage_bits;
} sksc_shader_spec_constant_t;

typedef struct {
	int32_t total;
	int32_t tex_read;
	int32_t dynamic_flow;
} sksc_shader_ops_t;

// All pointer members (buffers, resources, vertex_inputs, spec_constants) and
// their sub-arrays (per-buffer vars and defaults) are carved from a single
// allocation rooted at `buffers`. Call sksc_shader_meta_free() to release; do
// not free members individually.
typedef struct {
	char                        name[256];
	sksc_shader_buffer_t       *buffers;
	sksc_shader_resource_t     *resources;
	skr_vert_component_t       *vertex_inputs;
	sksc_shader_spec_constant_t*spec_constants;
	uint32_t                    buffer_count;
	uint32_t                    resource_count;
	int32_t                     global_buffer_id;
	int32_t                     vertex_input_count;
	uint32_t                    spec_constant_count;
	sksc_shader_ops_t           ops_vertex;
	sksc_shader_ops_t           ops_pixel;
	uint32_t                    wave_size;
	// Required device features (sksc_feature_bit_ indexes), derived from the
	// SPIR-V's capability/extension declarations; check before pipeline
	// creation. The second word is reserved growth room, always written.
	uint64_t                    features;
	uint64_t                    features_reserved;
} sksc_shader_meta_t;

typedef struct {
	skr_shader_lang_ language;
	skr_stage_       stage;
	uint32_t         code_size;
	void            *code;
	uint32_t         wave_size; // per-entry subgroup size (0 = unpinned)
} sksc_shader_file_stage_t;

typedef struct {
	sksc_shader_meta_t        meta;
	uint32_t                  stage_count;
	sksc_shader_file_stage_t *stages;
} sksc_shader_file_t;


///////////////////////////////////////////////////////////////////////////////

SKSC_API bool                     sksc_shader_file_verify         (const void *file_memory, uint32_t file_size, uint16_t *out_version, char *out_name, uint32_t out_name_size);
SKSC_API sksc_result_             sksc_shader_file_load_memory    (const void *file_memory, uint32_t file_size, sksc_shader_file_t *out_file);
SKSC_API void                     sksc_shader_file_destroy        (sksc_shader_file_t *ref_file);

SKSC_API skr_bind_t               sksc_shader_meta_get_bind       (const sksc_shader_meta_t*     meta, const char *name);
SKSC_API int32_t                  sksc_shader_meta_get_var_count  (const sksc_shader_meta_t*     meta);
SKSC_API int32_t                  sksc_shader_meta_get_var_index  (const sksc_shader_meta_t*     meta, const char *name);
SKSC_API int32_t                  sksc_shader_meta_get_var_index_h(const sksc_shader_meta_t*     meta, uint64_t name_hash);
SKSC_API const sksc_shader_var_t* sksc_shader_meta_get_var_info   (const sksc_shader_meta_t*     meta, int32_t  var_index);
SKSC_API void                     sksc_shader_meta_free           (      sksc_shader_meta_t* ref_meta);

#ifdef __cplusplus
}
#endif
