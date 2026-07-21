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
	skr_register_sample_weight, // VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM
	skr_register_block_match,   // VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM
	skr_register_tile_sampled,  // VK_QCOM_tile_shading attachments
	skr_register_tile_storage,
} skr_register_;

typedef enum {
	skr_shader_lang_hlsl,
	skr_shader_lang_spirv,
	skr_shader_lang_glsl,
	skr_shader_lang_glsl_es,
	skr_shader_lang_glsl_web,
	skr_shader_lang_wgsl,     // v12: WebGPU Shading Language text (see webgpu-backend-plan.md)
} skr_shader_lang_;

// Current .sks container version written by skshaderc. Version must match
// exactly to load — .sks files carry no back-compat by policy; recompile
// shaders when this bumps.
#define SKSC_FILE_VERSION 12

// WGSL stages replace the SPIR-V ViewIndex builtin with a system-owned spec
// constant named sk_view_index (default 0) using this reserved constant id —
// browser WebGPU has no multiview, so the runtime renders one pass per array
// layer and specializes the pipeline per view. User shaders must not declare
// [[vk::constant_id(999)]]; skshaderc errors if they do when targeting WGSL.
#define SKSC_WGSL_VIEW_INDEX_SPEC_ID 999

// Bind-slot register shifts. HLSL register spaces collapse into one binding
// namespace: a resource's meta slot is its register index plus the shift for
// its register type. skshaderc applies these when compiling; both runtimes
// use them to translate between raw register indices (e.g. global binds) and
// meta slots.
#define SKSC_SLOT_TEXTURE   100 // t registers (also merged samplers on Vulkan)
#define SKSC_SLOT_READWRITE 200 // u registers (RW buffers/textures)
#define SKSC_SLOT_INPUT_ATT 300 // input attachments, + [[vk::input_attachment_index]]
#define SKSC_SLOT_SAMPLER   400 // standalone samplers (split-sampler WGSL stages)

// sksc_shader_resource_t.shape bit layout (see the field's comment for the
// per-register-type meaning of the top two bits)
#define SKSC_SHAPE_DIM_MASK   0x07 // 0 = unreported
#define SKSC_SHAPE_DIM_2D     1
#define SKSC_SHAPE_DIM_3D     2
#define SKSC_SHAPE_DIM_CUBE   3
#define SKSC_SHAPE_DIM_1D     4
#define SKSC_SHAPE_ARRAYED    0x08
#define SKSC_SHAPE_MS         0x10
#define SKSC_SHAPE_COMPARISON 0x20 // paired with a comparison sampler
#define SKSC_SHAPE_WRITTEN    0x40 // storage images; QCOM image-processing sampler on sampled textures
#define SKSC_SHAPE_READ       0x80 // storage images

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
	// VkPhysicalDeviceScalarBlockLayoutFeatures.scalarBlockLayout
	// (VK_EXT_scalar_block_layout; optional-core in 1.2): a buffer layout in this
	// shader breaks core relaxed block layout rules. Has no SPIR-V capability, so
	// this bit is the only machine-readable signal.
	sksc_feature_bit_scalar_layout    = 16,
	// VkPhysicalDeviceImageProcessingFeaturesQCOM carries one feature per op
	// family, so each gets its own bit — a device that only enables
	// textureBoxFilter still passes box-filter shaders. All three are
	// VK_QCOM_image_processing, and the shaders are SPIR-V 1.4 modules:
	// Vulkan 1.2+, or VK_KHR_spirv_1_4
	sksc_feature_bit_qcom_sample_weighted = 17, // textureSampleWeighted
	sksc_feature_bit_qcom_box_filter      = 18, // textureBoxFilter
	sksc_feature_bit_qcom_block_match     = 19, // textureBlockMatch
	// VkPhysicalDeviceImageProcessing2FeaturesQCOM.textureBlockMatch2
	// (VK_QCOM_image_processing2 Window/Gather ops; implies block_match's
	// requirements)
	sksc_feature_bit_qcom_image_proc2     = 20,
	// VkPhysicalDeviceTileShadingFeaturesQCOM: tileShading, plus
	// tileShadingFragmentStage / tileShadingPerTileDispatch by stage; the render
	// pass must be a tile shading render pass (VK_QCOM_tile_shading)
	sksc_feature_bit_qcom_tile_shading    = 21,
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
	// For shader meta vertex_inputs, the SPIR-V input location this attribute
	// is decorated with; used at pipeline creation to wire mesh components to
	// shader inputs by semantic. Ignored (and zero) for user-built mesh vert
	// types, where component-to-buffer layout is described by binding/offset.
	uint8_t         location;
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
	// a comparison sampler. On sampled textures, bit 6 means the sampler serves
	// QCOM image-processing ops (create it with
	// VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM); on storage images, bits
	// 6/7 record write/read usage so WebGPU layouts can declare exact access
	uint8_t    shape;
	// SpvImageFormat of a storage image binding (0 = Unknown / not a storage
	// image); Vulkan requires the bound view's format to match when declared
	uint8_t    image_format;
} sksc_shader_resource_t;

// A standalone sampler binding (v12, WGSL stages only). The Vulkan path merges
// HLSL SamplerState objects into combined image samplers, but WGSL requires
// textures and samplers to stay separate, so WGSL stage blobs bind these as
// their own entries. `slot` uses the same register-shift scheme as resources
// (s register + 400). `paired_slot` names the texture resource this sampler
// samples — the runtime applies that texture's sampler settings, matching
// Vulkan's combined-sampler semantics. The Vulkan backend ignores this array.
typedef struct {
	char     name[32];
	uint64_t name_hash;
	uint16_t slot;        // WGSL @binding (s register + 400)
	// of type skr_stage_
	uint8_t  stage_bits;
	uint8_t  _pad;
	uint16_t paired_slot; // Bind slot of the texture resource this sampler samples (0xFFFF = unpaired)
	uint16_t _pad2;
} sksc_shader_sampler_t;

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
	sksc_shader_sampler_t      *samplers;      // v12: standalone samplers for WGSL stages (empty otherwise)
	bool                        samplers_owned;// samplers is its own malloc (not carved from `buffers`); meta_free releases it
	uint32_t                    buffer_count;
	uint32_t                    resource_count;
	int32_t                     global_buffer_id;
	int32_t                     vertex_input_count;
	uint32_t                    spec_constant_count;
	uint32_t                    sampler_count; // v12
	sksc_shader_ops_t           ops_vertex;
	sksc_shader_ops_t           ops_pixel;
	uint32_t                    wave_size;
	// //--apron: requested VkRenderPassTileShadingCreateInfoQCOM::tileApronSize
	// (width, height) for VK_QCOM_tile_shading passes; (0, 0) = no apron
	uint32_t                    tile_apron[2];
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
SKSC_API uint64_t                 sksc_shader_meta_missing_features(const sksc_shader_meta_t*     meta, uint64_t enabled_features);
SKSC_API int32_t                  sksc_shader_meta_get_var_count  (const sksc_shader_meta_t*     meta);
SKSC_API int32_t                  sksc_shader_meta_get_var_index  (const sksc_shader_meta_t*     meta, const char *name);
SKSC_API int32_t                  sksc_shader_meta_get_var_index_h(const sksc_shader_meta_t*     meta, uint64_t name_hash);
SKSC_API const sksc_shader_var_t* sksc_shader_meta_get_var_info   (const sksc_shader_meta_t*     meta, int32_t  var_index);
SKSC_API void                     sksc_shader_meta_free           (      sksc_shader_meta_t* ref_meta);

#ifdef __cplusplus
}
#endif
