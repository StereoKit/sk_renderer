// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_sk_renderer.h"
#include "skr_pipeline.h"
#include "skr_conversions.h"
#include "skr_scratch.h"
#include "skr_transient.h"

#include "skr_mipgen_2d.hlsl.h"
#include "skr_mipgen_cube.hlsl.h"

#define VOLK_IMPLEMENTATION
#include <volk.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Global state
///////////////////////////////////////////////////////////////////////////////

_skr_vk_t _skr_vk;

///////////////////////////////////////////////////////////////////////////////
// Memory allocation wrappers
///////////////////////////////////////////////////////////////////////////////

void* _skr_malloc(size_t size) {
	return _skr_vk.malloc_func(size);
}

void* _skr_calloc(size_t count, size_t size) {
	return _skr_vk.calloc_func(count, size);
}

void* _skr_realloc(void* ptr, size_t size) {
	return _skr_vk.realloc_func(ptr, size);
}

void _skr_free(void* ptr) {
	_skr_vk.free_func(ptr);
}

static char* _skr_strdup(const char* str) {
	size_t len    = strlen(str) + 1;
	char*  result = _skr_malloc(len);
	memcpy(result, str, len);
	return result;
}

///////////////////////////////////////////////////////////////////////////////
// Extension & feature requests
///////////////////////////////////////////////////////////////////////////////

// Request registry, grown on demand. Registration happens before skr_init
// provides the app allocators, so this uses stdlib malloc/realloc throughout.

typedef struct {
	int32_t offset; // Byte offset into the registry's feat_buffer
	int32_t size;
} _skr_req_feat_t;

typedef struct {
	const char* name;             // Interned; NULL for anonymous requests
	bool        required;
	int32_t     inst_ext_start, inst_ext_count; // Range in the registry's exts
	int32_t     dev_ext_start,  dev_ext_count;  // Range in the registry's exts
	int32_t     feat_start,     feat_count;     // Range in the registry's feats
	// Evaluation results, rebuilt by each skr_init
	bool        enabled;
	const char* missing;          // First unsatisfied piece, for logging
	char        missing_feature[40];
} _skr_req_t;

typedef struct {
	_skr_req_t*      reqs;
	int32_t          req_count,        req_cap;
	const char**     exts;             // Extension names, referenced by request ranges
	int32_t          ext_count,        ext_cap;
	_skr_req_feat_t* feats;
	int32_t          feat_count,       feat_cap;
	uint8_t*         feat_buffer;      // 8-aligned offsets keep structs pointer-aligned
	int32_t          feat_buffer_used, feat_buffer_cap;
	char**           strs;             // Interned strings, individually allocated so pointers stay stable
	int32_t          str_count,        str_cap;
} _skr_req_registry_t;

static _skr_req_registry_t _skr_reg = {0};

// Grow a registry array to at least `needed` elements, false on out-of-memory
static bool _skr_req_reserve(void* array_ptr, int32_t* cap, int32_t needed, size_t elem_size) {
	if (needed <= *cap) return true;
	int32_t new_cap = *cap < 16 ? 16 : *cap;
	while (new_cap < needed) new_cap *= 2;
	void** array = (void**)array_ptr;
	void*  grown = realloc(*array, (size_t)new_cap * elem_size);
	if (grown == NULL) return false;
	*array = grown;
	*cap   = new_cap;
	return true;
}

// Copy a string into the registry, reusing existing copies. NULL on OOM.
static const char* _skr_req_intern(const char* str) {
	for (int32_t i = 0; i < _skr_reg.str_count; i++)
		if (strcmp(_skr_reg.strs[i], str) == 0) return _skr_reg.strs[i];
	if (!_skr_req_reserve(&_skr_reg.strs, &_skr_reg.str_cap, _skr_reg.str_count + 1, sizeof(char*)))
		return NULL;
	size_t len  = strlen(str) + 1;
	char*  copy = malloc(len);
	if (copy == NULL) return NULL;
	memcpy(copy, str, len);
	_skr_reg.strs[_skr_reg.str_count++] = copy;
	return copy;
}

static const VkBaseInStructure* _skr_req_feat_struct(const _skr_req_feat_t* feat) {
	return (const VkBaseInStructure*)(_skr_reg.feat_buffer + feat->offset);
}

void skr_vk_request(const skr_vk_request_t* request) {
	if (_skr_vk.initialized) {
		skr_log(skr_log_warning, "skr_vk_request must be called before skr_init");
		return;
	}
	if (request == NULL) return;
	const char* label = request->name ? request->name : "(anonymous)";

	// Re-registering an existing name is a no-op
	if (request->name != NULL) {
		for (int32_t i = 0; i < _skr_reg.req_count; i++)
			if (_skr_reg.reqs[i].name != NULL && strcmp(_skr_reg.reqs[i].name, request->name) == 0) return;
	}

	for (int32_t i = 0; i < request->feature_count; i++) {
		const skr_vk_feature_t* f = &request->features[i];
		if (f->vk_struct == NULL || f->size < (int32_t)sizeof(VkBaseInStructure)) {
			skr_log(skr_log_warning, "skr_vk_request '%s': invalid feature struct, request dropped", label);
			return;
		}
		// Core features ride VkDeviceCreateInfo.pEnabledFeatures, and Vulkan
		// forbids combining that with a chained VkPhysicalDeviceFeatures2.
		if (((const VkBaseInStructure*)f->vk_struct)->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
			skr_log(skr_log_warning, "skr_vk_request '%s': pass VkPhysicalDevice*Features structs, not VkPhysicalDeviceFeatures2 itself; request dropped", label);
			return;
		}
	}

	// Save counts so an out-of-memory mid-copy rolls back; stray interned
	// strings from a dropped request are harmless
	int32_t save_ext  = _skr_reg.ext_count;
	int32_t save_feat = _skr_reg.feat_count;
	int32_t save_buf  = _skr_reg.feat_buffer_used;
	bool    ok        = _skr_req_reserve(&_skr_reg.reqs, &_skr_reg.req_cap, _skr_reg.req_count + 1, sizeof(_skr_req_t));

	_skr_req_t req = {0};
	req.required = request->required;
	if (ok && request->name != NULL) {
		req.name = _skr_req_intern(request->name);
		ok       = req.name != NULL;
	}
	req.inst_ext_start = _skr_reg.ext_count;
	for (int32_t i = 0; ok && i < request->instance_extension_count; i++) {
		const char* ext = _skr_req_reserve(&_skr_reg.exts, &_skr_reg.ext_cap, _skr_reg.ext_count + 1, sizeof(const char*))
			? _skr_req_intern(request->instance_extensions[i]) : NULL;
		if (ext) _skr_reg.exts[_skr_reg.ext_count++] = ext;
		else     ok = false;
	}
	req.inst_ext_count = _skr_reg.ext_count - req.inst_ext_start;
	req.dev_ext_start  = _skr_reg.ext_count;
	for (int32_t i = 0; ok && i < request->device_extension_count; i++) {
		const char* ext = _skr_req_reserve(&_skr_reg.exts, &_skr_reg.ext_cap, _skr_reg.ext_count + 1, sizeof(const char*))
			? _skr_req_intern(request->device_extensions[i]) : NULL;
		if (ext) _skr_reg.exts[_skr_reg.ext_count++] = ext;
		else     ok = false;
	}
	req.dev_ext_count = _skr_reg.ext_count - req.dev_ext_start;
	req.feat_start    = _skr_reg.feat_count;
	for (int32_t i = 0; ok && i < request->feature_count; i++) {
		int32_t aligned = (request->features[i].size + 7) & ~7;
		if (_skr_req_reserve(&_skr_reg.feats, &_skr_reg.feat_cap, _skr_reg.feat_count + 1, sizeof(_skr_req_feat_t)) &&
		    _skr_req_reserve(&_skr_reg.feat_buffer, &_skr_reg.feat_buffer_cap, _skr_reg.feat_buffer_used + aligned, sizeof(uint8_t))) {
			memcpy(_skr_reg.feat_buffer + _skr_reg.feat_buffer_used, request->features[i].vk_struct, request->features[i].size);
			_skr_reg.feats[_skr_reg.feat_count++] = (_skr_req_feat_t){ _skr_reg.feat_buffer_used, request->features[i].size };
			_skr_reg.feat_buffer_used += aligned;
		} else ok = false;
	}
	req.feat_count = _skr_reg.feat_count - req.feat_start;

	if (!ok) {
		_skr_reg.ext_count        = save_ext;
		_skr_reg.feat_count       = save_feat;
		_skr_reg.feat_buffer_used = save_buf;
		skr_log(skr_log_warning, "skr_vk_request '%s': out of memory, request dropped", label);
		return;
	}
	_skr_reg.reqs[_skr_reg.req_count++] = req;
}

bool skr_vk_request_enabled(const char* name) {
	if (name == NULL) return false;
	for (int32_t i = 0; i < _skr_reg.req_count; i++)
		if (_skr_reg.reqs[i].name != NULL && strcmp(_skr_reg.reqs[i].name, name) == 0) return _skr_reg.reqs[i].enabled;
	return false;
}

// Single-device-extension request, named by the extension
static void _skr_ext_request(const char* extension_name) {
	skr_vk_request(&(skr_vk_request_t){
		.name                   = extension_name,
		.device_extensions      = &extension_name,
		.device_extension_count = 1,
	});
}

// sk_renderer's own optional extensions and features, expressed as requests.
// Results feed the has_* flags and shader feature mask during skr_init.
static void _skr_register_internal_requests(void) {
	// VK_KHR_swapchain is optional so headless environments can init without
	// presentation, see skr_capability_presentation
	_skr_ext_request(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	_skr_ext_request(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);          // External memory for GL interop
	_skr_ext_request(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);     // DMA-BUF import
	_skr_ext_request(VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME);
	_skr_ext_request(VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME);
	_skr_ext_request(VK_QCOM_RENDER_PASS_SHADER_RESOLVE_EXTENSION_NAME);
	_skr_ext_request(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);           // Sync FD export for frame fences (VK_KHR_external_fence is core 1.1)
	_skr_ext_request(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);         // Postfx depth input attachments (per-reference aspect masks)
	_skr_ext_request(VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME); // Legacy instanced stereo: SV_RenderTargetArrayIndex from the vertex stage
	// All three alias the same enum value, so any one of them will do. The 1.3
	// core promotion is no help, since the instance targets Vulkan 1.1.
	_skr_ext_request(VK_KHR_LOAD_STORE_OP_NONE_EXTENSION_NAME);
	_skr_ext_request(VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME);
	_skr_ext_request(VK_QCOM_RENDER_PASS_STORE_OPS_EXTENSION_NAME);
#ifndef __ANDROID__
	_skr_ext_request(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME); // Push descriptors have performance overhead per call on Adreno?
#endif
#ifdef VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
	_skr_ext_request(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#endif
#ifdef VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME
	_skr_ext_request(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
#endif

	// On-tile depth resolve so postfx reads 1x depth under MSAA. SAMPLE_ZERO
	// resolve support is mandated by the extension, so no mode query.
	skr_vk_request(&(skr_vk_request_t){
		.name                   = "depth_stencil_resolve",
		.device_extensions      = (const char*[]){ VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
		                                           VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME },
		.device_extension_count = 2,
	});

	// Multiview is required for stereo/XR rendering. It's core 1.1 but still a
	// feature flag, so a 1.1 device can report it unsupported.
	skr_vk_request(&(skr_vk_request_t){
		.name          = "multiview",
		.required      = true,
		.features      = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceMultiviewFeatures){
			.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
			.multiview = VK_TRUE,
		}, sizeof(VkPhysicalDeviceMultiviewFeatures) } },
		.feature_count = 1,
	});

	// YCbCr conversion for YUV/NV12 textures and AHB external memory. Core 1.1
	// struct, but some drivers lack the feature (older Mesa lavapipe).
	skr_vk_request(&(skr_vk_request_t){
		.name                       = "ycbcr_conversion",
		.features                   = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceSamplerYcbcrConversionFeatures){
			.sType                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
			.samplerYcbcrConversion = VK_TRUE,
		}, sizeof(VkPhysicalDeviceSamplerYcbcrConversionFeatures) } },
		.feature_count              = 1,
	});

	// Subgroup size control lets compute pipelines request a specific subgroup
	// size at pipeline creation time (the Vulkan analog of HLSL [WaveSize(N)]).
	skr_vk_request(&(skr_vk_request_t){
		.name                    = "subgroup_size_control",
		.device_extensions       = (const char*[]){ VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME },
		.device_extension_count  = 1,
		.features                = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceSubgroupSizeControlFeaturesEXT){
			.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
			.subgroupSizeControl = VK_TRUE,
		}, sizeof(VkPhysicalDeviceSubgroupSizeControlFeaturesEXT) } },
		.feature_count           = 1,
	});

	// Reports whether drivers actually merge the postfx subpass chain
	skr_vk_request(&(skr_vk_request_t){
		.name                     = "subpass_merge_feedback",
		.device_extensions        = (const char*[]){ VK_EXT_SUBPASS_MERGE_FEEDBACK_EXTENSION_NAME },
		.device_extension_count   = 1,
		.features                 = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceSubpassMergeFeedbackFeaturesEXT){
			.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_MERGE_FEEDBACK_FEATURES_EXT,
			.subpassMergeFeedback = VK_TRUE,
		}, sizeof(VkPhysicalDeviceSubpassMergeFeedbackFeaturesEXT) } },
		.feature_count            = 1,
	});

	// Synchronization2 is required by video decode and by tile shading's
	// tile-attachment dependencies; enabled whenever the device offers it.
	skr_vk_request(&(skr_vk_request_t){
		.name                   = "sync2",
		.device_extensions      = (const char*[]){ VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME },
		.device_extension_count = 1,
		.features               = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceSynchronization2Features){
			.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
			.synchronization2   = VK_TRUE,
		}, sizeof(VkPhysicalDeviceSynchronization2Features) } },
		.feature_count          = 1,
	});

	// Present fences make swapchain teardown exact: a present's semaphore
	// waits and image use are otherwise unobservable, and surface resize must
	// fall back to a full device idle before destroying the old swapchain.
	// Promoted to KHR, but drivers keep advertising the EXT spelling.
	skr_vk_request(&(skr_vk_request_t){
		.name                     = "swapchain_maintenance1",
		.instance_extensions      = (const char*[]){ VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
		                                             VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME },
		.instance_extension_count = 2,
		.device_extensions        = (const char*[]){ VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME },
		.device_extension_count   = 1,
		.features                 = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT){
			.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
			.swapchainMaintenance1 = VK_TRUE,
		}, sizeof(VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT) } },
		.feature_count            = 1,
	});

	// QCOM image processing: each op family is its own feature and sksc bit, so
	// each is its own request. The shaders are SPIR-V 1.4, hence the ext chain.
	const char* image_proc_exts[] = {
		VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME,
		VK_KHR_SPIRV_1_4_EXTENSION_NAME,
		VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
		VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME,
	};
	skr_vk_request(&(skr_vk_request_t){
		.name                     = "qcom_sample_weighted",
		.device_extensions        = image_proc_exts,
		.device_extension_count   = 4,
		.features                 = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceImageProcessingFeaturesQCOM){
			.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM,
			.textureSampleWeighted = VK_TRUE,
		}, sizeof(VkPhysicalDeviceImageProcessingFeaturesQCOM) } },
		.feature_count            = 1,
	});
	skr_vk_request(&(skr_vk_request_t){
		.name                   = "qcom_box_filter",
		.device_extensions      = image_proc_exts,
		.device_extension_count = 4,
		.features               = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceImageProcessingFeaturesQCOM){
			.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM,
			.textureBoxFilter   = VK_TRUE,
		}, sizeof(VkPhysicalDeviceImageProcessingFeaturesQCOM) } },
		.feature_count          = 1,
	});
	skr_vk_request(&(skr_vk_request_t){
		.name                   = "qcom_block_match",
		.device_extensions      = image_proc_exts,
		.device_extension_count = 4,
		.features               = (skr_vk_feature_t[]){ { &(VkPhysicalDeviceImageProcessingFeaturesQCOM){
			.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM,
			.textureBlockMatch  = VK_TRUE,
		}, sizeof(VkPhysicalDeviceImageProcessingFeaturesQCOM) } },
		.feature_count          = 1,
	});
	// The Window/Gather block-match variants layer on VK_QCOM_image_processing2
	// and imply plain block match.
	skr_vk_request(&(skr_vk_request_t){
		.name                   = "qcom_block_match2",
		.device_extensions      = (const char*[]){ VK_QCOM_IMAGE_PROCESSING_2_EXTENSION_NAME,
		                                           VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME,
		                                           VK_KHR_SPIRV_1_4_EXTENSION_NAME,
		                                           VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
		                                           VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME },
		.device_extension_count = 5,
		.features               = (skr_vk_feature_t[]){
			{ &(VkPhysicalDeviceImageProcessingFeaturesQCOM){
				.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM,
				.textureBlockMatch  = VK_TRUE,
			}, sizeof(VkPhysicalDeviceImageProcessingFeaturesQCOM) },
			{ &(VkPhysicalDeviceImageProcessing2FeaturesQCOM){
				.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_2_FEATURES_QCOM,
				.textureBlockMatch2 = VK_TRUE,
			}, sizeof(VkPhysicalDeviceImageProcessing2FeaturesQCOM) } },
		.feature_count          = 2,
	});

	// Fragment-stage tile shading: reading color attachments as sampled tile
	// attachments with an apron. Needs sync2's 64-bit access flags + renderpass2.
	skr_vk_request(&(skr_vk_request_t){
		.name                   = "qcom_tile_shading",
		.device_extensions      = (const char*[]){ VK_QCOM_TILE_SHADING_EXTENSION_NAME,
		                                           VK_QCOM_TILE_PROPERTIES_EXTENSION_NAME,
		                                           VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
		                                           VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME },
		.device_extension_count = 4,
		.features               = (skr_vk_feature_t[]){
			{ &(VkPhysicalDeviceTileShadingFeaturesQCOM){
				.sType                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_FEATURES_QCOM,
				.tileShading                   = VK_TRUE,
				.tileShadingFragmentStage      = VK_TRUE,
				.tileShadingColorAttachments   = VK_TRUE,
				.tileShadingSampledAttachments = VK_TRUE,
				.tileShadingApron              = VK_TRUE,
			}, sizeof(VkPhysicalDeviceTileShadingFeaturesQCOM) },
			{ &(VkPhysicalDeviceSynchronization2Features){
				.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
				.synchronization2 = VK_TRUE,
			}, sizeof(VkPhysicalDeviceSynchronization2Features) } },
		.feature_count          = 2,
	});

	// 16-bit storage access (sksc_feature_bit_storage16). The push-constant bit
	// is a separate opportunistic request, devices commonly lack just that one.
	skr_vk_request(&(skr_vk_request_t){
		.name                                   = "storage16",
		.features                               = (skr_vk_feature_t[]){ { &(VkPhysicalDevice16BitStorageFeatures){
			.sType                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
			.storageBuffer16BitAccess           = VK_TRUE,
			.uniformAndStorageBuffer16BitAccess = VK_TRUE,
		}, sizeof(VkPhysicalDevice16BitStorageFeatures) } },
		.feature_count                          = 1,
	});
	skr_vk_request(&(skr_vk_request_t){
		.name                      = "storage16_push",
		.features                  = (skr_vk_feature_t[]){ { &(VkPhysicalDevice16BitStorageFeatures){
			.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
			.storagePushConstant16 = VK_TRUE,
		}, sizeof(VkPhysicalDevice16BitStorageFeatures) } },
		.feature_count             = 1,
	});

	// Float atomics (sksc_feature_bit_float_atomics): the coarse bit promises
	// exchange + add + min/max, and min/max lives in atomic_float2's struct.
	skr_vk_request(&(skr_vk_request_t){
		.name                   = "float_atomics",
		.device_extensions      = (const char*[]){ VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
		                                           VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME },
		.device_extension_count = 2,
		.features               = (skr_vk_feature_t[]){
			{ &(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT){
				.sType                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
				.shaderBufferFloat32Atomics   = VK_TRUE,
				.shaderBufferFloat32AtomicAdd = VK_TRUE,
				.shaderSharedFloat32Atomics   = VK_TRUE,
				.shaderSharedFloat32AtomicAdd = VK_TRUE,
			}, sizeof(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT) },
			{ &(VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT){
				.sType                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
				.shaderBufferFloat32AtomicMinMax = VK_TRUE,
				.shaderSharedFloat32AtomicMinMax = VK_TRUE,
			}, sizeof(VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT) } },
		.feature_count          = 2,
	});
}

///////////////////////////////////////////////////////////////////////////////
// Validation layers
///////////////////////////////////////////////////////////////////////////////

static VKAPI_ATTR VkBool32 VKAPI_CALL _skr_vk_debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
	VkDebugUtilsMessageTypeFlagsEXT             type,
	const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
	void*                                       user_data) {

	if (callback_data->messageIdNumber == 0)           return VK_FALSE; // A lot of noise?
	if (callback_data->messageIdNumber == -1744492148) return VK_FALSE; // vkCreateGraphicsPipelines: pCreateInfos[] Inside the fragment shader, it writes to output Location X but there is no VkSubpassDescription::pColorAttachments[X] and this write is unused. Spec information at https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-fragmentoutput
	if (callback_data->messageIdNumber == -937765618 ) return VK_FALSE; // vkCreateGraphicsPipelines: pCreateInfos[].pVertexInputState Vertex attribute at location X not consumed by shader.
	if (callback_data->messageIdNumber == -60244330  ) return VK_FALSE;
	if (callback_data->messageIdNumber ==  533026821 ) return VK_FALSE; // gl_Layer ?
	if (callback_data->messageIdNumber ==  115483881 ) return VK_FALSE; // Geometry shader req, might need attention

	const char *severity_str = severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT ? "VERBOSE" :
							   severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT    ? "INFO"    :
							   severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? "WARNING" :
							   severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT   ? "ERROR"   : "UNKNOWN";
	skr_log_    level        = severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT   ? skr_log_critical :
							   severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? skr_log_warning  : skr_log_info;

	skr_log(level, "[Vulkan:%s:%d] %s", severity_str, callback_data->messageIdNumber, callback_data->pMessage);

	return VK_FALSE;
}

static bool _skr_vk_create_debug_messenger(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT callback, VkDebugUtilsMessengerEXT* out_messenger) {
	VkDebugUtilsMessengerCreateInfoEXT create_info = {
		.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		.pfnUserCallback = callback,
	};

	VkResult vr = vkCreateDebugUtilsMessengerEXT(instance, &create_info, NULL, out_messenger);
	SKR_VK_CHECK_RET(vr, "vkCreateDebugUtilsMessengerEXT", false);
	return true;
}

///////////////////////////////////////////////////////////////////////////////
// Initialization
///////////////////////////////////////////////////////////////////////////////

// Helper to check if an extension is available
static bool _skr_ext_available(const char* name, const VkExtensionProperties* available, uint32_t count) {
	for (uint32_t i = 0; i < count; i++) {
		if (strcmp(name, available[i].extensionName) == 0) return true;
	}
	return false;
}

bool skr_init(skr_settings_t settings) {
	if (_skr_vk.initialized) {
		skr_log(skr_log_warning, "sk_renderer already initialized");
		return false;
	}

	// Validate memory allocators - either all provided or none provided
	int32_t allocator_count = (settings.malloc_func  != NULL) + (settings.calloc_func  != NULL) +
	                          (settings.realloc_func != NULL) + (settings.free_func    != NULL);
	if (allocator_count != 0 && allocator_count != 4) {
		skr_log(skr_log_critical, "sk_renderer: Memory allocators must be all provided or all NULL");
		return false;
	}

	_skr_vk = (_skr_vk_t){0};
	_skr_vk.validation_enabled        = settings.enable_validation;
	_skr_vk.current_renderpass_idx    = -1;
	_skr_vk.main_thread_id            = thrd_current();
	_skr_vk.destroy_list              = _skr_destroy_list_create();

	// Set up memory allocators (use stdlib if none provided)
	_skr_vk.malloc_func  = settings.malloc_func  ? settings.malloc_func  : malloc;
	_skr_vk.calloc_func  = settings.calloc_func  ? settings.calloc_func  : calloc;
	_skr_vk.realloc_func = settings.realloc_func ? settings.realloc_func : realloc;
	_skr_vk.free_func    = settings.free_func    ? settings.free_func    : free;

	_skr_bind_pool_init();
	_skr_sampler_cache_init();
	_skr_mipgen_materials_init();
	_skr_scratch_pool_init();
	_skr_transient_pool_init();

	// Set up bind slot configuration (use defaults if not provided)
	if (settings.bind_settings) {
		_skr_vk.bind_settings = *settings.bind_settings;
	} else {
		_skr_vk.bind_settings = (skr_bind_settings_t){
			.material_slot = 0,
			.system_slot   = 1,
			.instance_slot = 2,
		};
	}

	_skr_vk.buffering = settings.buffering == skr_buffering_default
		? skr_buffering_triple
		: settings.buffering;

	// sk_renderer's own requests register through the same path apps use
	_skr_register_internal_requests();

	// Initialize volk
	VkResult vr = volkInitialize();
	SKR_VK_CHECK_RET(vr, volkInitialize, false);

	// Save global vkGetInstanceProcAddr before volkLoadInstance replaces it.
	// OpenXR runtimes (Quest) call vkGetInstanceProcAddr(NULL, ...) which only
	// works with the global loader version, not the instance-specific one.
	PFN_vkGetInstanceProcAddr global_get_instance_proc_addr = vkGetInstanceProcAddr;

	///////////////////////////////////////////////////////////////////////////
	// Extension definitions
	///////////////////////////////////////////////////////////////////////////

	// Instance extensions. VK_KHR_surface is optional so headless/offscreen
	// environments (CI runners, render baking) can init without a surface;
	// presentation capability is surfaced via skr_capability_presentation.
	const char* optional_instance_exts[] = {
		VK_KHR_SURFACE_EXTENSION_NAME,
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
	};
	const uint32_t optional_instance_ext_count = sizeof(optional_instance_exts) / sizeof(optional_instance_exts[0]);

	// Device extensions all come from requests, both sk_renderer's own and any
	// the application registered before skr_init.

	// Video decode extensions (all required together for video support).
	// VK_KHR_synchronization2 is also required, but lives in the optional list
	// above since tile shading needs it too — the video check below requires it
	// separately.
	const char* video_device_exts[] = {
		VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
		VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
		VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
		VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
	};
	const uint32_t video_device_ext_count = sizeof(video_device_exts) / sizeof(video_device_exts[0]);

	///////////////////////////////////////////////////////////////////////////
	// Instance creation
	///////////////////////////////////////////////////////////////////////////

	VkApplicationInfo app_info = {
		.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName   = settings.app_name ? settings.app_name : "sk_renderer_app",
		.applicationVersion = settings.app_version,
		.pEngineName        = "sk_renderer",
		.engineVersion      = VK_MAKE_VERSION(0, 1, 0),
		.apiVersion         = VK_API_VERSION_1_1,
	};

	// Get available instance extensions
	uint32_t available_inst_ext_count = 0;
	vkEnumerateInstanceExtensionProperties(NULL, &available_inst_ext_count, NULL);
	VkExtensionProperties* available_inst_exts = _skr_malloc(available_inst_ext_count * sizeof(VkExtensionProperties));
	vkEnumerateInstanceExtensionProperties(NULL, &available_inst_ext_count, available_inst_exts);

	// Build final instance extension list, sized for every possible source
	uint32_t instance_ext_cap = settings.required_extension_count + optional_instance_ext_count;
	for (int32_t r = 0; r < _skr_reg.req_count; r++)
		instance_ext_cap += (uint32_t)_skr_reg.reqs[r].inst_ext_count;
	const char** instance_exts     = _skr_malloc(instance_ext_cap * sizeof(const char*));
	uint32_t     instance_ext_count = 0;

	// Add application-required extensions first
	for (uint32_t i = 0; i < settings.required_extension_count; i++) {
		if (_skr_ext_available(settings.required_extensions[i], available_inst_exts, available_inst_ext_count)) {
			instance_exts[instance_ext_count++] = settings.required_extensions[i];
		} else {
			skr_log(skr_log_critical, "Required instance extension '%s' not available", settings.required_extensions[i]);
			_skr_free(instance_exts);
			_skr_free(available_inst_exts);
			return false;
		}
	}

	// Add optional extensions if available
	bool has_surface = false;
	for (uint32_t i = 0; i < optional_instance_ext_count; i++) {
		// Skip debug utils if validation not enabled
		if (strcmp(optional_instance_exts[i], VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0 && !_skr_vk.validation_enabled) {
			continue;
		}
		if (_skr_ext_available(optional_instance_exts[i], available_inst_exts, available_inst_ext_count)) {
			instance_exts[instance_ext_count++] = optional_instance_exts[i];
			if (strcmp(optional_instance_exts[i], VK_KHR_SURFACE_EXTENSION_NAME) == 0) has_surface = true;
		}
	}

	// Requests missing an instance extension settle here; survivors contribute
	// their instance extensions and stay candidates for the device phase.
	for (int32_t r = 0; r < _skr_reg.req_count; r++) {
		_skr_req_t* req = &_skr_reg.reqs[r];
		req->enabled = true;
		req->missing = NULL;
		for (int32_t i = 0; i < req->inst_ext_count; i++) {
			if (!_skr_ext_available(_skr_reg.exts[req->inst_ext_start + i], available_inst_exts, available_inst_ext_count)) {
				req->enabled = false;
				req->missing = _skr_reg.exts[req->inst_ext_start + i];
				break;
			}
		}
		if (!req->enabled) {
			if (req->required) {
				skr_log(skr_log_critical, "Required request '%s': instance extension '%s' not available",
					req->name ? req->name : "(anonymous)", req->missing);
				_skr_free(instance_exts);
				_skr_free(available_inst_exts);
				return false;
			}
			continue;
		}
		for (int32_t i = 0; i < req->inst_ext_count; i++) {
			const char* name    = _skr_reg.exts[req->inst_ext_start + i];
			bool        present = false;
			for (uint32_t e = 0; e < instance_ext_count; e++)
				if (strcmp(instance_exts[e], name) == 0) { present = true; break; }
			if (!present) instance_exts[instance_ext_count++] = name;
		}
	}

	// Keep a copy of the final list for skr_vk_ext_enabled
	_skr_vk.enabled_instance_exts = _skr_malloc((instance_ext_count > 0 ? instance_ext_count : 1) * sizeof(char*));
	for (uint32_t i = 0; i < instance_ext_count; i++)
		_skr_vk.enabled_instance_exts[i] = _skr_strdup(instance_exts[i]);
	_skr_vk.enabled_instance_ext_count = instance_ext_count;

	_skr_free(available_inst_exts);

	// Build list of desired layers
	const char* desired_layers[8];
	uint32_t    desired_layer_count = 0;
	if (_skr_vk.validation_enabled) {
		desired_layers[desired_layer_count++] = "VK_LAYER_KHRONOS_validation";
	}

	// Get available layers
	uint32_t available_layer_count = 0;
	vkEnumerateInstanceLayerProperties(&available_layer_count, NULL);
	VkLayerProperties* available_layers = _skr_malloc(available_layer_count * sizeof(VkLayerProperties));
	vkEnumerateInstanceLayerProperties(&available_layer_count, available_layers);

	// Filter layers to only those available
	const char* layers[8];
	uint32_t    layer_count = 0;
	for (uint32_t i = 0; i < desired_layer_count; i++) {
		bool found = false;
		for (uint32_t j = 0; j < available_layer_count; j++) {
			if (strcmp(desired_layers[i], available_layers[j].layerName) == 0) {
				found = true;
				break;
			}
		}
		if (found) {
			layers[layer_count++] = desired_layers[i];
		} else {
			skr_log(skr_log_warning, "Layer '%s' not available, skipping", desired_layers[i]);
			if (strcmp(desired_layers[i], "VK_LAYER_KHRONOS_validation") == 0) _skr_vk.validation_enabled = false;
		}
	}
	_skr_free(available_layers);

	VkInstanceCreateInfo instance_info = {
		.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo        = &app_info,
		.enabledExtensionCount   = instance_ext_count,
		.ppEnabledExtensionNames = instance_exts,
		.enabledLayerCount       = layer_count,
		.ppEnabledLayerNames     = layers,
	};

	// Create VkInstance - either via callback (for OpenXR enable2) or directly
	if (settings.instance_create_callback) {
		skr_instance_create_info_t create_info = {
			.instance_create_info   = &instance_info,
			.get_instance_proc_addr = global_get_instance_proc_addr,
		};
		_skr_vk.instance = (VkInstance)settings.instance_create_callback(&create_info, settings.instance_create_user_data);
		if (_skr_vk.instance == VK_NULL_HANDLE) {
			skr_log(skr_log_critical, "Instance creation callback failed");
			_skr_free(instance_exts);
			return false;
		}
	} else {
		VkResult result = vkCreateInstance(&instance_info, NULL, &_skr_vk.instance);
		if (result != VK_SUCCESS) {
			skr_log(skr_log_critical, "Failed to create Vulkan instance: 0x%X", result);
			skr_log(skr_log_info,     "  Enabled extensions (%u):", instance_ext_count);
			for (uint32_t i = 0; i < instance_ext_count; i++)
				skr_log(skr_log_info, "    - %s", instance_exts[i]);
			if (layer_count > 0) {
				skr_log(skr_log_info, "  Enabled layers (%u):", layer_count);
				for (uint32_t i = 0; i < layer_count; i++) {
					skr_log(skr_log_info, "    - %s", layers[i]);
				}
			}
			skr_log(skr_log_info, "  Tip: If using RenderDoc, ensure it's launched with Vulkan support enabled");
			_skr_free(instance_exts);
			return false;
		}
	}
	_skr_free(instance_exts);

	volkLoadInstance(_skr_vk.instance);

	if (_skr_vk.validation_enabled) {
		if (!_skr_vk_create_debug_messenger(_skr_vk.instance, _skr_vk_debug_callback, &_skr_vk.debug_messenger )) {
			skr_log(skr_log_warning, "Failed to create debug messenger");
		} else {
			_skr_cmd_destroy_debug_messenger(&_skr_vk.destroy_list, _skr_vk.debug_messenger);
		}
	}

	// Call device initialization callback if provided (e.g., for OpenXR integration)
	// This allows external systems to query for physical device and device extensions
	// after VkInstance is available but before VkDevice is created.
	skr_device_request_t device_request = {0};
	if (settings.device_init_callback) {
		device_request = settings.device_init_callback(_skr_vk.instance, settings.device_init_user_data);
	}

	// Pick physical device - callback's choice takes precedence over settings
	VkPhysicalDevice requested_physical_device = device_request.physical_device
		? (VkPhysicalDevice)device_request.physical_device
		: (VkPhysicalDevice)settings.physical_device;

	if (requested_physical_device) {
		// Use the device specified by the application (e.g., from OpenXR)
		_skr_vk.physical_device = requested_physical_device;
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(_skr_vk.physical_device, &props);
		skr_log(skr_log_info, "Using application-specified GPU: %s", props.deviceName);
	} else {
		// Enumerate and select GPU based on require/prefer flags
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(_skr_vk.instance, &device_count, NULL);
		if (device_count == 0) {
			skr_log(skr_log_critical, "No Vulkan-compatible GPUs found");
			return false;
		}

		VkPhysicalDevice devices[32];
		vkEnumeratePhysicalDevices(_skr_vk.instance, &device_count, devices);

		// Score each device and find best match
		int32_t best_score = -1;
		int32_t best_idx   = -1;

		for (uint32_t i = 0; i < device_count; i++) {
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(devices[i], &props);

			// Determine device capabilities
			bool is_discrete   = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
			bool is_integrated = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
			bool has_video     = false;

			// Check for video decode support
			uint32_t ext_count = 0;
			vkEnumerateDeviceExtensionProperties(devices[i], NULL, &ext_count, NULL);
			if (ext_count > 0) {
				VkExtensionProperties* exts = _skr_malloc(sizeof(VkExtensionProperties) * ext_count);
				vkEnumerateDeviceExtensionProperties(devices[i], NULL, &ext_count, exts);

				uint32_t video_found = 0;
				for (uint32_t v = 0; v < video_device_ext_count; v++) {
					for (uint32_t e = 0; e < ext_count; e++) {
						if (strcmp(exts[e].extensionName, video_device_exts[v]) == 0) {
							video_found++;
							break;
						}
					}
				}
				_skr_free(exts);
				has_video = (video_found == video_device_ext_count);
			}

			// Check required flags - skip device if any required flag is missing
			skr_gpu_ req = settings.gpu_require;
			if ((req & skr_gpu_discrete)   && !is_discrete)   continue;
			if ((req & skr_gpu_integrated) && !is_integrated) continue;
			if ((req & skr_gpu_video)      && !has_video)     continue;

			// Calculate score based on preferred flags and device type
			int32_t score = 0;
			skr_gpu_ pref = settings.gpu_prefer;

			if (pref == skr_gpu_none) {
				// No preference: default to discrete > integrated
				if (is_discrete)   score += 1000;
				if (is_integrated) score += 100;
			} else {
				// Score based on matching preferred flags
				if ((pref & skr_gpu_discrete)   && is_discrete)   score += 1000;
				if ((pref & skr_gpu_integrated) && is_integrated) score += 1000;
				if ((pref & skr_gpu_video)      && has_video)     score += 500;
			}

			if (score > best_score) {
				best_score = score;
				best_idx   = (int32_t)i;
			}
		}

		if (best_idx < 0) {
			skr_log(skr_log_critical, "No GPU found matching required features (require=0x%X)", settings.gpu_require);
			return false;
		}

		_skr_vk.physical_device = devices[best_idx];
	}

	// Get device properties for timing and logging
	VkPhysicalDeviceProperties device_props;
	vkGetPhysicalDeviceProperties(_skr_vk.physical_device, &device_props);

	// Print selected device if we didn't find discrete GPU
	skr_log(skr_log_info, "Using GPU: %s", device_props.deviceName);

	// Store device limits
	_skr_vk.timestamp_period      = device_props.limits.timestampPeriod;
	_skr_vk.min_ubo_offset_align  = (uint32_t)device_props.limits.minUniformBufferOffsetAlignment;
	_skr_vk.min_ssbo_offset_align = (uint32_t)device_props.limits.minStorageBufferOffsetAlignment;

	// Calculate maximum supported MSAA sample count (intersection of color + depth)
	VkSampleCountFlags supported_samples =
		device_props.limits.framebufferColorSampleCounts &
		device_props.limits.framebufferDepthSampleCounts;

	_skr_vk.max_msaa_samples = 1;
	if      (supported_samples & VK_SAMPLE_COUNT_64_BIT) _skr_vk.max_msaa_samples = 64;
	else if (supported_samples & VK_SAMPLE_COUNT_32_BIT) _skr_vk.max_msaa_samples = 32;
	else if (supported_samples & VK_SAMPLE_COUNT_16_BIT) _skr_vk.max_msaa_samples = 16;
	else if (supported_samples & VK_SAMPLE_COUNT_8_BIT)  _skr_vk.max_msaa_samples = 8;
	else if (supported_samples & VK_SAMPLE_COUNT_4_BIT)  _skr_vk.max_msaa_samples = 4;
	else if (supported_samples & VK_SAMPLE_COUNT_2_BIT)  _skr_vk.max_msaa_samples = 2;

	// Find queue families
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(_skr_vk.physical_device, &queue_family_count, NULL);

	VkQueueFamilyProperties queue_families[32];
	vkGetPhysicalDeviceQueueFamilyProperties(_skr_vk.physical_device, &queue_family_count, queue_families);

	// Find graphics queue family
	_skr_vk.graphics_queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < queue_family_count; i++) {
		if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			_skr_vk.graphics_queue_family = i;
			_skr_vk.present_queue_family  = i; // Assume same for now
			break;
		}
	}

	if (_skr_vk.graphics_queue_family == UINT32_MAX) {
		skr_log(skr_log_critical, "Failed to find graphics queue family");
		return false;
	}

	// Find dedicated transfer queue (TRANSFER but not GRAPHICS)
	_skr_vk.transfer_queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < queue_family_count; i++) {
		if ((queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			!(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			_skr_vk.transfer_queue_family = i;
			_skr_vk.has_dedicated_transfer = true;
			break;
		}
	}

	// Fall back to graphics queue for transfers
	if (_skr_vk.transfer_queue_family == UINT32_MAX) {
		_skr_vk.transfer_queue_family = _skr_vk.graphics_queue_family;
		_skr_vk.has_dedicated_transfer = false;
	}

	// Find video decode queue family (requires VK_QUEUE_VIDEO_DECODE_BIT_KHR = 0x20)
	_skr_vk.video_decode_queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < queue_family_count; i++) {
		if (queue_families[i].queueFlags & 0x00000020) {  // VK_QUEUE_VIDEO_DECODE_BIT_KHR
			_skr_vk.video_decode_queue_family = i;
			break;
		}
	}

	// Create queue create infos
	float queue_priority = 1.0f;
	VkDeviceQueueCreateInfo queue_infos[3];
	uint32_t queue_info_count = 0;

	// Always create graphics queue
	queue_infos[queue_info_count++] = (VkDeviceQueueCreateInfo){
		.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = _skr_vk.graphics_queue_family,
		.queueCount       = 1,
		.pQueuePriorities = &queue_priority,
	};

	// Create dedicated transfer queue if available
	if (_skr_vk.has_dedicated_transfer) {
		queue_infos[queue_info_count++] = (VkDeviceQueueCreateInfo){
			.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = _skr_vk.transfer_queue_family,
			.queueCount       = 1,
			.pQueuePriorities = &queue_priority,
		};
	}

	// Create video decode queue if available and different from existing queues
	bool need_video_decode_queue = _skr_vk.video_decode_queue_family != UINT32_MAX &&
	                               _skr_vk.video_decode_queue_family != _skr_vk.graphics_queue_family &&
	                               _skr_vk.video_decode_queue_family != _skr_vk.transfer_queue_family;
	if (need_video_decode_queue) {
		queue_infos[queue_info_count++] = (VkDeviceQueueCreateInfo){
			.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = _skr_vk.video_decode_queue_family,
			.queueCount       = 1,
			.pQueuePriorities = &queue_priority,
		};
	}

	///////////////////////////////////////////////////////////////////////////
	// Device creation
	///////////////////////////////////////////////////////////////////////////

	// Get available device extensions
	uint32_t available_device_ext_count = 0;
	vkEnumerateDeviceExtensionProperties(_skr_vk.physical_device, NULL, &available_device_ext_count, NULL);
	VkExtensionProperties* available_device_exts = _skr_malloc(available_device_ext_count * sizeof(VkExtensionProperties));
	vkEnumerateDeviceExtensionProperties(_skr_vk.physical_device, NULL, &available_device_ext_count, available_device_exts);

	// Build final device extension list, sized for every possible source
	uint32_t device_ext_cap = device_request.required_device_extension_count + video_device_ext_count;
	for (int32_t r = 0; r < _skr_reg.req_count; r++)
		device_ext_cap += (uint32_t)_skr_reg.reqs[r].dev_ext_count;
	const char** device_exts     = _skr_malloc(device_ext_cap * sizeof(const char*));
	uint32_t     device_ext_count = 0;

	// Add device extensions from callback (e.g., from OpenXR)
	for (uint32_t i = 0; i < device_request.required_device_extension_count; i++) {
		if (_skr_ext_available(device_request.required_device_extensions[i], available_device_exts, available_device_ext_count)) {
			device_exts[device_ext_count++] = device_request.required_device_extensions[i];
		} else {
			skr_log(skr_log_critical, "Required device extension '%s' not available", device_request.required_device_extensions[i]);
			_skr_free(device_exts);
			_skr_free(available_device_exts);
			return false;
		}
	}

	// Settle each request's device-extension availability. Survivors are
	// candidates: their feature structs get queried before anything enables.
	for (int32_t r = 0; r < _skr_reg.req_count; r++) {
		_skr_req_t* req = &_skr_reg.reqs[r];
		if (!req->enabled) continue; // Already failed at the instance phase
		for (int32_t i = 0; i < req->dev_ext_count; i++) {
			if (!_skr_ext_available(_skr_reg.exts[req->dev_ext_start + i], available_device_exts, available_device_ext_count)) {
				req->enabled = false;
				req->missing = _skr_reg.exts[req->dev_ext_start + i];
				break;
			}
		}
	}

	// Query support for candidates' feature structs, merged by sType so shared
	// dependencies (e.g. sync2) chain once. Candidate exts were verified above.
	typedef struct {
		VkStructureType     stype;
		int32_t             size;
		VkBaseOutStructure* data;
	} _skr_feat_query_t;
	// Unique sTypes can't outnumber the total feature struct references
	_skr_feat_query_t* feat_queries = _skr_reg.feat_count > 0
		? _skr_malloc((uint32_t)_skr_reg.feat_count * sizeof(_skr_feat_query_t)) : NULL;
	int32_t feat_query_count = 0;
	int32_t feat_query_bytes = 0;
	for (int32_t r = 0; r < _skr_reg.req_count; r++) {
		if (!_skr_reg.reqs[r].enabled) continue;
		for (int32_t i = 0; i < _skr_reg.reqs[r].feat_count; i++) {
			const _skr_req_feat_t*   feat = &_skr_reg.feats[_skr_reg.reqs[r].feat_start + i];
			const VkBaseInStructure* s    = _skr_req_feat_struct(feat);
			int32_t q = 0;
			while (q < feat_query_count && feat_queries[q].stype != s->sType) q++;
			if (q == feat_query_count) feat_queries[feat_query_count++] = (_skr_feat_query_t){ s->sType, 0, NULL };
			if (feat->size > feat_queries[q].size) feat_queries[q].size = feat->size;
		}
	}
	for (int32_t q = 0; q < feat_query_count; q++)
		feat_query_bytes += (feat_queries[q].size + 7) & ~7;

	uint8_t* feat_query_mem = NULL;
	if (feat_query_count > 0) {
		feat_query_mem = _skr_malloc(feat_query_bytes);
		memset(feat_query_mem, 0, feat_query_bytes);
		VkPhysicalDeviceFeatures2 features2_query = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
		int32_t offset = 0;
		for (int32_t q = 0; q < feat_query_count; q++) {
			feat_queries[q].data        = (VkBaseOutStructure*)(feat_query_mem + offset);
			feat_queries[q].data->sType = feat_queries[q].stype;
			feat_queries[q].data->pNext = (VkBaseOutStructure*)features2_query.pNext;
			features2_query.pNext       = feat_queries[q].data;
			offset += (feat_queries[q].size + 7) & ~7;
		}
		vkGetPhysicalDeviceFeatures2(_skr_vk.physical_device, &features2_query);
	}

	// A candidate enables only if every bit it asked for is supported. Structs
	// are {sType, pNext, VkBool32...}, so compare dwords; only VK_TRUE counts.
	const int32_t feat_header = (int32_t)sizeof(VkBaseInStructure);
	for (int32_t r = 0; r < _skr_reg.req_count; r++) {
		_skr_req_t* req = &_skr_reg.reqs[r];
		for (int32_t i = 0; req->enabled && i < req->feat_count; i++) {
			const _skr_req_feat_t*   feat = &_skr_reg.feats[req->feat_start + i];
			const VkBaseInStructure* want = _skr_req_feat_struct(feat);
			int32_t q = 0;
			while (feat_queries[q].stype != want->sType) q++;
			const uint32_t* want_bits = (const uint32_t*)((const uint8_t*)want                 + feat_header);
			const uint32_t* have_bits = (const uint32_t*)((const uint8_t*)feat_queries[q].data + feat_header);
			for (int32_t b = 0; b < (feat->size - feat_header) / 4; b++) {
				if (want_bits[b] == VK_TRUE && have_bits[b] != VK_TRUE) {
					snprintf(req->missing_feature, sizeof(req->missing_feature), "feature bit %d of sType %u", b, (uint32_t)want->sType);
					req->missing = req->missing_feature;
					req->enabled = false;
					break;
				}
			}
		}
		if (!req->enabled && req->required) {
			skr_log(skr_log_critical, "Required request '%s' not satisfied by the selected GPU (missing %s)",
				req->name ? req->name : "(anonymous)", req->missing ? req->missing : "?");
			if (feat_queries)   _skr_free(feat_queries);
			if (feat_query_mem) _skr_free(feat_query_mem);
			_skr_free(device_exts);
			_skr_free(available_device_exts);
			return false;
		}
	}

	// Enabled requests contribute their device extensions, deduped. A failed
	// request keeps all its extensions out.
	for (int32_t r = 0; r < _skr_reg.req_count; r++) {
		_skr_req_t* req = &_skr_reg.reqs[r];
		if (!req->enabled) continue;
		for (int32_t i = 0; i < req->dev_ext_count; i++) {
			const char* name    = _skr_reg.exts[req->dev_ext_start + i];
			bool        present = false;
			for (uint32_t e = 0; e < device_ext_count; e++)
				if (strcmp(device_exts[e], name) == 0) { present = true; break; }
			if (!present) device_exts[device_ext_count++] = name;
		}
	}

	// Video decode is all-or-nothing and also gated on a decode queue family,
	// which a request can't express, so it stays special-cased.
	_skr_vk.has_video_decode = false;
	bool has_synchronization2 = skr_vk_request_enabled("sync2");
	if (_skr_vk.video_decode_queue_family != UINT32_MAX && has_synchronization2) {
		uint32_t video_found = 0;
		for (uint32_t v = 0; v < video_device_ext_count; v++) {
			if (_skr_ext_available(video_device_exts[v], available_device_exts, available_device_ext_count))
				video_found++;
		}
		if (video_found == video_device_ext_count) {
			for (uint32_t v = 0; v < video_device_ext_count; v++) {
				bool present = false; // An app request may already list one (e.g. timeline semaphore)
				for (uint32_t e = 0; e < device_ext_count; e++)
					if (strcmp(device_exts[e], video_device_exts[v]) == 0) { present = true; break; }
				if (!present) device_exts[device_ext_count++] = video_device_exts[v];
			}
			_skr_vk.has_video_decode = true;
			skr_log(skr_log_info, "Vulkan video decode extensions enabled");
		}
	}

	// Keep a copy of the final list for skr_vk_ext_enabled
	_skr_vk.enabled_device_exts = _skr_malloc((device_ext_count > 0 ? device_ext_count : 1) * sizeof(char*));
	for (uint32_t i = 0; i < device_ext_count; i++)
		_skr_vk.enabled_device_exts[i] = _skr_strdup(device_exts[i]);
	_skr_vk.enabled_device_ext_count = device_ext_count;

	_skr_free(available_device_exts);

	// Request-derived feature flags
	_skr_vk.has_push_descriptors        = skr_vk_request_enabled(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
	_skr_vk.has_external_memory_fd      = skr_vk_request_enabled(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
	_skr_vk.has_external_memory_dma_buf = skr_vk_request_enabled(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
	_skr_vk.has_drm_format_modifier     = skr_vk_request_enabled(VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME);
	_skr_vk.has_custom_resolve          = skr_vk_request_enabled(VK_QCOM_RENDER_PASS_SHADER_RESOLVE_EXTENSION_NAME);
	_skr_vk.has_external_fence_fd       = skr_vk_request_enabled(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
	_skr_vk.has_create_renderpass2      = skr_vk_request_enabled(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
	_skr_vk.has_depth_stencil_resolve   = skr_vk_request_enabled("depth_stencil_resolve");
	_skr_vk.has_present_fence           = skr_vk_request_enabled("swapchain_maintenance1");
	_skr_vk.has_subpass_merge_feedback  = skr_vk_request_enabled("subpass_merge_feedback");
	_skr_vk.has_subgroup_size_control   = skr_vk_request_enabled("subgroup_size_control");
	_skr_vk.has_ycbcr_conversion        = skr_vk_request_enabled("ycbcr_conversion");
	_skr_vk.has_qcom_tile_shading       = skr_vk_request_enabled("qcom_tile_shading");
	_skr_vk.has_store_op_none           = skr_vk_request_enabled(VK_KHR_LOAD_STORE_OP_NONE_EXTENSION_NAME)
	                                   || skr_vk_request_enabled(VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME)
	                                   || skr_vk_request_enabled(VK_QCOM_RENDER_PASS_STORE_OPS_EXTENSION_NAME);
	_skr_vk.has_external_memory_win32   = false;
	_skr_vk.has_android_hardware_buffer = false;
#ifdef VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
	_skr_vk.has_external_memory_win32   = skr_vk_request_enabled(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#endif
#ifdef VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME
	_skr_vk.has_android_hardware_buffer = skr_vk_request_enabled(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
#endif
	bool has_swapchain          = skr_vk_request_enabled(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	bool has_image_format_list  = skr_vk_request_enabled(VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME);
	bool has_output_layer_ext   = skr_vk_request_enabled(VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME);
	bool enable_sample_weighted = skr_vk_request_enabled("qcom_sample_weighted");
	bool enable_box_filter      = skr_vk_request_enabled("qcom_box_filter");
	bool enable_block_match     = skr_vk_request_enabled("qcom_block_match");
	bool enable_block_match2    = skr_vk_request_enabled("qcom_block_match2");
	// Any enabled op means image-processing bindings can occur, hence the sampler
	_skr_vk.has_qcom_image_proc = enable_sample_weighted || enable_box_filter || enable_block_match;
	bool enable_storage16       = skr_vk_request_enabled("storage16");
	bool enable_atomic_float    = skr_vk_request_enabled("float_atomics");

	// Query available device features
	VkPhysicalDeviceFeatures available_features;
	vkGetPhysicalDeviceFeatures(_skr_vk.physical_device, &available_features);

	// Track feature availability
	_skr_vk.has_depth_clamp            = available_features.depthClamp;
	_skr_vk.has_fill_mode_non_solid    = available_features.fillModeNonSolid;
	_skr_vk.has_storage_without_format = available_features.shaderStorageImageWriteWithoutFormat
	                                  && available_features.shaderStorageImageReadWithoutFormat;

	// Enable features we need (only if available)
	VkPhysicalDeviceFeatures device_features = {
		.samplerAnisotropy              = available_features.samplerAnisotropy,
		.sampleRateShading              = VK_FALSE, // Not using sample shading yet
		.fillModeNonSolid               = available_features.fillModeNonSolid,
		.depthClamp                     = available_features.depthClamp,
		.vertexPipelineStoresAndAtomics = available_features.vertexPipelineStoresAndAtomics,
		.fragmentStoresAndAtomics       = available_features.fragmentStoresAndAtomics,
		// shader-feature gates (sksc_feature_bit_*): enabled when available so
		// the support mask below can advertise them
		.geometryShader                  = available_features.geometryShader,
		.shaderStorageImageExtendedFormats = available_features.shaderStorageImageExtendedFormats,
		.shaderInt16                     = available_features.shaderInt16,
		// sksc compiles RWTextures with SPIR-V format Unknown (DXC-style), so
		// storage image access is gated on the WithoutFormat pair.
		.shaderStorageImageWriteWithoutFormat = available_features.shaderStorageImageWriteWithoutFormat,
		.shaderStorageImageReadWithoutFormat  = available_features.shaderStorageImageReadWithoutFormat,
	};

	// Query multiview properties (and subgroup size limits if supported)
	VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_props = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT,
	};
	VkPhysicalDeviceMultiviewProperties multiview_props = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES,
		.pNext = _skr_vk.has_subgroup_size_control ? &subgroup_props : NULL,
	};
	VkPhysicalDeviceProperties2 props2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &multiview_props,
	};
	VkPhysicalDeviceTileShadingPropertiesQCOM tile_shading_props = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_PROPERTIES_QCOM,
	};
	if (_skr_vk.has_qcom_tile_shading) {
		tile_shading_props.pNext = props2.pNext;
		props2.pNext             = &tile_shading_props;
	}
	vkGetPhysicalDeviceProperties2(_skr_vk.physical_device, &props2);
	_skr_vk.max_tile_apron = _skr_vk.has_qcom_tile_shading ? tile_shading_props.maxApronSize : 0;
	_skr_vk.max_multiview_view_count = multiview_props.maxMultiviewViewCount;
	if (_skr_vk.has_subgroup_size_control) {
		_skr_vk.min_subgroup_size             = subgroup_props.minSubgroupSize;
		_skr_vk.max_subgroup_size             = subgroup_props.maxSubgroupSize;
		_skr_vk.required_subgroup_size_stages = subgroup_props.requiredSubgroupSizeStages;
	}

	VkDeviceCreateInfo device_info = {
		.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount    = queue_info_count,
		.pQueueCreateInfos       = queue_infos,
		.enabledExtensionCount   = device_ext_count,
		.ppEnabledExtensionNames = device_exts,
		.pEnabledFeatures        = &device_features,
	};

	// Reuse the query structs as the enable chain: zero the bools, OR in each
	// enabled request's bits, and chain every struct that received any.
	for (int32_t q = 0; q < feat_query_count; q++)
		memset((uint8_t*)feat_queries[q].data + feat_header, 0, feat_queries[q].size - feat_header);
	for (int32_t r = 0; r < _skr_reg.req_count; r++) {
		if (!_skr_reg.reqs[r].enabled) continue;
		for (int32_t i = 0; i < _skr_reg.reqs[r].feat_count; i++) {
			const _skr_req_feat_t*   feat = &_skr_reg.feats[_skr_reg.reqs[r].feat_start + i];
			const VkBaseInStructure* want = _skr_req_feat_struct(feat);
			int32_t q = 0;
			while (feat_queries[q].stype != want->sType) q++;
			const uint32_t* want_bits = (const uint32_t*)((const uint8_t*)want                 + feat_header);
			uint32_t*       dst_bits  = (uint32_t*)      ((uint8_t*)      feat_queries[q].data + feat_header);
			for (int32_t b = 0; b < (feat->size - feat_header) / 4; b++)
				if (want_bits[b] == VK_TRUE) dst_bits[b] = VK_TRUE;
		}
	}
	for (int32_t q = 0; q < feat_query_count; q++) {
		const uint32_t* bits = (const uint32_t*)((const uint8_t*)feat_queries[q].data + feat_header);
		bool            any  = false;
		for (int32_t b = 0; b < (feat_queries[q].size - feat_header) / 4 && !any; b++)
			any = bits[b] == VK_TRUE;
		if (!any) continue;
		feat_queries[q].data->pNext = (VkBaseOutStructure*)device_info.pNext;
		device_info.pNext           = feat_queries[q].data;
	}

	// Create VkDevice - either via callback (for OpenXR enable2) or directly
	if (settings.device_create_callback) {
		skr_device_create_info_t create_info = {
			.vk_physical_device     = _skr_vk.physical_device,
			.device_create_info     = &device_info,
			.get_instance_proc_addr = global_get_instance_proc_addr,
		};
		_skr_vk.device = (VkDevice)settings.device_create_callback(&create_info, settings.device_create_user_data);
		if (_skr_vk.device == VK_NULL_HANDLE)
			skr_log(skr_log_critical, "Device creation callback failed");
	} else {
		vr = vkCreateDevice(_skr_vk.physical_device, &device_info, NULL, &_skr_vk.device);
		SKR_VK_CHECK_NRET(vr, "vkCreateDevice");
	}

	// The extension list and feature chain have been consumed by device creation
	_skr_free(device_exts);
	if (feat_queries)   _skr_free(feat_queries);
	if (feat_query_mem) _skr_free(feat_query_mem);
	if (_skr_vk.device == VK_NULL_HANDLE) return false;

	volkLoadDevice(_skr_vk.device);

	// Get graphics queue
	vkGetDeviceQueue(_skr_vk.device, _skr_vk.graphics_queue_family, 0, &_skr_vk.graphics_queue);
	_skr_vk.present_queue = _skr_vk.graphics_queue;

	// Get transfer queue (same as graphics if no dedicated queue)
	if (_skr_vk.has_dedicated_transfer) {
		vkGetDeviceQueue(_skr_vk.device, _skr_vk.transfer_queue_family, 0, &_skr_vk.transfer_queue);
	} else {
		_skr_vk.transfer_queue = _skr_vk.graphics_queue;
	}

	// Initialize queue mutexes for thread-safe queue submission
	// We use 4 slots but may only need 1-3 if queues are aliased
	for (int32_t i = 0; i < SKR_QUEUE_TYPE_COUNT; i++) mtx_init(&_skr_vk.queue_mutexes[i], mtx_plain);
	mtx_init(&_skr_vk.thread_pool_mutex, mtx_plain);

	// Set up mutex pointers based on queue aliasing
	_skr_vk.graphics_queue_mutex = &_skr_vk.queue_mutexes[0];

	// Present always aliases graphics
	_skr_vk.present_queue_mutex = &_skr_vk.queue_mutexes[0];

	// Transfer uses dedicated mutex if it has a dedicated queue, otherwise aliases graphics
	if (_skr_vk.has_dedicated_transfer) {
		_skr_vk.transfer_queue_mutex = &_skr_vk.queue_mutexes[2];
	} else {
		_skr_vk.transfer_queue_mutex = &_skr_vk.queue_mutexes[0];
	}

	// Video decode: dedicated mutex if separate family, otherwise alias the matching one
	if (_skr_vk.video_decode_queue_family == UINT32_MAX) {
		_skr_vk.video_decode_queue_mutex = NULL;
	} else if (_skr_vk.video_decode_queue_family == _skr_vk.graphics_queue_family) {
		_skr_vk.video_decode_queue_mutex = _skr_vk.graphics_queue_mutex;
	} else if (_skr_vk.video_decode_queue_family == _skr_vk.transfer_queue_family) {
		_skr_vk.video_decode_queue_mutex = _skr_vk.transfer_queue_mutex;
	} else {
		_skr_vk.video_decode_queue_mutex = &_skr_vk.queue_mutexes[3];
	}

	// Create command pool
	VkCommandPoolCreateInfo pool_info = {
		.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = _skr_vk.graphics_queue_family,
	};

	vr = vkCreateCommandPool(_skr_vk.device, &pool_info, NULL, &_skr_vk.command_pool);
	SKR_VK_CHECK_RET(vr, "vkCreateCommandPool", false);
	_skr_cmd_destroy_command_pool(&_skr_vk.destroy_list, _skr_vk.command_pool);

	// Allocate command buffers (one per frame in flight)
	VkCommandBufferAllocateInfo alloc_info = {
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool        = _skr_vk.command_pool,
		.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = SKR_MAX_FRAMES_IN_FLIGHT,
	};

	vr = vkAllocateCommandBuffers(_skr_vk.device, &alloc_info, _skr_vk.command_buffers);
	SKR_VK_CHECK_RET(vr, "vkAllocateCommandBuffers", false);

	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		vr = vkCreateFence(_skr_vk.device, &(VkFenceCreateInfo){
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT, // Start signaled so first frame doesn't wait
		}, NULL, &_skr_vk.frame_fences[i]);
		SKR_VK_CHECK_RET(vr, "vkCreateFence", false);
		_skr_cmd_destroy_fence(&_skr_vk.destroy_list, _skr_vk.frame_fences[i]);
	}

	vr = vkCreateQueryPool(_skr_vk.device, &(VkQueryPoolCreateInfo){
		.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType  = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = 2 * SKR_MAX_FRAMES_IN_FLIGHT,
	}, NULL, &_skr_vk.timestamp_pool);
	SKR_VK_CHECK_RET(vr, "vkCreateQueryPool", false);
	_skr_cmd_destroy_query_pool(&_skr_vk.destroy_list, _skr_vk.timestamp_pool);

	for (uint32_t i = 0; i < SKR_MAX_FRAMES_IN_FLIGHT; i++) {
		_skr_vk.timestamps_valid[i] = false;
	}

	vr = vkCreatePipelineCache(_skr_vk.device, &(VkPipelineCacheCreateInfo){
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
	}, NULL, &_skr_vk.pipeline_cache);
	SKR_VK_CHECK_RET(vr, "vkCreatePipelineCache", false);
	_skr_cmd_destroy_pipeline_cache(&_skr_vk.destroy_list, _skr_vk.pipeline_cache);

	// Create descriptor pool for compute shaders
	VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount = 1000 },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          .descriptorCount = 1000 },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1000 },
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         .descriptorCount = 1000 },
	};

	VkDescriptorPoolCreateInfo desc_pool_info = {
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets       = 1000,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes    = pool_sizes,
	};

	vr = vkCreateDescriptorPool(_skr_vk.device, &desc_pool_info, NULL, &_skr_vk.descriptor_pool);
	SKR_VK_CHECK_RET(vr, "vkCreateDescriptorPool", false);
	_skr_cmd_destroy_descriptor_pool(&_skr_vk.destroy_list, _skr_vk.descriptor_pool);

	_skr_pipeline_init();

	if (!_skr_cmd_init()) {
		skr_log(skr_log_critical, "Failed to initialize upload system");
		return false;
	}

	// Initialize main thread
	skr_thread_init();

	const skr_tex_sampler_t sampler = {
		.sample  = skr_tex_sample_linear,
		.address = skr_tex_address_clamp
	};
	uint32_t color = 0xFFFFFFFF;
	skr_tex_create( skr_tex_fmt_rgba32, skr_tex_flags_readable, sampler, (skr_vec3i_t){1, 1, 1}, 1, 1, &(skr_tex_data_t){.data = &color, .mip_count = 1, .layer_count = 1}, &_skr_vk.default_tex_white);
	color = 0xFF808080;
	skr_tex_create( skr_tex_fmt_rgba32, skr_tex_flags_readable, sampler, (skr_vec3i_t){1, 1, 1}, 1, 1, &(skr_tex_data_t){.data = &color, .mip_count = 1, .layer_count = 1}, &_skr_vk.default_tex_gray);
	color = 0xFF000000;
	skr_tex_create( skr_tex_fmt_rgba32, skr_tex_flags_readable, sampler, (skr_vec3i_t){1, 1, 1}, 1, 1, &(skr_tex_data_t){.data = &color, .mip_count = 1, .layer_count = 1}, &_skr_vk.default_tex_black);

	// Populate capability array
	_skr_vk.capabilities[skr_capability_external_vk ] = true;
	_skr_vk.capabilities[skr_capability_external_gl ] = _skr_vk.has_external_memory_fd || _skr_vk.has_external_memory_win32;
	_skr_vk.capabilities[skr_capability_external_ahb] = _skr_vk.has_android_hardware_buffer;
	_skr_vk.capabilities[skr_capability_external_dma] = _skr_vk.has_external_memory_dma_buf && _skr_vk.has_drm_format_modifier && has_image_format_list;
	_skr_vk.capabilities[skr_capability_vk_video    ] = _skr_vk.has_video_decode && _skr_vk.has_ycbcr_conversion;
	_skr_vk.capabilities[skr_capability_presentation] = has_surface && has_swapchain;

	// Build the shader-support mask: one bit per sksc_feature_bit_ that this
	// device actually enabled above, so skr_shader_check_support can reject a
	// shader before it reaches a material/pipeline. A bit left clear means the
	// feature simply isn't enabled here yet — enable it in the device_features /
	// pNext chain above, then set its bit here. sksc_feature_bit_unknown is
	// deliberately never set: a shader that needs an unclassified capability
	// can't be verified, so it must fail the check.
	_skr_vk.enabled_features = 0;
	// Multiview is a hard requirement enforced above, and basic subgroup ops are
	// core in Vulkan 1.1 (the floor multiview implies). The `subgroups` bit is
	// coarse — it does not distinguish op classes (clustered/quad/…) that some
	// devices lack — but the common case is covered.
	_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_multiview;
	_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_subgroups;
	// Storage-image atomics are gated by the bound image's VkFormat at draw
	// time, not by a device feature, so this never blocks pipeline creation.
	_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_image_atomics;
	if (_skr_vk.has_subgroup_size_control)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_wave_size;
	// The bit is joint read+write; both features are near-universal together.
	if (_skr_vk.has_storage_without_format)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_formatless;
	if (enable_sample_weighted)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_qcom_sample_weighted;
	if (enable_box_filter)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_qcom_box_filter;
	if (enable_block_match)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_qcom_block_match;
	if (enable_block_match2)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_qcom_image_proc2;
	if (_skr_vk.has_qcom_tile_shading)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_qcom_tile_shading;
	// The extension has no feature struct — its presence is the feature. Note
	// Vulkan forbids writing Layer inside a multiview render pass; that is a
	// per-pass rule the legacy instanced-stereo path already respects.
	if (has_output_layer_ext)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_output_layer;
	if (device_features.geometryShader)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_geometry;
	if (device_features.shaderStorageImageExtendedFormats)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_extended_formats;
	if (_skr_vk.has_storage_without_format) // Read/write to unknown format textures in SPIRV
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_formatless;
	if (device_features.shaderInt16)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_int16;
	if (enable_storage16)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_storage16;
	if (enable_atomic_float)
		_skr_vk.enabled_features |= (uint64_t)1 << sksc_feature_bit_float_atomics;

	// QCOM image-processing ops (BoxFilterQCOM etc.) are only legal with a
	// sampler created with the IMAGE_PROCESSING flag, and such a sampler's
	// other creation parameters are pinned by the spec (nearest filtering,
	// clamp-to-edge, lod 0, no anisotropy/compare) — the ops define their own
	// filtering. So one shared sampler covers every image-processing binding;
	// _skr_material_add_writes substitutes it when a binding's meta shape
	// declares bit 6.
	_skr_vk.sampler_image_proc = VK_NULL_HANDLE;
	if (_skr_vk.has_qcom_image_proc) {
		VkSamplerCreateInfo image_proc_sampler_info = {
			.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.flags        = VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM,
			.magFilter    = VK_FILTER_NEAREST,
			.minFilter    = VK_FILTER_NEAREST,
			.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};
		VkResult sampler_vr = vkCreateSampler(_skr_vk.device, &image_proc_sampler_info, NULL, &_skr_vk.sampler_image_proc);
		if (sampler_vr == VK_SUCCESS) {
			_skr_set_debug_name(_skr_vk.device, VK_OBJECT_TYPE_SAMPLER, (uint64_t)_skr_vk.sampler_image_proc, "sampler_image_processing_qcom");
		} else {
			skr_log(skr_log_warning, "vkCreateSampler (QCOM image processing) failed: 0x%X", (uint32_t)sampler_vr);
			_skr_vk.sampler_image_proc  = VK_NULL_HANDLE;
			_skr_vk.has_qcom_image_proc = false;
			_skr_vk.enabled_features   &= ~(((uint64_t)1 << sksc_feature_bit_qcom_sample_weighted)
			                              | ((uint64_t)1 << sksc_feature_bit_qcom_box_filter)
			                              | ((uint64_t)1 << sksc_feature_bit_qcom_block_match)
			                              | ((uint64_t)1 << sksc_feature_bit_qcom_image_proc2));
		}
	}

	// Built-in fallback mipgen shaders. Used by skr_tex_generate_mips when the
	// caller passes no shader and the texture format doesn't support blit
	// (e.g. B10G11R11_UFLOAT on Mesa llvmpipe). Created only after
	// enabled_features exists: skr_shader_create gates on that mask, and an
	// empty mask would refuse these (mipgen_cube needs multiview).
	if (skr_shader_create(sks_skr_mipgen_2d_hlsl,   sizeof(sks_skr_mipgen_2d_hlsl),   &_skr_vk.builtin_mipgen_2d  ) == skr_err_success) skr_shader_set_name(&_skr_vk.builtin_mipgen_2d,   "skr_builtin_mipgen_2d");
	if (skr_shader_create(sks_skr_mipgen_cube_hlsl, sizeof(sks_skr_mipgen_cube_hlsl), &_skr_vk.builtin_mipgen_cube) == skr_err_success) skr_shader_set_name(&_skr_vk.builtin_mipgen_cube, "skr_builtin_mipgen_cube");

	// Log request outcomes, including which piece a disabled request missed,
	// so a device log answers "why" by itself
	for (int32_t r = 0; r < _skr_reg.req_count; r++) {
		const _skr_req_t* req  = &_skr_reg.reqs[r];
		const char*       name = req->name ? req->name : "(anonymous)";
		if (req->enabled) skr_log(skr_log_info, "[%s] true", name);
		else              skr_log(skr_log_info, "[%s] false (missing %s)", name, req->missing ? req->missing : "?");
	}
	if (_skr_vk.has_qcom_tile_shading)
		skr_log(skr_log_info, "[%s] max apron %u",  VK_QCOM_TILE_SHADING_EXTENSION_NAME, _skr_vk.max_tile_apron);
	skr_log(skr_log_info, "[%s] max %u views", VK_KHR_MULTIVIEW_EXTENSION_NAME,     _skr_vk.max_multiview_view_count);

	_skr_vk.initialized = true;
	return true;
}

// vkDeviceWaitIdle implicitly uses every queue, so it needs the same external
// synchronization a submit does — take every queue mutex (aliased pointers all
// resolve into this array) so worker-thread submits can't overlap the wait
void _skr_device_wait_idle(void) {
	for (int32_t i = 0; i < SKR_QUEUE_TYPE_COUNT; i++) mtx_lock  (&_skr_vk.queue_mutexes[i]);
	vkDeviceWaitIdle(_skr_vk.device);
	for (int32_t i = 0; i < SKR_QUEUE_TYPE_COUNT; i++) mtx_unlock(&_skr_vk.queue_mutexes[i]);
}

void skr_shutdown(void) {
	if (!_skr_vk.initialized) return;

	_skr_device_wait_idle();

	skr_tex_destroy(&_skr_vk.default_tex_white);
	skr_tex_destroy(&_skr_vk.default_tex_gray);
	skr_tex_destroy(&_skr_vk.default_tex_black);

	skr_shader_destroy(&_skr_vk.builtin_mipgen_2d);
	skr_shader_destroy(&_skr_vk.builtin_mipgen_cube);

	_skr_mipgen_materials_shutdown(); // Builtin entries died with their shaders above; this drops app-shader ones
	_skr_scratch_pool_shutdown();   // Free pooled mipgen scratch textures before command shutdown
	_skr_transient_pool_shutdown(); // Free pooled transient postfx attachments before command shutdown

	_skr_cmd_shutdown      ();  // Executes per-command destroy lists (may free bind pool slots)
	_skr_pipeline_shutdown ();

	_skr_destroy_list_execute(&_skr_vk.destroy_list);  // Execute global destroy list
	_skr_destroy_list_free   (&_skr_vk.destroy_list);

	_skr_bind_pool_shutdown();     // Free bind pool after all deferred destroys are done
	_skr_sampler_cache_shutdown(); // Destroy cached samplers after GPU is idle
	if (_skr_vk.sampler_image_proc != VK_NULL_HANDLE) vkDestroySampler(_skr_vk.device, _skr_vk.sampler_image_proc, NULL);

	// Free dynamic arrays
	if (_skr_vk.pending_transitions)      _skr_free(_skr_vk.pending_transitions);
	if (_skr_vk.pending_transition_types) _skr_free(_skr_vk.pending_transition_types);
	for (uint32_t i = 0; i < _skr_vk.enabled_instance_ext_count; i++) _skr_free(_skr_vk.enabled_instance_exts[i]);
	for (uint32_t i = 0; i < _skr_vk.enabled_device_ext_count;   i++) _skr_free(_skr_vk.enabled_device_exts  [i]);
	if (_skr_vk.enabled_instance_exts) _skr_free(_skr_vk.enabled_instance_exts);
	if (_skr_vk.enabled_device_exts)   _skr_free(_skr_vk.enabled_device_exts);

	// Destroy mutexes
	for (int32_t i = 0; i < SKR_QUEUE_TYPE_COUNT; i++) mtx_destroy(&_skr_vk.queue_mutexes[i]);
	mtx_destroy(&_skr_vk.thread_pool_mutex);

	// Destroy device and instance directly (special cases not in destroy list)
	if (_skr_vk.device   != VK_NULL_HANDLE) { vkDestroyDevice  (_skr_vk.device,   NULL); }
	if (_skr_vk.instance != VK_NULL_HANDLE) { vkDestroyInstance(_skr_vk.instance, NULL); }

	_skr_vk = (_skr_vk_t){0};
}

VkInstance skr_get_vk_instance(void) {
	return _skr_vk.instance;
}

VkDevice skr_get_vk_device(void) {
	return _skr_vk.device;
}

VkPhysicalDevice skr_get_vk_physical_device(void) {
	return _skr_vk.physical_device;
}

VkQueue skr_get_vk_graphics_queue(void) {
	return _skr_vk.graphics_queue;
}

uint32_t skr_get_vk_graphics_queue_family(void) {
	return _skr_vk.graphics_queue_family;
}

uint32_t skr_get_vk_transfer_queue_family(void) {
	return _skr_vk.transfer_queue_family;
}

uint32_t skr_get_vk_video_decode_queue_family(void) {
	return _skr_vk.video_decode_queue_family;
}

void skr_vk_queue_lock(uint32_t queue_family) {
	mtx_t *m = NULL;
	if      (queue_family == _skr_vk.graphics_queue_family)      m = _skr_vk.graphics_queue_mutex;
	else if (queue_family == _skr_vk.transfer_queue_family)      m = _skr_vk.transfer_queue_mutex;
	else if (queue_family == _skr_vk.video_decode_queue_family)  m = _skr_vk.video_decode_queue_mutex;
	if (m) mtx_lock(m);
}

void skr_vk_queue_unlock(uint32_t queue_family) {
	mtx_t *m = NULL;
	if      (queue_family == _skr_vk.graphics_queue_family)      m = _skr_vk.graphics_queue_mutex;
	else if (queue_family == _skr_vk.transfer_queue_family)      m = _skr_vk.transfer_queue_mutex;
	else if (queue_family == _skr_vk.video_decode_queue_family)  m = _skr_vk.video_decode_queue_mutex;
	if (m) mtx_unlock(m);
}

bool skr_vk_ext_enabled(const char* extension_name) {
	if (extension_name == NULL) return false;
	for (uint32_t i = 0; i < _skr_vk.enabled_instance_ext_count; i++)
		if (strcmp(_skr_vk.enabled_instance_exts[i], extension_name) == 0) return true;
	for (uint32_t i = 0; i < _skr_vk.enabled_device_ext_count; i++)
		if (strcmp(_skr_vk.enabled_device_exts[i], extension_name) == 0) return true;
	return false;
}

void* skr_vk_get_function(const char* function_name) {
	PFN_vkVoidFunction fn = NULL;
	if (_skr_vk.device != VK_NULL_HANDLE)
		fn = vkGetDeviceProcAddr(_skr_vk.device, function_name);
	if (fn == NULL && _skr_vk.instance != VK_NULL_HANDLE)
		fn = vkGetInstanceProcAddr(_skr_vk.instance, function_name);
	return (void*)fn;
}

void skr_get_vk_device_uuid(uint8_t out_uuid[VK_UUID_SIZE]) {
	VkPhysicalDeviceIDProperties id_props = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
	};
	VkPhysicalDeviceProperties2 props2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &id_props,
	};
	vkGetPhysicalDeviceProperties2(_skr_vk.physical_device, &props2);
	memcpy(out_uuid, id_props.deviceUUID, VK_UUID_SIZE);
}

int32_t skr_get_max_msaa_samples(void) {
	return _skr_vk.max_msaa_samples;
}

bool skr_is_capable(skr_capability_ capability) {
	if ((int32_t)capability < 0 || capability >= skr_capability_max)
		return false;
	return _skr_vk.capabilities[capability];
}

