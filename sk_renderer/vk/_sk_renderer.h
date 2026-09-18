// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include "sk_renderer.h"
#include "skr_vulkan.h"
#include "../_skr_shared.h"

#include <volk.h>
#include <threads.h>

// Registries coordinate writers with one mutex and publish to lock-free
// readers with release/acquire pointer stores. MSVC gates C11 <stdatomic.h>
// behind /experimental:c11atomics, so use the intrinsics both compilers ship.
#if defined(_MSC_VER)
	#include <intrin.h>
	#define _skr_atomic(T)            T volatile
	#if defined(_M_ARM64) || defined(_M_ARM64EC)
		#define _skr_load_acquire(p)  ((void*)__ldar64((volatile __int64*)(p)))
	#else
		#define _skr_load_acquire(p)  (*(p)) // x86/x64 loads carry acquire already
	#endif
	#define _skr_store_release(p, v)  _InterlockedExchangePointer((void* volatile*)(p), (void*)(v))
#else
	#define _skr_atomic(T)            T
	#define _skr_load_acquire(p)      __atomic_load_n ((p),      __ATOMIC_ACQUIRE)
	#define _skr_store_release(p, v)  __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#endif

///////////////////////////////////////////////////////////////////////////////
// Memory allocation wrappers
///////////////////////////////////////////////////////////////////////////////

void* _skr_malloc (size_t size);
void* _skr_calloc (size_t count, size_t size);
void* _skr_realloc(void* ptr, size_t size);
void  _skr_free   (void* ptr);

///////////////////////////////////////////////////////////////////////////////
// Internal state
///////////////////////////////////////////////////////////////////////////////

// Boolean renderpass features, packed into skr_pipeline_renderpass_key_t.flags
// so a new boolean costs no struct layout churn.
typedef enum {
	skr_rp_flag_resolve_subpass    = 1 << 0, // Manual MSAA resolve subpass between geometry and postfx
	skr_rp_flag_custom_resolve     = 1 << 1, // VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM on resolve subpass
	skr_rp_flag_postfx_reads_depth = 1 << 2, // Postfx subpasses get depth as an input attachment. Under MSAA,
	                                         // geometry resolves depth on-tile (VK_KHR_depth_stencil_resolve)
	skr_rp_flag_postfx_depth_ms    = 1 << 3, // ...unless a depth-reading shader declared it as SubpassInputMS,
	                                         // which reads MSAA depth directly and skips that on-tile resolve
	skr_rp_flag_tile_shading       = 1 << 4, // VK_QCOM_tile_shading pass: a postfx shader reads an
	                                         // attachment as a tile attachment (neighborhood reads on-tile)
	skr_rp_flag_resolve_reads_depth= 1 << 5, // Tracked apart from the postfx flag so a depth-free resolve
	                                         // releases depth from tile memory after the geometry subpass
	skr_rp_flag_fragment_density_map = 1 << 6, // A fragment density map attachment drives foveation (VK_EXT_fragment_density_map)
	skr_rp_flag_msrtss             = 1 << 7, // Attachments are single-sample but rasterized at `samples` and
	                                         // resolved in-tile on store (VK_EXT_multisampled_render_to_single_sampled)
} skr_rp_flag_;

typedef struct {
	VkFormat              color_format;
	VkFormat              depth_format;
	VkFormat              resolve_format;
	VkSampleCountFlagBits samples;
	VkAttachmentStoreOp   depth_store_op;   // How to store depth (STORE or DONT_CARE)
	VkAttachmentLoadOp    color_load_op;    // How to load color (LOAD, CLEAR, or DONT_CARE)
	uint32_t              view_mask;        // 0 = single-view, non-zero = multiview (e.g. 0x3 for stereo)
	uint32_t              correlation_mask; // Bitmask of views that see the same geometry from different
	                                        // viewpoints (e.g. VR stereo eyes). The driver may reuse
	                                        // vertex processing, clipping, and binning across correlated
	                                        // views. Set to view_mask for stereo VR (0x3 for 2 eyes),
	                                        // 0 for cubemap faces (0x3F view_mask, 0x0 correlation) or
	                                        // independent array layers where each view sees different
	                                        // content. Single-view (view_mask=0x1) should use 0x1.
	uint8_t               subpass_index;    // 0 for geometry, 1+ for resolve/postfx subpasses
	uint8_t               postfx_count;     // 0 = single-subpass (legacy), 1+ = multi-subpass with postfx
	uint8_t               tile_apron[2];    // VkRenderPassTileShadingCreateInfoQCOM::tileApronSize, from
	                                        // the postfx shader's //--apron meta, clamped to maxApronSize
	uint32_t              flags;            // skr_rp_flag_* bits
	VkFormat              postfx_output_format;     // Format of the final postfx output attachment
	VkImageLayout         final_color_layout;       // 0 or COLOR_ATTACHMENT_OPTIMAL = default; SHADER_READ_ONLY = readable
	VkImageLayout         final_resolve_layout;     // 0 or COLOR_ATTACHMENT_OPTIMAL = default; SHADER_READ_ONLY = readable
	VkImageLayout         final_depth_layout;       // 0 or DEPTH_STENCIL_ATTACHMENT_OPTIMAL = default; DEPTH_STENCIL_READ_ONLY = readable
} skr_pipeline_renderpass_key_t;
// The renderpass cache deduplicates by memcmp over this struct, so its layout
// must have no padding holes — every byte a named field, single-byte fields in
// groups of 4/8. The assert trips when a new field breaks that; re-pack (or add
// explicit pad bytes) rather than letting uninitialized padding poison dedup.
_Static_assert(sizeof(skr_pipeline_renderpass_key_t) == 56, "renderpass key must stay padding-free for memcmp dedup");

#define SKR_QUEUE_TYPE_COUNT    4   // graphics, present, transfer, video_decode
#define skr_MAX_COMMAND_RING    8   // Number of command buffers per thread
#define skr_MAX_THREAD_POOLS    16  // Maximum concurrent threads

// Bind shifts (hardcoded to match skshaderc)
#define SKR_BIND_SHIFT_BUFFER  0
#define SKR_BIND_SHIFT_TEXTURE 100
#define SKR_BIND_SHIFT_UAV              200
#define SKR_BIND_SHIFT_INPUT_ATTACHMENT 300

// PostFX multi-subpass limits
#define SKR_POSTFX_MAX_ATTACHMENTS 8
#define SKR_POSTFX_MAX_SUBPASSES   (SKR_PASS_MAX_POSTFX + 2)  // geometry + resolve + postfx

// Stage the swapchain acquire semaphore is waited at (_skr_cmd_end_submit).
// A first write to an acquired image must source from this stage, or it runs
// ahead of the wait: barrier src scopes and external subpass dependencies.
#define _SKR_ACQUIRE_WAIT_STAGE VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT

#define SKR_VK_CHECK_RET(vkResult, fnName, returnVal) { VkResult __vr = (vkResult); if (__vr != VK_SUCCESS) { skr_log(skr_log_critical, "%s: 0x%X", fnName, (uint32_t)__vr); return returnVal; } }
#define SKR_VK_CHECK_NRET(vkResult, fnName) { VkResult __vr = (vkResult); if (__vr != VK_SUCCESS) { skr_log(skr_log_critical, "%s: 0x%X", fnName, (uint32_t)__vr); } }

// Deferred destruction system
typedef struct skr_destroy_list_t {
	void*    items;
	uint32_t count;
	uint32_t capacity;
	mtx_t    mutex;  // Thread-safe access for cross-thread destruction
} skr_destroy_list_t;

// Sampler cache for deduplicating VkSampler objects
// Most textures use one of a handful of sampler configurations
typedef struct {
	skr_tex_sampler_t settings;
	VkSampler         sampler;
	uint32_t          ref_count;
} _skr_sampler_entry_t;

typedef struct {
	_skr_sampler_entry_t* entries;
	uint32_t              count;
	uint32_t              capacity;
	mtx_t                 mutex;
} _skr_sampler_cache_t;

// The three bump-buffer bindings (material params, system data, instance
// data). Without push descriptors they are dynamic-offset descriptors, so a
// set survives across frames and the draw supplies the offsets, in binding
// order, for every dynamic binding the layout declares.
typedef struct {
	uint32_t bindings[3];
	uint32_t offsets [3];
	uint32_t count;
} _skr_dyn_offsets_t;

// Everything one bind needs: the writes, the infos they point at, and the
// dynamic offsets the draw supplies
#define _SKR_DESC_WRITES_MAX 32
#define _SKR_DESC_INFOS_MAX  16
typedef struct {
	VkWriteDescriptorSet   writes      [_SKR_DESC_WRITES_MAX];
	VkDescriptorBufferInfo buffer_infos[_SKR_DESC_INFOS_MAX];
	VkDescriptorImageInfo  image_infos [_SKR_DESC_INFOS_MAX];
	uint32_t               write_ct;
	uint32_t               buffer_ct;
	uint32_t               image_ct;
	_skr_dyn_offsets_t     dyn;
} _skr_desc_writes_t;

///////////////////////////////////////////////////////////////////////////////

typedef struct {
	VkCommandBuffer    cmd;
	VkFence            fence;
	skr_destroy_list_t destroy_list;
	bool               alive;
	uint64_t           generation;  // Incremented each time this slot is reused
} _skr_cmd_ring_slot_t;

// Per-frame transient data (material params, system data, instance data,
// compute $Globals) for one thread: one host-visible buffer shared by every
// command ring slot, each slot's writes forming one contiguous range that its
// fence retires. Positions are absolute and never wrap, so the tail scan can
// order them directly; the buffer offset is position % capacity.
typedef struct {
	skr_buffer_t                buffer;
	uint32_t                    capacity;
	uint64_t                    head;                          // next write position
	uint64_t                    start[skr_MAX_COMMAND_RING];   // each slot's live range, empty when start == end
	uint64_t                    end  [skr_MAX_COMMAND_RING];
	uint32_t                    slot;                          // the slot recording now
	const _skr_cmd_ring_slot_t* slots;                         // the owning thread's ring, for fence polling
	skr_buffer_type_            buffer_type;
	uint32_t                    alignment;
	uint32_t                    window;                        // bytes every offset must be able to reach
} _skr_bump_ring_t;

typedef struct skr_bump_result_t {
	VkBuffer buffer; // VK_NULL_HANDLE when the write failed
	uint32_t offset;
} skr_bump_result_t;

// Command context returned from command begin/acquire
typedef struct {
	VkCommandBuffer     cmd;
	skr_destroy_list_t* destroy_list;
	_skr_bump_ring_t*   const_ring;
	_skr_bump_ring_t*   storage_ring;
} _skr_cmd_ctx_t;

// Cached descriptor sets, one per bind slice, the only path on devices
// without push descriptors. Thread-private, so no locking.
#define _SKR_DESC_CACHE_SHIFT      8
#define _SKR_DESC_CACHE_CHUNK      (1u << _SKR_DESC_CACHE_SHIFT)
#define _SKR_DESC_CACHE_MAX_CHUNKS 1024 // must cover the bind pool's whole index range, see _SKR_BIND_MAX_CHUNKS
typedef struct {
	VkDescriptorSet set;
	uint64_t        hash; // of the writes the set was built from, 0 = empty
	uint8_t         pool; // index into _skr_desc_cache_t.pools
} _skr_desc_cache_entry_t;

typedef struct {
	VkDescriptorSet set;
	uint8_t         pool;
} _skr_desc_retired_t;

typedef struct {
	_skr_desc_cache_entry_t* chunks[_SKR_DESC_CACHE_MAX_CHUNKS];
	VkDescriptorPool*        pools;         // free-set pools, a new one opens when none has room
	uint16_t*                pool_used;     // live sets per pool
	bool*                    pool_no_room;  // refused an alloc with sets left; cleared by the next free from it
	uint32_t                 pool_count;
	uint32_t                 pool_capacity;
	// A replaced set may still be bound in any in-flight command buffer, so
	// it frees when the slot that replaced it next begins recording: that
	// slot's fence orders after every earlier submission
	_skr_desc_retired_t*     retired         [skr_MAX_COMMAND_RING];
	uint32_t                 retired_count   [skr_MAX_COMMAND_RING];
	uint32_t                 retired_capacity[skr_MAX_COMMAND_RING];
} _skr_desc_cache_t;

typedef struct {
	VkCommandPool          cmd_pool;
	_skr_cmd_ring_slot_t*  active_cmd;      // Currently recording command buffer
	_skr_cmd_ring_slot_t*  last_submitted;  // Most recently submitted command buffer
	_skr_cmd_ring_slot_t   cmd_ring[skr_MAX_COMMAND_RING];
	_skr_bump_ring_t       const_ring;      // material params, system data, compute $Globals
	_skr_bump_ring_t       storage_ring;    // instance data
	_skr_desc_cache_t      desc_cache;
	uint32_t               cmd_ring_index;
	uint32_t               thread_idx;
	int32_t                ref_count;
	bool                   alive;
} _skr_vk_thread_t;

typedef struct {
	VkInstance               instance;
	VkPhysicalDevice         physical_device;
	VkDevice                 device;
	VkQueue                  graphics_queue;
	VkQueue                  present_queue;
	VkQueue                  transfer_queue;
	uint32_t                 graphics_queue_family;
	uint32_t                 present_queue_family;
	uint32_t                 transfer_queue_family;
	uint32_t                 video_decode_queue_family;  // UINT32_MAX if not available
	mtx_t                    queue_mutexes[SKR_QUEUE_TYPE_COUNT]; // Mutexes for unique queues (graphics, present, transfer, video_decode)
	mtx_t*                   graphics_queue_mutex;     // Pointer to correct mutex (may alias)
	mtx_t*                   present_queue_mutex;      // Pointer to correct mutex (may alias)
	mtx_t*                   transfer_queue_mutex;     // Pointer to correct mutex (may alias)
	mtx_t*                   video_decode_queue_mutex; // Pointer to correct mutex (may alias, NULL if no video decode)
	VkCommandPool            command_pool;
	VkCommandBuffer          command_buffers[SKR_MAX_FRAMES_IN_FLIGHT];
	VkFence                  frame_fences[SKR_MAX_FRAMES_IN_FLIGHT];
	VkPipelineCache          pipeline_cache;
	VkDebugUtilsMessengerEXT debug_messenger;
	bool                     validation_enabled;
	bool                     has_push_descriptors;        // VK_KHR_push_descriptor support
	bool                     has_depth_clamp;             // VkPhysicalDeviceFeatures::depthClamp support
	bool                     has_fill_mode_non_solid;     // VkPhysicalDeviceFeatures::fillModeNonSolid support
	bool                     has_external_memory_fd;      // VK_KHR_external_memory_fd
	bool                     has_external_memory_win32;   // VK_KHR_external_memory_win32
	bool                     has_android_hardware_buffer; // VK_ANDROID_external_memory_android_hardware_buffer
	bool                     has_external_memory_dma_buf; // VK_EXT_external_memory_dma_buf
	bool                     has_drm_format_modifier;     // VK_EXT_image_drm_format_modifier
	bool                     has_external_fence_fd;       // VK_KHR_external_fence_fd
	bool                     has_video_decode;            // VK_KHR_video_decode_queue + related extensions
	bool                     has_ycbcr_conversion;        // VkPhysicalDeviceSamplerYcbcrConversionFeatures::samplerYcbcrConversion
	bool                     has_custom_resolve;          // VK_QCOM_render_pass_shader_resolve
	bool                     has_create_renderpass2;      // VK_KHR_create_renderpass2
	bool                     has_depth_stencil_resolve;   // VK_KHR_depth_stencil_resolve (implies create_renderpass2)
	bool                     has_fragment_density_map;    // VK_EXT_fragment_density_map + fragmentDensityMap feature (foveation)
	bool                     has_fdm_non_subsampled;      // fragmentDensityMapNonSubsampledImages: FDM passes can use ordinary attachments
	bool                     has_msrtss;                  // VK_EXT_multisampled_render_to_single_sampled + feature (implies renderpass2 + depth_stencil_resolve)
	bool                     has_present_fence;           // VK_EXT_swapchain_maintenance1, fences observe present completion
	bool                     has_present_timing;          // VK_EXT_present_timing, presentTiming feature
	bool                     has_present_wait2;           // VK_KHR_present_id2 + VK_KHR_present_wait2
	bool                     has_present_wait;            // VK_KHR_present_id + VK_KHR_present_wait
	bool                     has_display_timing_google;   // VK_GOOGLE_display_timing (Android)
	bool                     has_calibrated_timestamps;   // VK_KHR/EXT_calibrated_timestamps with a device + host domain pair
	PFN_vkGetCalibratedTimestampsKHR get_calibrated_timestamps; // Whichever spelling the driver enabled, NULL without has_calibrated_timestamps
	VkTimeDomainKHR          host_time_domain;            // The domain skr_time_now_ns reads
	uint64_t                 calib_device_ticks;          // Last calibration pair, device ticks and host ns
	uint64_t                 calib_host_ns;
	uint64_t                 frame_present_id[SKR_MAX_FRAMES_IN_FLIGHT]; // The present each frame fed, 0 if none
	skr_surface_t*           frame_surface   [SKR_MAX_FRAMES_IN_FLIGHT]; // and the surface it went to; cleared on surface destroy
	bool                     has_store_op_none;           // VK_ATTACHMENT_STORE_OP_NONE, from any of the KHR/EXT/QCOM extensions
	bool                     has_subpass_merge_feedback;  // VK_EXT_subpass_merge_feedback + feature bit
	bool                     has_subgroup_size_control;   // VK_EXT_subgroup_size_control + subgroupSizeControl feature
	bool                     has_storage_without_format;  // shaderStorageImage(Write+Read)WithoutFormat, for format Unknown storage access
	bool                     has_qcom_image_proc;         // VK_QCOM_image_processing + all three feature bits, plus its
	                                                      // SPIR-V 1.4 prerequisites (VK_KHR_spirv_1_4 + shader_float_controls)
	                                                      // and VK_KHR_format_feature_flags2
	bool                     has_qcom_tile_shading;       // VK_QCOM_tile_shading fragment-stage set (tileShading,
	                                                      // fragmentStage, colorAttachments, sampledAttachments, apron)
	                                                      // + VK_QCOM_tile_properties + synchronization2 + renderpass2
	uint32_t                 max_tile_apron;              // VkPhysicalDeviceTileShadingPropertiesQCOM::maxApronSize
	VkSampler                sampler_image_proc;          // The one legal QCOM image-processing sampler config (nearest,
	                                                      // clamp-to-edge, lod 0); bound in place of the texture's own
	                                                      // sampler on bindings whose meta shape declares bit 6
	uint32_t                 min_subgroup_size;           // From VkPhysicalDeviceSubgroupSizeControlPropertiesEXT
	uint32_t                 max_subgroup_size;
	VkShaderStageFlags       required_subgroup_size_stages; // Stages that allow VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
	bool                     initialized;
	uint32_t                 max_multiview_view_count;    // From VkPhysicalDeviceMultiviewProperties

	// Capability system (runtime-queried feature support)
	bool                     capabilities[skr_capability_max];

	char**                   enabled_instance_exts;       // Copied enabled ext names, for skr_vk_ext_enabled
	uint32_t                 enabled_instance_ext_count;
	char**                   enabled_device_exts;         // Copied enabled ext names, for skr_vk_ext_enabled
	uint32_t                 enabled_device_ext_count;

	// Shader-support mask: bit (1 << sksc_feature_bit_) set iff that device
	// feature is actually enabled at device creation. Compared against a
	// shader meta's `features` by skr_shader_check_support. See skr_initialize.c
	// where it's built, and keep it in sync as new features get enabled.
	uint64_t                 enabled_features;

	// Memory allocators
	void*                  (*malloc_func) (size_t size);
	void*                  (*calloc_func) (size_t count, size_t size);
	void*                  (*realloc_func)(void* ptr, size_t size);
	void                   (*free_func)   (void* ptr);

	// Bind slot configuration
	skr_bind_settings_t      bind_settings;
	skr_buffering_           buffering;
	bool                     in_frame;  // True when between frame_begin and frame_end
	thrd_t                   main_thread_id;  // Thread that calls skr_init
	uint32_t                 frame;
	uint32_t                 flight_idx;

	// GPU timing (single query pool, 2 queries per frame)
	VkQueryPool              timestamp_pool;
	float                    timestamp_period;       // ns per tick
	uint32_t                 min_ubo_offset_align;   // minUniformBufferOffsetAlignment
	uint32_t                 min_ssbo_offset_align;  // minStorageBufferOffsetAlignment
	int32_t                  max_msaa_samples;       // Maximum supported MSAA sample count
	uint64_t                 frame_timestamps[SKR_MAX_FRAMES_IN_FLIGHT][2];  // [frame][start/end]

	// CPU timing (wall-clock time for frame work, excluding vsync)
	uint64_t                 cpu_frame_start_ns  [SKR_MAX_FRAMES_IN_FLIGHT];
	uint64_t                 cpu_frame_end_ns    [SKR_MAX_FRAMES_IN_FLIGHT];
	uint64_t                 cpu_frame_wait_ns   [SKR_MAX_FRAMES_IN_FLIGHT];  // Accumulated wait time to subtract
	skr_frame_timing_t       last_frame_timing;      // Snapshot taken when a frame's timestamps are read back
	bool                     last_frame_timing_valid;

	// Current render pass (for pipeline lookup)
	int32_t                  current_renderpass_idx;
	skr_tex_t*               current_color_texture;    // Track color texture for layout transitions
	skr_tex_t*               current_depth_texture;    // Track depth texture for layout transitions
	skr_tex_t*               current_resolve_texture;  // Track resolve target for layout transitions

	// Global bindings (merged with material bindings at draw time)
	skr_buffer_t*            global_buffers[16];
	skr_tex_t*               global_textures[16];

	// Deferred compute→graphics barrier (flushed before next render pass)
	bool                     pending_compute_barrier;

	// Deferred texture transition tracking (to avoid in-renderpass barriers)
	skr_tex_t**              pending_transitions;
	uint8_t*                 pending_transition_types;  // 0=shader_read, 1=storage
	uint32_t                 pending_transition_count;
	uint32_t                 pending_transition_capacity;

	// Command system
	bool                     has_dedicated_transfer;
	_skr_vk_thread_t         thread_pools[skr_MAX_THREAD_POOLS];
	mtx_t                    thread_pool_mutex;

	// Default assets
	skr_tex_t                default_tex_white;
	skr_tex_t                default_tex_black;
	skr_tex_t                default_tex_gray;

	// Built-in mipgen fallbacks. Used by skr_tex_generate_mips when no shader
	// is passed and the texture format doesn't support blit.
	skr_shader_t             builtin_mipgen_2d;
	skr_shader_t             builtin_mipgen_cube;

	// Deferred destruction
	skr_destroy_list_t       destroy_list;

	// Sampler cache
	_skr_sampler_cache_t     sampler_cache;
} _skr_vk_t;

extern _skr_vk_t _skr_vk;

///////////////////////////////////////////////////////////////////////////////
// Internal helpers
///////////////////////////////////////////////////////////////////////////////

VkFramebuffer         _skr_create_framebuffer               (VkDevice device, VkRenderPass render_pass, skr_tex_t* color, skr_tex_t* depth, skr_tex_t* opt_resolve);
skr_tex_t*            _skr_framebuffer_fdm                  (skr_tex_t* opt_color, skr_tex_t* opt_resolve);
VkDeviceMemory        _skr_allocate_image_memory            (VkDevice device, VkPhysicalDevice phys_device, VkImage image, bool is_transient_attachment, VkDeviceMemory* out_memory);
VkSampler             _skr_sampler_create_vk                (VkDevice device, skr_tex_sampler_t settings);
skr_err_              _skr_tex_create_scratch               (const skr_tex_t* template_src, skr_tex_t* out_tex);
VkDescriptorSetLayout _skr_shader_make_layout               (VkDevice device, bool has_push_descriptors, const sksc_shader_meta_t* meta, skr_stage_ stage_mask, const VkSampler* immutable_samplers, const int32_t* immutable_sampler_slots, int32_t immutable_sampler_count);
void                  _skr_shader_resolve_spec_constants    (const sksc_shader_meta_t* meta, const skr_spec_constant_t* specs, uint32_t spec_count, uint32_t out_values[SKR_MAX_SPEC_CONSTANTS]);
const VkSpecializationInfo* _skr_shader_make_spec_info      (const sksc_shader_meta_t* meta, const uint32_t* spec_values, VkSpecializationMapEntry out_entries[SKR_MAX_SPEC_CONSTANTS], VkSpecializationInfo* out_info);

// Timing helpers
uint64_t              _skr_time_get_ns                      (void);
uint64_t              _skr_time_from_host_domain            (uint64_t stamp);  // A raw host_time_domain stamp onto the _skr_time_get_ns scale
void                  _skr_frame_note_present               (const skr_surface_t* surface, uint64_t id);  // Joins a present to the frame that just ended, when it was that frame's surface

// Material descriptor caching. Returns -1 on success, or the failing bind index if a resource is missing.
int32_t               _skr_material_add_writes              (const skr_material_bind_t* binds, uint32_t bind_ct, skr_stage_ stage_mask, const int32_t* ignore_slots, int32_t ignore_ct, _skr_desc_writes_t* ref_writes);
const char*           _skr_material_bind_name               (const sksc_shader_meta_t* meta, int32_t bind_idx);

// Bind pool management
void                  _skr_bind_pool_init                   (void);
void                  _skr_bind_pool_shutdown               (void);
int32_t               _skr_bind_pool_alloc                  (uint32_t count);  // Returns start index, -1 on failure
void                  _skr_bind_pool_free                   (int32_t start, uint32_t count);
skr_material_bind_t*  _skr_bind_pool_get                    (int32_t start);   // Get pointer to slot (NULL if invalid)

// Sampler cache management
void                  _skr_sampler_cache_init               (void);
void                  _skr_sampler_cache_shutdown           (void);
void                  _skr_mipgen_materials_init            (void);
void                  _skr_mipgen_materials_shutdown        (void);
void                  _skr_mipgen_material_release          (const skr_shader_t* shader);
void                  _skr_shader_param_write               (const sksc_shader_meta_t* meta, void* param_buffer, uint32_t param_buffer_size, const char* name, sksc_shader_var_ type, uint32_t count, const void* data);
VkSampler             _skr_sampler_cache_acquire            (skr_tex_sampler_t settings);  // Get or create sampler, increment ref
void                  _skr_sampler_cache_release            (skr_tex_sampler_t settings);  // Decrement ref, destroy if zero

// Bump allocator management
void                  _skr_bump_ring_init                   (_skr_bump_ring_t* out_ring, skr_buffer_type_ type, uint32_t alignment, const _skr_cmd_ring_slot_t* slots);
void                  _skr_bump_ring_destroy                (_skr_bump_ring_t* ref_ring);
void                  _skr_bump_ring_slot_begin             (_skr_bump_ring_t* ref_ring, uint32_t slot);
uint32_t              _skr_bump_ring_reserve_window         (_skr_bump_ring_t* ref_ring, uint32_t bytes);
skr_bump_result_t     _skr_bump_ring_write                  (_skr_bump_ring_t* ref_ring, const void* data, uint32_t size);

// Render list sorting
void                  _skr_render_list_sort                 (skr_render_list_t* ref_list);

// Debug
void                  _skr_set_debug_name                   (VkDevice device, VkObjectType type, uint64_t handle, const char* name);
void                  _skr_append_str                       (char* ref_str, size_t str_size, const char* text);
void                  _skr_append_vertex_format             (char* ref_str, size_t str_size, const skr_vert_component_t* components, uint32_t component_count);
const char*           _skr_semantic_name                    (skr_semantic_ semantic);
void                  _skr_append_material_config           (char* ref_str, size_t str_size, const _skr_pipeline_material_key_t* mat_key);
void                  _skr_append_renderpass_config         (char* ref_str, size_t str_size, const skr_pipeline_renderpass_key_t* rp_key);
void                  _skr_log_descriptor_writes            (const VkWriteDescriptorSet* writes, uint32_t write_ct, uint32_t buffer_ct, uint32_t image_ct);

// Barrier batch collector — accumulates barriers and flushes in one vkCmdPipelineBarrier
#define SKR_MAX_BATCHED_IMAGE_BARRIERS 32
typedef struct {
	VkImageMemoryBarrier image_barriers[SKR_MAX_BATCHED_IMAGE_BARRIERS];
	uint32_t             image_count;
	VkMemoryBarrier      mem_barrier;
	bool                 has_mem_barrier;
	VkPipelineStageFlags src_stages;
	VkPipelineStageFlags dst_stages;
} _skr_barrier_batch_t;

void                  _skr_barrier_batch_init               (_skr_barrier_batch_t* batch);
void                  _skr_barrier_batch_add                (_skr_barrier_batch_t* batch, VkCommandBuffer cmd, skr_tex_t* ref_tex, VkImageLayout new_layout, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access);
void                  _skr_barrier_batch_add_memory         (_skr_barrier_batch_t* batch, VkPipelineStageFlags src_stage, VkAccessFlags src_access, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access);
void                  _skr_barrier_batch_flush              (_skr_barrier_batch_t* batch, VkCommandBuffer cmd);

// Automatic layout transition system
void                  _skr_tex_transition                   (VkCommandBuffer cmd, skr_tex_t* ref_tex, VkImageLayout new_layout, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access);
void                  _skr_tex_barrier                      (VkCommandBuffer cmd, skr_tex_t* ref_tex, VkPipelineStageFlags src_stage, VkAccessFlags src_access, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access);
void                  _skr_tex_transition_for_shader_read   (VkCommandBuffer cmd, skr_tex_t* ref_tex, VkPipelineStageFlags dst_stage);
void                  _skr_tex_transition_for_storage       (VkCommandBuffer cmd, skr_tex_t* ref_tex);
void                  _skr_tex_transition_queue_family      (VkCommandBuffer cmd, skr_tex_t* ref_tex, uint32_t src_queue_family, uint32_t dst_queue_family, VkImageLayout layout);
void                  _skr_tex_transition_notify_layout     (      skr_tex_t* ref_tex, VkImageLayout new_layout);  // Called after render pass implicit transitions
bool                  _skr_tex_needs_transition             (const skr_tex_t*     tex, uint8_t type); // Check if texture needs transition for given type (0=shader_read, 1=storage)
VkImageLayout         _skr_tex_sample_layout                (const skr_tex_t*     tex); // Canonical layout for sampling a sk_renderer-owned texture (matches descriptor imageLayout)
VkImageLayout         _skr_tex_attachment_layout            (const skr_tex_t*     tex); // Canonical layout for using a texture as a render-pass attachment (color vs depth, picked from aspect_mask)
void                  _skr_tex_transition_enqueue           (      skr_tex_t* ref_tex, uint8_t type); // Deferred texture transition queue (to avoid in-renderpass barriers) type: 0=shader_read, 1=storage
void                  _skr_tex_transition_dequeue           (      skr_tex_t* ref_tex);               // Remove from deferred queue (called on texture destroy)

// vkDeviceWaitIdle with every queue mutex held; the plain call implicitly
// uses all queues and races worker-thread submits (skr_initialize.c)
void                  _skr_device_wait_idle                 (void);

// Command buffer management
bool                  _skr_cmd_init                         (void);
void                  _skr_cmd_shutdown                     (void);
_skr_vk_thread_t*     _skr_cmd_get_thread                   (void);
_skr_cmd_ctx_t        _skr_cmd_begin                        (void);
bool                  _skr_cmd_try_get_active               (_skr_cmd_ctx_t* out_ctx);
VkCommandBuffer       _skr_cmd_end                          (void);  // Ends and returns command buffer (caller must submit)
skr_future_t          _skr_cmd_end_submit                   (const VkSemaphore* wait_semaphores, uint32_t wait_count, const VkSemaphore* signal_semaphores, uint32_t signal_count);  // Ends and submits, returns future
_skr_cmd_ctx_t        _skr_cmd_acquire                      (void);
void                  _skr_cmd_release                      (VkCommandBuffer buffer);

// Deferred destruction API
skr_destroy_list_t    _skr_destroy_list_create              (void);
void                  _skr_destroy_list_free                (skr_destroy_list_t* ref_list);
void                  _skr_destroy_list_execute             (skr_destroy_list_t* ref_list);

// Add functions for each Vulkan resource type
void                  _skr_cmd_destroy_buffer               (skr_destroy_list_t* opt_ref_list, VkBuffer                 handle);
void                  _skr_cmd_destroy_image                (skr_destroy_list_t* opt_ref_list, VkImage                  handle);
void                  _skr_cmd_destroy_image_view           (skr_destroy_list_t* opt_ref_list, VkImageView              handle);
void                  _skr_cmd_destroy_sampler              (skr_destroy_list_t* opt_ref_list, VkSampler                handle);
void                  _skr_cmd_destroy_framebuffer          (skr_destroy_list_t* opt_ref_list, VkFramebuffer            handle);
void                  _skr_cmd_destroy_render_pass          (skr_destroy_list_t* opt_ref_list, VkRenderPass             handle);
void                  _skr_cmd_destroy_pipeline             (skr_destroy_list_t* opt_ref_list, VkPipeline               handle);
void                  _skr_cmd_destroy_pipeline_layout      (skr_destroy_list_t* opt_ref_list, VkPipelineLayout         handle);
void                  _skr_cmd_destroy_pipeline_cache       (skr_destroy_list_t* opt_ref_list, VkPipelineCache          handle);
void                  _skr_cmd_destroy_descriptor_set_layout(skr_destroy_list_t* opt_ref_list, VkDescriptorSetLayout    handle);
void                  _skr_cmd_destroy_descriptor_pool      (skr_destroy_list_t* opt_ref_list, VkDescriptorPool         handle);
void                  _skr_cmd_destroy_shader_module        (skr_destroy_list_t* opt_ref_list, VkShaderModule           handle);
void                  _skr_cmd_destroy_command_pool         (skr_destroy_list_t* opt_ref_list, VkCommandPool            handle);
void                  _skr_cmd_destroy_fence                (skr_destroy_list_t* opt_ref_list, VkFence                  handle);
void                  _skr_cmd_destroy_semaphore            (skr_destroy_list_t* opt_ref_list, VkSemaphore              handle);
void                  _skr_cmd_destroy_query_pool           (skr_destroy_list_t* opt_ref_list, VkQueryPool              handle);
void                  _skr_cmd_destroy_swapchain            (skr_destroy_list_t* opt_ref_list, VkSwapchainKHR           handle);
void                  _skr_cmd_destroy_surface              (skr_destroy_list_t* opt_ref_list, VkSurfaceKHR             handle);
void                  _skr_cmd_destroy_debug_messenger      (skr_destroy_list_t* opt_ref_list, VkDebugUtilsMessengerEXT handle);
void                  _skr_cmd_destroy_memory               (skr_destroy_list_t* opt_ref_list, VkDeviceMemory           handle);
void                  _skr_cmd_destroy_ycbcr_conversion     (skr_destroy_list_t* opt_ref_list, VkSamplerYcbcrConversion handle);

// Custom deferred destruction (non-Vulkan types)
void                  _skr_cmd_destroy_bind_pool_slots      (skr_destroy_list_t* opt_ref_list, int32_t start, uint32_t count);

// Descriptor binding: push descriptors, or the per-thread set cache without them
void                  _skr_desc_writes_begin                (_skr_desc_writes_t* out_writes, int32_t material_idx);
void                  _skr_write_buffer                     (_skr_desc_writes_t* ref_writes, uint32_t binding, bool storage, VkBuffer buffer, uint32_t offset, uint32_t range);
void                  _skr_write_image                      (_skr_desc_writes_t* ref_writes, uint32_t binding, VkDescriptorType type, VkSampler sampler, VkImageView view, VkImageLayout layout);
bool                  _skr_bind_descriptors                 (VkCommandBuffer cmd, VkPipelineBindPoint bind_point, int32_t bind_start, VkPipelineLayout layout, VkDescriptorSetLayout desc_layout, _skr_desc_writes_t* ref_writes);
void                  _skr_desc_cache_destroy               (_skr_desc_cache_t* ref_cache);
void                  _skr_desc_cache_retire                (_skr_desc_cache_t* ref_cache, uint32_t slot);
uint32_t              _skr_shader_dyn_bindings              (const sksc_shader_meta_t* meta, skr_stage_ stage_mask, bool has_push_descriptors, uint32_t out_bindings[3]);
uint32_t              _skr_pipeline_get_dyn_bindings        (int32_t material_idx, uint32_t out_bindings[3]);
