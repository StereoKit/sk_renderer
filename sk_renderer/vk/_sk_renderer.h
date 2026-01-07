// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include "sk_renderer.h"
#include "skr_vulkan.h"

#include <volk.h>
#include <threads.h>

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

typedef struct {
	VkFormat              color_format;
	VkFormat              depth_format;
	VkFormat              resolve_format;
	VkSampleCountFlagBits samples;
	VkAttachmentStoreOp   depth_store_op;   // How to store depth (STORE or DONT_CARE)
	VkAttachmentLoadOp    color_load_op;    // How to load color (LOAD, CLEAR, or DONT_CARE)
} skr_pipeline_renderpass_key_t;

#define SKR_MAX_FRAMES_IN_FLIGHT 3
#define skr_MAX_COMMAND_RING    8   // Number of command buffers per thread
#define skr_MAX_THREAD_POOLS    16  // Maximum concurrent threads

// Bind shifts (hardcoded to match skshaderc)
#define SKR_BIND_SHIFT_BUFFER  0
#define SKR_BIND_SHIFT_TEXTURE 100
#define SKR_BIND_SHIFT_UAV     200

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

// Bind pool for material resource bindings
// Manages consecutive runs of slots for safe lifetime management
typedef struct {
	uint32_t start;
	uint32_t count;
} _skr_bind_range_t;

typedef struct {
	skr_material_bind_t* binds;
	uint32_t             capacity;
	_skr_bind_range_t*   free_ranges;
	uint32_t             free_range_count;
	uint32_t             free_range_capacity;
	mtx_t                mutex;
} _skr_bind_pool_t;

///////////////////////////////////////////////////////////////////////////////
// Bump Allocator - provides (buffer, offset) pairs with overflow support
///////////////////////////////////////////////////////////////////////////////

// Result of a bump allocation
typedef struct skr_bump_result_t {
	skr_buffer_t* buffer;
	uint32_t      offset;
} skr_bump_result_t;

// Bump allocator with automatic overflow handling
typedef struct skr_bump_alloc_t {
	// Main buffer (resized between frames based on high-water mark)
	skr_buffer_t     main_buffer;
	uint32_t         main_used;
	bool             main_valid;

	// Overflow buffers (created mid-frame if main is exhausted)
	skr_buffer_t*    overflow;
	uint32_t         overflow_count;
	uint32_t         overflow_capacity;

	// High-water mark for next-frame sizing
	uint32_t         high_water_mark;

	// Configuration
	skr_buffer_type_ buffer_type;
	uint32_t         alignment;  // Minimum alignment for allocations (e.g., 256 for UBOs)
} skr_bump_alloc_t;

///////////////////////////////////////////////////////////////////////////////

typedef struct {
	VkCommandBuffer    cmd;
	VkFence            fence;
	VkDescriptorPool   descriptor_pool;  // Per-command descriptor pool (for non-push-descriptor fallback)
	skr_destroy_list_t destroy_list;
	skr_bump_alloc_t   const_bump;       // Bump allocator for constant buffers (compute $Globals, system, material params)
	skr_bump_alloc_t   storage_bump;     // Bump allocator for storage buffers (instance data)
	bool               alive;
	uint64_t           generation;  // Incremented each time this slot is reused
} _skr_cmd_ring_slot_t;

// Command context returned from command begin/acquire
typedef struct {
	VkCommandBuffer     cmd;
	VkDescriptorPool    descriptor_pool;  // Per-command descriptor pool (VK_NULL_HANDLE if push descriptors enabled)
	skr_destroy_list_t* destroy_list;
	skr_bump_alloc_t*   const_bump;       // Bump allocator for constant buffers
	skr_bump_alloc_t*   storage_bump;     // Bump allocator for storage buffers
} _skr_cmd_ctx_t;

typedef struct {
	VkCommandPool          cmd_pool;
	_skr_cmd_ring_slot_t*  active_cmd;      // Currently recording command buffer
	_skr_cmd_ring_slot_t*  last_submitted;  // Most recently submitted command buffer
	_skr_cmd_ring_slot_t   cmd_ring[skr_MAX_COMMAND_RING];
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
	mtx_t                    queue_mutexes[4];         // Mutexes for unique queues (graphics, present, transfer, video_decode)
	mtx_t*                   graphics_queue_mutex;     // Pointer to correct mutex (may alias)
	mtx_t*                   present_queue_mutex;      // Pointer to correct mutex (may alias)
	mtx_t*                   transfer_queue_mutex;     // Pointer to correct mutex (may alias)
	VkCommandPool            command_pool;
	VkCommandBuffer          command_buffers[SKR_MAX_FRAMES_IN_FLIGHT];
	VkFence                  frame_fences[SKR_MAX_FRAMES_IN_FLIGHT];
	VkPipelineCache          pipeline_cache;
	VkDescriptorPool         descriptor_pool;
	VkDebugUtilsMessengerEXT debug_messenger;
	bool                     validation_enabled;
	bool                     has_push_descriptors;  // VK_KHR_push_descriptor support
	bool                     has_depth_clamp;       // VkPhysicalDeviceFeatures::depthClamp support
	bool                     initialized;

	// Memory allocators
	void*                  (*malloc_func) (size_t size);
	void*                  (*calloc_func) (size_t count, size_t size);
	void*                  (*realloc_func)(void* ptr, size_t size);
	void                   (*free_func)   (void* ptr);

	// Bind slot configuration
	skr_bind_settings_t      bind_settings;
	bool                     in_frame;  // True when between frame_begin and frame_end
	thrd_t                   main_thread_id;  // Thread that calls skr_init
	uint32_t                 frame;
	uint32_t                 flight_idx;

	// GPU timing (single query pool, 2 queries per frame)
	VkQueryPool              timestamp_pool;
	float                    timestamp_period;  // ns per tick
	uint64_t                 frame_timestamps[SKR_MAX_FRAMES_IN_FLIGHT][2];  // [frame][start/end]
	bool                     timestamps_valid[SKR_MAX_FRAMES_IN_FLIGHT];

	// Current render pass (for pipeline lookup)
	int32_t                  current_renderpass_idx;
	skr_tex_t*               current_color_texture;  // Track color texture for layout transitions
	skr_tex_t*               current_depth_texture;  // Track depth texture for layout transitions

	// Global bindings (merged with material bindings at draw time)
	skr_buffer_t*            global_buffers[16];
	skr_tex_t*               global_textures[16];

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

	// Deferred destruction
	skr_destroy_list_t       destroy_list;

	// Material bind pool
	_skr_bind_pool_t         bind_pool;

	// Sampler cache
	_skr_sampler_cache_t     sampler_cache;
} _skr_vk_t;

extern _skr_vk_t _skr_vk;

///////////////////////////////////////////////////////////////////////////////
// Internal helpers
///////////////////////////////////////////////////////////////////////////////

VkFramebuffer         _skr_create_framebuffer               (VkDevice device, VkRenderPass render_pass, skr_tex_t* color, skr_tex_t* depth, skr_tex_t* opt_resolve);
VkSampler             _skr_sampler_create_vk                (VkDevice device, skr_tex_sampler_t settings);
VkDescriptorSetLayout _skr_shader_make_layout               (VkDevice device, bool has_push_descriptors, const sksc_shader_meta_t* meta, skr_stage_ stage_mask);

// Format helpers
bool                  _skr_format_has_stencil               (VkFormat format);

// Material descriptor caching. Returns -1 on success, or the failing bind index if a resource is missing.
int32_t               _skr_material_add_writes              (const skr_material_bind_t* binds, uint32_t bind_ct, const int32_t* ignore_slots, int32_t ignore_ct, VkWriteDescriptorSet* ref_writes, uint32_t write_max, VkDescriptorBufferInfo* ref_buffer_infos, uint32_t buffer_max, VkDescriptorImageInfo* ref_image_infos, uint32_t image_max, uint32_t* ref_write_ct, uint32_t* ref_buffer_ct, uint32_t* ref_image_ct);
const char*           _skr_material_bind_name               (const sksc_shader_meta_t* meta, int32_t bind_idx);

// Bind pool management
void                  _skr_bind_pool_init                   (void);
void                  _skr_bind_pool_shutdown               (void);
int32_t               _skr_bind_pool_alloc                  (uint32_t count);  // Returns start index, -1 on failure
void                  _skr_bind_pool_free                   (int32_t start, uint32_t count);
skr_material_bind_t*  _skr_bind_pool_get                    (int32_t start);   // Get pointer to slot (NULL if invalid)
void                  _skr_bind_pool_lock                   (void);            // Lock pool for safe pointer access
void                  _skr_bind_pool_unlock                 (void);            // Unlock pool after done with pointer

// Sampler cache management
void                  _skr_sampler_cache_init               (void);
void                  _skr_sampler_cache_shutdown           (void);
VkSampler             _skr_sampler_cache_acquire            (skr_tex_sampler_t settings);  // Get or create sampler, increment ref
void                  _skr_sampler_cache_release            (skr_tex_sampler_t settings);  // Decrement ref, destroy if zero

// Bump allocator management
void                  _skr_bump_alloc_init                  (skr_bump_alloc_t* ref_alloc, skr_buffer_type_ type, uint32_t alignment);
void                  _skr_bump_alloc_destroy               (skr_bump_alloc_t* ref_alloc);
void                  _skr_bump_alloc_reset                 (skr_bump_alloc_t* ref_alloc);  // Call at frame start: resize main buffer, clean overflow
skr_bump_result_t     _skr_bump_alloc_write                 (skr_bump_alloc_t* ref_alloc, const void* data, uint32_t size);  // Allocate + write, returns buffer+offset

// Render list sorting
void                  _skr_render_list_sort                 (skr_render_list_t* ref_list);

// Debug
void                  _skr_set_debug_name                   (VkDevice device, VkObjectType type, uint64_t handle, const char* name);
void                  _skr_append_vertex_format             (char* ref_str, size_t str_size, const skr_vert_component_t* components, uint32_t component_count);
void                  _skr_append_material_config           (char* ref_str, size_t str_size, const _skr_pipeline_material_key_t* mat_key);
void                  _skr_append_renderpass_config         (char* ref_str, size_t str_size, const skr_pipeline_renderpass_key_t* rp_key);
void                  _skr_log_descriptor_writes            (const VkWriteDescriptorSet* writes, const VkDescriptorBufferInfo* buffer_infos, const VkDescriptorImageInfo* image_infos, uint32_t write_ct, uint32_t buffer_ct, uint32_t image_ct);

// Automatic layout transition system
void                  _skr_tex_transition                   (VkCommandBuffer cmd, skr_tex_t* ref_tex, VkImageLayout new_layout, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access);
void                  _skr_tex_transition_for_shader_read   (VkCommandBuffer cmd, skr_tex_t* ref_tex, VkPipelineStageFlags dst_stage);
void                  _skr_tex_transition_for_storage       (VkCommandBuffer cmd, skr_tex_t* ref_tex);
void                  _skr_tex_transition_queue_family      (VkCommandBuffer cmd, skr_tex_t* ref_tex, uint32_t src_queue_family, uint32_t dst_queue_family, VkImageLayout layout);
void                  _skr_tex_transition_notify_layout     (      skr_tex_t* ref_tex, VkImageLayout new_layout);  // Called after render pass implicit transitions
bool                  _skr_tex_needs_transition             (const skr_tex_t*     tex, uint8_t type); // Check if texture needs transition for given type (0=shader_read, 1=storage)
void                  _skr_tex_transition_enqueue           (      skr_tex_t* ref_tex, uint8_t type); // Deferred texture transition queue (to avoid in-renderpass barriers) type: 0=shader_read, 1=storage

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
void                  _skr_destroy_list_clear               (skr_destroy_list_t* ref_list);

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

// Custom deferred destruction (non-Vulkan types)
void                  _skr_cmd_destroy_bind_pool_slots      (skr_destroy_list_t* opt_ref_list, int32_t start, uint32_t count);

// Descriptor helper (allocates and binds descriptor set, handles push descriptors vs fallback)
void                  _skr_bind_descriptors                 (VkCommandBuffer cmd, VkDescriptorPool pool, VkPipelineBindPoint bind_point, VkPipelineLayout layout, VkDescriptorSetLayout desc_layout, VkWriteDescriptorSet* writes, uint32_t write_count);
