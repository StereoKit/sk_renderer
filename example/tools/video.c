// SPDX-License-Identifier: MIT
// Video playback module using FFmpeg with Vulkan hardware acceleration

#include "video.h"
#include "scene_util.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>

#include <vulkan/vulkan.h>

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

///////////////////////////////////////////////////////////////////////////////
// FFmpeg queue lock/unlock callbacks (delegates to sk_renderer's queue mutexes)
///////////////////////////////////////////////////////////////////////////////

static void _ff_lock_queue(struct AVHWDeviceContext* ctx, uint32_t queue_family, uint32_t index) {
	(void)ctx; (void)index;
	skr_vk_queue_lock(queue_family);
}

static void _ff_unlock_queue(struct AVHWDeviceContext* ctx, uint32_t queue_family, uint32_t index) {
	(void)ctx; (void)index;
	skr_vk_queue_unlock(queue_family);
}

///////////////////////////////////////////////////////////////////////////////
// Internal structures
///////////////////////////////////////////////////////////////////////////////

typedef struct {
	AVBufferRef*         hw_device_ctx;
	PFN_vkWaitSemaphores pfn_wait_semaphores;
	uint32_t             decode_family;
} _vulkan_hw_context_t;

struct video_t {
	// Stream metadata (read-only after open)
	int32_t     width, height;
	int32_t     coded_width, coded_height;
	double      duration, framerate, time_base;
	bool        is_live, is_seekable;

	// Playback state
	double      current_pts;
	bool        valid, eof, needs_flush;
	atomic_bool abort_decode;

	// FFmpeg demuxer
	AVFormatContext* format_ctx;
	int32_t          stream_idx;
	AVPacket*        packet;

	// FFmpeg decoder
	AVCodecContext* codec_ctx;
	AVFrame*        frame;
	AVFrame*        held_frame;
	AVFrame*        sw_frame;

	// Vulkan hardware acceleration
	_vulkan_hw_context_t hw_ctx;
	const char*          enabled_exts[16];
	bool                 hw_accel;
	bool                 zero_copy;

	// Textures (zero-copy OR software, never both)
	skr_tex_t tex_y;
	skr_tex_t tex_uv;
	skr_tex_t sw_tex_y;
	skr_tex_t sw_tex_uv;

	// Rendering (owned by video module)
	skr_shader_t   shader;
	skr_material_t material;
	bool           material_ready;
};

///////////////////////////////////////////////////////////////////////////////
// Hardware context setup
///////////////////////////////////////////////////////////////////////////////

static enum AVPixelFormat _get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
	(void)ctx;
	for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
		if (*p == AV_PIX_FMT_VULKAN)
			return *p;
	}
	return AV_PIX_FMT_YUV420P;
}

// Initialize a shared Vulkan hardware context for FFmpeg video decode.
// Returns true on success. The caller-owned enabled_exts array is filled in
// with extension name pointers (string literals, no allocation needed).
static bool _init_vulkan_hwcontext(
	_vulkan_hw_context_t* out_ctx,
	const char**          ref_enabled_exts,
	int32_t*              out_ext_count)
{
	if (!skr_is_capable(skr_capability_vk_video)) return false;

	AVBufferRef* hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
	if (!hw_device_ctx) return false;

	AVHWDeviceContext     *device_ctx = (AVHWDeviceContext *)hw_device_ctx->data;
	AVVulkanDeviceContext *vk_ctx     = device_ctx->hwctx;

	// Share sk_renderer's Vulkan handles
	vk_ctx->get_proc_addr = vkGetInstanceProcAddr;
	vk_ctx->inst          = skr_get_vk_instance();
	vk_ctx->phys_dev      = skr_get_vk_physical_device();
	vk_ctx->act_dev       = skr_get_vk_device();

	// Queue families
	uint32_t gfx_family    = skr_get_vk_graphics_queue_family();
	uint32_t tx_family     = skr_get_vk_transfer_queue_family();
	uint32_t decode_family = skr_get_vk_video_decode_queue_family();

	// Query queue family properties for flags
	uint32_t qf_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(vk_ctx->phys_dev, &qf_count, NULL);
	VkQueueFamilyProperties qf_props[32];
	vkGetPhysicalDeviceQueueFamilyProperties(vk_ctx->phys_dev, &qf_count, qf_props);

	int32_t nb_qf = 0;
	vk_ctx->qf[nb_qf++] = (AVVulkanDeviceQueueFamily){
		.idx   = (int)gfx_family,
		.num   = 1,
		.flags = qf_props[gfx_family].queueFlags,
	};
	if (tx_family != gfx_family) {
		vk_ctx->qf[nb_qf++] = (AVVulkanDeviceQueueFamily){
			.idx   = (int)tx_family,
			.num   = 1,
			.flags = qf_props[tx_family].queueFlags,
		};
	}
	if (decode_family != UINT32_MAX && decode_family != gfx_family && decode_family != tx_family) {
		vk_ctx->qf[nb_qf++] = (AVVulkanDeviceQueueFamily){
			.idx        = (int)decode_family,
			.num        = 1,
			.flags      = qf_props[decode_family].queueFlags,
			.video_caps = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
		};
	}
	vk_ctx->nb_qf = nb_qf;

	// Lock/unlock callbacks for thread-safe queue submission. Deprecated by
	// FFmpeg, but still the only hook available on the versions we support.
#if defined(__GNUC__) || defined(__clang__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
	vk_ctx->lock_queue   = _ff_lock_queue;
	vk_ctx->unlock_queue = _ff_unlock_queue;
#if defined(__GNUC__) || defined(__clang__)
	#pragma GCC diagnostic pop
#endif

	// Enabled device extensions - stored in caller's array so FFmpeg can
	// access them after this function returns (FFmpeg keeps the pointer)
	int32_t ext_count = 0;
	ref_enabled_exts[ext_count++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
	ref_enabled_exts[ext_count++] = "VK_KHR_synchronization2";
	ref_enabled_exts[ext_count++] = "VK_KHR_timeline_semaphore";
	ref_enabled_exts[ext_count++] = "VK_KHR_video_queue";
	ref_enabled_exts[ext_count++] = "VK_KHR_video_decode_queue";
	ref_enabled_exts[ext_count++] = "VK_KHR_video_decode_h264";
	if (skr_is_capable(skr_capability_external_gl))  ref_enabled_exts[ext_count++] = "VK_KHR_external_memory_fd";
	if (skr_is_capable(skr_capability_external_dma)) ref_enabled_exts[ext_count++] = "VK_EXT_external_memory_dma_buf";
	if (skr_is_capable(skr_capability_external_dma)) ref_enabled_exts[ext_count++] = "VK_EXT_image_drm_format_modifier";
	vk_ctx->enabled_dev_extensions    = (const char* const*)ref_enabled_exts;
	vk_ctx->nb_enabled_dev_extensions = ext_count;
	*out_ext_count = ext_count;

	// Device features - query what the physical device supports
	VkPhysicalDeviceFeatures2 features2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	vkGetPhysicalDeviceFeatures2(vk_ctx->phys_dev, &features2);
	vk_ctx->device_features = features2;

	// Deprecated queue family fields (FFmpeg compat). Setting them is the point
	// here, so the deprecation notice is silenced rather than acted on.
#ifdef FF_API_VULKAN_FIXED_QUEUES
#if defined(__GNUC__) || defined(__clang__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
	vk_ctx->queue_family_index        = (int)gfx_family;
	vk_ctx->nb_graphics_queues        = 1;
	vk_ctx->queue_family_tx_index     = (int)tx_family;
	vk_ctx->nb_tx_queues              = 1;
	vk_ctx->queue_family_comp_index   = (int)gfx_family;
	vk_ctx->nb_comp_queues            = 1;
	vk_ctx->queue_family_decode_index = (decode_family != UINT32_MAX) ? (int)decode_family : -1;
	vk_ctx->nb_decode_queues          = (decode_family != UINT32_MAX) ? 1 : 0;
#if defined(__GNUC__) || defined(__clang__)
	#pragma GCC diagnostic pop
#endif
#endif

	int ret = av_hwdevice_ctx_init(hw_device_ctx);
	if (ret < 0) {
		char err_buf[256];
		av_strerror(ret, err_buf, sizeof(err_buf));
		fprintf(stderr, "[video] Failed to init shared Vulkan hw context: %s\n", err_buf);
		av_buffer_unref(&hw_device_ctx);
		return false;
	}

	// Resolve timeline semaphore wait function (Vulkan 1.2 core or KHR extension)
	VkDevice device = vk_ctx->act_dev;
	PFN_vkWaitSemaphores pfn_wait = (PFN_vkWaitSemaphores)vkGetDeviceProcAddr(device, "vkWaitSemaphores");
	if (!pfn_wait)
		pfn_wait = (PFN_vkWaitSemaphores)vkGetDeviceProcAddr(device, "vkWaitSemaphoresKHR");

	*out_ctx = (_vulkan_hw_context_t){
		.hw_device_ctx      = hw_device_ctx,
		.pfn_wait_semaphores = pfn_wait,
		.decode_family      = decode_family,
	};
	return true;
}

///////////////////////////////////////////////////////////////////////////////
// Vulkan helpers
///////////////////////////////////////////////////////////////////////////////

static VkImageView _create_plane_view(VkImage image, VkFormat format, VkImageAspectFlagBits plane_aspect) {
	VkImageViewCreateInfo view_info = {
		.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image    = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format   = format,
		.subresourceRange = {
			.aspectMask     = plane_aspect,
			.baseMipLevel   = 0,
			.levelCount     = 1,
			.baseArrayLayer = 0,
			.layerCount     = 1,
		},
	};
	VkImageView view = VK_NULL_HANDLE;
	VkResult vr = vkCreateImageView(skr_get_vk_device(), &view_info, NULL, &view);
	if (vr != VK_SUCCESS) {
		fprintf(stderr, "[video] _create_plane_view failed: vr=%d format=%d aspect=0x%x\n", vr, format, plane_aspect);
	}
	return view;
}

// Wait for FFmpeg's timeline semaphores to signal that decode is complete.
// Unlike vkQueueWaitIdle, this only waits for the specific decode operation
// and doesn't stall the main thread's frame pipeline.
static void _wait_vk_frame_semaphores(PFN_vkWaitSemaphores pfn_wait, VkDevice device, AVVkFrame* vk_frame) {
	if (!pfn_wait) return;

	VkSemaphore sems[AV_NUM_DATA_POINTERS];
	uint64_t    vals[AV_NUM_DATA_POINTERS];
	uint32_t    sem_count = 0;

	for (int32_t i = 0; i < AV_NUM_DATA_POINTERS && vk_frame->sem[i] != VK_NULL_HANDLE; i++) {
		sems[sem_count] = vk_frame->sem[i];
		vals[sem_count] = vk_frame->sem_value[i];
		sem_count++;
	}
	if (sem_count == 0) return;

	VkSemaphoreWaitInfo wait_info = {
		.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.semaphoreCount = sem_count,
		.pSemaphores    = sems,
		.pValues        = vals,
	};
	pfn_wait(device, &wait_info, UINT64_MAX);
}

// Detect actual VkImage dimensions on first frame (hw_frames_ctx is lazily
// updated by the codec, only valid after first decode). Returns true if
// coded dimensions differ from display dimensions.
static bool _detect_coded_dimensions(
	AVCodecContext* codec_ctx,
	int32_t         display_width,
	int32_t         display_height,
	int32_t*        out_coded_width,
	int32_t*        out_coded_height)
{
	if (!codec_ctx->hw_frames_ctx) return false;

	AVHWFramesContext *fc = (AVHWFramesContext *)codec_ctx->hw_frames_ctx->data;
	int32_t cw = (fc->width  > display_width)  ? fc->width  : display_width;
	int32_t ch = (fc->height > display_height) ? fc->height : display_height;

	*out_coded_width  = cw;
	*out_coded_height = ch;
	return (cw != display_width || ch != display_height);
}

// Update zero-copy textures from FFmpeg's VkImages. For multiplane NV12,
// creates plane-specific views from a single VkImage. For separate images,
// lets sk_renderer create views. skr_tex_update_external handles destroying
// old VkImageViews via deferred destruction.
static void _update_zero_copy_textures(
	AVVkFrame*  vk_frame,
	int32_t     display_width,
	int32_t     display_height,
	skr_tex_t*  ref_tex_y,
	skr_tex_t*  ref_tex_uv)
{
	skr_tex_sampler_t sampler = {
		.sample  = skr_tex_sample_linear,
		.address = skr_tex_address_clamp,
	};

	bool        multiplane = (vk_frame->img[1] == VK_NULL_HANDLE);
	VkImage     y_img      = vk_frame->img[0];
	VkImage     uv_img     = multiplane ? vk_frame->img[0] : vk_frame->img[1];
	VkImageLayout y_layout  = vk_frame->layout[0];
	VkImageLayout uv_layout = multiplane ? vk_frame->layout[0] : vk_frame->layout[1];

	// For multiplane NV12, create plane-specific views from the single image.
	// For separate images, let skr_tex handle view creation (VK_NULL_HANDLE).
	// Note: skr_tex_update_external defers destruction of old views, so creating
	// new views each frame is safe.
	VkImageView y_view  = multiplane ? _create_plane_view(y_img, VK_FORMAT_R8_UNORM,   VK_IMAGE_ASPECT_PLANE_0_BIT) : VK_NULL_HANDLE;
	VkImageView uv_view = multiplane ? _create_plane_view(y_img, VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT) : VK_NULL_HANDLE;

	// Y plane (R8, full resolution)
	if (!skr_tex_is_valid(ref_tex_y)) {
		skr_tex_create_external_vk((skr_tex_external_info_t){
			.image          = y_img,
			.view           = y_view,
			.memory         = VK_NULL_HANDLE,
			.format         = skr_tex_fmt_r8,
			.size           = {display_width, display_height, 1},
			.current_layout = y_layout,
			.sampler        = sampler,
			.owns_image     = false,
		}, ref_tex_y);
	} else {
		skr_tex_update_external(ref_tex_y, (skr_tex_external_update_t){
			.image          = y_img,
			.view           = y_view,
			.current_layout = y_layout,
		});
	}

	// UV plane (RG8, half resolution)
	if (!skr_tex_is_valid(ref_tex_uv)) {
		skr_tex_create_external_vk((skr_tex_external_info_t){
			.image          = uv_img,
			.view           = uv_view,
			.memory         = VK_NULL_HANDLE,
			.format         = skr_tex_fmt_r8g8,
			.size           = {display_width / 2, display_height / 2, 1},
			.current_layout = uv_layout,
			.sampler        = sampler,
			.owns_image     = false,
		}, ref_tex_uv);
	} else {
		skr_tex_update_external(ref_tex_uv, (skr_tex_external_update_t){
			.image          = uv_img,
			.view           = uv_view,
			.current_layout = uv_layout,
		});
	}
}

///////////////////////////////////////////////////////////////////////////////
// Software decode helpers
///////////////////////////////////////////////////////////////////////////////

static void _create_software_textures(
	int32_t    width,
	int32_t    height,
	skr_tex_t* out_tex_y,
	skr_tex_t* out_tex_uv)
{
	skr_tex_sampler_t sampler = {
		.sample  = skr_tex_sample_linear,
		.address = skr_tex_address_clamp,
	};

	skr_tex_create(
		skr_tex_fmt_r8,
		skr_tex_flags_dynamic,
		sampler,
		(skr_vec3i_t){width, height, 1},
		1, 1, NULL, out_tex_y);
	skr_tex_set_name(out_tex_y, "video_y");

	skr_tex_create(
		skr_tex_fmt_r8g8,
		skr_tex_flags_dynamic,
		sampler,
		(skr_vec3i_t){width / 2, height / 2, 1},
		1, 1, NULL, out_tex_uv);
	skr_tex_set_name(out_tex_uv, "video_uv");
}

static void _upload_software_frame(
	int32_t    width,
	int32_t    height,
	skr_tex_t* ref_tex_y,
	skr_tex_t* ref_tex_uv,
	AVFrame*   frame)
{
	int32_t uv_width  = width / 2;
	int32_t uv_height = height / 2;

	if (frame->format == AV_PIX_FMT_NV12) {
		// NV12: Y plane (full res) + interleaved UV plane (half res)
		skr_tex_set_data(ref_tex_y,  &(skr_tex_data_t){.data = frame->data[0], .mip_count = 1, .layer_count = 1, .row_pitch = frame->linesize[0]});
		skr_tex_set_data(ref_tex_uv, &(skr_tex_data_t){.data = frame->data[1], .mip_count = 1, .layer_count = 1, .row_pitch = frame->linesize[1]});

	} else if (frame->format == AV_PIX_FMT_YUV420P) {
		// Planar YUV420: Y + U + V separate planes
		skr_tex_set_data(ref_tex_y, &(skr_tex_data_t){.data = frame->data[0], .mip_count = 1, .layer_count = 1, .row_pitch = frame->linesize[0]});

		// Interleave U and V into UV texture (RG8 format)
		uint8_t* uv_buffer = malloc(uv_width * uv_height * 2);
		if (uv_buffer) {
			uint8_t* u_plane = frame->data[1];
			uint8_t* v_plane = frame->data[2];
			int32_t  u_pitch = frame->linesize[1];
			int32_t  v_pitch = frame->linesize[2];

			for (int32_t y = 0; y < uv_height; y++) {
				uint8_t* dst   = uv_buffer + y * uv_width * 2;
				uint8_t* u_row = u_plane + y * u_pitch;
				uint8_t* v_row = v_plane + y * v_pitch;
				for (int32_t x = 0; x < uv_width; x++) {
					dst[x * 2 + 0] = u_row[x];  // U -> R
					dst[x * 2 + 1] = v_row[x];  // V -> G
				}
			}
			skr_tex_set_data(ref_tex_uv, &(skr_tex_data_t){.data = uv_buffer, .mip_count = 1, .layer_count = 1});
			free(uv_buffer);
		}
	} else {
		fprintf(stderr, "[video] Unsupported pixel format: %d\n", frame->format);
	}
}

///////////////////////////////////////////////////////////////////////////////
// Material helpers
///////////////////////////////////////////////////////////////////////////////

static void _update_material(
	skr_material_t* ref_material,
	skr_tex_t*      tex_y,
	skr_tex_t*      tex_uv,
	int32_t         display_width,
	int32_t         display_height,
	int32_t         coded_width,
	int32_t         coded_height)
{
	skr_material_set_tex(ref_material, "tex_y",  tex_y);
	skr_material_set_tex(ref_material, "tex_uv", tex_uv);

	int32_t cw = coded_width  > 0 ? coded_width  : display_width;
	int32_t ch = coded_height > 0 ? coded_height : display_height;
	float uv_crop[2] = {
		(cw > 0) ? (float)display_width  / (float)cw : 1.0f,
		(ch > 0) ? (float)display_height / (float)ch : 1.0f,
	};
	skr_material_set_param(ref_material, "uv_crop", sksc_shader_var_float, 2, uv_crop);
}

///////////////////////////////////////////////////////////////////////////////
// Public API
///////////////////////////////////////////////////////////////////////////////

// FFmpeg interrupt callback: allows aborting blocking I/O (e.g. network reads)
// when a new seek arrives.
static int _interrupt_cb(void* opaque) {
	video_t* v = opaque;
	return atomic_load(&v->abort_decode) ? 1 : 0;
}

video_t* video_open(const char* uri) {
	video_t* v = calloc(1, sizeof(video_t));
	if (!v) return NULL;

	// Set up network options for URLs
	AVDictionary* opts = NULL;
	bool is_url = strstr(uri, "://") != NULL;
	if (is_url) {
		av_dict_set(&opts, "timeout",             "5000000", 0);
		av_dict_set(&opts, "reconnect",           "1",       0);
		av_dict_set(&opts, "reconnect_streamed",  "1",       0);
		av_dict_set(&opts, "reconnect_delay_max", "5",       0);
	}

	// Open input file/URL
	int ret = avformat_open_input(&v->format_ctx, uri, NULL, &opts);
	av_dict_free(&opts);
	if (ret < 0) {
		fprintf(stderr, "[video] Failed to open: %s\n", uri);
		goto cleanup;
	}

	// Set interrupt callback so blocking I/O can be aborted via video_abort_decode()
	v->format_ctx->interrupt_callback.callback = _interrupt_cb;
	v->format_ctx->interrupt_callback.opaque   = v;

	// Find stream info
	ret = avformat_find_stream_info(v->format_ctx, NULL);
	if (ret < 0) {
		fprintf(stderr, "[video] Failed to find stream info\n");
		goto cleanup;
	}

	// Find best video stream
	v->stream_idx = av_find_best_stream(v->format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	if (v->stream_idx < 0) {
		fprintf(stderr, "[video] No video stream found\n");
		goto cleanup;
	}

	AVStream* stream = v->format_ctx->streams[v->stream_idx];
	v->time_base = av_q2d(stream->time_base);
	v->framerate = av_q2d(stream->avg_frame_rate);

	// Handle duration - live streams have unknown/infinite duration
	if (v->format_ctx->duration == AV_NOPTS_VALUE || v->format_ctx->duration <= 0) {
		v->duration = 0.0;
		v->is_live  = true;
	} else {
		v->duration = (double)v->format_ctx->duration / AV_TIME_BASE;
		v->is_live  = false;
	}

	// Check if seeking is supported
	v->is_seekable = !v->is_live && v->format_ctx->pb && (v->format_ctx->pb->seekable & AVIO_SEEKABLE_NORMAL);

	// Find decoder
	const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
	if (!codec) {
		fprintf(stderr, "[video] No decoder found for codec\n");
		goto cleanup;
	}

	// Allocate codec context
	v->codec_ctx = avcodec_alloc_context3(codec);
	if (!v->codec_ctx) {
		fprintf(stderr, "[video] Failed to allocate codec context\n");
		goto cleanup;
	}

	ret = avcodec_parameters_to_context(v->codec_ctx, stream->codecpar);
	if (ret < 0) {
		fprintf(stderr, "[video] Failed to copy codec parameters\n");
		goto cleanup;
	}

	v->width  = v->codec_ctx->width;
	v->height = v->codec_ctx->height;

	// Try Vulkan hardware acceleration (shares sk_renderer's VkDevice)
	int32_t ext_count = 0;
	v->hw_accel = _init_vulkan_hwcontext(&v->hw_ctx, v->enabled_exts, &ext_count);
	if (v->hw_accel) {
		v->zero_copy                = true;
		v->codec_ctx->hw_device_ctx = av_buffer_ref(v->hw_ctx.hw_device_ctx);
		v->codec_ctx->get_format    = _get_hw_format;
	}

	// Open codec
	ret = avcodec_open2(v->codec_ctx, codec, NULL);
	if (ret < 0) {
		fprintf(stderr, "[video] Failed to open codec: %d\n", ret);
		goto cleanup;
	}

	// Coded dimensions detected on first frame decode
	v->coded_width  = v->width;
	v->coded_height = v->height;

	// Allocate frames and packet
	v->frame      = av_frame_alloc();
	v->held_frame = av_frame_alloc();
	v->sw_frame   = av_frame_alloc();
	v->packet     = av_packet_alloc();
	if (!v->frame || !v->held_frame || !v->sw_frame || !v->packet) {
		fprintf(stderr, "[video] Failed to allocate frames/packet\n");
		goto cleanup;
	}

	// Create software textures upfront for CPU decode path
	if (!v->zero_copy) {
		_create_software_textures(v->width, v->height, &v->sw_tex_y, &v->sw_tex_uv);
	}

	// Create shader and material for video rendering
	v->shader = su_shader_load("shaders/video.hlsl.sks", "video");
	skr_material_create((skr_material_info_t){
		.shader     = &v->shader,
		.cull       = skr_cull_none,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_always,
	}, &v->material);

	v->valid = true;

	// Decode first frame to avoid green flash on first render
	// (uninitialized UV textures produce green in YUV->RGB conversion).
	// For network streams, this may block briefly.
	video_decode_next_frame(v);

	return v;

cleanup:
	video_destroy(v);
	return NULL;
}

void video_destroy(video_t* v) {
	if (!v) return;

	// Destroy rendering resources
	skr_material_destroy(&v->material);
	skr_shader_destroy(&v->shader);

	// Destroy textures
	skr_tex_destroy(&v->tex_y);
	skr_tex_destroy(&v->tex_uv);
	skr_tex_destroy(&v->sw_tex_y);
	skr_tex_destroy(&v->sw_tex_uv);

	// Free FFmpeg resources
	if (v->packet)             av_packet_free      (&v->packet);
	if (v->sw_frame)           av_frame_free       (&v->sw_frame);
	if (v->held_frame)         av_frame_free       (&v->held_frame);
	if (v->frame)              av_frame_free       (&v->frame);
	if (v->hw_ctx.hw_device_ctx) av_buffer_unref   (&v->hw_ctx.hw_device_ctx);
	if (v->codec_ctx)          avcodec_free_context(&v->codec_ctx);
	if (v->format_ctx)         avformat_close_input(&v->format_ctx);

	free(v);
}

bool video_is_valid(const video_t* v) {
	return v && v->valid;
}

int32_t video_get_width(const video_t* v) {
	return v ? v->width : 0;
}

int32_t video_get_height(const video_t* v) {
	return v ? v->height : 0;
}

double video_get_duration(const video_t* v) {
	return v ? v->duration : 0.0;
}

double video_get_framerate(const video_t* v) {
	return v ? v->framerate : 0.0;
}

double video_get_current_time(const video_t* v) {
	return v ? v->current_pts : 0.0;
}

bool video_is_live(const video_t* v) {
	return v ? v->is_live : false;
}

bool video_is_seekable(const video_t* v) {
	return v ? v->is_seekable : false;
}

bool video_is_hw_accelerated(const video_t* v) {
	return v && v->hw_accel;
}

void video_abort_decode(video_t* v) {
	if (v) atomic_store(&v->abort_decode, true);
}

video_decode_status_ video_decode_next_frame(video_t* v) {
	if (!v || !v->valid) return video_decode_error;
	if (v->eof)          return video_decode_eof;

	// Clear abort flag so av_read_frame can proceed normally
	atomic_store(&v->abort_decode, false);

	// Flush decoder if a seek was requested. Deferred from video_seek to
	// coalesce rapid seeks (e.g. slider drag) into one flush.
	if (v->needs_flush) {
		avcodec_flush_buffers(v->codec_ctx);
		v->needs_flush = false;
	}

	while (true) {
		if (atomic_load(&v->abort_decode)) return video_decode_aborted;

		int ret = avcodec_receive_frame(v->codec_ctx, v->frame);
		if (ret == 0) {
			// Got a frame
			v->current_pts = v->frame->pts * v->time_base;

			if (v->frame->format == AV_PIX_FMT_VULKAN && v->hw_accel) {
				AVVkFrame* vk_frame = (AVVkFrame*)v->frame->data[0];
				if (vk_frame && v->zero_copy) {
					// Detect coded dimensions on first frame
					if (v->coded_width == v->width && v->coded_height == v->height) {
						if (_detect_coded_dimensions(v->codec_ctx, v->width, v->height, &v->coded_width, &v->coded_height)) {
							printf("[video] Display: %dx%d, VkImage: %dx%d (UV crop: %.4f, %.4f)\n",
								v->width, v->height, v->coded_width, v->coded_height,
								(float)v->width / (float)v->coded_width, (float)v->height / (float)v->coded_height);
						}
					}

					_wait_vk_frame_semaphores(v->hw_ctx.pfn_wait_semaphores, skr_get_vk_device(), vk_frame);
					_update_zero_copy_textures(vk_frame, v->width, v->height, &v->tex_y, &v->tex_uv);

					_update_material(&v->material, &v->tex_y, &v->tex_uv, v->width, v->height, v->coded_width, v->coded_height);
					v->material_ready = true;
				} else if (vk_frame) {
					// Fallback: transfer through CPU (hw decode but no zero-copy)
					int xfer = av_hwframe_transfer_data(v->sw_frame, v->frame, 0);
					if (xfer >= 0) {
						_upload_software_frame(v->width, v->height, &v->sw_tex_y, &v->sw_tex_uv, v->sw_frame);
						_update_material(&v->material, &v->sw_tex_y, &v->sw_tex_uv, v->width, v->height, v->coded_width, v->coded_height);
						v->material_ready = true;
					} else {
						fprintf(stderr, "[video] Failed to transfer hw frame to CPU\n");
					}
				}
			} else {
				// Software decoded frame
				_upload_software_frame(v->width, v->height, &v->sw_tex_y, &v->sw_tex_uv, v->frame);
				_update_material(&v->material, &v->sw_tex_y, &v->sw_tex_uv, v->width, v->height, v->coded_width, v->coded_height);
				v->material_ready = true;
			}

			// Hold a reference to the displayed frame so its VkImages survive
			// avcodec_flush_buffers during seek.
			av_frame_unref(v->held_frame);
			av_frame_ref  (v->held_frame, v->frame);

			return video_decode_ok;
		} else if (ret == AVERROR(EAGAIN)) {
			// Need more input
		} else if (ret == AVERROR_EOF) {
			v->eof = true;
			return video_decode_eof;
		} else {
			return video_decode_error;
		}

		// Read next packet
		ret = av_read_frame(v->format_ctx, v->packet);
		if (ret < 0) {
			if (ret == AVERROR_EOF) {
				avcodec_send_packet(v->codec_ctx, NULL);
				v->eof = true;
			}
			continue;
		}

		// Skip non-video packets
		if (v->packet->stream_index != v->stream_idx) {
			av_packet_unref(v->packet);
			continue;
		}

		ret = avcodec_send_packet(v->codec_ctx, v->packet);
		av_packet_unref(v->packet);

		if (ret < 0 && ret != AVERROR(EAGAIN)) {
			return video_decode_error;
		}
	}
}

bool video_seek(video_t* v, double time_seconds) {
	if (!v || !v->valid || !v->is_seekable) return false;

	int64_t timestamp = (int64_t)(time_seconds * AV_TIME_BASE);
	int ret = av_seek_frame(v->format_ctx, -1, timestamp, AVSEEK_FLAG_BACKWARD);
	if (ret < 0) return false;

	// Defer the flush until the next decode. During slider drag, video_seek
	// is called many times per second. Deferring coalesces all seeks into
	// a single flush right before the first actual decode.
	v->needs_flush = true;
	v->eof         = false;
	v->current_pts = time_seconds;

	return true;
}

skr_material_t* video_get_material(video_t* v) {
	if (!v || !v->material_ready) return NULL;
	return &v->material;
}

///////////////////////////////////////////////////////////////////////////////
// Thumbnail extraction
///////////////////////////////////////////////////////////////////////////////

skr_tex_t video_extract_thumbnail(const char* filename, int32_t max_size) {
	skr_tex_t result = {0};
	if (!filename || max_size <= 0) return result;

	// Open format context (lightweight, no hw accel)
	AVFormatContext* fmt_ctx = NULL;
	int ret = avformat_open_input(&fmt_ctx, filename, NULL, NULL);
	if (ret < 0) return result;

	ret = avformat_find_stream_info(fmt_ctx, NULL);
	if (ret < 0) { avformat_close_input(&fmt_ctx); return result; }

	int stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	if (stream_idx < 0) { avformat_close_input(&fmt_ctx); return result; }

	AVStream* stream = fmt_ctx->streams[stream_idx];
	const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
	if (!codec) { avformat_close_input(&fmt_ctx); return result; }

	AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
	if (!codec_ctx) { avformat_close_input(&fmt_ctx); return result; }

	ret = avcodec_parameters_to_context(codec_ctx, stream->codecpar);
	if (ret < 0) { avcodec_free_context(&codec_ctx); avformat_close_input(&fmt_ctx); return result; }

	ret = avcodec_open2(codec_ctx, codec, NULL);
	if (ret < 0) { avcodec_free_context(&codec_ctx); avformat_close_input(&fmt_ctx); return result; }

	// Decode first frame
	AVFrame*  frame  = av_frame_alloc();
	AVPacket* packet = av_packet_alloc();
	if (!frame || !packet) goto thumb_cleanup;

	bool got_frame = false;
	while (!got_frame) {
		ret = av_read_frame(fmt_ctx, packet);
		if (ret < 0) break;

		if (packet->stream_index != stream_idx) {
			av_packet_unref(packet);
			continue;
		}

		ret = avcodec_send_packet(codec_ctx, packet);
		av_packet_unref(packet);
		if (ret < 0) break;

		ret = avcodec_receive_frame(codec_ctx, frame);
		if (ret == 0) got_frame = true;
		else if (ret != AVERROR(EAGAIN)) break;
	}

	if (!got_frame) goto thumb_cleanup;

	// Calculate thumbnail dimensions (fit within max_size)
	int32_t src_w = codec_ctx->width;
	int32_t src_h = codec_ctx->height;
	int32_t dst_w, dst_h;

	if (src_w >= src_h) {
		dst_w = max_size;
		dst_h = (int32_t)((float)src_h / (float)src_w * max_size);
	} else {
		dst_h = max_size;
		dst_w = (int32_t)((float)src_w / (float)src_h * max_size);
	}
	if (dst_w < 1) dst_w = 1;
	if (dst_h < 1) dst_h = 1;

	// Convert to RGBA and scale
	struct SwsContext* sws = sws_getContext(
		src_w, src_h, frame->format,
		dst_w, dst_h, AV_PIX_FMT_RGBA,
		SWS_BILINEAR, NULL, NULL, NULL);
	if (!sws) goto thumb_cleanup;

	uint8_t* rgba_data = malloc(dst_w * dst_h * 4);
	if (!rgba_data) { sws_freeContext(sws); goto thumb_cleanup; }

	uint8_t* dst_planes[1] = { rgba_data };
	int      dst_stride[1] = { dst_w * 4 };
	sws_scale(sws, (const uint8_t* const*)frame->data, frame->linesize, 0, src_h, dst_planes, dst_stride);
	sws_freeContext(sws);

	// Create texture from RGBA data
	skr_tex_create(
		skr_tex_fmt_rgba32_srgb,
		skr_tex_flags_none,
		(skr_tex_sampler_t){ .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp },
		(skr_vec3i_t){ dst_w, dst_h, 1 },
		1, 1, NULL, &result);
	skr_tex_set_data(&result, &(skr_tex_data_t){ .data = rgba_data, .mip_count = 1, .layer_count = 1 });
	skr_tex_set_name(&result, "video_thumbnail");

	free(rgba_data);

thumb_cleanup:
	if (packet) av_packet_free(&packet);
	if (frame)  av_frame_free(&frame);
	avcodec_free_context(&codec_ctx);
	avformat_close_input(&fmt_ctx);
	return result;
}
