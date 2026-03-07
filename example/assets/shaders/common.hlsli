// common.hlsli - Shared definitions for sk_renderer shaders

#ifndef COMMON_HLSLI
#define COMMON_HLSLI

// System buffer - available to all shaders via register(b1, space0)
// Contains view/projection matrices and multi-view rendering state
cbuffer SystemBuffer : register(b1, space0) {
	float4x4 view          [6];  // View matrices (one per view)
	float4x4 view_inv      [6];  // Inverse view matrices
	float4x4 projection    [6];  // Per-view projection matrices
	float4x4 projection_inv[6];  // Inverse projection matrices
	float4x4 viewproj      [6];  // Precomputed view*projection matrices
	float4   cam_pos       [6];  // Camera position (xyz + padding)
	float4   cam_dir       [6];  // Camera forward direction (xyz + padding)
	float4   cubemap_info;       // .xy = size, .z = mip count, .w = unused
	float4   screen_size;        // .xy = width/height, .zw = 1/width, 1/height
	float    time;               // Time in seconds
	uint     view_count;         // Number of active views (1-6)
	uint     view_offset;        // Base view index for multi-view fallback
	uint     _pad;
};

///////////////////////////////////////////
// Multi-view variant abstractions
///////////////////////////////////////////

// Resolved instance and view indices from SV_InstanceID
struct skr_ids_t {
	uint inst;
	uint view;
};

// System vertex input — contains SV_InstanceID and (future) SV_ViewID
struct skr_input_t {
	uint instance_id : SV_InstanceID;
};

// Resolve instance and view indices from system input.
//
// Default variant: instances are multiplied by view_count, view index is
// packed into the instance ID via id % view_count.
//
// SKR_NO_LAYER_SELECT variant: renderer draws one pass per view, setting
// view_offset each pass. Instance ID is used directly.
skr_ids_t skr_resolve_ids(skr_input_t input) {
	skr_ids_t r;
#ifdef SKR_NO_LAYER_SELECT
	r.inst = input.instance_id;
	r.view = view_offset;
#else
	r.inst = input.instance_id / view_count;
	r.view = input.instance_id % view_count;
#endif
	return r;
}

// Vertex output layer declaration — include in your psIn struct.
// Expands to nothing when layer selection is unavailable.
#ifdef SKR_NO_LAYER_SELECT
	#define SKR_LAYER_OUTPUT
	#define SKR_SET_LAYER(output, val)
#else
	#define SKR_LAYER_OUTPUT  uint layer : SV_RenderTargetArrayIndex;
	#define SKR_SET_LAYER(output, val)  output.layer = val
#endif

#endif // COMMON_HLSLI
