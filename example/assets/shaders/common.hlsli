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
	uint     view_offset;        // Unused, kept for ABI compatibility
	uint     _pad;
};

///////////////////////////////////////////
// Multi-view abstractions (VK_KHR_multiview)
///////////////////////////////////////////

// System vertex IDs — multiview provides SV_ViewID via the driver,
// SV_InstanceID is the real instance index (no view packing).
struct skr_ids_t {
	uint inst : SV_InstanceID;
	uint view : SV_ViewID;
};

#endif // COMMON_HLSLI
