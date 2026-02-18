//--name = gi_debug_voxel

// Renders instanced cubes colored by voxel radiance.
// Each cube = one voxel cell. Empty voxels are discarded.
// Voxel position derived from SV_InstanceID (no instance data needed).

#include "common.hlsli"
#include "gi_voxel.hlsli"

///////////////////////////////////////////
// GI Probe Buffer (b12)
///////////////////////////////////////////

cbuffer GIBuffer : register(b12, space0) {
	float3 gi_volume_min;
	float  gi_intensity;
	float3 gi_volume_inv; // 1.0 / (volume_max - volume_min)
	uint   gi_grid_size;
};

///////////////////////////////////////////
// Voxel Texture
///////////////////////////////////////////

Texture3D<float4> gi_voxel_tex : register(t9);

///////////////////////////////////////////
// Vertex/Pixel Shader I/O
///////////////////////////////////////////

struct vsIn {
	float3 pos   : SV_POSITION;
	float3 norm  : NORMAL0;
	float2 uv    : TEXCOORD0;
	float4 color : COLOR0;
};

struct psIn {
	float4                     pos     : SV_POSITION;
	float3                     normal  : NORMAL0;
	nointerpolation float3     vox_col : TEXCOORD0;
	uint                       layer   : SV_RenderTargetArrayIndex;
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn output;

	uint view_idx = id % view_count;
	uint inst_idx = id / view_count;

	// Derive 3D voxel coordinate from linear instance index
	int3 vpos = int3(
		inst_idx % GI_VOXEL_RES,
		(inst_idx / GI_VOXEL_RES) % GI_VOXEL_RES,
		inst_idx / (GI_VOXEL_RES * GI_VOXEL_RES));

	float4 voxel = gi_voxel_tex.Load(int4(vpos, 0));

	if (voxel.a <= 0) {
		output.pos     = asfloat(0x7FC00000); // NaN kills the primitive
		output.normal  = float3(0, 0, 0);
		output.vox_col = float3(0, 0, 0);
		output.layer   = view_idx;
		return output;
	}

	output.vox_col = voxel.rgb;

	// Compute world position from voxel coordinate
	float3 vol_size = 1.0 / gi_volume_inv;
	float3 cell     = vol_size / (float)GI_VOXEL_RES;
	float3 center   = gi_volume_min + (float3(vpos) + 0.5) * cell;

	float3 world_pos = input.pos * cell + center;
	output.pos       = mul(float4(world_pos, 1), viewproj[view_idx]);
	output.normal    = input.norm;
	output.layer     = view_idx;
	return output;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	return float4(input.vox_col, 1.0);
}
