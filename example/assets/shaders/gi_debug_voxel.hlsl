//--name = gi_debug_voxel

// Renders instanced cubes colored by voxel radiance.
// Each cube = one voxel cell. Empty voxels are discarded.

#include "common.hlsli"

///////////////////////////////////////////
// Instance Data
///////////////////////////////////////////

float3 cell_size; // voxel cell dimensions (set per-material, same for all)

struct Inst {
	float3 pos;   // xyz=voxel center position
	float  _pad;
};
StructuredBuffer<Inst> inst : register(t2, space0);

///////////////////////////////////////////
// GI Probe Buffer (b12)
///////////////////////////////////////////

cbuffer GIBuffer : register(b12, space0) {
	float3 gi_volume_min;
	float  gi_intensity;
	float3 gi_volume_inv; // 1.0 / (volume_max - volume_min)
	float  _gi_pad;
};

///////////////////////////////////////////
// Voxel Texture
///////////////////////////////////////////

Texture3D<float4> gi_voxel   : register(t9);
SamplerState      gi_voxel_s : register(s9);

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
	float4 pos       : SV_POSITION;
	float3 normal    : NORMAL0;
	float3 color     : TEXCOORD0;
	SKR_LAYER_OUTPUT
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, skr_input_t sys) {
	skr_ids_t ids = skr_resolve_ids(sys);

	psIn output;

	float3 center = inst[ids.inst].pos;

	// Sample voxel from instance center position
	float3 uvw       = (center - gi_volume_min) * gi_volume_inv;
	float3 voxel_uvw = (floor(saturate(uvw) * 32.0) + 0.5) / 32.0;
	float4 voxel     = gi_voxel.SampleLevel(gi_voxel_s, voxel_uvw, 0);

	uint face_mask = uint(voxel.a + 0.5);
	if (face_mask == 0) {
		output.pos    = asfloat(0x7FC00000); // NaN kills the primitive
		output.normal = float3(0, 0, 0);
		output.color  = float3(0, 0, 0);
		SKR_SET_LAYER(output, ids.view);
		return output;
	}

	uint  count = countbits(face_mask);
	output.color = voxel.rgb / (float)count;

	float3 world_pos = input.pos * cell_size + center;
	output.pos       = mul(float4(world_pos, 1), viewproj[ids.view]);
	output.normal    = input.norm;
	SKR_SET_LAYER(output, ids.view);
	return output;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	return float4(input.color, 1.0);
}
