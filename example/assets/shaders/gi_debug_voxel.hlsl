//--name = gi_debug_voxel

// Renders instanced cubes colored by voxel radiance.
// Each cube = one voxel cell. Empty voxels are discarded.

#include "common.hlsli"
#include "gi_voxel.hlsli"

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
	uint   gi_grid_size;
};

///////////////////////////////////////////
// Voxel Buffer
///////////////////////////////////////////

StructuredBuffer<Voxel> gi_voxel_buf : register(t9);

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
	float4             pos       : SV_POSITION;
	float3             normal    : NORMAL0;
	nointerpolation uint vox_idx   : TEXCOORD0;
	nointerpolation uint face_mask : TEXCOORD1;
	uint               layer     : SV_RenderTargetArrayIndex;
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn output;

	uint view_idx = id % view_count;
	uint inst_idx = id / view_count;

	float3 center = inst[inst_idx].pos;

	// Compute voxel grid position from world position
	float3 uvw      = (center - gi_volume_min) * gi_volume_inv;
	uint3  vpos     = uint3(clamp(uvw * (float)GI_GRID, float3(0,0,0),
	                               float3(GI_GRID-1, GI_GRID-1, GI_GRID-1)));
	uint   vox_idx  = voxel_index(vpos);
	Voxel  v        = gi_voxel_buf[vox_idx];
	uint   face_mask = voxel_face_mask(v);

	if (face_mask == 0) {
		output.pos       = asfloat(0x7FC00000); // NaN kills the primitive
		output.normal    = float3(0, 0, 0);
		output.vox_idx   = 0;
		output.face_mask = 0;
		output.layer     = view_idx;
		return output;
	}

	output.vox_idx   = vox_idx;
	output.face_mask = face_mask;

	float3 world_pos = input.pos * cell_size + center;
	output.pos       = mul(float4(world_pos, 1), viewproj[view_idx]);
	output.normal    = input.norm;
	output.layer     = view_idx;
	return output;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	// Map cube face normal to face index: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
	float3 a = abs(input.normal);
	uint face_idx;
	if      (a.x >= a.y && a.x >= a.z) face_idx = input.normal.x > 0 ? 0u : 1u;
	else if (a.y >= a.x && a.y >= a.z) face_idx = input.normal.y > 0 ? 2u : 3u;
	else                                face_idx = input.normal.z > 0 ? 4u : 5u;

	if (!(input.face_mask & (1u << face_idx))) discard;

	Voxel v = gi_voxel_buf[input.vox_idx];
	return float4(unpack_rgba8_color(voxel_get_face(v, face_idx)), 1.0);
}
