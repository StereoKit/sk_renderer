//--name = gi_debug
//--debug_mode = 0

// Debug visualization shader for GI light probes.
// Renders instanced spheres colored by the SH irradiance at each probe position.
// debug_mode: 0=SH irradiance, 1=voxel occupancy, 2=raw L0, 3=SH detail,
//             4=voxel radiance

#include "common.hlsli"
#include "gi_voxel.hlsli"

uint debug_mode;

///////////////////////////////////////////
// Instance Data
///////////////////////////////////////////

struct Inst {
	float4 pos_scale; // xyz=position, w=uniform scale
};
 StructuredBuffer<Inst> inst : register(t2, space0);

///////////////////////////////////////////
// GI Buffers
///////////////////////////////////////////

StructuredBuffer<SHProbe> gi_sh_probes : register(t6);
Texture3D<float4>         gi_voxel_tex : register(t9);

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
	float3 world_pos : TEXCOORD0;
	uint   layer     : SV_RenderTargetArrayIndex;
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn output;

	uint view_idx = id % view_count;
	uint inst_idx = id / view_count;

	float  scale = inst[inst_idx].pos_scale.w;
	float3 center = inst[inst_idx].pos_scale.xyz;
	output.world_pos = input.pos * scale + center;
	output.pos       = mul(float4(output.world_pos, 1), viewproj[view_idx]);
	output.normal    = input.norm; // uniform scale, normal unchanged
	output.layer     = view_idx;
	return output;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	// World position -> probe volume UVW [0,1]
	float3 uvw = (input.world_pos - gi_cascades[0].volume_min) * gi_cascades[0].volume_inv;
	uvw = saturate(uvw);

	// Voxel grid position (higher res)
	uint3 voxel_pos = uint3(clamp(uvw * (float)GI_VOXEL_RES, float3(0,0,0),
	                               float3(GI_VOXEL_RES-1, GI_VOXEL_RES-1, GI_VOXEL_RES-1)));
	// Probe grid position (lower res)
	uint3 probe_pos = uint3(clamp(uvw * (float)GI_GRID, float3(0,0,0),
	                               float3(GI_GRID-1, GI_GRID-1, GI_GRID-1)));
	uint lidx = voxel_index(probe_pos);

	// Mode 1: voxel occupancy (occupied = white, empty = discarded)
	if (debug_mode == 1) {
		float4 voxel = gi_voxel_tex.Load(int4(voxel_pos, 0));
		if (voxel.a <= 0) discard;
		return float4(1.0, 1.0, 1.0, 1.0);
	}

	// Mode 4: voxel radiance (color from 3D texture)
	if (debug_mode == 4) {
		float4 voxel = gi_voxel_tex.Load(int4(voxel_pos, 0));
		if (voxel.a <= 0) discard;
		return float4(voxel.rgb, 1.0);
	}

	// Sample SH coefficients (nearest probe, linear-indexed)
	SHProbe sh  = gi_sh_probes[lidx];
	float4  shr = sh_unpack(sh.r);
	float4  shg = sh_unpack(sh.g);
	float4  shb = sh_unpack(sh.b);

	// Mode 2: show raw L0 coefficient (ambient, direction-independent)
	if (debug_mode == 2) {
		return float4(shr.x, shg.x, shb.x, 1.0);
	}

	// Mode 3: show all 4 SH coefficients of R channel as color
	if (debug_mode == 3) {
		return float4(abs(shr.x), abs(shr.y) + abs(shr.z) + abs(shr.w), 0, 1.0);
	}

	// Mode 0 (default): evaluate SH for surface normal direction
	float4 sh_basis = float4(
		0.88623,
		1.02333 * input.normal.y,
		1.02333 * input.normal.z,
		1.02333 * input.normal.x
	);

	float3 irradiance = float3(
		dot(shr, sh_basis),
		dot(shg, sh_basis),
		dot(shb, sh_basis)
	);
	irradiance = max(float3(0, 0, 0), irradiance) * gi_intensity;

	return float4(irradiance, 1.0);
}
