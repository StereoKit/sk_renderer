//--name = gi_debug
//--debug_mode = 0

// Debug visualization shader for GI light probes.
// Renders instanced spheres colored by the SH irradiance at each probe position.
// debug_mode: 0=SH irradiance, 1=UVW coords, 2=raw L0, 3=SH detail,
//             4=voxel radiance, 5=voxel opacity

#include "common.hlsli"

uint debug_mode;

///////////////////////////////////////////
// Instance Data
///////////////////////////////////////////

struct Inst {
	float4 pos_scale; // xyz=position, w=uniform scale
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
// GI SH Textures
///////////////////////////////////////////

Texture3D<float4> gi_sh_r   : register(t6);
SamplerState      gi_sh_r_s : register(s6);

Texture3D<float4> gi_sh_g   : register(t7);
SamplerState      gi_sh_g_s : register(s7);

Texture3D<float4> gi_sh_b   : register(t8);
SamplerState      gi_sh_b_s : register(s8);

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
	float3 uvw = (input.world_pos - gi_volume_min) * gi_volume_inv;
	uvw = saturate(uvw);

	// Mode 1: show UVW coordinates as RGB (verify spatial mapping)
	if (debug_mode == 1) {
		return float4(uvw, 1.0);
	}

	// Snap to texel center for voxel modes — the sphere surface has physical
	// extent so raw UVW varies across the sphere, causing linear filter blending
	// at texel boundaries. Snapping ensures each sphere shows its exact texel.
	float3 voxel_uvw = (floor(uvw * 32.0) + 0.5) / 32.0;

	// Mode 4: voxel radiance (normalized rgb)
	if (debug_mode == 4) {
		float4 voxel = gi_voxel.SampleLevel(gi_voxel_s, voxel_uvw, 0);
		float3 col = voxel.a > 0.01 ? voxel.rgb / voxel.a : float3(0, 0, 0);
		if (voxel.a < 0.01) discard;
		return float4(col, 1.0);
	}

	// Mode 5: voxel opacity (alpha as grayscale)
	if (debug_mode == 5) {
		float4 voxel = gi_voxel.SampleLevel(gi_voxel_s, voxel_uvw, 0);
		float  a = saturate(voxel.a);
		return float4(a, a, a, 1.0);
	}

	// Sample SH coefficients
	float4 shr = gi_sh_r.SampleLevel(gi_sh_r_s, uvw, 0);
	float4 shg = gi_sh_g.SampleLevel(gi_sh_g_s, uvw, 0);
	float4 shb = gi_sh_b.SampleLevel(gi_sh_b_s, uvw, 0);

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
