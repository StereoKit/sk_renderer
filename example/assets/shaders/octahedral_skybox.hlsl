//--name = octahedral_skybox

#include "common.hlsli"

struct vsIn {
	float3 pos    : SV_POSITION;
	float3 normal : NORMAL;
	float2 uv     : TEXCOORD0;
	float4 color  : COLOR;
};
struct psIn {
	float4 pos   : SV_POSITION;
	float3 dir   : TEXCOORD0;
	uint   layer : SV_RenderTargetArrayIndex;
};

Texture2D    octahedral_tex     : register(t0);
SamplerState octahedral_sampler : register(s0);

// Octahedral encoding adapted for Y-up from:
// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
float2 oct_wrap(float2 v) {
	return (1.0 - abs(v.yx)) * (v.xy >= 0.0 ? 1.0 : -1.0);
}

float2 direction_to_octahedral(float3 dir) {
	dir /= (abs(dir.x) + abs(dir.y) + abs(dir.z));
	dir.xz = dir.y >= 0.0 ? dir.xz : oct_wrap(dir.xz);
	return dir.xz * 0.5 + 0.5;
}

psIn vs(vsIn input, uint id : SV_InstanceID) {
	uint view_idx = id % view_count;

	psIn output;
	output.pos   = float4(input.pos, 1.0);
	output.pos.z = 1; // Force Z to the back

	// Calculate view direction from inverse projection and view matrices
	float4 proj_inv = mul(output.pos, projection_inv[view_idx]);
	output.dir = mul(float4(proj_inv.xyz, 0), transpose(view[view_idx])).xyz;

	output.layer = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	float3 dir = normalize(input.dir);
	float2 uv  = direction_to_octahedral(dir);
	return octahedral_tex.SampleLevel(octahedral_sampler, uv, 0);
}
