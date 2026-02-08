//--name = pbr_gi_vertex

#include "common.hlsli"

///////////////////////////////////////////
// Material Parameters
///////////////////////////////////////////

//--color:color           = 1,1,1,1
//--emission_factor:color = 0,0,0,0
//--tex_trans             = 0,0,1,1

float4 color;
float4 emission_factor;
float4 tex_trans;

///////////////////////////////////////////
// Instance Data
///////////////////////////////////////////

struct Inst {
	float4x4 world;
};
 StructuredBuffer<Inst> inst : register(t2);

///////////////////////////////////////////
// Shadow Buffer (b13)
///////////////////////////////////////////

cbuffer ShadowBuffer : register(b13, space0) {
	float4x4 shadow_transform;
	float3   light_direction;
	float    shadow_bias;
	float3   light_color;
	float    shadow_pixel_size;
};

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
// Textures
///////////////////////////////////////////

Texture2D    albedo_tex   : register(t1);
SamplerState albedo_tex_s : register(s1);

Texture2D    emission_tex   : register(t0);
SamplerState emission_tex_s : register(s0);

// GI SH probe textures (3D, one per color channel, 4 SH coefficients each)
Texture3D<float4> gi_sh_r   : register(t6);
SamplerState      gi_sh_r_s : register(s6);

Texture3D<float4> gi_sh_g   : register(t7);
SamplerState      gi_sh_g_s : register(s7);

Texture3D<float4> gi_sh_b   : register(t8);
SamplerState      gi_sh_b_s : register(s8);

// Shadow map (t14/s14)
Texture2D              shadow_map         : register(t14);
SamplerComparisonState shadow_map_sampler : register(s14);

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
	float4 pos           : SV_POSITION;
	float2 uv            : TEXCOORD0;
	float4 color         : COLOR0;
	float3 gi_irradiance : TEXCOORD1;  // GI irradiance evaluated per-vertex
	float4 shadow_uv     : TEXCOORD2;  // .xyz = shadow coords, .w = ndotl
	uint   layer         : SV_RenderTargetArrayIndex;
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn output;

	uint view_idx = id % view_count;
	uint inst_idx = id / view_count;

	float3 world_pos = mul(float4(input.pos, 1), inst[inst_idx].world).xyz;
	float3 normal    = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.pos       = mul(float4(world_pos, 1), viewproj[view_idx]);
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;

	// GI probe sampling (per-vertex)
	float3 uvw = saturate((world_pos - gi_volume_min) * gi_volume_inv);
	float4 shr = gi_sh_r.SampleLevel(gi_sh_r_s, uvw, 0);
	float4 shg = gi_sh_g.SampleLevel(gi_sh_g_s, uvw, 0);
	float4 shb = gi_sh_b.SampleLevel(gi_sh_b_s, uvw, 0);

	float3 sh_norm  = 1.02333 * normal.yzx;
	float4 sh_basis = float4(0.88623, sh_norm);
	output.gi_irradiance = max(float3(0, 0, 0), float3(
		dot(shr, sh_basis),
		dot(shg, sh_basis),
		dot(shb, sh_basis)
	)) * gi_intensity;

	// Shadow map projection + ndotl (reused in PS for diffuse lighting)
	float  ndotl = dot(normal, light_direction);
	float  slope = saturate(min(1.0, sqrt(1.0 - ndotl * ndotl) / max(0.001, ndotl)));
	float3 bias  = normal * (shadow_bias * slope);
	float4 shadow_pos  = mul(float4(world_pos + bias, 1), shadow_transform);
	output.shadow_uv   = float4(shadow_pos.xy * float2(0.5, -0.5) + 0.5, shadow_pos.z / shadow_pos.w, ndotl);

	output.layer = view_idx;
	return output;
}

///////////////////////////////////////////
// Shadow Sampling
///////////////////////////////////////////

float shadow_factor_pcf3(float3 uv, float scale) {
	float radius = shadow_pixel_size * scale;
	float shadow_factor = 0.0;

	[unroll]
	for (int x = -1; x <= 1; x++) {
		[unroll]
		for (int y = -1; y <= 1; y++) {
			float2 offset = float2(x, y) * radius;
			shadow_factor += shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + offset, uv.z);
		}
	}
	return shadow_factor / 9.0;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	float4 albedo   = albedo_tex  .Sample(albedo_tex_s,   input.uv) * input.color;
	float3 emissive = emission_tex.Sample(emission_tex_s, input.uv).rgb * emission_factor.rgb;

	// Shadow (ndotl from VS via shadow_uv.w)
	float ndotl  = input.shadow_uv.w;
	float shadow = 1.0;
	if (ndotl > 0.0) {
		shadow = shadow_factor_pcf3(input.shadow_uv.xyz, 1.0);
	}

	float3 diffuse     = albedo.rgb * input.gi_irradiance;
	float3 directional = albedo.rgb * light_color * saturate(ndotl) * shadow;

	return float4(diffuse + directional + emissive, albedo.a);
}
