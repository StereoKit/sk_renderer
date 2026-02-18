//--name = pbr_cubemap

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
// Textures
///////////////////////////////////////////

Texture2D    albedo_tex   : register(t1);
SamplerState albedo_tex_s : register(s1);

Texture2D    emission_tex   : register(t0);
SamplerState emission_tex_s : register(s0);

TextureCube  environment_map   : register(t5);
SamplerState environment_map_s : register(s5);

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
	float4 pos        : SV_POSITION;
	float3 normal     : NORMAL0;
	float2 uv         : TEXCOORD0;
	float4 color      : COLOR0;
	float3 shadow_uv  : TEXCOORD1;
	uint   layer      : SV_RenderTargetArrayIndex;
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
	output.normal    = normal;
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;

	// Shadow map projection
	float  ndotl = dot(normal, light_direction);
	float  slope = saturate(min(1.0, sqrt(1.0 - ndotl * ndotl) / max(0.001, ndotl)));
	float3 bias  = normal * (shadow_bias * slope);
	float4 shadow_pos = mul(float4(world_pos + bias, 1), shadow_transform);
	output.shadow_uv  = float3(shadow_pos.xy * float2(0.5, -0.5) + 0.5, shadow_pos.z / shadow_pos.w);

	output.layer = view_idx;
	return output;
}

///////////////////////////////////////////
// Shadow Sampling
///////////////////////////////////////////

float shadow_factor_pcf4(float3 uv, float scale) {
	float r = shadow_pixel_size * scale * 0.5;
	return (
		shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + float2(-r, -r), uv.z) +
		shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + float2( r, -r), uv.z) +
		shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + float2(-r,  r), uv.z) +
		shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + float2( r,  r), uv.z)
	) * 0.25;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	float4 albedo   = albedo_tex  .Sample(albedo_tex_s,   input.uv) * input.color;
	float3 emissive = emission_tex.Sample(emission_tex_s, input.uv).rgb * emission_factor.rgb;

	// Cubemap irradiance (sample at lowest mip for blurry ambient)
	float3 irradiance = environment_map.SampleLevel(environment_map_s, input.normal, cubemap_info.z - 1).rgb;

	// Shadow
	float ndotl  = dot(input.normal, light_direction);
	float shadow = 1.0;
	if (ndotl > 0.0) {
		shadow = shadow_factor_pcf4(input.shadow_uv, 1.0);
	}

	float3 diffuse     = albedo.rgb * irradiance;
	float3 directional = albedo.rgb * light_color * saturate(ndotl) * shadow;

	return float4(diffuse + directional + emissive, albedo.a);
}
