//--name = pbr_shadow

#include "common.hlsli"
#include "pbr.hlsli"

///////////////////////////////////////////
// Material Parameters
///////////////////////////////////////////

//--color:color           = 1,1,1,1
//--emission_factor:color = 0,0,0,0
//--metallic              = 0
//--roughness             = 1
//--tex_trans             = 0,0,1,1

float4 color;
float4 emission_factor;
float4 tex_trans;
float  metallic;
float  roughness;
float2 _mat_pad;

///////////////////////////////////////////
// Instance Data
///////////////////////////////////////////

struct Inst {
	float4x4 world;
};
 StructuredBuffer<Inst> inst : register(t2);

///////////////////////////////////////////
// Shadow Buffer (b13, matches shadow_receiver.hlsl)
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

Texture2D    metal_tex    : register(t3);
SamplerState metal_tex_s  : register(s3);

Texture2D    occlusion_tex   : register(t4);
SamplerState occlusion_tex_s : register(s4);

TextureCube  environment_map   : register(t5);
SamplerState environment_map_s : register(s5);

// Shadow map (t14/s14, matches shadow_receiver.hlsl)
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
	float3 world_pos  : TEXCOORD1;
	float3 view_dir   : TEXCOORD2;
	float3 shadow_uv  : TEXCOORD3;
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, skr_ids_t ids) {
	psIn output;

	output.world_pos = mul(float4(input.pos, 1), inst[ids.inst].world).xyz;
	output.pos       = mul(float4(output.world_pos, 1), viewproj[ids.view]);
	output.normal    = normalize(mul(float4(input.norm, 0), inst[ids.inst].world).xyz);
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;
	output.view_dir  = cam_pos[ids.view].xyz - output.world_pos;

	// Shadow map projection
	float  ndotl = dot(output.normal, light_direction);
	float  slope = saturate(min(1.0, sqrt(1.0 - ndotl * ndotl) / max(0.001, ndotl)));
	float3 bias  = output.normal * (shadow_bias * slope);
	float4 shadow_pos = mul(float4(output.world_pos + bias, 1), shadow_transform);
	output.shadow_uv  = float3(shadow_pos.xy * float2(0.5, -0.5) + 0.5, shadow_pos.z / shadow_pos.w);

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
	// Sample textures
	float3 metal_sample    = metal_tex   .Sample(metal_tex_s,     input.uv).rgb;
	float4 albedo_sample   = albedo_tex  .Sample(albedo_tex_s,    input.uv) * input.color;
	float3 emissive_sample = emission_tex.Sample(emission_tex_s,  input.uv).rgb * emission_factor.rgb;
	float  ao_sample       = occlusion_tex.Sample(occlusion_tex_s, input.uv).r;

	float roughness_final = metal_sample.g * roughness;
	float metallic_final  = metal_sample.b * metallic;
	float ao_final        = ao_sample;

	// IBL irradiance
	float3 irradiance = environment_map.SampleLevel(environment_map_s, input.normal, cubemap_info.z - 1).rgb;

	// Shadow
	float ndotl  = dot(input.normal, light_direction);
	float shadow = 1.0;
	if (ndotl > 0.0) {
		shadow = shadow_factor_pcf4(input.shadow_uv, 1.0);
	}

	// Directional light contribution, modulated by shadow
	float directional = saturate(ndotl) * shadow;

	// PBR shading (inlined to avoid SPIRV OpStore validation error from
	// passing TextureCube/SamplerState as function parameters)
	float3 view       = normalize(input.view_dir);
	float3 reflection = reflect(-view, input.normal);
	float  ndotv      = max(0, dot(input.normal, view));

	float3 norm_ddx        = ddx(input.normal.xyz);
	float3 norm_ddy        = ddy(input.normal.xyz);
	float  geometric_rough = pow(saturate(max(dot(norm_ddx, norm_ddx), dot(norm_ddy, norm_ddy))), 0.45);
	float  rough_clamped   = max(roughness_final, geometric_rough);

	float3 F0 = lerp(0.04, albedo_sample.rgb, metallic_final);
	float3 F  = pbr_fresnel_schlick_roughness(ndotv, F0, rough_clamped);
	float3 kS = F;

	float mip = (1 - pow(1 - rough_clamped, 2)) * cubemap_info.z;
	mip = max(mip, pbr_mip_level(ndotv));
	float3 prefilteredColor = environment_map.SampleLevel(environment_map_s, reflection, mip).rgb;
	float2 envBRDF          = pbr_brdf_appx(rough_clamped, ndotv);
	float3 specular         = prefilteredColor * (F * envBRDF.x + envBRDF.y);

	float3 kD = 1 - kS;
	kD *= 1.0 - metallic_final;

	float3 diffuse    = albedo_sample.rgb * irradiance * ao_final;
	float4 final_color = float4((kD * diffuse + specular * ao_final), albedo_sample.a);

	// Add directional light on top of IBL
	final_color.rgb += albedo_sample.rgb * light_color * directional * 0.5;

	// Add emissive
	final_color.rgb += emissive_sample;

	return final_color;
}
