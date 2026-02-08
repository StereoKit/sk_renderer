//--name = pbr_gi

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

Texture2D    metal_tex    : register(t3);
SamplerState metal_tex_s  : register(s3);

Texture2D    occlusion_tex   : register(t4);
SamplerState occlusion_tex_s : register(s4);

TextureCube  environment_map   : register(t5);
SamplerState environment_map_s : register(s5);

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
	float4 pos        : SV_POSITION;
	float3 normal     : NORMAL0;
	float2 uv         : TEXCOORD0;
	float4 color      : COLOR0;
	float3 world_pos  : TEXCOORD1;
	float3 view_dir   : TEXCOORD2;
	float3 shadow_uv  : TEXCOORD3;
	uint   layer      : SV_RenderTargetArrayIndex;
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn output;

	uint view_idx = id % view_count;
	uint inst_idx = id / view_count;

	output.world_pos = mul(float4(input.pos, 1), inst[inst_idx].world).xyz;
	output.pos       = mul(float4(output.world_pos, 1), viewproj[view_idx]);
	output.normal    = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;
	output.view_dir  = cam_pos[view_idx].xyz - output.world_pos;

	// Shadow map projection
	float  ndotl = dot(output.normal, light_direction);
	float  slope = saturate(min(1.0, sqrt(1.0 - ndotl * ndotl) / max(0.001, ndotl)));
	float3 bias  = output.normal * (shadow_bias * slope);
	float4 shadow_pos = mul(float4(output.world_pos + bias, 1), shadow_transform);
	output.shadow_uv  = float3(shadow_pos.xy * float2(0.5, -0.5) + 0.5, shadow_pos.z / shadow_pos.w);

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
// GI Probe Sampling
///////////////////////////////////////////

float3 gi_sample_irradiance(float3 world_pos, float3 normal) {
	// World position -> probe volume UVW [0,1]
	float3 uvw = (world_pos - gi_volume_min) * gi_volume_inv;
	uvw = saturate(uvw);

	// Sample SH coefficients from 3D textures (trilinear interpolation)
	float4 shr = gi_sh_r.SampleLevel(gi_sh_r_s, uvw, 0);
	float4 shg = gi_sh_g.SampleLevel(gi_sh_g_s, uvw, 0);
	float4 shb = gi_sh_b.SampleLevel(gi_sh_b_s, uvw, 0);

	// Evaluate SH for surface normal direction
	// Pre-multiplied with cosine lobe weights: A0=pi, A1=2pi/3
	// Y00*pi = 0.28209*3.14159 = 0.88623
	// Y1m*2pi/3 = 0.48860*2.09440 = 1.02333
	float4 sh_basis = float4(
		0.88623,
		1.02333 * normal.y,
		1.02333 * normal.z,
		1.02333 * normal.x
	);

	return float3(
		dot(shr, sh_basis),
		dot(shg, sh_basis),
		dot(shb, sh_basis)
	);
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	// Sample textures
	float4 albedo_sample   = albedo_tex  .Sample(albedo_tex_s,    input.uv) * input.color;
	float3 emissive_sample = emission_tex.Sample(emission_tex_s,  input.uv).rgb * emission_factor.rgb;
	float3 metal_sample    = metal_tex   .Sample(metal_tex_s,     input.uv).rgb;
	float  ao_sample       = occlusion_tex.Sample(occlusion_tex_s, input.uv).r;

	float roughness_final = metal_sample.g * roughness;
	float metallic_final  = metal_sample.b * metallic;
	float ao_final        = ao_sample;

	// GI probe irradiance (replaces IBL diffuse)
	float3 irradiance = max(float3(0, 0, 0), gi_sample_irradiance(input.world_pos, input.normal)) * gi_intensity;

	// Shadow
	float ndotl  = dot(input.normal, light_direction);
	float shadow = 1.0;
	if (ndotl > 0.0) {
		shadow = shadow_factor_pcf3(input.shadow_uv, 1.0);
	}

	// Directional light contribution, modulated by shadow
	float directional = saturate(ndotl) * shadow;

	// PBR shading
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
	float4 final_color = float4((kD * diffuse /* + specular * ao_final */), albedo_sample.a);

	// Add directional light
	final_color.rgb += albedo_sample.rgb * light_color * directional * 0.5;

	// Add emissive
	final_color.rgb += emissive_sample;

	return final_color;
}
