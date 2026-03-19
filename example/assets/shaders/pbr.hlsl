//--name = pbr

#include "common.hlsli"
#include "pbr.hlsli"

///////////////////////////////////////////
// Material Parameters
///////////////////////////////////////////

// Default material properties
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
// Textures - Using slots 3,4,5,6 to avoid conflicts with t2 (instance buffer)
///////////////////////////////////////////

// Albedo/diffuse texture
Texture2D    albedo_tex   : register(t1);
SamplerState albedo_tex_s : register(s1);

// Emission texture
Texture2D    emission_tex   : register(t0);
SamplerState emission_tex_s : register(s0);

// Metallic-roughness texture (R=occlusion, G=roughness, B=metallic)
Texture2D    metal_tex    : register(t3);
SamplerState metal_tex_s  : register(s3);

// Occlusion texture (packed in metal texture R channel, but can be separate)
Texture2D    occlusion_tex   : register(t4);
SamplerState occlusion_tex_s : register(s4);

// Environment cubemap for IBL
TextureCube  environment_map   : register(t5);
SamplerState environment_map_s : register(s5);

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
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, skr_ids_t ids) {
	psIn output;

	// Transform to world space
	output.world_pos = mul(float4(input.pos, 1), inst[ids.inst].world).xyz;
	output.pos       = mul(float4(output.world_pos, 1), viewproj[ids.view]);

	// Transform normal to world space
	output.normal = normalize(mul(float4(input.norm, 0), inst[ids.inst].world).xyz);

	// Apply texture transform and pass through vertex color
	output.uv    = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color = input.color * color;

	// Calculate view direction
	output.view_dir = cam_pos[ids.view].xyz - output.world_pos;

	return output;
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

	// Extract material properties from textures
	// glTF 2.0 standard: G=roughness, B=metallic, R=occlusion (optional)
	float roughness_final = metal_sample.g * roughness;
	float metallic_final  = metal_sample.b * metallic;
	float ao_final        = ao_sample;

	// Simple irradiance approximation - sample lower mips of cubemap
	// This replaces spherical harmonics with a simpler approach
	float3 irradiance = environment_map.SampleLevel(environment_map_s, input.normal, cubemap_info.z - 1).rgb;

	// Apply PBR shading
	float4 final_color = pbr_shade(
		albedo_sample,
		irradiance,
		ao_final,
		metallic_final,
		roughness_final,
		input.view_dir,
		input.normal,
		environment_map,
		environment_map_s
	);

	// Add emissive contribution
	final_color.rgb += emissive_sample;

	return final_color;
}
