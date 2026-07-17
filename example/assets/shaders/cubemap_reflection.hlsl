//--name = cubemap_reflection

#include "common.hlsli"

struct Inst {
	float4x4 world;
	float    roughness;  // 0 = smooth (mip 0), 1 = rough (highest mip)
	float3   _pad;       // Matches the C-side padding to a 16-aligned stride
};
StructuredBuffer<Inst> inst : register(t2, space0);

struct vsIn {
	float3 pos  : SV_POSITION;
	float3 norm : NORMAL;
	float2 uv   : TEXCOORD0;
	float4 color: COLOR0;
};
struct psIn {
	float4 pos       : SV_POSITION;
	float3 world_pos : TEXCOORD0;
	float3 normal    : TEXCOORD1;
	float  roughness : TEXCOORD2;
	float3 cam_pos   : TEXCOORD3;
};

TextureCube  cubemap         : register(t0);
SamplerState cubemap_sampler : register(s0);

psIn vs(vsIn input, skr_ids_t ids) {
	psIn output;
	float4 world_pos = mul(float4(input.pos, 1), inst[ids.inst].world);
	output.world_pos = world_pos.xyz;
	output.pos = mul(world_pos, viewproj[ids.view]);
	output.normal = normalize(mul(float4(input.norm, 0), inst[ids.inst].world).xyz);
	output.roughness = inst[ids.inst].roughness;
	output.cam_pos = cam_pos[ids.view].xyz;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	const float3 light_dir = normalize(float3(1, 4, 2));

	// Calculate view direction from camera to fragment (in world space)
	float3 view_dir = normalize(input.world_pos - input.cam_pos);

	// Calculate reflection vector
	float3 normal = normalize(input.normal);
	float3 reflection = reflect(view_dir, normal);

	// Sample cubemap using reflection vector with mip level based on roughness
	// roughness 0.0 = mip 0 (sharp), roughness 1.0 = highest mip (blurry)
	// Assuming ~9 mip levels for 512x512 texture
	float mip_level = input.roughness * 9.0;
	float3 reflected_color = cubemap.SampleLevel(cubemap_sampler, reflection, mip_level).rgb;

	// Apply lighting: ambient + diffuse
	float3 ambient = float3(0.05, 0.05, 0.1);
	float diffuse = saturate(dot(normal, light_dir) * 0.8);
	float3 lighting = ambient + diffuse;

	// Combine reflection with lighting
	float3 final_color = reflected_color + lighting*0.5;

	return float4(final_color, 1.0);
}
