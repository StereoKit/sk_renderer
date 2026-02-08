//--name = gi_capture
//--color:color = 1,1,1,1
//--tex_trans   = 0,0,1,1

// Simplified shader for GI voxel capture passes.
// Shadowed diffuse only (no PBR). Backfaces render as black so they
// contribute opacity without injecting incorrect radiance.

#include "common.hlsli"

float4 color;
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
	output.pos       = mul(float4(world_pos, 1), viewproj[view_idx]);
	output.normal    = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;

	// Shadow map projection
	float  ndotl = dot(output.normal, light_direction);
	float  slope = saturate(min(1.0, sqrt(1.0 - ndotl * ndotl) / max(0.001, ndotl)));
	float3 bias  = output.normal * (shadow_bias * slope);
	float4 shadow_pos = mul(float4(world_pos + bias, 1), shadow_transform);
	output.shadow_uv  = float3(shadow_pos.xy * float2(0.5, -0.5) + 0.5, shadow_pos.z / shadow_pos.w);

	output.layer = view_idx;
	return output;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input, bool is_front : SV_IsFrontFace) : SV_TARGET {
	// Backfaces: render black (contributes opacity to voxels without radiance)
	//if (!is_front) return float4(0, 0, 0, 0);

	// Albedo
	float4 albedo = albedo_tex.Sample(albedo_tex_s, input.uv) * input.color;

	// Shadow
	float ndotl  = dot(input.normal, light_direction);
	float shadow = 1.0;
	if (ndotl > 0.0) {
		float radius = shadow_pixel_size;
		float s = 0.0;
		[unroll] for (int x = -1; x <= 1; x++) {
			[unroll] for (int y = -1; y <= 1; y++) {
				s += shadow_map.SampleCmpLevelZero(shadow_map_sampler, input.shadow_uv.xy + float2(x, y) * radius, input.shadow_uv.z);
			}
		}
		shadow = s / 9.0;
	}

	// Simple diffuse: NdotL * shadow * light_color
	float diffuse = saturate(ndotl) * shadow;
	float3 lit = albedo.rgb * light_color * diffuse;

	return float4(lit, albedo.a);
}
