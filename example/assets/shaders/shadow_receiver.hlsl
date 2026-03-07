//--name = shadow_receiver
//--tex = white

#include "common.hlsli"

// Shadow receiver shader - samples from shadow map and applies shadows

struct Inst {
	float4x4 world;
};
StructuredBuffer<Inst> inst : register(t2, space0);

// Shadow buffer - contains shadow mapping parameters
cbuffer ShadowBuffer : register(b13, space0) {
	float4x4 shadow_transform;      // Transforms world space to shadow map UV space
	float3   light_direction;
	float    shadow_bias;
	float3   light_color;
	float    shadow_pixel_size;
};

struct vsIn {
	float3 pos  : SV_POSITION;
	float3 norm : NORMAL;
	float2 uv   : TEXCOORD0;
	float4 color: COLOR0;
};

struct psIn {
	float4 pos          : SV_POSITION;
	float2 uv           : TEXCOORD0;
	float3 shadow_uv    : TEXCOORD1;
	float3 world_pos    : TEXCOORD2;
	float  shadow_ndotl : TEXCOORD3;
	float3 color        : COLOR0;
	SKR_LAYER_OUTPUT
};

Texture2D              tex              : register(t3);
SamplerState           tex_sampler      : register(s3);
Texture2D              shadow_map       : register(t14);
SamplerComparisonState shadow_map_sampler : register(s14);

psIn vs(vsIn input, skr_input_t sys) {
	skr_ids_t ids = skr_resolve_ids(sys);

	psIn output;
	float4 world_pos = mul(float4(input.pos, 1), inst[ids.inst].world);
	output.pos = mul(world_pos, viewproj[ids.view]);
	output.world_pos = world_pos.xyz;

	// Calculate normal in world space
	float3 normal = normalize(mul(float4(input.norm, 0), inst[ids.inst].world).xyz);
	output.shadow_ndotl = dot(normal, light_direction);

	// Apply bias to shadow map position in world space
	// Slope-based bias to reduce shadow acne
	float slope = saturate(min(1.0, sqrt(1.0 - output.shadow_ndotl * output.shadow_ndotl) / max(0.001, output.shadow_ndotl)));
	float3 bias = normal * (shadow_bias * slope);
	float4 shadow_pos = mul(float4(world_pos.xyz + bias, 1), shadow_transform);

	// Transform to shadow map UV coordinates
	output.shadow_uv = float3(shadow_pos.xy, shadow_pos.z / shadow_pos.w);

	// Convert clip space [-1,1] to texture UV [0,1], flip Y for negative viewport
	output.shadow_uv.xy = output.shadow_uv.xy * float2(0.5, -0.5) + 0.5;

	output.uv = input.uv;

	// Simple lighting based on normal
	float3 ambient = float3(0.05, 0.05, 0.1);
	float diffuse = saturate(output.shadow_ndotl) * 0.8;
	output.color = (ambient + diffuse) * input.color.rgb;

	SKR_SET_LAYER(output, ids.view);
	return output;
}

// Fast shadow sampling - single sample
float shadow_factor_fast(float3 uv) {
	return shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy, uv.z);
}

// PCF 3x3 filter - smooth shadow edges
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

float4 ps(psIn input) : SV_TARGET {
	float4 col = tex.Sample(tex_sampler, input.uv);

	// Calculate shadow factor
	float light = 0.0;
	if (input.shadow_ndotl > 0.0) {
		// Use PCF filtering for smooth shadow edges
		float shadow = shadow_factor_pcf3(input.shadow_uv, 1.0);
		light = min(input.shadow_ndotl, shadow);
	}

	// Apply lighting
	float3 ambient = float3(0.05, 0.05, 0.1);
	col.rgb *= (ambient + light * light_color);

	return col;
}
