//--name = test
//--tex = white

#include "common.hlsli"

float4 emissive;

struct Inst {
	float4x4 world;
};
StructuredBuffer<Inst> inst : register(t2, space0);

struct vsIn {
	float3 pos  : SV_POSITION;
	float3 norm : NORMAL;
	float2 uv   : TEXCOORD0;
	float4 color: COLOR0;
};
struct psIn {
	float4 pos   : SV_POSITION;
	float2 uv    : TEXCOORD0;
	float3 color   : COLOR0;
};

Texture2D    tex         : register(t3);
SamplerState tex_sampler : register(s3);
float4       tex_trans;

//--tex_trans = 0,0,1,1

psIn vs(vsIn input, skr_ids_t ids) {
	const float3 light_dir = normalize(float3(1, 4, 2));

	psIn output;
	output.pos = mul(float4(input.pos, 1), inst[ids.inst].world);
	output.pos = mul(output.pos, viewproj[ids.view]);
	float3 normal = normalize(mul(float4(input.norm, 0), inst[ids.inst].world).xyz);
	output.color = (float3(.05,.05,.1) + saturate(dot(normal, light_dir)*0.8).xxx) * input.color.rgb;
	output.uv    = (input.uv * tex_trans.zw) + tex_trans.xy;
	return output;
}
float4 ps(psIn input) : SV_TARGET {
	return float4(input.color, 1) * tex.Sample(tex_sampler, input.uv) + emissive;
}
