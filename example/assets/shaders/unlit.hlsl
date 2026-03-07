//--name = unlit
//--tex = white

#include "common.hlsli"

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
	float4 color : COLOR0;
	SKR_LAYER_OUTPUT
};

Texture2D    tex         : register(t3);
SamplerState tex_sampler : register(s3);
float4       tex_trans;

//--tex_trans = 0,0,1,1

psIn vs(vsIn input, skr_input_t sys) {
	skr_ids_t ids = skr_resolve_ids(sys);

	psIn output;
	output.pos   = mul(float4(input.pos, 1), inst[ids.inst].world);
	output.pos   = mul(output.pos, viewproj[ids.view]);
	output.color = input.color;
	output.uv    = (input.uv * tex_trans.zw) + tex_trans.xy;
	SKR_SET_LAYER(output, ids.view);
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	return input.color * tex.Sample(tex_sampler, input.uv);
}
