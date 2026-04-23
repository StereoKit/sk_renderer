//--name = tex_compare
//--swipe = 0.5
//--brightness = 1
//--line_color:color = 1,1,1,1

// Side-by-side compare with a vertical swipe line. Left of the swipe shows
// tex_left (original); right shows tex_right (compressed). A 1-2 px white
// vertical line at the swipe position marks the transition.
//
// Single fullscreen-style quad — no quad pair, no per-side aspect math.

#include "common.hlsli"

struct Inst { float4x4 world; };
StructuredBuffer<Inst> inst : register(t2, space0);

struct vsIn {
	float3 pos  : SV_POSITION;
	float3 norm : NORMAL;
	float2 uv   : TEXCOORD0;
	float4 color: COLOR0;
};
struct psIn {
	float4 pos    : SV_POSITION;
	float2 uv     : TEXCOORD0;
	float4 color  : COLOR0;
	float2 screen : TEXCOORD1; // pixel coords for line thickness
};

Texture2D    tex_left         : register(t3);
SamplerState tex_left_sampler : register(s3);

Texture2D    tex_right         : register(t4);
SamplerState tex_right_sampler : register(s4);

float  swipe;       // 0..1, fraction of width showing tex_left
float  brightness;  // multiplier on the sampled color (before the swipe line)
float4 line_color;

psIn vs(vsIn input, skr_ids_t ids) {
	psIn output;
	float4 world  = mul(float4(input.pos, 1), inst[ids.inst].world);
	output.pos    = mul(world, viewproj[ids.view]);
	output.uv     = input.uv;
	output.color  = input.color;
	output.screen = output.pos.xy / output.pos.w; // NDC, used for line width
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	float4 c_left  = tex_left .Sample(tex_left_sampler,  input.uv);
	float4 c_right = tex_right.Sample(tex_right_sampler, input.uv);
	float4 col     = (input.uv.x < swipe) ? c_left : c_right;

	// Apply brightness to RGB only (alpha unchanged) — scale HDR content down
	// so bright sky pixels don't blow out and hide the detail we're comparing.
	col.rgb *= brightness;

	// Vertical line at uv.x == swipe. Width is fixed in UV-space (~0.002 of
	// the quad width) — gives a thin clean line at any zoom level. fwidth
	// would tie it to screen-pixels but adds a derivative cost we don't need.
	// Drawn after brightness so the line stays pure white regardless.
	float dist_to_line = abs(input.uv.x - swipe);
	float line_blend   = saturate(1.0 - dist_to_line * 500.0);
	col = lerp(col, line_color, line_blend);

	return col * input.color;
}
