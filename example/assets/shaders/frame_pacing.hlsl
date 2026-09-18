//--name = frame_pacing

#include "common.hlsli"

// One instance per draw; the scene fills this each frame
struct Inst {
	float scroll_px;   // fence offset in pixels
	float period_px;   // distance between bars
	float bar_px;      // bar width
	float marker_px;   // left edge of the moving marker
	uint  frame_index;
	uint  _pad0;
	uint  _pad1;
	uint  _pad2;
};
StructuredBuffer<Inst> inst : register(t2, space0);

struct vsIn {
	float3 pos  : SV_POSITION;
	float3 norm : NORMAL;
	float2 uv   : TEXCOORD0;
	float4 color: COLOR0;
};
struct psIn {
	float4 pos : SV_POSITION;
	float2 px  : TEXCOORD0;
};

// The quad is already in NDC and covers the viewport, so the camera is
// ignored and pixel coordinates come straight from the corners (y down).
psIn vs(vsIn input, skr_ids_t ids) {
	psIn o;
	o.pos = float4(input.pos.xy, 0, 1);
	o.px  = (input.pos.xy * float2(0.5, -0.5) + 0.5) * screen_size.xy;
	return o;
}

float3 hue_rgb(float h) {
	float3 p = abs(frac(h + float3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
	return saturate(p - 1.0);
}

float fence(float x, float period, float bar) {
	return frac(x / period) * period < bar ? 1.0 : 0.0;
}

float4 ps(psIn input) : SV_TARGET {
	Inst   i  = inst[0];
	float2 px = input.px;
	float  y  = px.y / screen_size.y;

	// Left strip: a new hue every frame. In a high speed camera capture a
	// repeated color is a repeated frame, and a skipped hue is a dropped one.
	if (px.x < 24) {
		float v = (i.frame_index & 1) ? 1.0 : 0.55;
		return float4(hue_rgb(frac(i.frame_index * 0.618034)) * v, 1);
	}

	float3 col = 0.12;
	if      (y > 0.08 && y < 0.42) col = fence(px.x - i.scroll_px,       i.period_px, i.bar_px);
	else if (y > 0.47 && y < 0.68) col = fence(px.x - i.scroll_px * 2.0, i.period_px, i.bar_px);
	else if (y > 0.73 && y < 0.84) {
		float d = px.x - i.marker_px;
		col = (d >= 0 && d < 32) ? float3(1.0, 0.6, 0.1) : 0.04;
	}
	else if (y > 0.89 && y < 0.96) {
		// 16 cells, one lit per frame, so a capture can count displayed frames
		uint cell = (uint)floor(px.x / screen_size.x * 16.0);
		col = cell == (i.frame_index & 15) ? float3(0.2, 1.0, 0.3) : 0.04;
	}
	return float4(col, 1);
}
