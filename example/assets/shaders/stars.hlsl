//--name = stars

#include "common.hlsli"

// Pins each star to a pixel center. Exact, but it quantizes motion, so the
// scene leaves it to the viewer.
[[vk::constant_id(0)]] const bool STAR_SNAP = false;

struct Star {
	float3 position;
	float  brightness;
};
StructuredBuffer<Star> stars : register(t3);

struct psIn {
	float4 pos   : SV_POSITION;
	float4 color : COLOR0;
};

psIn vs(uint vertex_id : SV_VertexID, skr_ids_t ids) {
	uint star_idx = vertex_id / 3;
	uint corner   = vertex_id % 3;

	Star star = stars[star_idx];

	psIn output;

	float4 clip_pos = mul(float4(star.position, 1), viewproj[ids.view]);

	// A pixel in clip units. Negated in y: window y runs down, clip y up.
	float2 px_to_clip = screen_size.zw * float2(2.0, -2.0) * clip_pos.w;

	// Snapping puts the triangle on the sample grid the offsets below were
	// solved against. Stars behind the eye are clipped away, so skip the divide
	// rather than hand it a negative w.
	float2 snap = 0;
	if (STAR_SNAP && clip_pos.w > 0) {
		float2 px = (clip_pos.xy / clip_pos.w * float2(0.5, -0.5) + 0.5) * screen_size.xy;
		snap = floor(px) + 0.5 - px;
	}

	// Snapped: equilateral, circumradius 0.833px, one vertex 3.7 degrees below +x.
	// That pose covers every standard 2x and 4x sample of its pixel and no other
	// pixel's, with 0.034px of margin. The rotation is not free: mirroring it to
	// +3.7 catches a neighbour, because the sample grid is chiral.
	float2 tri_snapped[3] = {
		float2( 0.83126, -0.05376),
		float2(-0.36908,  0.74677),
		float2(-0.46219, -0.69302)
	};
	// Free: isoceles, 1px base by 2px tall, area exactly 1px^2. Area alone fixes
	// the mean at 4 samples covered; this shape holds it to 3..5 wherever it
	// lands, where an equilateral of the same area swings 2..6. Axis alignment
	// is what buys that, and two degrees off loses most of it.
	float2 tri_free[3] = {
		float2(-0.5, -0.66667),
		float2( 0.5, -0.66667),
		float2( 0.0,  1.33333)
	};
	clip_pos.xy += (snap + (STAR_SNAP ? tri_snapped[corner] : tri_free[corner])) * px_to_clip;

	output.pos   = clip_pos;
	output.color = float4(star.brightness, star.brightness, star.brightness, 1);

	return output;
}

float4 ps(psIn input) : SV_TARGET {
	return input.color;
}
