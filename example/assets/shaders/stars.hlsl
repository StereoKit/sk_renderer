//--name = stars

#include "common.hlsli"

struct Star {
	float3 position;
	float  brightness;
};
StructuredBuffer<Star> stars : register(t3);

struct psIn {
	float4 pos   : SV_POSITION;
	float4 color : COLOR0;
	SKR_LAYER_OUTPUT
};

psIn vs(uint vertex_id : SV_VertexID, skr_input_t sys) {
	skr_ids_t ids = skr_resolve_ids(sys);

	uint star_idx = vertex_id / 3;
	uint corner   = vertex_id % 3;

	Star star = stars[star_idx];

	psIn output;

	float4 clip_pos = mul(float4(star.position, 1), viewproj[ids.view]);

	// Expand triangle to cover approximately 1 pixel
	float2 pixel_size = screen_size.zw * 2.0 * clip_pos.w;
	float2 offsets[3] = {
		float2( 0.0,    1.0),  // top
		float2(-0.866, -0.5), // bottom-left
		float2( 0.866, -0.5)  // bottom-right
	};
	clip_pos.xy += offsets[corner] * pixel_size;

	output.pos   = clip_pos;
	output.color = float4(star.brightness, star.brightness, star.brightness, 1);
	SKR_SET_LAYER(output, ids.view);

	return output;
}

float4 ps(psIn input) : SV_TARGET {
	return input.color;
}
