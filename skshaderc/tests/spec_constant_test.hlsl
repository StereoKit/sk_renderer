//--name = test/spec_constants
// Expected metadata: 4 spec constants, ids 0-3, types int/float/int(bool)/uint,
// defaults 4/0.25/1/8, split across vertex and pixel stages.

[[vk::constant_id(0)]] const int   LIGHT_COUNT  = 4;
[[vk::constant_id(1)]] const float FOG_DENSITY  = 0.25;
[[vk::constant_id(2)]] const bool  USE_SHADOWS  = true;
[[vk::constant_id(3)]] const uint  SAMPLE_COUNT = 8;

//--tint = 1,1,1,1
float4 tint;

struct vsIn {
	float4 pos : SV_Position;
};
struct psIn {
	float4 pos : SV_Position;
	float  fog : TEXCOORD0;
};

psIn vs(vsIn input) {
	psIn o;
	o.pos = input.pos;
	o.fog = FOG_DENSITY * (float)SAMPLE_COUNT;
	return o;
}

float4 ps(psIn input) : SV_Target {
	float4 col = tint;
	for (int i = 0; i < LIGHT_COUNT; i++)
		col.rgb += 0.01;
	if (USE_SHADOWS)
		col.rgb *= 0.5;
	col.rgb *= input.fog;
	return col;
}
