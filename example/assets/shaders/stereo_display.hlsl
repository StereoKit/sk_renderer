//--name = stereo_display

struct vsIn {
	float3 pos    : SV_POSITION;
	float3 normal : NORMAL;
	float2 uv     : TEXCOORD0;
	float4 color  : COLOR;
};

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

uint mode;

Texture2DArray array_tex      : register(t0);
SamplerState   array_sampler  : register(s0);

psIn vs(vsIn input) {
	psIn output;
	output.pos = float4(input.pos, 1.0);  // Convert float3 to float4
	output.uv  = input.uv;
	// Note: normal and color are not used, but must be in vertex layout
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	// Sample both layers of the array texture
	float3 left  = array_tex.Sample(array_sampler, float3(input.uv, 0)).rgb;  // Layer 0 (left eye)
	float3 right = array_tex.Sample(array_sampler, float3(input.uv, 1)).rgb;  // Layer 1 (right eye)

	float3 stereo;
	if (mode == 0) {
		float left_lum  = dot(left,  float3(0.299, 0.587, 0.114));
		float right_lum = dot(right, float3(0.299, 0.587, 0.114));
		// Red-blue anaglyph
		stereo = float3(left_lum, 0, right_lum);
	} else {
		// Red-cyan Dubois optimized anaglyph
		stereo = float3(
			dot(left,  float3( 0.4561,    0.500484,  0.176381)),
			dot(right, float3( 0.378476,  0.73364,  -0.0184503)),
			dot(right, float3(-0.0721527,-0.112961,  1.2264)));
	}

	return float4(stereo, 1);
}
