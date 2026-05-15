//--name = skr_mipgen_2d
// Built-in fallback mipmap generator for non-layered 2D textures.
// Single LINEAR tap at the destination texel center = exact 2x2 box filter
// (hardware bilinear gives 0.25 weight to each of the 4 surrounding source
// texels). Used when blit-based mipgen is unsupported for the source format
// (e.g. B10G11R11_UFLOAT on Mesa llvmpipe).

uint2 src_size;
uint2 dst_size;
uint  src_mip_level;
uint  mip_max;

Texture2D<float4> src_tex     : register(t1);
SamplerState      src_sampler : register(s1);

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

psIn vs(uint id : SV_VertexID) {
	psIn output;
	output.uv  = float2((id << 1) & 2, id & 2);
	output.pos = float4(output.uv * 2.0 - 1.0, 0, 1);
	return output;
}

float4 ps(psIn input) : SV_Target {
	return src_tex.SampleLevel(src_sampler, input.uv, src_mip_level);
}
