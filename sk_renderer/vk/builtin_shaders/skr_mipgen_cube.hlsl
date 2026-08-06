//--name = skr_mipgen_cube
// Built-in fallback mipmap generator for cubemap textures.
// Single LINEAR tap at the destination texel center = exact 2x2 box filter
// within the face (hardware bilinear gives 0.25 weight to each of the 4
// surrounding source texels). Used when blit-based mipgen is unsupported
// for the source format (e.g. B10G11R11_UFLOAT on Mesa llvmpipe). Each
// output face is rendered via multiview (SV_ViewID = face index).

uint2 src_size;
uint2 dst_size;
uint  src_mip_level;
uint  mip_max;

TextureCube<float4> src_tex     : register(t1);
SamplerState        src_sampler : register(s1);

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

float3 uv_to_direction(float2 uv, uint face) {
	float2 ndc = uv * 2.0 - 1.0;
	if (face == 0) return normalize(float3( 1.0, -ndc.y, -ndc.x));
	if (face == 1) return normalize(float3(-1.0, -ndc.y,  ndc.x));
	if (face == 2) return normalize(float3( ndc.x,  1.0,  ndc.y));
	if (face == 3) return normalize(float3( ndc.x, -1.0, -ndc.y));
	if (face == 4) return normalize(float3( ndc.x, -ndc.y,  1.0));
	               return normalize(float3(-ndc.x, -ndc.y, -1.0));
}

psIn vs(uint id : SV_VertexID) {
	psIn output;
	output.uv  = float2(id & 2, (id << 1) & 2);
	output.pos = float4(output.uv * float2(2, -2) + float2(-1, 1), 0, 1);
	return output;
}

float4 ps(psIn input, uint face : SV_ViewID) : SV_Target {
	return src_tex.SampleLevel(src_sampler, uv_to_direction(input.uv, face), src_mip_level);
}
