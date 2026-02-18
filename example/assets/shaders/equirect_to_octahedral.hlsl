//--name = equirect_to_octahedral
// Converts an equirectangular (lat/long) panorama to an octahedral map

Texture2D<float4> equirect_tex     : register(t0);
SamplerState      equirect_sampler : register(s0);

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

// Octahedral decoding adapted for Y-up from:
// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
// Lower hemisphere unwrap from https://twitter.com/Stubbesaurus/status/937994790553227264
float3 octahedral_to_direction(float2 f) {
	f = f * 2.0 - 1.0;

	float3 n = float3(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
	float  t = saturate(-n.y);
	n.xz += n.xz >= 0.0 ? -t : t;

	return normalize(n);
}

// Convert 3D direction to equirectangular UV coordinates
float2 direction_to_equirect_uv(float3 dir) {
	dir = normalize(dir);

	const float PI = 3.14159265359;
	float theta = atan2(dir.z, dir.x);
	float phi   = acos(dir.y);

	float u = (theta / (2.0 * PI)) + 0.5;
	float v = phi / PI;

	return float2(u, v);
}

// Vertex shader - fullscreen triangle (no mesh input needed for blit)
psIn vs(uint id : SV_VertexID) {
	psIn output;

	output.uv  = float2((id << 1) & 2, id & 2);
	output.pos = float4(output.uv * 2.0 - 1.0, 0, 1);

	return output;
}

// Pixel shader - convert octahedral UV -> direction -> equirect UV -> sample
// The octahedral texture uses mirrored repeat addressing, so bilinear
// filtering at texture edges naturally wraps correctly without padding.
float4 ps(psIn input) : SV_Target {
	float3 dir         = octahedral_to_direction(input.uv);
	float2 equirect_uv = direction_to_equirect_uv(dir);
	return equirect_tex.Sample(equirect_sampler, equirect_uv);
}
