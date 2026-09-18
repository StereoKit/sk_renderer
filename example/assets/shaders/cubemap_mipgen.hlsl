//--name = cubemap_mipgen
// GGX specular prefilter for IBL (Karis 2013, split-sum approximation).
// Cascade topology: each destination mip importance-samples the previous mip.

static const float PI = 3.14159265359;

uint2 src_size;      // Source mip dimensions
uint2 dst_size;      // Destination mip dimensions
uint  src_mip_level; // Source mip level to read from
uint  mip_max;
uint  _pad[2];

TextureCube<float4> src_tex     : register(t1);  // Source cubemap texture
SamplerState        src_sampler : register(s1);  // Linear sampler for source

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

// Convert UV coordinates to cubemap direction for a specific face
float3 uv_to_direction(float2 uv, uint face) {
	// UV goes from 0 to 1, convert to -1 to 1
	float2 ndc = uv * 2.0 - 1.0;

	// Map to cubemap face direction
	// Faces: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5
	float3 dir;
	if (face == 0) {      // +X
		dir = float3(1.0, -ndc.y, -ndc.x);
	} else if (face == 1) { // -X
		dir = float3(-1.0, -ndc.y, ndc.x);
	} else if (face == 2) { // +Y
		dir = float3(ndc.x, 1.0, ndc.y);
	} else if (face == 3) { // -Y
		dir = float3(ndc.x, -1.0, -ndc.y);
	} else if (face == 4) { // +Z
		dir = float3(ndc.x, -ndc.y, 1.0);
	} else {              // -Z
		dir = float3(-ndc.x, -ndc.y, -1.0);
	}

	return normalize(dir);
}

// Vertex shader - fullscreen triangle, face index from SV_ViewID
psIn vs(uint id : SV_VertexID) {
	psIn output;

	// Generate fullscreen triangle
	output.uv  = float2(id & 2, (id << 1) & 2);
	output.pos = float4(output.uv * float2(2, -2) + float2(-1, 1), 0, 1);

	return output;
}

// Low-discrepancy Hammersley sequence for importance sampling.
// See http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float radical_inverse_vdc(uint bits) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10;
}
float2 hammersley(uint i, uint n) {
	return float2(float(i) / float(n), radical_inverse_vdc(i));
}

// Sample a half-vector from the GGX distribution around normal N.
// Karis 2013, "Real Shading in Unreal Engine 4".
float3 importance_sample_ggx(float2 xi, float3 n, float roughness) {
	float  a        = roughness * roughness;
	float  phi      = 2.0 * PI * xi.x;
	float  cos_t    = sqrt((1.0 - xi.y) / (1.0 + (a*a - 1.0) * xi.y));
	float  sin_t    = sqrt(1.0 - cos_t * cos_t);
	float3 h_local  = float3(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);

	// Build orthonormal basis around N.
	float3 up       = abs(n.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
	float3 tangent  = normalize(cross(up, n));
	float3 bitang   = cross(n, tangent);
	return normalize(tangent * h_local.x + bitang * h_local.y + n * h_local.z);
}

// GGX importance-sampled prefilter with Karis's split-sum assumption (V = R = N).
// Mip 1 uses a single hardware-bilinear tap — GGX at roughness < ~0.15 is too
// sharp to benefit from stochastic sampling (the lobe barely covers one texel).
float4 ps(psIn input, uint face : SV_ViewID) : SV_Target {
	float3 n = uv_to_direction(input.uv, face);

	// Mip 1 bypass: direct linear sample, ~16x cheaper than GGX integration.
	if (src_mip_level == 0) {
		return float4(src_tex.SampleLevel(src_sampler, n, 0).rgb, 1);
	}

	// Destination mip's roughness. We're writing mip (src_mip_level + 1).
	float roughness = float(src_mip_level + 1) / float(mip_max - 1);

	// Sample count scales with output mip. Cascade topology doesn't let us use
	// filtered importance sampling (LOD bias), so we compensate with brute
	// force — variance drops as sqrt(N). Ramps from 64 → 256 top-to-bottom.
	uint sample_count = min(32u + src_mip_level * 32u, 256u);

	float3 v            = n;
	float3 color        = 0;
	float  total_weight = 0;
	for (uint i = 0; i < sample_count; i++) {
		float2 xi  = hammersley(i, sample_count);
		float3 h   = importance_sample_ggx(xi, n, roughness);
		float3 l   = normalize(2.0 * dot(v, h) * h - v);
		float  ndl = saturate(dot(n, l));
		if (ndl > 0) {
			// Firefly clamp: prevents bright HDRI pixels (sun, specular
			// highlights) from dominating low-probability GGX samples.
			float3 s      = min(src_tex.SampleLevel(src_sampler, l, src_mip_level).rgb, 10.0);
			color        += s * ndl;
			total_weight += ndl;
		}
	}

	return float4(color / max(total_weight, 1e-4), 1);
}
