//--name = mipgen_alpha_weighted_render
// Alpha-weighted mipmap generation fragment shader
// Properly handles transparent textures by weighting colors by alpha

uint2 src_size;      // Source mip dimensions
uint2 dst_size;      // Destination mip dimensions
uint  src_mip_level; // Source mip level to read from
uint  mip_max;
uint  _pad[2];

Texture2D<float4> src_tex     : register(t1);  // Source texture (read from previous mip)
SamplerState      src_sampler : register(s1);  // Linear sampler for source

struct vs_out {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

// Vertex shader - fullscreen triangle
vs_out vs(uint id : SV_VertexID) {
	vs_out output;
	// Generate fullscreen triangle, uv (0,0) at top-left
	output.uv  = float2(id & 2, (id << 1) & 2);
	output.pos = float4(output.uv * float2(2, -2) + float2(-1, 1), 0, 1);
	return output;
}

// Alpha-weighted 2x2 box filter
// This prevents dark halos around transparent edges
float4 ps(vs_out input) : SV_Target {
	// Calculate UV coordinates for the center of this destination pixel
	float2 dst_uv = input.uv;

	// Sample 4 pixels from source mip (2x2 box)
	// Offset by half a texel to sample pixel centers
	float2 texel_size = 1.0 / float2(src_size);
	float2 uv_offsets[4] = {
		float2(-0.25, -0.25) * texel_size,
		float2( 0.25, -0.25) * texel_size,
		float2(-0.25,  0.25) * texel_size,
		float2( 0.25,  0.25) * texel_size,
	};

	float4 weighted_sum = float4(0, 0, 0, 0);
	float  total_weight = 0.0;

	// Sample and weight by alpha
	for (int i = 0; i < 4; i++) {
		float4 sample = src_tex.SampleLevel(src_sampler, dst_uv + uv_offsets[i], src_mip_level);
		float weight = sample.a;  // Weight by alpha

		// Premultiply RGB by alpha for proper blending
		weighted_sum.rgb += sample.rgb * weight;
		weighted_sum.a += weight;
		total_weight += weight;
	}

	// Normalize
	if (total_weight > 0.0001) {
		weighted_sum.rgb /= total_weight;
		weighted_sum.a /= 4.0;  // Average alpha
	}

	return weighted_sum;
}
