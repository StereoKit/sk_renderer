//--name = cubemap_mipgen
// High-quality cubemap mipmap generation for IBL
// Uses wider kernel sampling to create proper roughness blur

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
	output.uv  = float2((id << 1) & 2, id & 2);
	output.pos = float4(output.uv * 2.0 - 1.0, 0, 1);

	return output;
}

// High-quality filter with multiple samples
// The number of samples increases with mip level for better quality
float4 ps(psIn input, uint face : SV_ViewID) : SV_Target {
	// Get the main direction for this pixel
	float3 main_dir = uv_to_direction(input.uv, face);

	// Calculate roughness based on destination mip level
	// We're generating mip (src_mip_level + 1), so higher src means more blur needed
	float roughness = float(src_mip_level + 1) / (mip_max-1);

	// Sample count increases with each mip level for better quality blur
	// Start with more samples to avoid aliasing artifacts
	// Mip 1: 16 samples, Mip 2: 20 samples, Mip 3: 24 samples, etc.
	int sample_count = 16 + src_mip_level * 4;
	sample_count = min(sample_count, 64); // Cap at 64 samples

	float4 color_sum = float4(0, 0, 0, 0);
	float total_weight = 0.0;

	// Create orthogonal basis around the main direction
	float3 up_vec = abs(main_dir.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
	float3 tangent = normalize(cross(up_vec, main_dir));
	float3 bitangent = cross(main_dir, tangent);

	// Sample in a cone around the main direction
	// Use a simple spiral pattern for better distribution
	const float PI = 3.14159265359;
	const float GOLDEN_ANGLE = PI * 0.7639320225; // Golden angle in radians

	float blur_radius = roughness * 0.5; // How wide the cone is

	for (int i = 0; i < sample_count; i++) {
		// Spiral pattern using golden angle
		float angle  = float(i) * GOLDEN_ANGLE;
		float radius = (float(i) / float(sample_count)) * blur_radius;

		// Convert to 2D offset
		float2 offset = float2(cos(angle), sin(angle)) * radius;

		// Apply to tangent space to get sample direction
		float3 sample_dir = normalize(
			main_dir +
			tangent * offset.x +
			bitangent * offset.y
		);

		// Sample the cubemap
		float4 sample_color = src_tex.SampleLevel(src_sampler, sample_dir, src_mip_level);

		// Weight samples based on distance from center (Gaussian-like)
		float dist = length(offset);
		float weight = exp(-dist * dist * 2.0);

		color_sum += sample_color * weight;
		total_weight += weight;
	}

	// Normalize
	if (total_weight > 0.0001) {
		color_sum /= total_weight;
	}

	return color_sum;
}
