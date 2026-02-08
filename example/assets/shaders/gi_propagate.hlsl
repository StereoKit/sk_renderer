//--name = gi_propagate

// LPV-style propagation: each probe receives light from its 6 face-adjacent
// neighbors, spreading captured radiance through the volume. Run multiple
// iterations to propagate further. Ping-pongs between source/dest textures.

Texture3D<float4>   sh_r_in  : register(t0);
Texture3D<float4>   sh_g_in  : register(t1);
Texture3D<float4>   sh_b_in  : register(t2);
RWTexture3D<float4> sh_r_out : register(u0);
RWTexture3D<float4> sh_g_out : register(u1);
RWTexture3D<float4> sh_b_out : register(u2);

float damping;
uint  grid_size;

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= grid_size || id.y >= grid_size || id.z >= grid_size) return;

	float4 r = sh_r_in[id];
	float4 g = sh_g_in[id];
	float4 b = sh_b_in[id];

	// 6 face-adjacent neighbor offsets and the direction from each neighbor to us
	static const int3 offsets[6] = {
		int3( 1, 0, 0), int3(-1, 0, 0),
		int3( 0, 1, 0), int3( 0,-1, 0),
		int3( 0, 0, 1), int3( 0, 0,-1)
	};
	static const float3 from_neighbor[6] = {
		float3(-1, 0, 0), float3( 1, 0, 0),
		float3( 0,-1, 0), float3( 0, 1, 0),
		float3( 0, 0,-1), float3( 0, 0, 1)
	};

	static const float W = 2.09440; // 4*pi/6, solid angle per face

	for (int i = 0; i < 6; i++) {
		int3 npos = int3(id) + offsets[i];
		if (npos.x < 0 || npos.x >= (int)grid_size ||
			npos.y < 0 || npos.y >= (int)grid_size ||
			npos.z < 0 || npos.z >= (int)grid_size)
			continue;

		float4 nr = sh_r_in[uint3(npos)];
		float4 ng = sh_g_in[uint3(npos)];
		float4 nb = sh_b_in[uint3(npos)];

		// SH basis for direction from neighbor toward us (raw, no cosine weights)
		float3 d = from_neighbor[i];
		float4 eval_basis = float4(0.28209, 0.48860 * d.y, 0.48860 * d.z, 0.48860 * d.x);

		// Radiance the neighbor has aimed toward us
		float inc_r = max(0, dot(nr, eval_basis));
		float inc_g = max(0, dot(ng, eval_basis));
		float inc_b = max(0, dot(nb, eval_basis));

		// Project into our SH as light arriving from the neighbor's direction.
		// Arrival direction at our probe is -d (toward the neighbor).
		float4 proj_basis = float4(0.28209, 0.48860 * (-d.y), 0.48860 * (-d.z), 0.48860 * (-d.x));

		r += proj_basis * inc_r * W * damping;
		g += proj_basis * inc_g * W * damping;
		b += proj_basis * inc_b * W * damping;
	}

	sh_r_out[id] = r;
	sh_g_out[id] = g;
	sh_b_out[id] = b;
}
