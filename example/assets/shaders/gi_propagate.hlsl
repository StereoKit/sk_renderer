//--name = gi_propagate

// LPV-style propagation: each probe receives light from its 6 face-adjacent
// neighbors, spreading captured radiance through the volume. Run multiple
// iterations to propagate further. Ping-pongs between source/dest textures.

#include "gi_voxel.hlsli"

StructuredBuffer<SHProbe>   sh_probes_in  : register(t0);
RWStructuredBuffer<SHProbe> sh_probes_out : register(u0);

float damping;
uint  grid_size;

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= GI_GRID || id.y >= GI_GRID || id.z >= GI_GRID) return;

	uint   idx = voxel_index(id);
	SHProbe sp = sh_probes_in[idx];
	float4 r   = sh_unpack(sp.r);
	float4 g   = sh_unpack(sp.g);
	float4 b   = sh_unpack(sp.b);

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
		if (npos.x < 0 || npos.x >= (int)GI_GRID ||
			npos.y < 0 || npos.y >= (int)GI_GRID ||
			npos.z < 0 || npos.z >= (int)GI_GRID)
			continue;

		SHProbe nsp = sh_probes_in[voxel_index(uint3(npos))];
		float4  nr  = sh_unpack(nsp.r);
		float4  ng  = sh_unpack(nsp.g);
		float4  nb  = sh_unpack(nsp.b);

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

	sh_probes_out[idx].r = sh_pack(r);
	sh_probes_out[idx].g = sh_pack(g);
	sh_probes_out[idx].b = sh_pack(b);
}
