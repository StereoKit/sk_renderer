//--name = gi_clear

// Compute shader to clear the voxel color buffer.

#include "gi_voxel.hlsli"

RWStructuredBuffer<Voxel> voxel : register(u0);

uint grid_size;

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= GI_GRID || id.y >= GI_GRID || id.z >= GI_GRID) return;
	uint idx = voxel_index(id);

	[unroll] for (uint i = 0; i < 6; i++)
		voxel[idx].faces[i] = 0;
}
