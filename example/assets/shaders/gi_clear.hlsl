//--name = gi_clear

// Compute shader to clear the voxel color buffer.

#include "gi_voxel.hlsli"

RWStructuredBuffer<Voxel> voxel : register(u0);

uint grid_size;

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= GI_GRID || id.y >= GI_GRID || id.z >= GI_GRID) return;
	uint idx = voxel_index(id);

	voxel[idx].faces_x = 0;
	voxel[idx].faces_y = 0;
	voxel[idx].faces_z = 0;
}
