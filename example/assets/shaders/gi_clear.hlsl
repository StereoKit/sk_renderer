//--name = gi_clear
//--cascade_offset = 0

// Compute shader to clear the 3D voxel texture.
// cascade_offset selects which cascade slice to clear (0, GI_VOXEL_RES, 2*GI_VOXEL_RES, ...).

#include "gi_voxel.hlsli"

uint cascade_offset;

RWTexture3D<float4> voxel_tex : register(u0);

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= GI_VOXEL_RES || id.y >= GI_VOXEL_RES || id.z >= GI_VOXEL_RES) return;
	voxel_tex[uint3(id.x, id.y, id.z + cascade_offset)] = float4(0, 0, 0, 0);
}
