//--name = gi_clear

// Compute shader to clear the 3D voxel texture.

#include "gi_voxel.hlsli"

RWTexture3D<float4> voxel_tex : register(u0);

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= GI_VOXEL_RES || id.y >= GI_VOXEL_RES || id.z >= GI_VOXEL_RES) return;
	voxel_tex[id] = float4(0, 0, 0, 0);
}
