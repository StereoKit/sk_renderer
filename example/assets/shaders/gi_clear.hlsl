//--name = gi_clear

// Compute shader to decay voxel radiance at cycle start.
// decay=0 clears to zero, decay<1 applies temporal fade (e.g. 0.85).
// SH decay is handled by the EMA in gi_voxel_to_sh instead.

RWTexture3D<float4> voxel : register(u0);

float decay;

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	voxel[id] *= decay;
}
