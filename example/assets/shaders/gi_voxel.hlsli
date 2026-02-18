#ifndef GI_VOXEL_HLSLI
#define GI_VOXEL_HLSLI

///////////////////////////////////////////
// Grid Constants
///////////////////////////////////////////

// SH probe grid (16^3)
#define GI_GRID      16
#define GI_GRID2     256       // GI_GRID * GI_GRID
#define GI_INV_GRID  0.0625    // 1.0 / GI_GRID

// Voxel 3D texture (128^3)
#define GI_VOXEL_RES     64
#define GI_INV_VOXEL_RES 0.015625 // 1.0 / GI_VOXEL_RES

///////////////////////////////////////////
// Linear Indexing (used for SH probes)
///////////////////////////////////////////

uint voxel_index(uint3 pos) {
	return pos.x + pos.y * GI_GRID + pos.z * GI_GRID2;
}

///////////////////////////////////////////
// SH Probe Storage (half-precision)
///////////////////////////////////////////

// Per-probe spherical harmonics (L1 = 4 coefficients per color channel).
// Stored as half-precision: 3 channels x uint2 (4 half values per uint2) = 24 bytes.
struct SHProbe {
	uint2 r; // 4 half-precision SH coefficients for red
	uint2 g; // 4 half-precision SH coefficients for green
	uint2 b; // 4 half-precision SH coefficients for blue
};

// Pack float4 SH coefficients into uint2 (2 halves per uint)
uint2 sh_pack(float4 v) {
	return uint2(
		f32tof16(v.x) | (f32tof16(v.y) << 16),
		f32tof16(v.z) | (f32tof16(v.w) << 16)
	);
}

// Unpack uint2 back to float4 SH coefficients
float4 sh_unpack(uint2 p) {
	return float4(
		f16tof32(p.x),
		f16tof32(p.x >> 16),
		f16tof32(p.y),
		f16tof32(p.y >> 16)
	);
}

#endif // GI_VOXEL_HLSLI
