#ifndef GI_VOXEL_HLSLI
#define GI_VOXEL_HLSLI

///////////////////////////////////////////
// Grid Constants
///////////////////////////////////////////

// SH probe grid
#define GI_GRID      32
#define GI_GRID2     (GI_GRID * GI_GRID)
#define GI_GRID3     (GI_GRID * GI_GRID * GI_GRID)
#define GI_INV_GRID  (1.0 / (float)GI_GRID)

// Voxel 3D texture (64^3 per cascade)
#define GI_VOXEL_RES     128
#define GI_INV_VOXEL_RES (1.0 / (float)GI_VOXEL_RES)

// Cascade configuration
#define GI_CASCADE_COUNT 3

///////////////////////////////////////////
// Per-Cascade Data
///////////////////////////////////////////

struct GICascade {
	float3 volume_min;
	float  cell_size;     // world-space size of one voxel
	float3 volume_inv;    // 1.0 / (volume_max - volume_min)
	int    scroll_x;      // toroidal scroll offset per axis
	int    scroll_y;
	int    scroll_z;
	uint   _pad0;
	uint   _pad1;
};

///////////////////////////////////////////
// GI Buffer (shared across all GI shaders)
///////////////////////////////////////////

cbuffer GIBuffer : register(b12, space0) {
	GICascade gi_cascades[GI_CASCADE_COUNT];
	float  gi_intensity;
	uint   gi_grid_size;
	uint   gi_cascade_count;
	uint   gi_active_cascade;
};

///////////////////////////////////////////
// Linear Indexing (used for SH probes)
///////////////////////////////////////////

uint voxel_index(uint3 pos) {
	return pos.x + pos.y * GI_GRID + pos.z * GI_GRID2;
}

uint probe_index(uint3 pos, uint cascade) {
	return cascade * GI_GRID3 + voxel_index(pos);
}

uint probe_index_scrolled(uint3 grid_pos, uint cascade) {
	int3 scroll = int3(gi_cascades[cascade].scroll_x, gi_cascades[cascade].scroll_y, gi_cascades[cascade].scroll_z);
	uint3 buf   = uint3((int3(grid_pos) + scroll + GI_GRID * 256) % GI_GRID);
	return cascade * GI_GRID3 + voxel_index(buf);
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
