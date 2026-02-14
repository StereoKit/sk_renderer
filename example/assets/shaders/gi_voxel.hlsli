#ifndef GI_VOXEL_HLSLI
#define GI_VOXEL_HLSLI

// Per-voxel storage: 6 faces as RGBA8 (one uint per face, 24 bytes total).
// Face order: faces[0]=+X, [1]=-X, [2]=+Y, [3]=-Y, [4]=+Z, [5]=-Z.
// Each face: [R:8][G:8][B:8][A:8] where A>0 means occupied.
struct Voxel {
	uint faces[6];
};

// Face index convention: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

///////////////////////////////////////////
// Grid Constants
///////////////////////////////////////////

#define GI_GRID      32
#define GI_GRID2     1024    // GI_GRID * GI_GRID
#define GI_INV_GRID  0.03125 // 1.0 / GI_GRID

///////////////////////////////////////////
// Linear Indexing
///////////////////////////////////////////

uint voxel_index(uint3 pos) {
	return pos.x + (pos.y << 5) + (pos.z << 10);
}

///////////////////////////////////////////
// RGBA8 Packing
///////////////////////////////////////////

// Pack color + occupancy into a 32-bit RGBA8 value.
// A=255 when occupied, A=0 when empty.
uint pack_rgba8(float3 color, bool occupied) {
	uint r = (uint)(saturate(color.r) * 255.0 + 0.5);
	uint g = (uint)(saturate(color.g) * 255.0 + 0.5);
	uint b = (uint)(saturate(color.b) * 255.0 + 0.5);
	uint a = occupied ? 255u : 0u;
	return (r << 24) | (g << 16) | (b << 8) | a;
}

float3 unpack_rgba8_color(uint packed) {
	float r = (float)((packed >> 24) & 0xFF) / 255.0;
	float g = (float)((packed >> 16) & 0xFF) / 255.0;
	float b = (float)((packed >> 8)  & 0xFF) / 255.0;
	return float3(r, g, b);
}

bool unpack_rgba8_occupied(uint packed) {
	return (packed & 0xFF) != 0;
}

///////////////////////////////////////////
// Per-Face Accessors
///////////////////////////////////////////

uint voxel_get_face(Voxel v, uint face_idx) {
	return v.faces[face_idx];
}

void voxel_set_face(inout Voxel v, uint face_idx, uint packed) {
	v.faces[face_idx] = packed;
}

///////////////////////////////////////////
// Voxel Queries
///////////////////////////////////////////

// Build 6-bit face mask from alpha of all 6 faces.
// Bit 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
uint voxel_face_mask(Voxel v) {
	uint mask = 0;
	[unroll] for (uint i = 0; i < 6; i++)
		if (v.faces[i] & 0xFF) mask |= (1u << i);
	return mask;
}

// Average color of occupied faces
float3 voxel_average_color(Voxel v) {
	uint   face_mask = voxel_face_mask(v);
	float3 total     = float3(0, 0, 0);
	uint   count     = 0;
	for (uint i = 0; i < 6; i++) {
		if (face_mask & (1u << i)) {
			total += unpack_rgba8_color(v.faces[i]);
			count++;
		}
	}
	return count > 0 ? total / (float)count : float3(0, 0, 0);
}

///////////////////////////////////////////
// Face Index Mapping
///////////////////////////////////////////

// Convert old face_bit (power of 2: 1,2,4,8,16,32) to face index (0-5)
uint face_bit_to_index(uint face_bit) {
	if (face_bit ==  1) return 0; // +X
	if (face_bit ==  2) return 1; // -X
	if (face_bit ==  4) return 2; // +Y
	if (face_bit ==  8) return 3; // -Y
	if (face_bit == 16) return 4; // +Z
	return 5;                     // -Z (32)
}

// Average outward normal from occupied faces. Returns (0,0,0) if faces cancel
// (e.g. thin wall with +Y and -Y) or no faces are set.
float3 voxel_face_normal(uint face_mask) {
	float3 n = float3(0, 0, 0);
	if (face_mask &  1u) n.x += 1;
	if (face_mask &  2u) n.x -= 1;
	if (face_mask &  4u) n.y += 1;
	if (face_mask &  8u) n.y -= 1;
	if (face_mask & 16u) n.z += 1;
	if (face_mask & 32u) n.z -= 1;
	return n;
}

// For ray marching: determine which face a ray enters through (dominant axis)
uint entry_mask_to_face(uint entry_mask, float3 dir) {
	float3 a = abs(dir);
	if (a.x >= a.y && a.x >= a.z) return (dir.x > 0) ? 1u : 0u; // entering -X or +X face
	if (a.y >= a.x && a.y >= a.z) return (dir.y > 0) ? 3u : 2u; // entering -Y or +Y face
	return (dir.z > 0) ? 5u : 4u;                                 // entering -Z or +Z face
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
