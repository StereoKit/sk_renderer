#ifndef GI_VOXEL_HLSLI
#define GI_VOXEL_HLSLI

// Per-voxel storage: 6 faces packed as 3 uint (two RGB5A1 per uint).
// Face order: +X/-X in faces_x, +Y/-Y in faces_y, +Z/-Z in faces_z.
// Lower 16 bits = positive axis face, upper 16 bits = negative axis face.
// Each face: [R:5][G:5][B:5][A:1] where A=1 means occupied.
struct Voxel {
	uint faces_x; // lower 16 = +X (RGB5A1), upper 16 = -X (RGB5A1)
	uint faces_y; // lower 16 = +Y, upper 16 = -Y
	uint faces_z; // lower 16 = +Z, upper 16 = -Z
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
// RGB5A1 Packing
///////////////////////////////////////////

// Pack color + occupancy into a 16-bit RGB5A1 value.
// Layout: [R:5][G:5][B:5][A:1] = 16 bits. A=1 means face is occupied.
uint pack_rgb5a1(float3 color, bool occupied) {
	uint r = (uint)(saturate(color.r) * 31.0 + 0.5);
	uint g = (uint)(saturate(color.g) * 31.0 + 0.5);
	uint b = (uint)(saturate(color.b) * 31.0 + 0.5);
	return (r << 11) | (g << 6) | (b << 1) | (occupied ? 1u : 0u);
}

float3 unpack_rgb5a1_color(uint packed) {
	float r = (float)((packed >> 11) & 0x1F) / 31.0;
	float g = (float)((packed >> 6)  & 0x1F) / 31.0;
	float b = (float)((packed >> 1)  & 0x1F) / 31.0;
	return float3(r, g, b);
}

bool unpack_rgb5a1_occupied(uint packed) {
	return (packed & 1u) != 0;
}

///////////////////////////////////////////
// Face Pair Packing (two faces per uint)
///////////////////////////////////////////

uint unpack_positive_face(uint packed_pair) {
	return packed_pair & 0xFFFF;
}

uint unpack_negative_face(uint packed_pair) {
	return (packed_pair >> 16) & 0xFFFF;
}

///////////////////////////////////////////
// Per-Face Accessors
///////////////////////////////////////////

// Get a specific face's packed RGB5A1 uint16 by face index
uint voxel_get_face(Voxel v, uint face_idx) {
	uint pair;
	if      (face_idx < 2) pair = v.faces_x;
	else if (face_idx < 4) pair = v.faces_y;
	else                   pair = v.faces_z;
	return (face_idx & 1) == 0 ? (pair & 0xFFFF) : ((pair >> 16) & 0xFFFF);
}

// Set a specific face in a Voxel
void voxel_set_face(inout Voxel v, uint face_idx, uint packed_rgb5a1) {
	uint val_lo = packed_rgb5a1 & 0xFFFF;
	uint val_hi = (packed_rgb5a1 & 0xFFFF) << 16;
	if      (face_idx == 0) v.faces_x = (v.faces_x & 0xFFFF0000) | val_lo;
	else if (face_idx == 1) v.faces_x = (v.faces_x & 0x0000FFFF) | val_hi;
	else if (face_idx == 2) v.faces_y = (v.faces_y & 0xFFFF0000) | val_lo;
	else if (face_idx == 3) v.faces_y = (v.faces_y & 0x0000FFFF) | val_hi;
	else if (face_idx == 4) v.faces_z = (v.faces_z & 0xFFFF0000) | val_lo;
	else                    v.faces_z = (v.faces_z & 0x0000FFFF) | val_hi;
}

///////////////////////////////////////////
// Voxel Queries
///////////////////////////////////////////

// Build 6-bit face mask from alpha bits of all 6 faces.
// Bit 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
uint voxel_face_mask(Voxel v) {
	uint mask = 0;
	if (v.faces_x        & 1u) mask |=  1u; // +X
	if ((v.faces_x >> 16) & 1u) mask |=  2u; // -X
	if (v.faces_y        & 1u) mask |=  4u; // +Y
	if ((v.faces_y >> 16) & 1u) mask |=  8u; // -Y
	if (v.faces_z        & 1u) mask |= 16u; // +Z
	if ((v.faces_z >> 16) & 1u) mask |= 32u; // -Z
	return mask;
}

// Average color of occupied faces
float3 voxel_average_color(Voxel v) {
	uint   face_mask = voxel_face_mask(v);
	float3 total     = float3(0, 0, 0);
	uint   count     = 0;
	for (uint i = 0; i < 6; i++) {
		if (face_mask & (1u << i)) {
			total += unpack_rgb5a1_color(voxel_get_face(v, i));
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
