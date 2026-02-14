//--name = gi_fast_voxelize

// Compute shader for fast depth-based voxelization.
// Processes all 6 views in one dispatch, reconstructing 3D voxel positions
// from (pixel, depth, view_index). Writes per-face RGB5A1 into the voxel grid.
// Occupancy is encoded in each face's alpha bit (A=1 means occupied).
//
// Capture textures are 2x voxel grid resolution (64x64 for 32^3 grid).
// Each sub-pixel independently maps to a voxel via depth — better coverage.
//
// Views: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

#include "gi_voxel.hlsli"

Texture2DArray<float4>    capture_color : register(t0);
Texture2DArray<float>     capture_depth : register(t1);
RWStructuredBuffer<Voxel> voxel_buf     : register(u0);

uint grid_size;   // 32
uint from_center; // 1 = center outward, 0 = edges inward

static const uint CAPTURE_RES = GI_GRID * 2;

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= CAPTURE_RES || id.y >= CAPTURE_RES || id.z >= 6) return;

	uint view_index  = id.z;
	uint axis        = view_index / 2;
	uint is_positive = (view_index % 2 == 0) ? 1 : 0;

	// Load color and depth for this pixel/view
	float4 color = capture_color.Load(int4(id.x, id.y, view_index, 0));
	float  depth = capture_depth.Load(int4(id.x, id.y, view_index, 0));

	// Skip empty pixels
	if (depth >= 1.0 || color.a <= 0) return;

	// Depth to normalized voxel coordinate along the view axis [0, 1]
	float voxel_frac;
	if (from_center) {
		// Center outward: depth 0 = center (0.5), depth 1 = edge
		voxel_frac = is_positive ? (0.5 + depth * 0.5) : (0.5 - depth * 0.5);
	} else {
		// Edge inward: depth 0 = near edge, depth 1 = far edge
		voxel_frac = is_positive ? depth : (1.0 - depth);
	}

	uint depth_idx = clamp(uint(voxel_frac * GI_GRID), 0, GI_GRID - 1);

	// Pixel to perpendicular voxel indices (flip in pixel space, then halve)
	uint flip_y = (axis != 1) ? 1 : 0;
	uint flip_x = (axis == 0) ? (1 - is_positive) : is_positive;

	uint px = flip_x ? (CAPTURE_RES - 1 - id.x) : id.x;
	uint py = flip_y ? (CAPTURE_RES - 1 - id.y) : id.y;
	px /= 2;
	py /= 2;

	// Map to 3D texel (same axis mapping as gi_voxelize.hlsl)
	uint3 texel;
	if      (axis == 0) texel = uint3(depth_idx, py, px);
	else if (axis == 1) texel = uint3(px, depth_idx, py);
	else                texel = uint3(px, py, depth_idx);

	// Face index from view_index: captures from +X see -X faces (index 1), etc.
	const uint face_indices[6] = { 1, 0, 3, 2, 5, 4 };
	uint face_idx = face_indices[view_index];

	uint idx    = voxel_index(texel);
	uint packed = pack_rgba8(color.rgb, true);

	Voxel v = voxel_buf[idx];
	voxel_set_face(v, face_idx, packed);
	voxel_buf[idx] = v;
}
