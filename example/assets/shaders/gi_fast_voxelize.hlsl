//--name = gi_fast_voxelize

// Compute shader for fast depth-based voxelization.
// Processes all 6 views in one dispatch, reconstructing 3D voxel positions
// from (pixel, depth, view_index). Accumulates into the voxel grid.
//
// Views: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

Texture2DArray<float4> capture_color : register(t0);
Texture2DArray<float>  capture_depth : register(t1);
RWTexture3D<float4>    voxel_tex    : register(u0);

uint grid_size;   // 32
uint from_center; // 1 = center outward, 0 = edges inward

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= grid_size || id.y >= grid_size || id.z >= 6) return;

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

	uint depth_idx = clamp(uint(voxel_frac * grid_size), 0, grid_size - 1);

	// Pixel to perpendicular voxel indices (matching gi_voxelize flip conventions)
	uint flip_y = (axis != 1) ? 1 : 0;
	uint flip_x = (axis == 0) ? (1 - is_positive) : is_positive;

	uint px = flip_x ? (grid_size - 1 - id.x) : id.x;
	uint py = flip_y ? (grid_size - 1 - id.y) : id.y;

	// Map to 3D texel (same axis mapping as gi_voxelize.hlsl)
	uint3 texel;
	if      (axis == 0) texel = uint3(depth_idx, py, px);
	else if (axis == 1) texel = uint3(px, depth_idx, py);
	else                texel = uint3(px, py, depth_idx);

	voxel_tex[texel] += float4(color.rgb, 1.0);
}
