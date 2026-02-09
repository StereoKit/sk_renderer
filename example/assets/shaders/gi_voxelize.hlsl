//--name = gi_voxelize

// Compute shader to write captured radiance into a 3D voxel texture.
// Each dispatch processes one 32x32 capture from one axis-aligned direction,
// accumulating lit color into the voxel grid.

Texture2D<float4>   capture_tex : register(t0);
RWTexture3D<float4> voxel_tex   : register(u0);

uint  axis;        // 0=X, 1=Y, 2=Z
uint  layer_index; // 0-31
uint  grid_size;   // 32
uint  flip_x;      // flip pixel X before mapping to texel
uint  flip_y;      // flip pixel Y before mapping to texel
uint  face_bit;    // 6-bit face mask for this capture direction

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= grid_size || id.y >= grid_size) return;

	float4 color = capture_tex[id.xy];

	// Correct for view-space axis flips (same mapping as gi_accumulate)
	uint px = flip_x ? (grid_size - 1 - id.x) : id.x;
	uint py = flip_y ? (grid_size - 1 - id.y) : id.y;

	// Map (pixel_x, pixel_y, layer_index) -> 3D texel based on capture axis
	uint3 texel;
	if      (axis == 0) texel = uint3(layer_index, py, px); // X-layer: image plane is ZY
	else if (axis == 1) texel = uint3(px, layer_index, py); // Y-layer: image plane is XZ
	else                texel = uint3(px, py, layer_index); // Z-layer: image plane is XY

	// Accumulate radiance (rgb) and opacity (a).
	// Each voxel receives 6 captures per cycle (2 per axis).
	voxel_tex[texel] += float4(color.rgb, color.a > 0 ? (float)face_bit : 0.0);
}
