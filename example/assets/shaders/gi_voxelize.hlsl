//--name = gi_voxelize

// Compute shader to write captured radiance into the voxel structured buffer.
// Each frame processes 1 axis x 3 consecutive layers (2 on the last batch).
// Dispatched with z = layer_count (1-3). Each thread handles one voxel,
// reading both +/- views and writing both face halves — no race condition.
//
// Capture texture is 2x voxel grid resolution (64x64 for 32^3 grid).
// Bilinear SampleLevel at the center of each 2x2 block gives a box-filter downsample.
//
// Texture layout: layers [0,1] = layer+0 (pos,neg), [2,3] = layer+1, [4,5] = layer+2
// Face convention: +axis camera sees the -axis face, -axis camera sees the +axis face.
// Negative view is always the horizontal mirror of the positive view.

#include "gi_voxel.hlsli"

Texture2DArray<float4>    capture_tex   : register(t0);
SamplerState              capture_tex_s : register(s0);
RWStructuredBuffer<Voxel> voxel_buf     : register(u0);

uint  axis;        // 0=X, 1=Y, 2=Z
uint  layer_start; // first layer index in this batch (0, 3, 6, ...)
uint  layer_count; // layers in this batch (3, or 2 for last batch)
uint  grid_size;   // 32

// Pixel-to-voxel flips for the positive direction camera.
// Negative direction always differs by exactly one X-mirror.
static const uint FLIP_X[3] = { 0, 1, 1 };
static const uint FLIP_Y[3] = { 1, 0, 1 };

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= GI_GRID || id.y >= GI_GRID || id.z >= layer_count) return;

	uint layer = layer_start + id.z;

	// id.x, id.y are voxel coordinates in the capture plane.
	// Invert the per-view flips to find the texture pixel for each direction.
	float pos_px = FLIP_X[axis] ? (GI_GRID - 1 - id.x) : id.x;
	float pos_py = FLIP_Y[axis] ? (GI_GRID - 1 - id.y) : id.y;
	float neg_px = GI_GRID - 1 - pos_px; // negative is always horizontal mirror
	float neg_py = pos_py;

	// Sample at center of 2x2 block — bilinear filter gives box-filter downsample
	float2 pos_uv = (float2(pos_px, pos_py) + 0.5) / GI_GRID;
	float2 neg_uv = (float2(neg_px, neg_py) + 0.5) / GI_GRID;

	float4 color_pos = capture_tex.SampleLevel(capture_tex_s, float3(pos_uv, id.z * 2),     0);
	float4 color_neg = capture_tex.SampleLevel(capture_tex_s, float3(neg_uv, id.z * 2 + 1), 0);

	// Map voxel plane coords to 3D texel (same convention as before)
	uint3 texel;
	if      (axis == 0) texel = uint3(layer, id.y, id.x);
	else if (axis == 1) texel = uint3(id.x, layer, id.y);
	else                texel = uint3(id.x, id.y, layer);

	// Face indices: +axis camera sees -axis face, -axis camera sees +axis face
	uint pos_face = axis * 2 + 1; // -X(1), -Y(3), -Z(5)
	uint neg_face = axis * 2;     // +X(0), +Y(2), +Z(4)

	// Write both faces of the same voxel — single thread, no race
	uint  idx = voxel_index(texel);
	Voxel v   = voxel_buf[idx];
	voxel_set_face(v, pos_face, pack_rgba8(color_pos.rgb, color_pos.a > 0));
	voxel_set_face(v, neg_face, pack_rgba8(color_neg.rgb, color_neg.a > 0));
	voxel_buf[idx] = v;
}
