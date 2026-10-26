// IEC 61966-2-1 transfer function. Used by the SRGB_ENCODE pipeline variants
// to gamma-encode linear-light sources (float or sRGB-view textures, which
// Load as linear) so endpoints quantize in the space the *_srgb output
// format decodes from.
float3 linear_to_srgb(float3 c) {
	c = max(c, 0.0);
	float3 lo = c * 12.92;
	float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
	return lerp(lo, hi, step(0.0031308, c));
}

// Axis-fit channel weights. Color leans on green the way the eye does (R2 G4
// B1 A2), while LINEAR_DATA weighs channels evenly since each one is its own
// signal. Encoders that use these declare LINEAR_DATA before including.
#define PW_R     (LINEAR_DATA ? 1.0 : 2.0)
#define PW_G     (LINEAR_DATA ? 1.0 : 4.0)
#define PW_A     (LINEAR_DATA ? 1.0 : 2.0)
#define PW_R_INV (LINEAR_DATA ? 1.0 : 0.5)
#define PW_G_INV (LINEAR_DATA ? 1.0 : 0.25)
#define PW_A_INV (LINEAR_DATA ? 1.0 : 0.5)

// One dispatch encodes every mip and layer. Each mip owns a run of 8x8 block
// groups in a flat group index, and the dispatch's z picks the layer.
uint  base_width;
uint  base_height;
uint  mip_count;
uint  groups_x;           // Dispatch width, the flat group index wraps on it
uint4 mip_group_start[4]; // First flat group of each mip, up to 16 mips. uint4 packs tight, a uint[16] pads to 16 bytes per mip
uint4 mip_block_start[4]; // First output block of each mip, counting every layer

struct texenc_block_t {
	uint x;
	uint y;
	uint layer;
	uint mip;
	uint width;     // The mip's size in pixels
	uint height;
	uint out_index;
};

// Group starts only grow, so counting the ones at or below flat finds its mip.
uint texenc_mip(uint flat) {
	uint mip = 0;
	[unroll] for (uint m = 1; m < 16; m++)
		mip += (m < mip_count && mip_group_start[m >> 2][m & 3] <= flat) ? 1u : 0u;
	return mip;
}

bool texenc_block(uint3 group, uint3 thread, uint block_w, uint block_h, out texenc_block_t b) {
	uint flat = group.y * groups_x + group.x;
	b.mip = texenc_mip(flat);
	b.width  = max(base_width  >> b.mip, 1u);
	b.height = max(base_height >> b.mip, 1u);

	uint blocks_x = (b.width  + block_w - 1u) / block_w;
	uint blocks_y = (b.height + block_h - 1u) / block_h;
	uint groups_w = (blocks_x + 7u) / 8u;
	uint groups_h = (blocks_y + 7u) / 8u;
	uint local    = flat - mip_group_start[b.mip >> 2][b.mip & 3];
	b.x       = (local % groups_w) * 8u + thread.x;
	b.y       = (local / groups_w) * 8u + thread.y;
	b.layer   = group.z;
	b.out_index = mip_block_start[b.mip >> 2][b.mip & 3] + (b.layer * blocks_y + b.y) * blocks_x + b.x;
	return local < groups_w * groups_h && b.x < blocks_x && b.y < blocks_y;
}
