//--name = msaa_resolve_wide
//
// Wide-kernel MSAA resolve with Mitchell-Netravali filter.
// Samples all 4 from the center pixel, plus the 2 nearest samples from each
// edge neighbor (cross pattern). Skips diagonals — they contribute <2% weight.
// 12 samples total vs 36 for full 3x3, nearly identical quality.
// PC only — requires MSAA target readable as Texture2DMS (not tile-local).

Texture2DMS<float4, 4> msaa_color : register(t0);

// Precomputed Mitchell-Netravali (B=C=1/3) weights for standard 4x MSAA positions.
// mitchell_2d(offset) = mitchell_1d(offset.x) * mitchell_1d(offset.y)
//
// Center pixel samples:
//   s0 (-0.125, -0.375): 0.8528 * 0.6471 = 0.5518
//   s1 ( 0.375, -0.125): 0.6471 * 0.8528 = 0.5518
//   s2 (-0.375,  0.125): 0.6471 * 0.8528 = 0.5518
//   s3 ( 0.125,  0.375): 0.8528 * 0.6471 = 0.5518
//
// Right (+1,0) nearest:
//   s2 ( 0.625,  0.125): 0.2630 * 0.8528 = 0.2243
//   s0 ( 0.875, -0.375): 0.0645 * 0.6471 = 0.0417
//
// Left (-1,0) nearest:
//   s1 (-0.625, -0.125): 0.2630 * 0.8528 = 0.2243
//   s3 (-0.875,  0.375): 0.0645 * 0.6471 = 0.0417
//
// Down (0,+1) nearest:
//   s0 (-0.125,  0.625): 0.8528 * 0.2630 = 0.2243
//   s1 ( 0.375,  0.875): 0.6471 * 0.0645 = 0.0417
//
// Up (0,-1) nearest:
//   s3 ( 0.125, -0.625): 0.8528 * 0.2630 = 0.2243
//   s2 (-0.375, -0.875): 0.6471 * 0.0645 = 0.0417

static const float W_CENTER = 0.5518;  // All 4 center samples have same weight
static const float W_NEAR   = 0.2243;  // Nearest neighbor sample
static const float W_FAR    = 0.0417;  // Second-nearest neighbor sample

// Total weight = 4*0.5518 + 4*0.2243 + 4*0.0417 = 3.2712
static const float W_TOTAL_INV = 1.0 / (4.0 * W_CENTER + 4.0 * W_NEAR + 4.0 * W_FAR);

struct psIn {
	float4 pos : SV_POSITION;
};

psIn vs(uint id : SV_VertexID) {
	psIn o;
	float2 uv = float2(id & 2, (id << 1) & 2);
	o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	int2 p = int2(input.pos.xy);

	// Center pixel: all 4 samples (equal weight)
	float4 accum = (msaa_color.Load(p, 0)
	              + msaa_color.Load(p, 1)
	              + msaa_color.Load(p, 2)
	              + msaa_color.Load(p, 3)) * W_CENTER;

	// Right (+1,0): samples 2 (near) and 0 (far)
	int2 r = p + int2(1, 0);
	accum += msaa_color.Load(r, 2) * W_NEAR + msaa_color.Load(r, 0) * W_FAR;

	// Left (-1,0): samples 1 (near) and 3 (far)
	int2 l = p - int2(1, 0);
	accum += msaa_color.Load(l, 1) * W_NEAR + msaa_color.Load(l, 3) * W_FAR;

	// Down (0,+1): samples 0 (near) and 1 (far)
	int2 d = p + int2(0, 1);
	accum += msaa_color.Load(d, 0) * W_NEAR + msaa_color.Load(d, 1) * W_FAR;

	// Up (0,-1): samples 3 (near) and 2 (far)
	int2 u = p - int2(0, 1);
	accum += msaa_color.Load(u, 3) * W_NEAR + msaa_color.Load(u, 2) * W_FAR;

	return accum * W_TOTAL_INV;
}
