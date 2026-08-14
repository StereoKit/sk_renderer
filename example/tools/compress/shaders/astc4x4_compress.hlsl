//--name = astc4x4_compress

// ASTC 4x4 GPU texture compression — RGB + RGBA per-block selector.
//
// Each thread compresses one 4x4 block. Two encoder paths, picked per-block
// from the alpha bbox:
//
//   Mode A — CEM 8 (RGB-only), 4x4 grid, trit+2bit BISE weights (12 levels),
//            range-256 endpoints (block mode 0x251). Per-pixel grid with
//            12-level axial precision — 16 × 12 = 192 effective steps, up
//            from 128 on the previous R3 config. Used for opaque blocks.
//
//   Mode B — CEM 12 (RGBA), 4x4 grid, trit+1bit BISE weights (6 levels),
//            range-256 endpoints (block mode 0x043). Per-pixel grid with
//            6-level axial precision — 16 × 6 = 96 effective steps, up
//            from 64 on the previous R2 config. Used for non-opaque blocks.
//
// Selection is purely from the alpha bbox: opaque → Mode A, otherwise Mode B.
// (Same decision as before — the axial upgrade doesn't change this calculus.)
//
// Weight packing uses the BISE trit group helpers. 16 weights = 3 full
// 5-trit groups + 1 partial-1 group (18×3 + 4 = 58 bits for B=2; 13×3 + 3
// = 42 bits for B=1).

Texture2D<float4>         source_tex    : register(t0);
RWStructuredBuffer<uint4> output_blocks : register(u1);

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

#include "astc_common.hlsli"

// Trit-BISE weight unquant table, indexed by the ENCODED v value (not level
// index). For trit+Bbit encoding the decoder's unquantize(v) is not monotonic
// in v — it produces pairs that mirror across 0.5, which is why LS-refine
// endpoint swap can use `v ^ 1` as the weight mirror. (The trit+2bit table,
// UNQ_R12_V, lives in astc_common.hlsli.)

// trit+1bit (6 lvl): v → unquantized weight in [0, 1]. Raw values
// (0,63,12,51,25,38) with the decoder's "> 32 → +1" bump, over 64.
// v: 0, 1,      2,     3,     4,     5
// w: 0, 64/64, 12/64, 52/64, 25/64, 39/64
static const float UNQ_R6_V[6] = {
	 0.0/64.0,  1.0,       12.0/64.0, 52.0/64.0,
	25.0/64.0, 39.0/64.0,
};

///////////////////////////////////////////////////////////////////////////////
// Mode A: CEM 8 (RGB-only), 4x4 grid, trit+2bit weights (12 levels)
///////////////////////////////////////////////////////////////////////////////

uint4 encode_mode_4x4_rgb(in float3 pixels[16]) {
	// FPS seed: 2 passes of "find pixel most distant from current anchor,
	// anchor = that pixel" — converges to the block's diametric pixel pair,
	// two real colors defining the principal axis. Bbox corners are fictional
	// colors on anti-correlated content (e.g. red/green foliage) and collapse
	// the projection there.
	uint3 fps_a = uint3(pixels[0] * 255.0 + 0.5);
	uint3 fps_b = fps_a;
	[unroll] for (uint iter = 0; iter < 2u; iter++) {
		float max_d2 = -1.0;
		uint3 far_p  = fps_a;
		[unroll] for (uint fi = 0; fi < 16; fi++) {
			uint3 pi   = uint3(pixels[fi] * 255.0 + 0.5);
			int3  diff = int3(pi) - int3(fps_a);
			float d2   = dot(float3(diff), float3(diff));
			far_p  = d2 > max_d2 ? pi : far_p;
			max_d2 = max(max_d2, d2);
		}
		fps_b = fps_a;
		fps_a = far_p;
	}
	uint3 e0 = fps_b;
	uint3 e1 = fps_a;
	if (e0.r + e0.g + e0.b > e1.r + e1.g + e1.b) {
		uint3 tmp = e0; e0 = e1; e1 = tmp;
	}

	float3 faxis = float3(
		float(int(e1.r) - int(e0.r)) * 2.0,
		float(int(e1.g) - int(e0.g)) * 4.0,
		float(int(e1.b) - int(e0.b)));
	float faxis_len_sq = (faxis.r * faxis.r * 0.5 + faxis.g * faxis.g * 0.25 + faxis.b * faxis.b) / 255.0;
	float fc0_proj     = dot(float3(e0) / 255.0, faxis);

	// Quantize pixel projections to trit+2bit weight (12 levels). weights[]
	// stores the ENCODED v directly (not level index), so packing skips any
	// level→v remapping.
	uint weights[16];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 16; i++) weights[i] = 0;
	} else {
		float inv_axis_len_sq = 1.0 / faxis_len_sq;
		[unroll] for (uint i = 0; i < 16; i++) {
			float p       = dot(pixels[i], faxis);
			float target  = saturate((p - fc0_proj) * inv_axis_len_sq);
			weights[i]    = astc_quantize_weight_trit_2bit(target);
		}
	}

	// LS refine (same 2x2 normal-equation solve as before, indexed through
	// UNQ_R12_V since weights are encoded-v values).
	if (faxis_len_sq >= 1e-6) {
		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0);
		float3 E = float3(0, 0, 0);
		[unroll] for (uint i = 0; i < 16; i++) {
			float  w  = UNQ_R12_V[weights[i]];
			float3 p  = pixels[i];
			float  iw = 1.0 - w;
			A += iw * iw; B +=  w * iw; C +=  w *  w;
			D += iw * p;  E +=  w * p;
		}

		float det = A * C - B * B;
		if (det > 1e-6 && det > 0.01 * A * C) {
			float  inv_det = 1.0 / det;
			float3 e0_ref  = saturate((C * D - B * E) * inv_det);
			float3 e1_ref  = saturate((A * E - B * D) * inv_det);
			uint3 n0 = uint3(e0_ref * 255.0 + 0.5);
			uint3 n1 = uint3(e1_ref * 255.0 + 0.5);
			// No weight mirror needed on swap — weights are re-derived from
			// the refined endpoints below, in whichever order they land.
			if (n0.r + n0.g + n0.b > n1.r + n1.g + n1.b) {
				uint3 tmp = n0; n0 = n1; n1 = tmp;
			}
			e0 = n0; e1 = n1;

			// Re-derive weights against the refined endpoints. Range-256
			// endpoints are exact 8-bit values, so this axis is exactly the
			// decoder's — closing the fit-against-stale-axis gap.
			faxis = float3(
				float(int(e1.r) - int(e0.r)) * 2.0,
				float(int(e1.g) - int(e0.g)) * 4.0,
				float(int(e1.b) - int(e0.b)));
			faxis_len_sq = (faxis.r * faxis.r * 0.5 + faxis.g * faxis.g * 0.25 + faxis.b * faxis.b) / 255.0;
			if (faxis_len_sq >= 1e-6) {
				fc0_proj = dot(float3(e0) / 255.0, faxis);
				float inv_len = 1.0 / faxis_len_sq;
				[unroll] for (uint i = 0; i < 16; i++) {
					float p      = dot(pixels[i], faxis);
					float target = saturate((p - fc0_proj) * inv_len);
					weights[i]   = astc_quantize_weight_trit_2bit(target);
				}
			}
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_4x4_rgb_r12(block);
	astc_write_endpoints_rgb8(block, e0, e1);

	// 16 weights = 3 full 5-trit groups + 1 partial-1 group.
	astc_write_weights16_trit_2bit(block, weights);

	return block;
}

///////////////////////////////////////////////////////////////////////////////
// Mode B: CEM 12 (RGBA), 4x4 grid, trit+1bit weights (6 levels)
///////////////////////////////////////////////////////////////////////////////

uint4 encode_mode_4x4_rgba(in float4 pixels[16]) {
	// 4D FPS seed — same diametric-pair search as Mode A, in RGBA space so
	// alpha extremes participate in the seeding too.
	uint4 fps_a = uint4(pixels[0] * 255.0 + 0.5);
	uint4 fps_b = fps_a;
	[unroll] for (uint iter = 0; iter < 2u; iter++) {
		float max_d2 = -1.0;
		uint4 far_p  = fps_a;
		[unroll] for (uint fi = 0; fi < 16; fi++) {
			uint4 pi   = uint4(pixels[fi] * 255.0 + 0.5);
			int4  diff = int4(pi) - int4(fps_a);
			float d2   = dot(float4(diff), float4(diff));
			far_p  = d2 > max_d2 ? pi : far_p;
			max_d2 = max(max_d2, d2);
		}
		fps_b = fps_a;
		fps_a = far_p;
	}
	uint3 e0 = fps_b.rgb;
	uint3 e1 = fps_a.rgb;
	uint  a0 = fps_b.a;
	uint  a1 = fps_a.a;
	if (e0.r + e0.g + e0.b > e1.r + e1.g + e1.b) {
		uint3 tmp = e0; e0 = e1; e1 = tmp;
		uint  at  = a0; a0 = a1; a1 = at;
	}

	// 4D perceptual axis. CEM 12 single-plane couples alpha to the same
	// per-pixel weight as RGB, so alpha must participate in the fit — with an
	// RGB-only axis, a block of flat color under an alpha gradient projects
	// every pixel to the same weight and the gradient collapses to constant
	// alpha. Alpha gets red's perceptual weight (2).
	float4 faxis = float4(
		float(int(e1.r) - int(e0.r)) * 2.0,
		float(int(e1.g) - int(e0.g)) * 4.0,
		float(int(e1.b) - int(e0.b)),
		float(int(a1)   - int(a0))   * 2.0);
	float faxis_len_sq = (faxis.r * faxis.r * 0.5 + faxis.g * faxis.g * 0.25 + faxis.b * faxis.b + faxis.a * faxis.a * 0.5) / 255.0;
	float fc0_proj     = dot(float4(float3(e0), float(a0)) / 255.0, faxis);

	// Quantize to trit+1bit weight (6 levels); weights[] holds encoded v.
	uint weights[16];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 16; i++) weights[i] = 0;
	} else {
		float inv_axis_len_sq = 1.0 / faxis_len_sq;
		[unroll] for (uint i = 0; i < 16; i++) {
			float p      = dot(pixels[i], faxis);
			float target = saturate((p - fc0_proj) * inv_axis_len_sq);
			weights[i]   = astc_quantize_weight_trit_1bit(target);
		}
	}

	// LS refine all four endpoint channels against the shared weight.
	if (faxis_len_sq >= 1e-6) {
		float  A = 0, B = 0, C = 0;
		float4 D = float4(0, 0, 0, 0);
		float4 E = float4(0, 0, 0, 0);
		[unroll] for (uint i = 0; i < 16; i++) {
			float  w  = UNQ_R6_V[weights[i]];
			float4 p  = pixels[i];
			float  iw = 1.0 - w;
			A += iw * iw; B +=  w * iw; C +=  w *  w;
			D += iw * p;  E +=  w * p;
		}

		float det = A * C - B * B;
		if (det > 1e-6 && det > 0.01 * A * C) {
			float  inv_det = 1.0 / det;
			float4 e0_ref  = saturate((C * D - B * E) * inv_det);
			float4 e1_ref  = saturate((A * E - B * D) * inv_det);
			uint3 n0 = uint3(e0_ref.rgb * 255.0 + 0.5);
			uint3 n1 = uint3(e1_ref.rgb * 255.0 + 0.5);
			uint  m0 = uint(e0_ref.a * 255.0 + 0.5);
			uint  m1 = uint(e1_ref.a * 255.0 + 0.5);
			// Decoder's swap test is on RGB sums only; alpha rides along.
			// No weight mirror — weights are re-derived below.
			if (n0.r + n0.g + n0.b > n1.r + n1.g + n1.b) {
				uint3 tmp = n0; n0 = n1; n1 = tmp;
				uint  at  = m0; m0 = m1; m1 = at;
			}
			e0 = n0; e1 = n1;
			a0 = m0; a1 = m1;

			// Re-derive weights against the refined endpoints (exact 8-bit,
			// so decoder-exact).
			faxis = float4(
				float(int(e1.r) - int(e0.r)) * 2.0,
				float(int(e1.g) - int(e0.g)) * 4.0,
				float(int(e1.b) - int(e0.b)),
				float(int(a1)   - int(a0))   * 2.0);
			faxis_len_sq = (faxis.r * faxis.r * 0.5 + faxis.g * faxis.g * 0.25 + faxis.b * faxis.b + faxis.a * faxis.a * 0.5) / 255.0;
			if (faxis_len_sq >= 1e-6) {
				fc0_proj = dot(float4(float3(e0), float(a0)) / 255.0, faxis);
				float inv_len = 1.0 / faxis_len_sq;
				[unroll] for (uint i = 0; i < 16; i++) {
					float p      = dot(pixels[i], faxis);
					float target = saturate((p - fc0_proj) * inv_len);
					weights[i]   = astc_quantize_weight_trit_1bit(target);
				}
			}
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_4x4_rgba_r6(block);
	astc_write_endpoints_rgba8(block, e0, e1, a0, a1);

	// 16 weights = 3 full 5-trit groups + 1 partial-1 group (B=1, single-bit m).
	astc_write_weights16_trit_1bit(block, weights);

	return block;
}

///////////////////////////////////////////////////////////////////////////////
// Main entry point
///////////////////////////////////////////////////////////////////////////////

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint bx = id.x;
	uint by = id.y;
	uint blocks_y = (image_height + 3u) / 4u;
	if (bx >= blocks_x || by >= blocks_y)
		return;

	// Load 4x4 block of RGBA pixels; alpha bbox drives mode selection.
	float4 pixels[16];
	float  amin = 1.0;
	float  amax = 0.0;
	[unroll] for (uint py = 0; py < 4; py++) {
		[unroll] for (uint px = 0; px < 4; px++) {
			uint sx = min(bx * 4u + px, image_width  - 1u);
			uint sy = min(by * 4u + py, image_height - 1u);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			pixels[py * 4u + px] = texel;
			amin = min(amin, texel.a);
			amax = max(amax, texel.a);
		}
	}

	// Alpha bbox drives mode selection; endpoint seeding itself is FPS-based
	// inside each encoder, so the RGB bbox is no longer needed here.
	uint a0 = uint(amin * 255.0 + 0.5);
	uint a1 = uint(amax * 255.0 + 0.5);

	// Per-block selector: opaque blocks take the higher-precision CEM 8
	// path; anything else (uniform translucent or varying alpha) takes
	// CEM 12 to actually encode the alpha values.
	uint4 block;
	if (a0 == 255u && a1 == 255u) {
		float3 rgb_pixels[16];
		[unroll] for (uint i = 0; i < 16; i++) rgb_pixels[i] = pixels[i].rgb;
		block = encode_mode_4x4_rgb(rgb_pixels);
	} else {
		block = encode_mode_4x4_rgba(pixels);
	}

	output_blocks[buffer_offset + by * blocks_x + bx] = block;
}
