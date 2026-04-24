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
SamplerState              source_tex_s  : register(s0);
RWStructuredBuffer<uint4> output_blocks : register(u1);

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

#include "astc_common.hlsli"

// Trit-BISE weight unquant tables, indexed by the ENCODED v value (not level
// index). For trit+Bbit encoding the decoder's unquantize(v) is not monotonic
// in v — it produces pairs that mirror across 0.5, which is why LS-refine
// endpoint swap can use `v ^ 1` as the weight mirror.

// trit+2bit (12 lvl): v → unquantized weight in [0, 1]
// v: 0,      1,  2,     3,     4,    5,     6,     7,     8,     9,     10,    11
// w: 0/63, 63/63, 18/63, 45/63, 5/63, 58/63, 24/63, 39/63, 11/63, 52/63, 30/63, 33/63
static const float UNQ_R12_V[12] = {
	 0.0/63.0,  1.0,       18.0/63.0, 45.0/63.0,
	 5.0/63.0, 58.0/63.0,  24.0/63.0, 39.0/63.0,
	11.0/63.0, 52.0/63.0,  30.0/63.0, 33.0/63.0,
};

// trit+1bit (6 lvl): v → unquantized weight in [0, 1]
// v: 0, 1,      2,     3,     4,     5
// w: 0, 63/63, 12/63, 51/63, 25/63, 38/63
static const float UNQ_R6_V[6] = {
	 0.0/63.0,  1.0,       12.0/63.0, 51.0/63.0,
	25.0/63.0, 38.0/63.0,
};

///////////////////////////////////////////////////////////////////////////////
// Mode A: CEM 8 (RGB-only), 4x4 grid, trit+2bit weights (12 levels)
///////////////////////////////////////////////////////////////////////////////

uint4 encode_mode_4x4_rgb(in float3 pixels[16], in uint3 imin, in uint3 imax) {
	uint3 e0 = imin;
	uint3 e1 = imax;

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
			if (n0.r + n0.g + n0.b > n1.r + n1.g + n1.b) {
				uint3 tmp = n0; n0 = n1; n1 = tmp;
				// Mirror weight v across 0.5. Trit encoding pairs (0,1),
				// (2,3), (4,5), … always have `v ^ 1` as the "opposite"
				// unquant level — simpler than the LEVEL→V round-trip.
				[unroll] for (uint i = 0; i < 16; i++) weights[i] = weights[i] ^ 1u;
			}
			e0 = n0; e1 = n1;
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_4x4_rgb_r12(block);
	astc_write_endpoints_rgb8(block, e0, e1);

	// 16 weights = 3 full 5-trit groups + 1 partial-1 group.
	uint base = 0u;
	[unroll] for (uint g = 0; g < 3; g++) {
		uint v0 = weights[g * 5u + 0u];
		uint v1 = weights[g * 5u + 1u];
		uint v2 = weights[g * 5u + 2u];
		uint v3 = weights[g * 5u + 3u];
		uint v4 = weights[g * 5u + 4u];
		uint t0 = v0 >> 2, m0 = v0 & 3u;
		uint t1 = v1 >> 2, m1 = v1 & 3u;
		uint t2 = v2 >> 2, m2 = v2 & 3u;
		uint t3 = v3 >> 2, m3 = v3 & 3u;
		uint t4 = v4 >> 2, m4 = v4 & 3u;
		uint T  = astc_trit_pack_lut[t0 + 3u*t1 + 9u*t2 + 27u*t3 + 81u*t4];
		astc_write_weight_trit_2bit_full(block, base, T, m0, m1, m2, m3, m4);
		base += 18u;
	}
	// Partial-1 for weight 15.
	uint v15 = weights[15];
	uint T_p = astc_trit_pack_lut[v15 >> 2];
	astc_write_weight_trit_2bit_partial1(block, base, T_p, v15 & 3u);

	return block;
}

///////////////////////////////////////////////////////////////////////////////
// Mode B: CEM 12 (RGBA), 4x4 grid, trit+1bit weights (6 levels)
///////////////////////////////////////////////////////////////////////////////

uint4 encode_mode_4x4_rgba(in float4 pixels[16], in uint3 imin, in uint3 imax, in uint amin, in uint amax) {
	uint3 e0 = imin;
	uint3 e1 = imax;
	uint  a0 = amin;
	uint  a1 = amax;

	float3 faxis = float3(
		float(int(e1.r) - int(e0.r)) * 2.0,
		float(int(e1.g) - int(e0.g)) * 4.0,
		float(int(e1.b) - int(e0.b)));
	float faxis_len_sq = (faxis.r * faxis.r * 0.5 + faxis.g * faxis.g * 0.25 + faxis.b * faxis.b) / 255.0;
	float fc0_proj     = dot(float3(e0) / 255.0, faxis);

	// Quantize to trit+1bit weight (6 levels); weights[] holds encoded v.
	uint weights[16];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 16; i++) weights[i] = 0;
	} else {
		float inv_axis_len_sq = 1.0 / faxis_len_sq;
		[unroll] for (uint i = 0; i < 16; i++) {
			float p      = dot(pixels[i].rgb, faxis);
			float target = saturate((p - fc0_proj) * inv_axis_len_sq);
			weights[i]   = astc_quantize_weight_trit_1bit(target);
		}
	}

	// LS refine RGB endpoints (alpha stays at bbox; CEM 12 single-plane
	// couples alpha to RGB axis, no separate solve needed).
	if (faxis_len_sq >= 1e-6) {
		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0);
		float3 E = float3(0, 0, 0);
		[unroll] for (uint i = 0; i < 16; i++) {
			float  w  = UNQ_R6_V[weights[i]];
			float3 p  = pixels[i].rgb;
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
			if (n0.r + n0.g + n0.b > n1.r + n1.g + n1.b) {
				uint3 tmp = n0; n0 = n1; n1 = tmp;
				uint  at  = a0; a0 = a1; a1 = at;
				// Mirror weight v across 0.5 via v ^ 1 (same trit-pair
				// symmetry as Mode A).
				[unroll] for (uint i = 0; i < 16; i++) weights[i] = weights[i] ^ 1u;
			}
			e0 = n0; e1 = n1;
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_4x4_rgba_r6(block);
	astc_write_endpoints_rgba8(block, e0, e1, a0, a1);

	// 16 weights = 3 full 5-trit groups + 1 partial-1 group. B=1 so each m
	// is a single bit.
	uint base = 0u;
	[unroll] for (uint g = 0; g < 3; g++) {
		uint v0 = weights[g * 5u + 0u];
		uint v1 = weights[g * 5u + 1u];
		uint v2 = weights[g * 5u + 2u];
		uint v3 = weights[g * 5u + 3u];
		uint v4 = weights[g * 5u + 4u];
		uint t0 = v0 >> 1, m0 = v0 & 1u;
		uint t1 = v1 >> 1, m1 = v1 & 1u;
		uint t2 = v2 >> 1, m2 = v2 & 1u;
		uint t3 = v3 >> 1, m3 = v3 & 1u;
		uint t4 = v4 >> 1, m4 = v4 & 1u;
		uint T  = astc_trit_pack_lut[t0 + 3u*t1 + 9u*t2 + 27u*t3 + 81u*t4];
		astc_write_weight_trit_1bit_full(block, base, T, m0, m1, m2, m3, m4);
		base += 13u;
	}
	uint v15 = weights[15];
	uint T_p = astc_trit_pack_lut[v15 >> 1];
	astc_write_weight_trit_1bit_partial1(block, base, T_p, v15 & 1u);

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

	// Load 4x4 block of RGBA pixels, compute RGB + alpha bboxes.
	float4 pixels[16];
	float3 cmin = float3(1, 1, 1);
	float3 cmax = float3(0, 0, 0);
	float  amin = 1.0;
	float  amax = 0.0;
	[unroll] for (uint py = 0; py < 4; py++) {
		[unroll] for (uint px = 0; px < 4; px++) {
			uint sx = min(bx * 4u + px, image_width  - 1u);
			uint sy = min(by * 4u + py, image_height - 1u);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			pixels[py * 4u + px] = texel;
			cmin = min(cmin, texel.rgb);
			cmax = max(cmax, texel.rgb);
			amin = min(amin, texel.a);
			amax = max(amax, texel.a);
		}
	}

	// Quantize RGB endpoints to [0,255], inset by 1/16 (BC1-style outlier
	// trim). Alpha keeps raw bbox.
	uint3 imin = uint3(cmin * 255.0 + 0.5);
	uint3 imax = uint3(cmax * 255.0 + 0.5);
	uint3 range = imax - imin;
	imin += range >> 4;
	imax -= range >> 4;
	uint a0 = uint(amin * 255.0 + 0.5);
	uint a1 = uint(amax * 255.0 + 0.5);

	// Per-block selector: opaque blocks take the higher-precision CEM 8
	// path; anything else (uniform translucent or varying alpha) takes
	// CEM 12 to actually encode the alpha values.
	uint4 block;
	if (a0 == 255u && a1 == 255u) {
		float3 rgb_pixels[16];
		[unroll] for (uint i = 0; i < 16; i++) rgb_pixels[i] = pixels[i].rgb;
		block = encode_mode_4x4_rgb(rgb_pixels, imin, imax);
	} else {
		block = encode_mode_4x4_rgba(pixels, imin, imax, a0, a1);
	}

	output_blocks[buffer_offset + by * blocks_x + bx] = block;
}
