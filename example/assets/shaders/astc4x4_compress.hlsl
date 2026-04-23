//--name = astc4x4_compress

// ASTC 4x4 GPU texture compression — RGB + RGBA per-block selector.
//
// Each thread compresses one 4x4 block. Two encoder paths, picked per-block
// from the alpha bbox:
//
//   Mode A — CEM 8 (RGB-only), 4x4 grid, 3-bit weights, 8-bit endpoints
//            (block mode 0x053). 8 weight levels, 256-level color, no alpha
//            in the encoded block (decoder writes alpha = 1.0). Used when
//            the source block is opaque (a0 == a1 == 255).
//
//   Mode B — CEM 12 (RGBA), 4x4 grid, 2-bit weights, 8-bit endpoints
//            (block mode 0x042). 4 weight levels, 256-level color including
//            alpha. Used for any non-opaque block. Pure binary throughout —
//            no BISE, no LUTs needed.
//
// Selection is purely from the alpha bbox: opaque → Mode A, otherwise Mode B.
// We don't bother with an SSE comparison because the outcome is decided:
// * Opaque blocks pay zero alpha cost in Mode A and have 2× the weight
//   precision; Mode B can't possibly win.
// * Non-opaque blocks pay a large constant alpha penalty in Mode A
//   (sum((1-α)²) per pixel) that Mode B trivially beats.

Texture2D<float4>         source_tex    : register(t0);
SamplerState              source_tex_s  : register(s0);
RWStructuredBuffer<uint4> output_blocks : register(u1);

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

#include "astc_common.hlsli"

static const float UNQ_R8[8] = {
	0.0,         9.0 / 64.0, 18.0 / 64.0, 27.0 / 64.0,
	37.0 / 64.0, 46.0 / 64.0, 55.0 / 64.0, 1.0
};
static const float UNQ_R4[4] = { 0.0, 21.0 / 64.0, 43.0 / 64.0, 1.0 };

///////////////////////////////////////////////////////////////////////////////
// Mode A: CEM 8 (RGB-only), 4x4 grid, 3-bit weights
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

	float T[7];
	T[0] = faxis_len_sq * 0.0703125 + fc0_proj;
	T[1] = faxis_len_sq * 0.2109375 + fc0_proj;
	T[2] = faxis_len_sq * 0.3515625 + fc0_proj;
	T[3] = faxis_len_sq * 0.5000000 + fc0_proj;
	T[4] = faxis_len_sq * 0.6484375 + fc0_proj;
	T[5] = faxis_len_sq * 0.7890625 + fc0_proj;
	T[6] = faxis_len_sq * 0.9296875 + fc0_proj;

	uint weights[16];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 16; i++) weights[i] = 0;
	} else {
		[unroll] for (uint i = 0; i < 16; i++) {
			float p = dot(pixels[i], faxis);
			uint  q = uint(p >= T[0]) + uint(p >= T[1]) + uint(p >= T[2])
			        + uint(p >= T[3]) + uint(p >= T[4]) + uint(p >= T[5])
			        + uint(p >= T[6]);
			weights[i] = q;
		}
	}

	if (faxis_len_sq >= 1e-6) {
		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0);
		float3 E = float3(0, 0, 0);
		[unroll] for (uint i = 0; i < 16; i++) {
			float  w  = UNQ_R8[weights[i]];
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
				[unroll] for (uint i = 0; i < 16; i++) weights[i] = 7u - weights[i];
			}
			e0 = n0; e1 = n1;
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_4x4_rgb(block);
	astc_write_endpoints_rgb8(block, e0, e1);
	[unroll] for (uint wi = 0; wi < 16; wi++) {
		astc_write_weight_3bit(block, wi, weights[wi]);
	}
	return block;
}

///////////////////////////////////////////////////////////////////////////////
// Mode B: CEM 12 (RGBA), 4x4 grid, 2-bit weights
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

	// 2-bit weight thresholds (4 levels) — same as the 6x6 5x5/dual-plane path.
	float T[3];
	T[0] = faxis_len_sq * 0.1640625 + fc0_proj;
	T[1] = faxis_len_sq * 0.5000000 + fc0_proj;
	T[2] = faxis_len_sq * 0.8359375 + fc0_proj;

	uint weights[16];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 16; i++) weights[i] = 0;
	} else {
		[unroll] for (uint i = 0; i < 16; i++) {
			float p = dot(pixels[i].rgb, faxis);
			uint  q = uint(p >= T[0]) + uint(p >= T[1]) + uint(p >= T[2]);
			weights[i] = q;
		}
	}

	// LS refine RGB endpoints (alpha stays at bbox; CEM 12 single-plane
	// couples alpha to RGB axis, no separate solve needed).
	if (faxis_len_sq >= 1e-6) {
		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0);
		float3 E = float3(0, 0, 0);
		[unroll] for (uint i = 0; i < 16; i++) {
			float  w  = UNQ_R4[weights[i]];
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
				[unroll] for (uint i = 0; i < 16; i++) weights[i] = 3u - weights[i];
			}
			e0 = n0; e1 = n1;
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_4x4_rgba(block);
	astc_write_endpoints_rgba8(block, e0, e1, a0, a1);
	[unroll] for (uint wi = 0; wi < 16; wi++) {
		astc_write_weight_2bit(block, wi, weights[wi]);
	}
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
		// CEM 8 path doesn't need the alpha array — pass RGB only.
		float3 rgb_pixels[16];
		[unroll] for (uint i = 0; i < 16; i++) rgb_pixels[i] = pixels[i].rgb;
		block = encode_mode_4x4_rgb(rgb_pixels, imin, imax);
	} else {
		block = encode_mode_4x4_rgba(pixels, imin, imax, a0, a1);
	}

	output_blocks[buffer_offset + by * blocks_x + bx] = block;
}
