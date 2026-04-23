//--name = astc6x6_compress

// ASTC 6x6 GPU texture compression — multi-mode per-block selector.
//
// One shader handles all 6x6 ASTC encoding. ASTC at the Vulkan level has
// only one format per block size (always RGBA at decode, with the CEM
// determining whether alpha is encoded or hardcoded to 1.0), so a single
// encoder that picks the right CEM and grid per block is strictly more
// capable than separate "RGB" and "RGBA" encoders. The per-block alpha-
// pattern early-out (in main cs() below) makes the cost on opaque content
// match what a dedicated RGB-only encoder would pay.
//
// For each 6x6 block, encodes up to five candidate block configurations
// (two CEM 8 RGB modes + three CEM 12 RGBA modes) and picks
// whichever minimizes reconstruction SSE against the source RGBA pixels.
// SSE is measured in dequantized [0,1] RGBA space using each mode's actual
// decoder reconstruction. Per-block CEM/grid choice means a single texture
// can mix opaque and translucent blocks at no extra cost.
//
//   Mode A — CEM 8, 4x4 grid, 3-bit weights, 8-bit endpoints (0x053)
//            RGB-only. Decoder writes alpha=1.0 unconditionally, so blocks
//            whose source alpha is uniformly opaque win cleanly here:
//            zero alpha error, full 8-bit color precision, more bits free
//            because we don't waste them on alpha endpoints.
//
//   Mode B — CEM 8, 6x6 per-pixel grid, 2-bit weights, range-80 endpoints (0x108)
//            RGB-only at higher spatial precision but coarser color (80 levels).
//            Wins on opaque blocks with sharp color edges.
//
//   Mode C — CEM 12, 3x3 grid, 3-bit weights, 8-bit endpoints (0x1BF)
//            Full 256-level color precision; alpha couples to the RGB axis.
//            Wins on translucent blocks where alpha tracks color brightness.
//
//   Mode D — CEM 12, 5x5 grid, 2-bit weights, range-192 endpoints (0x0E2)
//            Denser weight grid for sharper transitions; endpoints drop to
//            192 levels via trit BISE. Wins when alpha mostly tracks RGB
//            but the spatial detail needs more weight samples.
//
//   Mode E — CEM 12 dual-plane, 3x3 grid, 2-bit weights, 8-bit endpoints (0x5AE)
//            Alpha gets its OWN per-pixel weight independent of RGB. Wins on
//            blocks where alpha doesn't correlate with color brightness —
//            sprite cutouts, font edges, alpha masks. CCS=3 selects alpha
//            as the secondary plane.

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
// Mode A: 3x3 weight grid, 3-bit weights, 8-bit RGBA endpoints
///////////////////////////////////////////////////////////////////////////////

void encode_mode_3x3_rgba(
	in  float4 pixels[36],
	in  float3 faxis,     in  float faxis_len_sq, in float fc0_proj,
	in  float  pproj[36], in  uint3 imin_rgb,     in uint3 imax_rgb,
	in  uint   amin,      in  uint  amax,
	out uint4  out_block, out float out_sse)
{
	uint3 e0 = imin_rgb;
	uint3 e1 = imax_rgb;
	uint  a0 = amin;
	uint  a1 = amax;

	float T[7];
	T[0] = faxis_len_sq * 0.0703125 + fc0_proj;
	T[1] = faxis_len_sq * 0.2109375 + fc0_proj;
	T[2] = faxis_len_sq * 0.3515625 + fc0_proj;
	T[3] = faxis_len_sq * 0.5000000 + fc0_proj;
	T[4] = faxis_len_sq * 0.6484375 + fc0_proj;
	T[5] = faxis_len_sq * 0.7890625 + fc0_proj;
	T[6] = faxis_len_sq * 0.9296875 + fc0_proj;

	// Sample 9 grid points via bilinear downsample of pproj (RGB axis only).
	static const float wcoord[3] = { 0.0, 2.5, 5.0 };
	uint weights[9];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 9; i++) weights[i] = 0;
	} else {
		[unroll] for (uint wy = 0; wy < 3; wy++) {
			[unroll] for (uint wx = 0; wx < 3; wx++) {
				float fx = wcoord[wx], fy = wcoord[wy];
				int   ix0 = int(fx),   iy0 = int(fy);
				int   ix1 = min(ix0 + 1, 5);
				int   iy1 = min(iy0 + 1, 5);
				float tx = fx - float(ix0), ty = fy - float(iy0);
				float p00 = pproj[iy0 * 6 + ix0];
				float p10 = pproj[iy0 * 6 + ix1];
				float p01 = pproj[iy1 * 6 + ix0];
				float p11 = pproj[iy1 * 6 + ix1];
				float p   = lerp(lerp(p00, p10, tx), lerp(p01, p11, tx), ty);
				uint  q = uint(p >= T[0]) + uint(p >= T[1]) + uint(p >= T[2])
				        + uint(p >= T[3]) + uint(p >= T[4]) + uint(p >= T[5])
				        + uint(p >= T[6]);
				weights[wy * 3 + wx] = q;
			}
		}
	}

	// LS refine RGB endpoints — alpha LS would require a separate axis since
	// CEM 12 single-plane shares one weight per pixel for all four channels.
	// We solve LS on the dominant variation (RGB axis) and let alpha endpoint
	// stay at bbox; alpha precision then comes from the 8-bit endpoint range.
	if (faxis_len_sq >= 1e-6) {
		float gw[9];
		[unroll] for (uint i = 0; i < 9; i++) gw[i] = UNQ_R8[weights[i]];

		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0), E = float3(0, 0, 0);
		[unroll] for (uint py = 0; py < 6; py++) {
			[unroll] for (uint px = 0; px < 6; px++) {
				float fx = ASTC_3X3_AXIS_POS[px];
				float fy = ASTC_3X3_AXIS_POS[py];
				int   ix = int(fx), iy = int(fy);
				int   ix1 = min(ix + 1, 2);
				int   iy1 = min(iy + 1, 2);
				float tx = fx - float(ix), ty = fy - float(iy);
				float w00 = gw[iy  * 3 + ix ];
				float w10 = gw[iy  * 3 + ix1];
				float w01 = gw[iy1 * 3 + ix ];
				float w11 = gw[iy1 * 3 + ix1];
				float w   = lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty);
				float3 p  = pixels[py * 6 + px].rgb;
				float  iw = 1.0 - w;
				A += iw*iw; B += w*iw; C += w*w;
				D += iw * p; E += w * p;
			}
		}

		float det = A * C - B * B;
		if (det > 1e-6 && det > 0.01 * A * C) {
			float  inv_det = 1.0 / det;
			float3 e0_ref  = saturate((C * D - B * E) * inv_det);
			float3 e1_ref  = saturate((A * E - B * D) * inv_det);
			uint3  n0      = uint3(e0_ref * 255.0 + 0.5);
			uint3  n1      = uint3(e1_ref * 255.0 + 0.5);
			if (n0.r + n0.g + n0.b > n1.r + n1.g + n1.b) {
				uint3 tmp = n0; n0 = n1; n1 = tmp;
				uint  at  = a0; a0 = a1; a1 = at;
				[unroll] for (uint i = 0; i < 9; i++) weights[i] = 7u - weights[i];
			}
			e0 = n0; e1 = n1;
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_3x3_rgba(block);
	astc_write_endpoints_rgba8(block, e0, e1, a0, a1);
	[unroll] for (uint wi = 0; wi < 9; wi++) {
		astc_write_weight_3bit(block, wi, weights[wi]);
	}
	out_block = block;

	// SSE — full RGBA reconstruction with bilinear weight upsampling.
	float4 e0_f = float4(float3(e0) / 255.0, float(a0) / 255.0);
	float4 e1_f = float4(float3(e1) / 255.0, float(a1) / 255.0);
	float  gw2[9];
	[unroll] for (uint i = 0; i < 9; i++) gw2[i] = UNQ_R8[weights[i]];
	float sse = 0.0;
	[unroll] for (uint py = 0; py < 6; py++) {
		[unroll] for (uint px = 0; px < 6; px++) {
			float fx = ASTC_3X3_AXIS_POS[px];
			float fy = ASTC_3X3_AXIS_POS[py];
			int   ix = int(fx), iy = int(fy);
			int   ix1 = min(ix + 1, 2);
			int   iy1 = min(iy + 1, 2);
			float tx = fx - float(ix), ty = fy - float(iy);
			float w00 = gw2[iy  * 3 + ix ];
			float w10 = gw2[iy  * 3 + ix1];
			float w01 = gw2[iy1 * 3 + ix ];
			float w11 = gw2[iy1 * 3 + ix1];
			float w   = lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty);
			float4 recon = lerp(e0_f, e1_f, w);
			float4 d     = pixels[py * 6 + px] - recon;
			sse += dot(d, d);
		}
	}
	out_sse = sse;
}

///////////////////////////////////////////////////////////////////////////////
// Mode B: 5x5 weight grid, 2-bit weights, range-192 RGBA endpoints (BISE trit)
///////////////////////////////////////////////////////////////////////////////

void encode_mode_5x5_rgba(
	in  float4 pixels[36],
	in  float3 faxis,     in  float faxis_len_sq, in float fc0_proj,
	in  float  pproj[36], in  uint3 imin_rgb,     in uint3 imax_rgb,
	in  uint   amin,      in  uint  amax,
	out uint4  out_block, out float out_sse)
{
	uint3 e0 = imin_rgb;
	uint3 e1 = imax_rgb;
	uint  a0 = amin;
	uint  a1 = amax;

	// 2-bit weights, 4 levels — same threshold table as the RGB 6x6pp variant.
	float T[3];
	T[0] = faxis_len_sq * 0.1640625 + fc0_proj;
	T[1] = faxis_len_sq * 0.5000000 + fc0_proj;
	T[2] = faxis_len_sq * 0.8359375 + fc0_proj;

	// 25 grid points via bilinear downsample of pproj. Coords (0, 1.25, 2.5,
	// 3.75, 5.0) per axis on a 6x6 block.
	static const float wcoord[5] = { 0.0, 1.25, 2.5, 3.75, 5.0 };
	uint weights[25];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 25; i++) weights[i] = 0;
	} else {
		[unroll] for (uint wy = 0; wy < 5; wy++) {
			[unroll] for (uint wx = 0; wx < 5; wx++) {
				float fx = wcoord[wx], fy = wcoord[wy];
				int   ix0 = int(fx),   iy0 = int(fy);
				int   ix1 = min(ix0 + 1, 5);
				int   iy1 = min(iy0 + 1, 5);
				float tx = fx - float(ix0), ty = fy - float(iy0);
				float p00 = pproj[iy0 * 6 + ix0];
				float p10 = pproj[iy0 * 6 + ix1];
				float p01 = pproj[iy1 * 6 + ix0];
				float p11 = pproj[iy1 * 6 + ix1];
				float p   = lerp(lerp(p00, p10, tx), lerp(p01, p11, tx), ty);
				uint  q = uint(p >= T[0]) + uint(p >= T[1]) + uint(p >= T[2]);
				weights[wy * 5 + wx] = q;
			}
		}
	}

	if (faxis_len_sq >= 1e-6) {
		float gw[25];
		[unroll] for (uint i = 0; i < 25; i++) gw[i] = UNQ_R4[weights[i]];

		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0), E = float3(0, 0, 0);
		[unroll] for (uint py = 0; py < 6; py++) {
			[unroll] for (uint px = 0; px < 6; px++) {
				float fx = ASTC_5X5_AXIS_POS[px];
				float fy = ASTC_5X5_AXIS_POS[py];
				int   ix = int(fx), iy = int(fy);
				int   ix1 = min(ix + 1, 4);
				int   iy1 = min(iy + 1, 4);
				float tx = fx - float(ix), ty = fy - float(iy);
				float w00 = gw[iy  * 5 + ix ];
				float w10 = gw[iy  * 5 + ix1];
				float w01 = gw[iy1 * 5 + ix ];
				float w11 = gw[iy1 * 5 + ix1];
				float w   = lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty);
				float3 p  = pixels[py * 6 + px].rgb;
				float  iw = 1.0 - w;
				A += iw*iw; B += w*iw; C += w*w;
				D += iw * p; E += w * p;
			}
		}

		float det = A * C - B * B;
		if (det > 1e-6 && det > 0.01 * A * C) {
			float  inv_det = 1.0 / det;
			float3 e0_ref  = saturate((C * D - B * E) * inv_det);
			float3 e1_ref  = saturate((A * E - B * D) * inv_det);
			uint3  n0      = uint3(e0_ref * 255.0 + 0.5);
			uint3  n1      = uint3(e1_ref * 255.0 + 0.5);
			if (n0.r + n0.g + n0.b > n1.r + n1.g + n1.b) {
				uint3 tmp = n0; n0 = n1; n1 = tmp;
				uint  at  = a0; a0 = a1; a1 = at;
				[unroll] for (uint i = 0; i < 25; i++) weights[i] = 3u - weights[i];
			}
			e0 = n0; e1 = n1;
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_5x5_rgba(block);
	astc_write_endpoints_rgba6trit(block, e0, e1, a0, a1);
	[unroll] for (uint wi = 0; wi < 25; wi++) {
		astc_write_weight_2bit(block, wi, weights[wi]);
	}
	out_block = block;

	// SSE with range-192 dequantized endpoints (mirrors the decoder).
	uint v_r0 = astc_r192_quant_lut[min(e0.r, 255u)];
	uint v_r1 = astc_r192_quant_lut[min(e1.r, 255u)];
	uint v_g0 = astc_r192_quant_lut[min(e0.g, 255u)];
	uint v_g1 = astc_r192_quant_lut[min(e1.g, 255u)];
	uint v_b0 = astc_r192_quant_lut[min(e0.b, 255u)];
	uint v_b1 = astc_r192_quant_lut[min(e1.b, 255u)];
	uint v_a0 = astc_r192_quant_lut[min(a0,   255u)];
	uint v_a1 = astc_r192_quant_lut[min(a1,   255u)];
	float4 e0_f = float4(
		astc_dequant_r192(v_r0),
		astc_dequant_r192(v_g0),
		astc_dequant_r192(v_b0),
		astc_dequant_r192(v_a0)) / 255.0;
	float4 e1_f = float4(
		astc_dequant_r192(v_r1),
		astc_dequant_r192(v_g1),
		astc_dequant_r192(v_b1),
		astc_dequant_r192(v_a1)) / 255.0;

	float gw2[25];
	[unroll] for (uint i = 0; i < 25; i++) gw2[i] = UNQ_R4[weights[i]];
	float sse = 0.0;
	[unroll] for (uint py = 0; py < 6; py++) {
		[unroll] for (uint px = 0; px < 6; px++) {
			float fx = ASTC_5X5_AXIS_POS[px];
			float fy = ASTC_5X5_AXIS_POS[py];
			int   ix = int(fx), iy = int(fy);
			int   ix1 = min(ix + 1, 4);
			int   iy1 = min(iy + 1, 4);
			float tx = fx - float(ix), ty = fy - float(iy);
			float w00 = gw2[iy  * 5 + ix ];
			float w10 = gw2[iy  * 5 + ix1];
			float w01 = gw2[iy1 * 5 + ix ];
			float w11 = gw2[iy1 * 5 + ix1];
			float w   = lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty);
			float4 recon = lerp(e0_f, e1_f, w);
			float4 d     = pixels[py * 6 + px] - recon;
			sse += dot(d, d);
		}
	}
	out_sse = sse;
}

///////////////////////////////////////////////////////////////////////////////
// Mode A: CEM 8 (RGB-only), 4x4 weight grid, 3-bit weights, 8-bit endpoints
//         Decoder writes alpha = 1.0 unconditionally, so SSE includes
//         alpha error against the source — opaque blocks (α=1.0 everywhere)
//         get zero alpha penalty AND benefit from not spending bits on α.
///////////////////////////////////////////////////////////////////////////////

void encode_mode_4x4_rgb_only(
	in  float4 pixels[36],
	in  float3 faxis,     in  float faxis_len_sq, in float fc0_proj,
	in  float  pproj[36], in  uint3 imin_rgb,     in uint3 imax_rgb,
	out uint4  out_block, out float out_sse)
{
	uint3 e0 = imin_rgb;
	uint3 e1 = imax_rgb;

	float T[7];
	T[0] = faxis_len_sq * 0.0703125 + fc0_proj;
	T[1] = faxis_len_sq * 0.2109375 + fc0_proj;
	T[2] = faxis_len_sq * 0.3515625 + fc0_proj;
	T[3] = faxis_len_sq * 0.5000000 + fc0_proj;
	T[4] = faxis_len_sq * 0.6484375 + fc0_proj;
	T[5] = faxis_len_sq * 0.7890625 + fc0_proj;
	T[6] = faxis_len_sq * 0.9296875 + fc0_proj;

	static const float wcoord[4] = { 0.0, 5.0/3.0, 10.0/3.0, 5.0 };
	uint weights[16];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 16; i++) weights[i] = 0;
	} else {
		[unroll] for (uint wy = 0; wy < 4; wy++) {
			[unroll] for (uint wx = 0; wx < 4; wx++) {
				float fx = wcoord[wx], fy = wcoord[wy];
				int   ix0 = int(fx),   iy0 = int(fy);
				int   ix1 = min(ix0 + 1, 5);
				int   iy1 = min(iy0 + 1, 5);
				float tx = fx - float(ix0), ty = fy - float(iy0);
				float p00 = pproj[iy0 * 6 + ix0];
				float p10 = pproj[iy0 * 6 + ix1];
				float p01 = pproj[iy1 * 6 + ix0];
				float p11 = pproj[iy1 * 6 + ix1];
				float p   = lerp(lerp(p00, p10, tx), lerp(p01, p11, tx), ty);
				uint  q = uint(p >= T[0]) + uint(p >= T[1]) + uint(p >= T[2])
				        + uint(p >= T[3]) + uint(p >= T[4]) + uint(p >= T[5])
				        + uint(p >= T[6]);
				weights[wy * 4 + wx] = q;
			}
		}
	}

	if (faxis_len_sq >= 1e-6) {
		float gw[16];
		[unroll] for (uint i = 0; i < 16; i++) gw[i] = UNQ_R8[weights[i]];

		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0), E = float3(0, 0, 0);
		[unroll] for (uint py = 0; py < 6; py++) {
			[unroll] for (uint px = 0; px < 6; px++) {
				float fx = ASTC_4X4_AXIS_POS[px];
				float fy = ASTC_4X4_AXIS_POS[py];
				int   ix = int(fx), iy = int(fy);
				int   ix1 = min(ix + 1, 3);
				int   iy1 = min(iy + 1, 3);
				float tx = fx - float(ix), ty = fy - float(iy);
				float w00 = gw[iy  * 4 + ix ];
				float w10 = gw[iy  * 4 + ix1];
				float w01 = gw[iy1 * 4 + ix ];
				float w11 = gw[iy1 * 4 + ix1];
				float w   = lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty);
				float3 p  = pixels[py * 6 + px].rgb;
				float  iw = 1.0 - w;
				A += iw*iw; B += w*iw; C += w*w;
				D += iw * p; E += w * p;
			}
		}

		float det = A * C - B * B;
		if (det > 1e-6 && det > 0.01 * A * C) {
			float  inv_det = 1.0 / det;
			float3 e0_ref  = saturate((C * D - B * E) * inv_det);
			float3 e1_ref  = saturate((A * E - B * D) * inv_det);
			uint3  n0      = uint3(e0_ref * 255.0 + 0.5);
			uint3  n1      = uint3(e1_ref * 255.0 + 0.5);
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
	out_block = block;

	// SSE — RGB reconstruction + alpha-against-1.0 penalty.
	float3 e0_f = float3(e0) / 255.0;
	float3 e1_f = float3(e1) / 255.0;
	float  gw2[16];
	[unroll] for (uint i = 0; i < 16; i++) gw2[i] = UNQ_R8[weights[i]];
	float sse = 0.0;
	[unroll] for (uint py = 0; py < 6; py++) {
		[unroll] for (uint px = 0; px < 6; px++) {
			float fx = ASTC_4X4_AXIS_POS[px];
			float fy = ASTC_4X4_AXIS_POS[py];
			int   ix = int(fx), iy = int(fy);
			int   ix1 = min(ix + 1, 3);
			int   iy1 = min(iy + 1, 3);
			float tx = fx - float(ix), ty = fy - float(iy);
			float w00 = gw2[iy  * 4 + ix ];
			float w10 = gw2[iy  * 4 + ix1];
			float w01 = gw2[iy1 * 4 + ix ];
			float w11 = gw2[iy1 * 4 + ix1];
			float w   = lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty);
			float3 recon_rgb = lerp(e0_f, e1_f, w);
			float4 d         = float4(pixels[py * 6 + px].rgb - recon_rgb,
			                          pixels[py * 6 + px].a   - 1.0);
			sse += dot(d, d);
		}
	}
	out_sse = sse;
}

///////////////////////////////////////////////////////////////////////////////
// Mode B: CEM 8 (RGB-only), 6x6 per-pixel grid, 2-bit weights, range-80
//         Sharp opaque blocks. Same alpha-against-1.0 SSE penalty.
///////////////////////////////////////////////////////////////////////////////

void encode_mode_6x6pp_rgb_only(
	in  float4 pixels[36],
	in  float3 faxis,     in  float faxis_len_sq, in float fc0_proj,
	in  float  pproj[36], in  uint3 imin_rgb,     in uint3 imax_rgb,
	out uint4  out_block, out float out_sse)
{
	uint3 e0 = imin_rgb;
	uint3 e1 = imax_rgb;

	float T[3];
	T[0] = faxis_len_sq * 0.1640625 + fc0_proj;
	T[1] = faxis_len_sq * 0.5000000 + fc0_proj;
	T[2] = faxis_len_sq * 0.8359375 + fc0_proj;

	uint weights[36];
	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 36; i++) weights[i] = 0;
	} else {
		[unroll] for (uint i = 0; i < 36; i++) {
			float p = pproj[i];
			uint  q = uint(p >= T[0]) + uint(p >= T[1]) + uint(p >= T[2]);
			weights[i] = q;
		}
	}

	if (faxis_len_sq >= 1e-6) {
		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0), E = float3(0, 0, 0);
		[unroll] for (uint i = 0; i < 36; i++) {
			float  w  = UNQ_R4[weights[i]];
			float3 p  = pixels[i].rgb;
			float  iw = 1.0 - w;
			A += iw*iw; B += w*iw; C += w*w;
			D += iw * p; E += w * p;
		}

		float det = A * C - B * B;
		if (det > 1e-6 && det > 0.01 * A * C) {
			float  inv_det = 1.0 / det;
			float3 e0_ref  = saturate((C * D - B * E) * inv_det);
			float3 e1_ref  = saturate((A * E - B * D) * inv_det);
			uint3  n0      = uint3(e0_ref * 255.0 + 0.5);
			uint3  n1      = uint3(e1_ref * 255.0 + 0.5);
			if (n0.r + n0.g + n0.b > n1.r + n1.g + n1.b) {
				uint3 tmp = n0; n0 = n1; n1 = tmp;
				[unroll] for (uint i = 0; i < 36; i++) weights[i] = 3u - weights[i];
			}
			e0 = n0; e1 = n1;
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_6x6_rgb(block);
	astc_write_endpoints_rgb6(block, e0, e1);
	[unroll] for (uint wi = 0; wi < 36; wi++) {
		astc_write_weight_2bit(block, wi, weights[wi]);
	}
	out_block = block;

	// SSE: same range-80 dequant pattern as the standalone 6x6pp shader,
	// plus alpha-against-1.0 penalty.
	uint3 q0 = uint3(astc_r80_quant_lut[min(e0.r, 255u)],
	                 astc_r80_quant_lut[min(e0.g, 255u)],
	                 astc_r80_quant_lut[min(e0.b, 255u)]);
	uint3 q1 = uint3(astc_r80_quant_lut[min(e1.r, 255u)],
	                 astc_r80_quant_lut[min(e1.g, 255u)],
	                 astc_r80_quant_lut[min(e1.b, 255u)]);
	float3 e0_f = float3(astc_dequant_r80(q0.r),
	                     astc_dequant_r80(q0.g),
	                     astc_dequant_r80(q0.b)) / 255.0;
	float3 e1_f = float3(astc_dequant_r80(q1.r),
	                     astc_dequant_r80(q1.g),
	                     astc_dequant_r80(q1.b)) / 255.0;
	float sse = 0.0;
	[unroll] for (uint i = 0; i < 36; i++) {
		float  w     = UNQ_R4[weights[i]];
		float3 recon = lerp(e0_f, e1_f, w);
		float4 d     = float4(pixels[i].rgb - recon, pixels[i].a - 1.0);
		sse += dot(d, d);
	}
	out_sse = sse;
}

///////////////////////////////////////////////////////////////////////////////
// Mode E: CEM 12 dual-plane, 3x3 grid, 2-bit weights, 8-bit endpoints
//
// Block layout (128 bits):
//   bits   0-10 : block mode (0x5AE — 3x3 grid, 2-bit weights, D=1)
//   bits  11-12 : partition count = 0
//   bits  13-16 : CEM 12
//   bits  17-80 : 8 endpoint values × 8 bits = 64 bits (R0,R1,G0,G1,B0,B1,A0,A1)
//   bits  81-89 : unused (9 bits)
//   bits  90-91 : CCS = 3 (alpha as secondary plane)
//   bits  92-127: 36-bit dual-plane weight area (interleaved primary/secondary,
//                 bit-reversed at the top)
//
// The primary weight controls RGB interpolation; the secondary weight (CCS=3
// → alpha) is independent, so alpha cutouts can have sharp transitions even
// when RGB varies smoothly.
///////////////////////////////////////////////////////////////////////////////

void encode_mode_dual_3x3_rgba(
	in  float4 pixels[36],
	in  float3 faxis,     in  float faxis_len_sq, in float fc0_proj,
	in  float  pproj[36], in  uint3 imin_rgb,     in uint3 imax_rgb,
	in  uint   amin,      in  uint  amax,
	out uint4  out_block, out float out_sse)
{
	uint3 e0 = imin_rgb;
	uint3 e1 = imax_rgb;
	uint  a0 = amin;
	uint  a1 = amax;

	// Primary (RGB) weight thresholds: 2-bit = 4 levels.
	float T_rgb[3];
	T_rgb[0] = faxis_len_sq * 0.1640625 + fc0_proj;
	T_rgb[1] = faxis_len_sq * 0.5000000 + fc0_proj;
	T_rgb[2] = faxis_len_sq * 0.8359375 + fc0_proj;

	// Secondary (alpha) is 1D — quantize per-pixel alpha to 4 levels using
	// (alpha - amin) / (amax - amin) against {0.164, 0.5, 0.836} thresholds.
	float a_range = float(int(amax) - int(amin));
	bool  alpha_degenerate = a_range < 1.0;

	// Sample 9 grid weights per plane via bilinear downsample.
	static const float wcoord[3] = { 0.0, 2.5, 5.0 };
	uint weights_pri[9];
	uint weights_sec[9];

	if (faxis_len_sq < 1e-6) {
		[unroll] for (uint i = 0; i < 9; i++) weights_pri[i] = 0;
	} else {
		[unroll] for (uint wy = 0; wy < 3; wy++) {
			[unroll] for (uint wx = 0; wx < 3; wx++) {
				float fx = wcoord[wx], fy = wcoord[wy];
				int   ix0 = int(fx),   iy0 = int(fy);
				int   ix1 = min(ix0 + 1, 5);
				int   iy1 = min(iy0 + 1, 5);
				float tx = fx - float(ix0), ty = fy - float(iy0);
				float p00 = pproj[iy0 * 6 + ix0];
				float p10 = pproj[iy0 * 6 + ix1];
				float p01 = pproj[iy1 * 6 + ix0];
				float p11 = pproj[iy1 * 6 + ix1];
				float p   = lerp(lerp(p00, p10, tx), lerp(p01, p11, tx), ty);
				uint  q   = uint(p >= T_rgb[0]) + uint(p >= T_rgb[1]) + uint(p >= T_rgb[2]);
				weights_pri[wy * 3 + wx] = q;
			}
		}
	}

	if (alpha_degenerate) {
		[unroll] for (uint i = 0; i < 9; i++) weights_sec[i] = 0;
	} else {
		// Bilinear-downsample per-pixel alpha-t (in [0,1]) to the 9 grid
		// points, then quantize to 4 levels via the same midpoint thresholds.
		float a_t[36];
		float inv_arange = 1.0 / a_range;
		[unroll] for (uint i = 0; i < 36; i++) {
			a_t[i] = (pixels[i].a * 255.0 - float(amin)) * inv_arange;
		}
		[unroll] for (uint wy = 0; wy < 3; wy++) {
			[unroll] for (uint wx = 0; wx < 3; wx++) {
				float fx = wcoord[wx], fy = wcoord[wy];
				int   ix0 = int(fx),   iy0 = int(fy);
				int   ix1 = min(ix0 + 1, 5);
				int   iy1 = min(iy0 + 1, 5);
				float tx = fx - float(ix0), ty = fy - float(iy0);
				float p00 = a_t[iy0 * 6 + ix0];
				float p10 = a_t[iy0 * 6 + ix1];
				float p01 = a_t[iy1 * 6 + ix0];
				float p11 = a_t[iy1 * 6 + ix1];
				float p   = lerp(lerp(p00, p10, tx), lerp(p01, p11, tx), ty);
				uint  q   = uint(p >= 0.1640625) + uint(p >= 0.5) + uint(p >= 0.8359375);
				weights_sec[wy * 3 + wx] = q;
			}
		}
	}

	// LS refine RGB endpoints using primary weights (same pattern as
	// encode_mode_3x3_rgba). Skip alpha LS — keep a0/a1 at bbox.
	if (faxis_len_sq >= 1e-6) {
		float gw[9];
		[unroll] for (uint i = 0; i < 9; i++) gw[i] = UNQ_R4[weights_pri[i]];

		float  A = 0, B = 0, C = 0;
		float3 D = float3(0, 0, 0), E = float3(0, 0, 0);
		[unroll] for (uint py = 0; py < 6; py++) {
			[unroll] for (uint px = 0; px < 6; px++) {
				float fx = ASTC_3X3_AXIS_POS[px];
				float fy = ASTC_3X3_AXIS_POS[py];
				int   ix = int(fx), iy = int(fy);
				int   ix1 = min(ix + 1, 2);
				int   iy1 = min(iy + 1, 2);
				float tx = fx - float(ix), ty = fy - float(iy);
				float w00 = gw[iy  * 3 + ix ];
				float w10 = gw[iy  * 3 + ix1];
				float w01 = gw[iy1 * 3 + ix ];
				float w11 = gw[iy1 * 3 + ix1];
				float w   = lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty);
				float3 p  = pixels[py * 6 + px].rgb;
				float  iw = 1.0 - w;
				A += iw*iw; B += w*iw; C += w*w;
				D += iw * p; E += w * p;
			}
		}

		float det = A * C - B * B;
		if (det > 1e-6 && det > 0.01 * A * C) {
			float  inv_det = 1.0 / det;
			float3 e0_ref  = saturate((C * D - B * E) * inv_det);
			float3 e1_ref  = saturate((A * E - B * D) * inv_det);
			uint3  n0      = uint3(e0_ref * 255.0 + 0.5);
			uint3  n1      = uint3(e1_ref * 255.0 + 0.5);
			// Blue-contract check uses RGB only; alpha endpoints/weights
			// are not affected by an RGB swap (independent plane).
			if (n0.r + n0.g + n0.b > n1.r + n1.g + n1.b) {
				uint3 tmp = n0; n0 = n1; n1 = tmp;
				[unroll] for (uint i = 0; i < 9; i++) weights_pri[i] = 3u - weights_pri[i];
			}
			e0 = n0; e1 = n1;
		}
	}

	uint4 block = uint4(0, 0, 0, 0);
	astc_write_header_3x3_rgba_dual(block);
	astc_write_endpoints_rgba8(block, e0, e1, a0, a1);
	astc_write_ccs(block, 90u, ASTC_CCS_ALPHA);
	[unroll] for (uint wi = 0; wi < 9; wi++) {
		astc_write_dual_weight_2bit(block, wi, weights_pri[wi], weights_sec[wi]);
	}
	out_block = block;

	// SSE: RGB uses primary weight, alpha uses secondary weight — both
	// bilinearly upsampled from the 9 grid points to per-pixel.
	float3 e0_rgb_f = float3(e0) / 255.0;
	float3 e1_rgb_f = float3(e1) / 255.0;
	float  a0_f     = float(a0) / 255.0;
	float  a1_f     = float(a1) / 255.0;
	float gw_pri[9], gw_sec[9];
	[unroll] for (uint i = 0; i < 9; i++) {
		gw_pri[i] = UNQ_R4[weights_pri[i]];
		gw_sec[i] = UNQ_R4[weights_sec[i]];
	}
	float sse = 0.0;
	[unroll] for (uint py = 0; py < 6; py++) {
		[unroll] for (uint px = 0; px < 6; px++) {
			float fx = ASTC_3X3_AXIS_POS[px];
			float fy = ASTC_3X3_AXIS_POS[py];
			int   ix = int(fx), iy = int(fy);
			int   ix1 = min(ix + 1, 2);
			int   iy1 = min(iy + 1, 2);
			float tx = fx - float(ix), ty = fy - float(iy);
			// Primary
			float wp00 = gw_pri[iy  * 3 + ix ];
			float wp10 = gw_pri[iy  * 3 + ix1];
			float wp01 = gw_pri[iy1 * 3 + ix ];
			float wp11 = gw_pri[iy1 * 3 + ix1];
			float wp   = lerp(lerp(wp00, wp10, tx), lerp(wp01, wp11, tx), ty);
			// Secondary
			float ws00 = gw_sec[iy  * 3 + ix ];
			float ws10 = gw_sec[iy  * 3 + ix1];
			float ws01 = gw_sec[iy1 * 3 + ix ];
			float ws11 = gw_sec[iy1 * 3 + ix1];
			float ws   = lerp(lerp(ws00, ws10, tx), lerp(ws01, ws11, tx), ty);

			float3 recon_rgb = lerp(e0_rgb_f, e1_rgb_f, wp);
			float  recon_a   = lerp(a0_f, a1_f, ws);
			float4 d = pixels[py * 6 + px] - float4(recon_rgb, recon_a);
			sse += dot(d, d);
		}
	}
	out_sse = sse;
}

///////////////////////////////////////////////////////////////////////////////
// Main entry point
///////////////////////////////////////////////////////////////////////////////

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint bx = id.x;
	uint by = id.y;
	uint blocks_y = (image_height + 5u) / 6u;
	if (bx >= blocks_x || by >= blocks_y)
		return;

	// Load 6x6 block of RGBA pixels, compute bbox.
	float4 pixels[36];
	float3 cmin_rgb = float3(1, 1, 1);
	float3 cmax_rgb = float3(0, 0, 0);
	float  amin     = 1.0;
	float  amax     = 0.0;
	[unroll] for (uint py = 0; py < 6; py++) {
		[unroll] for (uint px = 0; px < 6; px++) {
			uint sx = min(bx * 6u + px, image_width  - 1u);
			uint sy = min(by * 6u + py, image_height - 1u);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			pixels[py * 6u + px] = texel;
			cmin_rgb = min(cmin_rgb, texel.rgb);
			cmax_rgb = max(cmax_rgb, texel.rgb);
			amin     = min(amin,     texel.a);
			amax     = max(amax,     texel.a);
		}
	}

	uint3 imin_rgb = uint3(cmin_rgb * 255.0 + 0.5);
	uint3 imax_rgb = uint3(cmax_rgb * 255.0 + 0.5);
	uint3 range = imax_rgb - imin_rgb;
	imin_rgb += range >> 4;
	imax_rgb -= range >> 4;

	uint a0 = uint(amin * 255.0 + 0.5);
	uint a1 = uint(amax * 255.0 + 0.5);

	// Perceptual axis on RGB only (CEM 12 single-plane couples alpha to RGB).
	float3 faxis = float3(
		float(int(imax_rgb.r) - int(imin_rgb.r)) * 2.0,
		float(int(imax_rgb.g) - int(imin_rgb.g)) * 4.0,
		float(int(imax_rgb.b) - int(imin_rgb.b)));
	float faxis_len_sq = (faxis.r * faxis.r * 0.5 + faxis.g * faxis.g * 0.25 + faxis.b * faxis.b) / 255.0;
	float fc0_proj     = dot(float3(imin_rgb) / 255.0, faxis);

	float pproj[36];
	[unroll] for (uint i = 0; i < 36; i++) {
		pproj[i] = dot(pixels[i].rgb, faxis);
	}

	// Per-block early-out: pick the mode set that can possibly win based on
	// the alpha content. Skipping unwinnable modes is safe because their
	// SSE would be strictly higher than at least one mode we still run.
	//
	//   Opaque (α=255 everywhere)        → run only CEM 8 modes (2)
	//   Uniform translucent (α<255)      → run only single-plane CEM 12 (2)
	//   Varying alpha                    → run all 3 CEM 12 modes (3)
	//
	// For the typical mostly-opaque image this drops average mode count from
	// 5 to ~2, matching the cost of the RGB-only encoder on those blocks.
	uint4 best     = uint4(0, 0, 0, 0);
	float best_sse = 1e10;
	uint4 b; float s;

	if (a0 == 255u && a1 == 255u) {
		// Opaque block: CEM 12 modes are strictly worse than CEM 8 here —
		// they spend endpoint bits on alpha that doesn't need encoding,
		// at no quality benefit (alpha=1.0 either way).
		encode_mode_4x4_rgb_only  (pixels, faxis, faxis_len_sq, fc0_proj, pproj, imin_rgb, imax_rgb, b, s);
		best = b; best_sse = s;
		encode_mode_6x6pp_rgb_only(pixels, faxis, faxis_len_sq, fc0_proj, pproj, imin_rgb, imax_rgb, b, s);
		if (s < best_sse) { best = b; best_sse = s; }
	} else if (a0 == a1) {
		// Uniform translucent block: CEM 8 has a large constant alpha
		// penalty and would lose. Dual-plane has no per-pixel alpha
		// variation to capture — same result as single-plane but coarser
		// RGB weights. Run only the two single-plane CEM 12 modes.
		encode_mode_3x3_rgba(pixels, faxis, faxis_len_sq, fc0_proj, pproj, imin_rgb, imax_rgb, a0, a1, b, s);
		best = b; best_sse = s;
		encode_mode_5x5_rgba(pixels, faxis, faxis_len_sq, fc0_proj, pproj, imin_rgb, imax_rgb, a0, a1, b, s);
		if (s < best_sse) { best = b; best_sse = s; }
	} else {
		// Varying alpha: try all 3 CEM 12 modes. CEM 8 still loses.
		encode_mode_3x3_rgba     (pixels, faxis, faxis_len_sq, fc0_proj, pproj, imin_rgb, imax_rgb, a0, a1, b, s);
		best = b; best_sse = s;
		encode_mode_5x5_rgba     (pixels, faxis, faxis_len_sq, fc0_proj, pproj, imin_rgb, imax_rgb, a0, a1, b, s);
		if (s < best_sse) { best = b; best_sse = s; }
		encode_mode_dual_3x3_rgba(pixels, faxis, faxis_len_sq, fc0_proj, pproj, imin_rgb, imax_rgb, a0, a1, b, s);
		if (s < best_sse) { best = b; best_sse = s; }
	}

	output_blocks[buffer_offset + by * blocks_x + bx] = best;
}
