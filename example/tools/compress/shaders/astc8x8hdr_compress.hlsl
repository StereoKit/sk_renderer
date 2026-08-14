//--name = astc8x8hdr_compress

// ASTC 8x8 HDR GPU texture compression — RGB only (CEM 11), 2-mode selector.
//
// Two candidate modes per block; the one with lower reconstruction SSE wins:
//
//   Mode A — 6x5 grid, 2-bit weights, range-256 endpoints
//     30 weights × 4 levels = 120 effective steps
//     Best for: detail/edges/textured content (more spatial points)
//     Bit budget: 17 + 60 + 48 = 125 / 128 used (3 spare)
//
//   Mode B — 4x4 grid, trit+2bit (12 lvl) weights, range-256 endpoints
//     16 weights × 12 levels = 192 effective steps
//     Best for: smooth gradients (finer axial interpolation via BISE trit)
//     Bit budget: 17 + 58 + 48 = 123 / 128 used (5 spare)
//
// Both modes share the underlying endpoint encoding (CEM 11 via
// astc_quantize_hdr_rgb at QUANT_256), so the 8-mode HDR endpoint search
// runs only once per block. Mode B packs its 16 weights as 3 full 5-trit
// groups (18 bits each) + 1 partial-1 group (4 bits), total 58 wt bits.
//
// SSE comparison is in normalized projection space along the shared axis
// (lns_max - lns_min). Per-pixel weights use bilinear interpolation of the
// grid weights — matches what the decoder actually does.
//
// Source must be a float-format texture (RGBA16/RGBA32 float). Alpha unused.
//
// NOTE: HDR ASTC is *not* decodable by mesa's software path; AMD desktop
// will display magenta. Hardware ASTC (Adreno, Mali, Apple, NV with HDR
// extension) decodes correctly. astcenc -dh decodes for offline validation.

Texture2D<float4>         source_tex    : register(t0);
RWStructuredBuffer<uint4> output_blocks : register(u1);

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

#include "astc_common.hlsli"

///////////////////////////////////////////////////////////////////////////////
// Mode A: 6x5 grid, 2-bit weights — Voronoi assignment + bilinear interp.
///////////////////////////////////////////////////////////////////////////////

// 2-bit weight unquant levels (UNQ_R4) and midpoint thresholds (UNQ_R4_T0..T2)
// come from astc_common.hlsli.

// Voronoi assignment for 6x5 grid: 6 X grid points, 5 Y. X cell counts
// {1,2,1,1,2,1}, Y {1,2,2,2,1}. Total = 2·8 + 3·16 = 64 pixels ✓.
static const uint A_PIX_X_TO_GRID[8] = { 0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u };
static const uint A_PIX_Y_TO_GRID[8] = { 0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u };
static const uint A_GRID_PIX_COUNT[30] = {
	1u, 2u, 1u, 1u, 2u, 1u,   // gy=0 (corner row)
	2u, 4u, 2u, 2u, 4u, 2u,   // gy=1
	2u, 4u, 2u, 2u, 4u, 2u,   // gy=2
	2u, 4u, 2u, 2u, 4u, 2u,   // gy=3
	1u, 2u, 1u, 1u, 2u, 1u,   // gy=4 (corner row)
};

// Decoder bilinear-interp coordinates for SSE — X has 6 grid points, Y has 5.
static const uint A_BL_JX[8] = { 0u, 0u, 1u, 2u, 2u, 3u, 4u, 5u };
static const uint A_BL_WX[8] = { 0u, 11u, 7u, 2u, 14u, 9u, 4u, 0u };
static const uint A_BL_JY[8] = { 0u, 0u, 1u, 1u, 2u, 2u, 3u, 4u };
static const uint A_BL_WY[8] = { 0u, 9u, 2u, 11u, 5u, 14u, 7u, 0u };

///////////////////////////////////////////////////////////////////////////////
// Mode B: 4x4 grid, trit+2bit (12 lvl) BISE weights.
///////////////////////////////////////////////////////////////////////////////

// trit+2bit unquant indexed by ENCODED v ∈ [0, 11] is UNQ_R12_V from
// astc_common.hlsli. Trit-encoded unquant is non-monotonic in v, pairs as
// (v, v^1) mirror across 0.5 — so the LS-style endpoint-swap would use v^1
// to flip weights if needed (we don't LS in HDR but use this for SSE lookup
// after quantization).

// 4x4 grid in 8x8 block: each grid cell covers a 2x2 pixel patch. Pixel
// positions (in /16 units) are 0,7,14,21,27,34,41,48 against grid posts at
// 0,16,32,48 → Voronoi sets {0,1}, {2,3}, {4,5}, {6,7}. All cells own 2x2=4
// source pixels (uniform), so no per-cell count table is needed.
static const uint B_PIX_TO_GRID[8] = { 0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u };

// Bilinear-interp coords for 4x4 grid in 8x8 block (symmetric in x and y).
static const uint B_BL_J[8] = { 0u, 0u, 0u, 1u, 1u, 2u, 2u, 3u };
static const uint B_BL_W[8] = { 0u, 7u, 14u, 5u, 11u, 2u, 9u, 0u };

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint bx = id.x;
	uint by = id.y;
	uint blocks_y = (image_height + 7u) / 8u;
	if (bx >= blocks_x || by >= blocks_y)
		return;

	// Load 8x8 block; convert each pixel to LNS so all subsequent math
	// (bbox, axis projection, weight quantization, endpoint pack, SSE)
	// lives in the same space the HDR decoder interpolates in.
	float3 lns_pixels[64];
	float3 lns_min = float3(65535.0, 65535.0, 65535.0);
	float3 lns_max = float3(0, 0, 0);
	[unroll] for (uint py = 0; py < 8; py++) {
		[unroll] for (uint px = 0; px < 8; px++) {
			uint sx = min(bx * 8u + px, image_width  - 1u);
			uint sy = min(by * 8u + py, image_height - 1u);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			float3 lns = float3(
				astc_float_to_lns(max(texel.r, 0.0)),
				astc_float_to_lns(max(texel.g, 0.0)),
				astc_float_to_lns(max(texel.b, 0.0)));
			lns_pixels[py * 8u + px] = lns;
			lns_min = min(lns_min, lns);
			lns_max = max(lns_max, lns);
		}
	}

	// Endpoint encoding is shared between modes (both use CEM 11 + range
	// 256). Run the 8-mode HDR search once.
	uint v0, v1, v2, v3, v4, v5;
	astc_quantize_hdr_rgb(lns_min, lns_max, v0, v1, v2, v3, v4, v5);

	// Shared axis.
	float3 axis        = lns_max - lns_min;
	float  axis_len_sq = dot(axis, axis);
	float  c0_proj     = dot(lns_min, axis);

	// Degenerate flat block — every mode would output all-zero weights.
	// Use Mode B (smallest encoded weight area at all-zeros).
	if (axis_len_sq < 1.0) {
		uint4 b = uint4(0, 0, 0, 0);
		astc_write_header_8x8_hdr_rgb_4x4_r12(b);
		astc_write_endpoints_v6(b, v0, v1, v2, v3, v4, v5);
		output_blocks[buffer_offset + by * blocks_x + bx] = b;
		return;
	}

	// Precompute per-pixel projection. The normalized-to-[0,1] form is
	// recomputed at each use site as (pixel_proj - c0_proj) * inv_axis_len_sq
	// rather than stored — keeps register pressure down.
	float pixel_proj[64];
	float inv_axis_len_sq = 1.0 / axis_len_sq;
	[unroll] for (uint pi = 0; pi < 64; pi++) {
		pixel_proj[pi] = dot(lns_pixels[pi], axis);
	}

	///////////////////////////////////////////////////////////////////////
	// Mode A: 6x5 + 2-bit (4 levels)
	///////////////////////////////////////////////////////////////////////
	uint  weights_A[30];
	float sse_A = 0.0;
	{
		// 2-bit thresholds in projection space.
		float TA[3];
		TA[0] = axis_len_sq * UNQ_R4_T0 + c0_proj;
		TA[1] = axis_len_sq * UNQ_R4_T1 + c0_proj;
		TA[2] = axis_len_sq * UNQ_R4_T2 + c0_proj;

		// Voronoi-bin pixel projections into 30 grid cells.
		float grid_proj[30];
		[unroll] for (uint i = 0; i < 30; i++) grid_proj[i] = 0;
		[unroll] for (uint py = 0; py < 8; py++) {
			[unroll] for (uint px = 0; px < 8; px++) {
				grid_proj[A_PIX_Y_TO_GRID[py] * 6u + A_PIX_X_TO_GRID[px]] += pixel_proj[py * 8u + px];
			}
		}
		[unroll] for (uint gi = 0; gi < 30; gi++) {
			float p = grid_proj[gi] / float(A_GRID_PIX_COUNT[gi]);
			weights_A[gi] = uint(p >= TA[0]) + uint(p >= TA[1]) + uint(p >= TA[2]);
		}

		[unroll] for (uint sy = 0; sy < 8; sy++) {
			[unroll] for (uint sx = 0; sx < 8; sx++) {
				uint  jx  = A_BL_JX[sx]; uint jx1 = min(jx + 1u, 5u);
				uint  jy  = A_BL_JY[sy]; uint jy1 = min(jy + 1u, 4u);
				float fx  = float(A_BL_WX[sx]) * (1.0 / 16.0);
				float fy  = float(A_BL_WY[sy]) * (1.0 / 16.0);
				float w00 = UNQ_R4[weights_A[jy  * 6u + jx ]];
				float w10 = UNQ_R4[weights_A[jy  * 6u + jx1]];
				float w01 = UNQ_R4[weights_A[jy1 * 6u + jx ]];
				float w11 = UNQ_R4[weights_A[jy1 * 6u + jx1]];
				float w_i = lerp(lerp(w00, w10, fx), lerp(w01, w11, fx), fy);
				float p_n = (pixel_proj[sy * 8u + sx] - c0_proj) * inv_axis_len_sq;
				float err = p_n - w_i;
				sse_A += err * err;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////
	// Mode B: 4x4 + trit+2bit (12 levels)
	///////////////////////////////////////////////////////////////////////
	uint  weights_B[16];
	float sse_B = 0.0;
	{
		// Voronoi-bin pixel projections into 16 cells (uniform 4 pixels each).
		float grid_proj[16];
		[unroll] for (uint i = 0; i < 16; i++) grid_proj[i] = 0;
		[unroll] for (uint py = 0; py < 8; py++) {
			[unroll] for (uint px = 0; px < 8; px++) {
				grid_proj[B_PIX_TO_GRID[py] * 4u + B_PIX_TO_GRID[px]] += pixel_proj[py * 8u + px];
			}
		}
		// Quantize cell-mean projections to trit+2bit (12 lvl) weights. Output
		// is ENCODED v ∈ [0, 11]; UNQ_R12_V[v] is the dequantized value.
		[unroll] for (uint gi = 0; gi < 16; gi++) {
			float p     = grid_proj[gi] * 0.25;
			float norm  = (p - c0_proj) * inv_axis_len_sq;
			weights_B[gi] = astc_quantize_weight_trit_2bit(saturate(norm));
		}

		[unroll] for (uint sy = 0; sy < 8; sy++) {
			[unroll] for (uint sx = 0; sx < 8; sx++) {
				uint  jx  = B_BL_J[sx]; uint jx1 = min(jx + 1u, 3u);
				uint  jy  = B_BL_J[sy]; uint jy1 = min(jy + 1u, 3u);
				float fx  = float(B_BL_W[sx]) * (1.0 / 16.0);
				float fy  = float(B_BL_W[sy]) * (1.0 / 16.0);
				float w00 = UNQ_R12_V[weights_B[jy  * 4u + jx ]];
				float w10 = UNQ_R12_V[weights_B[jy  * 4u + jx1]];
				float w01 = UNQ_R12_V[weights_B[jy1 * 4u + jx ]];
				float w11 = UNQ_R12_V[weights_B[jy1 * 4u + jx1]];
				float w_i = lerp(lerp(w00, w10, fx), lerp(w01, w11, fx), fy);
				float p_n = (pixel_proj[sy * 8u + sx] - c0_proj) * inv_axis_len_sq;
				float err = p_n - w_i;
				sse_B += err * err;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////
	// Pick winner + pack the chosen block.
	///////////////////////////////////////////////////////////////////////
	uint4 block = uint4(0, 0, 0, 0);
	if (sse_A <= sse_B) {
		astc_write_header_8x8_hdr_rgb_6x5(block);
		astc_write_endpoints_v6          (block, v0, v1, v2, v3, v4, v5);
		[unroll] for (uint wi = 0; wi < 30; wi++) {
			astc_write_weight_2bit(block, wi, weights_A[wi]);
		}
	} else {
		astc_write_header_8x8_hdr_rgb_4x4_r12(block);
		astc_write_endpoints_v6              (block, v0, v1, v2, v3, v4, v5);
		// 16 weights = 3 full 5-trit groups + 1 partial-1 group.
		astc_write_weights16_trit_2bit(block, weights_B);
	}
	output_blocks[buffer_offset + by * blocks_x + bx] = block;
}
