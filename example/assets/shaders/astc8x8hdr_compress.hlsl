//--name = astc8x8hdr_compress

// ASTC 8x8 HDR GPU texture compression — RGB only (CEM 11), 2-mode selector.
//
// Two candidate modes per block; the one with lower reconstruction SSE wins:
//
//   Mode A — 6x5 grid, 2-bit weights, range-256 endpoints
//     30 weights × 4 levels = 120 effective steps
//     Best for: detail/edges/textured content (more spatial points)
//     Bit budget: 17 + 60 + 48 = 125 / 128 used
//
//   Mode B — 4x4 grid, 3-bit weights, range-256 endpoints
//     16 weights × 8 levels = 128 effective steps
//     Best for: smooth gradients (finer along-axis interpolation)
//     Bit budget: 17 + 48 + 48 = 113 / 128 used
//
// Both modes share the underlying endpoint encoding (CEM 11 via
// astc_quantize_hdr_rgb at QUANT_256), so the 8-mode HDR endpoint search
// runs only once per block.
//
// An 8x4 + range-192 third mode was tried and removed — only ~0.1 dB on
// foliage-heavy content at ~33% encoder cost. The supporting helpers
// (astc_write_endpoints_v6_r192_hdr, bit-preserving range-192 LUTs,
// astc_write_trit_group_partial1, ASTC_BLOCK_MODE_8x4_R2) stay in
// astc_common.hlsli for future re-use.
//
// SSE comparison is in normalized projection space along the shared axis
// (lns_max - lns_min). The off-axis component of error is identical for
// both modes (depends only on how far each pixel deviates from the e0-e1
// line) and cancels out, so axial-only SSE is sufficient for picking a
// winner. Per-pixel weights use bilinear interpolation of the grid weights
// — matches what the decoder actually does, and only this matches the
// downsample patterns of the two grids fairly.
//
// Source must be a float-format texture (RGBA16/RGBA32 float). Alpha unused.
//
// NOTE: HDR ASTC is *not* decodable by mesa's software path; AMD desktop
// will display magenta. Hardware ASTC (Adreno, Mali, Apple, NV with HDR
// extension) decodes correctly. astcenc -dh decodes for offline validation.

Texture2D<float4>         source_tex    : register(t0);
SamplerState              source_tex_s  : register(s0);
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

// 2-bit weight unquant levels (UNQ_R4) and midpoint thresholds.
static const float UNQ_R4[4]  = { 0.0, 21.0/64.0, 43.0/64.0, 1.0 };
static const float UNQ_R4_T0  = 21.0 / 128.0;  // ≈ 0.164
static const float UNQ_R4_T1  = 0.5;
static const float UNQ_R4_T2  = 107.0 / 128.0; // ≈ 0.836

// Voronoi assignment — which 6x5 grid point is each source pixel nearest to?
// X = {1,2,1,1,2,1} pixel-counts per grid; Y = {1,2,2,2,1}. Used to bin
// projected pixel values for cell-mean weight quantization.
static const uint A_PIX_X_TO_GRID[8] = { 0u, 1u, 1u, 2u, 3u, 4u, 4u, 5u };
static const uint A_PIX_Y_TO_GRID[8] = { 0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u };
static const uint A_GRID_PIX_COUNT[30] = {
	1u, 2u, 1u, 1u, 2u, 1u,   // gy=0
	2u, 4u, 2u, 2u, 4u, 2u,   // gy=1
	2u, 4u, 2u, 2u, 4u, 2u,   // gy=2
	2u, 4u, 2u, 2u, 4u, 2u,   // gy=3
	1u, 2u, 1u, 1u, 2u, 1u,   // gy=4
};

// Decoder bilinear-interp coordinates for SSE: per pixel, integer grid
// index + fractional /16 across both axes (X has 6 grid points, Y has 5).
static const uint A_BL_JX [8] = { 0u, 0u, 1u, 2u, 2u, 3u, 4u, 5u };
static const uint A_BL_WX [8] = { 0u, 11u, 7u, 2u, 14u, 9u, 4u, 0u };
static const uint A_BL_JY [8] = { 0u, 0u, 1u, 1u, 2u, 2u, 3u, 4u };
static const uint A_BL_WY [8] = { 0u, 9u, 2u, 11u, 5u, 14u, 7u, 0u };

///////////////////////////////////////////////////////////////////////////////
// Mode B: 4x4 grid, 3-bit weights — Voronoi assignment + bilinear interp.
///////////////////////////////////////////////////////////////////////////////

// 3-bit weight unquant levels (UNQ_R8) and midpoint thresholds (7 of them).
static const float UNQ_R8[8] = {
	0.0,         9.0  / 64.0, 18.0 / 64.0, 27.0 / 64.0,
	37.0 / 64.0, 46.0 / 64.0, 55.0 / 64.0, 1.0
};
static const float UNQ_R8_T[7] = {
	 4.5 / 64.0, 13.5 / 64.0, 22.5 / 64.0, 32.0 / 64.0,
	41.5 / 64.0, 50.5 / 64.0, 59.5 / 64.0
};

// 4x4 grid in 8x8 block: each grid cell owns a 2-pixel-wide column/row in
// each axis. Pixel positions (in /16 grid units) are 0,7,14,21,27,34,41,48
// against grid posts at 0,16,32,48 → Voronoi sets {0,1}, {2,3}, {4,5}, {6,7}.
static const uint B_PIX_TO_GRID[8] = { 0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u };
// All 16 cells own exactly 2x2 = 4 source pixels, so no per-cell count table
// is needed — the divisor is constant 4.

// Bilinear-interp coords for 4x4 grid in 8x8 block (symmetric in x and y):
static const uint B_BL_J [8] = { 0u, 0u, 0u, 1u, 1u, 2u, 2u, 3u };
static const uint B_BL_W [8] = { 0u, 7u, 14u, 5u, 11u, 2u, 9u, 0u };

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

	// Endpoint encoding is shared between modes (both write CEM 11 + range
	// 256, only the block mode bits + weight area differ). Run the 8-mode
	// HDR search once.
	uint v0, v1, v2, v3, v4, v5;
	astc_quantize_hdr_rgb(lns_min, lns_max, v0, v1, v2, v3, v4, v5);

	// Shared axis. Per-pixel projection (and its normalized [0,1]
	// counterpart) is precomputed once; both modes read from the same
	// pixel_proj_norm table.
	float3 axis        = lns_max - lns_min;
	float  axis_len_sq = dot(axis, axis);
	float  c0_proj     = dot(lns_min, axis);

	// Degenerate flat block — every mode would output all-zero weights and
	// give identical reconstruction. Skip the SSE compare and use Mode B
	// (range-256 endpoints, smallest weight area).
	if (axis_len_sq < 1.0) {
		uint4 b = uint4(0, 0, 0, 0);
		astc_write_header_8x8_hdr_rgb_4x4(b);
		astc_write_endpoints_v6(b, v0, v1, v2, v3, v4, v5);
		// All weights = 0 (unwritten), block goes out as just header + endpoints.
		output_blocks[buffer_offset + by * blocks_x + bx] = b;
		return;
	}

	// Precompute per-pixel projection (raw and normalized to [0,1]).
	// The raw form is what the threshold compares need; the normalized form
	// is the ideal weight value, used for SSE accumulation.
	float pixel_proj    [64];
	float pixel_proj_norm[64];
	float inv_axis_len_sq = 1.0 / axis_len_sq;
	[unroll] for (uint pi = 0; pi < 64; pi++) {
		pixel_proj[pi]      = dot(lns_pixels[pi], axis);
		pixel_proj_norm[pi] = (pixel_proj[pi] - c0_proj) * inv_axis_len_sq;
	}

	///////////////////////////////////////////////////////////////////////
	// Mode A: 6x5 + 2-bit
	///////////////////////////////////////////////////////////////////////
	uint  weights_A[30];
	float sse_A = 0.0;
	{
		// 2-bit thresholds in projection space.
		float TA[3];
		TA[0] = axis_len_sq * UNQ_R4_T0 + c0_proj;
		TA[1] = axis_len_sq * UNQ_R4_T1 + c0_proj;
		TA[2] = axis_len_sq * UNQ_R4_T2 + c0_proj;

		// Voronoi-bin pixel projections into the 30 grid cells.
		float grid_proj[30];
		[unroll] for (uint i = 0; i < 30; i++) grid_proj[i] = 0;
		[unroll] for (uint py = 0; py < 8; py++) {
			[unroll] for (uint px = 0; px < 8; px++) {
				grid_proj[A_PIX_Y_TO_GRID[py] * 6u + A_PIX_X_TO_GRID[px]] += pixel_proj[py * 8u + px];
			}
		}
		// Cell-mean → quantized weight via threshold compare.
		[unroll] for (uint gi = 0; gi < 30; gi++) {
			float p = grid_proj[gi] / float(A_GRID_PIX_COUNT[gi]);
			weights_A[gi] = uint(p >= TA[0]) + uint(p >= TA[1]) + uint(p >= TA[2]);
		}

		// SSE: per pixel, bilinearly interpolate the grid weights to predict
		// the decoder's actual reconstruction, then accumulate squared
		// difference vs the ideal weight (pixel_proj_norm).
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
				float err = pixel_proj_norm[sy * 8u + sx] - w_i;
				sse_A += err * err;
			}
		}
	}

	///////////////////////////////////////////////////////////////////////
	// Mode B: 4x4 + 3-bit
	///////////////////////////////////////////////////////////////////////
	uint  weights_B[16];
	float sse_B = 0.0;
	{
		// 3-bit thresholds in projection space.
		float TB[7];
		[unroll] for (uint ti = 0; ti < 7; ti++) {
			TB[ti] = axis_len_sq * UNQ_R8_T[ti] + c0_proj;
		}

		// Voronoi-bin pixel projections into the 16 grid cells (each cell
		// covers a 2x2 source-pixel patch — counts are uniform = 4).
		float grid_proj[16];
		[unroll] for (uint i = 0; i < 16; i++) grid_proj[i] = 0;
		[unroll] for (uint py = 0; py < 8; py++) {
			[unroll] for (uint px = 0; px < 8; px++) {
				grid_proj[B_PIX_TO_GRID[py] * 4u + B_PIX_TO_GRID[px]] += pixel_proj[py * 8u + px];
			}
		}
		[unroll] for (uint gi = 0; gi < 16; gi++) {
			float p = grid_proj[gi] * 0.25;  // /4 (uniform cell size)
			uint q = uint(p >= TB[0]) + uint(p >= TB[1]) + uint(p >= TB[2])
			       + uint(p >= TB[3]) + uint(p >= TB[4]) + uint(p >= TB[5])
			       + uint(p >= TB[6]);
			weights_B[gi] = q;
		}

		[unroll] for (uint sy = 0; sy < 8; sy++) {
			[unroll] for (uint sx = 0; sx < 8; sx++) {
				uint  jx  = B_BL_J[sx]; uint jx1 = min(jx + 1u, 3u);
				uint  jy  = B_BL_J[sy]; uint jy1 = min(jy + 1u, 3u);
				float fx  = float(B_BL_W[sx]) * (1.0 / 16.0);
				float fy  = float(B_BL_W[sy]) * (1.0 / 16.0);
				float w00 = UNQ_R8[weights_B[jy  * 4u + jx ]];
				float w10 = UNQ_R8[weights_B[jy  * 4u + jx1]];
				float w01 = UNQ_R8[weights_B[jy1 * 4u + jx ]];
				float w11 = UNQ_R8[weights_B[jy1 * 4u + jx1]];
				float w_i = lerp(lerp(w00, w10, fx), lerp(w01, w11, fx), fy);
				float err = pixel_proj_norm[sy * 8u + sx] - w_i;
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
		astc_write_header_8x8_hdr_rgb_4x4(block);
		astc_write_endpoints_v6          (block, v0, v1, v2, v3, v4, v5);
		[unroll] for (uint wi = 0; wi < 16; wi++) {
			astc_write_weight_3bit(block, wi, weights_B[wi]);
		}
	}
	output_blocks[buffer_offset + by * blocks_x + bx] = block;
}
