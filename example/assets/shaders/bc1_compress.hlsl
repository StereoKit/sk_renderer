//--name = bc1_compress

// BC1 (DXT1) GPU texture compression
//
// Each thread compresses one 4x4 block. Output is an RWStructuredBuffer of
// uint2 (8 bytes per block), laid out in row-major block order.
//
// Port of the CPU bounding-box encoder from tools/tex_compress.c.

Texture2D<float4>         source_tex    : register(t0);
RWStructuredBuffer<uint2> output_blocks : register(u1);

// Pipeline-specialized: the C side creates one compute per value, so the
// branches below fold away at pipeline-compile time.
[[vk::constant_id(0)]] const bool ENABLE_ALPHA = false;  // false = opaque only (4-color), true = punch-through alpha
// Encode linear-light input to sRGB before quantizing, for use with float or
// sRGB-view sources (which Load as linear). BC1's 5:6:5 endpoints band badly
// in the darks if fed linear values; quantizing in sRGB space matches how the
// hardware decodes a *_srgb output format back to linear at sample time.
[[vk::constant_id(1)]] const bool SRGB_ENCODE = false;

float3 linear_to_srgb(float3 c) {
	// IEC 61966-2-1 transfer function
	c = max(c, 0.0);
	float3 lo = c * 12.92;
	float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
	return lerp(lo, hi, step(0.0031308, c));
}

// Round-to-nearest RGB565 from a [0,1] color. Rounding directly in 5/6-bit
// space halves the worst-case endpoint error vs the old 8-bit-round-then-
// truncate path (max ~4/255 vs ~7/255 per channel).
uint pack565(float3 c) {
	uint r = uint(saturate(c.r) * 31.0 + 0.5);
	uint g = uint(saturate(c.g) * 63.0 + 0.5);
	uint b = uint(saturate(c.b) * 31.0 + 0.5);
	return (r << 11) | (g << 5) | b;
}

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

///////////////////////////////////////////////////////////////////////////////
// BC1 block encoder
///////////////////////////////////////////////////////////////////////////////

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint bx = id.x;
	uint by = id.y;
	uint blocks_y = (image_height + 3) / 4;
	if (bx >= blocks_x || by >= blocks_y)
		return;

	// Load 4x4 block.
	float3 pixels[16];

	// Pack transparency into a bitmask instead of bool[16] to save VGPRs
	uint trans_mask = 0;

	[unroll] for (uint py = 0; py < 4; py++) {
		[unroll] for (uint px = 0; px < 4; px++) {
			uint sx = min(bx * 4 + px, image_width  - 1);
			uint sy = min(by * 4 + py, image_height - 1);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			if (SRGB_ENCODE)
				texel.rgb = linear_to_srgb(texel.rgb);

			uint idx = py * 4 + px;
			pixels[idx] = texel.rgb;

			if (ENABLE_ALPHA && texel.a < 0.5)
				trans_mask |= (1u << idx);
		}
	}

	bool has_transparent = (trans_mask != 0);
	bool has_opaque      = (trans_mask != 0xFFFF);

	// Alpha path: fully transparent block early-out
	if (ENABLE_ALPHA && !has_opaque) {
		output_blocks[buffer_offset + by * blocks_x + bx] = uint2(0, 0xFFFFFFFF);
		return;
	}

	// FPS seed: 2 passes of "find pixel most distant from current anchor,
	// anchor = that pixel" — converges on the block's diametric real pixel
	// pair, a better color axis than bbox corners on anti-correlated content.
	// Transparent pixels can't seed endpoints (their color is invisible).
	float3 fps_a = pixels[0];
	float3 fps_b = fps_a;
	[unroll] for (uint iter = 0; iter < 2u; iter++) {
		float  max_d2 = -1.0;
		float3 far_p  = fps_a;
		[unroll] for (uint fi = 0; fi < 16; fi++) {
			float3 diff = pixels[fi] - fps_a;
			float  d2   = dot(diff, diff);
			if (ENABLE_ALPHA && ((trans_mask >> fi) & 1u))
				d2 = -2.0;
			far_p  = d2 > max_d2 ? pixels[fi] : far_p;
			max_d2 = max(max_d2, d2);
		}
		fps_b = fps_a;
		fps_a = far_p;
	}

	// Round-to-nearest 565 endpoints; mode ordering is normalized inside the
	// refinement loop below.
	uint c0 = pack565(fps_a);
	uint c1 = pack565(fps_b);

	bool alpha_mode = ENABLE_ALPHA && has_transparent;
	uint indices    = 0;

	// LS_ROUNDS == 1 is the plain bounding-box encoder; 2 adds one
	// least-squares refinement pass (see below).
	#define LS_ROUNDS 2

	// Two rounds: round 0 assigns indices from the bounding-box endpoints and
	// accumulates least-squares sums; between rounds we solve for the
	// endpoints that minimize squared error for that assignment, requantize
	// to 565, and round 1 re-assigns indices against them. The bbox diagonal
	// is a poor color axis in smooth gradients — this removes most of the
	// banding it causes, for one extra index pass and a 2x2 solve.
	//
	// Each round's actual reconstruction SSE is tracked and the best round
	// wins: LS is optimal for round 0's assignment, but after requantization
	// and re-assignment it can lose range on noisy blocks (photos), where the
	// bbox endpoints — which exactly span the block — were already better.
	uint  best_indices = 0;
	uint  best_c0 = 0, best_c1 = 0;
	float best_sse = 1e30;

	[unroll] for (uint round = 0; round < LS_ROUNDS; round++) {
		// Mode ordering: 3-color + alpha needs c0 <= c1, 4-color opaque needs c0 > c1
		if (alpha_mode) {
			if (c0 > c1) { uint t = c0; c0 = c1; c1 = t; }
			if (c0 == c1 && c0 > 0) c0--;
		} else {
			if (c0 < c1) { uint t = c0; c0 = c1; c1 = t; }
			if (c0 == c1 && c0 < 0xFFFF) c0++;
		}

		// Expand endpoints back to RGB888 via bit replication
		uint3 color0, color1;
		{
			uint r0 = (c0 >> 11) & 0x1F; uint g0 = (c0 >> 5) & 0x3F; uint b0 = c0 & 0x1F;
			uint r1 = (c1 >> 11) & 0x1F; uint g1 = (c1 >> 5) & 0x3F; uint b1 = c1 & 0x1F;
			color0 = uint3((r0 << 3) | (r0 >> 2), (g0 << 2) | (g0 >> 4), (b0 << 3) | (b0 >> 2));
			color1 = uint3((r1 << 3) | (r1 >> 2), (g1 << 2) | (g1 >> 4), (b1 << 3) | (b1 >> 2));
		}

		// Perceptually weighted axis in float space: avoids per-pixel float->uint
		// conversion and replaces 4-cycle v_mul_lo_u32 with 1-cycle v_fma_f32.
		float3 faxis = float3(
			float(int(color1.r) - int(color0.r)) * 2.0,
			float(int(color1.g) - int(color0.g)) * 4.0,
			float(int(color1.b) - int(color0.b)));

		// axis_len_sq with matching weights, scaled by 1/255 to match [0,1] pixel space.
		// Projection gives proj_int/255, so thresholds must also be /255.
		float faxis_len_sq = (faxis.r * faxis.r * 0.5 + faxis.g * faxis.g * 0.25 + faxis.b * faxis.b) / 255.0;
		float fc0_proj     = dot(float3(color0) / 255.0, faxis);

		indices = 0;

		if (faxis_len_sq < 1e-6) {
			// Degenerate: all same color — everything index 0, except transparent
			// pixels which need index 3. Indices are 2 bits per pixel, so spread
			// each trans_mask bit j out to bit 2j, then *3 fills both bits.
			// Nothing for least squares to refine here. Only reachable in
			// round 0 for a genuinely flat block (refined endpoints of a
			// non-flat block can't collapse); keep it and stop.
			if (alpha_mode) {
				uint spread = trans_mask;
				spread = (spread | (spread << 8)) & 0x00FF00FFu;
				spread = (spread | (spread << 4)) & 0x0F0F0F0Fu;
				spread = (spread | (spread << 2)) & 0x33333333u;
				spread = (spread | (spread << 1)) & 0x55555555u;
				indices = spread * 3u;
			}
			if (round == 0) { best_indices = indices; best_c0 = c0; best_c1 = c1; }
			break;
		}

		float3 color0f = float3(color0) / 255.0;
		float3 color1f = float3(color1) / 255.0;
		float  sse     = 0;

		// Least-squares accumulators: reconstruction is e0*(1-w) + e1*w with
		// w the bucket's position along the c0->c1 axis.
		float  sum_aa = 0, sum_ab = 0, sum_bb = 0;
		float3 sum_ap = float3(0, 0, 0);
		float3 sum_bp = float3(0, 0, 0);

		if (alpha_mode) {
			// 3-color + alpha mode
			// Original: (dot(p, faxis) - fc0_proj) * 4 >= faxis_len_sq * N
			// Folded:    dot(p, faxis) >= faxis_len_sq * N / 4 + fc0_proj
			float t1 = faxis_len_sq * 0.25 + fc0_proj;
			float t3 = faxis_len_sq * 0.75 + fc0_proj;

			[unroll] for (uint j = 0; j < 16; j++) {
				float d = dot(pixels[j], faxis);
				uint bucket = uint(d >= t1) + uint(d >= t3);
				uint color_idx = (0x18u >> (bucket * 2)) & 3u;
				uint is_trans = (trans_mask >> j) & 1u;
				uint final_idx = is_trans ? 3u : color_idx;
				indices |= (final_idx << (j * 2));

				// Transparent pixels don't constrain the endpoints
				float opq = is_trans ? 0.0 : 1.0;
				float w = float(bucket) * 0.5;
				float a = (1.0 - w) * opq;
				float b = w * opq;
				sum_aa += a * a; sum_ab += a * b; sum_bb += b * b;
				sum_ap += a * pixels[j]; sum_bp += b * pixels[j];

				float3 diff = (pixels[j] - lerp(color0f, color1f, w)) * opq;
				sse += dot(diff, diff);
			}
		} else {
			// 4-color opaque mode
			// Original: (dot(p, faxis) - fc0_proj) * 6 >= faxis_len_sq * N
			// Folded:    dot(p, faxis) >= faxis_len_sq * N / 6 + fc0_proj
			float rcp6 = 1.0 / 6.0;
			float t1 = faxis_len_sq       * rcp6 + fc0_proj;
			float t3 = faxis_len_sq * 3.0 * rcp6 + fc0_proj;
			float t5 = faxis_len_sq * 5.0 * rcp6 + fc0_proj;

			[unroll] for (uint j = 0; j < 16; j++) {
				float d = dot(pixels[j], faxis);
				uint bucket = uint(d >= t1) + uint(d >= t3) + uint(d >= t5);
				indices |= (((0x78u >> (bucket * 2)) & 3u) << (j * 2));

				float w = float(bucket) * (1.0 / 3.0);
				float a = 1.0 - w;
				sum_aa += a * a; sum_ab += a * w; sum_bb += w * w;
				sum_ap += a * pixels[j]; sum_bp += w * pixels[j];

				float3 diff = pixels[j] - lerp(color0f, color1f, w);
				sse += dot(diff, diff);
			}
		}

		if (sse < best_sse) {
			best_sse     = sse;
			best_indices = indices;
			best_c0      = c0;
			best_c1      = c1;
		}

		if (round == LS_ROUNDS - 1) break;

		// Solve the 2x2 normal equations. A near-singular system means the
		// pixels all landed in one bucket — keep the bbox endpoints.
		float det = sum_aa * sum_bb - sum_ab * sum_ab;
		if (abs(det) < 1e-8) break;
		float  inv_det = 1.0 / det;
		float3 ep0 = saturate((sum_ap * sum_bb - sum_bp * sum_ab) * inv_det);
		float3 ep1 = saturate((sum_bp * sum_aa - sum_ap * sum_ab) * inv_det);

		c0 = pack565(ep0);
		c1 = pack565(ep1);
	}

	// Pack output: [c0_lo, c0_hi, c1_lo, c1_hi] [idx0..idx3]
	output_blocks[buffer_offset + by * blocks_x + bx] = uint2(best_c0 | (best_c1 << 16), best_indices);
}
