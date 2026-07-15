//--name = bc7_compress

// BC7 GPU texture compression — mode 6 only.
//
// Each thread compresses one 4x4 block into 128 bits: 7-bit mode field
// (six zeros then a 1), two RGBA endpoints at 7 bits/channel plus one
// shared p-bit each (effectively 8-bit endpoints), and 63 bits of 4-bit
// indices (pixel 0's MSB is implicit-zero — the anchor rule). One subset,
// one index per pixel shared by all four channels — the real-time design
// point: smooth RGBA content compresses ~6-10 dB better than BC1 at 2x
// the bits, with real interpolated alpha instead of punch-through.
//
// Encoder skeleton mirrors bc1_compress.hlsl lifted to float4: FPS 2-pass
// endpoint seeding, LS_ROUNDS of index-assign + least-squares endpoint
// refinement with best-of-rounds SSE tracking. The color axis uses the
// same perceptual channel weights as BC1 (R2 G4 B1) plus A2 (matching the
// ASTC 4x4 RGBA mode); LS solve and SSE stay unweighted.
//
// Endpoint quantization: e8 = (e7 << 1) | p, exact 8-bit when p lands
// right. The p-bit is shared across a endpoint's four channels, so it is
// chosen per endpoint by trying both values and keeping the lower total
// squared endpoint error.
//
// v1 limitations (deliberate, same flavor as BC6H's mode-11-only):
// sharp alpha edges would prefer mode 5's separate alpha indices, and
// hard color edges would prefer mode 1/3 partitions. Measure first.

Texture2D<float4>         source_tex    : register(t0);
RWStructuredBuffer<uint4> output_blocks : register(u1);

// Encode linear-light input to sRGB before quantizing, for cube sources
// (float or sRGB-view textures Load as linear). Same rationale as BC1's
// variant: quantize in the space the *_srgb output format decodes from.
[[vk::constant_id(0)]] const bool SRGB_ENCODE = false;

float3 linear_to_srgb(float3 c) {
	c = max(c, 0.0);
	float3 lo = c * 12.92;
	float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
	return lerp(lo, hi, step(0.0031308, c));
}

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

///////////////////////////////////////////////////////////////////////////////
// Mode 6 quantization + packing
///////////////////////////////////////////////////////////////////////////////

// BC7 4-bit index weights in 1/64ths (same table as BC6H).
static const float W4[16] = {
	 0.0 / 64.0,  4.0 / 64.0,  9.0 / 64.0, 13.0 / 64.0,
	17.0 / 64.0, 21.0 / 64.0, 26.0 / 64.0, 30.0 / 64.0,
	34.0 / 64.0, 38.0 / 64.0, 43.0 / 64.0, 47.0 / 64.0,
	51.0 / 64.0, 55.0 / 64.0, 60.0 / 64.0, 64.0 / 64.0,
};

// Quantize a [0,255] endpoint to 7 bits + shared p-bit. p is chosen per
// endpoint (both candidates evaluated); this returns e7 for a given p.
uint4 quantize7(float4 target, uint p) {
	// e8 = e7*2 + p, so the nearest e7 for target t is round((t - p) / 2).
	return uint4(clamp((target - float(p)) * 0.5 + 0.5, 0.0, 127.0));
}

float4 dequant7(uint4 e7, uint p) {
	return float4((e7 << 1) | p);
}

// OR count bits of value into the 128-bit block at a bit offset (LSB-first).
void write_bits(inout uint4 b, uint offset, uint count, uint value) {
	value &= (1u << count) - 1u;
	uint word = offset >> 5u;
	uint bit  = offset & 31u;
	b[word] |= value << bit;
	if (bit + count > 32u)
		b[word + 1u] |= value >> (32u - bit);
}

///////////////////////////////////////////////////////////////////////////////
// BC7 mode 6 block encoder
///////////////////////////////////////////////////////////////////////////////

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint bx = id.x;
	uint by = id.y;
	uint blocks_y = (image_height + 3) / 4;
	if (bx >= blocks_x || by >= blocks_y)
		return;

	// Load 4x4 block in [0,255] float4 — the space mode 6 endpoints live in.
	float4 pixels[16];
	[unroll] for (uint py = 0; py < 4; py++) {
		[unroll] for (uint px = 0; px < 4; px++) {
			uint sx = min(bx * 4 + px, image_width  - 1);
			uint sy = min(by * 4 + py, image_height - 1);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			if (SRGB_ENCODE)
				texel.rgb = linear_to_srgb(texel.rgb);
			pixels[py * 4 + px] = saturate(texel) * 255.0;
		}
	}

	// FPS seed: 2 passes of "find pixel most distant from current anchor,
	// anchor = that pixel" — converges on the block's diametric real pixel
	// pair, a better color axis than bbox corners on anti-correlated content.
	float4 fps_a = pixels[0];
	float4 fps_b = fps_a;
	[unroll] for (uint iter = 0; iter < 2u; iter++) {
		float  max_d2 = -1.0;
		float4 far_p  = fps_a;
		[unroll] for (uint fi = 0; fi < 16; fi++) {
			float4 diff = pixels[fi] - fps_a;
			float  d2   = dot(diff, diff);
			far_p  = d2 > max_d2 ? pixels[fi] : far_p;
			max_d2 = max(max_d2, d2);
		}
		fps_b = fps_a;
		fps_a = far_p;
	}

	// Choose each endpoint's p-bit once from the seed targets, then keep it
	// fixed through the LS rounds — the refined endpoints stay in the same
	// neighborhood, and a stable p keeps rounds comparable.
	uint p0, p1;
	{
		float4 d0a = fps_b - dequant7(quantize7(fps_b, 0u), 0u);
		float4 d0b = fps_b - dequant7(quantize7(fps_b, 1u), 1u);
		p0 = dot(d0b, d0b) < dot(d0a, d0a) ? 1u : 0u;
		float4 d1a = fps_a - dequant7(quantize7(fps_a, 0u), 0u);
		float4 d1b = fps_a - dequant7(quantize7(fps_a, 1u), 1u);
		p1 = dot(d1b, d1b) < dot(d1a, d1a) ? 1u : 0u;
	}

	uint4 q0 = quantize7(fps_b, p0);
	uint4 q1 = quantize7(fps_a, p1);

	#define LS_ROUNDS 3

	// Round 0 assigns indices from the seeded endpoints and accumulates
	// least-squares sums; between rounds we solve for the endpoints that
	// minimize squared error for that assignment, requantize, and re-assign.
	// Each round's actual reconstruction SSE is tracked and the best round
	// wins (LS can lose range on noisy blocks after requantization).
	uint4 best_q0 = q0, best_q1 = q1;
	uint  best_p0 = p0, best_p1 = p1;
	uint  best_idx_lo = 0, best_idx_hi = 0;   // 4 bits/pixel: 0-7 in lo, 8-15 in hi
	float best_sse = 1e30;

	[unroll] for (uint round = 0; round < LS_ROUNDS; round++) {
		float4 e0 = dequant7(q0, p0);
		float4 e1 = dequant7(q1, p1);

		// Perceptually weighted projection axis (R2 G4 B1 A2, as BC1 + the
		// ASTC 4x4 RGBA mode); reconstruction SSE below stays unweighted.
		float4 axis   = (e1 - e0) * float4(2.0, 4.0, 1.0, 2.0);
		float  len_sq = dot(e1 - e0, axis);

		if (len_sq < 1e-3) {
			// Degenerate: flat block — all indices 0 reconstruct e0 exactly.
			// Only reachable in round 0; keep the seed endpoints and stop.
			if (round == 0) {
				best_q0 = q0; best_q1 = q1; best_p0 = p0; best_p1 = p1;
				best_idx_lo = 0; best_idx_hi = 0;
			}
			break;
		}

		float inv_len = 1.0 / len_sq;
		float e0_proj = dot(e0, axis);

		uint  idx_lo = 0, idx_hi = 0;
		float sse    = 0;

		// Least-squares accumulators: reconstruction is e0*(1-w) + e1*w.
		float  sum_aa = 0, sum_ab = 0, sum_bb = 0;
		float4 sum_ap = float4(0, 0, 0, 0);
		float4 sum_bp = float4(0, 0, 0, 0);

		[unroll] for (uint j = 0; j < 16; j++) {
			float wn = saturate((dot(pixels[j], axis) - e0_proj) * inv_len);
			uint  k  = uint(wn * 15.0 + 0.5);
			float w  = W4[k];

			if (j < 8) idx_lo |= k << (j * 4);
			else       idx_hi |= k << ((j - 8) * 4);

			float a = 1.0 - w;
			sum_aa += a * a; sum_ab += a * w; sum_bb += w * w;
			sum_ap += a * pixels[j]; sum_bp += w * pixels[j];

			float4 diff = pixels[j] - lerp(e0, e1, w);
			sse += dot(diff, diff);
		}

		if (sse < best_sse) {
			best_sse    = sse;
			best_q0     = q0;  best_q1 = q1;
			best_p0     = p0;  best_p1 = p1;
			best_idx_lo = idx_lo;
			best_idx_hi = idx_hi;
		}

		if (round == LS_ROUNDS - 1) break;

		// Solve the 2x2 normal equations. Near-singular means every pixel
		// landed in one bucket — keep the current endpoints.
		float det = sum_aa * sum_bb - sum_ab * sum_ab;
		if (abs(det) < 1e-8) break;
		float  inv_det = 1.0 / det;
		float4 ep0 = clamp((sum_ap * sum_bb - sum_bp * sum_ab) * inv_det, 0.0, 255.0);
		float4 ep1 = clamp((sum_bp * sum_aa - sum_ap * sum_ab) * inv_det, 0.0, 255.0);

		q0 = quantize7(ep0, p0);
		q1 = quantize7(ep1, p1);
	}

	// Anchor rule: pixel 0 stores only 3 index bits, MSB implied zero. If its
	// index has the MSB set, swap the endpoints (with their p-bits) and
	// complement every index (k -> 15-k == XOR the whole nibble array).
	if ((best_idx_lo & 0x8u) != 0u) {
		uint4 tq = best_q0; best_q0 = best_q1; best_q1 = tq;
		uint  tp = best_p0; best_p0 = best_p1; best_p1 = tp;
		best_idx_lo ^= 0xFFFFFFFFu;
		best_idx_hi ^= 0xFFFFFFFFu;
	}

	// Pack mode 6: mode field 0000001 (bit 6 set), then R0 R1 G0 G1 B0 B1
	// A0 A1 (7 bits each), P0 P1, then 63 index bits (pixel 0 is 3 bits).
	uint4 block = uint4(0, 0, 0, 0);
	write_bits(block,  0, 7, 0x40u);
	write_bits(block,  7, 7, best_q0.r);
	write_bits(block, 14, 7, best_q1.r);
	write_bits(block, 21, 7, best_q0.g);
	write_bits(block, 28, 7, best_q1.g);
	write_bits(block, 35, 7, best_q0.b);
	write_bits(block, 42, 7, best_q1.b);
	write_bits(block, 49, 7, best_q0.a);
	write_bits(block, 56, 7, best_q1.a);
	write_bits(block, 63, 1, best_p0);
	write_bits(block, 64, 1, best_p1);
	write_bits(block, 65, 3, best_idx_lo & 0x7u);
	[unroll] for (uint j = 1; j < 16; j++) {
		uint k = (j < 8) ? ((best_idx_lo >> (j * 4)) & 0xFu)
		                 : ((best_idx_hi >> ((j - 8) * 4)) & 0xFu);
		write_bits(block, 68 + (j - 1) * 4, 4, k);
	}

	output_blocks[buffer_offset + by * blocks_x + bx] = block;
}
