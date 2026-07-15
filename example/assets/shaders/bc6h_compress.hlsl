//--name = bc6h_compress

// BC6H (UF16) GPU texture compression — mode 11 only.
//
// Each thread compresses one 4x4 block into 128 bits: 5-bit mode (00011),
// two absolute 10-bit-per-channel RGB endpoints, 63 bits of 4-bit indices
// (pixel 0's MSB is implicit-zero — the anchor rule). One subset, no
// partitions, no delta endpoints — the real-time design point: skybox/IBL
// content is smooth, and mode 11 has no delta-representability fallbacks.
// See docs/BC6H_format.md for the full format writeup.
//
// Working space: the decoder interpolates (an unquantization of) raw half-
// float bit patterns, so the encoder converts pixels with f32tof16() and does
// ALL math — seeding, projection, least squares, SSE — on those integers as
// floats [0, 31743]. That space is piecewise-linear in log2(value); SSE here
// is a perceptually reasonable HDR metric for free, and it's exactly the
// metric the decoder's lerp realizes.
//
// Mode 11 decode chain (interior codes), which quantize10/dequant10 invert:
//   unq(q)  = (q << 6) + 32            10-bit -> 17-bit work value
//   half(u) = (u * 31) >> 6            so half(q) = 31*q + 15
// with q=0 pinned to 0 and q=1023 pinned to 0x7BFF (65504.0, max finite
// half) — BC6H cannot emit Inf/NaN by construction.
//
// Encoder skeleton mirrors bc1_compress.hlsl: FPS 2-pass endpoint seeding,
// LS_ROUNDS of index-assign + least-squares endpoint refinement with
// best-of-rounds SSE tracking. Channels are unweighted: half-bit space is
// already log-ish, which does the perceptual work 565-space needed explicit
// weights for.
//
// Source must be a float-format texture (RGBA16/RGBA32 float). Alpha is
// ignored (BC6H has none; samples as 1.0). Negative inputs clamp to 0
// (UF16 is unsigned).

Texture2D<float4>         source_tex    : register(t0);
RWStructuredBuffer<uint4> output_blocks : register(u1);

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

///////////////////////////////////////////////////////////////////////////////
// Mode 11 quantization
///////////////////////////////////////////////////////////////////////////////

// Nearest 10-bit code for a half-bit value h in [0, 31743]. Interior decode
// is 31*q + 15, so round((h-15)/31) = (2h+1)/62; both ends are special-cased
// because codes 0 and 1023 are pinned (decode to 0 and 31743, not the 31q+15
// line): the 0/1 midpoint is 23 and the 1022/1023 midpoint is 31720.
// Verified nearest for all 31744 half codes
// (tools/compress/validate/bc6h_check_quant10.py).
uint quantize10(float hf) {
	uint h = uint(hf + 0.5);
	uint q = (h * 2u + 1u) / 62u;
	if (q == 0u && h > 22u) q = 1u;
	q = min(q, 1023u);
	if (q == 1023u && h < 31720u) q = 1022u;
	return q;
}

float dequant10(uint q) {
	return q == 0u ? 0.0 : (q == 1023u ? 31743.0 : float(31u * q + 15u));
}

float3 dequant10_3(uint3 q) {
	return float3(dequant10(q.x), dequant10(q.y), dequant10(q.z));
}

uint3 quantize10_3(float3 h) {
	return uint3(quantize10(h.x), quantize10(h.y), quantize10(h.z));
}

///////////////////////////////////////////////////////////////////////////////
// Bit packing
///////////////////////////////////////////////////////////////////////////////

// BC6H 4-bit index weights in 1/64ths — rounded k*64/15 so indices 0 and 15
// reproduce the endpoints exactly.
static const float W4[16] = {
	 0.0 / 64.0,  4.0 / 64.0,  9.0 / 64.0, 13.0 / 64.0,
	17.0 / 64.0, 21.0 / 64.0, 26.0 / 64.0, 30.0 / 64.0,
	34.0 / 64.0, 38.0 / 64.0, 43.0 / 64.0, 47.0 / 64.0,
	51.0 / 64.0, 55.0 / 64.0, 60.0 / 64.0, 64.0 / 64.0,
};

// OR count bits of value into the 128-bit block at a bit offset (LSB-first,
// matching the D3D bitstream). count <= 24 so a value spans at most 2 dwords.
void write_bits(inout uint4 b, uint offset, uint count, uint value) {
	value &= (1u << count) - 1u;
	uint word = offset >> 5u;
	uint bit  = offset & 31u;
	b[word] |= value << bit;
	if (bit + count > 32u)
		b[word + 1u] |= value >> (32u - bit);
}

///////////////////////////////////////////////////////////////////////////////
// BC6H mode 11 block encoder
///////////////////////////////////////////////////////////////////////////////

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint bx = id.x;
	uint by = id.y;
	uint blocks_y = (image_height + 3) / 4;
	if (bx >= blocks_x || by >= blocks_y)
		return;

	// Load 4x4 block as half-bit integers (see working-space note above).
	// min() against 0x7BFF turns +Inf (0x7C00) and NaN payloads into the max
	// finite half instead of leaking exponent-field garbage into the math.
	float3 pixels[16];
	[unroll] for (uint py = 0; py < 4; py++) {
		[unroll] for (uint px = 0; px < 4; px++) {
			uint sx = min(bx * 4 + px, image_width  - 1);
			uint sy = min(by * 4 + py, image_height - 1);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			uint3  hbits = min(f32tof16(max(texel.rgb, 0.0)), 0x7BFFu);
			pixels[py * 4 + px] = float3(hbits);
		}
	}

	// FPS seed: 2 passes of "find pixel most distant from current anchor,
	// anchor = that pixel" — converges on the block's diametric real pixel
	// pair, a better color axis than bbox corners on anti-correlated content.
	float3 fps_a = pixels[0];
	float3 fps_b = fps_a;
	[unroll] for (uint iter = 0; iter < 2u; iter++) {
		float  max_d2 = -1.0;
		float3 far_p  = fps_a;
		[unroll] for (uint fi = 0; fi < 16; fi++) {
			float3 diff = pixels[fi] - fps_a;
			float  d2   = dot(diff, diff);
			far_p  = d2 > max_d2 ? pixels[fi] : far_p;
			max_d2 = max(max_d2, d2);
		}
		fps_b = fps_a;
		fps_a = far_p;
	}

	uint3 q0 = quantize10_3(fps_b);
	uint3 q1 = quantize10_3(fps_a);

	#define LS_ROUNDS 3

	// Round 0 assigns indices from the seeded endpoints and accumulates
	// least-squares sums; between rounds we solve for the endpoints that
	// minimize squared error for that assignment, requantize, and re-assign.
	// Each round's actual reconstruction SSE is tracked and the best round
	// wins (LS can lose range on noisy blocks after requantization).
	uint3 best_q0 = q0, best_q1 = q1;
	uint  best_idx_lo = 0, best_idx_hi = 0;   // 4 bits/pixel: 0-7 in lo, 8-15 in hi
	float best_sse = 1e30;

	[unroll] for (uint round = 0; round < LS_ROUNDS; round++) {
		float3 e0 = dequant10_3(q0);
		float3 e1 = dequant10_3(q1);

		float3 axis        = e1 - e0;
		float  axis_len_sq = dot(axis, axis);

		if (axis_len_sq < 1e-3) {
			// Degenerate: flat block — all indices 0 reconstruct e0 exactly.
			// Only reachable in round 0 (LS of a non-flat block can't
			// collapse); keep the seed endpoints and stop.
			if (round == 0) { best_q0 = q0; best_q1 = q1; best_idx_lo = 0; best_idx_hi = 0; }
			break;
		}

		float inv_len = 1.0 / axis_len_sq;
		float e0_proj = dot(e0, axis);

		uint  idx_lo = 0, idx_hi = 0;
		float sse    = 0;

		// Least-squares accumulators: reconstruction is e0*(1-w) + e1*w.
		float  sum_aa = 0, sum_ab = 0, sum_bb = 0;
		float3 sum_ap = float3(0, 0, 0);
		float3 sum_bp = float3(0, 0, 0);

		[unroll] for (uint j = 0; j < 16; j++) {
			float wn = saturate((dot(pixels[j], axis) - e0_proj) * inv_len);
			// Uniform nearest-of-15 is within half a weight step of exact
			// nearest in the W4 table; the SSE below uses the true table
			// value, so any misassignment costs (and can lose) honestly.
			uint  k  = uint(wn * 15.0 + 0.5);
			float w  = W4[k];

			if (j < 8) idx_lo |= k << (j * 4);
			else       idx_hi |= k << ((j - 8) * 4);

			float a = 1.0 - w;
			sum_aa += a * a; sum_ab += a * w; sum_bb += w * w;
			sum_ap += a * pixels[j]; sum_bp += w * pixels[j];

			float3 diff = pixels[j] - lerp(e0, e1, w);
			sse += dot(diff, diff);
		}

		if (sse < best_sse) {
			best_sse    = sse;
			best_q0     = q0;
			best_q1     = q1;
			best_idx_lo = idx_lo;
			best_idx_hi = idx_hi;
		}

		if (round == LS_ROUNDS - 1) break;

		// Solve the 2x2 normal equations. Near-singular means every pixel
		// landed in one bucket — keep the current endpoints.
		float det = sum_aa * sum_bb - sum_ab * sum_ab;
		if (abs(det) < 1e-8) break;
		float  inv_det = 1.0 / det;
		float3 ep0 = clamp((sum_ap * sum_bb - sum_bp * sum_ab) * inv_det, 0.0, 31743.0);
		float3 ep1 = clamp((sum_bp * sum_aa - sum_ap * sum_ab) * inv_det, 0.0, 31743.0);

		q0 = quantize10_3(ep0);
		q1 = quantize10_3(ep1);
	}

	// Anchor rule: pixel 0 stores only 3 index bits, MSB implied zero. If its
	// index has the MSB set, swap the endpoints and complement every index
	// (k -> 15-k == k ^ 0xF, i.e. XOR the whole nibble array).
	if ((best_idx_lo & 0x8u) != 0u) {
		uint3 t = best_q0; best_q0 = best_q1; best_q1 = t;
		best_idx_lo ^= 0xFFFFFFFFu;
		best_idx_hi ^= 0xFFFFFFFFu;
	}

	// Pack mode 11: m[4:0]=00011, rw gw bw rx gx bx (10 bits each), then
	// 63 index bits — pixel 0 is 3 bits, pixels 1..15 are 4 bits.
	uint4 block = uint4(0, 0, 0, 0);
	write_bits(block,  0,  5, 0x03u);
	write_bits(block,  5, 10, best_q0.x);
	write_bits(block, 15, 10, best_q0.y);
	write_bits(block, 25, 10, best_q0.z);
	write_bits(block, 35, 10, best_q1.x);
	write_bits(block, 45, 10, best_q1.y);
	write_bits(block, 55, 10, best_q1.z);
	write_bits(block, 65,  3, best_idx_lo & 0x7u);
	[unroll] for (uint j = 1; j < 16; j++) {
		uint k = (j < 8) ? ((best_idx_lo >> (j * 4)) & 0xFu)
		                 : ((best_idx_hi >> ((j - 8) * 4)) & 0xFu);
		write_bits(block, 68 + (j - 1) * 4, 4, k);
	}

	output_blocks[buffer_offset + by * blocks_x + bx] = block;
}
