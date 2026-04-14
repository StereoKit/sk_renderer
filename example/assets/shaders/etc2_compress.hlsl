//--name = etc2_compress

// ETC2 RGB8 GPU texture compression
//
// Each thread compresses one 4x4 block. Output is an RWStructuredBuffer of
// uint2 (8 bytes per block), laid out in row-major block order.
//
// Implements quality mode 3 from the CPU encoder with both flip directions:
//   - Differential mode (RGB555 + delta333) - most common
//   - Individual mode (RGB444 + RGB444) - fallback when delta overflows
//   - Planar mode - for smooth gradients (conservative trigger)

Texture2D<float4>         source_tex    : register(t0);
SamplerState              source_tex_s  : register(s0);
RWStructuredBuffer<uint2> output_blocks : register(u1);

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

// ETC2 blocks are big-endian but GPU memory is little-endian.
uint etc_bswap(uint v) {
	return ((v >> 24) & 0xFFu) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
}

int etc_expand4(int c) { return (c << 4) | c; }
int etc_expand5(int c) { return (c << 3) | (c >> 2); }
int etc_expand6(int c) { return (c << 2) | (c >> 4); }
int etc_expand7(int c) { return (c << 1) | (c >> 6); }
int etc_quantize4(int c8) { return min((c8 + 8) >> 4, 15); }
int etc_quantize5(int c8) { return min((c8 * 31 + 127) / 255, 31); }

void etc_get_mod(int t, out int s, out int l) {
	switch (t) {
		case 0: s= 2; l=  8; break; case 1: s= 5; l= 17; break;
		case 2: s= 9; l= 29; break; case 3: s=13; l= 42; break;
		case 4: s=18; l= 60; break; case 5: s=24; l= 80; break;
		case 6: s=33; l=106; break; default:s=47; l=183; break;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Sub-block encoding
///////////////////////////////////////////////////////////////////////////////

// Float-space distance: v_fma_f32 is 1 cycle vs v_mul_lo_u32 at 4 cycles on RDNA3.
// Precision is sufficient: max squared distance is 3*438^2 = 575208, within float32.
int subblock_error(int3 pixels[8], int3 base, int table_idx, out uint indices[8]) {
	int ms, ml;
	etc_get_mod(table_idx, ms, ml);

	float3 fb  = float3(base);
	float  fms = float(ms);
	float  fml = float(ml);
	float3 fc0 = clamp(fb + fms, 0.0, 255.0);
	float3 fc1 = clamp(fb + fml, 0.0, 255.0);
	float3 fc2 = clamp(fb - fms, 0.0, 255.0);
	float3 fc3 = clamp(fb - fml, 0.0, 255.0);

	int total = 0;
	[unroll] for (uint i = 0; i < 8; i++) {
		float3 fp = float3(pixels[i]);
		float3 d0 = fp - fc0; float e0 = d0.r*d0.r + d0.g*d0.g + d0.b*d0.b;
		float3 d1 = fp - fc1; float e1 = d1.r*d1.r + d1.g*d1.g + d1.b*d1.b;
		float3 d2 = fp - fc2; float e2 = d2.r*d2.r + d2.g*d2.g + d2.b*d2.b;
		float3 d3 = fp - fc3; float e3 = d3.r*d3.r + d3.g*d3.g + d3.b*d3.b;

		int best_e = int(e0); uint best_i = 0;
		if (e1 < e0) { best_e = int(e1); best_i = 1; }
		if (e2 < float(best_e)) { best_e = int(e2); best_i = 2; }
		if (e3 < float(best_e)) { best_e = int(e3); best_i = 3; }
		total += best_e;
		indices[i] = best_i;
	}
	return total;
}

// Float abs() is free on GPU (sign bit mask) vs integer sub+max(x,-x).
int encode_subblock(int3 pixels[8], int3 base, out int out_table, out uint out_indices[8]) {
	float3 fb = float3(base);
	float max_dev = 0.0;
	[unroll] for (uint i = 0; i < 8; i++) {
		float3 d = abs(float3(pixels[i]) - fb);
		max_dev = max(max_dev, max(max(d.r, d.g), d.b));
	}

	int t;
	if      (max_dev <=  8.0) t = 0;
	else if (max_dev <= 17.0) t = 1;
	else if (max_dev <= 29.0) t = 2;
	else if (max_dev <= 42.0) t = 3;
	else if (max_dev <= 60.0) t = 4;
	else if (max_dev <= 80.0) t = 5;
	else if (max_dev <=106.0) t = 6;
	else                      t = 7;

	out_table = t;
	return subblock_error(pixels, base, t, out_indices);
}

///////////////////////////////////////////////////////////////////////////////
// Index packing (column-major per ETC spec, big-endian)
///////////////////////////////////////////////////////////////////////////////

void pack_subblock_indices(uint i0[8], uint i1[8], int flip,
                           out uint out_msb, out uint out_lsb) {
	uint msb = 0, lsb = 0;
	[unroll] for (uint i = 0; i < 8; i++) {
		int x = flip ? (int(i) % 4) : (int(i) % 2);
		int y = flip ? (int(i) / 4) : (int(i) / 2);
		int b = x * 4 + y;
		msb |= ((i0[i] >> 1) & 1u) << b;
		lsb |= ((i0[i] >> 0) & 1u) << b;
	}
	[unroll] for (uint j = 0; j < 8; j++) {
		int x = flip ? (int(j) % 4) : (2 + int(j) % 2);
		int y = flip ? (2 + int(j) / 4) : (int(j) / 2);
		int b = x * 4 + y;
		msb |= ((i1[j] >> 1) & 1u) << b;
		lsb |= ((i1[j] >> 0) & 1u) << b;
	}
	out_msb = msb;
	out_lsb = lsb;
}

///////////////////////////////////////////////////////////////////////////////
// T/H mode shared: block error with 4 arbitrary paint colors (16 pixels)
///////////////////////////////////////////////////////////////////////////////

int etc_th_dist_val(int di) {
	switch(di) {
		case 0: return 3;  case 1: return 6;  case 2: return 11; case 3: return 16;
		case 4: return 23; case 5: return 32; case 6: return 41; default: return 64;
	}
}

int th_block_error(int3 pixels[16], int3 paint[4], out uint indices[16]) {
	int total = 0;
	[unroll] for (uint i = 0; i < 16; i++) {
		float3 fp = float3(pixels[i]);
		float3 d0 = fp - float3(paint[0]); float e0 = d0.r*d0.r + d0.g*d0.g + d0.b*d0.b;
		float3 d1 = fp - float3(paint[1]); float e1 = d1.r*d1.r + d1.g*d1.g + d1.b*d1.b;
		float3 d2 = fp - float3(paint[2]); float e2 = d2.r*d2.r + d2.g*d2.g + d2.b*d2.b;
		float3 d3 = fp - float3(paint[3]); float e3 = d3.r*d3.r + d3.g*d3.g + d3.b*d3.b;

		int best_e = int(e0); uint best_i = 0;
		if (e1 < e0)           { best_e = int(e1); best_i = 1; }
		if (e2 < float(best_e)) { best_e = int(e2); best_i = 2; }
		if (e3 < float(best_e)) { best_e = int(e3); best_i = 3; }
		total += best_e;
		indices[i] = best_i;
	}
	return total;
}

// Full-block index packing (column-major, for T/H/planar modes)
void pack_indices_16(uint indices[16], out uint out_msb, out uint out_lsb) {
	uint msb = 0, lsb = 0;
	[unroll] for (uint i = 0; i < 16; i++) {
		int x = int(i) % 4;
		int y = int(i) / 4;
		int b = x * 4 + y;
		msb |= ((indices[i] >> 1) & 1u) << b;
		lsb |= ((indices[i] >> 0) & 1u) << b;
	}
	out_msb = msb;
	out_lsb = lsb;
}

///////////////////////////////////////////////////////////////////////////////
// T-mode: paint0=base1, paint1=base2+d, paint2=base2, paint3=base2-d
///////////////////////////////////////////////////////////////////////////////

uint2 try_t_mode(int3 pixels[16], int3 c1, int3 c2, out int out_error) {
	int3 q1 = int3(etc_quantize4(c1.r), etc_quantize4(c1.g), etc_quantize4(c1.b));
	int3 q2 = int3(etc_quantize4(c2.r), etc_quantize4(c2.g), etc_quantize4(c2.b));
	int3 b1 = int3(etc_expand4(q1.r), etc_expand4(q1.g), etc_expand4(q1.b));
	int3 b2 = int3(etc_expand4(q2.r), etc_expand4(q2.g), etc_expand4(q2.b));

	int best_error = 0x7FFFFFFF;
	int best_di = 0;
	uint best_idx[16];

	[unroll] for (int di = 0; di < 8; di++) {
		int d = etc_th_dist_val(di);
		int3 paint[4] = { b1, clamp(b2 + d, 0, 255), b2, clamp(b2 - d, 0, 255) };

		uint idx[16];
		int err = th_block_error(pixels, paint, idx);
		if (err < best_error) {
			best_error = err;
			best_di = di;
			best_idx = idx;
		}
	}

	// Pack T-mode block (triggers R overflow)
	int use_pos = (q1.r == 7 || q1.r == 10 || q1.r == 11 ||
	               q1.r == 13 || q1.r == 14 || q1.r == 15) ? 1 : 0;

	uint hi = 0;
	if (use_pos)  hi |= 7u << 29;
	hi |= uint((q1.r >> 2) & 3) << 27;
	if (!use_pos) hi |= 1u << 26;
	hi |= uint(q1.r & 3) << 24;
	hi |= uint(q1.g & 0xF) << 20;
	hi |= uint(q1.b & 0xF) << 16;
	hi |= uint(q2.r & 0xF) << 12;
	hi |= uint(q2.g & 0xF) << 8;
	hi |= uint(q2.b & 0xF) << 4;
	hi |= uint((best_di >> 1) & 3) << 2;
	hi |= 1u << 1;
	hi |= uint(best_di & 1);

	uint msb, lsb_val;
	pack_indices_16(best_idx, msb, lsb_val);

	out_error = best_error;
	return uint2(etc_bswap(hi), etc_bswap((msb << 16) | lsb_val));
}

///////////////////////////////////////////////////////////////////////////////
// H-mode: paint0=b1+d, paint1=b1-d, paint2=b2+d, paint3=b2-d
///////////////////////////////////////////////////////////////////////////////

uint2 try_h_mode(int3 pixels[16], int3 c1, int3 c2, out int out_error) {
	int3 q1 = int3(etc_quantize4(c1.r), etc_quantize4(c1.g), etc_quantize4(c1.b));
	int3 q2 = int3(etc_quantize4(c2.r), etc_quantize4(c2.g), etc_quantize4(c2.b));
	int3 b1 = int3(etc_expand4(q1.r), etc_expand4(q1.g), etc_expand4(q1.b));
	int3 b2 = int3(etc_expand4(q2.r), etc_expand4(q2.g), etc_expand4(q2.b));

	int best_error = 0x7FFFFFFF;
	int best_di = 0;
	uint best_idx[16];
	int best_swap = 0;

	[unroll] for (int swap = 0; swap <= 1; swap++) {
		int3 qa = swap ? q2 : q1, qb = swap ? q1 : q2;
		int3 ba = swap ? b2 : b1, bb = swap ? b1 : b2;

		int val_a = (qa.r << 8) | (qa.g << 4) | qa.b;
		int val_b = (qb.r << 8) | (qb.g << 4) | qb.b;
		int ordering_bit = (val_a >= val_b) ? 1 : 0;

		[unroll] for (int di = 0; di < 8; di++) {
			if ((di & 1) != ordering_bit) continue;
			int d = etc_th_dist_val(di);
			int3 paint[4] = { clamp(ba+d,0,255), clamp(ba-d,0,255), clamp(bb+d,0,255), clamp(bb-d,0,255) };

			uint idx[16];
			int err = th_block_error(pixels, paint, idx);
			if (err < best_error) {
				best_error = err;
				best_di = di;
				best_swap = swap;
				best_idx = idx;
			}
		}
	}

	// Apply winning swap
	int3 fq1 = best_swap ? q2 : q1;
	int3 fq2 = best_swap ? q1 : q2;

	// Pack H-mode block (triggers G overflow)
	int g_sum = ((fq1.g & 1) * 2) + (((fq1.b >> 2) & 1) * 2) + ((fq1.b >> 3) & 1) + ((fq1.b >> 1) & 1);
	int use_pos = (g_sum >= 4) ? 1 : 0;

	uint hi = 0;
	if ((fq1.g >> 3) & 1) hi |= 1u << 31;
	hi |= uint(fq1.r & 0xF) << 27;
	hi |= uint((fq1.g >> 1) & 7) << 24;
	if (use_pos) hi |= 7u << 21;
	hi |= uint(fq1.g & 1) << 20;
	hi |= uint((fq1.b >> 3) & 1) << 19;
	if (!use_pos) hi |= 1u << 18;
	hi |= uint(fq1.b & 7) << 15;
	hi |= uint(fq2.r & 0xF) << 11;
	hi |= uint(fq2.g & 0xF) << 7;
	hi |= uint(fq2.b & 0xF) << 3;
	hi |= uint((best_di >> 2) & 1) << 2;
	hi |= 1u << 1;
	hi |= uint((best_di >> 1) & 1);

	uint msb, lsb_val;
	pack_indices_16(best_idx, msb, lsb_val);

	out_error = best_error;
	return uint2(etc_bswap(hi), etc_bswap((msb << 16) | lsb_val));
}

///////////////////////////////////////////////////////////////////////////////
// Planar mode
///////////////////////////////////////////////////////////////////////////////

bool planar_will_overflow(int o_b6) {
	return (((o_b6 >> 3) & 3) + ((o_b6 >> 1) & 3)) < 4;
}

int find_best_overflow_bo(int target_b8) {
	int best_b6 = 0, best_err = 0x7FFFFFFF;
	[unroll] for (int b6 = 0; b6 < 64; b6++) {
		if (!planar_will_overflow(b6)) continue;
		int b8 = etc_expand6(b6);
		int d = b8 - target_b8;
		int err = d * d;
		if (err < best_err) { best_err = err; best_b6 = b6; }
	}
	return best_b6;
}

void planar_fix_overflow(inout int o_r6, inout int o_g7, inout int o_b6) {
	int r0_5   = (o_r6 >> 2) & 0xF;
	int dr_raw = ((o_r6 & 3) << 1) | ((o_g7 >> 6) & 1);
	if (r0_5 + ((dr_raw >= 4) ? (dr_raw - 8) : dr_raw) < 0) o_r6 &= ~2;
	int g0_5   = (o_g7 >> 2) & 0xF;
	int dg_raw = ((o_g7 & 3) << 1) | ((o_b6 >> 5) & 1);
	if (g0_5 + ((dg_raw >= 4) ? (dg_raw - 8) : dg_raw) < 0) o_g7 &= ~2;
}

uint2 encode_planar(int3 pixels[16]) {
	int3 total = 0, sum_h = 0, sum_v = 0;
	[unroll] for (int y = 0; y < 4; y++) {
		[unroll] for (int x = 0; x < 4; x++) {
			int3 c = pixels[y * 4 + x];
			total += c;
			sum_h += x * c;
			sum_v += y * c;
		}
	}
	int3 sum_o = 4 * total - sum_h - sum_v;

	int det_div4 = 25600;
	int3 o8 = clamp((sum_o * 1840 - sum_h *   80 - sum_v *   80) / det_div4, 0, 255);
	int3 h8 = clamp((-sum_o *  80 + sum_h * 3120 - sum_v * 2000) / det_div4, 0, 255);
	int3 v8 = clamp((-sum_o *  80 - sum_h * 2000 + sum_v * 3120) / det_div4, 0, 255);

	int3 oq = int3(min((o8.r+2)>>2,63), min((o8.g+1)>>1,127), min((o8.b+2)>>2,63));
	int3 hq = int3(min((h8.r+2)>>2,63), min((h8.g+1)>>1,127), min((h8.b+2)>>2,63));
	int3 vq = int3(min((v8.r+2)>>2,63), min((v8.g+1)>>1,127), min((v8.b+2)>>2,63));

	if (!planar_will_overflow(oq.b)) oq.b = find_best_overflow_bo(o8.b);
	planar_fix_overflow(oq.r, oq.g, oq.b);
	if (!planar_will_overflow(oq.b)) oq.b = find_best_overflow_bo(o8.b);

	uint hi = 0;
	hi |= uint(oq.r & 0x3F) << 25;
	hi |= uint((oq.g >> 6) & 1) << 24;
	hi |= uint(oq.g & 0x3F) << 17;
	hi |= uint((oq.b >> 5) & 1) << 16;
	hi |= uint((oq.b >> 3) & 3) << 11;
	hi |= 1u << 10;
	hi |= uint(oq.b & 7) << 7;
	hi |= uint((hq.r >> 1) & 0x1F) << 2;
	hi |= 1u << 1;
	hi |= uint(hq.r & 1);

	uint lo = 0;
	lo |= uint(hq.g & 0x7F) << 25;
	lo |= uint(hq.b & 0x3F) << 19;
	lo |= uint(vq.r & 0x3F) << 13;
	lo |= uint(vq.g & 0x7F) << 6;
	lo |= uint(vq.b & 0x3F);

	return uint2(etc_bswap(hi), etc_bswap(lo));
}

///////////////////////////////////////////////////////////////////////////////
// NN Mode Classifier (25 -> 16 -> 5)
//
// Predicts: 0=differential, 1=individual, 2=planar, 3=T-mode, 4=H-mode
// Trained on 500k blocks, 74.7% accuracy, 10.3% SSE overhead vs exhaustive.
///////////////////////////////////////////////////////////////////////////////

// Weights, biases, and standardization parameters are baked from training.
// clang-format off
#include "etc2_nn_weights.hlsli"
// clang-format on

// 5-class mode family prediction from 16 pixel colors.
// Returns: 0=diff, 1=individual, 2=planar, 3=T, 4=H
int nn_predict_mode(int3 pixels[16]) {
	// ---- Compute 25 derived features ----
	float3 sum_c   = 0;
	float3 min_c   = 255, max_c = 0;
	float3 sum_l   = 0, sum_r = 0, sum_t = 0, sum_b = 0;
	float  hz      = 0, vt = 0, max_adj = 0;
	float  min_lum = 9999, max_lum = -1;
	float3 min_lum_rgb = float3(pixels[0]);
	float3 max_lum_rgb = float3(pixels[0]);

	[unroll] for (int y = 0; y < 4; y++) {
		[unroll] for (int x = 0; x < 4; x++) {
			float3 p = float3(pixels[y * 4 + x]) / 255.0;
			sum_c += p;
			min_c = min(min_c, p); max_c = max(max_c, p);
			if (x < 2) sum_l += p; else sum_r += p;
			if (y < 2) sum_t += p; else sum_b += p;

			float lum = (2.0 * p.r + 4.0 * p.g + p.b) / 7.0;
			if (lum < min_lum) { min_lum = lum; min_lum_rgb = p; }
			if (lum > max_lum) { max_lum = lum; max_lum_rgb = p; }

			if (x < 3) { float3 d = abs(p - float3(pixels[y*4+x+1]) / 255.0); float s = d.r+d.g+d.b; hz += s; max_adj = max(max_adj, max(d.r, max(d.g, d.b))); }
			if (y < 3) { float3 d = abs(p - float3(pixels[(y+1)*4+x]) / 255.0); float s = d.r+d.g+d.b; vt += s; max_adj = max(max_adj, max(d.r, max(d.g, d.b))); }
		}
	}

	float3 range_c   = max_c - min_c;
	float  max_range = max(range_c.r, max(range_c.g, range_c.b));
	float3 diff_h    = abs(sum_l - sum_r) / 8.0;
	float3 diff_v    = abs(sum_t - sum_b) / 8.0;
	float  sub_diff_h = diff_h.r + diff_h.g + diff_h.b;
	float  sub_diff_v = diff_v.r + diff_v.g + diff_v.b;

	// Diff validity: check if 5-bit delta fits [-4,+3]
	float diff_val_v = 1.0, diff_val_h = 1.0;
	[unroll] for (int c = 0; c < 3; c++) {
		float avg0v = (c==0 ? sum_l.r : c==1 ? sum_l.g : sum_l.b) / 8.0 * 255.0 + 0.5;
		float avg1v = (c==0 ? sum_r.r : c==1 ? sum_r.g : sum_r.b) / 8.0 * 255.0 + 0.5;
		int q0v = min(int(avg0v) * 31 / 255, 31);
		int q1v = min(int(avg1v) * 31 / 255, 31);
		if (q1v - q0v < -4 || q1v - q0v > 3) diff_val_v = 0.0;

		float avg0h = (c==0 ? sum_t.r : c==1 ? sum_t.g : sum_t.b) / 8.0 * 255.0 + 0.5;
		float avg1h = (c==0 ? sum_b.r : c==1 ? sum_b.g : sum_b.b) / 8.0 * 255.0 + 0.5;
		int q0h = min(int(avg0h) * 31 / 255, 31);
		int q1h = min(int(avg1h) * 31 / 255, 31);
		if (q1h - q0h < -4 || q1h - q0h > 3) diff_val_h = 0.0;
	}

	// Block variance
	float mean_var = 0;
	[unroll] for (int ch = 0; ch < 3; ch++) {
		float m = (ch==0 ? sum_c.r : ch==1 ? sum_c.g : sum_c.b) / 16.0;
		float v = 0;
		[unroll] for (int i = 0; i < 16; i++) {
			float pv = float(ch==0 ? pixels[i].r : ch==1 ? pixels[i].g : pixels[i].b) / 255.0;
			float dd = pv - m; v += dd * dd;
		}
		mean_var += v / 16.0;
	}
	mean_var /= 3.0;

	// Color pair: quantize min/max lum to 4-bit
	float3 b1, b2;
	[unroll] for (int qc = 0; qc < 3; qc++) {
		float v1 = qc==0 ? min_lum_rgb.r : qc==1 ? min_lum_rgb.g : min_lum_rgb.b;
		float v2 = qc==0 ? max_lum_rgb.r : qc==1 ? max_lum_rgb.g : max_lum_rgb.b;
		int i1 = min(int(v1 * 255.0 + 0.5 + 8) >> 4, 15);
		int i2 = min(int(v2 * 255.0 + 0.5 + 8) >> 4, 15);
		float e1 = float((i1 << 4) | i1) / 255.0;
		float e2 = float((i2 << 4) | i2) / 255.0;
		if (qc==0) { b1.r=e1; b2.r=e2; }
		else if (qc==1) { b1.g=e1; b2.g=e2; }
		else { b1.b=e1; b2.b=e2; }
	}
	float cpd = length(b1 - b2);

	// Cluster analysis
	float sum_near = 0, max_near = 0;
	int   cnt1 = 0;
	float wvar1 = 0, wvar2 = 0;
	int   cnt2 = 0;
	[unroll] for (int ci = 0; ci < 16; ci++) {
		float3 pp = float3(pixels[ci]) / 255.0;
		float d1 = dot(pp - b1, pp - b1);
		float d2 = dot(pp - b2, pp - b2);
		float nearest = min(d1, d2);
		sum_near += nearest;
		max_near = max(max_near, nearest);
		if (d1 < d2) { cnt1++; wvar1 += d1; }
		else          { cnt2++; wvar2 += d2; }
	}
	float clust_tight = sum_near / 16.0;
	float between_d   = dot(b1-b2, b1-b2);
	float within_avg  = 0;
	if (cnt1 > 0) within_avg += wvar1 / float(cnt1);
	if (cnt2 > 0) within_avg += wvar2 / float(cnt2);
	within_avg = within_avg * 0.5 + 1e-6;
	float bimodal   = between_d / within_avg;
	float total_w   = wvar1 + wvar2;
	float total_v   = mean_var * 3.0 * 16.0 + 1e-6;
	float var_ratio = total_w / total_v;

	// Interaction features
	float tight_cpd  = clust_tight * cpd;
	float resid_vr   = max_near * var_ratio;
	float range_vr   = max_range * var_ratio;
	float bimod_cpd  = bimodal * cpd;
	float lum_tight  = (max_lum - min_lum) * clust_tight;
	float smooth     = max(0.0, 1.0 - max_adj);
	float grad_smth  = sub_diff_v * smooth + sub_diff_h * smooth;

	// ---- Assemble feature vector ----
	float feat[25] = {
		max_range, sub_diff_h, max_adj, hz / 36.0, vt / 36.0,
		diff_val_v, diff_val_h,
		range_c.r, range_c.g, range_c.b, max_lum - min_lum,
		sub_diff_v, mean_var, cpd,
		clust_tight, max_near, float(cnt1) / 16.0, bimodal, var_ratio,
		tight_cpd, resid_vr, range_vr, bimod_cpd, lum_tight, grad_smth,
	};

	// ---- Standardize ----
	[unroll] for (int fi = 0; fi < 25; fi++)
		feat[fi] = (feat[fi] - nn_feat_mean[fi]) / nn_feat_std[fi];

	// ---- Layer 1: 25 -> 16, ReLU ----
	float hidden[16];
	[unroll] for (int j = 0; j < 16; j++) {
		float s = nn_b1[j];
		[unroll] for (int i = 0; i < 25; i++)
			s += feat[i] * nn_w1[i * 16 + j];
		hidden[j] = max(0.0, s);
	}

	// ---- Layer 2: 16 -> 5 (argmax, no softmax needed) ----
	int   best_class = 0;
	float best_score = nn_b2[0];
	[unroll] for (int j2 = 0; j2 < 16; j2++)
		best_score += hidden[j2] * nn_w2[j2 * 5 + 0];
	[unroll] for (int k = 1; k < 5; k++) {
		float score = nn_b2[k];
		[unroll] for (int j2 = 0; j2 < 16; j2++)
			score += hidden[j2] * nn_w2[j2 * 5 + k];
		if (score > best_score) { best_score = score; best_class = k; }
	}
	return best_class;
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint bx = id.x;
	uint by = id.y;
	if (bx >= blocks_x || by >= (image_height + 3) / 4)
		return;

	// Load 4x4 block (premultiply alpha so semi-transparent pixels → black)
	int3 pixels[16];
	[unroll] for (uint py = 0; py < 4; py++) {
		[unroll] for (uint px = 0; px < 4; px++) {
			uint sx = min(bx * 4 + px, image_width  - 1);
			uint sy = min(by * 4 + py, image_height - 1);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			pixels[py * 4 + px] = int3(texel.rgb * texel.a * 255.0 + 0.5);
		}
	}

	// =====================================================================
	// NN classifier predicts mode family, then we only run that encoder
	// 0=diff, 1=individual, 2=planar, 3=T, 4=H
	// =====================================================================

	int predicted_mode = nn_predict_mode(pixels);

	uint2 best_block = uint2(0, 0);

	if (predicted_mode == 3 || predicted_mode == 4) {
		// T or H mode
		int min_lum = 0x7FFFFFFF, max_lum = 0;
		int3 min_c = pixels[0], max_c = pixels[0];
		[unroll] for (uint i = 0; i < 16; i++) {
			int3 p = pixels[i];
			int lum = p.r * 2 + p.g * 4 + p.b;
			if (lum < min_lum) { min_lum = lum; min_c = p; }
			if (lum > max_lum) { max_lum = lum; max_c = p; }
		}

		int err;
		if (predicted_mode == 3)
			best_block = try_t_mode(pixels, min_c, max_c, err);
		else
			best_block = try_h_mode(pixels, min_c, max_c, err);
	} else if (predicted_mode == 2) {
		// Planar mode
		best_block = encode_planar(pixels);
	} else {
		// Differential (0) or Individual (1): try both flips, pick best
		int best_error = 0x7FFFFFFF;
		[unroll] for (int flip = 0; flip <= 1; flip++) {
			int3 sb0[8], sb1[8];
			if (flip == 0) {
				[unroll] for (int sy = 0; sy < 4; sy++) {
					sb0[sy*2+0] = pixels[sy*4+0]; sb0[sy*2+1] = pixels[sy*4+1];
					sb1[sy*2+0] = pixels[sy*4+2]; sb1[sy*2+1] = pixels[sy*4+3];
				}
			} else {
				[unroll] for (int sx = 0; sx < 4; sx++) {
					sb0[sx+0] = pixels[0*4+sx]; sb0[sx+4] = pixels[1*4+sx];
					sb1[sx+0] = pixels[2*4+sx]; sb1[sx+4] = pixels[3*4+sx];
				}
			}

			int3 avg0 = 0, avg1 = 0;
			[unroll] for (uint ai = 0; ai < 8; ai++) { avg0 += sb0[ai]; avg1 += sb1[ai]; }
			avg0 = (avg0 + 4) / 8;
			avg1 = (avg1 + 4) / 8;

			int3 q5_0 = int3(etc_quantize5(avg0.r), etc_quantize5(avg0.g), etc_quantize5(avg0.b));
			int3 q5_1 = int3(etc_quantize5(avg1.r), etc_quantize5(avg1.g), etc_quantize5(avg1.b));
			int3 delta = q5_1 - q5_0;

			uint out_hi;
			int  t0, t1;
			uint idx0[8], idx1[8];
			int  err0, err1;

			bool use_individual = (predicted_mode == 1) ||
			                      (delta.r < -4 || delta.r > 3 ||
			                       delta.g < -4 || delta.g > 3 ||
			                       delta.b < -4 || delta.b > 3);

			if (use_individual) {
				int3 q4_0 = int3(etc_quantize4(avg0.r), etc_quantize4(avg0.g), etc_quantize4(avg0.b));
				int3 q4_1 = int3(etc_quantize4(avg1.r), etc_quantize4(avg1.g), etc_quantize4(avg1.b));
				int3 b0 = int3(etc_expand4(q4_0.r), etc_expand4(q4_0.g), etc_expand4(q4_0.b));
				int3 b1 = int3(etc_expand4(q4_1.r), etc_expand4(q4_1.g), etc_expand4(q4_1.b));
				err0 = encode_subblock(sb0, b0, t0, idx0);
				err1 = encode_subblock(sb1, b1, t1, idx1);
				out_hi = ((q4_0.r << 4) | q4_1.r) << 24
				       | ((q4_0.g << 4) | q4_1.g) << 16
				       | ((q4_0.b << 4) | q4_1.b) << 8
				       | (uint(t0) << 5) | (uint(t1) << 2) | uint(flip);
			} else {
				int3 d = clamp(delta, -4, 3);
				int3 a1 = q5_0 + d;
				int3 b0 = int3(etc_expand5(q5_0.r), etc_expand5(q5_0.g), etc_expand5(q5_0.b));
				int3 b1 = int3(etc_expand5(a1.r), etc_expand5(a1.g), etc_expand5(a1.b));
				err0 = encode_subblock(sb0, b0, t0, idx0);
				err1 = encode_subblock(sb1, b1, t1, idx1);
				out_hi = (uint(q5_0.r << 3) | uint(d.r & 7)) << 24
				       | (uint(q5_0.g << 3) | uint(d.g & 7)) << 16
				       | (uint(q5_0.b << 3) | uint(d.b & 7)) << 8
				       | (uint(t0) << 5) | (uint(t1) << 2) | (1u << 1) | uint(flip);
			}

			int total_err = err0 + err1;
			if (total_err < best_error) {
				best_error = total_err;
				uint msb, lsb_val;
				pack_subblock_indices(idx0, idx1, flip, msb, lsb_val);
				best_block = uint2(etc_bswap(out_hi), etc_bswap((msb << 16) | lsb_val));
			}
		}
	}

	output_blocks[buffer_offset + by * blocks_x + bx] = best_block;
}
