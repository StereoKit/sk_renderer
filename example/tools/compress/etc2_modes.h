#ifndef ETC2_MODES_H
#define ETC2_MODES_H

/*
 * ETC2 mode evaluation functions (error only, no block encoding).
 * Input: 48 bytes of RGB pixel data (16 pixels, row-major, 3 bytes/pixel).
 * pixel(x,y) channel c = pixels[(y*4+x)*3 + c]
 *
 * 7 modes:
 *   0: Differential flip=0   1: Differential flip=1
 *   2: Individual flip=0     3: Individual flip=1
 *   4: Planar                5: T-mode
 *   6: H-mode
 */

#include <stdint.h>
#include <limits.h>

#define _ETC2_PX(p,x,y,c) ((int32_t)(p)[((y)*4+(x))*3+(c)])

/* Modifier table: index 0-7, each has [small, large] offsets */
static const int32_t _etc2_mod[8][2] = {
	{ 2,  8}, { 5, 17}, { 9, 29}, {13, 42},
	{18, 60}, {24, 80}, {33,106}, {47,183},
};

/* T/H distance table */
static const int32_t _etc2_th_dist[8] = {3, 6, 11, 16, 23, 32, 41, 64};

static inline int32_t _etc2_clamp(int32_t v)      { return v < 0 ? 0 : (v > 255 ? 255 : v); }
static inline int32_t _etc2_expand4(int32_t c)     { return (c << 4) | c; }
static inline int32_t _etc2_expand5(int32_t c)     { return (c << 3) | (c >> 2); }
static inline int32_t _etc2_expand6(int32_t c)     { return (c << 2) | (c >> 4); }
static inline int32_t _etc2_expand7(int32_t c)     { return (c << 1) | (c >> 6); }
static inline int32_t _etc2_quantize4(int32_t c8)  { if (c8 < 0) return 0; int32_t q = (c8 + 8) >> 4; return q > 15 ? 15 : q; }
static inline int32_t _etc2_quantize5(int32_t c8)  { if (c8 < 0) return 0; int32_t q = (c8 * 31 + 127) / 255; return q > 31 ? 31 : q; }

/* Sub-block bounds: [flip][sub] = {x0, y0, width, height} */
static const int32_t _etc2_sub_bounds[2][2][4] = {
	{{0, 0, 2, 4}, {2, 0, 2, 4}},  /* flip=0: vertical split   */
	{{0, 0, 4, 2}, {0, 2, 4, 2}},  /* flip=1: horizontal split */
};

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/* Average RGB of 8 pixels in a sub-block */
static void _etc2_sub_avg(const uint8_t *px, int32_t flip, int32_t sub,
                          int32_t *out_r, int32_t *out_g, int32_t *out_b) {
	const int32_t *b = _etc2_sub_bounds[flip][sub];
	int32_t sr = 0, sg = 0, sb = 0;
	for (int32_t y = b[1]; y < b[1] + b[3]; y++)
		for (int32_t x = b[0]; x < b[0] + b[2]; x++) {
			sr += _ETC2_PX(px, x, y, 0);
			sg += _ETC2_PX(px, x, y, 1);
			sb += _ETC2_PX(px, x, y, 2);
		}
	*out_r = (sr + 4) / 8;
	*out_g = (sg + 4) / 8;
	*out_b = (sb + 4) / 8;
}

/* Error for a sub-block with given base color and modifier table index */
static int32_t _etc2_sub_err(const uint8_t *px, int32_t flip, int32_t sub,
                             int32_t br, int32_t bg, int32_t bb, int32_t ti) {
	int32_t s = _etc2_mod[ti][0], l = _etc2_mod[ti][1];
	int32_t c[4][3] = {
		{_etc2_clamp(br + s), _etc2_clamp(bg + s), _etc2_clamp(bb + s)},
		{_etc2_clamp(br + l), _etc2_clamp(bg + l), _etc2_clamp(bb + l)},
		{_etc2_clamp(br - s), _etc2_clamp(bg - s), _etc2_clamp(bb - s)},
		{_etc2_clamp(br - l), _etc2_clamp(bg - l), _etc2_clamp(bb - l)},
	};

	const int32_t *bd = _etc2_sub_bounds[flip][sub];
	int32_t total = 0;
	for (int32_t y = bd[1]; y < bd[1] + bd[3]; y++)
		for (int32_t x = bd[0]; x < bd[0] + bd[2]; x++) {
			int32_t pr = _ETC2_PX(px, x, y, 0);
			int32_t pg = _ETC2_PX(px, x, y, 1);
			int32_t pb = _ETC2_PX(px, x, y, 2);
			int32_t d0 = (pr-c[0][0])*(pr-c[0][0]) + (pg-c[0][1])*(pg-c[0][1]) + (pb-c[0][2])*(pb-c[0][2]);
			int32_t d1 = (pr-c[1][0])*(pr-c[1][0]) + (pg-c[1][1])*(pg-c[1][1]) + (pb-c[1][2])*(pb-c[1][2]);
			int32_t d2 = (pr-c[2][0])*(pr-c[2][0]) + (pg-c[2][1])*(pg-c[2][1]) + (pb-c[2][2])*(pb-c[2][2]);
			int32_t d3 = (pr-c[3][0])*(pr-c[3][0]) + (pg-c[3][1])*(pg-c[3][1]) + (pb-c[3][2])*(pb-c[3][2]);
			int32_t best = d0;
			if (d1 < best) best = d1;
			if (d2 < best) best = d2;
			if (d3 < best) best = d3;
			total += best;
		}
	return total;
}

/* Find best modifier table via max-deviation heuristic, return sub-block error */
static int32_t _etc2_encode_sub(const uint8_t *px, int32_t flip, int32_t sub,
                                int32_t br, int32_t bg, int32_t bb) {
	const int32_t *bd = _etc2_sub_bounds[flip][sub];
	int32_t max_dev = 0;
	for (int32_t y = bd[1]; y < bd[1] + bd[3]; y++)
		for (int32_t x = bd[0]; x < bd[0] + bd[2]; x++) {
			int32_t dr = _ETC2_PX(px, x, y, 0) - br; if (dr < 0) dr = -dr;
			int32_t dg = _ETC2_PX(px, x, y, 1) - bg; if (dg < 0) dg = -dg;
			int32_t db = _ETC2_PX(px, x, y, 2) - bb; if (db < 0) db = -db;
			int32_t dev = dr > dg ? dr : dg;
			if (db > dev) dev = db;
			if (dev > max_dev) max_dev = dev;
		}

	int32_t ti;
	if      (max_dev <=   8) ti = 0;
	else if (max_dev <=  17) ti = 1;
	else if (max_dev <=  29) ti = 2;
	else if (max_dev <=  42) ti = 3;
	else if (max_dev <=  60) ti = 4;
	else if (max_dev <=  80) ti = 5;
	else if (max_dev <= 106) ti = 6;
	else                     ti = 7;

	return _etc2_sub_err(px, flip, sub, br, bg, bb, ti);
}

/* T/H block error: best-match each of 16 pixels to 4 paint colors */
static int32_t _etc2_th_err(const uint8_t *px, int32_t paint[4][3]) {
	int32_t total = 0;
	for (int32_t y = 0; y < 4; y++)
		for (int32_t x = 0; x < 4; x++) {
			int32_t pr = _ETC2_PX(px, x, y, 0);
			int32_t pg = _ETC2_PX(px, x, y, 1);
			int32_t pb = _ETC2_PX(px, x, y, 2);
			int32_t best = INT32_MAX;
			for (int32_t k = 0; k < 4; k++) {
				int32_t d = (pr - paint[k][0]) * (pr - paint[k][0])
				          + (pg - paint[k][1]) * (pg - paint[k][1])
				          + (pb - paint[k][2]) * (pb - paint[k][2]);
				if (d < best) best = d;
			}
			total += best;
		}
	return total;
}

/* Find min/max luminance colors (lum = 2*R + 4*G + B) */
static void _etc2_lum_colors(const uint8_t *px,
                             int32_t *c1_r, int32_t *c1_g, int32_t *c1_b,
                             int32_t *c2_r, int32_t *c2_g, int32_t *c2_b) {
	int32_t min_lum = INT32_MAX, max_lum = -1;
	*c1_r = *c1_g = *c1_b = 0;
	*c2_r = *c2_g = *c2_b = 0;
	for (int32_t i = 0; i < 16; i++) {
		int32_t r = px[i * 3 + 0], g = px[i * 3 + 1], b = px[i * 3 + 2];
		int32_t lum = r * 2 + g * 4 + b;
		if (lum < min_lum) { min_lum = lum; *c1_r = r; *c1_g = g; *c1_b = b; }
		if (lum > max_lum) { max_lum = lum; *c2_r = r; *c2_g = g; *c2_b = b; }
	}
}

/* ------------------------------------------------------------------ */
/*  Public mode error functions                                        */
/* ------------------------------------------------------------------ */

/* Differential mode error (modes 0, 1) — RGB555 + delta333 */
static int32_t etc2_differential_error(const uint8_t *px, int32_t flip) {
	int32_t avg0_r, avg0_g, avg0_b, avg1_r, avg1_g, avg1_b;
	_etc2_sub_avg(px, flip, 0, &avg0_r, &avg0_g, &avg0_b);
	_etc2_sub_avg(px, flip, 1, &avg1_r, &avg1_g, &avg1_b);

	int32_t r0 = _etc2_quantize5(avg0_r), g0 = _etc2_quantize5(avg0_g), b0 = _etc2_quantize5(avg0_b);
	int32_t r1 = _etc2_quantize5(avg1_r), g1 = _etc2_quantize5(avg1_g), b1 = _etc2_quantize5(avg1_b);

	/* Delta must fit in [-4, +3] for each channel */
	int32_t dr = r1 - r0, dg = g1 - g0, db = b1 - b0;
	if (dr < -4 || dr > 3 || dg < -4 || dg > 3 || db < -4 || db > 3)
		return INT32_MAX;

	int32_t base0_r = _etc2_expand5(r0), base0_g = _etc2_expand5(g0), base0_b = _etc2_expand5(b0);
	int32_t base1_r = _etc2_expand5(r1), base1_g = _etc2_expand5(g1), base1_b = _etc2_expand5(b1);

	return _etc2_encode_sub(px, flip, 0, base0_r, base0_g, base0_b)
	     + _etc2_encode_sub(px, flip, 1, base1_r, base1_g, base1_b);
}

/* Individual mode error (modes 2, 3) — RGB444 + RGB444 */
static int32_t etc2_individual_error(const uint8_t *px, int32_t flip) {
	int32_t avg0_r, avg0_g, avg0_b, avg1_r, avg1_g, avg1_b;
	_etc2_sub_avg(px, flip, 0, &avg0_r, &avg0_g, &avg0_b);
	_etc2_sub_avg(px, flip, 1, &avg1_r, &avg1_g, &avg1_b);

	int32_t base0_r = _etc2_expand4(_etc2_quantize4(avg0_r));
	int32_t base0_g = _etc2_expand4(_etc2_quantize4(avg0_g));
	int32_t base0_b = _etc2_expand4(_etc2_quantize4(avg0_b));
	int32_t base1_r = _etc2_expand4(_etc2_quantize4(avg1_r));
	int32_t base1_g = _etc2_expand4(_etc2_quantize4(avg1_g));
	int32_t base1_b = _etc2_expand4(_etc2_quantize4(avg1_b));

	return _etc2_encode_sub(px, flip, 0, base0_r, base0_g, base0_b)
	     + _etc2_encode_sub(px, flip, 1, base1_r, base1_g, base1_b);
}

/* Planar mode error (mode 4) — 3-point interpolation */
static int32_t etc2_planar_error(const uint8_t *px) {
	int32_t oq[3], hq[3], vq[3];

	for (int32_t ch = 0; ch < 3; ch++) {
		int32_t sum_o = 0, sum_h = 0, sum_v = 0;
		for (int32_t y = 0; y < 4; y++)
			for (int32_t x = 0; x < 4; x++) {
				int32_t c = _ETC2_PX(px, x, y, ch);
				sum_o += (4 - x - y) * c;
				sum_h += x * c;
				sum_v += y * c;
			}

		/* Cramer's rule: det/4 = 25600 */
		int32_t o8 = _etc2_clamp((sum_o * 1840 - sum_h * 80   - sum_v * 80)   / 25600);
		int32_t h8 = _etc2_clamp((-sum_o * 80  + sum_h * 3120 - sum_v * 2000) / 25600);
		int32_t v8 = _etc2_clamp((-sum_o * 80  - sum_h * 2000 + sum_v * 3120) / 25600);

		/* Quantize to 6/7/6 bits */
		if (ch == 1) { /* green = 7 bits */
			int32_t oq7 = (o8 + 1) >> 1; if (oq7 > 127) oq7 = 127;
			int32_t hq7 = (h8 + 1) >> 1; if (hq7 > 127) hq7 = 127;
			int32_t vq7 = (v8 + 1) >> 1; if (vq7 > 127) vq7 = 127;
			oq[ch] = _etc2_expand7(oq7);
			hq[ch] = _etc2_expand7(hq7);
			vq[ch] = _etc2_expand7(vq7);
		} else { /* red, blue = 6 bits */
			int32_t oq6 = (o8 + 2) >> 2; if (oq6 > 63) oq6 = 63;
			int32_t hq6 = (h8 + 2) >> 2; if (hq6 > 63) hq6 = 63;
			int32_t vq6 = (v8 + 2) >> 2; if (vq6 > 63) vq6 = 63;
			oq[ch] = _etc2_expand6(oq6);
			hq[ch] = _etc2_expand6(hq6);
			vq[ch] = _etc2_expand6(vq6);
		}
	}

	int32_t total = 0;
	for (int32_t y = 0; y < 4; y++)
		for (int32_t x = 0; x < 4; x++)
			for (int32_t ch = 0; ch < 3; ch++) {
				int32_t pred = _etc2_clamp((x * (hq[ch] - oq[ch]) + y * (vq[ch] - oq[ch]) + 4 * oq[ch] + 2) >> 2);
				int32_t diff = _ETC2_PX(px, x, y, ch) - pred;
				total += diff * diff;
			}
	return total;
}

/* T-mode error (mode 5) — paint: {base1, base2+d, base2, base2-d} */
static int32_t etc2_t_mode_error(const uint8_t *px) {
	int32_t c1_r, c1_g, c1_b, c2_r, c2_g, c2_b;
	_etc2_lum_colors(px, &c1_r, &c1_g, &c1_b, &c2_r, &c2_g, &c2_b);

	int32_t base1_r = _etc2_expand4(_etc2_quantize4(c1_r));
	int32_t base1_g = _etc2_expand4(_etc2_quantize4(c1_g));
	int32_t base1_b = _etc2_expand4(_etc2_quantize4(c1_b));
	int32_t base2_r = _etc2_expand4(_etc2_quantize4(c2_r));
	int32_t base2_g = _etc2_expand4(_etc2_quantize4(c2_g));
	int32_t base2_b = _etc2_expand4(_etc2_quantize4(c2_b));

	int32_t best = INT32_MAX;
	for (int32_t di = 0; di < 8; di++) {
		int32_t d = _etc2_th_dist[di];
		int32_t paint[4][3] = {
			{base1_r,                    base1_g,                    base1_b                   },
			{_etc2_clamp(base2_r + d),   _etc2_clamp(base2_g + d),  _etc2_clamp(base2_b + d)  },
			{base2_r,                    base2_g,                    base2_b                   },
			{_etc2_clamp(base2_r - d),   _etc2_clamp(base2_g - d),  _etc2_clamp(base2_b - d)  },
		};
		int32_t err = _etc2_th_err(px, paint);
		if (err < best) best = err;
	}
	return best;
}

/* H-mode error (mode 6) — paint: {base_a+d, base_a-d, base_b+d, base_b-d} */
static int32_t etc2_h_mode_error(const uint8_t *px) {
	int32_t c1_r, c1_g, c1_b, c2_r, c2_g, c2_b;
	_etc2_lum_colors(px, &c1_r, &c1_g, &c1_b, &c2_r, &c2_g, &c2_b);

	int32_t r1_4 = _etc2_quantize4(c1_r), g1_4 = _etc2_quantize4(c1_g), b1_4 = _etc2_quantize4(c1_b);
	int32_t r2_4 = _etc2_quantize4(c2_r), g2_4 = _etc2_quantize4(c2_g), b2_4 = _etc2_quantize4(c2_b);
	int32_t base1_r = _etc2_expand4(r1_4), base1_g = _etc2_expand4(g1_4), base1_b = _etc2_expand4(b1_4);
	int32_t base2_r = _etc2_expand4(r2_4), base2_g = _etc2_expand4(g2_4), base2_b = _etc2_expand4(b2_4);

	int32_t best = INT32_MAX;
	for (int32_t swap = 0; swap <= 1; swap++) {
		int32_t ra = swap ? r2_4 : r1_4, ga = swap ? g2_4 : g1_4, ba = swap ? b2_4 : b1_4;
		int32_t rb = swap ? r1_4 : r2_4, gb = swap ? g1_4 : g2_4, bb = swap ? b1_4 : b2_4;
		int32_t ba_r = swap ? base2_r : base1_r, ba_g = swap ? base2_g : base1_g, ba_b = swap ? base2_b : base1_b;
		int32_t bb_r = swap ? base1_r : base2_r, bb_g = swap ? base1_g : base2_g, bb_b = swap ? base1_b : base2_b;

		int32_t val_a = (ra << 8) | (ga << 4) | ba;
		int32_t val_b = (rb << 8) | (gb << 4) | bb;
		int32_t ordering_bit = (val_a >= val_b) ? 1 : 0;

		for (int32_t di = 0; di < 8; di++) {
			if ((di & 1) != ordering_bit) continue;
			int32_t d = _etc2_th_dist[di];
			int32_t paint[4][3] = {
				{_etc2_clamp(ba_r + d), _etc2_clamp(ba_g + d), _etc2_clamp(ba_b + d)},
				{_etc2_clamp(ba_r - d), _etc2_clamp(ba_g - d), _etc2_clamp(ba_b - d)},
				{_etc2_clamp(bb_r + d), _etc2_clamp(bb_g + d), _etc2_clamp(bb_b + d)},
				{_etc2_clamp(bb_r - d), _etc2_clamp(bb_g - d), _etc2_clamp(bb_b - d)},
			};
			int32_t err = _etc2_th_err(px, paint);
			if (err < best) best = err;
		}
	}
	return best;
}

/* Dispatcher: compute error for a specific mode (0-6) */
static int32_t etc2_mode_error(const uint8_t *px, int32_t mode) {
	switch (mode) {
	case 0: return etc2_differential_error(px, 0);
	case 1: return etc2_differential_error(px, 1);
	case 2: return etc2_individual_error(px, 0);
	case 3: return etc2_individual_error(px, 1);
	case 4: return etc2_planar_error(px);
	case 5: return etc2_t_mode_error(px);
	case 6: return etc2_h_mode_error(px);
	default: return INT32_MAX;
	}
}

/* Evaluate all 7 modes, write errors to out_errors[7], return best mode index */
static int32_t etc2_evaluate_all(const uint8_t *px, int32_t out_errors[7]) {
	int32_t best_mode = 0;
	int32_t best_err  = INT32_MAX;
	for (int32_t m = 0; m < 7; m++) {
		out_errors[m] = etc2_mode_error(px, m);
		if (out_errors[m] < best_err) {
			best_err  = out_errors[m];
			best_mode = m;
		}
	}
	return best_mode;
}

#endif /* ETC2_MODES_H */
