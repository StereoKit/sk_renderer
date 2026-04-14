/*
 * train.c — ETC2 mode classifier trainer.
 * Loads training data, trains decision tree + linear + neural net classifiers,
 * reports accuracy, exports models as C header files.
 *
 * Usage: ./train <training_data.bin> [--val-split 0.2] [--hidden 16,32,64]
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <float.h>

#include "etc2_modes.h"

#define NUM_CLASSES      5
#define NUM_RAW_FEATURES 48
#define NUM_DERIVED      25
#define NUM_FEATURES     (NUM_RAW_FEATURES + NUM_DERIVED)
#define NUM_BUCKETS      256

/* Runtime feature count — equals NUM_FEATURES normally, NUM_DERIVED with --derived-only */
static int32_t g_n_feat = NUM_FEATURES;

/* ================================================================== */
/*  RNG (xorshift64, seed=42)                                         */
/* ================================================================== */

static uint64_t _rng_state = 42;

static uint64_t _rng_next(void) {
	_rng_state ^= _rng_state << 13;
	_rng_state ^= _rng_state >> 7;
	_rng_state ^= _rng_state << 17;
	return _rng_state;
}

static float _rng_uniform(void) {
	return (float)(_rng_next() & 0xFFFFFF) / (float)0xFFFFFF;
}

static float _rng_normal(float mean, float stddev) {
	float u1 = _rng_uniform();
	float u2 = _rng_uniform();
	while (u1 <= 1e-7f) u1 = _rng_uniform();
	return mean + stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void _shuffle_indices(int32_t *arr, int32_t n) {
	for (int32_t i = n - 1; i > 0; i--) {
		int32_t j = (int32_t)(_rng_next() % (uint64_t)(i + 1));
		int32_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
	}
}

/* ================================================================== */
/*  Softmax                                                            */
/* ================================================================== */

static void _softmax(const float *logits, float *out, int32_t n) {
	float mx = logits[0];
	for (int32_t i = 1; i < n; i++)
		if (logits[i] > mx) mx = logits[i];
	float sum = 0;
	for (int32_t i = 0; i < n; i++) {
		out[i] = expf(logits[i] - mx);
		sum += out[i];
	}
	for (int32_t i = 0; i < n; i++)
		out[i] /= sum;
}

/* ================================================================== */
/*  Feature computation                                                */
/* ================================================================== */

/*
 * Derived features (computed from the 48 normalized [0,1] pixel values):
 *   [48]  max_range      — max channel range across block
 *   [49]  sub_diff_h     — left vs right half average RGB difference
 *   [50]  max_adj_diff   — max adjacent pixel channel difference
 *   [51]  hz_diff        — horizontal edge strength
 *   [52]  vt_diff        — vertical edge strength
 *   [53]  diff_valid_v   — 1 if differential flip=0 delta fits [-4,+3]
 *   [54]  diff_valid_h   — 1 if differential flip=1 delta fits [-4,+3]
 *   [55]  r_range        — red channel range
 *   [56]  g_range        — green channel range
 *   [57]  b_range        — blue channel range
 *   [58]  lum_range      — luminance range (2R+4G+B weighting) / 7.0
 *   [59]  sub_diff_v     — top vs bottom half average RGB difference
 *   [60]  block_variance — mean per-channel variance across block
 *   [61]  color_pair_dist— quantized min/max lum color distance
 */
static void _compute_derived(const float *raw48, float *out) {
	/* pixel(x,y) channel c = raw48[(y*4+x)*3+c] */
	float min_c[3] = {1,1,1}, max_c[3] = {0,0,0};
	float sum_c[3] = {0};
	float sum_left[3] = {0}, sum_right[3] = {0};
	float sum_top[3]  = {0}, sum_bot[3]   = {0};
	float hz = 0, vt = 0;
	float max_adj = 0;

	/* Sub-block averages for differential validity check */
	/* flip=0: sub0=cols 0-1, sub1=cols 2-3 */
	/* flip=1: sub0=rows 0-1, sub1=rows 2-3 */
	float sub_v0[3] = {0}, sub_v1[3] = {0}; /* flip=0 vertical split */
	float sub_h0[3] = {0}, sub_h1[3] = {0}; /* flip=1 horizontal split */

	/* Luminance tracking for T/H mode features */
	float min_lum = 99.0f, max_lum = -1.0f;
	float min_lum_rgb[3] = {0}, max_lum_rgb[3] = {0};

	for (int32_t y = 0; y < 4; y++)
		for (int32_t x = 0; x < 4; x++) {
			int32_t idx = (y * 4 + x) * 3;
			float r = raw48[idx + 0], g = raw48[idx + 1], b = raw48[idx + 2];

			for (int32_t c = 0; c < 3; c++) {
				float v = raw48[idx + c];
				if (v < min_c[c]) min_c[c] = v;
				if (v > max_c[c]) max_c[c] = v;
				sum_c[c] += v;
				if (x < 2) { sum_left[c] += v; sub_v0[c] += v; }
				else        { sum_right[c] += v; sub_v1[c] += v; }
				if (y < 2) { sum_top[c] += v; sub_h0[c] += v; }
				else        { sum_bot[c] += v; sub_h1[c] += v; }
			}

			/* Luminance (2R + 4G + B) / 7, normalized to [0,1] */
			float lum = (2.0f * r + 4.0f * g + b) / 7.0f;
			if (lum < min_lum) { min_lum = lum; min_lum_rgb[0] = r; min_lum_rgb[1] = g; min_lum_rgb[2] = b; }
			if (lum > max_lum) { max_lum = lum; max_lum_rgb[0] = r; max_lum_rgb[1] = g; max_lum_rgb[2] = b; }

			/* Horizontal adjacency */
			if (x < 3) {
				int32_t idx2 = (y * 4 + x + 1) * 3;
				for (int32_t c = 0; c < 3; c++) {
					float d = raw48[idx + c] - raw48[idx2 + c];
					if (d < 0) d = -d;
					hz += d;
					if (d > max_adj) max_adj = d;
				}
			}
			/* Vertical adjacency */
			if (y < 3) {
				int32_t idx2 = ((y + 1) * 4 + x) * 3;
				for (int32_t c = 0; c < 3; c++) {
					float d = raw48[idx + c] - raw48[idx2 + c];
					if (d < 0) d = -d;
					vt += d;
					if (d > max_adj) max_adj = d;
				}
			}
		}

	/* Original 5 features */
	float max_range = 0, sub_diff_h = 0;
	for (int32_t c = 0; c < 3; c++) {
		float rng = max_c[c] - min_c[c];
		if (rng > max_range) max_range = rng;
		float d = (sum_left[c] - sum_right[c]) / 8.0f;
		if (d < 0) d = -d;
		sub_diff_h += d;
	}

	out[0] = max_range;
	out[1] = sub_diff_h;
	out[2] = max_adj;
	out[3] = hz / (12.0f * 3.0f);
	out[4] = vt / (12.0f * 3.0f);

	/* [53-54] Differential validity: check if 5-bit delta fits in [-4,+3] */
	/* Convert normalized [0,1] averages back to 8-bit, quantize to 5-bit */
	float diff_valid_v = 1.0f, diff_valid_h = 1.0f;
	for (int32_t c = 0; c < 3; c++) {
		/* flip=0 sub-blocks */
		int32_t avg0_v = (int32_t)(sub_v0[c] / 8.0f * 255.0f + 0.5f);
		int32_t avg1_v = (int32_t)(sub_v1[c] / 8.0f * 255.0f + 0.5f);
		int32_t q0_v = (avg0_v * 31 + 127) / 255;
		int32_t q1_v = (avg1_v * 31 + 127) / 255;
		int32_t dv = q1_v - q0_v;
		if (dv < -4 || dv > 3) diff_valid_v = 0.0f;

		/* flip=1 sub-blocks */
		int32_t avg0_h = (int32_t)(sub_h0[c] / 8.0f * 255.0f + 0.5f);
		int32_t avg1_h = (int32_t)(sub_h1[c] / 8.0f * 255.0f + 0.5f);
		int32_t q0_h = (avg0_h * 31 + 127) / 255;
		int32_t q1_h = (avg1_h * 31 + 127) / 255;
		int32_t dh = q1_h - q0_h;
		if (dh < -4 || dh > 3) diff_valid_h = 0.0f;
	}
	out[5] = diff_valid_v;
	out[6] = diff_valid_h;

	/* [55-57] Per-channel ranges */
	out[7] = max_c[0] - min_c[0];
	out[8] = max_c[1] - min_c[1];
	out[9] = max_c[2] - min_c[2];

	/* [58] Luminance range */
	out[10] = max_lum - min_lum;

	/* [59] Top vs bottom half diff */
	float sub_diff_v = 0;
	for (int32_t c = 0; c < 3; c++) {
		float d = (sum_top[c] - sum_bot[c]) / 8.0f;
		if (d < 0) d = -d;
		sub_diff_v += d;
	}
	out[11] = sub_diff_v;

	/* [60] Block variance — mean per-channel variance */
	float mean_var = 0;
	for (int32_t c = 0; c < 3; c++) {
		float mean = sum_c[c] / 16.0f;
		float var = 0;
		for (int32_t i = 0; i < 16; i++) {
			float d = raw48[i * 3 + c] - mean;
			var += d * d;
		}
		mean_var += var / 16.0f;
	}
	out[12] = mean_var / 3.0f;

	/* [61] Color pair distance — quantized min/max luminance color distance */
	float cpd = 0;
	for (int32_t c = 0; c < 3; c++) {
		/* Quantize to 4-bit and expand back (mimics T/H mode) */
		int32_t v1 = (int32_t)(min_lum_rgb[c] * 255.0f + 0.5f);
		int32_t v2 = (int32_t)(max_lum_rgb[c] * 255.0f + 0.5f);
		int32_t q1 = (v1 + 8) >> 4; if (q1 > 15) q1 = 15;
		int32_t q2 = (v2 + 8) >> 4; if (q2 > 15) q2 = 15;
		int32_t e1 = (q1 << 4) | q1;  /* expand4 */
		int32_t e2 = (q2 << 4) | q2;
		float d = (float)(e1 - e2) / 255.0f;
		cpd += d * d;
	}
	out[13] = sqrtf(cpd);

	/* ---- T/H mode focused features ---- */
	/* Assign each pixel to nearest base color (min/max luminance pair),
	 * then measure how well this 2-cluster split explains the block. */

	/* Quantized base colors (what T/H mode actually uses) */
	float b1[3], b2[3];
	for (int32_t c = 0; c < 3; c++) {
		int32_t v1 = (int32_t)(min_lum_rgb[c] * 255.0f + 0.5f);
		int32_t v2 = (int32_t)(max_lum_rgb[c] * 255.0f + 0.5f);
		int32_t q1 = (v1 + 8) >> 4; if (q1 > 15) q1 = 15;
		int32_t q2 = (v2 + 8) >> 4; if (q2 > 15) q2 = 15;
		b1[c] = (float)((q1 << 4) | q1) / 255.0f;
		b2[c] = (float)((q2 << 4) | q2) / 255.0f;
	}

	/* [62] cluster_tightness — avg distance of each pixel to nearest base color */
	/* [63] max_cluster_residual — worst-case pixel distance to nearest base */
	/* [64] cluster_balance — fraction of pixels closer to base1 (0.5 = even split) */
	float sum_nearest_dist = 0;
	float max_nearest_dist = 0;
	int32_t count_b1 = 0;
	float within_var_1 = 0, within_var_2 = 0;
	int32_t count_b2 = 0;

	for (int32_t i = 0; i < 16; i++) {
		float r = raw48[i*3+0], g = raw48[i*3+1], bb_ = raw48[i*3+2];
		float d1 = (r-b1[0])*(r-b1[0]) + (g-b1[1])*(g-b1[1]) + (bb_-b1[2])*(bb_-b1[2]);
		float d2 = (r-b2[0])*(r-b2[0]) + (g-b2[1])*(g-b2[1]) + (bb_-b2[2])*(bb_-b2[2]);
		float nearest = d1 < d2 ? d1 : d2;
		sum_nearest_dist += nearest;
		if (nearest > max_nearest_dist) max_nearest_dist = nearest;
		if (d1 < d2) { count_b1++; within_var_1 += d1; }
		else          { count_b2++; within_var_2 += d2; }
	}

	out[14] = sum_nearest_dist / 16.0f;  /* avg distance to nearest base */
	out[15] = max_nearest_dist;           /* max distance to nearest base */
	out[16] = (float)count_b1 / 16.0f;   /* cluster balance */

	/* [65] bimodality — Fisher criterion: between-cluster / within-cluster variance */
	/* High value = block is clearly two-toned → T/H mode wins */
	float between_dist = (b1[0]-b2[0])*(b1[0]-b2[0])
	                   + (b1[1]-b2[1])*(b1[1]-b2[1])
	                   + (b1[2]-b2[2])*(b1[2]-b2[2]);
	float within_avg = 0;
	if (count_b1 > 0) within_avg += within_var_1 / (float)count_b1;
	if (count_b2 > 0) within_avg += within_var_2 / (float)count_b2;
	within_avg = within_avg * 0.5f + 1e-6f;
	out[17] = between_dist / within_avg;

	/* [66] variance_ratio — within-cluster variance / total block variance
	 * Low value = two clusters explain most of the variance → T/H mode */
	float total_within = (count_b1 > 0 ? within_var_1 : 0)
	                   + (count_b2 > 0 ? within_var_2 : 0);
	float total_var = mean_var * 3.0f * 16.0f + 1e-6f; /* unnormalized total variance */
	out[18] = total_within / total_var;

	/* ---- Interaction features (products — nonlinear, can't be learned by linear) ---- */
	float tight = sum_nearest_dist / 16.0f;
	float vr    = total_within / total_var;

	/* [67] tight * cpair — tight clusters AND separated colors → T/H */
	out[19] = tight * cpd;

	/* [68] resid * vr — high residual in clustered blocks → needs T/H distance */
	out[20] = max_nearest_dist * vr;

	/* [69] range * vr — high range with good cluster fit → individual */
	out[21] = max_range * vr;

	/* [70] bimodal * cpair — strong two-tone with color separation */
	out[22] = (between_dist / within_avg) * cpd;

	/* [71] lum * tight — luminance spread in tight clusters → T/H confidence */
	out[23] = (max_lum - min_lum) * tight;

	/* [72] gradient * (1-max_adj) — smooth block with no edges → planar confidence */
	float smooth = 1.0f - max_adj;
	if (smooth < 0) smooth = 0;
	out[24] = sub_diff_v * smooth + sub_diff_h * smooth;
}

/* ================================================================== */
/*  Decision Tree                                                      */
/* ================================================================== */

#define MAX_TREE_NODES 4096

typedef struct {
	int32_t feature;    /* -1 for leaf */
	float   threshold;
	int32_t left;       /* child index */
	int32_t right;      /* child index */
	int32_t prediction; /* leaf class  */
} dt_node_t;

typedef struct {
	dt_node_t nodes[MAX_TREE_NODES];
	int32_t   node_count;
} dt_tree_t;

static int32_t _dt_build(dt_tree_t *tree, const float *features, const uint8_t *labels,
                         int32_t *indices, int32_t count, int32_t depth, int32_t max_depth) {
	if (tree->node_count >= MAX_TREE_NODES - 2) {
		/* Safety: force leaf */
		int32_t node = tree->node_count++;
		tree->nodes[node] = (dt_node_t){.feature = -1, .prediction = 0};
		return node;
	}

	int32_t node = tree->node_count++;

	/* Class counts and majority */
	int32_t class_counts[NUM_CLASSES] = {0};
	for (int32_t i = 0; i < count; i++)
		class_counts[labels[indices[i]]]++;

	int32_t majority = 0;
	for (int32_t c = 1; c < NUM_CLASSES; c++)
		if (class_counts[c] > class_counts[majority])
			majority = c;

	/* Leaf conditions */
	if (depth >= max_depth || count <= 2) {
		tree->nodes[node] = (dt_node_t){.feature = -1, .prediction = majority};
		return node;
	}

	/* Check purity */
	int32_t pure = 1;
	for (int32_t c = 0; c < NUM_CLASSES; c++)
		if (class_counts[c] > 0 && class_counts[c] < count) { pure = 0; break; }
	if (pure) {
		tree->nodes[node] = (dt_node_t){.feature = -1, .prediction = majority};
		return node;
	}

	/* Find best split using bucketing */
	int32_t best_feature = -1;
	float   best_threshold = 0;
	float   best_gini = FLT_MAX;

	for (int32_t f = 0; f < g_n_feat; f++) {
		float fmin = FLT_MAX, fmax = -FLT_MAX;
		for (int32_t i = 0; i < count; i++) {
			float v = features[indices[i] * g_n_feat + f];
			if (v < fmin) fmin = v;
			if (v > fmax) fmax = v;
		}
		if (fmax - fmin < 1e-8f) continue;

		int32_t buckets[NUM_BUCKETS][NUM_CLASSES];
		memset(buckets, 0, sizeof(buckets));
		float scale = (NUM_BUCKETS - 1.0f) / (fmax - fmin);

		for (int32_t i = 0; i < count; i++) {
			float v = features[indices[i] * g_n_feat + f];
			int32_t b = (int32_t)((v - fmin) * scale);
			if (b >= NUM_BUCKETS) b = NUM_BUCKETS - 1;
			if (b < 0) b = 0;
			buckets[b][labels[indices[i]]]++;
		}

		int32_t left_counts[NUM_CLASSES] = {0};
		int32_t n_left = 0;

		for (int32_t b = 0; b < NUM_BUCKETS - 1; b++) {
			int32_t bucket_total = 0;
			for (int32_t c = 0; c < NUM_CLASSES; c++) {
				left_counts[c] += buckets[b][c];
				bucket_total += buckets[b][c];
			}
			n_left += bucket_total;
			int32_t n_right = count - n_left;

			if (n_left == 0 || n_right == 0) continue;

			/* Gini impurity */
			float gini_l = 1.0f, gini_r = 1.0f;
			for (int32_t c = 0; c < NUM_CLASSES; c++) {
				float pl = (float)left_counts[c] / (float)n_left;
				gini_l -= pl * pl;
				float pr = (float)(class_counts[c] - left_counts[c]) / (float)n_right;
				gini_r -= pr * pr;
			}
			float weighted = ((float)n_left * gini_l + (float)n_right * gini_r) / (float)count;

			if (weighted < best_gini) {
				best_gini = weighted;
				best_feature = f;
				best_threshold = fmin + (b + 1.0f) / scale;
			}
		}
	}

	if (best_feature == -1) {
		tree->nodes[node] = (dt_node_t){.feature = -1, .prediction = majority};
		return node;
	}

	/* Partition indices */
	int32_t left_count = 0;
	for (int32_t i = 0; i < count; i++) {
		if (features[indices[i] * g_n_feat + best_feature] < best_threshold) {
			int32_t tmp = indices[left_count];
			indices[left_count] = indices[i];
			indices[i] = tmp;
			left_count++;
		}
	}

	if (left_count == 0 || left_count == count) {
		tree->nodes[node] = (dt_node_t){.feature = -1, .prediction = majority};
		return node;
	}

	int32_t left_child  = _dt_build(tree, features, labels, indices, left_count, depth + 1, max_depth);
	int32_t right_child = _dt_build(tree, features, labels, indices + left_count, count - left_count, depth + 1, max_depth);

	tree->nodes[node] = (dt_node_t){
		.feature   = best_feature,
		.threshold = best_threshold,
		.left      = left_child,
		.right     = right_child,
		.prediction = -1,
	};
	return node;
}

static int32_t _dt_predict(const dt_tree_t *tree, int32_t node, const float *features) {
	const dt_node_t *n = &tree->nodes[node];
	if (n->feature == -1) return n->prediction;
	if (features[n->feature] < n->threshold)
		return _dt_predict(tree, n->left, features);
	else
		return _dt_predict(tree, n->right, features);
}

static float _dt_accuracy(const dt_tree_t *tree, int32_t root, const float *features,
                          const uint8_t *labels, const uint8_t *opt_labels2,
                          const int32_t *indices, int32_t count) {
	int32_t correct = 0;
	for (int32_t i = 0; i < count; i++) {
		int32_t idx = indices[i];
		int32_t pred = _dt_predict(tree, root, features + idx * g_n_feat);
		if (pred == labels[idx]) correct++;
		else if (opt_labels2 && opt_labels2[idx] != 0xFF && pred == opt_labels2[idx]) correct++;
	}
	return (float)correct / (float)count * 100.0f;
}

static void _dt_export_ternary(const dt_tree_t *tree, int32_t node, FILE *f, int32_t indent) {
	const dt_node_t *n = &tree->nodes[node];
	if (n->feature == -1) {
		fprintf(f, "%d", n->prediction);
		return;
	}
	fprintf(f, "(f[%d] < %.6ff\n", n->feature, (double)n->threshold);
	fprintf(f, "%*s? ", indent + 4, "");
	_dt_export_ternary(tree, n->left, f, indent + 4);
	fprintf(f, "\n%*s: ", indent + 4, "");
	_dt_export_ternary(tree, n->right, f, indent + 4);
	fprintf(f, ")");
}

static void _dt_export(const dt_tree_t *tree, int32_t root, int32_t depth, float val_acc,
                       const char *filename) {
	FILE *f = fopen(filename, "w");
	if (!f) { fprintf(stderr, "Cannot write %s\n", filename); return; }

	fprintf(f, "/* decision_tree_export.h — Decision tree (depth %d, val accuracy %.1f%%)\n", depth, (double)val_acc);
	fprintf(f, " *\n");
	fprintf(f, " * Features: [0-47]=pixel RGB/255, [48]=max_range, [49]=sub_diff_h,\n");
	fprintf(f, " *   [50]=max_adj_diff, [51]=hz_diff, [52]=vt_diff,\n");
	fprintf(f, " *   [53]=diff_valid_v, [54]=diff_valid_h,\n");
	fprintf(f, " *   [55-57]=r/g/b_range, [58]=lum_range,\n");
	fprintf(f, " *   [59]=sub_diff_v, [60]=block_var, [61]=color_pair_dist,\n");
	fprintf(f, " *   [62]=cluster_tight, [63]=max_cluster_resid, [64]=cluster_bal,\n");
	fprintf(f, " *   [65]=bimodality, [66]=variance_ratio\n");
	fprintf(f, " */\n");
	fprintf(f, "#ifndef DECISION_TREE_EXPORT_H\n");
	fprintf(f, "#define DECISION_TREE_EXPORT_H\n\n");
	fprintf(f, "static inline int dt_predict(const float f[%d]) {\n", g_n_feat);
	fprintf(f, "    return ");
	_dt_export_ternary(tree, root, f, 4);
	fprintf(f, ";\n}\n\n");
	fprintf(f, "#endif /* DECISION_TREE_EXPORT_H */\n");
	fclose(f);
	printf("  Exported to %s\n", filename);
}

/* ================================================================== */
/*  Linear Classifier (53 -> 7)                                        */
/* ================================================================== */

typedef struct {
	float w[NUM_FEATURES * NUM_CLASSES]; /* max size */
	float b[NUM_CLASSES];                /* [7]    */
} linear_t;

static void _linear_init(linear_t *m) {
	float scale = sqrtf(2.0f / (float)g_n_feat);
	for (int32_t i = 0; i < g_n_feat * NUM_CLASSES; i++)
		m->w[i] = _rng_normal(0, scale);
	for (int32_t i = 0; i < NUM_CLASSES; i++)
		m->b[i] = 0;
}

static void _linear_forward(const linear_t *m, const float *x, float *out_probs) {
	float logits[NUM_CLASSES];
	for (int32_t j = 0; j < NUM_CLASSES; j++) {
		float sum = m->b[j];
		for (int32_t i = 0; i < g_n_feat; i++)
			sum += x[i] * m->w[i * NUM_CLASSES + j];
		logits[j] = sum;
	}
	_softmax(logits, out_probs, NUM_CLASSES);
}

static void _linear_train(linear_t *m, const float *data, const uint8_t *labels,
                          const int32_t *indices, int32_t count,
                          const float *class_weights,
                          float lr, int32_t batch_size, int32_t epochs) {
	float dw[NUM_FEATURES * NUM_CLASSES];
	float db[NUM_CLASSES];
	float vw[NUM_FEATURES * NUM_CLASSES];
	float vb[NUM_CLASSES];
	memset(vw, 0, sizeof(vw));
	memset(vb, 0, sizeof(vb));
	const float momentum = 0.9f;

	int32_t *shuf = malloc((size_t)count * sizeof(int32_t));
	memcpy(shuf, indices, (size_t)count * sizeof(int32_t));

	for (int32_t epoch = 0; epoch < epochs; epoch++) {
		float cur_lr = lr * 0.5f * (1.0f + cosf((float)M_PI * (float)epoch / (float)epochs));
		_shuffle_indices(shuf, count);

		for (int32_t batch_start = 0; batch_start < count; batch_start += batch_size) {
			int32_t bs = batch_start + batch_size <= count ? batch_size : count - batch_start;
			memset(dw, 0, sizeof(dw));
			memset(db, 0, sizeof(db));

			for (int32_t b = 0; b < bs; b++) {
				int32_t idx = shuf[batch_start + b];
				const float *x = data + idx * g_n_feat;
				int32_t label = labels[idx];

				float probs[NUM_CLASSES];
				_linear_forward(m, x, probs);

				float w = class_weights ? class_weights[label] : 1.0f;
				for (int32_t j = 0; j < NUM_CLASSES; j++) {
					float grad = w * (probs[j] - (j == label ? 1.0f : 0.0f));
					db[j] += grad;
					for (int32_t i = 0; i < g_n_feat; i++)
						dw[i * NUM_CLASSES + j] += x[i] * grad;
				}
			}

			/* Momentum SGD update */
			float scale = cur_lr / (float)bs;
			for (int32_t i = 0; i < g_n_feat * NUM_CLASSES; i++) {
				vw[i] = momentum * vw[i] + scale * dw[i];
				m->w[i] -= vw[i];
			}
			for (int32_t i = 0; i < NUM_CLASSES; i++) {
				vb[i] = momentum * vb[i] + scale * db[i];
				m->b[i] -= vb[i];
			}
		}
	}
	free(shuf);
}

static float _linear_accuracy(const linear_t *m, const float *data, const uint8_t *labels,
                              const uint8_t *opt_labels2,
                              const int32_t *indices, int32_t count) {
	int32_t correct = 0;
	for (int32_t i = 0; i < count; i++) {
		int32_t idx = indices[i];
		float probs[NUM_CLASSES];
		_linear_forward(m, data + idx * g_n_feat, probs);
		int32_t pred = 0;
		for (int32_t j = 1; j < NUM_CLASSES; j++)
			if (probs[j] > probs[pred]) pred = j;
		if (pred == labels[idx]) correct++;
		else if (opt_labels2 && opt_labels2[idx] != 0xFF && pred == opt_labels2[idx]) correct++;
	}
	return (float)correct / (float)count * 100.0f;
}

static void _linear_export(const linear_t *m, float val_acc, const char *filename) {
	FILE *f = fopen(filename, "w");
	if (!f) { fprintf(stderr, "Cannot write %s\n", filename); return; }

	fprintf(f, "/* linear_export.h — Linear classifier (%d->7, val accuracy %.1f%%) */\n", g_n_feat, (double)val_acc);
	fprintf(f, "#ifndef LINEAR_EXPORT_H\n");
	fprintf(f, "#define LINEAR_EXPORT_H\n\n");

	fprintf(f, "static const float linear_weights[%d] = {\n", g_n_feat * NUM_CLASSES);
	for (int32_t i = 0; i < g_n_feat * NUM_CLASSES; i++) {
		fprintf(f, "  %14.8ef,", (double)m->w[i]);
		if ((i + 1) % 7 == 0) fprintf(f, "\n");
	}
	fprintf(f, "};\n\n");

	fprintf(f, "static const float linear_bias[%d] = {\n  ", NUM_CLASSES);
	for (int32_t i = 0; i < NUM_CLASSES; i++)
		fprintf(f, "%14.8ef, ", (double)m->b[i]);
	fprintf(f, "\n};\n\n");

	fprintf(f, "#endif /* LINEAR_EXPORT_H */\n");
	fclose(f);
	printf("  Exported to %s\n", filename);
}

/* ================================================================== */
/*  Neural Network (48 -> H -> 7)                                      */
/* ================================================================== */

typedef struct {
	int32_t hidden;
	float  *w1;   /* [48 * hidden] */
	float  *b1;   /* [hidden]      */
	float  *w2;   /* [hidden * 7]  */
	float  *b2;   /* [7]           */
} nn_t;

static void _nn_init(nn_t *m, int32_t hidden) {
	m->hidden = hidden;
	m->w1 = malloc((size_t)(g_n_feat * hidden) * sizeof(float));
	m->b1 = calloc((size_t)hidden, sizeof(float));
	m->w2 = malloc((size_t)(hidden * NUM_CLASSES) * sizeof(float));
	m->b2 = calloc((size_t)NUM_CLASSES, sizeof(float));

	float s1 = sqrtf(2.0f / (float)g_n_feat);
	for (int32_t i = 0; i < g_n_feat * hidden; i++)
		m->w1[i] = _rng_normal(0, s1);
	float s2 = sqrtf(2.0f / (float)hidden);
	for (int32_t i = 0; i < hidden * NUM_CLASSES; i++)
		m->w2[i] = _rng_normal(0, s2);
}

static void _nn_free(nn_t *m) {
	free(m->w1); free(m->b1); free(m->w2); free(m->b2);
}

static void _nn_forward(const nn_t *m, const float *x, float *out_h, float *out_probs) {
	int32_t H = m->hidden;

	/* Hidden layer with ReLU */
	for (int32_t j = 0; j < H; j++) {
		float sum = m->b1[j];
		for (int32_t i = 0; i < g_n_feat; i++)
			sum += x[i] * m->w1[i * H + j];
		out_h[j] = sum > 0 ? sum : 0; /* ReLU */
	}

	/* Output layer */
	float logits[NUM_CLASSES];
	for (int32_t k = 0; k < NUM_CLASSES; k++) {
		float sum = m->b2[k];
		for (int32_t j = 0; j < H; j++)
			sum += out_h[j] * m->w2[j * NUM_CLASSES + k];
		logits[k] = sum;
	}
	_softmax(logits, out_probs, NUM_CLASSES);
}

static float _nn_accuracy(const nn_t *m, const float *data, const uint8_t *labels,
                          const uint8_t *opt_labels2,
                          const int32_t *indices, int32_t count);

static void _nn_train(nn_t *m, const float *data, const uint8_t *labels,
                      const int32_t *indices, int32_t count,
                      const int32_t *val_indices, int32_t val_count,
                      const float *class_weights,
                      float lr, int32_t batch_size, int32_t epochs) {
	int32_t H = m->hidden;
	const float momentum = 0.9f;

	float *dw1 = calloc((size_t)(g_n_feat * H), sizeof(float));
	float *db1 = calloc((size_t)H, sizeof(float));
	float *dw2 = calloc((size_t)(H * NUM_CLASSES), sizeof(float));
	float  db2[NUM_CLASSES];

	/* Momentum velocity buffers */
	float *vw1 = calloc((size_t)(g_n_feat * H), sizeof(float));
	float *vb1 = calloc((size_t)H, sizeof(float));
	float *vw2 = calloc((size_t)(H * NUM_CLASSES), sizeof(float));
	float  vb2[NUM_CLASSES];
	memset(vb2, 0, sizeof(vb2));

	float *h_buf    = malloc((size_t)H * sizeof(float));
	float *dh       = malloc((size_t)H * sizeof(float));

	int32_t *shuf = malloc((size_t)count * sizeof(int32_t));
	memcpy(shuf, indices, (size_t)count * sizeof(int32_t));

	float epoch_loss = 0;
	int32_t epoch_correct = 0;

	for (int32_t epoch = 0; epoch < epochs; epoch++) {
		epoch_loss = 0;
		epoch_correct = 0;
		float cur_lr = lr * 0.5f * (1.0f + cosf((float)M_PI * (float)epoch / (float)epochs));
		_shuffle_indices(shuf, count);

		for (int32_t batch_start = 0; batch_start < count; batch_start += batch_size) {
			int32_t bs = batch_start + batch_size <= count ? batch_size : count - batch_start;

			memset(dw1, 0, (size_t)(g_n_feat * H) * sizeof(float));
			memset(db1, 0, (size_t)H * sizeof(float));
			memset(dw2, 0, (size_t)(H * NUM_CLASSES) * sizeof(float));
			memset(db2, 0, sizeof(db2));

			for (int32_t b = 0; b < bs; b++) {
				int32_t idx = shuf[batch_start + b];
				const float *x = data + idx * g_n_feat;
				int32_t label = labels[idx];

				float probs[NUM_CLASSES];
				_nn_forward(m, x, h_buf, probs);

				/* Track loss and accuracy */
				float p = probs[label] > 1e-7f ? probs[label] : 1e-7f;
				epoch_loss -= logf(p);
				int32_t pred = 0;
				for (int32_t k = 1; k < NUM_CLASSES; k++)
					if (probs[k] > probs[pred]) pred = k;
				if (pred == label) epoch_correct++;

				float w = class_weights ? class_weights[label] : 1.0f;
				float d_out[NUM_CLASSES];
				for (int32_t k = 0; k < NUM_CLASSES; k++)
					d_out[k] = w * (probs[k] - (k == label ? 1.0f : 0.0f));

				for (int32_t j = 0; j < H; j++)
					for (int32_t k = 0; k < NUM_CLASSES; k++)
						dw2[j * NUM_CLASSES + k] += h_buf[j] * d_out[k];
				for (int32_t k = 0; k < NUM_CLASSES; k++)
					db2[k] += d_out[k];

				for (int32_t j = 0; j < H; j++) {
					float s = 0;
					for (int32_t k = 0; k < NUM_CLASSES; k++)
						s += d_out[k] * m->w2[j * NUM_CLASSES + k];
					dh[j] = (h_buf[j] > 0) ? s : 0;
				}

				for (int32_t i = 0; i < g_n_feat; i++)
					for (int32_t j = 0; j < H; j++)
						dw1[i * H + j] += x[i] * dh[j];
				for (int32_t j = 0; j < H; j++)
					db1[j] += dh[j];
			}

			/* Momentum SGD update */
			float scale = cur_lr / (float)bs;
			for (int32_t i = 0; i < g_n_feat * H; i++) {
				vw1[i] = momentum * vw1[i] + scale * dw1[i];
				m->w1[i] -= vw1[i];
			}
			for (int32_t i = 0; i < H; i++) {
				vb1[i] = momentum * vb1[i] + scale * db1[i];
				m->b1[i] -= vb1[i];
			}
			for (int32_t i = 0; i < H * NUM_CLASSES; i++) {
				vw2[i] = momentum * vw2[i] + scale * dw2[i];
				m->w2[i] -= vw2[i];
			}
			for (int32_t i = 0; i < NUM_CLASSES; i++) {
				vb2[i] = momentum * vb2[i] + scale * db2[i];
				m->b2[i] -= vb2[i];
			}
		}

		/* Per-epoch diagnostics (every 10 epochs + first + last) */
		if (epoch % 10 == 0 || epoch == epochs - 1) {
			float train_acc = (float)epoch_correct / (float)count * 100.0f;
			float avg_loss  = epoch_loss / (float)count;
			if (val_indices && val_count > 0) {
				float val_acc = _nn_accuracy(m, data, labels, NULL, val_indices, val_count);
				printf("    epoch %3d: loss=%.4f  train=%.1f%%  val=%.1f%%  lr=%.5f\n",
					epoch, (double)avg_loss, (double)train_acc, (double)val_acc, (double)cur_lr);
			} else {
				printf("    epoch %3d: loss=%.4f  train=%.1f%%  lr=%.5f\n",
					epoch, (double)avg_loss, (double)train_acc, (double)cur_lr);
			}
		}
	}

	free(dw1); free(db1); free(dw2);
	free(vw1); free(vb1); free(vw2);
	free(h_buf); free(dh); free(shuf);
}

static float _nn_accuracy(const nn_t *m, const float *data, const uint8_t *labels,
                          const uint8_t *opt_labels2,
                          const int32_t *indices, int32_t count) {
	float *h_buf = malloc((size_t)m->hidden * sizeof(float));
	int32_t correct = 0;
	for (int32_t i = 0; i < count; i++) {
		int32_t idx = indices[i];
		float probs[NUM_CLASSES];
		_nn_forward(m, data + idx * g_n_feat, h_buf, probs);
		int32_t pred = 0;
		for (int32_t j = 1; j < NUM_CLASSES; j++)
			if (probs[j] > probs[pred]) pred = j;
		if (pred == labels[idx]) correct++;
		else if (opt_labels2 && opt_labels2[idx] != 0xFF && pred == opt_labels2[idx]) correct++;
	}
	free(h_buf);
	return (float)correct / (float)count * 100.0f;
}

static void _nn_export(const nn_t *m, float val_acc, const char *filename,
                       const float *feat_mean, const float *feat_std) {
	int32_t H = m->hidden;

	/* Write C header (nn_export.h) */
	FILE *f = fopen(filename, "w");
	if (!f) { fprintf(stderr, "Cannot write %s\n", filename); return; }

	fprintf(f, "/* nn_export.h — Neural net (%d->%d->%d, val accuracy %.1f%%) */\n",
		g_n_feat, H, NUM_CLASSES, (double)val_acc);
	fprintf(f, "#ifndef NN_EXPORT_H\n");
	fprintf(f, "#define NN_EXPORT_H\n\n");

	/* Shared: write arrays to a file handle */
	#define WRITE_ARRAYS(fh) do { \
		fprintf(fh, "static const float nn_w1[%d * %d] = {\n", g_n_feat, H); \
		for (int32_t i = 0; i < g_n_feat * H; i++) { \
			fprintf(fh, "  %14.8ef,", (double)m->w1[i]); \
			if ((i + 1) % 8 == 0) fprintf(fh, "\n"); \
		} fprintf(fh, "};\n\n"); \
		fprintf(fh, "static const float nn_b1[%d] = {\n  ", H); \
		for (int32_t i = 0; i < H; i++) { \
			fprintf(fh, "%14.8ef, ", (double)m->b1[i]); \
			if ((i + 1) % 8 == 0 && i + 1 < H) fprintf(fh, "\n  "); \
		} fprintf(fh, "\n};\n\n"); \
		fprintf(fh, "static const float nn_w2[%d * %d] = {\n", H, NUM_CLASSES); \
		for (int32_t i = 0; i < H * NUM_CLASSES; i++) { \
			fprintf(fh, "  %14.8ef,", (double)m->w2[i]); \
			if ((i + 1) % NUM_CLASSES == 0) fprintf(fh, "\n"); \
		} fprintf(fh, "};\n\n"); \
		fprintf(fh, "static const float nn_b2[%d] = {\n  ", NUM_CLASSES); \
		for (int32_t i = 0; i < NUM_CLASSES; i++) \
			fprintf(fh, "%14.8ef, ", (double)m->b2[i]); \
		fprintf(fh, "\n};\n\n"); \
		if (feat_mean && feat_std) { \
			fprintf(fh, "static const float nn_feat_mean[%d] = {\n  ", g_n_feat); \
			for (int32_t i = 0; i < g_n_feat; i++) { \
				fprintf(fh, "%14.8ef, ", (double)feat_mean[i]); \
				if ((i + 1) % 5 == 0 && i + 1 < g_n_feat) fprintf(fh, "\n  "); \
			} fprintf(fh, "\n};\n\n"); \
			fprintf(fh, "static const float nn_feat_std[%d] = {\n  ", g_n_feat); \
			for (int32_t i = 0; i < g_n_feat; i++) { \
				fprintf(fh, "%14.8ef, ", (double)feat_std[i]); \
				if ((i + 1) % 5 == 0 && i + 1 < g_n_feat) fprintf(fh, "\n  "); \
			} fprintf(fh, "\n};\n\n"); \
		} \
	} while(0)

	WRITE_ARRAYS(f);
	fprintf(f, "#endif /* NN_EXPORT_H */\n");
	fclose(f);
	printf("  Exported to %s\n", filename);

	/* Write HLSL include (etc2_nn_weights.hlsli) for the shader */
	const char *hlsli_path = "../../assets/shaders/etc2_nn_weights.hlsli";
	FILE *fh = fopen(hlsli_path, "w");
	if (!fh) { fprintf(stderr, "Cannot write %s\n", hlsli_path); return; }
	fprintf(fh, "// Auto-generated by train.c — %d->%d->%d mode classifier (%.1f%% acc)\n",
		g_n_feat, H, NUM_CLASSES, (double)val_acc);
	fprintf(fh, "// Classes: 0=diff, 1=individual, 2=planar, 3=T, 4=H\n\n");
	WRITE_ARRAYS(fh);
	fclose(fh);
	printf("  Exported to %s\n", hlsli_path);

	#undef WRITE_ARRAYS
}

/* ================================================================== */
/*  Two-Layer Neural Network (N -> H1 -> H2 -> 7)                      */
/* ================================================================== */

typedef struct {
	int32_t h1, h2;
	float *w1, *b1;  /* [n_feat × h1], [h1] */
	float *w2, *b2;  /* [h1 × h2],     [h2] */
	float *w3, *b3;  /* [h2 × 7],      [7]  */
} nn2_t;

static void _nn2_init(nn2_t *m, int32_t h1, int32_t h2) {
	m->h1 = h1; m->h2 = h2;
	m->w1 = malloc((size_t)(g_n_feat * h1) * sizeof(float));
	m->b1 = calloc((size_t)h1, sizeof(float));
	m->w2 = malloc((size_t)(h1 * h2) * sizeof(float));
	m->b2 = calloc((size_t)h2, sizeof(float));
	m->w3 = malloc((size_t)(h2 * NUM_CLASSES) * sizeof(float));
	m->b3 = calloc((size_t)NUM_CLASSES, sizeof(float));

	float s1 = sqrtf(2.0f / (float)g_n_feat);
	for (int32_t i = 0; i < g_n_feat * h1; i++) m->w1[i] = _rng_normal(0, s1);
	float s2 = sqrtf(2.0f / (float)h1);
	for (int32_t i = 0; i < h1 * h2; i++) m->w2[i] = _rng_normal(0, s2);
	float s3 = sqrtf(2.0f / (float)h2);
	for (int32_t i = 0; i < h2 * NUM_CLASSES; i++) m->w3[i] = _rng_normal(0, s3);
}

static void _nn2_free(nn2_t *m) {
	free(m->w1); free(m->b1); free(m->w2); free(m->b2); free(m->w3); free(m->b3);
}

static void _nn2_forward(const nn2_t *m, const float *x,
                         float *out_a1, float *out_a2, float *out_probs) {
	/* Layer 1 */
	for (int32_t j = 0; j < m->h1; j++) {
		float sum = m->b1[j];
		for (int32_t i = 0; i < g_n_feat; i++)
			sum += x[i] * m->w1[i * m->h1 + j];
		out_a1[j] = sum > 0 ? sum : 0;
	}
	/* Layer 2 */
	for (int32_t j = 0; j < m->h2; j++) {
		float sum = m->b2[j];
		for (int32_t i = 0; i < m->h1; i++)
			sum += out_a1[i] * m->w2[i * m->h2 + j];
		out_a2[j] = sum > 0 ? sum : 0;
	}
	/* Output */
	float logits[NUM_CLASSES];
	for (int32_t k = 0; k < NUM_CLASSES; k++) {
		float sum = m->b3[k];
		for (int32_t j = 0; j < m->h2; j++)
			sum += out_a2[j] * m->w3[j * NUM_CLASSES + k];
		logits[k] = sum;
	}
	_softmax(logits, out_probs, NUM_CLASSES);
}

static float _nn2_accuracy(const nn2_t *m, const float *data, const uint8_t *labels,
                           const int32_t *indices, int32_t count) {
	float *a1 = malloc((size_t)m->h1 * sizeof(float));
	float *a2 = malloc((size_t)m->h2 * sizeof(float));
	int32_t correct = 0;
	for (int32_t i = 0; i < count; i++) {
		float probs[NUM_CLASSES];
		_nn2_forward(m, data + indices[i] * g_n_feat, a1, a2, probs);
		int32_t pred = 0;
		for (int32_t j = 1; j < NUM_CLASSES; j++)
			if (probs[j] > probs[pred]) pred = j;
		if (pred == labels[indices[i]]) correct++;
	}
	free(a1); free(a2);
	return (float)correct / (float)count * 100.0f;
}

static void _nn2_train(nn2_t *m, const float *data, const uint8_t *labels,
                       const int32_t *indices, int32_t count,
                       const int32_t *val_indices, int32_t val_count,
                       float lr, int32_t batch_size, int32_t epochs) {
	int32_t H1 = m->h1, H2 = m->h2;
	const float momentum = 0.9f;

	/* Gradient buffers */
	float *dw1 = calloc((size_t)(g_n_feat * H1), sizeof(float));
	float *db1 = calloc((size_t)H1, sizeof(float));
	float *dw2 = calloc((size_t)(H1 * H2), sizeof(float));
	float *db2 = calloc((size_t)H2, sizeof(float));
	float *dw3 = calloc((size_t)(H2 * NUM_CLASSES), sizeof(float));
	float  db3[NUM_CLASSES];

	/* Velocity buffers */
	float *vw1 = calloc((size_t)(g_n_feat * H1), sizeof(float));
	float *vb1 = calloc((size_t)H1, sizeof(float));
	float *vw2 = calloc((size_t)(H1 * H2), sizeof(float));
	float *vb2 = calloc((size_t)H2, sizeof(float));
	float *vw3 = calloc((size_t)(H2 * NUM_CLASSES), sizeof(float));
	float  vb3[NUM_CLASSES]; memset(vb3, 0, sizeof(vb3));

	float *a1  = malloc((size_t)H1 * sizeof(float));
	float *a2  = malloc((size_t)H2 * sizeof(float));
	float *dh1 = malloc((size_t)H1 * sizeof(float));
	float *dh2 = malloc((size_t)H2 * sizeof(float));

	int32_t *shuf = malloc((size_t)count * sizeof(int32_t));
	memcpy(shuf, indices, (size_t)count * sizeof(int32_t));

	for (int32_t epoch = 0; epoch < epochs; epoch++) {
		float cur_lr = lr * 0.5f * (1.0f + cosf((float)M_PI * (float)epoch / (float)epochs));
		_shuffle_indices(shuf, count);
		float epoch_loss = 0;
		int32_t epoch_correct = 0;

		for (int32_t batch_start = 0; batch_start < count; batch_start += batch_size) {
			int32_t bs = batch_start + batch_size <= count ? batch_size : count - batch_start;

			memset(dw1, 0, (size_t)(g_n_feat * H1) * sizeof(float));
			memset(db1, 0, (size_t)H1 * sizeof(float));
			memset(dw2, 0, (size_t)(H1 * H2) * sizeof(float));
			memset(db2, 0, (size_t)H2 * sizeof(float));
			memset(dw3, 0, (size_t)(H2 * NUM_CLASSES) * sizeof(float));
			memset(db3, 0, sizeof(db3));

			for (int32_t b = 0; b < bs; b++) {
				int32_t idx = shuf[batch_start + b];
				const float *x = data + idx * g_n_feat;
				int32_t label = labels[idx];

				float probs[NUM_CLASSES];
				_nn2_forward(m, x, a1, a2, probs);

				float p = probs[label] > 1e-7f ? probs[label] : 1e-7f;
				epoch_loss -= logf(p);
				int32_t pred = 0;
				for (int32_t k = 1; k < NUM_CLASSES; k++)
					if (probs[k] > probs[pred]) pred = k;
				if (pred == label) epoch_correct++;

				/* Backprop: output gradient */
				float d_out[NUM_CLASSES];
				for (int32_t k = 0; k < NUM_CLASSES; k++)
					d_out[k] = probs[k] - (k == label ? 1.0f : 0.0f);

				/* dw3, db3 */
				for (int32_t j = 0; j < H2; j++)
					for (int32_t k = 0; k < NUM_CLASSES; k++)
						dw3[j * NUM_CLASSES + k] += a2[j] * d_out[k];
				for (int32_t k = 0; k < NUM_CLASSES; k++)
					db3[k] += d_out[k];

				/* dh2 (through ReLU) */
				for (int32_t j = 0; j < H2; j++) {
					float s = 0;
					for (int32_t k = 0; k < NUM_CLASSES; k++)
						s += d_out[k] * m->w3[j * NUM_CLASSES + k];
					dh2[j] = (a2[j] > 0) ? s : 0;
				}

				/* dw2, db2 */
				for (int32_t i = 0; i < H1; i++)
					for (int32_t j = 0; j < H2; j++)
						dw2[i * H2 + j] += a1[i] * dh2[j];
				for (int32_t j = 0; j < H2; j++)
					db2[j] += dh2[j];

				/* dh1 (through ReLU) */
				for (int32_t i = 0; i < H1; i++) {
					float s = 0;
					for (int32_t j = 0; j < H2; j++)
						s += dh2[j] * m->w2[i * H2 + j];
					dh1[i] = (a1[i] > 0) ? s : 0;
				}

				/* dw1, db1 */
				for (int32_t f = 0; f < g_n_feat; f++)
					for (int32_t i = 0; i < H1; i++)
						dw1[f * H1 + i] += x[f] * dh1[i];
				for (int32_t i = 0; i < H1; i++)
					db1[i] += dh1[i];
			}

			/* Momentum SGD */
			float scale = cur_lr / (float)bs;
			for (int32_t i = 0; i < g_n_feat * H1; i++) { vw1[i] = momentum*vw1[i] + scale*dw1[i]; m->w1[i] -= vw1[i]; }
			for (int32_t i = 0; i < H1; i++)             { vb1[i] = momentum*vb1[i] + scale*db1[i]; m->b1[i] -= vb1[i]; }
			for (int32_t i = 0; i < H1 * H2; i++)        { vw2[i] = momentum*vw2[i] + scale*dw2[i]; m->w2[i] -= vw2[i]; }
			for (int32_t i = 0; i < H2; i++)              { vb2[i] = momentum*vb2[i] + scale*db2[i]; m->b2[i] -= vb2[i]; }
			for (int32_t i = 0; i < H2 * NUM_CLASSES; i++){ vw3[i] = momentum*vw3[i] + scale*dw3[i]; m->w3[i] -= vw3[i]; }
			for (int32_t i = 0; i < NUM_CLASSES; i++)     { vb3[i] = momentum*vb3[i] + scale*db3[i]; m->b3[i] -= vb3[i]; }
		}

		if (epoch % 10 == 0 || epoch == epochs - 1) {
			float train_acc = (float)epoch_correct / (float)count * 100.0f;
			float avg_loss  = epoch_loss / (float)count;
			float val_acc = _nn2_accuracy(m, data, labels, val_indices, val_count);
			printf("    epoch %3d: loss=%.4f  train=%.1f%%  val=%.1f%%  lr=%.5f\n",
				epoch, (double)avg_loss, (double)train_acc, (double)val_acc, (double)cur_lr);
		}
	}

	free(dw1); free(db1); free(dw2); free(db2); free(dw3);
	free(vw1); free(vb1); free(vw2); free(vb2); free(vw3);
	free(a1); free(a2); free(dh1); free(dh2); free(shuf);
}

/* ================================================================== */
/*  Confusion Matrix                                                   */
/* ================================================================== */

static void _print_confusion(const char *model_name, const float *data, const uint8_t *labels,
                             const int32_t *indices, int32_t count,
                             /* prediction function: returns class given sample index */
                             int32_t (*predict_fn)(const void *model, const float *x),
                             const void *model) {
	static const char *mode_names[5] = {"diff ","ind  ","plan ","T    ","H    "};
	int32_t cm[NUM_CLASSES][NUM_CLASSES] = {{0}};

	for (int32_t i = 0; i < count; i++) {
		const float *x = data + indices[i] * g_n_feat;
		int32_t pred = predict_fn(model, x);
		int32_t true_label = labels[indices[i]];
		cm[true_label][pred]++;
	}

	printf("\nConfusion matrix (%s):\n", model_name);
	printf("           ");
	for (int32_t j = 0; j < NUM_CLASSES; j++)
		printf(" %5s", mode_names[j]);
	printf("  |  acc\n");

	for (int32_t i = 0; i < NUM_CLASSES; i++) {
		int32_t row_total = 0, row_correct = 0;
		for (int32_t j = 0; j < NUM_CLASSES; j++) row_total += cm[i][j];
		row_correct = cm[i][i];
		printf("True %s: ", mode_names[i]);
		for (int32_t j = 0; j < NUM_CLASSES; j++)
			printf(" %5d", cm[i][j]);
		printf("  | %5.1f%%\n", row_total > 0 ? (float)row_correct / (float)row_total * 100.0f : 0.0f);
	}
	printf("\n");
}

/* Wrapper predict functions for confusion matrix */
typedef struct { const nn_t *nn; float *h_buf; } _nn_predict_ctx;

static int32_t _predict_linear(const void *model, const float *x) {
	float probs[NUM_CLASSES];
	_linear_forward((const linear_t *)model, x, probs);
	int32_t pred = 0;
	for (int32_t j = 1; j < NUM_CLASSES; j++)
		if (probs[j] > probs[pred]) pred = j;
	return pred;
}

static int32_t _predict_nn(const void *model, const float *x) {
	const _nn_predict_ctx *ctx = (const _nn_predict_ctx *)model;
	float probs[NUM_CLASSES];
	_nn_forward(ctx->nn, x, ctx->h_buf, probs);
	int32_t pred = 0;
	for (int32_t j = 1; j < NUM_CLASSES; j++)
		if (probs[j] > probs[pred]) pred = j;
	return pred;
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <training_data.bin> [--val-split 0.2] [--hidden 16,32,64]\n", argv[0]);
		return 1;
	}

	const char *data_path = argv[1];
	float val_split = 0.2f;
	int32_t hidden_sizes[16] = {16};
	int32_t num_hidden = 1;
	int32_t max_samples = 0; /* 0 = use all */
	int32_t num_epochs = 200;
	float   learn_rate = 0.05f;
	int32_t derived_only = 0;
	int32_t raw_only     = 0;

	for (int32_t i = 2; i < argc; i++) {
		if (strcmp(argv[i], "--val-split") == 0 && i + 1 < argc) {
			val_split = (float)atof(argv[++i]);
		} else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
			i++;
			num_hidden = 0;
			char buf[256];
			strncpy(buf, argv[i], sizeof(buf) - 1);
			buf[sizeof(buf) - 1] = 0;
			char *tok = strtok(buf, ",");
			while (tok && num_hidden < 16) {
				int32_t h = atoi(tok);
				if (h > 0) hidden_sizes[num_hidden++] = h;
				tok = strtok(NULL, ",");
			}
		} else if (strcmp(argv[i], "--max-samples") == 0 && i + 1 < argc) {
			max_samples = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
			num_epochs = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
			learn_rate = (float)atof(argv[++i]);
		} else if (strcmp(argv[i], "--derived-only") == 0) {
			derived_only = 1;
		} else if (strcmp(argv[i], "--raw-only") == 0) {
			raw_only = 1;
		}
	}

	/* ---- Load data ---- */
	FILE *f = fopen(data_path, "rb");
	if (!f) { fprintf(stderr, "Cannot open %s\n", data_path); return 1; }

	uint32_t magic, block_count;
	if (fread(&magic, 4, 1, f) != 1 || fread(&block_count, 4, 1, f) != 1) {
		fprintf(stderr, "Failed to read header\n");
		fclose(f);
		return 1;
	}
	if (magic != 0x45544332 && magic != 0x45544333) {
		fprintf(stderr, "Invalid magic (expected 0x45544332/33, got 0x%08X)\n", magic);
		fclose(f);
		return 1;
	}
	/* Strided sampling: if max_samples < total, read every Nth block
	 * to spread evenly across all images instead of front-loading */
	uint32_t stride = 1;
	uint32_t load_count = block_count;
	if (max_samples > 0 && block_count > (uint32_t)max_samples) {
		stride = block_count / (uint32_t)max_samples;
		if (stride < 1) stride = 1;
		load_count = block_count / stride;
		if (load_count > (uint32_t)max_samples) load_count = (uint32_t)max_samples;
	}
	printf("Loading %u blocks (stride %u from %u total)...\n", load_count, stride, block_count);

	/* Detect format version from magic */
	int32_t has_label2 = (magic == 0x45544333); /* ETC3 = v2 with label2 */
	int32_t record_size = has_label2 ? 50 : 49;

	uint8_t *raw_pixels = malloc((size_t)load_count * 48);
	uint8_t *labels     = malloc((size_t)load_count);
	uint8_t *labels2    = malloc((size_t)load_count);
	uint8_t skip_buf[50];
	uint32_t loaded = 0;
	for (uint32_t i = 0; i < block_count && loaded < load_count; i++) {
		if (i % stride == 0) {
			if (fread(raw_pixels + loaded * 48, 48, 1, f) != 1 ||
			    fread(labels + loaded, 1, 1, f) != 1) {
				break;
			}
			if (has_label2) {
				if (fread(labels2 + loaded, 1, 1, f) != 1) break;
			} else {
				labels2[loaded] = 0xFF;
			}
			loaded++;
		} else {
			if (fread(skip_buf, (size_t)record_size, 1, f) != 1) break;
		}
	}
	fclose(f);
	block_count = loaded;

	/* Report label2 stats */
	{
		int32_t n_with_alt = 0;
		for (uint32_t i = 0; i < block_count; i++)
			if (labels2[i] != 0xFF) n_with_alt++;
		printf("Loaded %u blocks (%d have 2nd-best within 2.5%%)\n",
			block_count, n_with_alt);
	}

	/* ---- Normalize to floats ---- */
	float *data_f = malloc((size_t)block_count * NUM_RAW_FEATURES * sizeof(float));
	for (uint32_t i = 0; i < block_count * 48; i++)
		data_f[i] = (float)raw_pixels[i] / 255.0f;
	/* Keep raw_pixels for SSE error analysis later */

	/* ---- Compute derived features (all classifiers use 53 features) ---- */
	/* Always build full feature array first, then optionally compact */
	float *dt_features = malloc((size_t)block_count * NUM_FEATURES * sizeof(float));
	for (uint32_t i = 0; i < block_count; i++) {
		memcpy(dt_features + i * NUM_FEATURES, data_f + i * NUM_RAW_FEATURES,
		       NUM_RAW_FEATURES * sizeof(float));
		_compute_derived(data_f + i * NUM_RAW_FEATURES,
		                 dt_features + i * NUM_FEATURES + NUM_RAW_FEATURES);
	}
	free(data_f);

	/* If --derived-only, repack to only the 19 derived features */
	if (derived_only) {
		g_n_feat = NUM_DERIVED;
		float *compact = malloc((size_t)block_count * NUM_DERIVED * sizeof(float));
		for (uint32_t i = 0; i < block_count; i++)
			memcpy(compact + i * NUM_DERIVED,
			       dt_features + i * NUM_FEATURES + NUM_RAW_FEATURES,
			       NUM_DERIVED * sizeof(float));
		free(dt_features);
		dt_features = compact;
		printf("Derived-only mode: using %d features (no raw pixels)\n", g_n_feat);
	} else if (raw_only) {
		g_n_feat = NUM_RAW_FEATURES;
		float *compact = malloc((size_t)block_count * NUM_RAW_FEATURES * sizeof(float));
		for (uint32_t i = 0; i < block_count; i++)
			memcpy(compact + i * NUM_RAW_FEATURES,
			       dt_features + i * NUM_FEATURES,
			       NUM_RAW_FEATURES * sizeof(float));
		free(dt_features);
		dt_features = compact;
		printf("Raw-only mode: using %d features (16 RGB pixels)\n", g_n_feat);
	}

	/* ---- Shuffle and split ---- */
	int32_t N = (int32_t)block_count;
	int32_t *all_indices = malloc((size_t)N * sizeof(int32_t));
	for (int32_t i = 0; i < N; i++) all_indices[i] = i;
	_rng_state = 42; /* reset seed for reproducibility */
	_shuffle_indices(all_indices, N);

	int32_t n_val   = (int32_t)((float)N * val_split);
	int32_t n_train = N - n_val;
	int32_t *train_idx = all_indices;
	int32_t *val_idx   = all_indices + n_train;

	/* ---- Feature standardization (compute on train set, apply to all) ---- */
	float feat_mean[NUM_FEATURES] = {0};
	float feat_std[NUM_FEATURES]  = {0};
	for (int32_t i = 0; i < n_train; i++)
		for (int32_t f = 0; f < g_n_feat; f++)
			feat_mean[f] += dt_features[train_idx[i] * g_n_feat + f];
	for (int32_t f = 0; f < g_n_feat; f++)
		feat_mean[f] /= (float)n_train;

	for (int32_t i = 0; i < n_train; i++)
		for (int32_t f = 0; f < g_n_feat; f++) {
			float d = dt_features[train_idx[i] * g_n_feat + f] - feat_mean[f];
			feat_std[f] += d * d;
		}
	for (int32_t f = 0; f < g_n_feat; f++) {
		feat_std[f] = sqrtf(feat_std[f] / (float)n_train);
		if (feat_std[f] < 1e-6f) feat_std[f] = 1e-6f;
	}

	/* Apply standardization to ALL samples */
	for (int32_t i = 0; i < N; i++)
		for (int32_t f = 0; f < g_n_feat; f++)
			dt_features[i * g_n_feat + f] =
				(dt_features[i * g_n_feat + f] - feat_mean[f]) / feat_std[f];

	printf("Feature standardization applied (mean/std computed from %d training samples)\n", n_train);

	/* ---- Mode distribution ---- */
	int32_t mode_counts[NUM_CLASSES] = {0};
	for (int32_t i = 0; i < N; i++)
		mode_counts[labels[i]]++;

	printf("\n=== Results (%d training, %d validation blocks) ===\n\n", n_train, n_val);
	printf("Mode distribution: [diff=%d%% ind=%d%% plan=%d%% T=%d%% H=%d%%]\n",
		mode_counts[0]*100/N, mode_counts[1]*100/N, mode_counts[2]*100/N,
		mode_counts[3]*100/N, mode_counts[4]*100/N);

	/* Class weights: sqrt of inverse frequency (soft rebalancing) */
	float class_weights[NUM_CLASSES];
	for (int32_t c = 0; c < NUM_CLASSES; c++) {
		if (mode_counts[c] > 0)
			class_weights[c] = sqrtf((float)N / ((float)NUM_CLASSES * (float)mode_counts[c]));
		else
			class_weights[c] = 1.0f;
	}
	printf("Class weights: [");
	for (int32_t c = 0; c < NUM_CLASSES; c++)
		printf("%.2f%s", (double)class_weights[c], c < 6 ? ", " : "");
	printf("]\n\n");

	/* ---- Decision Trees ---- */
	float best_dt_val = 0;
	int32_t best_dt_depth = 4;
	dt_tree_t best_dt_tree;

	for (int32_t depth = 7; depth <= 7; depth++) {
		dt_tree_t tree = {.node_count = 0};

		/* Need a copy of train indices since build modifies them in-place */
		int32_t *idx_copy = malloc((size_t)n_train * sizeof(int32_t));
		memcpy(idx_copy, train_idx, (size_t)n_train * sizeof(int32_t));

		int32_t root = _dt_build(&tree, dt_features, labels, idx_copy, n_train, 0, depth);
		free(idx_copy);

		float train_acc = _dt_accuracy(&tree, root, dt_features, labels, NULL, train_idx, n_train);
		float val_acc   = _dt_accuracy(&tree, root, dt_features, labels, NULL, val_idx, n_val);
		float val_acc_r = _dt_accuracy(&tree, root, dt_features, labels, labels2, val_idx, n_val);
		printf("Decision Tree depth=%d:  train=%.1f%%  val=%.1f%%  val(relaxed)=%.1f%%  (%d nodes)\n",
			depth, (double)train_acc, (double)val_acc, (double)val_acc_r, tree.node_count);

		if (val_acc > best_dt_val) {
			best_dt_val = val_acc;
			best_dt_depth = depth;
			best_dt_tree = tree;
		}
	}

	printf("\n");

	/* ---- Linear Classifier ---- */
	_rng_state = 42;
	linear_t linear;
	_linear_init(&linear);
	printf("Training linear classifier (%d->7)...\n", g_n_feat);
	_linear_train(&linear, dt_features, labels, train_idx, n_train, NULL, learn_rate, 256, num_epochs);
	float lin_train = _linear_accuracy(&linear, dt_features, labels, NULL, train_idx, n_train);
	float lin_val   = _linear_accuracy(&linear, dt_features, labels, NULL, val_idx, n_val);
	float lin_val_r = _linear_accuracy(&linear, dt_features, labels, labels2, val_idx, n_val);
	printf("Linear (%d->7):        train=%.1f%%  val=%.1f%%  val(relaxed)=%.1f%%\n\n",
		g_n_feat, (double)lin_train, (double)lin_val, (double)lin_val_r);

	/* ---- Neural Networks ---- */
	float best_nn_val = 0;
	int32_t best_nn_idx = 0;
	nn_t *nns = malloc((size_t)num_hidden * sizeof(nn_t));

	for (int32_t hi = 0; hi < num_hidden; hi++) {
		int32_t H = hidden_sizes[hi];
		_rng_state = 42;
		_nn_init(&nns[hi], H);
		printf("Training NN (%d->%d->7)...\n", g_n_feat, H);
		_nn_train(&nns[hi], dt_features, labels, train_idx, n_train, val_idx, n_val, NULL, learn_rate, 256, num_epochs);
		float train_acc   = _nn_accuracy(&nns[hi], dt_features, labels, NULL, train_idx, n_train);
		float val_acc     = _nn_accuracy(&nns[hi], dt_features, labels, NULL, val_idx, n_val);
		float val_acc_r   = _nn_accuracy(&nns[hi], dt_features, labels, labels2, val_idx, n_val);
		printf("NN (%d->%d->7):       train=%.1f%%  val=%.1f%%  val(relaxed)=%.1f%%\n",
			g_n_feat, H, (double)train_acc, (double)val_acc, (double)val_acc_r);

		if (val_acc > best_nn_val) {
			best_nn_val = val_acc;
			best_nn_idx = hi;
		}
	}

	/* ---- Two-Layer Neural Networks ---- */
	printf("\n");
	int32_t nn2_configs[][2] = {{16,8}, {32,16}, {32,32}};
	int32_t num_nn2 = 3;
	for (int32_t ci = 0; ci < num_nn2; ci++) {
		int32_t h1 = nn2_configs[ci][0], h2 = nn2_configs[ci][1];
		_rng_state = 42;
		nn2_t net2;
		_nn2_init(&net2, h1, h2);
		printf("Training NN2 (%d->%d->%d->7)...\n", g_n_feat, h1, h2);
		_nn2_train(&net2, dt_features, labels, train_idx, n_train, val_idx, n_val, learn_rate, 256, num_epochs);
		float train_acc = _nn2_accuracy(&net2, dt_features, labels, train_idx, n_train);
		float val_acc   = _nn2_accuracy(&net2, dt_features, labels, val_idx, n_val);
		printf("NN2 (%d->%d->%d->7):  train=%.1f%%  val=%.1f%%\n\n",
			g_n_feat, h1, h2, (double)train_acc, (double)val_acc);
		_nn2_free(&net2);
	}

	/* ---- Find overall best model for confusion matrix ---- */
	printf("\n--- Best models ---\n");
	printf("  Decision Tree: depth=%d (val %.1f%%)\n", best_dt_depth, (double)best_dt_val);
	printf("  Linear:        %d->7  (val %.1f%%)\n", g_n_feat, (double)lin_val);
	printf("  Neural Net:    %d->%d->7 (val %.1f%%)\n", g_n_feat, hidden_sizes[best_nn_idx], (double)best_nn_val);

	/* Confusion matrix for the best overall model */
	float best_overall = best_dt_val;
	if (lin_val > best_overall) best_overall = lin_val;
	if (best_nn_val > best_overall) best_overall = best_nn_val;

	if (best_overall == best_nn_val) {
		_nn_predict_ctx ctx = {.nn = &nns[best_nn_idx],
		                       .h_buf = malloc((size_t)nns[best_nn_idx].hidden * sizeof(float))};
		char name[64];
		snprintf(name, sizeof(name), "NN %d->%d->7", g_n_feat, hidden_sizes[best_nn_idx]);
		_print_confusion(name, dt_features, labels, val_idx, n_val, _predict_nn, &ctx);
		free(ctx.h_buf);
	} else if (best_overall == lin_val) {
		char name[64];
		snprintf(name, sizeof(name), "Linear %d->7", g_n_feat);
		_print_confusion(name, dt_features, labels, val_idx, n_val, _predict_linear, &linear);
	}
	/* DT confusion matrix would need a different wrapper; skip for simplicity
	 * since NNs are expected to win */

	/* ---- Feature importance ---- */
	{
		nn_t *best_nn = &nns[best_nn_idx];
		int32_t H = best_nn->hidden;

		static const char *feat_names[] = {
			"R0","G0","B0","R1","G1","B1","R2","G2","B2","R3","G3","B3",
			"R4","G4","B4","R5","G5","B5","R6","G6","B6","R7","G7","B7",
			"R8","G8","B8","R9","G9","B9","R10","G10","B10","R11","G11","B11",
			"R12","G12","B12","R13","G13","B13","R14","G14","B14","R15","G15","B15",
			"max_range","sub_diff_h","max_adj","hz_diff","vt_diff",
			"diff_val_v","diff_val_h",
			"r_range","g_range","b_range","lum_range",
			"sub_diff_v","blk_var","cpair_dist",
			"clust_tight","max_resid","clust_bal","bimodal","var_ratio",
			"tight*cpd","resid*vr","range*vr","bimod*cpd","lum*tight","grad*smth",
		};

		int32_t name_offset = derived_only ? NUM_RAW_FEATURES : 0;

		/* --- NN: propagated importance (|w1| * |w2| through both layers) --- */
		float nn_imp[NUM_FEATURES];
		float nn_max = 0;
		for (int32_t f = 0; f < g_n_feat; f++) {
			float sum = 0;
			for (int32_t j = 0; j < H; j++) {
				float w1_abs = best_nn->w1[f * H + j];
				if (w1_abs < 0) w1_abs = -w1_abs;
				float w2_sum = 0;
				for (int32_t k = 0; k < NUM_CLASSES; k++) {
					float w2 = best_nn->w2[j * NUM_CLASSES + k];
					w2_sum += (w2 < 0 ? -w2 : w2);
				}
				sum += w1_abs * w2_sum;
			}
			nn_imp[f] = sum;
			if (sum > nn_max) nn_max = sum;
		}

		/* --- Linear: direct L1 norm per feature --- */
		float lin_imp[NUM_FEATURES];
		float lin_max = 0;
		for (int32_t f = 0; f < g_n_feat; f++) {
			float sum = 0;
			for (int32_t j = 0; j < NUM_CLASSES; j++) {
				float w = linear.w[f * NUM_CLASSES + j];
				sum += (w < 0 ? -w : w);
			}
			lin_imp[f] = sum;
			if (sum > lin_max) lin_max = sum;
		}

		/* Sort by NN propagated importance */
		int32_t order[NUM_FEATURES];
		for (int32_t i = 0; i < g_n_feat; i++) order[i] = i;
		for (int32_t i = 0; i < g_n_feat - 1; i++)
			for (int32_t j = i + 1; j < g_n_feat; j++)
				if (nn_imp[order[j]] > nn_imp[order[i]]) {
					int32_t tmp = order[i]; order[i] = order[j]; order[j] = tmp;
				}

		printf("\n=== Feature Importance ===\n\n");
		printf("  %-12s  %8s %5s  %8s %5s  %s\n",
			"Feature", "NN Prop", "Rel%", "Linear", "Rel%", "NN / Linear");
		printf("  %-12s  %8s %5s  %8s %5s  %s\n",
			"------------", "--------", "-----", "--------", "-----", "-----------");
		for (int32_t i = 0; i < g_n_feat; i++) {
			int32_t f = order[i];
			float nn_rel  = nn_imp[f] / nn_max * 100.0f;
			float lin_rel = lin_imp[f] / lin_max * 100.0f;
			int32_t nn_bar  = (int32_t)(nn_rel / 10.0f + 0.5f);
			int32_t lin_bar = (int32_t)(lin_rel / 10.0f + 0.5f);
			char bar_nn[11] = {0}, bar_lin[11] = {0};
			for (int32_t b = 0; b < nn_bar && b < 10; b++) bar_nn[b] = '#';
			for (int32_t b = 0; b < lin_bar && b < 10; b++) bar_lin[b] = '#';
			int32_t ni = f + name_offset;
			printf("  %-12s  %8.1f %4.0f%%  %8.1f %4.0f%%  %-10s %-10s\n",
				ni < NUM_FEATURES ? feat_names[ni] : "?",
				(double)nn_imp[f], (double)nn_rel,
				(double)lin_imp[f], (double)lin_rel,
				bar_nn, bar_lin);
		}
		printf("\n");

		/* --- Hidden unit analysis: what is each unit computing? --- */
		static const char *class_names[5] = {"df","in","pl","T ","H "};
		printf("=== Hidden Unit Analysis (what each ReLU neuron detects) ===\n\n");
		for (int32_t j = 0; j < H; j++) {
			/* Find top 3 positive and top 3 negative input weights */
			int32_t top_pos[3] = {-1,-1,-1};
			int32_t top_neg[3] = {-1,-1,-1};
			float   top_pos_w[3] = {0,0,0};
			float   top_neg_w[3] = {0,0,0};
			for (int32_t f = 0; f < g_n_feat; f++) {
				float w = best_nn->w1[f * H + j];
				for (int32_t t = 0; t < 3; t++) {
					if (w > top_pos_w[t]) {
						for (int32_t s = 2; s > t; s--) { top_pos[s] = top_pos[s-1]; top_pos_w[s] = top_pos_w[s-1]; }
						top_pos[t] = f; top_pos_w[t] = w; break;
					}
				}
				if (w < 0) {
					float aw = -w;
					for (int32_t t = 0; t < 3; t++) {
						if (aw > -top_neg_w[t]) {
							for (int32_t s = 2; s > t; s--) { top_neg[s] = top_neg[s-1]; top_neg_w[s] = top_neg_w[s-1]; }
							top_neg[t] = f; top_neg_w[t] = w; break;
						}
					}
				}
			}

			/* Output weights: which classes does this unit push toward? */
			float max_out = 0;
			for (int32_t k = 0; k < NUM_CLASSES; k++) {
				float w = best_nn->w2[j * NUM_CLASSES + k];
				if (w > 0 && w > max_out) max_out = w;
				if (-w > max_out) max_out = -w;
			}

			printf("  Unit %2d (bias=%+.2f): ", j, (double)best_nn->b1[j]);
			/* Print formula */
			printf("WHEN ");
			for (int32_t t = 0; t < 3 && top_pos[t] >= 0; t++) {
				int32_t ni = top_pos[t] + name_offset;
				printf("%s%s(+%.1f)", t > 0 ? " + " : "",
					ni < NUM_FEATURES ? feat_names[ni] : "?", (double)top_pos_w[t]);
			}
			for (int32_t t = 0; t < 3 && top_neg[t] >= 0; t++) {
				int32_t ni = top_neg[t] + name_offset;
				printf(" - %s(%.1f)",
					ni < NUM_FEATURES ? feat_names[ni] : "?", (double)(-top_neg_w[t]));
			}
			printf("\n%25s→ ", "");
			for (int32_t k = 0; k < NUM_CLASSES; k++) {
				float w = best_nn->w2[j * NUM_CLASSES + k];
				if (w > 0.3f || w < -0.3f)
					printf("%s%s%s ", w > 0 ? "+" : "-", class_names[k],
						(w > 1.0f || w < -1.0f) ? "!" : "");
			}
			printf("\n");
		}
		printf("\n");
	}

	/* ---- SSE error analysis for best NN ---- */
	{
		static const char *mode_names[5] = {"diff  ","ind   ","planar","T     ","H     "};
		nn_t *best_nn = &nns[best_nn_idx];
		float *h_buf = malloc((size_t)best_nn->hidden * sizeof(float));

		/* Per true-mode stats */
		int64_t  mode_count[NUM_CLASSES]       = {0};
		int64_t  mode_correct[NUM_CLASSES]     = {0};
		double   mode_opt_sse[NUM_CLASSES]     = {0};
		double   mode_pred_sse[NUM_CLASSES]    = {0};
		int64_t  mode_valid_wrong[NUM_CLASSES]  = {0};
		double   mode_valid_overhead[NUM_CLASSES] = {0};
		int32_t  mode_worst_extra[NUM_CLASSES]  = {0};
		int64_t  mode_invalid[NUM_CLASSES]      = {0};

		double total_opt_sse  = 0;
		double total_pred_sse = 0;
		int64_t total_invalid = 0;

		/* Flip heuristic verification */
		int32_t diff_flip_total = 0, diff_flip_correct = 0;
		int32_t ind_flip_total  = 0, ind_flip_correct  = 0;
		double  diff_flip_sse_heuristic = 0, diff_flip_sse_optimal = 0;
		double  ind_flip_sse_heuristic  = 0, ind_flip_sse_optimal  = 0;

		for (int32_t i = 0; i < n_val; i++) {
			int32_t idx = val_idx[i];
			const uint8_t *block = raw_pixels + idx * 48;
			int32_t true_mode = labels[idx];

			float probs[NUM_CLASSES];
			_nn_forward(best_nn, dt_features + idx * g_n_feat, h_buf, probs);
			int32_t pred = 0;
			for (int32_t j = 1; j < NUM_CLASSES; j++)
				if (probs[j] > probs[pred]) pred = j;

			/* Map 5-class to best-of-both-flips for SSE */
			int32_t errors_7[7];
			etc2_evaluate_all(block, errors_7);
			int32_t merged_sse[5] = {
				errors_7[0] < errors_7[1] ? errors_7[0] : errors_7[1],
				errors_7[2] < errors_7[3] ? errors_7[2] : errors_7[3],
				errors_7[4], errors_7[5], errors_7[6],
			};
			int32_t opt_sse  = merged_sse[true_mode];
			int32_t pred_sse = (pred == true_mode) ? opt_sse : merged_sse[pred];

			/* Verify flip heuristic: compare sub_diff_h vs sub_diff_v
			 * to pick flip direction, check against exhaustive best */
			if (pred == 0 || true_mode == 0) { /* differential */
				/* Compute sub-block diffs from raw pixels */
				float sl[3]={0}, sr[3]={0}, st[3]={0}, sb[3]={0};
				for (int32_t y=0;y<4;y++) for (int32_t x=0;x<4;x++) {
					for (int32_t c=0;c<3;c++) {
						float v = (float)block[(y*4+x)*3+c];
						if (x<2) sl[c]+=v; else sr[c]+=v;
						if (y<2) st[c]+=v; else sb[c]+=v;
					}
				}
				float sdh=0, sdv=0;
				for (int32_t c=0;c<3;c++) {
					float dh = (sl[c]-sr[c])/8.0f; if(dh<0)dh=-dh; sdh+=dh;
					float dv = (st[c]-sb[c])/8.0f; if(dv<0)dv=-dv; sdv+=dv;
				}
				int32_t heuristic_flip = (sdh > sdv) ? 0 : 1; /* 0=vert, 1=horiz */
				int32_t best_flip = (errors_7[0] <= errors_7[1]) ? 0 : 1;
				diff_flip_total++;
				if (heuristic_flip == best_flip) diff_flip_correct++;
				diff_flip_sse_heuristic += errors_7[heuristic_flip];
				diff_flip_sse_optimal   += errors_7[best_flip];
			}
			if (pred == 1 || true_mode == 1) { /* individual */
				float sl[3]={0}, sr[3]={0}, st[3]={0}, sb[3]={0};
				for (int32_t y=0;y<4;y++) for (int32_t x=0;x<4;x++) {
					for (int32_t c=0;c<3;c++) {
						float v = (float)block[(y*4+x)*3+c];
						if (x<2) sl[c]+=v; else sr[c]+=v;
						if (y<2) st[c]+=v; else sb[c]+=v;
					}
				}
				float sdh=0, sdv=0;
				for (int32_t c=0;c<3;c++) {
					float dh = (sl[c]-sr[c])/8.0f; if(dh<0)dh=-dh; sdh+=dh;
					float dv = (st[c]-sb[c])/8.0f; if(dv<0)dv=-dv; sdv+=dv;
				}
				int32_t heuristic_flip = (sdh > sdv) ? 0 : 1;
				int32_t best_flip = (errors_7[2] <= errors_7[3]) ? 0 : 1;
				ind_flip_total++;
				if (heuristic_flip == best_flip) ind_flip_correct++;
				ind_flip_sse_heuristic += errors_7[2 + heuristic_flip];
				ind_flip_sse_optimal   += errors_7[2 + best_flip];
			}

			mode_count[true_mode]++;
			mode_opt_sse[true_mode] += opt_sse;
			total_opt_sse += opt_sse;

			if (pred == true_mode) {
				mode_correct[true_mode]++;
				mode_pred_sse[true_mode] += pred_sse;
				total_pred_sse += pred_sse;
			} else if (pred_sse >= INT32_MAX / 2) {
				/* Predicted mode is invalid for this block (e.g. diff overflow) */
				mode_invalid[true_mode]++;
				total_invalid++;
				/* Use opt_sse as fallback for totals (a real system would retry) */
				mode_pred_sse[true_mode] += opt_sse;
				total_pred_sse += opt_sse;
			} else {
				/* Wrong but valid — measure the quality cost */
				mode_pred_sse[true_mode] += pred_sse;
				total_pred_sse += pred_sse;
				int32_t extra = pred_sse - opt_sse;
				if (extra > 0) {
					mode_valid_wrong[true_mode]++;
					mode_valid_overhead[true_mode] += extra;
					if (extra > mode_worst_extra[true_mode])
						mode_worst_extra[true_mode] = extra;
				}
			}
		}

		printf("\n=== SSE Error Analysis (best NN, %d val blocks) ===\n\n", n_val);
		printf("%-8s  %7s  %5s  %7s  %10s  %10s  %8s  %10s  %10s\n",
			"Mode", "Count", "Acc%", "Inv%", "Avg OptSSE", "AvgPredSSE", "SSE Ovh%", "AvgExtra", "WorstExtra");
		printf("%-8s  %7s  %5s  %7s  %10s  %10s  %8s  %10s  %10s\n",
			"--------", "-------", "-----", "-------", "----------", "----------", "--------", "----------", "----------");

		for (int32_t c = 0; c < NUM_CLASSES; c++) {
			if (mode_count[c] == 0) continue;
			double avg_opt  = mode_opt_sse[c]  / (double)mode_count[c];
			double avg_pred = mode_pred_sse[c]  / (double)mode_count[c];
			double pct_overhead = avg_opt > 0 ? (avg_pred - avg_opt) / avg_opt * 100.0 : 0.0;
			double avg_extra = mode_valid_wrong[c] > 0
				? mode_valid_overhead[c] / (double)mode_valid_wrong[c] : 0.0;
			float acc = (float)mode_correct[c] / (float)mode_count[c] * 100.0f;
			float inv = (float)mode_invalid[c]  / (float)mode_count[c] * 100.0f;

			printf("%-8s  %7" PRId64 "  %5.1f  %6.1f%%  %10.1f  %10.1f  %7.1f%%  %10.1f  %10d\n",
				mode_names[c], mode_count[c], (double)acc, (double)inv,
				avg_opt, avg_pred, pct_overhead,
				avg_extra, mode_worst_extra[c]);
		}

		double total_overhead_pct = total_opt_sse > 0
			? (total_pred_sse - total_opt_sse) / total_opt_sse * 100.0 : 0.0;
		printf("%-8s  %7d  %5.1f  %6.1f%%  %10.1f  %10.1f  %7.1f%%\n",
			"TOTAL", n_val,
			(double)(best_nn_val),
			(double)total_invalid / n_val * 100.0,
			total_opt_sse / n_val, total_pred_sse / n_val, total_overhead_pct);

		printf("\nInvalid predictions: %" PRId64 " / %d (%.1f%%) — these need mode fallback\n",
			total_invalid, n_val, (double)total_invalid / n_val * 100.0);

		printf("\n--- Flip Direction Analysis ---\n");
		printf("  sub_diff heuristic:  diff %d/%d (%.1f%%)   ind %d/%d (%.1f%%)\n",
			diff_flip_correct, diff_flip_total,
			diff_flip_total > 0 ? (double)diff_flip_correct / diff_flip_total * 100.0 : 0.0,
			ind_flip_correct, ind_flip_total,
			ind_flip_total > 0 ? (double)ind_flip_correct / ind_flip_total * 100.0 : 0.0);
		printf("  → Heuristic is unreliable for diff (%.0f%%). But both flips cost\n",
			diff_flip_total > 0 ? (double)diff_flip_correct / diff_flip_total * 100.0 : 0.0);
		printf("    only ~128 comparisons each — just try both and pick the better one.\n");
		printf("  Individual flip heuristic SSE overhead: %.2f%%\n",
			ind_flip_sse_optimal > 0 ? (ind_flip_sse_heuristic - ind_flip_sse_optimal) / ind_flip_sse_optimal * 100.0 : 0.0);
		printf("\n  Practical shader strategy:\n");
		printf("    1. Classifier predicts mode family (5-class: diff/ind/plan/T/H)\n");
		printf("    2. For diff/ind: try BOTH flips (cheap), pick lower error\n");
		printf("    3. For T/H/plan: run the single predicted mode\n");
		printf("    → Equivalent to trying 2-3 modes instead of all 7\n");

		/* --- Flip direction analysis --- */
		printf("\n--- Flip Direction: What Predicts It? ---\n");

		/* Break down heuristic accuracy by flip gap size */
		{
			int32_t gap_buckets[4][2] = {{0}}; /* [gap_bucket][correct/total] */
			/* 0: tie (0%), 1: <5%, 2: 5-25%, 3: >25% */
			for (int32_t i = 0; i < n_val; i++) {
				int32_t idx = val_idx[i];
				if (labels[idx] != 0) continue;
				const uint8_t *blk = raw_pixels + idx * 48;
				int32_t e0 = etc2_mode_error(blk, 0);
				int32_t e1 = etc2_mode_error(blk, 1);
				if (e0 >= INT32_MAX/2 && e1 >= INT32_MAX/2) continue;

				int32_t best_e = e0 < e1 ? e0 : e1;
				int32_t worst_e = e0 < e1 ? e1 : e0;
				float gap = best_e > 0 ? (float)(worst_e - best_e) / (float)best_e * 100.0f : 0.0f;

				int32_t bucket;
				if (gap <= 0.01f) bucket = 0;
				else if (gap <= 5.0f) bucket = 1;
				else if (gap <= 25.0f) bucket = 2;
				else bucket = 3;

				/* sub_diff heuristic */
				float sl[3]={0}, sr[3]={0}, st[3]={0}, sb[3]={0};
				for (int32_t y=0;y<4;y++) for (int32_t x=0;x<4;x++)
					for (int32_t c=0;c<3;c++) {
						float v = (float)blk[(y*4+x)*3+c];
						if (x<2) sl[c]+=v; else sr[c]+=v;
						if (y<2) st[c]+=v; else sb[c]+=v;
					}
				float sdh=0, sdv=0;
				for (int32_t c=0;c<3;c++) {
					float dh = sl[c]-sr[c]; if(dh<0)dh=-dh; sdh+=dh;
					float dv = st[c]-sb[c]; if(dv<0)dv=-dv; sdv+=dv;
				}
				int32_t heur = (sdh > sdv) ? 0 : 1;
				int32_t truth = (e0 <= e1) ? 0 : 1;

				gap_buckets[bucket][1]++; /* total */
				if (heur == truth) gap_buckets[bucket][0]++; /* correct */
			}

			printf("  sub_diff heuristic accuracy by flip gap:\n");
			static const char *gap_names[] = {"Tie (0%%)    ", "<5%% gap     ", "5-25%% gap   ", ">25%% gap    "};
			for (int32_t b = 0; b < 4; b++) {
				if (gap_buckets[b][1] == 0) continue;
				printf("    %s: %5d/%5d = %5.1f%%\n", gap_names[b],
					gap_buckets[b][0], gap_buckets[b][1],
					(double)gap_buckets[b][0] / gap_buckets[b][1] * 100.0);
			}
			printf("\n");
		}

		/* Can we predict whether flip matters? Try several cheap metrics */
		{
			printf("  Can we predict WHEN flip matters?\n");
			printf("  (Testing: does metric X separate 'uniform' from 'directional' diff blocks?)\n\n");

			/* For each candidate metric, bin diff blocks into low/high and check
			 * what fraction of each bin has a large flip gap */
			typedef struct { const char *name; float threshold; int32_t lo_uni; int32_t lo_dir; int32_t hi_uni; int32_t hi_dir; } flip_metric_t;
			flip_metric_t metrics[] = {
				{"max(sdh,sdv)",        0, 0,0,0,0},
				{"|sdh-sdv|",           0, 0,0,0,0},
				{"max_range",           0, 0,0,0,0},
				{"block_variance",      0, 0,0,0,0},
				{"max_adj",             0, 0,0,0,0},
				{"sdh+sdv",             0, 0,0,0,0},
			};
			int32_t n_metrics = 6;

			/* First pass: collect metric values to find medians for threshold */
			int32_t n_diff_blocks = 0;
			for (int32_t i = 0; i < n_val; i++)
				if (labels[val_idx[i]] == 0) n_diff_blocks++;

			float *metric_vals = malloc((size_t)(n_diff_blocks * n_metrics) * sizeof(float));
			int32_t di = 0;
			for (int32_t i = 0; i < n_val; i++) {
				int32_t idx = val_idx[i];
				if (labels[idx] != 0) continue;
				const uint8_t *blk = raw_pixels + idx * 48;

				/* Compute raw metrics (not standardized) */
				float sl[3]={0}, sr[3]={0}, st[3]={0}, sb[3]={0};
				float sum[3]={0};
				float max_ch_range = 0, max_adj_v = 0;
				float min_c[3]={255,255,255}, max_c[3]={0,0,0};
				for (int32_t y=0;y<4;y++) for (int32_t x=0;x<4;x++) {
					for (int32_t c=0;c<3;c++) {
						float v = (float)blk[(y*4+x)*3+c];
						sum[c] += v;
						if (x<2) sl[c]+=v; else sr[c]+=v;
						if (y<2) st[c]+=v; else sb[c]+=v;
						if (v < min_c[c]) min_c[c] = v;
						if (v > max_c[c]) max_c[c] = v;
					}
					if (x < 3) for (int32_t c=0;c<3;c++) {
						float d = (float)blk[(y*4+x)*3+c] - (float)blk[(y*4+x+1)*3+c];
						if (d<0) d=-d; if (d > max_adj_v) max_adj_v = d;
					}
					if (y < 3) for (int32_t c=0;c<3;c++) {
						float d = (float)blk[(y*4+x)*3+c] - (float)blk[((y+1)*4+x)*3+c];
						if (d<0) d=-d; if (d > max_adj_v) max_adj_v = d;
					}
				}
				float sdh=0, sdv=0;
				for (int32_t c=0;c<3;c++) {
					float dh = (sl[c]-sr[c])/8.0f; if(dh<0)dh=-dh; sdh+=dh;
					float dv = (st[c]-sb[c])/8.0f; if(dv<0)dv=-dv; sdv+=dv;
					float r = max_c[c]-min_c[c]; if(r>max_ch_range) max_ch_range=r;
				}
				float bvar = 0;
				for (int32_t c=0;c<3;c++) {
					float m = sum[c]/16.0f;
					for (int32_t p=0;p<16;p++) { float d=(float)blk[p*3+c]-m; bvar+=d*d; }
				}
				bvar /= 48.0f;

				metric_vals[di * n_metrics + 0] = sdh > sdv ? sdh : sdv;
				metric_vals[di * n_metrics + 1] = sdh - sdv; if(metric_vals[di*n_metrics+1]<0) metric_vals[di*n_metrics+1]=-metric_vals[di*n_metrics+1];
				metric_vals[di * n_metrics + 2] = max_ch_range;
				metric_vals[di * n_metrics + 3] = bvar;
				metric_vals[di * n_metrics + 4] = max_adj_v;
				metric_vals[di * n_metrics + 5] = sdh + sdv;
				di++;
			}

			/* Find median of each metric for threshold */
			for (int32_t m = 0; m < n_metrics; m++) {
				/* Approximate median: average of values near the middle */
				float sum_v = 0;
				for (int32_t i = 0; i < n_diff_blocks; i++)
					sum_v += metric_vals[i * n_metrics + m];
				metrics[m].threshold = sum_v / (float)n_diff_blocks;
			}

			/* Second pass: bin by metric and check flip gap */
			di = 0;
			for (int32_t i = 0; i < n_val; i++) {
				int32_t idx = val_idx[i];
				if (labels[idx] != 0) continue;
				const uint8_t *blk = raw_pixels + idx * 48;
				int32_t e0 = etc2_mode_error(blk, 0);
				int32_t e1 = etc2_mode_error(blk, 1);
				if (e0 >= INT32_MAX/2 && e1 >= INT32_MAX/2) { di++; continue; }

				int32_t best_e = e0 < e1 ? e0 : e1;
				int32_t worst_e = e0 < e1 ? e1 : e0;
				float gap = best_e > 0 ? (float)(worst_e - best_e) / (float)best_e * 100.0f : 0.0f;
				int32_t is_dir = (gap > 10.0f) ? 1 : 0; /* >10% gap = directional */

				for (int32_t m = 0; m < n_metrics; m++) {
					float v = metric_vals[di * n_metrics + m];
					if (v < metrics[m].threshold) {
						if (is_dir) metrics[m].lo_dir++; else metrics[m].lo_uni++;
					} else {
						if (is_dir) metrics[m].hi_dir++; else metrics[m].hi_uni++;
					}
				}
				di++;
			}
			free(metric_vals);

			printf("  %-16s  %20s  %20s\n", "Metric", "Below mean", "Above mean");
			printf("  %-16s  %9s %9s  %9s %9s\n", "", "uniform", "dir.", "uniform", "dir.");
			printf("  %-16s  %9s %9s  %9s %9s\n", "----------------", "---------", "---------", "---------", "---------");
			for (int32_t m = 0; m < n_metrics; m++) {
				int32_t lo_total = metrics[m].lo_uni + metrics[m].lo_dir;
				int32_t hi_total = metrics[m].hi_uni + metrics[m].hi_dir;
				printf("  %-16s  %7d %s %7d %s  %7d %s %7d %s\n",
					metrics[m].name,
					metrics[m].lo_uni, lo_total > 0 ? "" : " ",
					metrics[m].lo_dir, lo_total > 0 ? "" : " ",
					metrics[m].hi_uni, hi_total > 0 ? "" : " ",
					metrics[m].hi_dir, hi_total > 0 ? "" : " ");
			}
			printf("\n  'uniform' = flip gap <10%%, 'directional' = flip gap >10%%\n");
			printf("  A good metric has most 'dir.' in the 'Above mean' column.\n\n");
		}
		{
			/* Train a tiny linear flip classifier on raw pixel features
			 * to see which pixel positions matter for flip direction */
			float flip_w[48] = {0}; /* weight per pixel channel */
			float flip_b = 0;
			float flip_lr = 0.01f;

			/* Simple logistic regression: P(flip=1) = sigmoid(w*x + b) */
			for (int32_t epoch = 0; epoch < 100; epoch++) {
				for (int32_t i = 0; i < n_val; i++) {
					int32_t idx = val_idx[i];
					if (labels[idx] != 0) continue;
					const uint8_t *blk = raw_pixels + idx * 48;

					/* True flip: which flip gives lower error? */
					int32_t e0 = etc2_mode_error(blk, 0); /* diff flip=0 */
					int32_t e1 = etc2_mode_error(blk, 1); /* diff flip=1 */
					if (e0 >= INT32_MAX/2 && e1 >= INT32_MAX/2) continue;
					float target = (e1 < e0) ? 1.0f : 0.0f;
					if (e0 == e1) continue; /* skip ties */

					/* Forward */
					float logit = flip_b;
					for (int32_t f = 0; f < 48; f++)
						logit += flip_w[f] * ((float)blk[f] / 255.0f);
					float pred = 1.0f / (1.0f + expf(-logit));

					/* Gradient */
					float err = pred - target;
					flip_b -= flip_lr * err;
					for (int32_t f = 0; f < 48; f++)
						flip_w[f] -= flip_lr * err * ((float)blk[f] / 255.0f);
				}
			}

			/* Evaluate accuracy */
			int32_t flip_correct = 0, flip_total = 0;
			for (int32_t i = 0; i < n_val; i++) {
				int32_t idx = val_idx[i];
				if (labels[idx] != 0) continue;
				const uint8_t *blk = raw_pixels + idx * 48;
				int32_t e0 = etc2_mode_error(blk, 0);
				int32_t e1 = etc2_mode_error(blk, 1);
				if (e0 >= INT32_MAX/2 && e1 >= INT32_MAX/2) continue;
				if (e0 == e1) { flip_total++; flip_correct++; continue; }
				float logit = flip_b;
				for (int32_t f = 0; f < 48; f++)
					logit += flip_w[f] * ((float)blk[f] / 255.0f);
				int32_t pred_flip = (logit > 0) ? 1 : 0;
				int32_t true_flip = (e1 < e0) ? 1 : 0;
				if (pred_flip == true_flip) flip_correct++;
				flip_total++;
			}
			printf("  Linear pixel classifier: %d/%d (%.1f%%)\n",
				flip_correct, flip_total,
				flip_total > 0 ? (double)flip_correct / flip_total * 100.0 : 0.0);

			/* Show weight heatmap — which pixel positions matter?
			 * Layout: 4x4 grid, each pixel has RGB weights, show magnitude */
			printf("\n  Pixel position weight magnitudes (4x4 grid):\n");
			printf("  (high = important for flip direction, sign = which flip it favors)\n\n");
			for (int32_t y = 0; y < 4; y++) {
				printf("  ");
				for (int32_t x = 0; x < 4; x++) {
					int32_t pi = (y * 4 + x) * 3;
					float mag = 0;
					for (int32_t c = 0; c < 3; c++)
						mag += flip_w[pi + c] * flip_w[pi + c];
					mag = sqrtf(mag);
					float sign = flip_w[pi] + flip_w[pi+1] + flip_w[pi+2];
					char s = sign > 0 ? '+' : '-';
					printf("  %c%5.2f", s, (double)mag);
				}
				printf("\n");
			}
			printf("\n  + = favors flip=1 (horizontal), - = favors flip=0 (vertical)\n");
			printf("  If left/right columns differ: flip=0 for vertical split\n");
			printf("  If top/bottom rows differ:    flip=1 for horizontal split\n");
		}

		free(h_buf);
	}

	/* ---- Dataset ambiguity analysis ---- */
	{
		/* For each block: compute all 7 mode errors, measure how close
		 * the 2nd-best is to the best. If they're within X%, no classifier
		 * can reliably distinguish them from features alone. */
		int32_t bucket_0   = 0; /* 2nd best within 0% of best (tie) */
		int32_t bucket_5   = 0; /* within 5% */
		int32_t bucket_10  = 0; /* within 10% */
		int32_t bucket_25  = 0; /* within 25% */
		int32_t bucket_50  = 0; /* within 50% */
		int32_t bucket_inf = 0; /* >50% gap — clear winner */
		int32_t n_check = n_val < 50000 ? n_val : 50000; /* cap for speed */

		for (int32_t i = 0; i < n_check; i++) {
			int32_t idx = val_idx[i];
			const uint8_t *block = raw_pixels + idx * 48;

			int32_t errors[7];
			etc2_evaluate_all(block, errors);

			/* Find best and 2nd best */
			int32_t best = 0, second = 1;
			if (errors[1] < errors[0]) { best = 1; second = 0; }
			for (int32_t m = 2; m < 7; m++) {
				if (errors[m] < errors[best]) { second = best; best = m; }
				else if (errors[m] < errors[second]) { second = m; }
			}

			int32_t best_err = errors[best];
			int32_t second_err = errors[second];
			if (best_err <= 0) { bucket_0++; continue; }
			float gap_pct = (float)(second_err - best_err) / (float)best_err * 100.0f;

			if      (gap_pct <= 0.01f) bucket_0++;
			else if (gap_pct <= 5.0f)  bucket_5++;
			else if (gap_pct <= 10.0f) bucket_10++;
			else if (gap_pct <= 25.0f) bucket_25++;
			else if (gap_pct <= 50.0f) bucket_50++;
			else                       bucket_inf++;
		}

		printf("\n=== Dataset Ambiguity (how close is 2nd-best mode to best?) ===\n\n");
		int32_t cumul = 0;
		printf("  Gap          Count     %%   Cumul%%   Interpretation\n");
		printf("  ----------  ------  -----  ------   ----------------------------\n");
		cumul += bucket_0;
		printf("  Tie (0%%)    %6d  %5.1f%%  %5.1f%%   Impossible to distinguish\n",
			bucket_0, (double)bucket_0/n_check*100, (double)cumul/n_check*100);
		cumul += bucket_5;
		printf("  < 5%%        %6d  %5.1f%%  %5.1f%%   Nearly identical quality\n",
			bucket_5, (double)bucket_5/n_check*100, (double)cumul/n_check*100);
		cumul += bucket_10;
		printf("  5-10%%       %6d  %5.1f%%  %5.1f%%   Very similar quality\n",
			bucket_10, (double)bucket_10/n_check*100, (double)cumul/n_check*100);
		cumul += bucket_25;
		printf("  10-25%%      %6d  %5.1f%%  %5.1f%%   Noticeable but minor\n",
			bucket_25, (double)bucket_25/n_check*100, (double)cumul/n_check*100);
		cumul += bucket_50;
		printf("  25-50%%      %6d  %5.1f%%  %5.1f%%   Significant gap\n",
			bucket_50, (double)bucket_50/n_check*100, (double)cumul/n_check*100);
		cumul += bucket_inf;
		printf("  > 50%%       %6d  %5.1f%%  %5.1f%%   Clear winner\n",
			bucket_inf, (double)bucket_inf/n_check*100, (double)cumul/n_check*100);

		/* Recount with finer buckets + per-mode breakdown for clear winners */
		static const char *mn[5] = {"diff ","ind  ","plan ","T    ","H    "};
		int32_t b_0 = 0, b_2p5 = 0, b_5 = 0, b_10 = 0, b_rest = 0;
		int32_t clear_by_mode[NUM_CLASSES] = {0};     /* >10% gap, per best mode */
		int32_t ambig_by_mode[NUM_CLASSES] = {0};     /* <=10% gap */
		int32_t tie_by_mode[NUM_CLASSES]   = {0};     /* <=2.5% gap (relaxed-free) */
		int32_t clear_confused[NUM_CLASSES] = {0};     /* >10% gap but NN got wrong */

		for (int32_t i = 0; i < n_check; i++) {
			int32_t idx = val_idx[i];
			const uint8_t *block = raw_pixels + idx * 48;
			int32_t errors[7];
			etc2_evaluate_all(block, errors);

			int32_t best = 0;
			for (int32_t m = 1; m < 7; m++)
				if (errors[m] < errors[best]) best = m;
			int32_t second_err = INT32_MAX;
			for (int32_t m = 0; m < 7; m++)
				if (m != best && errors[m] < second_err) second_err = errors[m];

			int32_t best_err = errors[best];
			float gap = best_err > 0 ? (float)(second_err - best_err) / (float)best_err * 100.0f : 0.0f;
			if (best_err == 0 && second_err == 0) gap = 0;

			if      (gap <= 0.01f) { b_0++;   tie_by_mode[best]++; }
			else if (gap <= 2.5f)  { b_2p5++;  tie_by_mode[best]++; }
			else if (gap <= 5.0f)  { b_5++;    ambig_by_mode[best]++; }
			else if (gap <= 10.0f) { b_10++;   ambig_by_mode[best]++; }
			else                   { b_rest++; clear_by_mode[best]++; }
		}

		printf("\n  --- Accuracy ceilings ---\n");
		printf("  STRICT:  any prediction on tie/near-tie blocks is a coin flip.\n");
		printf("    Tie (0%%):       %5.1f%%  ← can't distinguish, ~50%% chance of matching label\n", (double)b_0/n_check*100);
		printf("    0-2.5%%:        %5.1f%%  ← nearly tied\n", (double)b_2p5/n_check*100);
		printf("    2.5-5%%:        %5.1f%%  ← close\n", (double)b_5/n_check*100);
		printf("    5-10%%:         %5.1f%%  ← distinguishable\n", (double)b_10/n_check*100);
		printf("    >10%%:          %5.1f%%  ← clear winner\n", (double)b_rest/n_check*100);
		printf("    Strict ceiling: ~%.0f%%  (assuming ~50%% on ties, 100%% on >5%% gap)\n",
			(double)(b_0/2 + b_2p5/2 + b_5/2 + b_10 + b_rest) / n_check * 100.0);
		printf("\n  RELAXED (either top-1 or top-2 within 2.5%% counts as correct):\n");
		printf("    Always correct: %5.1f%%  (tie + within 2.5%%)\n", (double)(b_0 + b_2p5)/n_check*100);
		printf("    Must get right: %5.1f%%  (gap > 2.5%%)\n", (double)(b_5 + b_10 + b_rest)/n_check*100);
		printf("    Relaxed ceiling: ~%.0f%%\n",
			(double)(b_0 + b_2p5 + b_10 + b_rest) / n_check * 100.0);

		printf("\n  --- Per-mode breakdown ---\n");
		printf("  %-7s  %7s  %7s  %7s  %7s\n",
			"Mode", "Clear", "Ambig", "Tied", "Total");
		printf("  %-7s  %7s  %7s  %7s  %7s\n",
			"-------", "-------", "-------", "-------", "-------");
		for (int32_t c = 0; c < NUM_CLASSES; c++) {
			int32_t total = clear_by_mode[c] + ambig_by_mode[c] + tie_by_mode[c];
			if (total == 0) continue;
			printf("  %-7s  %5d %s  %5d %s  %5d %s  %5d\n",
				mn[c],
				clear_by_mode[c],
				clear_by_mode[c] > 0 ? "" : " ",
				ambig_by_mode[c],
				ambig_by_mode[c] > 0 ? "" : " ",
				tie_by_mode[c],
				tie_by_mode[c] > 0 ? "" : " ",
				total);
		}
		printf("\n  Clear winners = >10%% gap between best and 2nd-best mode.\n");
		printf("  Ambiguous     = 2.5-10%% gap. Tied = <2.5%% gap.\n");
	}

	/* ---- Export best models ---- */
	printf("\n--- Exporting models ---\n");
	_dt_export(&best_dt_tree, 0, best_dt_depth, best_dt_val, "decision_tree_export.h");
	_linear_export(&linear, lin_val, "linear_export.h");
	_nn_export(&nns[best_nn_idx], best_nn_val, "nn_export.h", feat_mean, feat_std);

	/* Cleanup */
	for (int32_t i = 0; i < num_hidden; i++)
		_nn_free(&nns[i]);
	free(nns);
	free(raw_pixels);
	free(dt_features);
	free(labels);
	free(all_indices);

	return 0;
}
