// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "metrics.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

// Per-channel PSNR from MSE over a W×H array. Returns 100.0 for MSE == 0.
static double _psnr(double mse) {
	if (mse <= 0.0) return 100.0;
	return 10.0 * log10((255.0 * 255.0) / mse);
}

// Single-channel SSIM using an 8x8 block-mean window. We use block-mean
// instead of the standard 11x11 Gaussian because it needs no tuning constants
// and the numeric difference is small for compression-quality comparisons
// (SSIM is insensitive to the exact window within reason).
static double _ssim_channel(const uint8_t* a, const uint8_t* b, int32_t w, int32_t h, int32_t stride, int32_t channel) {
	const int32_t block = 8;
	const double  c1    = (0.01 * 255.0) * (0.01 * 255.0);
	const double  c2    = (0.03 * 255.0) * (0.03 * 255.0);

	double sum = 0.0;
	int32_t count = 0;

	for (int32_t by = 0; by <= h - block; by += block) {
		for (int32_t bx = 0; bx <= w - block; bx += block) {
			double mean_a = 0, mean_b = 0;
			for (int32_t y = 0; y < block; y++) {
				for (int32_t x = 0; x < block; x++) {
					int32_t i = ((by + y) * w + (bx + x)) * stride + channel;
					mean_a += a[i];
					mean_b += b[i];
				}
			}
			double n_inv = 1.0 / (double)(block * block);
			mean_a *= n_inv;
			mean_b *= n_inv;

			double var_a = 0, var_b = 0, cov_ab = 0;
			for (int32_t y = 0; y < block; y++) {
				for (int32_t x = 0; x < block; x++) {
					int32_t i = ((by + y) * w + (bx + x)) * stride + channel;
					double  da = (double)a[i] - mean_a;
					double  db = (double)b[i] - mean_b;
					var_a  += da * da;
					var_b  += db * db;
					cov_ab += da * db;
				}
			}
			var_a  *= n_inv;
			var_b  *= n_inv;
			cov_ab *= n_inv;

			double numer = (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2);
			double denom = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);
			sum += numer / denom;
			count++;
		}
	}
	return count > 0 ? sum / count : 0.0;
}

quality_metrics_t quality_compare_rgba8(const uint8_t* a, const uint8_t* b, int32_t width, int32_t height) {
	quality_metrics_t out = {0};
	if (!a || !b || width <= 0 || height <= 0) return out;

	const int32_t stride = 4;
	const int64_t n      = (int64_t)width * (int64_t)height;

	double sq_r = 0, sq_g = 0, sq_b = 0, sq_a = 0, sq_pm = 0;
	for (int64_t i = 0; i < n; i++) {
		double dr = (double)a[i * stride + 0] - (double)b[i * stride + 0];
		double dg = (double)a[i * stride + 1] - (double)b[i * stride + 1];
		double db = (double)a[i * stride + 2] - (double)b[i * stride + 2];
		double da = (double)a[i * stride + 3] - (double)b[i * stride + 3];
		sq_r += dr * dr;
		sq_g += dg * dg;
		sq_b += db * db;
		sq_a += da * da;

		// Premultiplied compare: what compositing actually shows
		double aa = (double)a[i * stride + 3] / 255.0;
		double ba = (double)b[i * stride + 3] / 255.0;
		for (int32_t c = 0; c < 3; c++) {
			double d = (double)a[i * stride + c] * aa - (double)b[i * stride + c] * ba;
			sq_pm += d * d;
		}
	}

	double mse_r   = sq_r / (double)n;
	double mse_g   = sq_g / (double)n;
	double mse_b   = sq_b / (double)n;
	double mse_rgb = (sq_r + sq_g + sq_b) / (double)(n * 3);

	out.psnr_r   = _psnr(mse_r);
	out.psnr_g   = _psnr(mse_g);
	out.psnr_b   = _psnr(mse_b);
	out.psnr_rgb = _psnr(mse_rgb);
	out.psnr_a   = _psnr(sq_a / (double)n);
	out.psnr_pm  = _psnr(sq_pm / (double)(n * 3));

	out.ssim_r   = _ssim_channel(a, b, width, height, stride, 0);
	out.ssim_g   = _ssim_channel(a, b, width, height, stride, 1);
	out.ssim_b   = _ssim_channel(a, b, width, height, stride, 2);
	out.ssim_rgb = (out.ssim_r + out.ssim_g + out.ssim_b) / 3.0;

	return out;
}
