// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include <stdint.h>

typedef struct {
	double psnr_r, psnr_g, psnr_b, psnr_rgb;
	double psnr_a;   // 100.0 when alpha is identical (e.g. fully opaque both sides)
	double psnr_pm;  // PSNR of alpha-premultiplied RGB — errors hidden behind
	                 // alpha≈0 don't count; the honest metric for cutouts
	double ssim_r, ssim_g, ssim_b, ssim_rgb;
} quality_metrics_t;

// Compute per-channel PSNR + SSIM between two equal-sized RGBA8 buffers.
// SSIM ignores alpha; PSNR reports it separately in psnr_a. Returns {0} if
// the sizes disagree.
quality_metrics_t quality_compare_rgba8(const uint8_t* a, const uint8_t* b, int32_t width, int32_t height);
