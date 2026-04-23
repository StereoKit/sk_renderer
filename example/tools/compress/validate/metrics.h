// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include <stdint.h>

typedef struct {
	double psnr_r, psnr_g, psnr_b, psnr_rgb;
	double ssim_r, ssim_g, ssim_b, ssim_rgb;
} quality_metrics_t;

// Compute per-channel PSNR + SSIM between two equal-sized RGBA8 buffers.
// Alpha is ignored. Returns {0} if the sizes disagree.
quality_metrics_t quality_compare_rgba8(const uint8_t* a, const uint8_t* b, int32_t width, int32_t height);
