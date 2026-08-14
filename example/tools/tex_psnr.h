// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#pragma once

#include <sk_renderer.h>

// GPU-side PSNR measurement
//
// Compares mip 0 of two same-size textures through the hardware decoder and
// returns PSNR in dB using the standard 8-bit RGB metric
// (10*log10(255^2 / MSE), RGB channels only, computed in sRGB-encoded space).
// Works for any format the GPU can sample — BC formats on desktop, ASTC on
// device — so it measures exactly what rendering sees.
//
// Stalls the GPU (dispatch + readback); intended for load-time quality
// logging, not per-frame use. Returns a negative value on failure, and
// INFINITY for identical images.

void   tex_psnr_init    (void);
void   tex_psnr_shutdown(void);
double tex_psnr         (const skr_tex_t* reference, const skr_tex_t* compressed);
