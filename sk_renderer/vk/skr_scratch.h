// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include "skr_vulkan.h"

// Scratch texture pool for render-based mipmap generation.
//
// Final textures avoid COLOR_ATTACHMENT_BIT usage (which disables lossless
// framebuffer compression on AMD DCC, Mali AFBC, Adreno UBWC, etc.). Instead
// the render-based mipgen path allocates a scratch image with color-attachment
// usage, filters mips into it, and copies the result back to the real texture.
// Scratch images are pooled and reused across mipgen calls to avoid repeated
// vkCreateImage / vkAllocateMemory overhead during scene-load bursts; entries
// are evicted after SKR_SCRATCH_IDLE_FRAMES of inactivity.

void       _skr_scratch_pool_init    (void);
void       _skr_scratch_pool_shutdown(void);
void       _skr_scratch_pool_tick    (void);                            // Called once per frame from frame_end

skr_tex_t* _skr_scratch_acquire      (const skr_tex_t* template_src);   // Returns scratch matching fmt/size/layers/mips; NULL on error
void       _skr_scratch_release      (skr_tex_t* scratch);              // Marks scratch available for reuse
