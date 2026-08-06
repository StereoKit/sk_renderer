// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include "skr_vulkan.h"

// Transient attachment pool for postfx chain intermediates and depth resolve
// targets. These images live entirely within a single render pass (loadOp and
// storeOp DONT_CARE, lazily allocated on tilers), so their contents never
// survive a pass — pooling avoids per-frame vkCreateImage/vkAllocateMemory
// hitches. Cross-pass reuse is made safe by the EXTERNAL subpass dependencies
// the multisubpass renderpass places on each attachment's first-use subpass;
// entries are evicted after SKR_TRANSIENT_IDLE_FRAMES of inactivity.

void       _skr_transient_pool_init    (void);
void       _skr_transient_pool_shutdown(void);
void       _skr_transient_pool_tick    (void);  // Called once per frame from frame_end

skr_tex_t* _skr_transient_acquire      (VkFormat format, int32_t width, int32_t height, int32_t layers, bool depth);  // NULL on error
void       _skr_transient_release      (skr_tex_t* transient);  // Pass recorded — entry may be reused by later passes
