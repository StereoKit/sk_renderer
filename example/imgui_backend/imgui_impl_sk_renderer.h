// SPDX-License-Identifier: MIT
// Dear ImGui sk_renderer Backend
// C-compatible header for integration with both C and C++ code

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ImDrawData;
struct skr_tex_t;

bool ImGui_ImplSkRenderer_Init    (void);
void ImGui_ImplSkRenderer_Shutdown(void);
void ImGui_ImplSkRenderer_NewFrame(void);

// Two-phase rendering for ImGui:
// 1. PrepareDrawData - uploads mesh data (MUST be called OUTSIDE render pass)
// 2. RenderDrawData - draws ImGui (MUST be called INSIDE render pass)

// Call OUTSIDE render pass to upload mesh data
void ImGui_ImplSkRenderer_PrepareDrawData(void);

// Call INSIDE render pass to draw ImGui
void ImGui_ImplSkRenderer_RenderDrawData(int width, int height);

#ifdef __cplusplus
}
#endif
