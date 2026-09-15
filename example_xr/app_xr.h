#pragma once

#include <sk_renderer.h>
#include <openxr/openxr.h>

// Initialize application resources
void app_xr_init(void);

// Shutdown application resources
void app_xr_shutdown(void);

// Update application state (called once per frame)
void app_xr_update(void);

// Update with predicted time (called after input prediction)
void app_xr_update_predicted(void);

// Render all views in a single pass (stereo array texture)
// color_target: MSAA array texture to render to
// resolve_target: non-MSAA texture to resolve to (can be NULL if no MSAA)
// depth_target: MSAA depth array texture
// delta_time, refresh_hz: from the runtime's predicted display time and period
void app_xr_render_stereo(skr_tex_t* color_target, skr_tex_t* resolve_target,
                          skr_tex_t* depth_target,
                          const XrView* views, uint32_t view_count,
                          int32_t width, int32_t height,
                          float delta_time, float refresh_hz);
