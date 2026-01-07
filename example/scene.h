// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include <sk_renderer.h>
#include "app.h"
#include "tools/float_math.h"
#include "tools/scene_util.h"
#include <stdbool.h>
#include <stdint.h>

// Forward declarations
typedef struct scene_t scene_t;

// Camera info structure
typedef struct {
	float3 position;
	float3 target;
	float3 up;
} scene_camera_t;

// Scene interface - each scene must implement these functions
typedef struct {
	const char* name;
	scene_t*    (*create)     (void);
	void        (*destroy)    (scene_t* scene);
	void        (*update)     (scene_t* scene, float delta_time);
	void        (*render)     (scene_t* scene, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer);
	bool        (*get_camera) (scene_t* scene, scene_camera_t* out_camera);  // Optional - return true to override camera
	void        (*render_ui)  (scene_t* scene);  // Optional - scene-specific ImGui controls
} scene_vtable_t;

// Base scene structure - just holds size for "inheritance" pattern
struct scene_t {
	size_t size;
};

// Scene registry - add new scenes here
extern const scene_vtable_t scene_meshes_vtable;
extern const scene_vtable_t scene_reaction_diffusion_vtable;
extern const scene_vtable_t scene_orbital_particles_vtable;
extern const scene_vtable_t scene_impostor_vtable;
extern const scene_vtable_t scene_array_texture_vtable;
extern const scene_vtable_t scene_3d_texture_vtable;
extern const scene_vtable_t scene_cubemap_vtable;
extern const scene_vtable_t scene_gltf_vtable;
extern const scene_vtable_t scene_shadows_vtable;
extern const scene_vtable_t scene_cloth_vtable;
extern const scene_vtable_t scene_text_vtable;
extern const scene_vtable_t scene_tex_copy_vtable;
extern const scene_vtable_t scene_lifetime_stress_vtable;
extern const scene_vtable_t scene_gaussian_splat_vtable;
#ifdef SKR_HAS_VIDEO
extern const scene_vtable_t scene_video_vtable;
#endif

// Helper macros for scene methods (vtable passed separately)
#define scene_create(vtable)                                ((vtable)->create())
#define scene_destroy(vtable, scene)                        ((vtable)->destroy(scene))
#define scene_update(vtable, scene, delta_time)             ((vtable)->update(scene, delta_time))
#define scene_render(vtable, scene, w, h, render_list, buf) ((vtable)->render(scene, w, h, render_list, buf))
#define scene_render_ui(vtable, scene)                      ((vtable)->render_ui ? (vtable)->render_ui(scene) : (void)0)
#define scene_get_name(vtable)                              ((vtable)->name)
