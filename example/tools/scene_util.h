// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#include <sk_renderer.h>
#include "float_math.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
// Logging
///////////////////////////////////////////////////////////////////////////////

typedef enum su_log_ {
	su_log_info,
	su_log_warning,
	su_log_critical,
} su_log_;

static inline void su_log(su_log_ level, const char* text, ...) {
	va_list args;
	va_start(args, text);

	char buffer[1024];
	vsnprintf(buffer, sizeof(buffer), text, args);

#ifdef __ANDROID__
	android_LogPriority priority = 
		level == su_log_info     ? ANDROID_LOG_INFO  :
		level == su_log_warning  ? ANDROID_LOG_WARN  :
		level == su_log_critical ? ANDROID_LOG_ERROR : ANDROID_LOG_UNKNOWN;
	__android_log_write(priority, "sk_example", buffer);
#else
	const char* prefix = 
		level == su_log_info     ? "[app:info] " :
		level == su_log_warning  ? "[app:warn] " :
		level == su_log_critical ? "[app:crit] " : "[app:unkn] ";
	printf("%s%s\n", prefix, buffer);
#endif

	va_end(args);
}

///////////////////////////////////////////////////////////////////////////////
// Rendering system buffer
///////////////////////////////////////////////////////////////////////////////

// Rendering system buffer (matching SystemBuffer from common.hlsli)
#define SU_MAX_VIEWS 6
typedef struct {
	float4x4 view          [SU_MAX_VIEWS];  // View matrices (one per view)
	float4x4 view_inv      [SU_MAX_VIEWS];  // Inverse view matrices
	float4x4 projection    [SU_MAX_VIEWS];  // Per-view projection matrices
	float4x4 projection_inv[SU_MAX_VIEWS];  // Inverse projection matrices
	float4x4 viewproj      [SU_MAX_VIEWS];  // Precomputed view*projection matrices
	float4   cam_pos       [SU_MAX_VIEWS];  // Camera position (xyz + padding)
	float4   cam_dir       [SU_MAX_VIEWS];  // Camera forward direction (xyz + padding)
	float4   cubemap_info;                  // .xy = size, .z = mip count, .w = unused
	float4   screen_size;                   // .xy = width/height, .zw = 1/width, 1/height
	float    time;                          // Time in seconds
	uint32_t view_count;                    // Number of active views (1-6)
	uint32_t view_offset;                   // Unused, kept for ABI compatibility
	uint32_t _pad;
} su_system_buffer_t;

///////////////////////////////////////////////////////////////////////////////
// Common Vertex Format
///////////////////////////////////////////////////////////////////////////////

// Standard vertex format: Position + Normal + UV + Color
typedef struct {
	skr_vec3_t position;
	skr_vec3_t normal;
	skr_vec2_t uv;
	uint32_t   color;
} su_vertex_t;

// Standard vertex type for PNUC format (available after su_initialize)
extern skr_vert_type_t su_vertex_type;

///////////////////////////////////////////////////////////////////////////////
// Initialization
///////////////////////////////////////////////////////////////////////////////

// File loading callback type
// filename: Path to file
// out_data: Pointer to receive allocated data (must be freed by caller with free())
// out_size: Pointer to receive file size in bytes
// user_data: User data passed to su_initialize
// Returns: true on success, false on failure
typedef bool (*su_file_read_fn)(const char* filename, void** out_data, size_t* out_size, void* user_data);

// Initialize scene utilities (call once at startup, after sk_renderer init)
// Starts the asset loading thread and initializes standard vertex types
// file_read_callback: Optional custom file loader (NULL uses default fopen/fread)
// user_data: User data passed to file_read_callback
void su_initialize(su_file_read_fn file_read_callback, void* user_data);

// Shutdown scene utilities (call before sk_renderer shutdown)
// Waits for pending loads and stops the asset loading thread
void su_shutdown(void);

///////////////////////////////////////////////////////////////////////////////
// Common Texture Samplers
///////////////////////////////////////////////////////////////////////////////

extern const skr_tex_sampler_t su_sampler_linear_clamp;
extern const skr_tex_sampler_t su_sampler_linear_wrap;
extern const skr_tex_sampler_t su_sampler_point_clamp;

///////////////////////////////////////////////////////////////////////////////
// Mesh Generation
///////////////////////////////////////////////////////////////////////////////

// Creates a UV sphere mesh
// segments: Horizontal divisions (e.g., 16 or 32)
// rings: Vertical divisions (e.g., 12 or 24)
// radius: Sphere radius
// color: Vertex color for all vertices
skr_mesh_t su_mesh_create_sphere(
	int32_t          segments,
	int32_t          rings,
	float            radius,
	skr_vec4_t       color
);

// Creates a cube mesh with optional per-face colors
// size: Cube edge length (e.g., 1.0f for unit cube)
// opt_face_colors: Array of 6 colors [Front, Back, Top, Bottom, Right, Left], or NULL for white
skr_mesh_t su_mesh_create_cube(
	float             size,
	const skr_vec4_t* opt_face_colors
);

// Creates a pyramid mesh (square base + apex)
// base_size: Width/depth of square base
// height: Height from base center to apex
// color: Vertex color
skr_mesh_t su_mesh_create_pyramid(
	float            base_size,
	float            height,
	skr_vec4_t       color
);

// Creates a quad mesh
// width: Quad width
// height: Quad height
// normal: Normal direction (determines plane: (0,1,0) = XZ plane, etc.)
// double_sided: If true, creates both front and back faces
// color: Vertex color
skr_mesh_t su_mesh_create_quad(
	float            width,
	float            height,
	skr_vec3_t       normal,
	bool             double_sided,
	skr_vec4_t       color
);

// Creates a fullscreen quad for post-processing effects
// Uses NDC coordinates (-1 to +1) in XY with Z=0, uses su_vertex_type_pnuc
// UV coordinates go from (0,0) to (1,1)
skr_mesh_t su_mesh_create_fullscreen_quad(void);

///////////////////////////////////////////////////////////////////////////////
// Texture Generation
///////////////////////////////////////////////////////////////////////////////

// Creates a checkerboard texture with two alternating colors
// resolution: Texture size (e.g., 512 for 512x512)
// square_size: Size of each square in pixels (e.g., 32)
// color1: First color (RGBA format 0xAABBGGRR)
// color2: Second color (RGBA format 0xAABBGGRR)
// generate_mips: Whether to generate mipmaps
skr_tex_t su_tex_create_checkerboard(
	int32_t  resolution,
	int32_t  square_size,
	uint32_t color1,
	uint32_t color2,
	bool     generate_mips
);

// Creates a 1x1 solid color texture
// color: RGBA color (format 0xAABBGGRR)
skr_tex_t su_tex_create_solid_color(uint32_t color);

// Loads a texture from an image file with optional mipmap generation
// filename: Path to image file (png, jpg, etc.)
// opt_name: Optional debug name for the texture (can be NULL)
// generate_mips: Whether to generate mipmaps
// Returns: Loaded texture, or invalid texture on failure (check with skr_tex_is_valid)
skr_tex_t su_tex_load(const char* filename, const char* opt_name, bool generate_mips);

///////////////////////////////////////////////////////////////////////////////
// Image Loading
///////////////////////////////////////////////////////////////////////////////

// Loads an image from file using stb_image
// filename: Path to image file (png, jpg, etc.)
// opt_out_width:  Optional output for image width
// opt_out_height: Optional output for image height
// opt_out_format: Optional output for recommended texture format (rgba32_srgb for LDR, rgba128 for HDR)
// force_channels: Number of channels to force (4 for RGBA), or 0 for original
// Returns: Pixel data (must be freed with su_image_free), or NULL on failure
//          For LDR: returns 8-bit per channel data
//          For HDR: returns 32-bit float per channel data (cast to void*)
void* su_image_load(
	const char*    filename,
	int32_t*       opt_out_width,
	int32_t*       opt_out_height,
	skr_tex_fmt_*  opt_out_format,
	int32_t        force_channels
);

// Loads an image from memory using stb_image
// data: Pointer to image file data in memory
// size: Size of image data in bytes
// opt_out_width:  Optional output for image width
// opt_out_height: Optional output for image height
// opt_out_format: Optional output for recommended texture format (rgba32_srgb for LDR, rgba128 for HDR)
// force_channels: Number of channels to force (4 for RGBA), or 0 for original
// Returns: Pixel data (must be freed with su_image_free), or NULL on failure
void* su_image_load_from_memory(
	const void*    data,
	size_t         size,
	int32_t*       opt_out_width,
	int32_t*       opt_out_height,
	skr_tex_fmt_*  opt_out_format,
	int32_t        force_channels
);

// Frees image data allocated by su_image_load or su_image_load_from_memory
void su_image_free(void* pixels);

///////////////////////////////////////////////////////////////////////////////
// File I/O
///////////////////////////////////////////////////////////////////////////////

// Reads a file into memory (platform-agnostic, handles Android assets)
// filename: Path to file (relative paths work on all platforms)
// out_data: Pointer to receive allocated data (must be freed by caller)
// out_size: Pointer to receive file size in bytes
// Returns: true on success, false on failure
bool su_file_read(const char* filename, void** out_data, size_t* out_size);

///////////////////////////////////////////////////////////////////////////////
// Shader Loading
///////////////////////////////////////////////////////////////////////////////

// Loads a shader from a compiled .sks file
// filename: Path to shader file (e.g., "shaders/my_shader.hlsl.sks")
// opt_name: Optional debug name for the shader (can be NULL)
// Returns: Loaded shader, or invalid shader on failure (check with skr_shader_is_valid)
skr_shader_t su_shader_load(const char* filename, const char* opt_name);

///////////////////////////////////////////////////////////////////////////////
// Utility Functions
///////////////////////////////////////////////////////////////////////////////

// Generates a pseudo-random float [0.0, 1.0] from integer position and seed
// Useful for procedural generation with consistent results
float su_hash_f(int32_t position, uint32_t seed);

///////////////////////////////////////////////////////////////////////////////
// File Dialog
///////////////////////////////////////////////////////////////////////////////

// Check if file dialogs are supported on this platform
// Returns true on Windows and Linux (non-Android)
bool su_file_dialog_supported(void);

// Opens a native file dialog and returns the selected path
// title: Dialog window title
// filter_desc: Human-readable description (e.g., "Image Files")
// filter_exts: Semicolon-separated extensions (e.g., "jpg;png;hdr")
// Returns: Selected file path (caller must free), or NULL if cancelled
char* su_file_dialog_open(const char* title, const char* filter_desc, const char* filter_exts);

///////////////////////////////////////////////////////////////////////////////
// GLTF Loading
///////////////////////////////////////////////////////////////////////////////

// Opaque GLTF model handle
typedef struct su_gltf_t su_gltf_t;

// GLTF loading state
typedef enum {
	su_gltf_state_loading,  // Still loading in background
	su_gltf_state_ready,    // Fully loaded and ready to render
	su_gltf_state_failed,   // Loading failed
} su_gltf_state_;

// Axis-aligned bounding box
typedef struct {
	float3 min;
	float3 max;
} su_bounds_t;

// Load a GLTF model asynchronously with multiple shader/material sets
// Each material_info provides a shader and pipeline state template. For each
// mesh, a material is created per template with GLTF textures and params
// bound. GLTF double_sided overrides cull_back→cull_none but respects
// templates that already specify cull_none.
su_gltf_t* su_gltf_load_ex(const char* filename, const skr_material_info_t* material_infos, int32_t material_info_count);

// Load a GLTF model asynchronously (single shader convenience wrapper)
// filename: Path to .gltf or .glb file
// shader: PBR shader to use for materials (caller retains ownership)
su_gltf_t* su_gltf_load(const char* filename, skr_shader_t* shader);

// Destroy a GLTF model and free all resources
void su_gltf_destroy(su_gltf_t* gltf);

// Get the current loading state
su_gltf_state_ su_gltf_get_state(su_gltf_t* gltf);

// Get the overall bounding box of the model (only valid when state is ready)
su_bounds_t su_gltf_get_bounds(su_gltf_t* gltf);

// Add GLTF model meshes to a render list for drawing (uses shader set 0)
// transform: Optional additional transform to apply (can be NULL for identity)
// Does nothing if model is not ready yet
void su_gltf_add_to_render_list(su_gltf_t* gltf, skr_render_list_t* list, const float4x4* opt_transform);

// Add GLTF model meshes to a render list using a specific shader set
// shader_index: Index into the material_infos array from su_gltf_load_ex
void su_gltf_add_to_render_list_shader(su_gltf_t* gltf, skr_render_list_t* list, const float4x4* opt_transform, int32_t shader_index);

// Add GLTF model meshes to a render list with a material override
// opt_material: If non-NULL, all meshes use this material instead of their own
// Useful for shadow caster passes where a cheap depth-only material is needed
void su_gltf_add_to_render_list_override(su_gltf_t* gltf, skr_render_list_t* list, const float4x4* opt_transform, skr_material_t* opt_material);

#ifdef __cplusplus
}
#endif

