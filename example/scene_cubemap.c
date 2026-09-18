// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "tools/compress/tex_compress.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// Cubemap scene - displays a reflective sphere and skybox using a generated cubemap
typedef struct {
	scene_t       base;

	// Meshes
	skr_mesh_t     sphere_mesh;
	skr_mesh_t     skybox_mesh;

	// Shaders
	skr_shader_t   reflection_shader;
	skr_shader_t   skybox_shader;
	skr_shader_t   mipgen_shader;

	// Materials
	skr_material_t sphere_material;
	skr_material_t skybox_material;

	// Textures
	skr_tex_t      cubemap_texture;

	// GPU-compressed copy of the cubemap (BC1 on desktop, ASTC 6x6 where
	// that's the displayable format) with a UI toggle to compare.
	skr_tex_t      cubemap_compressed;
	const char*    compressed_fmt_name;
	bool           show_compressed;

	float rotation;
} scene_cubemap_t;

static scene_t* _scene_cubemap_create(void) {
	scene_cubemap_t* scene = calloc(1, sizeof(scene_cubemap_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_cubemap_t);
	scene->rotation  = 0.0f;

	// Create sphere mesh using utility function (32 segments, 24 rings for smooth reflections)
	skr_vec4_t white = {1.0f, 1.0f, 1.0f, 1.0f};
	scene->sphere_mesh = su_mesh_create_sphere(32, 24, 1.0f, white);
	skr_mesh_set_name(&scene->sphere_mesh, "reflective_sphere");

	// Create fullscreen triangle for skybox
	scene->skybox_mesh = su_mesh_create_fullscreen_quad();
	skr_mesh_set_name(&scene->skybox_mesh, "skybox_fullscreen_quad");

	// Create cubemap texture with different colors per face
	const int cube_size = 512;
	const int face_size = cube_size * cube_size;
	uint32_t* cubemap_data = malloc(face_size * 6 * sizeof(uint32_t));

	// Define colors for each cubemap face - distinct vibrant palette in linear color space
	// Order: +X (right), -X (left), +Y (top), -Y (bottom), +Z (front), -Z (back)
	// Converted from sRGB to linear RGB
	uint32_t face_colors[6] = {
		0xFF0F0AC9,  // +X: Bright Red (#E63946)
		0xFF0036ED,  // -X: Vibrant Orange (#F77F00)
		0xFF1084F8,  // +Y: Golden Yellow (#FCBF49)
		0xFF346200,  // +Z: Emerald Green (#06A77D)
		0xFF180903,  // -Y: Deep Blue (#1D3557)
		0xFF4A1224,  // -Z: Rich Purple (#6A4C93)
	};

	// Fill each face with its color
	for (int face = 0; face < 6; face++) {
		uint32_t* face_data = cubemap_data + (face * face_size);
		for (int i = 0; i < face_size; i++) {
			face_data[i] = face_colors[face];
		}
	}

	// Create cubemap texture (6-layer texture with cubemap flag)
	skr_tex_create(
		skr_tex_fmt_rgba32_srgb,
		skr_tex_flags_readable | skr_tex_flags_cubemap | skr_tex_flags_gen_mips,
		su_sampler_linear_clamp,
		(skr_vec3i_t){cube_size, cube_size, 6},  // 6 faces
		1, 0,
		&(skr_tex_data_t){.data = cubemap_data, .mip_count = 1, .layer_count = 6},
		&scene->cubemap_texture );
	skr_tex_set_name(&scene->cubemap_texture, "color_cubemap");
	free(cubemap_data);

	// Load cubemap mipgen shader for high-quality IBL filtering
	scene->mipgen_shader = su_shader_load("shaders/cubemap_mipgen.hlsl.sks", "cubemap_mipgen");

	// Generate mips for the cubemap using our custom shader
	skr_tex_generate_mips(&scene->cubemap_texture, &scene->mipgen_shader);

	// GPU-compress the cubemap (per-face 2D compression assembled back into a
	// cube). Runs at load so the path is always exercised; the UI toggle picks
	// which version the materials sample.
	tex_compress_init(tex_compress_load_auto);
	if (tex_compress_available(tex_compress_fmt_bc1)) {
		scene->cubemap_compressed  = tex_compress_cube(&scene->cubemap_texture, tex_compress_fmt_bc1);
		scene->compressed_fmt_name = "BC1";
	} else if (tex_compress_available(tex_compress_fmt_astc6x6)) {
		scene->cubemap_compressed  = tex_compress_cube(&scene->cubemap_texture, tex_compress_fmt_astc6x6);
		scene->compressed_fmt_name = "ASTC 6x6";
	}
	if (skr_tex_is_valid(&scene->cubemap_compressed)) {
		skr_tex_set_name(&scene->cubemap_compressed, "color_cubemap_compressed");
		su_log(su_log_info, "Cubemap: GPU-compressed to %s", scene->compressed_fmt_name);
	} else {
		su_log(su_log_warning, "Cubemap: no displayable compressed format, toggle disabled");
	}

	// Load reflection shader
	scene->reflection_shader = su_shader_load("shaders/cubemap_reflection.hlsl.sks", "reflection_shader");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->reflection_shader,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->sphere_material);
	skr_material_set_tex(&scene->sphere_material, "cubemap", &scene->cubemap_texture);

	// Load skybox shader
	scene->skybox_shader = su_shader_load("shaders/cubemap_skybox.hlsl.sks", "skybox_shader");
	skr_material_create((skr_material_info_t){
		.shader       = &scene->skybox_shader,
		.write_mask   = skr_write_default,
		.depth_test   = skr_compare_less_or_eq,  // Less-equal for skybox
		.cull         = skr_cull_front,          // Cull front faces since we're inside
		.queue_offset = 100,                     // Draw last (after sphere)
	}, &scene->skybox_material);
	skr_material_set_tex(&scene->skybox_material, "cubemap", &scene->cubemap_texture);

	return (scene_t*)scene;
}

static void _scene_cubemap_destroy(scene_t* base) {
	scene_cubemap_t* scene = (scene_cubemap_t*)base;

	skr_mesh_destroy(&scene->sphere_mesh);
	skr_mesh_destroy(&scene->skybox_mesh);
	skr_material_destroy(&scene->sphere_material);
	skr_material_destroy(&scene->skybox_material);
	skr_shader_destroy(&scene->reflection_shader);
	skr_shader_destroy(&scene->skybox_shader);
	skr_shader_destroy(&scene->mipgen_shader);
	skr_tex_destroy(&scene->cubemap_texture);
	if (skr_tex_is_valid(&scene->cubemap_compressed))
		skr_tex_destroy(&scene->cubemap_compressed);
	tex_compress_shutdown();

	free(scene);
}

static void _scene_cubemap_update(scene_t* base, float delta_time) {
	scene_cubemap_t* scene = (scene_cubemap_t*)base;
	scene->rotation += delta_time;
}

static void _scene_cubemap_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_cubemap_t* scene = (scene_cubemap_t*)base;

	// Build instance data
	// NOTE: Explicit padding to match HLSL StructuredBuffer alignment (16 bytes)
	typedef struct {
		float4x4 world;
		float    roughness;
		float    _pad[3];  // Pad to 16-byte alignment for HLSL
	} instance_data_t;

	// Create 3x3 grid of spheres with varying roughness
	#define grid_size 3
	instance_data_t sphere_instances[grid_size * grid_size];

	for (int z = 0; z < grid_size; z++) {
		for (int x = 0; x < grid_size; x++) {
			int idx = x + z * grid_size;

			// Position in grid (4 units apart)
			float xpos = (x - (grid_size - 1) * 0.5f) * 4.0f;
			float zpos = (z - (grid_size - 1) * 0.5f) * 4.0f;

			// Apply time-based animation to roughness
			float roughness_cycle = sinf(scene->rotation * 0.5f + x * 2 + z * 7) * 0.5f + 0.5f;
			float roughness       = roughness_cycle;

			sphere_instances[idx].world = float4x4_trs(
				(float3){xpos, 0.0f, zpos},
				float4_quat_from_euler((float3){0.0f, scene->rotation * 0.3f + idx, 0.0f}),
				(float3){1.5f, 1.5f, 1.5f}
			);
			sphere_instances[idx].roughness = roughness;
		}
	}

	// Add to the provided render list
	skr_render_list_add(ref_render_list, &scene->sphere_mesh, &scene->sphere_material, sphere_instances, sizeof(instance_data_t), grid_size * grid_size);
	skr_render_list_add(ref_render_list, &scene->skybox_mesh, &scene->skybox_material, NULL, 0, 1);
}

static bool _scene_cubemap_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_cubemap_t* scene = (scene_cubemap_t*)base;

	// Orbit camera around the grid of spheres
	float radius = 12.0f;  // Further back to see whole grid
	float height = 4.0f;   // Higher up for better view
	float angle  = scene->rotation * 0.4f;  // Smooth orbit

	out_camera->position = (float3){cosf(angle) * radius, height, sinf(angle) * radius};
	out_camera->target   = (float3){0.0f, 0.0f, 0.0f};  // Look at center of grid
	out_camera->up       = (float3){0.0f, 1.0f, 0.0f};

	return true;  // Use this camera
}

static void _scene_cubemap_render_ui(scene_t* base) {
	scene_cubemap_t* scene = (scene_cubemap_t*)base;

	igText("Cubemap");
	igSeparator();
	if (skr_tex_is_valid(&scene->cubemap_compressed)) {
		if (igCheckbox("Show compressed", &scene->show_compressed)) {
			skr_tex_t* tex = scene->show_compressed ? &scene->cubemap_compressed : &scene->cubemap_texture;
			skr_material_set_tex(&scene->sphere_material, "cubemap", tex);
			skr_material_set_tex(&scene->skybox_material, "cubemap", tex);
		}
		igText("Compressed as: %s", scene->compressed_fmt_name);
	} else {
		igTextColored((ImVec4){1.0f, 0.5f, 0.5f, 1.0f}, "No displayable compressed format");
	}
}

const scene_vtable_t scene_cubemap_vtable = {
	.name       = "Cubemap (Reflection & Skybox)",
	.create     = _scene_cubemap_create,
	.destroy    = _scene_cubemap_destroy,
	.update     = _scene_cubemap_update,
	.render     = _scene_cubemap_render,
	.get_camera = _scene_cubemap_get_camera,
	.render_ui  = _scene_cubemap_render_ui,
};
