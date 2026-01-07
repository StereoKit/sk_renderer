// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// GLTF scene - displays a loaded GLTF model with environment mapping
typedef struct scene_gltf_t {
	scene_t        base;

	// GLTF model (async loaded)
	su_gltf_t*     model;
	char*          model_path;    // Path to currently loaded model (for UI display)
	skr_shader_t   shader;

	// Placeholder sphere while loading
	skr_mesh_t     placeholder_mesh;
	skr_material_t placeholder_material;
	skr_tex_t      white_texture;
	skr_tex_t      black_texture;

	// Cubemap skybox
	skr_tex_t      cubemap_texture;
	skr_tex_t      equirect_texture;
	skr_material_t equirect_convert_material;
	skr_shader_t   equirect_to_cubemap_shader;
	skr_shader_t   skybox_shader;
	skr_shader_t   mipgen_shader;
	skr_material_t skybox_material;
	skr_mesh_t     skybox_mesh;
	bool           cubemap_ready;
	char*          skybox_path;   // Path to currently loaded skybox (for UI display)

	float          rotation;
	float          model_scale;   // User-adjustable model scale multiplier
} scene_gltf_t;

static void _destroy_skybox(scene_gltf_t* scene) {
	if (!scene->cubemap_ready) return;

	skr_mesh_destroy    (&scene->skybox_mesh);
	skr_material_destroy(&scene->skybox_material);
	skr_shader_destroy  (&scene->skybox_shader);
	skr_shader_destroy  (&scene->mipgen_shader);
	skr_shader_destroy  (&scene->equirect_to_cubemap_shader);
	skr_tex_destroy     (&scene->cubemap_texture);

	free(scene->skybox_path);
	scene->skybox_path   = NULL;
	scene->cubemap_ready = false;
}

static void _load_skybox(scene_gltf_t* scene, const char* path) {
	// Destroy existing skybox first
	_destroy_skybox(scene);

	int32_t       equirect_width  = 0;
	int32_t       equirect_height = 0;
	skr_tex_fmt_  equirect_format = skr_tex_fmt_rgba32_srgb;
	void*         equirect_data   = su_image_load(path, &equirect_width, &equirect_height, &equirect_format, 4);

	if (!equirect_data || equirect_width <= 0 || equirect_height <= 0) {
		su_log(su_log_warning, "Failed to load skybox: %s", path);
		return;
	}

	// For HDR (rgb9e5), use same format for cubemap to preserve HDR values
	skr_tex_fmt_ cubemap_format = (equirect_format == skr_tex_fmt_rgb9e5)
		? skr_tex_fmt_rgb9e5
		: skr_tex_fmt_rgba32_srgb;

	// Create equirectangular texture
	skr_tex_create(
		equirect_format,
		skr_tex_flags_readable,
		su_sampler_linear_wrap,
		(skr_vec3i_t){equirect_width, equirect_height, 1},
		1, 0,
		&(skr_tex_data_t){.data = equirect_data, .mip_count = 1, .layer_count = 1},
		&scene->equirect_texture
	);
	skr_tex_set_name(&scene->equirect_texture, "equirect_source");
	su_image_free(equirect_data);

	// Create empty cubemap texture (matches source format for HDR preservation)
	const int32_t cube_size = equirect_height / 2;
	skr_tex_create(
		cubemap_format,
		skr_tex_flags_readable | skr_tex_flags_writeable | skr_tex_flags_cubemap | skr_tex_flags_gen_mips,
		su_sampler_linear_clamp,
		(skr_vec3i_t){cube_size, cube_size, 6},
		1, 0, NULL, &scene->cubemap_texture
	);
	skr_tex_set_name(&scene->cubemap_texture, "environment_cubemap");

	// Load equirect to cubemap shader and convert
	scene->equirect_to_cubemap_shader = su_shader_load("shaders/equirect_to_cubemap.hlsl.sks", "equirect_to_cubemap");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->equirect_to_cubemap_shader,
		.write_mask = skr_write_rgba,
		.cull       = skr_cull_none,
	}, &scene->equirect_convert_material);
	skr_material_set_tex(&scene->equirect_convert_material, "equirect_tex", &scene->equirect_texture);

	// Convert equirectangular to cubemap (wait for completion since we destroy the temp resources immediately)
	skr_renderer_blit(&scene->equirect_convert_material, &scene->cubemap_texture, (skr_recti_t){0, 0, cube_size, cube_size});

	skr_material_destroy(&scene->equirect_convert_material);
	skr_tex_destroy(&scene->equirect_texture);

	// Generate mips with custom shader for IBL
	scene->mipgen_shader = su_shader_load("shaders/cubemap_mipgen.hlsl.sks", "cubemap_mipgen");
	skr_tex_generate_mips(&scene->cubemap_texture, &scene->mipgen_shader);

	// Create skybox
	scene->skybox_shader = su_shader_load("shaders/cubemap_skybox.hlsl.sks", "skybox_shader");
	skr_material_create((skr_material_info_t){
		.shader       = &scene->skybox_shader,
		.write_mask   = skr_write_rgba,
		.depth_test   = skr_compare_less_or_eq,
		.cull         = skr_cull_none,
		.queue_offset = 100,
	}, &scene->skybox_material);
	skr_material_set_tex(&scene->skybox_material, "cubemap", &scene->cubemap_texture);

	scene->skybox_mesh = su_mesh_create_fullscreen_quad();
	skr_mesh_set_name(&scene->skybox_mesh, "skybox_fullscreen_quad");

	scene->cubemap_ready = true;
	scene->skybox_path   = strdup(path);

	su_log(su_log_info, "Loaded skybox: %s (%dx%d)", path, cube_size, cube_size);
}

static void _load_model(scene_gltf_t* scene, const char* path) {
	// Destroy existing model
	if (scene->model) {
		//su_gltf_destroy(scene->model);
	}
	free(scene->model_path);

	// Load new model
	scene->model      = su_gltf_load(path, &scene->shader);
	scene->model_path = strdup(path);

	su_log(su_log_info, "Loading model: %s", path);
}

static scene_t* _scene_gltf_create(void) {
	scene_gltf_t* scene = calloc(1, sizeof(scene_gltf_t));
	if (!scene) return NULL;

	scene->base.size   = sizeof(scene_gltf_t);
	scene->rotation    = 0.0f;
	scene->model_scale = 1.0f;

	// Create fallback textures for placeholder
	scene->white_texture = su_tex_create_solid_color(0xFFFFFFFF);
	scene->black_texture = su_tex_create_solid_color(0xFF000000);
	skr_tex_set_name(&scene->white_texture, "gltf_white_fallback");
	skr_tex_set_name(&scene->black_texture, "gltf_black_fallback");

	// Create placeholder sphere
	skr_vec4_t gray = {0.5f, 0.5f, 0.5f, 1.0f};
	scene->placeholder_mesh = su_mesh_create_sphere(16, 12, 1.0f, gray);
	skr_mesh_set_name(&scene->placeholder_mesh, "gltf_placeholder_sphere");

	// Load PBR shader
	scene->shader = su_shader_load("shaders/pbr.hlsl.sks", "pbr_shader");

	// Create placeholder material
	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.cull       = skr_cull_back,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->placeholder_material);

	// Set default textures for placeholder
	skr_material_set_tex  (&scene->placeholder_material, "albedo_tex",    &scene->white_texture);
	skr_material_set_tex  (&scene->placeholder_material, "emission_tex",  &scene->black_texture);
	skr_material_set_tex  (&scene->placeholder_material, "metal_tex",     &scene->white_texture);
	skr_material_set_tex  (&scene->placeholder_material, "occlusion_tex", &scene->white_texture);
	skr_vec4_t color = {0.5f, 0.5f, 0.5f, 1.0f};
	skr_material_set_param(&scene->placeholder_material, "color", sksc_shader_var_float, 4, &color);
	skr_vec4_t emission = {0.0f, 0.0f, 0.0f, 1.0f};
	skr_material_set_param(&scene->placeholder_material, "emission_factor", sksc_shader_var_float, 4, &emission);
	skr_vec4_t tex_trans = {0.0f, 0.0f, 1.0f, 1.0f};
	skr_material_set_param(&scene->placeholder_material, "tex_trans", sksc_shader_var_float, 4, &tex_trans);
	float metallic = 0.0f;
	skr_material_set_param(&scene->placeholder_material, "metallic", sksc_shader_var_float, 1, &metallic);
	float roughness = 0.8f;
	skr_material_set_param(&scene->placeholder_material, "roughness", sksc_shader_var_float, 1, &roughness);

	// Load GLTF model asynchronously
	scene->model      = su_gltf_load("DamagedHelmet.glb", &scene->shader);
	scene->model_path = strdup("DamagedHelmet.glb");

	// Load default skybox
	_load_skybox(scene, "cubemap.jpg");

	return (scene_t*)scene;
}

static void _scene_gltf_destroy(scene_t* base) {
	scene_gltf_t* scene = (scene_gltf_t*)base;

	// Destroy GLTF model
	su_gltf_destroy(scene->model);
	free(scene->model_path);

	// Destroy placeholder
	skr_mesh_destroy    (&scene->placeholder_mesh);
	skr_material_destroy(&scene->placeholder_material);
	skr_tex_destroy     (&scene->white_texture);
	skr_tex_destroy     (&scene->black_texture);
	skr_shader_destroy  (&scene->shader);

	// Destroy cubemap resources
	_destroy_skybox(scene);

	free(scene);
}

static void _scene_gltf_update(scene_t* base, float delta_time) {
	scene_gltf_t* scene = (scene_gltf_t*)base;
	scene->rotation += delta_time * 0.5f;
}

static void _scene_gltf_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_gltf_t*  scene = (scene_gltf_t*)base;
	su_gltf_state_ state = su_gltf_get_state(scene->model);

	// Set up environment cubemap info in system buffer
	if (scene->cubemap_ready && ref_system_buffer) {
		ref_system_buffer->cubemap_info = (float4){(float)scene->cubemap_texture.size.x, (float)scene->cubemap_texture.size.y, (float)scene->cubemap_texture.mip_levels, 0.0f};
		ref_system_buffer->time         = scene->rotation;

		// Bind environment cubemap globally for all PBR materials (t5 in pbr.hlsl)
		skr_renderer_set_global_texture(5, &scene->cubemap_texture);
	}

	// Render skybox
	if (scene->cubemap_ready) {
		skr_render_list_add(ref_render_list, &scene->skybox_mesh, &scene->skybox_material, NULL, 0, 1);
	}

	if (state != su_gltf_state_ready) {
		// Show placeholder while loading
		float4x4 world = float4x4_trs(
			(float3){0.0f, 0.0f, 0.0f},
			float4_quat_from_euler((float3){0.0f, scene->rotation * 2.0f, 0.0f}),
			(float3){1.0f, 1.0f, 1.0f}
		);
		skr_render_list_add(ref_render_list, &scene->placeholder_mesh, &scene->placeholder_material, &world, sizeof(float4x4), 1);
		return;
	}

	// Compute normalization transform: center the model and scale to fit in a unit sphere
	su_bounds_t bounds = su_gltf_get_bounds(scene->model);
	float3 center = {
		(bounds.min.x + bounds.max.x) * 0.5f,
		(bounds.min.y + bounds.max.y) * 0.5f,
		(bounds.min.z + bounds.max.z) * 0.5f,
	};
	float3 extents = {
		bounds.max.x - bounds.min.x,
		bounds.max.y - bounds.min.y,
		bounds.max.z - bounds.min.z,
	};
	float max_extent = fmaxf(fmaxf(extents.x, extents.y), extents.z);
	float scale = (max_extent > 0.0001f) ? (4.0f / max_extent) : 1.0f;
	scale *= scene->model_scale;  // Apply user scale multiplier

	// Build transform: rotate, then scale, then translate to center at origin
	float4x4 rotation     = float4x4_r(float4_quat_from_euler((float3){0.0f, scene->rotation, 0.0f}));
	float4x4 scale_matrix = float4x4_s((float3){scale, scale, scale});
	float4x4 offset       = float4x4_t((float3){-center.x, -center.y, -center.z});

	// Order: offset (move center to origin) -> scale -> rotate
	float4x4 transform = float4x4_mul(rotation, float4x4_mul(scale_matrix, offset));

	su_gltf_add_to_render_list(scene->model, ref_render_list, &transform);
}

static bool _scene_gltf_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_gltf_t* scene = (scene_gltf_t*)base;

	// Orbit camera
	float radius = 5.0f;
	float height = 2.0f;
	float angle  = scene->rotation * 0.3f;

	out_camera->position = (float3){cosf(angle) * radius, height, sinf(angle) * radius};
	out_camera->target   = (float3){0.0f, 0.0f, 0.0f};
	out_camera->up       = (float3){0.0f, 1.0f, 0.0f};

	return true;
}

// Helper to get just the filename from a path
static const char* _get_filename(const char* path) {
	if (!path) return "(none)";
	const char* last_slash = strrchr(path, '/');
	const char* last_bslash = strrchr(path, '\\');
	const char* name = path;
	if (last_slash && last_slash > name) name = last_slash + 1;
	if (last_bslash && last_bslash > name) name = last_bslash + 1;
	return name;
}

static void _scene_gltf_render_ui(scene_t* base) {
	scene_gltf_t* scene = (scene_gltf_t*)base;

	// Model info and loading
	su_gltf_state_ state = su_gltf_get_state(scene->model);
	const char*    state_str = state == su_gltf_state_ready   ? "Ready" :
	                           state == su_gltf_state_loading ? "Loading..." : "Failed";

	igText("Model: %s", _get_filename(scene->model_path));
	igText("Status: %s", state_str);
	igSliderFloat("Scale", &scene->model_scale, 0.1f, 5.0f, "%.2f", 0);

	if (su_file_dialog_supported()) {
		if (igButton("Load GLTF...", (ImVec2){-1, 0})) {
			char* path = su_file_dialog_open("Select GLTF Model", "GLTF Files", "glb;gltf");
			if (path) {
				_load_model(scene, path);
				free(path);
			}
		}
	} else {
		igBeginDisabled(true);
		igButton("Load GLTF...", (ImVec2){-1, 0});
		igEndDisabled();
		igTextDisabled("(File dialog not available)");
	}

	igSeparator();

	// Skybox info and loading
	igText("Skybox: %s", _get_filename(scene->skybox_path));
	igText("Cubemap: %s", scene->cubemap_ready ? "Ready" : "Not loaded");

	if (su_file_dialog_supported()) {
		if (igButton("Load Skybox...", (ImVec2){-1, 0})) {
			char* path = su_file_dialog_open("Select Skybox Image", "Image Files", "hdr;jpg;png");
			if (path) {
				_load_skybox(scene, path);
				free(path);
			}
		}
	} else {
		igBeginDisabled(true);
		igButton("Load Skybox...", (ImVec2){-1, 0});
		igEndDisabled();
	}
}

const scene_vtable_t scene_gltf_vtable = {
	.name       = "GLTF Model",
	.create     = _scene_gltf_create,
	.destroy    = _scene_gltf_destroy,
	.update     = _scene_gltf_update,
	.render     = _scene_gltf_render,
	.get_camera = _scene_gltf_get_camera,
	.render_ui  = _scene_gltf_render_ui,
};
