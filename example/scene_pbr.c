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

// A grid of PBR spheres varying in metallic (rows) and roughness (columns).
// Classic "PBR chart" for eyeballing the specular prefilter and material response.

#define PBR_GRID_SIZE 6

typedef struct scene_pbr_t {
	scene_t        base;

	skr_shader_t   pbr_shader;
	skr_mesh_t     sphere_mesh;
	skr_material_t materials[PBR_GRID_SIZE * PBR_GRID_SIZE];
	skr_tex_t      white_texture;
	skr_tex_t      black_texture;

	// Cubemap skybox (mirrors scene_gltf's pattern)
	skr_tex_t      cubemap_texture;
	skr_tex_t      equirect_texture;
	skr_material_t equirect_convert_material;
	skr_shader_t   equirect_to_cubemap_shader;
	skr_shader_t   skybox_shader;
	skr_shader_t   mipgen_shader;
	skr_material_t skybox_material;
	skr_mesh_t     skybox_mesh;
	bool           cubemap_ready;
	char*          skybox_path;

	float          time;
	float          delta_time;

	// Shared albedo color for all grid materials
	float          albedo_color[3];

	// Arc-ball camera state
	float  cam_yaw;
	float  cam_pitch;
	float  cam_distance;
	float  cam_yaw_vel;
	float  cam_pitch_vel;
	float  cam_distance_vel;
	float3 cam_target;
	float3 cam_target_vel;
} scene_pbr_t;

// ImGui's color picker operates in sRGB (the perceptual values users expect to
// see in the swatch). Shaders want linear-space albedo, so convert at the seam.
static float _srgb_to_linear(float c) {
	return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

static skr_vec4_t _albedo_srgb_to_linear(const float srgb[3]) {
	return (skr_vec4_t){
		_srgb_to_linear(srgb[0]),
		_srgb_to_linear(srgb[1]),
		_srgb_to_linear(srgb[2]),
		1.0f,
	};
}

static void _reset_camera(scene_pbr_t* scene) {
	scene->cam_yaw          = 0.0f;
	scene->cam_pitch        = 0.0f;
	scene->cam_distance     = 12.0f;
	scene->cam_target       = (float3){0.0f, 0.0f, 0.0f};
	scene->cam_yaw_vel      = 0.0f;
	scene->cam_pitch_vel    = 0.0f;
	scene->cam_distance_vel = 0.0f;
	scene->cam_target_vel   = (float3){0.0f, 0.0f, 0.0f};
}

static void _destroy_skybox(scene_pbr_t* scene) {
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

static void _load_skybox(scene_pbr_t* scene, const char* path) {
	_destroy_skybox(scene);

	int32_t      equirect_width  = 0;
	int32_t      equirect_height = 0;
	skr_tex_fmt_ equirect_format = skr_tex_fmt_rgba32_srgb;
	void*        equirect_data   = su_image_load(path, &equirect_width, &equirect_height, &equirect_format, 4);

	if (!equirect_data || equirect_width <= 0 || equirect_height <= 0) {
		su_log(su_log_warning, "Failed to load skybox: %s", path);
		return;
	}

	skr_tex_create(
		equirect_format,
		skr_tex_flags_readable,
		su_sampler_linear_wrap,
		(skr_vec3i_t){equirect_width, equirect_height, 1},
		1, 0,
		&(skr_tex_data_t){.data = equirect_data, .mip_count = 1, .layer_count = 1},
		&scene->equirect_texture
	);
	skr_tex_set_name(&scene->equirect_texture, "pbr_equirect_source");
	su_image_free(equirect_data);

	// rg11b10 cubemap (HDR-capable, cache-friendly); fall back to rgba64f if
	// the device can't use it as a color attachment.
	const int32_t        cube_size  = equirect_height / 2;
	const skr_tex_flags_ cube_flags = skr_tex_flags_readable | skr_tex_flags_writeable | skr_tex_flags_cubemap | skr_tex_flags_gen_mips;
	skr_tex_fmt_         cube_fmt   = skr_tex_fmt_rg11b10uf;
	if (!skr_tex_fmt_is_supported(cube_fmt, cube_flags, 1)) {
		su_log(su_log_warning, "rg11b10 cubemap unsupported, falling back to rgba64f");
		cube_fmt = skr_tex_fmt_rgba64f;
	}
	skr_tex_create(
		cube_fmt, cube_flags,
		su_sampler_linear_clamp,
		(skr_vec3i_t){cube_size, cube_size, 6},
		1, 0, NULL, &scene->cubemap_texture
	);
	skr_tex_set_name(&scene->cubemap_texture, "pbr_environment_cubemap");

	scene->equirect_to_cubemap_shader = su_shader_load("shaders/equirect_to_cubemap.hlsl.sks", "equirect_to_cubemap");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->equirect_to_cubemap_shader,
		.write_mask = skr_write_rgba,
		.cull       = skr_cull_none,
	}, &scene->equirect_convert_material);
	skr_material_set_tex(&scene->equirect_convert_material, "equirect_tex", &scene->equirect_texture);

	skr_renderer_blit(&scene->equirect_convert_material, &scene->cubemap_texture, (skr_recti_t){0, 0, cube_size, cube_size});

	skr_material_destroy(&scene->equirect_convert_material);
	skr_tex_destroy     (&scene->equirect_texture);

	scene->mipgen_shader = su_shader_load("shaders/cubemap_mipgen.hlsl.sks", "cubemap_mipgen");
	skr_tex_generate_mips(&scene->cubemap_texture, &scene->mipgen_shader);

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
	skr_mesh_set_name(&scene->skybox_mesh, "pbr_skybox_fullscreen_quad");

	scene->cubemap_ready = true;
	scene->skybox_path   = strdup(path);

	su_log(su_log_info, "Loaded skybox: %s (%dx%d)", path, cube_size, cube_size);
}

static scene_t* _scene_pbr_create(void) {
	scene_pbr_t* scene = calloc(1, sizeof(scene_pbr_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_pbr_t);
	scene->time      = 0.0f;
	scene->albedo_color[0] = 1.0f;
	scene->albedo_color[1] = 1.0f;
	scene->albedo_color[2] = 1.0f;
	_reset_camera(scene);

	scene->white_texture = su_tex_create_solid_color(0xFFFFFFFF);
	scene->black_texture = su_tex_create_solid_color(0xFF000000);
	skr_tex_set_name(&scene->white_texture, "pbr_white");
	skr_tex_set_name(&scene->black_texture, "pbr_black");

	skr_vec4_t sphere_color = {1.0f, 1.0f, 1.0f, 1.0f};
	scene->sphere_mesh = su_mesh_create_sphere(32, 24, 0.5f, sphere_color);
	skr_mesh_set_name(&scene->sphere_mesh, "pbr_sphere");

	scene->pbr_shader = su_shader_load("shaders/pbr.hlsl.sks", "pbr_shader");

	// One material per (roughness, metallic) cell. Shared shader, shared mesh —
	// the renderer sorts by material so descriptor switches batch well.
	const skr_vec4_t initial_color = _albedo_srgb_to_linear(scene->albedo_color);
	const skr_vec4_t black         = {0.0f, 0.0f, 0.0f, 1.0f};
	const skr_vec4_t tex_trans     = {0.0f, 0.0f, 1.0f, 1.0f};
	for (int32_t row = 0; row < PBR_GRID_SIZE; row++) {
		for (int32_t col = 0; col < PBR_GRID_SIZE; col++) {
			int32_t         idx = row * PBR_GRID_SIZE + col;
			skr_material_t* mat = &scene->materials[idx];

			skr_material_create((skr_material_info_t){
				.shader     = &scene->pbr_shader,
				.cull       = skr_cull_back,
				.write_mask = skr_write_default,
				.depth_test = skr_compare_less,
			}, mat);

			// metal_tex is (R=AO, G=roughness, B=metallic) — white texture gives
			// multiplier 1, so the scalar `roughness` / `metallic` uniforms drive
			// the final values.
			skr_material_set_tex(mat, "albedo_tex",    &scene->white_texture);
			skr_material_set_tex(mat, "emission_tex",  &scene->black_texture);
			skr_material_set_tex(mat, "metal_tex",     &scene->white_texture);
			skr_material_set_tex(mat, "occlusion_tex", &scene->white_texture);

			skr_material_set_param(mat, "color",           sksc_shader_var_float, 4, &initial_color);
			skr_material_set_param(mat, "emission_factor", sksc_shader_var_float, 4, &black);
			skr_material_set_param(mat, "tex_trans",       sksc_shader_var_float, 4, &tex_trans);

			float metallic  = (float)col / (float)(PBR_GRID_SIZE - 1);
			float roughness = (float)row / (float)(PBR_GRID_SIZE - 1);
			skr_material_set_param(mat, "metallic",  sksc_shader_var_float, 1, &metallic);
			skr_material_set_param(mat, "roughness", sksc_shader_var_float, 1, &roughness);
		}
	}

	_load_skybox(scene, "cubemap.jpg");

	return (scene_t*)scene;
}

static void _scene_pbr_destroy(scene_t* base) {
	scene_pbr_t* scene = (scene_pbr_t*)base;

	for (int32_t i = 0; i < PBR_GRID_SIZE * PBR_GRID_SIZE; i++) {
		skr_material_destroy(&scene->materials[i]);
	}
	skr_mesh_destroy  (&scene->sphere_mesh);
	skr_tex_destroy   (&scene->white_texture);
	skr_tex_destroy   (&scene->black_texture);
	skr_shader_destroy(&scene->pbr_shader);

	_destroy_skybox(scene);

	free(scene);
}

static void _scene_pbr_update(scene_t* base, float delta_time) {
	scene_pbr_t* scene = (scene_pbr_t*)base;
	scene->time      += delta_time;
	scene->delta_time = delta_time;
}

static void _scene_pbr_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_pbr_t* scene = (scene_pbr_t*)base;

	if (scene->cubemap_ready && ref_system_buffer) {
		ref_system_buffer->cubemap_info = (float4){(float)scene->cubemap_texture.size.x, (float)scene->cubemap_texture.size.y, (float)scene->cubemap_texture.mip_levels, 0.0f};
		ref_system_buffer->time         = scene->time;
		skr_renderer_set_global_texture(5, &scene->cubemap_texture);
	}

	if (scene->cubemap_ready) {
		skr_render_list_add(ref_render_list, &scene->skybox_mesh, &scene->skybox_material, NULL, 0, 1);
	}

	// Lay the grid out in XY, centered at origin.
	const float spacing = 1.25f;
	const float offset  = -0.5f * (float)(PBR_GRID_SIZE - 1) * spacing;
	for (int32_t row = 0; row < PBR_GRID_SIZE; row++) {
		for (int32_t col = 0; col < PBR_GRID_SIZE; col++) {
			float3   pos = {offset + col * spacing, offset + row * spacing, 0.0f};
			float4x4 world = float4x4_t(pos);
			skr_render_list_add(ref_render_list, &scene->sphere_mesh, &scene->materials[row * PBR_GRID_SIZE + col], &world, sizeof(float4x4), 1);
		}
	}
}

static bool _scene_pbr_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_pbr_t* scene      = (scene_pbr_t*)base;
	float        delta_time = scene->delta_time;

	const float rotate_sensitivity = 0.0002f;
	const float pan_sensitivity    = 0.0001f;
	const float zoom_sensitivity   = 0.2f;
	const float velocity_damping   = 0.0001f;
	const float pitch_limit        = 1.5f;
	const float min_distance       = 1.0f;
	const float max_distance       = 40.0f;

	ImGuiIO* io = igGetIO();

	if (!io->WantCaptureMouse) {
		if (io->MouseDown[0]) {
			scene->cam_yaw_vel   -= io->MouseDelta.x * rotate_sensitivity;
			scene->cam_pitch_vel += io->MouseDelta.y * rotate_sensitivity;
		}
		if (io->MouseDown[1]) {
			float  cos_yaw   = cosf(scene->cam_yaw);
			float  sin_yaw   = sinf(scene->cam_yaw);
			float3 right     = {cos_yaw, 0.0f, -sin_yaw};
			float  pan_scale = scene->cam_distance * pan_sensitivity;
			scene->cam_target_vel.x -= right.x * io->MouseDelta.x * pan_scale;
			scene->cam_target_vel.z -= right.z * io->MouseDelta.x * pan_scale;
			scene->cam_target_vel.y += io->MouseDelta.y * pan_scale;
		}
		if (io->MouseWheel != 0.0f) {
			scene->cam_distance_vel -= io->MouseWheel * zoom_sensitivity;
		}
	}

	scene->cam_yaw      += scene->cam_yaw_vel;
	scene->cam_pitch    += scene->cam_pitch_vel;
	scene->cam_distance += scene->cam_distance_vel;
	scene->cam_target.x += scene->cam_target_vel.x;
	scene->cam_target.y += scene->cam_target_vel.y;
	scene->cam_target.z += scene->cam_target_vel.z;

	if (scene->cam_pitch    >  pitch_limit ) scene->cam_pitch    =  pitch_limit;
	if (scene->cam_pitch    < -pitch_limit ) scene->cam_pitch    = -pitch_limit;
	if (scene->cam_distance <  min_distance) scene->cam_distance =  min_distance;
	if (scene->cam_distance >  max_distance) scene->cam_distance =  max_distance;

	float damping = powf(velocity_damping, delta_time);
	scene->cam_yaw_vel      *= damping;
	scene->cam_pitch_vel    *= damping;
	scene->cam_distance_vel *= damping;
	scene->cam_target_vel.x *= damping;
	scene->cam_target_vel.y *= damping;
	scene->cam_target_vel.z *= damping;

	float cos_pitch = cosf(scene->cam_pitch);
	float sin_pitch = sinf(scene->cam_pitch);
	float cos_yaw   = cosf(scene->cam_yaw);
	float sin_yaw   = sinf(scene->cam_yaw);

	out_camera->position = (float3){
		scene->cam_target.x + scene->cam_distance * cos_pitch * sin_yaw,
		scene->cam_target.y + scene->cam_distance * sin_pitch,
		scene->cam_target.z + scene->cam_distance * cos_pitch * cos_yaw,
	};
	out_camera->target = scene->cam_target;
	out_camera->up     = (float3){0, 1, 0};
	return true;
}

static const char* _get_filename(const char* path) {
	if (!path) return "(none)";
	const char* last_slash  = strrchr(path, '/');
	const char* last_bslash = strrchr(path, '\\');
	const char* name = path;
	if (last_slash  && last_slash  > name) name = last_slash  + 1;
	if (last_bslash && last_bslash > name) name = last_bslash + 1;
	return name;
}

static void _scene_pbr_render_ui(scene_t* base) {
	scene_pbr_t* scene = (scene_pbr_t*)base;

	igText("Grid: %dx%d spheres", PBR_GRID_SIZE, PBR_GRID_SIZE);
	igText("Rows = roughness (shiny bottom -> rough top)");
	igText("Cols = metallic (dielectric left -> metal right)");

	if (igColorEdit3("Albedo", scene->albedo_color, 0)) {
		skr_vec4_t color = _albedo_srgb_to_linear(scene->albedo_color);
		for (int32_t i = 0; i < PBR_GRID_SIZE * PBR_GRID_SIZE; i++) {
			skr_material_set_param(&scene->materials[i], "color", sksc_shader_var_float, 4, &color);
		}
	}

	igSeparator();

	igText("Distance: %.1f", scene->cam_distance);
	if (igButton("Reset Camera", (ImVec2){0, 0})) {
		_reset_camera(scene);
	}
	igTextWrapped("Left drag: rotate, Right drag: pan, Scroll: zoom");

	igSeparator();

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
		igTextDisabled("(File dialog not available)");
	}
}

const scene_vtable_t scene_pbr_vtable = {
	.name       = "PBR Material Grid",
	.create     = _scene_pbr_create,
	.destroy    = _scene_pbr_destroy,
	.update     = _scene_pbr_update,
	.render     = _scene_pbr_render,
	.get_camera = _scene_pbr_get_camera,
	.render_ui  = _scene_pbr_render_ui,
};
