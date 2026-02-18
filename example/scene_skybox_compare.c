// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// Skybox comparison scene - compare cubemap vs octahedral mapping,
// and fullscreen quad vs fullscreen triangle geometry.
typedef struct {
	scene_t        base;

	// Cubemap resources
	skr_tex_t      cubemap_texture;
	skr_shader_t   cubemap_skybox_shader;
	skr_material_t cubemap_skybox_material;
	skr_shader_t   equirect_to_cubemap_shader;
	skr_shader_t   cubemap_mipgen_shader;

	// Octahedral resources
	skr_tex_t      octahedral_texture;
	skr_shader_t   octahedral_skybox_shader;
	skr_material_t octahedral_skybox_material;
	skr_shader_t   equirect_to_octahedral_shader;

	// Fullscreen geometry
	skr_mesh_t     fullscreen_quad;
	skr_mesh_t     fullscreen_triangle;

	// State
	int32_t        mapping_mode;   // 0 = cubemap, 1 = octahedral
	int32_t        geometry_mode;  // 0 = quad, 1 = triangle
	bool           ready;
	char*          skybox_path;

	// Arc rotation camera
	float          cam_yaw;
	float          cam_pitch;
	float          cam_yaw_vel;
	float          cam_pitch_vel;
	float          delta_time;
} scene_skybox_compare_t;

static void _destroy_skybox(scene_skybox_compare_t* scene) {
	if (!scene->ready) return;

	skr_material_destroy(&scene->cubemap_skybox_material);
	skr_shader_destroy  (&scene->cubemap_skybox_shader);
	skr_shader_destroy  (&scene->equirect_to_cubemap_shader);
	skr_shader_destroy  (&scene->cubemap_mipgen_shader);
	skr_tex_destroy     (&scene->cubemap_texture);

	skr_material_destroy(&scene->octahedral_skybox_material);
	skr_shader_destroy  (&scene->octahedral_skybox_shader);
	skr_shader_destroy  (&scene->equirect_to_octahedral_shader);
	skr_tex_destroy     (&scene->octahedral_texture);

	free(scene->skybox_path);
	scene->skybox_path = NULL;
	scene->ready       = false;
}

static void _load_skybox(scene_skybox_compare_t* scene, const char* path) {
	_destroy_skybox(scene);

	int32_t       equirect_width  = 0;
	int32_t       equirect_height = 0;
	skr_tex_fmt_  equirect_format = skr_tex_fmt_rgba32_srgb;
	void*         equirect_data   = su_image_load(path, &equirect_width, &equirect_height, &equirect_format, 4);

	if (!equirect_data || equirect_width <= 0 || equirect_height <= 0) {
		su_log(su_log_warning, "Failed to load skybox: %s", path);
		return;
	}

	// Create equirectangular source texture
	skr_tex_t equirect_texture;
	skr_tex_create(
		equirect_format,
		skr_tex_flags_readable,
		su_sampler_linear_wrap,
		(skr_vec3i_t){equirect_width, equirect_height, 1},
		1, 0,
		&(skr_tex_data_t){.data = equirect_data, .mip_count = 1, .layer_count = 1},
		&equirect_texture
	);
	skr_tex_set_name(&equirect_texture, "skybox_compare_equirect");
	su_image_free(equirect_data);

	///////////////////////////////////////////////////////////////////////////
	// Cubemap conversion

	const int32_t cube_size = equirect_height / 2;
	skr_tex_create(
		equirect_format,
		skr_tex_flags_readable | skr_tex_flags_writeable | skr_tex_flags_cubemap | skr_tex_flags_gen_mips,
		su_sampler_linear_clamp,
		(skr_vec3i_t){cube_size, cube_size, 6},
		1, 0, NULL, &scene->cubemap_texture
	);
	skr_tex_set_name(&scene->cubemap_texture, "skybox_compare_cubemap");

	scene->equirect_to_cubemap_shader = su_shader_load("shaders/equirect_to_cubemap.hlsl.sks", "equirect_to_cubemap");
	skr_material_t convert_cubemap_mat;
	skr_material_create((skr_material_info_t){
		.shader     = &scene->equirect_to_cubemap_shader,
		.write_mask = skr_write_rgba,
		.cull       = skr_cull_none,
	}, &convert_cubemap_mat);
	skr_material_set_tex(&convert_cubemap_mat, "equirect_tex", &equirect_texture);
	skr_renderer_blit(&convert_cubemap_mat, &scene->cubemap_texture, (skr_recti_t){0, 0, cube_size, cube_size});
	skr_material_destroy(&convert_cubemap_mat);

	scene->cubemap_mipgen_shader = su_shader_load("shaders/cubemap_mipgen.hlsl.sks", "cubemap_mipgen");
	skr_tex_generate_mips(&scene->cubemap_texture, &scene->cubemap_mipgen_shader);

	///////////////////////////////////////////////////////////////////////////
	// Octahedral conversion
	// Mirrored repeat addressing handles edge seams: the octahedral fold's
	// topology means a mirrored copy is the correct neighbor at every edge.
	// See: https://lottes.dev/octahedron/

	const int32_t octa_size = equirect_height;
	skr_tex_sampler_t octa_sampler = { .sample = skr_tex_sample_linear, .address = skr_tex_address_mirror };
	skr_tex_create(
		equirect_format,
		skr_tex_flags_readable | skr_tex_flags_writeable,
		octa_sampler,
		(skr_vec3i_t){octa_size, octa_size, 1},
		1, 0, NULL, &scene->octahedral_texture
	);
	skr_tex_set_name(&scene->octahedral_texture, "skybox_compare_octahedral");

	scene->equirect_to_octahedral_shader = su_shader_load("shaders/equirect_to_octahedral.hlsl.sks", "equirect_to_octahedral");
	skr_material_t convert_octa_mat;
	skr_material_create((skr_material_info_t){
		.shader     = &scene->equirect_to_octahedral_shader,
		.write_mask = skr_write_rgba,
		.cull       = skr_cull_none,
	}, &convert_octa_mat);
	skr_material_set_tex(&convert_octa_mat, "equirect_tex", &equirect_texture);
	skr_renderer_blit(&convert_octa_mat, &scene->octahedral_texture, (skr_recti_t){0, 0, octa_size, octa_size});
	skr_material_destroy(&convert_octa_mat);

	// Done with equirect source
	skr_tex_destroy(&equirect_texture);

	///////////////////////////////////////////////////////////////////////////
	// Skybox materials

	scene->cubemap_skybox_shader = su_shader_load("shaders/cubemap_skybox.hlsl.sks", "cubemap_skybox");
	skr_material_create((skr_material_info_t){
		.shader       = &scene->cubemap_skybox_shader,
		.write_mask   = skr_write_rgba,
		.depth_test   = skr_compare_less_or_eq,
		.cull         = skr_cull_none,
		.queue_offset = 100,
	}, &scene->cubemap_skybox_material);
	skr_material_set_tex(&scene->cubemap_skybox_material, "cubemap", &scene->cubemap_texture);

	scene->octahedral_skybox_shader = su_shader_load("shaders/octahedral_skybox.hlsl.sks", "octahedral_skybox");
	skr_material_create((skr_material_info_t){
		.shader       = &scene->octahedral_skybox_shader,
		.write_mask   = skr_write_rgba,
		.depth_test   = skr_compare_less_or_eq,
		.cull         = skr_cull_none,
		.queue_offset = 100,
	}, &scene->octahedral_skybox_material);
	skr_material_set_tex(&scene->octahedral_skybox_material, "octahedral_tex", &scene->octahedral_texture);

	scene->ready       = true;
	scene->skybox_path = strdup(path);
	su_log(su_log_info, "Skybox compare loaded: %s (cube %dx%d, octa %dx%d)", path, cube_size, cube_size, octa_size, octa_size);
}

static scene_t* _scene_skybox_compare_create(void) {
	scene_skybox_compare_t* scene = calloc(1, sizeof(scene_skybox_compare_t));
	if (!scene) return NULL;

	scene->base.size     = sizeof(scene_skybox_compare_t);
	scene->mapping_mode  = 0;
	scene->geometry_mode = 0;
	scene->cam_yaw       = 0.4f;
	scene->cam_pitch     = 0.0f;

	scene->fullscreen_quad     = su_mesh_create_fullscreen_quad();
	scene->fullscreen_triangle = su_mesh_create_fullscreen_triangle();
	skr_mesh_set_name(&scene->fullscreen_quad,     "skybox_compare_quad");
	skr_mesh_set_name(&scene->fullscreen_triangle,  "skybox_compare_triangle");

	_load_skybox(scene, "cubemap.jpg");

	return (scene_t*)scene;
}

static void _scene_skybox_compare_destroy(scene_t* base) {
	scene_skybox_compare_t* scene = (scene_skybox_compare_t*)base;

	_destroy_skybox(scene);
	skr_mesh_destroy(&scene->fullscreen_quad);
	skr_mesh_destroy(&scene->fullscreen_triangle);

	free(scene);
}

static void _scene_skybox_compare_update(scene_t* base, float delta_time) {
	scene_skybox_compare_t* scene = (scene_skybox_compare_t*)base;
	scene->delta_time = delta_time;
}

static void _scene_skybox_compare_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_skybox_compare_t* scene = (scene_skybox_compare_t*)base;
	if (!scene->ready) return;

	skr_material_t* material = scene->mapping_mode == 0
		? &scene->cubemap_skybox_material
		: &scene->octahedral_skybox_material;
	skr_mesh_t* mesh = scene->geometry_mode == 0
		? &scene->fullscreen_quad
		: &scene->fullscreen_triangle;

	skr_render_list_add(ref_render_list, mesh, material, NULL, 0, 1);
}

static bool _scene_skybox_compare_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_skybox_compare_t* scene = (scene_skybox_compare_t*)base;
	float delta_time = scene->delta_time;

	ImGuiIO* io = igGetIO();

	const float rotate_sensitivity = 0.003f;
	const float velocity_damping   = 0.0001f;
	const float pitch_limit        = 1.5f;

	if (!io->WantCaptureMouse && io->MouseDown[0] && io->MouseDownDuration[0] > 0.0f) {
		scene->cam_yaw_vel   -= io->MouseDelta.x * rotate_sensitivity;
		scene->cam_pitch_vel += io->MouseDelta.y * rotate_sensitivity;
	}

	scene->cam_yaw   += scene->cam_yaw_vel;
	scene->cam_pitch += scene->cam_pitch_vel;

	if (scene->cam_pitch >  pitch_limit) scene->cam_pitch =  pitch_limit;
	if (scene->cam_pitch < -pitch_limit) scene->cam_pitch = -pitch_limit;

	float damping = powf(velocity_damping, delta_time);
	scene->cam_yaw_vel   *= damping;
	scene->cam_pitch_vel *= damping;

	float cos_pitch = cosf(scene->cam_pitch);
	float sin_pitch = sinf(scene->cam_pitch);
	float cos_yaw   = cosf(scene->cam_yaw);
	float sin_yaw   = sinf(scene->cam_yaw);

	float3 forward = {cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw};

	out_camera->position = (float3){0.0f, 0.0f, 0.0f};
	out_camera->target   = forward;
	out_camera->up       = (float3){0.0f, 1.0f, 0.0f};

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

static void _scene_skybox_compare_render_ui(scene_t* base) {
	scene_skybox_compare_t* scene = (scene_skybox_compare_t*)base;

	igText("Skybox: %s", _get_filename(scene->skybox_path));

	igSeparator();
	igText("Mapping");
	igRadioButton_IntPtr("Cubemap",    &scene->mapping_mode, 0);
	igRadioButton_IntPtr("Octahedral", &scene->mapping_mode, 1);

	igSeparator();
	igText("Geometry");
	igRadioButton_IntPtr("Fullscreen Quad",     &scene->geometry_mode, 0);
	igRadioButton_IntPtr("Fullscreen Triangle", &scene->geometry_mode, 1);

	igSeparator();
	igTextWrapped("Left-click drag to look around");
	if (igButton("Reset Camera", (ImVec2){-1, 0})) {
		scene->cam_yaw       = 0.4f;
		scene->cam_pitch     = 0.0f;
		scene->cam_yaw_vel   = 0.0f;
		scene->cam_pitch_vel = 0.0f;
	}

	igSeparator();
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

const scene_vtable_t scene_skybox_compare_vtable = {
	.name       = "Skybox Compare",
	.create     = _scene_skybox_compare_create,
	.destroy    = _scene_skybox_compare_destroy,
	.update     = _scene_skybox_compare_update,
	.render     = _scene_skybox_compare_render,
	.get_camera = _scene_skybox_compare_get_camera,
	.render_ui  = _scene_skybox_compare_render_ui,
};
