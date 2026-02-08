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

// GI scene - GLTF model with environment skybox and shadow mapping
// Desktop:  WASD+QE fly camera
// Android:  Arc rotation camera (touch drag to orbit, pinch to zoom)

///////////////////////////////////////////
// Shadow configuration
///////////////////////////////////////////

static const float   SHADOW_MAP_SIZE       = 20.0f;
static const int32_t SHADOW_MAP_RESOLUTION = 2048;
static const float   SHADOW_MAP_NEAR       = 0.01f;
static const float   SHADOW_MAP_FAR        = 50.0f;

///////////////////////////////////////////
// GI probe configuration
///////////////////////////////////////////

#define GI_GRID_SIZE       32
#define GI_CYCLE_LENGTH    (GI_GRID_SIZE * 3) // 96 frames per full update
#define GI_CAM_DISTANCE    50.0f
#define GI_CAM_NEAR        0.0f

///////////////////////////////////////////
// Shadow buffer (matches ShadowBuffer in pbr_shadow.hlsl / shadow_receiver.hlsl)
///////////////////////////////////////////

typedef struct {
	float4x4 shadow_transform;
	float3   light_direction;
	float    shadow_bias;
	float3   light_color;
	float    shadow_pixel_size;
} shadow_buffer_data_t;

///////////////////////////////////////////
// GI buffer (matches GIBuffer in pbr_gi.hlsl)
///////////////////////////////////////////

typedef struct {
	float3 gi_volume_min;
	float  gi_intensity;
	float3 gi_volume_inv; // 1.0 / (volume_max - volume_min)
	float  _gi_pad;
} gi_buffer_data_t;

///////////////////////////////////////////
// Scene state
///////////////////////////////////////////

typedef struct {
	scene_t        base;

	// GLTF model
	su_gltf_t*     model;
	char*          model_path;
	skr_shader_t   shader;        // pbr_shadow shader
	float          model_scale;

	// Placeholder while loading
	skr_mesh_t     placeholder_mesh;
	skr_material_t placeholder_material;
	skr_tex_t      white_texture;
	skr_tex_t      black_texture;

	// Skybox / environment cubemap
	skr_tex_t      cubemap_texture;
	skr_tex_t      equirect_texture;
	skr_material_t equirect_convert_material;
	skr_shader_t   equirect_to_cubemap_shader;
	skr_shader_t   mipgen_shader;
	skr_shader_t   skybox_shader;
	skr_material_t skybox_material;
	skr_mesh_t     skybox_mesh;
	bool           cubemap_ready;
	char*          skybox_path;

	// Shadow mapping
	skr_tex_t          shadow_map;
	skr_shader_t       shadow_caster_shader;
	skr_material_t     shadow_caster_material;
	skr_render_list_t  shadow_list;
	skr_buffer_t       shadow_buffer;
	float              light_angle;     // degrees, 0-360
	float              light_elevation; // 0 = horizontal, 1 = straight down
	float3             light_dir;       // computed from angle/elevation
	float3             light_color;

	// Floor
	skr_mesh_t     floor_mesh;
	skr_material_t floor_material;
	skr_tex_t      floor_texture;

	// GI probe system (single buffer, continuous temporal accumulation)
	skr_tex_t          gi_sh_r;              // SH R (32^3 rgba128, 3D + compute)
	skr_tex_t          gi_sh_g;              // SH G
	skr_tex_t          gi_sh_b;              // SH B
	skr_tex_t          gi_capture_color;     // 32x32 color render target for captures
	skr_tex_t          gi_capture_depth;     // 32x32 depth for capture occlusion
	skr_shader_t       gi_decay_shader;      // Voxel decay (gi_clear.hlsl)
	skr_compute_t      gi_decay_compute;
	skr_render_list_t  gi_capture_list;
	skr_shader_t       gi_capture_shader;    // Simplified diffuse for captures
	skr_material_t     gi_capture_material;  // cull_none, backfaces black
	skr_buffer_t       gi_const_buffer;      // GI constant buffer (b12)
	int32_t            gi_frame;             // 0 to GI_CYCLE_LENGTH-1
	bool               gi_enabled;
	float              gi_intensity;
	float              gi_sh_decay;          // SH EMA decay per frame (0.98 = ~50 frame window)
	uint32_t           gi_total_frames;      // Monotonic frame counter for random seeds
	int32_t            gi_ray_count;         // Rays per probe per frame (1-8)
	float              gi_env_mip;           // Cubemap mip for environment fallback
	float              gi_env_strength;      // Multiplier for environment contribution
	float3             gi_volume_min;
	float3             gi_volume_max;

	// Voxel radiance (double-buffered: build one while reading the other)
	skr_tex_t          gi_voxel[2];           // 32^3 rgba128, radiance + opacity
	int32_t            gi_voxel_write;        // Index we're building into (0 or 1)
	skr_shader_t       gi_voxelize_shader;
	skr_compute_t      gi_voxelize_compute;

	// Voxel-to-SH conversion (per-frame random ray march with EMA)
	skr_shader_t       gi_voxel_to_sh_shader;
	skr_compute_t      gi_voxel_to_sh_compute;

	// GI debug visualization
	skr_shader_t       gi_debug_shader;
	skr_material_t     gi_debug_material;
	skr_mesh_t         gi_debug_mesh;
	bool               gi_show_probes;
	int32_t            gi_debug_mode;    // 0=SH, 1=UVW, 2=L0, 3=SH detail
	bool               show_floor;

	// Camera state
	float  time;
#ifdef __ANDROID__
	// Arc rotation camera (touch)
	float  cam_yaw;
	float  cam_pitch;
	float  cam_distance;
	float  cam_yaw_vel;
	float  cam_pitch_vel;
	float  cam_distance_vel;
	float3 cam_target;
	float3 cam_target_vel;
#else
	// Fly camera (WASD+QE)
	float3 cam_pos;
	float  cam_yaw;
	float  cam_pitch;
#endif
} scene_gi_t;

///////////////////////////////////////////
// Shadow map helpers
///////////////////////////////////////////

static float3 _quantize_light_pos(float3 pos, float4x4 view_matrix, float texel_size) {
	float4 view_pos = float4x4_transform_float4(view_matrix, (float4){pos.x, pos.y, pos.z, 1.0f});
	view_pos.x = roundf(view_pos.x / texel_size) * texel_size;
	view_pos.y = roundf(view_pos.y / texel_size) * texel_size;
	float4x4 view_inv  = float4x4_invert(view_matrix);
	float4   world_pos = float4x4_transform_float4(view_inv, view_pos);
	return (float3){world_pos.x, world_pos.y, world_pos.z};
}

///////////////////////////////////////////
// Skybox loading (mirrors scene_gltf pattern)
///////////////////////////////////////////

static void _destroy_skybox(scene_gi_t* scene) {
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

static void _load_skybox(scene_gi_t* scene, const char* path) {
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
	skr_tex_set_name(&scene->equirect_texture, "gi_equirect_source");
	su_image_free(equirect_data);

	const int32_t cube_size = equirect_height / 2;
	skr_tex_create(
		equirect_format,
		skr_tex_flags_readable | skr_tex_flags_writeable | skr_tex_flags_cubemap | skr_tex_flags_gen_mips,
		su_sampler_linear_clamp,
		(skr_vec3i_t){cube_size, cube_size, 6},
		1, 0, NULL, &scene->cubemap_texture
	);
	skr_tex_set_name(&scene->cubemap_texture, "gi_environment_cubemap");

	scene->equirect_to_cubemap_shader = su_shader_load("shaders/equirect_to_cubemap.hlsl.sks", "equirect_to_cubemap");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->equirect_to_cubemap_shader,
		.write_mask = skr_write_rgba,
		.cull       = skr_cull_none,
	}, &scene->equirect_convert_material);
	skr_material_set_tex(&scene->equirect_convert_material, "equirect_tex", &scene->equirect_texture);

	skr_renderer_blit(&scene->equirect_convert_material, &scene->cubemap_texture, (skr_recti_t){0, 0, cube_size, cube_size});

	skr_material_destroy(&scene->equirect_convert_material);
	skr_tex_destroy(&scene->equirect_texture);

	scene->mipgen_shader = su_shader_load("shaders/cubemap_mipgen.hlsl.sks", "cubemap_mipgen");
	skr_tex_generate_mips(&scene->cubemap_texture, &scene->mipgen_shader);

	scene->skybox_shader = su_shader_load("shaders/cubemap_skybox.hlsl.sks", "gi_skybox_shader");
	skr_material_create((skr_material_info_t){
		.shader       = &scene->skybox_shader,
		.write_mask   = skr_write_rgba,
		.depth_test   = skr_compare_less_or_eq,
		.cull         = skr_cull_none,
		.queue_offset = 100,
	}, &scene->skybox_material);
	skr_material_set_tex(&scene->skybox_material, "cubemap", &scene->cubemap_texture);

	scene->skybox_mesh = su_mesh_create_fullscreen_quad();
	skr_mesh_set_name(&scene->skybox_mesh, "gi_skybox_quad");

	scene->cubemap_ready = true;
	scene->skybox_path   = strdup(path);

	su_log(su_log_info, "GI: Loaded skybox: %s (%dx%d)", path, cube_size, cube_size);
}

///////////////////////////////////////////
// Model loading
///////////////////////////////////////////

static void _load_model(scene_gi_t* scene, const char* path) {
	if (scene->model) {
		su_gltf_destroy(scene->model);
	}
	free(scene->model_path);

	scene->model      = su_gltf_load(path, &scene->shader);
	scene->model_path = strdup(path);

	su_log(su_log_info, "GI: Loading model: %s", path);
}

///////////////////////////////////////////
// Create / Destroy
///////////////////////////////////////////

static scene_t* _scene_gi_create(void) {
	scene_gi_t* scene = calloc(1, sizeof(scene_gi_t));
	if (!scene) return NULL;

	scene->base.size   = sizeof(scene_gi_t);
	scene->model_scale = 1.0f;
	scene->time        = 0.0f;
	scene->light_angle     = 20.0f;
	scene->light_elevation = 0.76f;
	scene->light_color     = (float3){4, 4, 3.5f};

	// Camera defaults
#ifdef __ANDROID__
	scene->cam_yaw      = 0.4f;
	scene->cam_pitch    = 0.3f;
	scene->cam_distance = 8.0f;
	scene->cam_target   = (float3){0.0f, 1.0f, 0.0f};
#else
	scene->cam_pos   = (float3){0.0f, 3.0f, 6.0f};
	scene->cam_yaw   = 0.0f;
	scene->cam_pitch = -0.2f;
#endif

	// Fallback textures
	scene->white_texture = su_tex_create_solid_color(0xFFFFFFFF);
	scene->black_texture = su_tex_create_solid_color(0xFF000000);
	skr_tex_set_name(&scene->white_texture, "gi_white_fallback");
	skr_tex_set_name(&scene->black_texture, "gi_black_fallback");

	// Placeholder mesh
	skr_vec4_t gray = {0.5f, 0.5f, 0.5f, 1.0f};
	scene->placeholder_mesh = su_mesh_create_sphere(16, 12, 1.0f, gray);
	skr_mesh_set_name(&scene->placeholder_mesh, "gi_placeholder_sphere");

	// PBR+GI shader (extends pbr_shadow with GI probe sampling)
	scene->shader = su_shader_load("shaders/pbr_gi.hlsl.sks", "pbr_gi");

	// Placeholder material
	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.cull       = skr_cull_back,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->placeholder_material);
	skr_material_set_tex  (&scene->placeholder_material, "albedo_tex",    &scene->white_texture);
	skr_material_set_tex  (&scene->placeholder_material, "emission_tex",  &scene->black_texture);
	skr_material_set_tex  (&scene->placeholder_material, "metal_tex",     &scene->white_texture);
	skr_material_set_tex  (&scene->placeholder_material, "occlusion_tex", &scene->white_texture);
	skr_vec4_t mat_color = {0.5f, 0.5f, 0.5f, 1.0f};
	skr_material_set_param(&scene->placeholder_material, "color", sksc_shader_var_float, 4, &mat_color);
	skr_vec4_t emission = {0.0f, 0.0f, 0.0f, 1.0f};
	skr_material_set_param(&scene->placeholder_material, "emission_factor", sksc_shader_var_float, 4, &emission);
	skr_vec4_t tex_trans = {0.0f, 0.0f, 1.0f, 1.0f};
	skr_material_set_param(&scene->placeholder_material, "tex_trans", sksc_shader_var_float, 4, &tex_trans);
	float mat_metallic  = 0.0f;
	float mat_roughness = 0.8f;
	skr_material_set_param(&scene->placeholder_material, "metallic",  sksc_shader_var_float, 1, &mat_metallic);
	skr_material_set_param(&scene->placeholder_material, "roughness", sksc_shader_var_float, 1, &mat_roughness);

	// Shadow map
	skr_tex_create(
		skr_tex_fmt_depth16,
		skr_tex_flags_writeable | skr_tex_flags_readable,
		(skr_tex_sampler_t){
			.sample         = skr_tex_sample_linear,
			.address        = skr_tex_address_clamp,
			.sample_compare = skr_compare_less_or_eq,
			.anisotropy     = 1,
		},
		(skr_vec3i_t){SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION, 1},
		1, 0, NULL, &scene->shadow_map
	);
	skr_tex_set_name(&scene->shadow_map, "gi_shadow_map");

	// Shadow caster shader + material
	scene->shadow_caster_shader = su_shader_load("shaders/shadow_caster.hlsl.sks", "gi_shadow_caster");
	skr_material_create((skr_material_info_t){
		.shader      = &scene->shadow_caster_shader,
		.write_mask  = skr_write_depth,
		.depth_test  = skr_compare_less,
		.depth_clamp = true,
		.cull        = skr_cull_none,
	}, &scene->shadow_caster_material);

	skr_render_list_create(&scene->shadow_list);

	// Shadow constant buffer
	shadow_buffer_data_t shadow_data = {0};
	skr_buffer_create(&shadow_data, sizeof(shadow_buffer_data_t), 1, skr_buffer_type_constant, skr_use_dynamic, &scene->shadow_buffer);
	skr_buffer_set_name(&scene->shadow_buffer, "gi_shadow_constants");

	// Floor
	skr_vec4_t white = {1.0f, 1.0f, 1.0f, 1.0f};
	scene->floor_mesh = su_mesh_create_quad(20.0f, 20.0f, (skr_vec3_t){0, 1, 0}, false, white);
	skr_mesh_set_name(&scene->floor_mesh, "gi_floor");

	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->floor_material);
	scene->floor_texture = su_tex_create_checkerboard(512, 32, 0xFFFFFFFF, 0xFF888888, true);
	skr_tex_set_name(&scene->floor_texture, "gi_floor_checker");
	skr_material_set_tex  (&scene->floor_material, "albedo_tex",    &scene->floor_texture);
	skr_material_set_tex  (&scene->floor_material, "emission_tex",  &scene->black_texture);
	skr_material_set_tex  (&scene->floor_material, "metal_tex",     &scene->white_texture);
	skr_material_set_tex  (&scene->floor_material, "occlusion_tex", &scene->white_texture);
	skr_vec4_t floor_color    = {1.0f, 1.0f, 1.0f, 1.0f};
	skr_vec4_t floor_emission = {0.0f, 0.0f, 0.0f, 0.0f};
	skr_vec4_t floor_tex_trans = {0.0f, 0.0f, 1.0f, 1.0f};
	float      floor_metallic  = 0.0f;
	float      floor_roughness = 0.9f;
	skr_material_set_param(&scene->floor_material, "color",           sksc_shader_var_float, 4, &floor_color);
	skr_material_set_param(&scene->floor_material, "emission_factor", sksc_shader_var_float, 4, &floor_emission);
	skr_material_set_param(&scene->floor_material, "tex_trans",       sksc_shader_var_float, 4, &floor_tex_trans);
	skr_material_set_param(&scene->floor_material, "metallic",        sksc_shader_var_float, 1, &floor_metallic);
	skr_material_set_param(&scene->floor_material, "roughness",       sksc_shader_var_float, 1, &floor_roughness);

	// GI probe system
	scene->gi_enabled    = true;
	scene->gi_intensity  = 1.0f;
	scene->gi_volume_min = (float3){-10.0f, -0.5f, -10.0f};
	scene->gi_volume_max = (float3){ 10.0f, 10.0f,  10.0f};
	scene->gi_sh_decay      = 0.985f;
	scene->gi_total_frames  = 0;
	scene->gi_ray_count     = 2;
	scene->gi_env_mip       = 5.0f;
	scene->gi_env_strength  = 1.0f;
	scene->gi_frame         = 0;

	// Create 3D SH textures (single buffer, continuous temporal accumulation)
	skr_tex_sampler_t gi_sampler = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };
	skr_tex_create(skr_tex_fmt_rgba128, skr_tex_flags_3d | skr_tex_flags_readable | skr_tex_flags_compute,
		gi_sampler, (skr_vec3i_t){GI_GRID_SIZE, GI_GRID_SIZE, GI_GRID_SIZE}, 1, 1, NULL, &scene->gi_sh_r);
	skr_tex_create(skr_tex_fmt_rgba128, skr_tex_flags_3d | skr_tex_flags_readable | skr_tex_flags_compute,
		gi_sampler, (skr_vec3i_t){GI_GRID_SIZE, GI_GRID_SIZE, GI_GRID_SIZE}, 1, 1, NULL, &scene->gi_sh_g);
	skr_tex_create(skr_tex_fmt_rgba128, skr_tex_flags_3d | skr_tex_flags_readable | skr_tex_flags_compute,
		gi_sampler, (skr_vec3i_t){GI_GRID_SIZE, GI_GRID_SIZE, GI_GRID_SIZE}, 1, 1, NULL, &scene->gi_sh_b);
	skr_tex_set_name(&scene->gi_sh_r, "gi_sh_r");
	skr_tex_set_name(&scene->gi_sh_g, "gi_sh_g");
	skr_tex_set_name(&scene->gi_sh_b, "gi_sh_b");

	// Capture render targets (32x32)
	skr_tex_create(skr_tex_fmt_rgba64f, skr_tex_flags_writeable | skr_tex_flags_readable,
		(skr_tex_sampler_t){ .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp },
		(skr_vec3i_t){GI_GRID_SIZE, GI_GRID_SIZE, 1}, 1, 0, NULL, &scene->gi_capture_color);
	skr_tex_set_name(&scene->gi_capture_color, "gi_capture_color");

	skr_tex_create(skr_tex_fmt_depth16, skr_tex_flags_writeable,
		(skr_tex_sampler_t){ .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp, .anisotropy = 1 },
		(skr_vec3i_t){GI_GRID_SIZE, GI_GRID_SIZE, 1}, 1, 0, NULL, &scene->gi_capture_depth);
	skr_tex_set_name(&scene->gi_capture_depth, "gi_capture_depth");

	// GI decay compute shader (voxel-only decay at cycle start, SH handled by EMA)
	scene->gi_decay_shader = su_shader_load("shaders/gi_clear.hlsl.sks", "gi_decay");
	skr_compute_create(&scene->gi_decay_shader, &scene->gi_decay_compute);

	// Voxel radiance textures (double-buffered)
	scene->gi_voxel_write = 0;
	for (int32_t i = 0; i < 2; i++) {
		skr_tex_create(skr_tex_fmt_rgba128, skr_tex_flags_3d | skr_tex_flags_readable | skr_tex_flags_compute,
			gi_sampler, (skr_vec3i_t){GI_GRID_SIZE, GI_GRID_SIZE, GI_GRID_SIZE}, 1, 1, NULL, &scene->gi_voxel[i]);
		char name[32];
		snprintf(name, sizeof(name), "gi_voxel_%d", i);
		skr_tex_set_name(&scene->gi_voxel[i], name);
	}

	// Voxelize compute shader
	scene->gi_voxelize_shader = su_shader_load("shaders/gi_voxelize.hlsl.sks", "gi_voxelize");
	skr_compute_create(&scene->gi_voxelize_shader, &scene->gi_voxelize_compute);
	skr_compute_set_tex  (&scene->gi_voxelize_compute, "capture_tex", &scene->gi_capture_color);
	skr_compute_set_param(&scene->gi_voxelize_compute, "grid_size", sksc_shader_var_uint, 1, &(uint32_t){GI_GRID_SIZE});

	// Voxel-to-SH conversion (ray march the completed voxel volume → SH)
	scene->gi_voxel_to_sh_shader = su_shader_load("shaders/gi_voxel_to_sh.hlsl.sks", "gi_voxel_to_sh");
	skr_compute_create(&scene->gi_voxel_to_sh_shader, &scene->gi_voxel_to_sh_compute);
	skr_compute_set_tex  (&scene->gi_voxel_to_sh_compute, "sh_r", &scene->gi_sh_r);
	skr_compute_set_tex  (&scene->gi_voxel_to_sh_compute, "sh_g", &scene->gi_sh_g);
	skr_compute_set_tex  (&scene->gi_voxel_to_sh_compute, "sh_b", &scene->gi_sh_b);
	skr_compute_set_param(&scene->gi_voxel_to_sh_compute, "grid_size", sksc_shader_var_uint, 1, &(uint32_t){GI_GRID_SIZE});

	// GI capture render list and simplified capture material (no backface culling)
	skr_render_list_create(&scene->gi_capture_list);
	scene->gi_capture_shader = su_shader_load("shaders/gi_capture.hlsl.sks", "gi_capture");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->gi_capture_shader,
		.depth_test = skr_compare_less,
		.cull       = skr_cull_none,
	}, &scene->gi_capture_material);

	// GI constant buffer
	gi_buffer_data_t gi_data = {0};
	skr_buffer_create(&gi_data, sizeof(gi_buffer_data_t), 1, skr_buffer_type_constant, skr_use_dynamic, &scene->gi_const_buffer);
	skr_buffer_set_name(&scene->gi_const_buffer, "gi_constants");

	// GI debug visualization
	scene->gi_show_probes  = true;
	scene->show_floor      = true;
	scene->gi_debug_shader = su_shader_load("shaders/gi_debug.hlsl.sks", "gi_debug");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->gi_debug_shader,
		.cull       = skr_cull_back,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->gi_debug_material);
	scene->gi_debug_mesh = su_mesh_create_sphere(8, 6, 1.0f, (skr_vec4_t){1, 1, 1, 1});
	skr_mesh_set_name(&scene->gi_debug_mesh, "gi_debug_probe_sphere");

	// /home/koujaku/Art/Modeling/BounceRoom.glb
	// Load default assets
	scene->model      = su_gltf_load("LightingRoom.glb", &scene->shader);
	scene->model_path = strdup("LightingRoom.glb");
	scene->model_scale = 4;
	_load_skybox(scene, "cubemap.jpg");

	return (scene_t*)scene;
}

static void _scene_gi_destroy(scene_t* base) {
	scene_gi_t* scene = (scene_gi_t*)base;

	// Model
	su_gltf_destroy(scene->model);
	free(scene->model_path);

	// Placeholder
	skr_mesh_destroy    (&scene->placeholder_mesh);
	skr_material_destroy(&scene->placeholder_material);
	skr_tex_destroy     (&scene->white_texture);
	skr_tex_destroy     (&scene->black_texture);
	skr_shader_destroy  (&scene->shader);

	// Skybox
	_destroy_skybox(scene);

	// Shadow
	skr_render_list_destroy(&scene->shadow_list);
	skr_material_destroy   (&scene->shadow_caster_material);
	skr_shader_destroy     (&scene->shadow_caster_shader);
	skr_tex_destroy        (&scene->shadow_map);
	skr_buffer_destroy     (&scene->shadow_buffer);

	// Floor
	skr_mesh_destroy    (&scene->floor_mesh);
	skr_material_destroy(&scene->floor_material);
	skr_tex_destroy     (&scene->floor_texture);

	// GI probes
	skr_tex_destroy        (&scene->gi_sh_r);
	skr_tex_destroy        (&scene->gi_sh_g);
	skr_tex_destroy        (&scene->gi_sh_b);
	skr_tex_destroy        (&scene->gi_capture_color);
	skr_tex_destroy        (&scene->gi_capture_depth);
	skr_compute_destroy    (&scene->gi_decay_compute);
	skr_shader_destroy     (&scene->gi_decay_shader);
	skr_render_list_destroy(&scene->gi_capture_list);
	skr_material_destroy   (&scene->gi_capture_material);
	skr_shader_destroy     (&scene->gi_capture_shader);
	skr_buffer_destroy     (&scene->gi_const_buffer);
	skr_tex_destroy        (&scene->gi_voxel[0]);
	skr_tex_destroy        (&scene->gi_voxel[1]);
	skr_compute_destroy    (&scene->gi_voxelize_compute);
	skr_shader_destroy     (&scene->gi_voxelize_shader);
	skr_compute_destroy    (&scene->gi_voxel_to_sh_compute);
	skr_shader_destroy     (&scene->gi_voxel_to_sh_shader);

	// GI debug
	skr_mesh_destroy    (&scene->gi_debug_mesh);
	skr_material_destroy(&scene->gi_debug_material);
	skr_shader_destroy  (&scene->gi_debug_shader);

	free(scene);
}

///////////////////////////////////////////
// Update (camera controls)
///////////////////////////////////////////

static void _scene_gi_update(scene_t* base, float delta_time) {
	scene_gi_t* scene = (scene_gi_t*)base;
	scene->time += delta_time;

	ImGuiIO* io = igGetIO();

#ifdef __ANDROID__
	// Arc rotation camera (same as scene_stars)
	const float rotate_sensitivity = 0.0002f;
	const float pan_sensitivity    = 0.0001f;
	const float zoom_sensitivity   = 0.2f;
	const float velocity_damping   = 0.0001f;
	const float pitch_limit        = 1.5f;
	const float min_distance       = 1.0f;
	const float max_distance       = 40.0f;

	if (!io->WantCaptureMouse) {
		if (io->MouseDown[0]) {
			scene->cam_yaw_vel   -= io->MouseDelta.x * rotate_sensitivity;
			scene->cam_pitch_vel += io->MouseDelta.y * rotate_sensitivity;
		}
		if (io->MouseDown[1]) {
			float cos_yaw = cosf(scene->cam_yaw);
			float sin_yaw = sinf(scene->cam_yaw);
			float3 right  = { cos_yaw, 0.0f, -sin_yaw };
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

	if (scene->cam_pitch >  pitch_limit) scene->cam_pitch =  pitch_limit;
	if (scene->cam_pitch < -pitch_limit) scene->cam_pitch = -pitch_limit;
	if (scene->cam_distance < min_distance) scene->cam_distance = min_distance;
	if (scene->cam_distance > max_distance) scene->cam_distance = max_distance;

	float damping = powf(velocity_damping, delta_time);
	scene->cam_yaw_vel      *= damping;
	scene->cam_pitch_vel    *= damping;
	scene->cam_distance_vel *= damping;
	scene->cam_target_vel.x *= damping;
	scene->cam_target_vel.y *= damping;
	scene->cam_target_vel.z *= damping;

#else
	// WASD+QE fly camera
	const float move_speed   = 5.0f;
	const float look_speed   = 0.002f;

	// Mouse look (right-click drag)
	if (!io->WantCaptureMouse && io->MouseDown[1]) {
		scene->cam_yaw   -= io->MouseDelta.x * look_speed;
		scene->cam_pitch -= io->MouseDelta.y * look_speed;

		const float pitch_limit = 1.5f;
		if (scene->cam_pitch >  pitch_limit) scene->cam_pitch =  pitch_limit;
		if (scene->cam_pitch < -pitch_limit) scene->cam_pitch = -pitch_limit;
	}

	// Compute forward/right/up from yaw/pitch
	float cos_p = cosf(scene->cam_pitch);
	float sin_p = sinf(scene->cam_pitch);
	float cos_y = cosf(scene->cam_yaw);
	float sin_y = sinf(scene->cam_yaw);

	float3 forward = { -sin_y * cos_p, sin_p, -cos_y * cos_p };
	float3 right   = {  cos_y, 0.0f, -sin_y };
	float3 up      = {  0.0f,  1.0f,  0.0f  };

	float speed = move_speed * delta_time;
	if (!io->WantCaptureKeyboard) {
		if (igIsKeyDown_Nil(ImGuiKey_W)) { scene->cam_pos = float3_add(scene->cam_pos, float3_mul_s(forward,  speed)); }
		if (igIsKeyDown_Nil(ImGuiKey_S)) { scene->cam_pos = float3_add(scene->cam_pos, float3_mul_s(forward, -speed)); }
		if (igIsKeyDown_Nil(ImGuiKey_D)) { scene->cam_pos = float3_add(scene->cam_pos, float3_mul_s(right,    speed)); }
		if (igIsKeyDown_Nil(ImGuiKey_A)) { scene->cam_pos = float3_add(scene->cam_pos, float3_mul_s(right,   -speed)); }
		if (igIsKeyDown_Nil(ImGuiKey_E)) { scene->cam_pos = float3_add(scene->cam_pos, float3_mul_s(up,       speed)); }
		if (igIsKeyDown_Nil(ImGuiKey_Q)) { scene->cam_pos = float3_add(scene->cam_pos, float3_mul_s(up,      -speed)); }
	}
#endif
}

///////////////////////////////////////////
// GI capture helpers
///////////////////////////////////////////

// Build an orthographic view/proj for capturing one axis-aligned direction.
// axis: 0=X, 1=Y, 2=Z. dir_sign: +1 or -1.
// layer_pos: world-space position along the capture axis for this layer.
// Camera is placed AT the layer position, looking outward along dir_sign.
// Near clips right at the layer; objects behind the layer are behind the camera.
static void _gi_build_capture_camera(
	float3 vol_min, float3 vol_max,
	int32_t axis, float dir_sign, float layer_pos,
	float4x4* out_view, float4x4* out_proj, float4x4* out_viewproj)
{
	float3 center = {
		(vol_min.x + vol_max.x) * 0.5f,
		(vol_min.y + vol_max.y) * 0.5f,
		(vol_min.z + vol_max.z) * 0.5f,
	};
	float3 cam_pos = center;
	float3 up      = {0, 1, 0};

	// Place camera at the layer position along the capture axis.
	if      (axis == 0) { cam_pos.x = layer_pos; }
	else if (axis == 1) { cam_pos.y = layer_pos; up = (float3){0, 0, -1}; }
	else                { cam_pos.z = layer_pos; }

	// Look outward in the capture direction.
	float3 forward = {0, 0, 0};
	if      (axis == 0) forward = (float3){dir_sign, 0, 0};
	else if (axis == 1) forward = (float3){0, dir_sign, 0};
	else                forward = (float3){0, 0, dir_sign};

	float3 cam_target = float3_add(cam_pos, forward);
	*out_view = float4x4_lookat(cam_pos, cam_target, up);

	// Orthographic half-extents cover the perpendicular axes of the volume.
	float half_a, half_b;
	if (axis == 0) {
		half_a = (vol_max.z - vol_min.z) * 0.5f;
		half_b = (vol_max.y - vol_min.y) * 0.5f;
	} else if (axis == 1) {
		half_a = (vol_max.x - vol_min.x) * 0.5f;
		half_b = (vol_max.z - vol_min.z) * 0.5f;
	} else {
		half_a = (vol_max.x - vol_min.x) * 0.5f;
		half_b = (vol_max.y - vol_min.y) * 0.5f;
	}

	// Far plane clips to half a cell — camera is at cell center, so half a cell
	// reaches the cell edge. The +dir and -dir captures together cover the full
	// cell without bleeding into neighboring cells.
	float cell_size;
	if      (axis == 0) cell_size = (vol_max.x - vol_min.x) / GI_GRID_SIZE;
	else if (axis == 1) cell_size = (vol_max.y - vol_min.y) / GI_GRID_SIZE;
	else                cell_size = (vol_max.z - vol_min.z) / GI_GRID_SIZE;
	float far_dist = cell_size * 0.5f;

	*out_proj     = float4x4_orthographic(-half_a, half_a, -half_b, half_b, GI_CAM_NEAR, far_dist);
	*out_viewproj = float4x4_mul(*out_proj, *out_view);
}

///////////////////////////////////////////
// Render
///////////////////////////////////////////

static void _scene_gi_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_gi_t* scene = (scene_gi_t*)base;

	// Compute light direction from angle/elevation
	float angle_rad  = scene->light_angle * (3.14159265f / 180.0f);
	float horiz      = sqrtf(1.0f - scene->light_elevation * scene->light_elevation);
	scene->light_dir = (float3){ horiz * cosf(angle_rad), -scene->light_elevation, horiz * sinf(angle_rad) };

	// Environment cubemap setup
	if (scene->cubemap_ready && ref_system_buffer) {
		ref_system_buffer->cubemap_info = (float4){
			(float)scene->cubemap_texture.size.x,
			(float)scene->cubemap_texture.size.y,
			(float)scene->cubemap_texture.mip_levels,
			0.0f
		};
		ref_system_buffer->time = scene->time;
		skr_renderer_set_global_texture(5, &scene->cubemap_texture);
	}

	// Compute model transform (needed by both GI captures and main pass)
	su_gltf_state_ state           = su_gltf_get_state(scene->model);
	float4x4       model_transform = float4x4_identity();
	if (state == su_gltf_state_ready) {
		su_bounds_t bounds = su_gltf_get_bounds(scene->model);
		float3 extents = {
			bounds.max.x - bounds.min.x,
			bounds.max.y - bounds.min.y,
			bounds.max.z - bounds.min.z,
		};
		float max_extent = fmaxf(fmaxf(extents.x, extents.y), extents.z);
		float scale = (max_extent > 0.0001f) ? (4.0f / max_extent) : 1.0f;
		scale *= scene->model_scale;

		float center_x = (bounds.min.x + bounds.max.x) * 0.5f;
		float center_z = (bounds.min.z + bounds.max.z) * 0.5f;

		float4x4 scale_mat = float4x4_s((float3){scale, scale, scale});
		float4x4 offset    = float4x4_t((float3){-center_x, -bounds.min.y, -center_z});
		model_transform     = float4x4_mul(scale_mat, offset);
	}
	float4x4 floor_instance = float4x4_identity();

	// --- Compute shadow data early (needed by GI capture draws and shadow pass) ---
	float  texel_size       = SHADOW_MAP_SIZE / SHADOW_MAP_RESOLUTION;
	float3 light_pos_init   = float3_add((float3){0, 0, 0}, float3_mul_s(scene->light_dir, -20.0f));
	float4x4 shadow_view_pre = float4x4_lookat(light_pos_init, float3_add(light_pos_init, scene->light_dir), (float3){0, 1, 0});
	float3 light_pos         = _quantize_light_pos(light_pos_init, shadow_view_pre, texel_size);

	float4x4 shadow_view = float4x4_lookat(light_pos, float3_add(light_pos, scene->light_dir), (float3){0, 1, 0});
	float4x4 shadow_proj = float4x4_orthographic(
		-SHADOW_MAP_SIZE * 0.5f, SHADOW_MAP_SIZE * 0.5f,
		-SHADOW_MAP_SIZE * 0.5f, SHADOW_MAP_SIZE * 0.5f,
		SHADOW_MAP_NEAR, SHADOW_MAP_FAR
	);
	float4x4 shadow_transform = float4x4_mul(shadow_proj, shadow_view);

	float slope_scale = fmaxf((SHADOW_MAP_FAR - SHADOW_MAP_NEAR) / 65536.0f, texel_size);
	shadow_buffer_data_t shadow_data = {
		.shadow_transform  = shadow_transform,
		.light_direction   = float3_mul_s(scene->light_dir, -1.0f),
		.shadow_bias       = slope_scale * 2.0f,
		.light_color       = scene->light_color,
		.shadow_pixel_size = 1.0f / SHADOW_MAP_RESOLUTION,
	};
	skr_buffer_set(&scene->shadow_buffer, &shadow_data, sizeof(shadow_buffer_data_t));

	// --- GI probe capture (continuous temporal accumulation) ---
	// Voxel decay once at cycle start, SH decay handled per-frame by EMA.
	if (scene->gi_enabled) {
		if (scene->gi_frame == 0) {
			// Clear voxel texture at cycle start
			skr_compute_set_tex  (&scene->gi_decay_compute, "voxel", &scene->gi_voxel[scene->gi_voxel_write]);
			skr_compute_set_param(&scene->gi_decay_compute, "decay", sksc_shader_var_float, 1, &(float){0.0f});
			int32_t clear_dispatch = (GI_GRID_SIZE + 3) / 4;
			skr_compute_execute(&scene->gi_decay_compute, clear_dispatch, clear_dispatch, clear_dispatch);
		}

		int32_t axis  = scene->gi_frame / GI_GRID_SIZE;  // 0=X, 1=Y, 2=Z
		int32_t layer = scene->gi_frame % GI_GRID_SIZE;

		// Compute layer world position (cell-centered: texel i at (i+0.5)/N)
		float3 vol_size = float3_sub(scene->gi_volume_max, scene->gi_volume_min);
		float  layer_pos;
		if      (axis == 0) { layer_pos = scene->gi_volume_min.x + (layer + 0.5f) * vol_size.x / GI_GRID_SIZE; }
		else if (axis == 1) { layer_pos = scene->gi_volume_min.y + (layer + 0.5f) * vol_size.y / GI_GRID_SIZE; }
		else                { layer_pos = scene->gi_volume_min.z + (layer + 0.5f) * vol_size.z / GI_GRID_SIZE; }

		// During captures: set gi_intensity=0 so pbr_gi shader doesn't sample GI (no feedback)
		gi_buffer_data_t gi_capture_data = {
			.gi_volume_min = scene->gi_volume_min,
			.gi_intensity  = 0.0f,
			.gi_volume_inv = {
				1.0f / vol_size.x,
				1.0f / vol_size.y,
				1.0f / vol_size.z,
			},
		};
		skr_buffer_set(&scene->gi_const_buffer, &gi_capture_data, sizeof(gi_buffer_data_t));
		skr_renderer_set_global_constants(12, &scene->gi_const_buffer);

		// Bind SH textures globally (shader sees zeros since gi_intensity=0)
		skr_renderer_set_global_texture(6, &scene->gi_sh_r);
		skr_renderer_set_global_texture(7, &scene->gi_sh_g);
		skr_renderer_set_global_texture(8, &scene->gi_sh_b);

		// Capture +direction and -direction
		float dir_signs[2] = { 1.0f, -1.0f };
		for (int32_t d = 0; d < 2; d++) {
			float    dir_sign = dir_signs[d];
			float4x4 cap_view, cap_proj, cap_viewproj;
			_gi_build_capture_camera(scene->gi_volume_min, scene->gi_volume_max,
				axis, dir_sign, layer_pos, &cap_view, &cap_proj, &cap_viewproj);

			su_system_buffer_t cap_sys = {0};
			cap_sys.view_count   = 1;
			cap_sys.view      [0] = cap_view;
			cap_sys.projection[0] = cap_proj;
			cap_sys.viewproj  [0] = cap_viewproj;
			cap_sys.time          = scene->time;
			if (scene->cubemap_ready) {
				cap_sys.cubemap_info = (float4){
					(float)scene->cubemap_texture.size.x,
					(float)scene->cubemap_texture.size.y,
					(float)scene->cubemap_texture.mip_levels,
					0.0f
				};
			}

			// Shadow data needs to be bound for the PBR shader during capture
			skr_renderer_set_global_constants(13, &scene->shadow_buffer);
			skr_renderer_set_global_texture  (14, &scene->shadow_map);

			// Render scene into capture RT
			skr_renderer_begin_pass(&scene->gi_capture_color, &scene->gi_capture_depth, NULL,
				skr_clear_all, (skr_vec4_t){0, 0, 0, 0}, 1.0f, 0);
			skr_renderer_set_viewport((skr_rect_t ){0, 0, (float)GI_GRID_SIZE, (float)GI_GRID_SIZE});
			skr_renderer_set_scissor ((skr_recti_t){0, 0, GI_GRID_SIZE, GI_GRID_SIZE});

			// Add floor and model to capture list with simplified capture material
			// (cull_none so backfaces are captured, backfaces render as black)
			if (scene->show_floor)
				skr_render_list_add(&scene->gi_capture_list, &scene->floor_mesh, &scene->gi_capture_material, &floor_instance, sizeof(float4x4), 1);
			if (state == su_gltf_state_ready) {
				su_gltf_add_to_render_list_override(scene->model, &scene->gi_capture_list, &model_transform, &scene->gi_capture_material);
			}

			skr_renderer_draw    (&scene->gi_capture_list, &cap_sys, sizeof(su_system_buffer_t), 1);
			skr_render_list_clear(&scene->gi_capture_list);
			skr_renderer_end_pass();

			// Voxelize: write captured radiance into the voxel build buffer
			uint32_t flip_x_val = 0, flip_y_val = (axis != 1) ? 1 : 0;
			if      (axis == 0) { flip_x_val = (dir_sign > 0) ? 0 : 1; }
			else if (axis == 1) { flip_x_val = (dir_sign > 0) ? 1 : 0; }
			else                { flip_x_val = (dir_sign > 0) ? 1 : 0; }
			skr_compute_set_tex  (&scene->gi_voxelize_compute, "voxel_tex",   &scene->gi_voxel[scene->gi_voxel_write]);
			skr_compute_set_param(&scene->gi_voxelize_compute, "axis",        sksc_shader_var_uint, 1, &(uint32_t){axis});
			skr_compute_set_param(&scene->gi_voxelize_compute, "layer_index", sksc_shader_var_uint, 1, &(uint32_t){layer});
			skr_compute_set_param(&scene->gi_voxelize_compute, "flip_x",      sksc_shader_var_uint, 1, &flip_x_val);
			skr_compute_set_param(&scene->gi_voxelize_compute, "flip_y",      sksc_shader_var_uint, 1, &flip_y_val);
			skr_compute_execute(&scene->gi_voxelize_compute, (GI_GRID_SIZE + 7) / 8, (GI_GRID_SIZE + 7) / 8, 1);
		}

		// Ray march low-discrepancy directions per probe each frame (EMA accumulation).
		// Reads from the stable voxel read buffer for consistent results.
		skr_compute_set_tex  (&scene->gi_voxel_to_sh_compute, "voxel_tex",  &scene->gi_voxel[1 - scene->gi_voxel_write]);
		skr_compute_set_param(&scene->gi_voxel_to_sh_compute, "frame_seed", sksc_shader_var_uint,  1, &scene->gi_total_frames);
		skr_compute_set_param(&scene->gi_voxel_to_sh_compute, "sh_decay",   sksc_shader_var_float, 1, &scene->gi_sh_decay);
		skr_compute_set_param(&scene->gi_voxel_to_sh_compute, "ray_count",  sksc_shader_var_uint,  1, &scene->gi_ray_count);
		skr_compute_set_tex  (&scene->gi_voxel_to_sh_compute, "env_cubemap", &scene->cubemap_texture);
		skr_compute_set_param(&scene->gi_voxel_to_sh_compute, "env_mip",      sksc_shader_var_float, 1, &scene->gi_env_mip);
		skr_compute_set_param(&scene->gi_voxel_to_sh_compute, "env_strength", sksc_shader_var_float, 1, &scene->gi_env_strength);
		int32_t vts_dispatch = (GI_GRID_SIZE + 3) / 4;
		skr_compute_execute(&scene->gi_voxel_to_sh_compute, vts_dispatch, vts_dispatch, vts_dispatch);

		// Swap voxel buffers at cycle end
		if (scene->gi_frame == GI_CYCLE_LENGTH - 1) {
			scene->gi_voxel_write = 1 - scene->gi_voxel_write;
		}

		scene->gi_frame = (scene->gi_frame + 1) % GI_CYCLE_LENGTH;
		scene->gi_total_frames++;
	}

	// --- Bind GI for main pass ---
	{
		float3 vol_size = float3_sub(scene->gi_volume_max, scene->gi_volume_min);
		gi_buffer_data_t gi_data = {
			.gi_volume_min = scene->gi_volume_min,
			.gi_intensity  = scene->gi_enabled ? scene->gi_intensity : 0.0f,
			.gi_volume_inv = {
				1.0f / vol_size.x,
				1.0f / vol_size.y,
				1.0f / vol_size.z,
			},
		};
		skr_buffer_set(&scene->gi_const_buffer, &gi_data, sizeof(gi_buffer_data_t));
		skr_renderer_set_global_constants(12, &scene->gi_const_buffer);
		skr_renderer_set_global_texture(6, &scene->gi_sh_r);
		skr_renderer_set_global_texture(7, &scene->gi_sh_g);
		skr_renderer_set_global_texture(8, &scene->gi_sh_b);
		skr_renderer_set_global_texture(9, &scene->gi_voxel[1 - scene->gi_voxel_write]); // voxel read buffer
	}

	// --- Shadow pass ---
	// Shadow pass system buffer (orthographic from light's perspective)
	su_system_buffer_t shadow_sys = {0};
	shadow_sys.view_count   = 1;
	shadow_sys.view      [0] = shadow_view;
	shadow_sys.projection[0] = shadow_proj;
	shadow_sys.viewproj  [0] = shadow_transform;

	// --- Shadow pass (depth only, using cheap shadow_caster shader) ---
	// Bind a dummy texture to slot 14 so the shadow map isn't simultaneously
	// bound as a texture while being used as the depth render target.
	skr_renderer_set_global_texture(14, &scene->white_texture);

	skr_renderer_begin_pass(NULL, &scene->shadow_map, NULL, skr_clear_depth, (skr_vec4_t){0, 0, 0, 0}, 1.0f, 0);
	skr_renderer_set_viewport((skr_rect_t ){0, 0, (float)SHADOW_MAP_RESOLUTION, (float)SHADOW_MAP_RESOLUTION});
	skr_renderer_set_scissor ((skr_recti_t){0, 0, SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION});

	if (scene->show_floor)
		skr_render_list_add(&scene->shadow_list, &scene->floor_mesh, &scene->shadow_caster_material, &floor_instance, sizeof(float4x4), 1);
	if (state == su_gltf_state_ready) {
		su_gltf_add_to_render_list_override(scene->model, &scene->shadow_list, &model_transform, &scene->shadow_caster_material);
	}

	skr_renderer_draw    (&scene->shadow_list, &shadow_sys, sizeof(su_system_buffer_t), 1);
	skr_render_list_clear(&scene->shadow_list);
	skr_renderer_end_pass();

	// --- Main pass setup ---
	skr_renderer_set_global_constants(13, &scene->shadow_buffer);
	skr_renderer_set_global_texture  (14, &scene->shadow_map);

	// Skybox
	if (scene->cubemap_ready) {
		skr_render_list_add(ref_render_list, &scene->skybox_mesh, &scene->skybox_material, NULL, 0, 1);
	}

	// Floor
	if (scene->show_floor)
		skr_render_list_add(ref_render_list, &scene->floor_mesh, &scene->floor_material, &floor_instance, sizeof(float4x4), 1);

	// GLTF model
	if (state != su_gltf_state_ready) {
		float4x4 world = float4x4_trs(
			(float3){0.0f, 1.0f, 0.0f},
			float4_quat_from_euler((float3){0.0f, scene->time * 2.0f, 0.0f}),
			(float3){1.0f, 1.0f, 1.0f}
		);
		skr_render_list_add(ref_render_list, &scene->placeholder_mesh, &scene->placeholder_material, &world, sizeof(float4x4), 1);
	} else {
		su_gltf_add_to_render_list(scene->model, ref_render_list, &model_transform);
	}

	// --- Debug probe visualization ---
	if (scene->gi_show_probes) {
		uint32_t dm = (uint32_t)scene->gi_debug_mode;
		skr_material_set_param(&scene->gi_debug_material, "debug_mode", sksc_shader_var_uint, 1, &dm);
		float3 vol_size  = float3_sub(scene->gi_volume_max, scene->gi_volume_min);
		float3 cell_size = {
			vol_size.x / GI_GRID_SIZE,
			vol_size.y / GI_GRID_SIZE,
			vol_size.z / GI_GRID_SIZE,
		};
		float probe_radius = fminf(fminf(cell_size.x, cell_size.y), cell_size.z) * 0.07f;

		// Batch probes in chunks to avoid huge stack allocations
		#define PROBE_BATCH 1024
		float4x4 probe_transforms[PROBE_BATCH];
		int32_t  batch_count = 0;

		for (int32_t z = 0; z < GI_GRID_SIZE; z++) {
			for (int32_t y = 0; y < GI_GRID_SIZE; y++) {
				for (int32_t x = 0; x < GI_GRID_SIZE; x++) {
					float3 pos = {
						scene->gi_volume_min.x + (x + 0.5f) * cell_size.x,
						scene->gi_volume_min.y + (y + 0.5f) * cell_size.y,
						scene->gi_volume_min.z + (z + 0.5f) * cell_size.z,
					};
					probe_transforms[batch_count] = float4x4_trs(
						pos,
						(float4){0, 0, 0, 1},
						(float3){probe_radius, probe_radius, probe_radius}
					);
					batch_count++;

					if (batch_count == PROBE_BATCH) {
						skr_render_list_add(ref_render_list, &scene->gi_debug_mesh, &scene->gi_debug_material, probe_transforms, sizeof(float4x4), batch_count);
						batch_count = 0;
					}
				}
			}
		}
		if (batch_count > 0) {
			skr_render_list_add(ref_render_list, &scene->gi_debug_mesh, &scene->gi_debug_material, probe_transforms, sizeof(float4x4), batch_count);
		}
		#undef PROBE_BATCH
	}
}

///////////////////////////////////////////
// Camera
///////////////////////////////////////////

static bool _scene_gi_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_gi_t* scene = (scene_gi_t*)base;

#ifdef __ANDROID__
	float cos_pitch = cosf(scene->cam_pitch);
	float sin_pitch = sinf(scene->cam_pitch);
	float cos_yaw   = cosf(scene->cam_yaw);
	float sin_yaw   = sinf(scene->cam_yaw);

	out_camera->position = (float3){
		scene->cam_target.x + scene->cam_distance * cos_pitch * sin_yaw,
		scene->cam_target.y + scene->cam_distance * sin_pitch,
		scene->cam_target.z + scene->cam_distance * cos_pitch * cos_yaw
	};
	out_camera->target = scene->cam_target;
	out_camera->up     = (float3){0, 1, 0};
#else
	float cos_p = cosf(scene->cam_pitch);
	float sin_p = sinf(scene->cam_pitch);
	float cos_y = cosf(scene->cam_yaw);
	float sin_y = sinf(scene->cam_yaw);

	float3 forward = { -sin_y * cos_p, sin_p, -cos_y * cos_p };

	out_camera->position = scene->cam_pos;
	out_camera->target   = float3_add(scene->cam_pos, forward);
	out_camera->up       = (float3){0, 1, 0};
#endif

	return true;
}

///////////////////////////////////////////
// UI
///////////////////////////////////////////

static const char* _get_filename(const char* path) {
	if (!path) return "(none)";
	const char* last_slash  = strrchr(path, '/');
	const char* last_bslash = strrchr(path, '\\');
	const char* name = path;
	if (last_slash  && last_slash  > name) name = last_slash  + 1;
	if (last_bslash && last_bslash > name) name = last_bslash + 1;
	return name;
}

static void _scene_gi_render_ui(scene_t* base) {
	scene_gi_t* scene = (scene_gi_t*)base;

	// Model info
	su_gltf_state_ state = su_gltf_get_state(scene->model);
	const char* state_str = state == su_gltf_state_ready   ? "Ready" :
	                        state == su_gltf_state_loading ? "Loading..." : "Failed";

	igText("Model: %s", _get_filename(scene->model_path));
	igText("Status: %s", state_str);
	igSliderFloat("Scale", &scene->model_scale, 0.1f, 5.0f, "%.2f", 0);
	igCheckbox("Show Floor", &scene->show_floor);

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

	// Skybox
	igText("Skybox: %s", _get_filename(scene->skybox_path));

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

	igSeparator();

	// Light direction (editable)
	igText("Light");
	igSliderFloat("Angle",     &scene->light_angle,     0.0f, 360.0f, "%.0f deg", 0);
	igSliderFloat("Elevation", &scene->light_elevation,  0.0f, 1.0f,  "%.2f",     0);
	igColorEdit3("Color", &scene->light_color.x, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);

	igSeparator();

	// GI probes
	igText("GI Probes");
	igCheckbox("Enabled", &scene->gi_enabled);
	igSliderFloat("Intensity", &scene->gi_intensity, 0.0f, 5.0f, "%.2f", 0);
	igText("Cycle: %d/%d", scene->gi_frame, GI_CYCLE_LENGTH);
	igCheckbox("Show Probes", &scene->gi_show_probes);
	{
		const char* modes[] = { "SH Irradiance", "UVW Coords", "Raw L0", "SH Detail", "Voxel Radiance", "Voxel Opacity" };
		igCombo_Str_arr("Debug Mode", &scene->gi_debug_mode, modes, 6, 0);
	}
	igDragFloat3("Vol Min", &scene->gi_volume_min.x, 0.5f, -50.0f, 50.0f, "%.1f", 0);
	igDragFloat3("Vol Max", &scene->gi_volume_max.x, 0.5f, -50.0f, 50.0f, "%.1f", 0);
	igSliderFloat("SH Decay", &scene->gi_sh_decay, 0.9f, 0.999f, "%.3f", 0);
	igSliderInt("Rays/Probe", &scene->gi_ray_count, 1, 8, "%d", 0);
	igSliderFloat("Env Mip", &scene->gi_env_mip, 0.0f, 10.0f, "%.1f", 0);
	igSliderFloat("Env Strength", &scene->gi_env_strength, 0.0f, 5.0f, "%.2f", 0);

	// GI capture preview
	if (scene->gi_enabled) {
		const char *axis_names[] = {"X", "Y", "Z"};
		int32_t axis  = scene->gi_frame / GI_GRID_SIZE;
		int32_t layer = scene->gi_frame % GI_GRID_SIZE;
		igText("Capture: %s layer %d", axis_names[axis], layer);

		ImVec2 avail;
		igGetContentRegionAvail(&avail);
		float  img_size = avail.x > 0 ? avail.x : 128;
		igImage((ImTextureID)(uintptr_t)&scene->gi_capture_color,
			(ImVec2){img_size, img_size},
			(ImVec2){0, 0}, (ImVec2){1, 1},
			(ImVec4){1, 1, 1, 1}, (ImVec4){0.5f, 0.5f, 0.5f, 0.5f});
	}

	igSeparator();

	// Camera controls hint
#ifdef __ANDROID__
	igTextWrapped("Drag: rotate, Two-finger: pan, Pinch: zoom");
#else
	igTextWrapped("WASD: move, QE: up/down, Right-click drag: look");
#endif

	if (igButton("Reset Camera", (ImVec2){-1, 0})) {
#ifdef __ANDROID__
		scene->cam_yaw          = 0.4f;
		scene->cam_pitch        = 0.3f;
		scene->cam_distance     = 8.0f;
		scene->cam_target       = (float3){0.0f, 1.0f, 0.0f};
		scene->cam_yaw_vel      = 0.0f;
		scene->cam_pitch_vel    = 0.0f;
		scene->cam_distance_vel = 0.0f;
		scene->cam_target_vel   = (float3){0.0f, 0.0f, 0.0f};
#else
		scene->cam_pos   = (float3){0.0f, 3.0f, 8.0f};
		scene->cam_yaw   = 0.0f;
		scene->cam_pitch = -0.2f;
#endif
	}
}

///////////////////////////////////////////
// Vtable
///////////////////////////////////////////

const scene_vtable_t scene_gi_vtable = {
	.name       = "GI Scene",
	.create     = _scene_gi_create,
	.destroy    = _scene_gi_destroy,
	.update     = _scene_gi_update,
	.render     = _scene_gi_render,
	.get_camera = _scene_gi_get_camera,
	.render_ui  = _scene_gi_render_ui,
};
