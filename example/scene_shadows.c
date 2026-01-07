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

// Shadow mapping scene - demonstrates basic single-cascade shadow mapping
typedef struct {
	scene_t            base;
	skr_render_list_t  render_list;
	skr_render_list_t  shadow_list;

	// Shadow map rendering
	skr_tex_t      shadow_map;
	skr_shader_t   shadow_caster_shader;
	skr_material_t shadow_caster_material;

	// Scene rendering
	skr_mesh_t     cube_mesh;
	skr_mesh_t     floor_mesh;
	skr_shader_t   shadow_receiver_shader;
	skr_material_t cube_material;
	skr_material_t floor_material;
	skr_tex_t      checkerboard_texture;
	skr_tex_t      white_texture;

	// Shadow buffer (sent to shader as constants)
	skr_buffer_t   shadow_buffer;

	float  rotation;
	float3 light_dir;
} scene_shadows_t;

// Shadow buffer structure - matches the HLSL cbuffer
typedef struct {
	float4x4 shadow_transform;       // Transforms world space to shadow map UV space
	float3   light_direction;
	float    shadow_bias;
	float3   light_color;
	float    shadow_pixel_size;
} shadow_buffer_data_t;

// Configuration constants
static const float    SHADOW_SCENE_SIZE     = 10.0f;
static const float    SHADOW_MAP_SIZE       = 15.0f;
static const int32_t  SHADOW_MAP_RESOLUTION = 2048;
static const float    SHADOW_MAP_NEAR_CLIP  = 0.01f;
static const float    SHADOW_MAP_FAR_CLIP   = 30.0f;

// Quantize light position to avoid shadow shimmering
static float3 _quantize_light_pos(float3 pos, float4x4 view_matrix, float texel_size) {
	// Transform position to light view space
	float4 view_pos = float4x4_transform_float4(view_matrix, (float4){pos.x, pos.y, pos.z, 1.0f});

	// Quantize x and y to texel grid
	view_pos.x = roundf(view_pos.x / texel_size) * texel_size;
	view_pos.y = roundf(view_pos.y / texel_size) * texel_size;

	// Transform back to world space
	float4x4 view_inv = float4x4_invert(view_matrix);
	float4 world_pos = float4x4_transform_float4(view_inv, view_pos);

	return (float3){world_pos.x, world_pos.y, world_pos.z};
}

static scene_t* _scene_shadows_create(void) {
	scene_shadows_t* scene = calloc(1, sizeof(scene_shadows_t));
	if (!scene) return NULL;

	scene->base.size  = sizeof(scene_shadows_t);
	scene->rotation   = 0.0f;
	scene->light_dir  = float3_norm((float3){1, -1, 0});
	skr_render_list_create(&scene->render_list);
	skr_render_list_create(&scene->shadow_list);

	// Create shadow map (depth texture)
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
		1, 0, NULL, &scene->shadow_map );
	skr_tex_set_name(&scene->shadow_map, "shadow_map");

	// Create cube mesh
	skr_vec4_t cube_colors[6] = {
		{0.8f, 0.3f, 0.3f, 1.0f},  // Front: Red
		{0.3f, 0.8f, 0.3f, 1.0f},  // Back: Green
		{0.3f, 0.3f, 0.8f, 1.0f},  // Top: Blue
		{0.8f, 0.8f, 0.3f, 1.0f},  // Bottom: Yellow
		{0.8f, 0.3f, 0.8f, 1.0f},  // Right: Magenta
		{0.3f, 0.8f, 0.8f, 1.0f},  // Left: Cyan
	};
	scene->cube_mesh = su_mesh_create_cube(1.0f, cube_colors);
	skr_mesh_set_name(&scene->cube_mesh, "shadow_cube");

	// Create floor mesh (large quad on XZ plane)
	skr_vec4_t white = {1.0f, 1.0f, 1.0f, 1.0f};
	skr_vec3_t normal_up = {0, 1, 0};
	scene->floor_mesh = su_mesh_create_quad(SHADOW_SCENE_SIZE, SHADOW_SCENE_SIZE, normal_up, false, white);
	skr_mesh_set_name(&scene->floor_mesh, "shadow_floor");

	// Load shadow caster shader
	scene->shadow_caster_shader = su_shader_load("shaders/shadow_caster.hlsl.sks", "shadow_caster");
	skr_material_create((skr_material_info_t){
		.shader      = &scene->shadow_caster_shader,
		.write_mask  = skr_write_depth,
		.depth_test  = skr_compare_less,
		.depth_clamp = true, // Clamp depth to prevent shadow acne from near/far clipping
	}, &scene->shadow_caster_material);

	// Load shadow receiver shader
	scene->shadow_receiver_shader = su_shader_load("shaders/shadow_receiver.hlsl.sks", "shadow_receiver");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->shadow_receiver_shader,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->cube_material);

	skr_material_create((skr_material_info_t){
		.shader     = &scene->shadow_receiver_shader,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->floor_material);

	// Create textures
	scene->checkerboard_texture = su_tex_create_checkerboard(512, 32, 0xFFFFFFFF, 0xFF888888, true);
	skr_tex_set_name(&scene->checkerboard_texture, "floor_checker");
	scene->white_texture = su_tex_create_solid_color(0xFFFFFFFF);
	skr_tex_set_name(&scene->white_texture, "white_1x1");

	// Bind textures to materials (shadow map will be bound globally per frame)
	skr_material_set_tex(&scene->cube_material,  "tex", &scene->white_texture);
	skr_material_set_tex(&scene->floor_material, "tex", &scene->checkerboard_texture);

	// Create shadow buffer (constant buffer for shadow parameters)
	shadow_buffer_data_t shadow_data = {0};
	skr_buffer_create(&shadow_data, sizeof(shadow_buffer_data_t), 1, skr_buffer_type_constant, skr_use_dynamic, &scene->shadow_buffer);
	skr_buffer_set_name(&scene->shadow_buffer, "shadow_constants");

	return (scene_t*)scene;
}

static void _scene_shadows_destroy(scene_t* base) {
	scene_shadows_t* scene = (scene_shadows_t*)base;

	skr_render_list_destroy(&scene->render_list);
	skr_render_list_destroy(&scene->shadow_list);
	skr_mesh_destroy(&scene->cube_mesh);
	skr_mesh_destroy(&scene->floor_mesh);
	skr_material_destroy(&scene->shadow_caster_material);
	skr_material_destroy(&scene->cube_material);
	skr_material_destroy(&scene->floor_material);
	skr_shader_destroy(&scene->shadow_caster_shader);
	skr_shader_destroy(&scene->shadow_receiver_shader);
	skr_tex_destroy(&scene->shadow_map);
	skr_tex_destroy(&scene->checkerboard_texture);
	skr_tex_destroy(&scene->white_texture);
	skr_buffer_destroy(&scene->shadow_buffer);

	free(scene);
}

static void _scene_shadows_update(scene_t* base, float delta_time) {
	scene_shadows_t* scene = (scene_shadows_t*)base;
	scene->rotation += delta_time * 0.5f;
	scene->light_dir = float3_norm((float3){cosf(scene->rotation), -1, sinf(scene->rotation)});
}

static void _scene_shadows_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_shadows_t* scene = (scene_shadows_t*)base;

	// Setup shadow map matrices
	float3 scene_center = {0, 0, 0};
	float  texel_size = SHADOW_MAP_SIZE / SHADOW_MAP_RESOLUTION;

	// Initial light position (before quantization)
	float3 light_pos_initial = float3_add(scene_center, float3_mul_s(scene->light_dir, -15.0f));

	// Create preliminary shadow view matrix for quantization
	float4x4 shadow_view_prelim = float4x4_lookat(light_pos_initial, float3_add(light_pos_initial, scene->light_dir), (float3){0, 1, 0});

	// Quantize light position to reduce shadow shimmering
	float3 light_pos = _quantize_light_pos(light_pos_initial, shadow_view_prelim, texel_size);

	// Create final shadow map view/projection matrices
	float4x4 shadow_view = float4x4_lookat(light_pos, float3_add(light_pos, scene->light_dir), (float3){0, 1, 0});
	float4x4 shadow_proj = float4x4_orthographic(
		-SHADOW_MAP_SIZE * 0.5f, SHADOW_MAP_SIZE * 0.5f,
		-SHADOW_MAP_SIZE * 0.5f, SHADOW_MAP_SIZE * 0.5f,
		SHADOW_MAP_NEAR_CLIP, SHADOW_MAP_FAR_CLIP
	);

	// Compute shadow transform matrix (world space -> shadow clip space)
	float4x4 shadow_transform = float4x4_mul(shadow_proj, shadow_view);

	// Update shadow buffer
	float slope_scale = fmaxf((SHADOW_MAP_FAR_CLIP - SHADOW_MAP_NEAR_CLIP) / 65536.0f, texel_size);
	shadow_buffer_data_t shadow_data = {
		.shadow_transform  = shadow_transform,
		.light_direction   = float3_mul_s(scene->light_dir, -1.0f),
		.shadow_bias       = slope_scale * 2.0f,
		.light_color       = {1, 1, 1},
		.shadow_pixel_size = 1.0f / SHADOW_MAP_RESOLUTION,
	};
	skr_buffer_set(&scene->shadow_buffer, &shadow_data, sizeof(shadow_buffer_data_t));

	// Build instance data for scene objects
	const int32_t cube_count = 20;
	float4x4 cube_instances[20];

	// Generate random cubes (using deterministic random seed)
	for (int32_t i = 0; i < cube_count; i++) {
		float rand_seed = (float)i / (float)cube_count;
		float x      = (su_hash_f(i * 3 + 0, 1) - 0.5f) * (SHADOW_SCENE_SIZE - 1.0f);
		float y      = (su_hash_f(i * 3 + 1, 1) - 0.5f) * (SHADOW_SCENE_SIZE - 1.0f);
		float size_x = 0.2f + su_hash_f(i * 3 + 2, 1) * 0.4f;
		float size_y = 0.3f + su_hash_f(i * 3 + 2, 1) * 1.5f;
		float size_z = 0.2f + su_hash_f(i * 3 + 2, 1) * 0.4f;

		cube_instances[i] = float4x4_trs(
			(float3){x, 0.01f + size_y * 0.5f, y},
			(float4){0, 0, 0, 1},
			(float3){size_x, size_y, size_z} );
	}

	// Floor instance
	float4x4 floor_instance = float4x4_trs(
		(float3){0.0f, 0.0f, 0.0f},
		(float4){0, 0, 0, 1},
		(float3){1.0f, 1.0f, 1.0f}
	);

	// Create system buffer for shadow pass (orthographic projection)
	su_system_buffer_t shadow_sys_buffer = {0};
	shadow_sys_buffer.view_count = 1;

	float4x4 shadow_viewproj = float4x4_mul(shadow_proj, shadow_view);

	shadow_sys_buffer.view      [0] = shadow_view;
	shadow_sys_buffer.projection[0] = shadow_proj;
	shadow_sys_buffer.viewproj  [0] = shadow_viewproj;

	// Clear global texture/constants that shadow caster doesn't use
	skr_renderer_set_global_constants(13, NULL);
	skr_renderer_set_global_texture  (14, NULL);

	// Render shadow map (depth-only pass)
	skr_renderer_begin_pass(NULL, &scene->shadow_map, NULL, skr_clear_depth, (skr_vec4_t){0, 0, 0, 0}, 1.0f, 0);
	skr_renderer_set_viewport((skr_rect_t ){0, 0, (float)SHADOW_MAP_RESOLUTION, (float)SHADOW_MAP_RESOLUTION});
	skr_renderer_set_scissor ((skr_recti_t){0, 0, SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION});

	skr_render_list_add  (&scene->shadow_list, &scene->cube_mesh,  &scene->shadow_caster_material,  cube_instances, sizeof(float4x4), cube_count);
	skr_render_list_add  (&scene->shadow_list, &scene->floor_mesh, &scene->shadow_caster_material, &floor_instance, sizeof(float4x4), 1);
	skr_renderer_draw    (&scene->shadow_list, &shadow_sys_buffer, sizeof(su_system_buffer_t), 1);
	skr_render_list_clear(&scene->shadow_list);
	skr_renderer_end_pass();

	// Bind shadow buffer and shadow map globally (b13 for constants, t14 for texture to avoid slot conflicts)
	skr_renderer_set_global_constants(13, &scene->shadow_buffer);
	skr_renderer_set_global_texture  (14, &scene->shadow_map);

	// Render scene with shadows to main render target
	skr_render_list_add(ref_render_list, &scene->floor_mesh, &scene->floor_material, &floor_instance, sizeof(float4x4), 1);
	skr_render_list_add(ref_render_list, &scene->cube_mesh,  &scene->cube_material,  cube_instances,  sizeof(float4x4), cube_count);
}

const scene_vtable_t scene_shadows_vtable = {
	.name       = "Shadow Mapping",
	.create     = _scene_shadows_create,
	.destroy    = _scene_shadows_destroy,
	.update     = _scene_shadows_update,
	.render     = _scene_shadows_render,
	.get_camera = NULL,
};
