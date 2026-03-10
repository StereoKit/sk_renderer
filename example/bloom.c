// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "bloom.h"
#include "tools/scene_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	float    texel_size[2];
	float    radius;
	float    intensity;
	uint32_t output_size[2];
	float    _pad[2];
} bloom_params_t;

typedef struct {
	float bloom_strength;
	float _pad[3];
} composite_params_t;

static bloom_t g_bloom = {0};

void bloom_create(int32_t width, int32_t height, int32_t mip_count) {
	g_bloom.bloom_mips = mip_count;
	g_bloom.width      = width;
	g_bloom.height     = height;

	su_log(su_log_info, "Creating bloom mip chain...");
	skr_tex_sampler_t linear_clamp = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };
	int32_t           mip_width    = width  / 2;
	int32_t           mip_height   = height / 2;
	for (int32_t i = 0; i < mip_count; i++) {
		skr_tex_create(skr_tex_fmt_rgba128, skr_tex_flags_readable | skr_tex_flags_compute, linear_clamp, (skr_vec3i_t){mip_width, mip_height, 1}, 1, 1, NULL, &g_bloom.bloom_chain[i]);
		skr_tex_create(skr_tex_fmt_rgba128, skr_tex_flags_readable | skr_tex_flags_compute, linear_clamp, (skr_vec3i_t){mip_width, mip_height, 1}, 1, 1, NULL, &g_bloom.bloom_upsample[i]);
		su_log(su_log_info, "  Bloom mip %d: %dx%d (valid=%d)", i, mip_width, mip_height, skr_tex_is_valid(&g_bloom.bloom_chain[i]));
		mip_width  /= 2;
		mip_height /= 2;
	}

	su_log(su_log_info, "Loading bloom shaders...");
	g_bloom.bloom_downsample_shader = su_shader_load("shaders/bloom_downsample.hlsl.sks", NULL);
	su_log(su_log_info, "  Downsample shader loaded: %d", skr_shader_is_valid(&g_bloom.bloom_downsample_shader));

	g_bloom.bloom_upsample_shader = su_shader_load("shaders/bloom_upsample.hlsl.sks", NULL);
	su_log(su_log_info, "  Upsample shader loaded:   %d", skr_shader_is_valid(&g_bloom.bloom_upsample_shader));

	g_bloom.bloom_composite_shader = su_shader_load("shaders/bloom_composite.hlsl.sks", NULL);
	su_log(su_log_info, "  Composite shader loaded:  %d", skr_shader_is_valid(&g_bloom.bloom_composite_shader));

	// Create compute instances
	skr_material_create((skr_material_info_t){
		.shader       = &g_bloom.bloom_composite_shader,
		.cull         = skr_cull_none,
		.write_mask   = skr_write_r | skr_write_g | skr_write_b | skr_write_a,
		.depth_test   = skr_compare_always,
	}, &g_bloom.bloom_composite_mat);
	for (int32_t i = 0; i < mip_count; i++) {
		skr_compute_create(&g_bloom.bloom_downsample_shader, &g_bloom.bloom_downsample_comp[i]);
		skr_compute_create(&g_bloom.bloom_upsample_shader, &g_bloom.bloom_upsample_comp[i]);
	}

	// Create fullscreen quad mesh
	typedef struct {
		skr_vec2_t position;
		skr_vec2_t uv;
	} vertex_t;
	skr_vert_type_create(
		(skr_vert_component_t[]){
			{ .format = skr_vertex_fmt_f32, .count = 2, .semantic = skr_semantic_position, .semantic_slot = 0 },
			{ .format = skr_vertex_fmt_f32, .count = 2, .semantic = skr_semantic_texcoord, .semantic_slot = 0 }}, 2, &g_bloom.vertex_type);


	skr_mesh_create(&g_bloom.vertex_type, skr_index_fmt_u16,
		(vertex_t[]){{ .position = {-1.0f, -1.0f}, .uv = {0.0f, 0.0f} },
		             { .position = { 1.0f, -1.0f}, .uv = {1.0f, 0.0f} },
		             { .position = { 1.0f,  1.0f}, .uv = {1.0f, 1.0f} },
		             { .position = {-1.0f,  1.0f}, .uv = {0.0f, 1.0f} }}, 4,
		(uint16_t[]) { 0, 1, 2, 2, 3, 0 }, 6, &g_bloom.fullscreen_quad);

	// Create parameter buffers
	mip_width  = width  / 2;
	mip_height = height / 2;
	for (int32_t i = 0; i < mip_count; i++) {
		bloom_params_t params = {
			.texel_size  = { 1.0f / mip_width, 1.0f / mip_height },
			.radius      = 0.8f,
			.intensity   = 0.5f,
			.output_size = { (uint32_t)mip_width, (uint32_t)mip_height }, };
		skr_buffer_create(&params, 1, sizeof(bloom_params_t), skr_buffer_type_constant, skr_use_dynamic, &g_bloom.bloom_params_buffers[i]);
		mip_width  /= 2;
		mip_height /= 2;
	}

	composite_params_t composite_params = { .bloom_strength = 0.0f };
	skr_buffer_create(&composite_params, 1, sizeof(composite_params_t), skr_buffer_type_constant, skr_use_dynamic, &g_bloom.composite_params_buffer);

	su_log(su_log_info, "Bloom system initialized");
}

void bloom_apply(skr_tex_t* scene_color, skr_tex_t* target, float bloom_strength, float radius, float intensity) {
	// Update parameter buffers with new values
	int32_t param_width  = g_bloom.width  / 2;
	int32_t param_height = g_bloom.height / 2;
	for (int32_t i = 0; i < g_bloom.bloom_mips; i++) {
		bloom_params_t params = {
			.texel_size  = { 1.0f / param_width, 1.0f / param_height },
			.radius      = radius,
			.intensity   = intensity,
			.output_size = { (uint32_t)param_width, (uint32_t)param_height },
		};
		skr_buffer_set(&g_bloom.bloom_params_buffers[i], &params, sizeof(bloom_params_t));
		param_width  /= 2;
		param_height /= 2;
	}

	composite_params_t composite_params = { .bloom_strength = bloom_strength };
	skr_buffer_set(&g_bloom.composite_params_buffer, &composite_params, sizeof(composite_params_t));

	// Downsample passes
	int32_t mip_width  = g_bloom.width  / 2;
	int32_t mip_height = g_bloom.height / 2;

	for (int32_t i = 0; i < g_bloom.bloom_mips; i++) {
		skr_tex_t* source = (i == 0) ? scene_color : &g_bloom.bloom_chain[i - 1];
		skr_tex_t* dest   = &g_bloom.bloom_chain[i];

		skr_compute_set_buffer(&g_bloom.bloom_downsample_comp[i], "BloomParams", &g_bloom.bloom_params_buffers[i]);
		skr_compute_set_tex   (&g_bloom.bloom_downsample_comp[i], "source_tex",  source);
		skr_compute_set_tex   (&g_bloom.bloom_downsample_comp[i], "dest_tex",    dest);

		uint32_t dispatch_x = (mip_width  + 7) / 8;
		uint32_t dispatch_y = (mip_height + 7) / 8;
		skr_compute_execute(&g_bloom.bloom_downsample_comp[i], dispatch_x, dispatch_y, 1);

		mip_width  /= 2;
		mip_height /= 2;
	}

	// Upsample passes (smallest mip -> full res, blending up)
	for (int32_t i = g_bloom.bloom_mips - 1; i >= 0; i--) {
		// Calculate the actual size of mip level i
		mip_width  = g_bloom.width / 2;
		mip_height = g_bloom.height / 2;
		for (int32_t j = 0; j < i; j++) {
			mip_width  /= 2;
			mip_height /= 2;
		}

		skr_tex_t* source = &g_bloom.bloom_chain[i];
		skr_tex_t* blend  = (i == g_bloom.bloom_mips - 1) ? &g_bloom.bloom_chain[i] : &g_bloom.bloom_upsample[i + 1];
		skr_tex_t* dest   = &g_bloom.bloom_upsample[i];

		skr_compute_set_buffer(&g_bloom.bloom_upsample_comp[i], "BloomParams", &g_bloom.bloom_params_buffers[i]);
		skr_compute_set_tex   (&g_bloom.bloom_upsample_comp[i], "source_tex",  source);
		skr_compute_set_tex   (&g_bloom.bloom_upsample_comp[i], "blend_tex",   blend);
		skr_compute_set_tex   (&g_bloom.bloom_upsample_comp[i], "dest_tex",    dest);

		uint32_t dispatch_x = (mip_width  + 7) / 8;
		uint32_t dispatch_y = (mip_height + 7) / 8;
		skr_compute_execute(&g_bloom.bloom_upsample_comp[i], dispatch_x, dispatch_y, 1);
	}

	// Composite pass: render fullscreen quad with bloom to target
	skr_material_set_params(&g_bloom.bloom_composite_mat, &g_bloom.composite_params_buffer, sizeof(g_bloom.composite_params_buffer));
	skr_material_set_tex   (&g_bloom.bloom_composite_mat, "scene_tex", scene_color);
	skr_material_set_tex   (&g_bloom.bloom_composite_mat, "bloom_tex", &g_bloom.bloom_upsample[0]);

	skr_render_list_t render_list;
	skr_render_list_create(&render_list);
	skr_renderer_begin_pass(target, NULL, NULL, skr_clear_none, (skr_vec4_t){0, 0, 0, 0}, 1.0f, 0, 0x1, 0x1);
	skr_render_list_add    (&render_list, &g_bloom.fullscreen_quad, &g_bloom.bloom_composite_mat, NULL, 0, 1);
	skr_renderer_draw      (&render_list, NULL, 0);  // No system buffer needed for fullscreen quad
	skr_render_list_destroy(&render_list);
	skr_renderer_end_pass  ();
}

void bloom_resize(int32_t width, int32_t height) {
	g_bloom.width  = width;
	g_bloom.height = height;

	// Destroy old textures
	for (int32_t i = 0; i < g_bloom.bloom_mips; i++) {
		skr_tex_destroy   (&g_bloom.bloom_chain         [i]);
		skr_tex_destroy   (&g_bloom.bloom_upsample      [i]);
		skr_buffer_destroy(&g_bloom.bloom_params_buffers[i]);
	}

	// Recreate textures and buffers
	skr_tex_sampler_t linear_clamp = { .sample = skr_tex_sample_linear, .address = skr_tex_address_clamp };
	int32_t           mip_width    = width  / 2;
	int32_t           mip_height   = height / 2;
	for (int32_t i = 0; i < g_bloom.bloom_mips; i++) {
		skr_tex_create(skr_tex_fmt_rgba128, skr_tex_flags_readable | skr_tex_flags_compute, linear_clamp, (skr_vec3i_t){mip_width, mip_height, 1}, 1, 1, NULL, &g_bloom.bloom_chain[i]);
		skr_tex_create(skr_tex_fmt_rgba128, skr_tex_flags_readable | skr_tex_flags_compute, linear_clamp, (skr_vec3i_t){mip_width, mip_height, 1}, 1, 1, NULL, &g_bloom.bloom_upsample[i]);

		bloom_params_t params = {
			.texel_size  = { 1.0f / mip_width, 1.0f / mip_height },
			.radius      = 0.8f,
			.intensity   = 0.5f,
			.output_size = { (uint32_t)mip_width, (uint32_t)mip_height }, };
		skr_buffer_create(&params, 1, sizeof(bloom_params_t), skr_buffer_type_constant, skr_use_dynamic, &g_bloom.bloom_params_buffers[i]);
		mip_width  /= 2;
		mip_height /= 2;
	}
}

void bloom_destroy(void) {
	for (int32_t i = 0; i < g_bloom.bloom_mips; i++) {
		skr_tex_destroy    (&g_bloom.bloom_chain          [i]);
		skr_tex_destroy    (&g_bloom.bloom_upsample       [i]);
		skr_compute_destroy(&g_bloom.bloom_downsample_comp[i]);
		skr_compute_destroy(&g_bloom.bloom_upsample_comp  [i]);
		skr_buffer_destroy (&g_bloom.bloom_params_buffers [i]);
	}
	skr_material_destroy(&g_bloom.bloom_composite_mat);
	skr_mesh_destroy    (&g_bloom.fullscreen_quad);
	skr_buffer_destroy  (&g_bloom.composite_params_buffer);
	skr_shader_destroy  (&g_bloom.bloom_downsample_shader);
	skr_shader_destroy  (&g_bloom.bloom_upsample_shader);
	skr_shader_destroy  (&g_bloom.bloom_composite_shader);
}
