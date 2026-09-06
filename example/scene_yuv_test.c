// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>

#include <sk_app.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// YUV Texture Format Demo
// Tests the skr_tex_create() YUV path with generated checker patterns.
// Displays NV12, P010, and YUV420P textures side-by-side for visual comparison.

#define YUV_TEX_WIDTH  256
#define YUV_TEX_HEIGHT 256
#define YUV_CHECKER_SIZE 32

typedef struct {
	scene_t        base;
	skr_mesh_t     quad_mesh;
	skr_shader_t   shader;

	// One material+texture per format
	skr_material_t material_nv12;
	skr_material_t material_p010;
	skr_material_t material_yuv420p;
	skr_tex_t      texture_nv12;
	skr_tex_t      texture_p010;
	skr_tex_t      texture_yuv420p;

	// Reference RGBA texture for comparison
	skr_material_t material_rgba;
	skr_tex_t      texture_rgba;

	bool           nv12_supported;
	bool           p010_supported;
	bool           yuv420p_supported;

	float          cam_distance;
} scene_yuv_test_t;

///////////////////////////////////////////////////////////////////////////////
// YUV Data Generation
///////////////////////////////////////////////////////////////////////////////

// BT.709 RGB -> YCbCr (narrow range) conversion
// Y  = 0.2126*R + 0.7152*G + 0.0722*B  -> [16..235]
// Cb = -0.1146*R - 0.3854*G + 0.5*B     -> [16..240] (offset by 128)
// Cr = 0.5*R - 0.4542*G - 0.0458*B      -> [16..240] (offset by 128)
static void _rgb_to_ycbcr_709(uint8_t r, uint8_t g, uint8_t b,
                               uint8_t* out_y, uint8_t* out_cb, uint8_t* out_cr) {
	float rf = r / 255.0f;
	float gf = g / 255.0f;
	float bf = b / 255.0f;

	float y  =  0.2126f * rf + 0.7152f * gf + 0.0722f * bf;
	float cb = -0.1146f * rf - 0.3854f * gf + 0.5000f * bf;
	float cr =  0.5000f * rf - 0.4542f * gf - 0.0458f * bf;

	// Scale to narrow range
	int32_t yi  = (int32_t)(y  * 219.0f + 16.5f);
	int32_t cbi = (int32_t)(cb * 224.0f + 128.5f);
	int32_t cri = (int32_t)(cr * 224.0f + 128.5f);

	*out_y  = (uint8_t)(yi  < 0 ? 0 : yi  > 255 ? 255 : yi);
	*out_cb = (uint8_t)(cbi < 0 ? 0 : cbi > 255 ? 255 : cbi);
	*out_cr = (uint8_t)(cri < 0 ? 0 : cri > 255 ? 255 : cri);
}

// Generate a checker pattern with 4 colored quadrants
// Top-left: Red, Top-right: Green, Bottom-left: Blue, Bottom-right: Yellow
static void _get_checker_rgb(int32_t x, int32_t y, int32_t w, int32_t h,
                             uint8_t* out_r, uint8_t* out_g, uint8_t* out_b) {
	// Determine quadrant color
	uint8_t base_r, base_g, base_b;
	if (x < w / 2 && y < h / 2) {
		base_r = 220; base_g = 50;  base_b = 50;  // Red
	} else if (x >= w / 2 && y < h / 2) {
		base_r = 50;  base_g = 200; base_b = 50;  // Green
	} else if (x < w / 2) {
		base_r = 50;  base_g = 80;  base_b = 220; // Blue
	} else {
		base_r = 230; base_g = 220; base_b = 50;  // Yellow
	}

	// Checker pattern modulates brightness
	int32_t cx = (x / YUV_CHECKER_SIZE) & 1;
	int32_t cy = (y / YUV_CHECKER_SIZE) & 1;
	float   scale = (cx ^ cy) ? 1.0f : 0.5f;

	*out_r = (uint8_t)(base_r * scale);
	*out_g = (uint8_t)(base_g * scale);
	*out_b = (uint8_t)(base_b * scale);
}

// Generate NV12 data: Y plane (WxH bytes) + interleaved UV plane (Wx(H/2) bytes)
static void* _generate_nv12(int32_t w, int32_t h) {
	uint32_t y_size  = w * h;
	uint32_t uv_size = w * (h / 2);
	uint8_t* data = malloc(y_size + uv_size);
	if (!data) return NULL;

	uint8_t* y_plane  = data;
	uint8_t* uv_plane = data + y_size;

	// Fill Y plane
	for (int32_t py = 0; py < h; py++) {
		for (int32_t px = 0; px < w; px++) {
			uint8_t r, g, b, y_val, cb, cr;
			_get_checker_rgb(px, py, w, h, &r, &g, &b);
			_rgb_to_ycbcr_709(r, g, b, &y_val, &cb, &cr);
			y_plane[py * w + px] = y_val;
			(void)cb; (void)cr;
		}
	}

	// Fill UV plane (interleaved, 2x2 averaged chroma)
	for (int32_t py = 0; py < h / 2; py++) {
		for (int32_t px = 0; px < w / 2; px++) {
			uint32_t cb_sum = 0, cr_sum = 0;
			for (int32_t dy = 0; dy < 2; dy++) {
				for (int32_t dx = 0; dx < 2; dx++) {
					uint8_t r, g, b, y_val, cb, cr;
					_get_checker_rgb(px * 2 + dx, py * 2 + dy, w, h, &r, &g, &b);
					_rgb_to_ycbcr_709(r, g, b, &y_val, &cb, &cr);
					cb_sum += cb;
					cr_sum += cr;
				}
			}
			uv_plane[py * w + px * 2 + 0] = (uint8_t)(cb_sum / 4);
			uv_plane[py * w + px * 2 + 1] = (uint8_t)(cr_sum / 4);
		}
	}

	return data;
}

// Generate P010 data: Y plane (Wx H x 2 bytes) + interleaved UV plane (Wx(H/2)x2 bytes)
// P010 uses 16-bit words with data in the upper 10 bits
static void* _generate_p010(int32_t w, int32_t h) {
	uint32_t  y_size  = w * h * 2;
	uint32_t  uv_size = w * (h / 2) * 2;
	uint8_t*  data = malloc(y_size + uv_size);
	if (!data) return NULL;

	uint16_t* y_plane  = (uint16_t*)data;
	uint16_t* uv_plane = (uint16_t*)(data + y_size);

	// Fill Y plane (10-bit in upper bits of 16-bit word)
	for (int32_t py = 0; py < h; py++) {
		for (int32_t px = 0; px < w; px++) {
			uint8_t r, g, b, y_val, cb, cr;
			_get_checker_rgb(px, py, w, h, &r, &g, &b);
			_rgb_to_ycbcr_709(r, g, b, &y_val, &cb, &cr);
			y_plane[py * w + px] = (uint16_t)(y_val << 8); // 8-bit value shifted to upper 10 bits (roughly)
			(void)cb; (void)cr;
		}
	}

	// Fill UV plane (interleaved 16-bit Cb Cr)
	for (int32_t py = 0; py < h / 2; py++) {
		for (int32_t px = 0; px < w / 2; px++) {
			uint32_t cb_sum = 0, cr_sum = 0;
			for (int32_t dy = 0; dy < 2; dy++) {
				for (int32_t dx = 0; dx < 2; dx++) {
					uint8_t r, g, b, y_val, cb, cr;
					_get_checker_rgb(px * 2 + dx, py * 2 + dy, w, h, &r, &g, &b);
					_rgb_to_ycbcr_709(r, g, b, &y_val, &cb, &cr);
					cb_sum += cb;
					cr_sum += cr;
				}
			}
			uv_plane[py * w + px * 2 + 0] = (uint16_t)((cb_sum / 4) << 8);
			uv_plane[py * w + px * 2 + 1] = (uint16_t)((cr_sum / 4) << 8);
		}
	}

	return data;
}

// Generate YUV420P data: Y plane (WxH) + U plane (W/2 x H/2) + V plane (W/2 x H/2)
static void* _generate_yuv420p(int32_t w, int32_t h) {
	uint32_t y_size = w * h;
	uint32_t u_size = (w / 2) * (h / 2);
	uint32_t v_size = u_size;
	uint8_t* data = malloc(y_size + u_size + v_size);
	if (!data) return NULL;

	uint8_t* y_plane = data;
	uint8_t* u_plane = data + y_size;
	uint8_t* v_plane = data + y_size + u_size;

	// Fill Y plane
	for (int32_t py = 0; py < h; py++) {
		for (int32_t px = 0; px < w; px++) {
			uint8_t r, g, b, y_val, cb, cr;
			_get_checker_rgb(px, py, w, h, &r, &g, &b);
			_rgb_to_ycbcr_709(r, g, b, &y_val, &cb, &cr);
			y_plane[py * w + px] = y_val;
			(void)cb; (void)cr;
		}
	}

	// Fill U and V planes (2x2 averaged)
	int32_t half_w = w / 2;
	for (int32_t py = 0; py < h / 2; py++) {
		for (int32_t px = 0; px < half_w; px++) {
			uint32_t cb_sum = 0, cr_sum = 0;
			for (int32_t dy = 0; dy < 2; dy++) {
				for (int32_t dx = 0; dx < 2; dx++) {
					uint8_t r, g, b, y_val, cb, cr;
					_get_checker_rgb(px * 2 + dx, py * 2 + dy, w, h, &r, &g, &b);
					_rgb_to_ycbcr_709(r, g, b, &y_val, &cb, &cr);
					cb_sum += cb;
					cr_sum += cr;
				}
			}
			u_plane[py * half_w + px] = (uint8_t)(cb_sum / 4);
			v_plane[py * half_w + px] = (uint8_t)(cr_sum / 4);
		}
	}

	return data;
}

// Generate RGBA reference texture with the same checker pattern
static void* _generate_rgba_reference(int32_t w, int32_t h) {
	uint8_t* data = malloc(w * h * 4);
	if (!data) return NULL;

	for (int32_t py = 0; py < h; py++) {
		for (int32_t px = 0; px < w; px++) {
			uint8_t r, g, b;
			_get_checker_rgb(px, py, w, h, &r, &g, &b);
			uint8_t* pixel = &data[(py * w + px) * 4];
			pixel[0] = r;
			pixel[1] = g;
			pixel[2] = b;
			pixel[3] = 255;
		}
	}

	return data;
}

///////////////////////////////////////////////////////////////////////////////
// Scene Implementation
///////////////////////////////////////////////////////////////////////////////

static scene_t* _scene_yuv_create(void) {
	scene_yuv_test_t* scene = calloc(1, sizeof(scene_yuv_test_t));
	if (!scene) return NULL;

	scene->base.size    = sizeof(scene_yuv_test_t);
	scene->cam_distance = 5.0f;

	// Check format support
	scene->nv12_supported    = skr_tex_fmt_is_supported(skr_tex_fmt_nv12,    skr_tex_flags_none, 1);
	scene->p010_supported    = skr_tex_fmt_is_supported(skr_tex_fmt_p010,    skr_tex_flags_none, 1);
	scene->yuv420p_supported = skr_tex_fmt_is_supported(skr_tex_fmt_yuv420p, skr_tex_flags_none, 1);

	su_log(su_log_info, "YUV Test: NV12 %s, P010 %s, YUV420P %s",
		scene->nv12_supported    ? "supported" : "not supported",
		scene->p010_supported    ? "supported" : "not supported",
		scene->yuv420p_supported ? "supported" : "not supported");

	// Create quad mesh
	scene->quad_mesh = su_mesh_create_quad(2.0f, 2.0f, (skr_vec3_t){0, 0, 1}, false, (skr_vec4_t){1, 1, 1, 1});
	skr_mesh_set_name(&scene->quad_mesh, "yuv_test_quad");

	scene->shader = su_shader_load("shaders/yuv_unlit.hlsl.sks", "yuv_test_shader");

	// Create materials
	skr_material_info_t mat_info = {
		.shader      = &scene->shader,
		.cull        = skr_cull_back,
		.depth_test  = skr_compare_less,
		.blend_state = skr_blend_off,
	};
	skr_material_create(mat_info, &scene->material_nv12);
	skr_material_create(mat_info, &scene->material_p010);
	skr_material_create(mat_info, &scene->material_yuv420p);
	skr_material_create(mat_info, &scene->material_rgba);

	int32_t w = YUV_TEX_WIDTH;
	int32_t h = YUV_TEX_HEIGHT;

	// Create RGBA reference
	void* rgba_data = _generate_rgba_reference(w, h);
	if (rgba_data) {
		skr_tex_create(skr_tex_fmt_rgba32_srgb,
			skr_tex_flags_none,
			su_sampler_point_clamp,
			(skr_vec3i_t){w, h, 1}, 1, 0,
			&(skr_tex_data_t){.data = rgba_data, .mip_count = 1, .layer_count = 1},
			&scene->texture_rgba);
		skr_tex_set_name(&scene->texture_rgba, "yuv_ref_rgba");
		skr_material_set_tex(&scene->material_rgba, "tex", &scene->texture_rgba);
		free(rgba_data);
	}

	// Create NV12 texture
	if (scene->nv12_supported) {
		void* nv12_data = _generate_nv12(w, h);
		if (nv12_data) {
			skr_tex_create(skr_tex_fmt_nv12,
				skr_tex_flags_none,
				su_sampler_point_clamp,
				(skr_vec3i_t){w, h, 1}, 1, 0,
				&(skr_tex_data_t){.data = nv12_data, .mip_count = 1, .layer_count = 1},
				&scene->texture_nv12);
			skr_tex_set_name(&scene->texture_nv12, "yuv_nv12");
			skr_material_set_tex(&scene->material_nv12, "tex", &scene->texture_nv12);
			free(nv12_data);
		}
	}

	// Create P010 texture
	if (scene->p010_supported) {
		void* p010_data = _generate_p010(w, h);
		if (p010_data) {
			skr_tex_create(skr_tex_fmt_p010,
				skr_tex_flags_none,
				su_sampler_point_clamp,
				(skr_vec3i_t){w, h, 1}, 1, 0,
				&(skr_tex_data_t){.data = p010_data, .mip_count = 1, .layer_count = 1},
				&scene->texture_p010);
			skr_tex_set_name(&scene->texture_p010, "yuv_p010");
			skr_material_set_tex(&scene->material_p010, "tex", &scene->texture_p010);
			free(p010_data);
		}
	}

	// Create YUV420P texture
	if (scene->yuv420p_supported) {
		void* yuv420p_data = _generate_yuv420p(w, h);
		if (yuv420p_data) {
			skr_tex_create(skr_tex_fmt_yuv420p,
				skr_tex_flags_none,
				su_sampler_point_clamp,
				(skr_vec3i_t){w, h, 1}, 1, 0,
				&(skr_tex_data_t){.data = yuv420p_data, .mip_count = 1, .layer_count = 1},
				&scene->texture_yuv420p);
			skr_tex_set_name(&scene->texture_yuv420p, "yuv_420p");
			skr_material_set_tex(&scene->material_yuv420p, "tex", &scene->texture_yuv420p);
			free(yuv420p_data);
		}
	}

	return (scene_t*)scene;
}

static void _scene_yuv_destroy(scene_t* base) {
	scene_yuv_test_t* scene = (scene_yuv_test_t*)base;

	skr_mesh_destroy    (&scene->quad_mesh);
	skr_material_destroy(&scene->material_nv12);
	skr_material_destroy(&scene->material_p010);
	skr_material_destroy(&scene->material_yuv420p);
	skr_material_destroy(&scene->material_rgba);
	skr_shader_destroy  (&scene->shader);

	if (skr_tex_is_valid(&scene->texture_nv12))    skr_tex_destroy(&scene->texture_nv12);
	if (skr_tex_is_valid(&scene->texture_p010))    skr_tex_destroy(&scene->texture_p010);
	if (skr_tex_is_valid(&scene->texture_yuv420p)) skr_tex_destroy(&scene->texture_yuv420p);
	if (skr_tex_is_valid(&scene->texture_rgba))    skr_tex_destroy(&scene->texture_rgba);

	free(scene);
}

static void _scene_yuv_update(scene_t* base, float delta_time) {
	(void)base;
	(void)delta_time;
}

static void _scene_yuv_render(scene_t* base, int32_t width, int32_t height,
                              skr_render_list_t* ref_render_list,
                              su_system_buffer_t* ref_system_buffer) {
	scene_yuv_test_t* scene = (scene_yuv_test_t*)base;
	(void)width; (void)height; (void)ref_system_buffer;

	// Count how many textures we have to display (RGBA ref + supported YUV formats)
	int32_t count = 1; // Always have RGBA reference
	if (scene->nv12_supported)    count++;
	if (scene->p010_supported)    count++;
	if (scene->yuv420p_supported) count++;

	// Layout: spread quads evenly
	float spacing   = 2.2f;
	float total     = (count - 1) * spacing;
	float start_x   = -total / 2.0f;
	int32_t idx     = 0;

	// RGBA reference (always first)
	if (skr_tex_is_valid(&scene->texture_rgba)) {
		float4x4 world = float4x4_trs((float3){start_x + idx * spacing, 0.0f, 0.0f}, (float4){0, 0, 0, 1}, (float3){1, 1, 1});
		skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->material_rgba, &world, sizeof(float4x4), 1);
		idx++;
	}

	if (scene->nv12_supported && skr_tex_is_valid(&scene->texture_nv12)) {
		float4x4 world = float4x4_trs((float3){start_x + idx * spacing, 0.0f, 0.0f}, (float4){0, 0, 0, 1}, (float3){1, 1, 1});
		skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->material_nv12, &world, sizeof(float4x4), 1);
		idx++;
	}

	if (scene->p010_supported && skr_tex_is_valid(&scene->texture_p010)) {
		float4x4 world = float4x4_trs((float3){start_x + idx * spacing, 0.0f, 0.0f}, (float4){0, 0, 0, 1}, (float3){1, 1, 1});
		skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->material_p010, &world, sizeof(float4x4), 1);
		idx++;
	}

	if (scene->yuv420p_supported && skr_tex_is_valid(&scene->texture_yuv420p)) {
		float4x4 world = float4x4_trs((float3){start_x + idx * spacing, 0.0f, 0.0f}, (float4){0, 0, 0, 1}, (float3){1, 1, 1});
		skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->material_yuv420p, &world, sizeof(float4x4), 1);
		idx++;
	}
}

static bool _scene_yuv_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_yuv_test_t* scene = (scene_yuv_test_t*)base;

	ImGuiIO* io = igGetIO_Nil();
	if (!io->WantCaptureMouse) {
		if (io->MouseWheel != 0.0f)
			scene->cam_distance -= io->MouseWheel * 0.5f;
		if (io->MouseDown[0])
			scene->cam_distance += io->MouseDelta.y * 0.02f;
		if (scene->cam_distance < 1.0f)  scene->cam_distance = 1.0f;
		if (scene->cam_distance > 20.0f) scene->cam_distance = 20.0f;
	}

	out_camera->position = (float3){0, 0, scene->cam_distance};
	out_camera->target   = (float3){0, 0, 0};
	out_camera->up       = (float3){0, 1, 0};
	return true;
}

static void _scene_yuv_render_ui(scene_t* base) {
	scene_yuv_test_t* scene = (scene_yuv_test_t*)base;

	igText("YUV Texture Formats");
	igSeparator();

	igText("Format Support:");
	igTextColored(scene->nv12_supported    ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  NV12:    %s", scene->nv12_supported    ? "Yes" : "No");
	igTextColored(scene->p010_supported    ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  P010:    %s", scene->p010_supported    ? "Yes" : "No");
	igTextColored(scene->yuv420p_supported ? (ImVec4){0.5f, 1.0f, 0.5f, 1.0f} : (ImVec4){1.0f, 0.5f, 0.5f, 1.0f},
		"  YUV420P: %s", scene->yuv420p_supported ? "Yes" : "No");

	igSeparator();

	igText("Pattern: %dx%d checker", YUV_TEX_WIDTH, YUV_TEX_HEIGHT);
	igText("Block size: %d px", YUV_CHECKER_SIZE);

	igSeparator();

	// Labels for the quads
	igTextColored((ImVec4){0.7f, 0.7f, 0.7f, 1.0f}, "Left to right:");
	igText("  RGBA (reference)");
	if (scene->nv12_supported)    igText("  NV12 (8-bit 4:2:0)");
	if (scene->p010_supported)    igText("  P010 (10-bit 4:2:0)");
	if (scene->yuv420p_supported) igText("  YUV420P (planar)");

	igSeparator();
	igTextColored((ImVec4){0.7f, 0.7f, 0.7f, 1.0f}, "BT.709 narrow range");
	igTextColored((ImVec4){0.7f, 0.7f, 0.7f, 1.0f}, "VkSamplerYcbcrConversion");
}

const scene_vtable_t scene_yuv_test_vtable = {
	.name       = "YUV Formats",
	.create     = _scene_yuv_create,
	.destroy    = _scene_yuv_destroy,
	.update     = _scene_yuv_update,
	.render     = _scene_yuv_render,
	.get_camera = _scene_yuv_get_camera,
	.render_ui  = _scene_yuv_render_ui,
};
