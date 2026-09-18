// Descriptor stress: many distinct materials, each with its own texture, so
// every material run needs its own descriptor state, and every (material,
// mesh) pair draws once per frame. The recreate toggle rebuilds materials as
// it goes, so nothing gets to reuse a set cached at creation.

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

#define STRESS_WARMUP 30

typedef struct {
	scene_t         base;
	skr_shader_t    shader;
	skr_mesh_t*     meshes;
	skr_tex_t*      textures;
	skr_material_t* materials;
	int32_t         mesh_count;
	int32_t         material_count;
	int32_t         next_mesh_count;      // GUI edits these, Rebuild applies them
	int32_t         next_material_count;
	bool            recreate;             // destroy and recreate materials every frame
	int32_t         recreate_per_frame;
	int32_t         recreate_next;
	float           rotation;
	int32_t         frames;
	double          cpu_ms_sum;
	double          gpu_ms_sum;
	int32_t         gpu_frames;
} scene_desc_stress_t;

static void _material_create(scene_desc_stress_t* scene, int32_t i) {
	uint32_t color = 0xFF000000 | ((uint32_t)(i * 2654435761u) >> 8);
	scene->textures[i] = su_tex_create_solid_color(color);
	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.write_mask = skr_write_default,
		.depth_test = skr_compare_less,
	}, &scene->materials[i]);
	skr_material_set_tex(&scene->materials[i], "tex", &scene->textures[i]);
}

static void _content_create(scene_desc_stress_t* scene) {
	scene->material_count = scene->next_material_count;
	scene->mesh_count     = scene->next_mesh_count;
	scene->meshes         = calloc(scene->mesh_count,     sizeof(skr_mesh_t));
	scene->textures       = calloc(scene->material_count, sizeof(skr_tex_t));
	scene->materials      = calloc(scene->material_count, sizeof(skr_material_t));

	skr_vec4_t white[6] = { {1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1} };
	for (int32_t i = 0; i < scene->mesh_count; i++)
		scene->meshes[i] = su_mesh_create_cube(0.5f + 0.001f * i, white);
	for (int32_t i = 0; i < scene->material_count; i++)
		_material_create(scene, i);

	scene->frames     = 0;
	scene->cpu_ms_sum = 0;
	scene->gpu_ms_sum = 0;
	scene->gpu_frames = 0;
}

static void _content_destroy(scene_desc_stress_t* scene) {
	for (int32_t i = 0; i < scene->material_count; i++) {
		skr_material_destroy(&scene->materials[i]);
		skr_tex_destroy(&scene->textures[i]);
	}
	for (int32_t i = 0; i < scene->mesh_count; i++)
		skr_mesh_destroy(&scene->meshes[i]);
	free(scene->materials);
	free(scene->textures);
	free(scene->meshes);
}

static scene_t* _create(void) {
	scene_desc_stress_t* scene = calloc(1, sizeof(scene_desc_stress_t));
	scene->base.size           = sizeof(scene_desc_stress_t);
	scene->next_material_count = 1500;
	scene->next_mesh_count     = 1;
	scene->recreate_per_frame  = 20;
	scene->shader              = su_shader_load("shaders/test.hlsl.sks", "stress_shader");
	_content_create(scene);
	return (scene_t*)scene;
}

static void _destroy(scene_t* base) {
	scene_desc_stress_t* scene = (scene_desc_stress_t*)base;
	int32_t frames = scene->frames - STRESS_WARMUP;
	if (frames > 0)
		su_log(su_log_info, "desc_stress: %d draws/frame over %d frames, avg CPU %.3f ms, avg GPU %.3f ms",
			scene->material_count * scene->mesh_count, frames, scene->cpu_ms_sum / frames, scene->gpu_frames > 0 ? scene->gpu_ms_sum / scene->gpu_frames : 0.0);
	_content_destroy(scene);
	skr_shader_destroy(&scene->shader);
	free(scene);
}

static void _update(scene_t* base, float delta_time) {
	scene_desc_stress_t* scene = (scene_desc_stress_t*)base;
	scene->rotation += delta_time;
	scene->frames++;

	if (scene->recreate) {
		for (int32_t c = 0; c < scene->recreate_per_frame; c++) {
			int32_t i = scene->recreate_next++ % scene->material_count;
			skr_material_destroy(&scene->materials[i]);
			skr_tex_destroy(&scene->textures[i]);
			_material_create(scene, i);
		}
	}

	if (scene->frames <= STRESS_WARMUP) return;
	skr_frame_timing_t t;
	if (!skr_renderer_get_frame_timing(&t)) return;
	scene->cpu_ms_sum += (double)(t.cpu_end_ns - t.cpu_begin_ns - t.cpu_wait_ns) / 1e6;
	if (t.gpu_time_ns > 0) {
		scene->gpu_ms_sum += (double)t.gpu_time_ns / 1e6;
		scene->gpu_frames++;
	}
}

static void _render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_desc_stress_t* scene = (scene_desc_stress_t*)base;
	int32_t total = scene->material_count * scene->mesh_count;
	int32_t side  = 1;
	while (side * side < total) side++;

	for (int32_t m = 0; m < scene->material_count; m++) {
		for (int32_t k = 0; k < scene->mesh_count; k++) {
			int32_t  i = m * scene->mesh_count + k;
			float4x4 transform = float4x4_trs(
				(float3){(i % side - side * 0.5f) * 0.8f, 0.0f, (i / side - side * 0.5f) * 0.8f},
				float4_quat_from_euler((float3){0.0f, scene->rotation + i * 0.1f, 0.0f}),
				(float3){1.0f, 1.0f, 1.0f});
			skr_render_list_add(ref_render_list, &scene->meshes[k], &scene->materials[m], &transform, sizeof(float4x4), 1);
		}
	}
}

static void _render_ui(scene_t* base) {
	scene_desc_stress_t* scene  = (scene_desc_stress_t*)base;
	int32_t              frames = scene->frames - STRESS_WARMUP;

	igTextWrapped("Validation layers cost far more than the draws here. CPU numbers only mean something with enable_validation off in main.c.");
	igSeparator();
	igText("Draws: %d (%d materials x %d meshes)", scene->material_count * scene->mesh_count, scene->material_count, scene->mesh_count);
	if (frames > 0)
		igText("Avg CPU %.3f ms, GPU %.3f ms over %d frames", scene->cpu_ms_sum / frames, scene->gpu_frames > 0 ? scene->gpu_ms_sum / scene->gpu_frames : 0.0, frames);

	igSliderInt("Materials",           &scene->next_material_count, 1, 4000, "%d", 0);
	igSliderInt("Meshes per material", &scene->next_mesh_count,     1, 400,  "%d", 0);
	if (igButton("Rebuild", (ImVec2){0, 0})) {
		_content_destroy(scene);
		_content_create (scene);
	}

	igCheckbox("Recreate materials each frame", &scene->recreate);
	if (igIsItemHovered(0))
		igSetTooltip("Destroys and recreates materials every frame, so their descriptors are allocated fresh instead of reusing the ones cached at creation.");
	if (scene->recreate)
		igSliderInt("Materials per frame", &scene->recreate_per_frame, 1, 200, "%d", 0);
}

const scene_vtable_t scene_desc_stress_vtable = {
	.name      = "Descriptor Stress",
	.create    = _create,
	.destroy   = _destroy,
	.update    = _update,
	.render    = _render,
	.render_ui = _render_ui,
};
