// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith
// Copyright (c) 2026 Qualcomm Technologies, Inc.

#include "tex_psnr.h"
#include "scene_util.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
	skr_shader_t  shader;
	skr_compute_t compute;
	bool          initialized;
} tex_psnr_state_t;

static tex_psnr_state_t g_psnr = {0};

void tex_psnr_init(void) {
	if (g_psnr.initialized) return;

	g_psnr.shader = su_shader_load("shaders/tex_psnr.hlsl.sks", "tex_psnr");
	if (skr_shader_is_valid(&g_psnr.shader))
		skr_compute_create(&g_psnr.shader, (skr_compute_info_t){0}, &g_psnr.compute);
	g_psnr.initialized = true;
}

void tex_psnr_shutdown(void) {
	if (skr_compute_is_valid(&g_psnr.compute)) skr_compute_destroy(&g_psnr.compute);
	if (skr_shader_is_valid (&g_psnr.shader))  skr_shader_destroy (&g_psnr.shader);
	g_psnr = (tex_psnr_state_t){0};
}

double tex_psnr(const skr_tex_t* reference, const skr_tex_t* compressed) {
	if (!g_psnr.initialized || !skr_compute_is_valid(&g_psnr.compute)) return -1.0;
	if (!skr_tex_is_valid(reference) || !skr_tex_is_valid(compressed)) return -1.0;

	skr_vec3i_t size = skr_tex_get_size(reference);
	skr_vec3i_t csz  = skr_tex_get_size(compressed);
	if (size.x != csz.x || size.y != csz.y) return -1.0;

	uint32_t groups_x = ((uint32_t)size.x + 7) / 8;
	uint32_t groups_y = ((uint32_t)size.y + 7) / 8;
	uint32_t groups   = groups_x * groups_y;

	skr_buffer_t partials;
	if (skr_buffer_create(NULL, groups, sizeof(float),
			skr_buffer_type_storage,
			(skr_use_)(skr_use_dynamic | skr_use_compute_readwrite),
			&partials) != skr_err_success) {
		return -1.0;
	}
	skr_buffer_set_name(&partials, "tex_psnr_partials");

	// Same begin/flush/end sandwich as the compression readback path — works
	// both inside and outside an open frame (see tex_compress_gpu.c).
	skr_cmd_begin();

	skr_compute_set_tex   (&g_psnr.compute, "tex_ref",     (skr_tex_t*)reference);
	skr_compute_set_tex   (&g_psnr.compute, "tex_cmp",     (skr_tex_t*)compressed);
	skr_compute_set_buffer(&g_psnr.compute, "partial_sse", &partials);

	uint32_t val;
	val = (uint32_t)size.x; skr_compute_set_param(&g_psnr.compute, "image_width",  sksc_shader_var_uint, 1, &val);
	val = (uint32_t)size.y; skr_compute_set_param(&g_psnr.compute, "image_height", sksc_shader_var_uint, 1, &val);
	val = groups_x;         skr_compute_set_param(&g_psnr.compute, "groups_x",     sksc_shader_var_uint, 1, &val);

	skr_compute_execute(&g_psnr.compute, groups_x, groups_y, 1);

	skr_future_t f = skr_cmd_flush();
	skr_future_wait(&f);
	skr_cmd_end();

	float* sums = (float*)malloc(groups * sizeof(float));
	if (!sums) {
		skr_buffer_destroy(&partials);
		return -1.0;
	}
	skr_buffer_get(&partials, sums, groups * sizeof(float));
	skr_buffer_destroy(&partials);

	double sse = 0.0;
	for (uint32_t i = 0; i < groups; i++)
		sse += sums[i];
	free(sums);

	double mse = sse / ((double)size.x * size.y * 3.0);
	return mse > 0.0 ? 10.0 * log10(255.0 * 255.0 / mse) : INFINITY;
}
