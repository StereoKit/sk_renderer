// Application rendering layer for OpenXR example
// Hosts scenes from the example/ directory with VR stereo rendering

#include "app_xr.h"
#include "openxr_util.h"

#include "scene.h"
#include "scene_util.h"
#include "float_math.h"

#include <sk_renderer.h>
#include <sksc_file.h>
#include <sk_app.h>

#include <stdlib.h>
#include <string.h>
#include <math.h>

///////////////////////////////////////////
// File I/O callback for su_initialize
///////////////////////////////////////////

static bool _xr_file_read(const char* filename, void** out_data, size_t* out_size, void* user_data) {
	(void)user_data;
	if (ska_asset_read(filename, out_data, out_size)) return true;
	if (ska_file_read (filename, out_data, out_size)) return true;
	skr_log(skr_log_critical, "XR: Failed to open file '%s'", filename);
	return false;
}

///////////////////////////////////////////
// Scene registry
///////////////////////////////////////////

static const scene_vtable_t* s_scene_types[] = {
	&scene_meshes_vtable,
	&scene_reaction_diffusion_vtable,
	//&scene_orbital_particles_vtable,
	&scene_impostor_vtable,
	//&scene_array_texture_vtable,
	&scene_3d_texture_vtable,
	&scene_cubemap_vtable,
	&scene_gltf_vtable,
	&scene_shadows_vtable,
	&scene_cloth_vtable,
	&scene_text_vtable,
	//&scene_tex_copy_vtable,
	//&scene_lifetime_stress_vtable, // looks okay?
	//&scene_gaussian_splat_vtable,
	//&scene_tex_compress_vtable,
	&scene_stars_vtable,
	//&scene_yuv_test_vtable,
	&scene_gi_vtable,
};
static const int32_t s_scene_count = sizeof(s_scene_types) / sizeof(s_scene_types[0]);

///////////////////////////////////////////
// Module state
///////////////////////////////////////////

static skr_render_list_t s_render_list;
static scene_t*          s_scene_current = NULL;
static int32_t           s_scene_index   = -1;
static float             s_time          = 0.0f;

///////////////////////////////////////////
// Helper functions
///////////////////////////////////////////

// Create projection matrix from OpenXR asymmetric FOV
static float4x4 xr_projection(XrFovf fov, float clip_near, float clip_far) {
	const float tan_left   = tanf(fov.angleLeft);
	const float tan_right  = tanf(fov.angleRight);
	const float tan_down   = tanf(fov.angleDown);
	const float tan_up     = tanf(fov.angleUp);

	const float tan_width  = tan_right - tan_left;
	const float tan_height = tan_up - tan_down;
	const float range      = clip_far / (clip_near - clip_far);

	float4x4 sk = {{
		2.0f / tan_width, 0.0f, 0.0f, 0.0f,
		0.0f, 2.0f / tan_height, 0.0f, 0.0f,
		(tan_right + tan_left) / tan_width, (tan_up + tan_down) / tan_height, range, -1.0f,
		0.0f, 0.0f, range * clip_near, 0.0f
	}};

	return float4x4_transpose(sk);
}

// Create view matrix from OpenXR pose (inverse of pose transform)
static float4x4 xr_view_matrix(XrPosef pose) {
	float4 q = (float4){pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
	float3 p = (float3){pose.position.x, pose.position.y, pose.position.z};

	float4 q_inv = float4_quat_conjugate(q);
	float3 p_inv = float4_quat_rotate(q_inv, float3_mul_s(p, -1.0f));

	return float4x4_trs(p_inv, q_inv, (float3){1, 1, 1});
}

///////////////////////////////////////////
// Scene management
///////////////////////////////////////////

static void _switch_scene(int32_t new_index) {
	if (new_index < 0 || new_index >= s_scene_count) return;
	if (new_index == s_scene_index) return;

	if (s_scene_current) {
		scene_destroy(s_scene_types[s_scene_index], s_scene_current);
		s_scene_current = NULL;
	}

	s_scene_index   = new_index;
	s_scene_current = scene_create(s_scene_types[new_index]);

	skr_log(skr_log_info, "XR: Switched to scene: %s", scene_get_name(s_scene_types[new_index]));
}

///////////////////////////////////////////
// Public API
///////////////////////////////////////////

void app_xr_init(void) {
	su_initialize(_xr_file_read, NULL);

	skr_render_list_create(&s_render_list);

	_switch_scene(8);
}

void app_xr_shutdown(void) {
	if (s_scene_current) {
		scene_destroy(s_scene_types[s_scene_index], s_scene_current);
		s_scene_current = NULL;
	}

	skr_render_list_destroy(&s_render_list);

	su_shutdown();
}

void app_xr_update(void) {
	// Left hand select: previous scene, right hand select: next scene
	if (xr_input.handSelect[0]) {
		_switch_scene((s_scene_index - 1 + s_scene_count) % s_scene_count);
	}
	if (xr_input.handSelect[1]) {
		_switch_scene((s_scene_index + 1) % s_scene_count);
	}
}

void app_xr_update_predicted(void) {
	// Nothing needed - scenes don't use hand tracking directly
}

void app_xr_render_stereo(skr_tex_t* color_target, skr_tex_t* resolve_target, skr_tex_t* depth_target, const XrView* views, uint32_t view_count, int32_t width, int32_t height) {
	if (!s_scene_current) return;

	float delta_time = 1.0f / 72.0f;
	s_time += delta_time;

	// Update scene
	scene_update(s_scene_types[s_scene_index], s_scene_current, delta_time);

	// Build system buffer with all views
	su_system_buffer_t sys = {0};
	sys.time        = s_time;
	sys.view_count  = view_count;
	sys.screen_size = (float4){(float)width, (float)height, 1.0f / width, 1.0f / height};

	for (uint32_t v = 0; v < view_count && v < SU_MAX_VIEWS; v++) {
		const XrView* view = &views[v];

		float4x4 view_mat = xr_view_matrix(view->pose);
		float4x4 proj_mat = xr_projection(view->fov, 0.05f, 100.0f);

		float4 q       = (float4){view->pose.orientation.x, view->pose.orientation.y, view->pose.orientation.z, view->pose.orientation.w};
		float3 cam_pos = (float3){view->pose.position.x, view->pose.position.y, view->pose.position.z};
		float3 cam_dir = float4_quat_rotate(q, (float3){0, 0, -1});

		sys.view[v]           = view_mat;
		sys.view_inv[v]       = float4x4_invert(view_mat);
		sys.projection[v]     = proj_mat;
		sys.projection_inv[v] = float4x4_invert(proj_mat);
		sys.viewproj[v]       = float4x4_mul(proj_mat, view_mat);
		sys.cam_pos[v]        = (float4){cam_pos.x, cam_pos.y, cam_pos.z, 1};
		sys.cam_dir[v]        = (float4){cam_dir.x, cam_dir.y, cam_dir.z, 0};
	}

	// Let the scene populate the render list
	scene_render(s_scene_types[s_scene_index], s_scene_current, width, height, &s_render_list, &sys);

	// Begin render pass with MSAA resolve
	skr_renderer_begin_pass(
		color_target,
		depth_target,
		resolve_target,
		skr_clear_all,
		(skr_vec4_t){ 0.0f, 0.0f, 0.0f, 0.0f },
		1.0f,
		0
	);

	skr_renderer_set_viewport((skr_rect_t){ 0, 0, (float)width, (float)height });
	skr_renderer_set_scissor((skr_recti_t){ 0, 0, width, height });

	// Draw with multi-view instancing
	skr_renderer_draw(&s_render_list, &sys, sizeof(sys), view_count);

	skr_renderer_end_pass();

	skr_render_list_clear(&s_render_list);
}
