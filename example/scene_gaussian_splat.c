// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Nick Klingensmith

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"
#include "tools/float_math.h"

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

#define MICRO_PLY_IMPL
#include "tools/micro_ply.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

// Unpacked Gaussian splat for PLY loading (intermediate format)
typedef struct {
	float3   position;
	float    opacity;
	float3   sh_dc;
	float    _pad1;
	float3   scale;
	float    _pad2;
	float4   rotation;
	float4   sh_rest[15];
} gaussian_splat_unpacked_t;

// Packed Gaussian splat (124 bytes, must match shader's structured buffer layout)
// Uses half precision and smallest-3 quaternion encoding for ~59% size reduction
typedef struct {
	float    pos_x, pos_y, pos_z;  // 12 bytes - full precision position
	uint32_t rot_packed;           // 4 bytes  - smallest-3 quaternion (10.10.10.2)
	uint32_t scale_xy;             // 4 bytes  - scale.x | scale.y (half floats)
	uint32_t scale_z_opacity;      // 4 bytes  - scale.z | opacity (half floats)
	uint32_t sh_dc_rg;             // 4 bytes  - sh_dc.r | sh_dc.g (half floats)
	uint32_t sh_dc_b_pad;          // 4 bytes  - sh_dc.b | padding (half floats)
	uint32_t sh_rest[23];          // 92 bytes - 45 half floats packed (+ 1 padding)
} gaussian_splat_t;  // Total: 124 bytes

// Half-float conversion (IEEE 754 binary16)
static inline uint16_t f32_to_f16(float f) {
	uint32_t x = *(uint32_t*)&f;
	uint32_t sign = (x >> 16) & 0x8000;
	int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
	uint32_t mant = (x >> 13) & 0x3FF;

	if (exp <= 0) {
		// Denormal or zero
		if (exp < -10) return (uint16_t)sign;  // Too small, flush to zero
		mant = (mant | 0x400) >> (1 - exp);
		return (uint16_t)(sign | mant);
	} else if (exp >= 31) {
		// Infinity or NaN
		return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
	}
	return (uint16_t)(sign | (exp << 10) | mant);
}

// Pack two half floats into one uint32
static inline uint32_t pack_halfs(float a, float b) {
	return (uint32_t)f32_to_f16(a) | ((uint32_t)f32_to_f16(b) << 16);
}

// Pack quaternion using smallest-3 encoding (10.10.10.2 bits)
static inline uint32_t pack_quat_smallest3(float4 q) {
	// Find largest absolute component
	float absQ[4] = { fabsf(q.x), fabsf(q.y), fabsf(q.z), fabsf(q.w) };
	int32_t idx = 0;
	float maxV = absQ[0];
	for (int32_t i = 1; i < 4; i++) {
		if (absQ[i] > maxV) { idx = i; maxV = absQ[i]; }
	}

	// Reorder so largest is last (will be reconstructed)
	float three[3];
	float sign = 1.0f;
	switch (idx) {
		case 0: three[0] = q.y; three[1] = q.z; three[2] = q.w; sign = (q.x >= 0) ? 1.0f : -1.0f; break;
		case 1: three[0] = q.x; three[1] = q.z; three[2] = q.w; sign = (q.y >= 0) ? 1.0f : -1.0f; break;
		case 2: three[0] = q.x; three[1] = q.y; three[2] = q.w; sign = (q.z >= 0) ? 1.0f : -1.0f; break;
		case 3: three[0] = q.x; three[1] = q.y; three[2] = q.z; sign = (q.w >= 0) ? 1.0f : -1.0f; break;
	}

	// Normalize to 0-1 range (components are in -1/sqrt(2) to 1/sqrt(2))
	const float scale = 0.70710678118f;  // 1/sqrt(2)
	uint32_t a = (uint32_t)((three[0] * sign / scale * 0.5f + 0.5f) * 1023.0f + 0.5f) & 0x3FF;
	uint32_t b = (uint32_t)((three[1] * sign / scale * 0.5f + 0.5f) * 1023.0f + 0.5f) & 0x3FF;
	uint32_t c = (uint32_t)((three[2] * sign / scale * 0.5f + 0.5f) * 1023.0f + 0.5f) & 0x3FF;

	return a | (b << 10) | (c << 20) | ((uint32_t)idx << 30);
}

// Radix sort constants (must match shader)
#define RADIX_BINS      256
#define RADIX_PART_SIZE 3840  // 256 threads * 15 keys

// Scene state
typedef struct {
	scene_t         base;

	// Splat data
	uint32_t        splat_count;
	skr_buffer_t    splat_buffer;

	// Radix sort buffers
	skr_buffer_t    sort_keys_a;      // uint keys (float depths converted to sortable uint)
	skr_buffer_t    sort_keys_b;      // uint keys alt (ping-pong)
	skr_buffer_t    sort_payload_a;   // uint payloads (splat indices)
	skr_buffer_t    sort_payload_b;   // uint payloads alt (ping-pong)
	skr_buffer_t    global_hist;      // Global histogram (RADIX * 4 = 1024)
	skr_buffer_t    pass_hist;        // Per-partition histograms (RADIX * thread_blocks)
	skr_buffer_t    sort_indices;     // Final sorted indices for rendering

	// Rendering
	skr_mesh_t      quad_mesh;
	skr_shader_t    render_shader;
	skr_material_t  render_material;

	// GPU Sort compute shaders (GPUSorting library)
	skr_shader_t    sort_init_shader;
	skr_shader_t    sort_upsweep_shader;
	skr_shader_t    sort_scan_shader;
	skr_shader_t    sort_downsweep_shader;
	skr_compute_t   sort_init;
	skr_compute_t   sort_upsweep;
	skr_compute_t   sort_scan;
	skr_compute_t   sort_downsweep;
	uint32_t        thread_blocks;    // Number of partitions for radix sort
	bool            sort_buffers_swapped; // Track ping-pong state

	// UI controls
	float           splat_scale;
	float           opacity_scale;
	int32_t         sh_degree;
	float           max_radius;   // Max splat radius in pixels (0 = unlimited)
	bool            enable_sort;
	bool            flip_y;       // Flip Y axis (for assets with different coordinate conventions)
	char*           ply_path;

	// Sort state (incremental sorting across frames)
	uint32_t        sort_stage;
	uint32_t        sort_step;
	bool            sort_initialized;     // Keys/payloads initialized from depths
	bool            initial_sort_complete;
	uint32_t        radix_byte;           // Current radix byte being sorted (0-3)
	uint32_t        radix_step;           // Step within current byte (0=upsweep, 1=scan, 2=downsweep, 3=swap)
	float3          last_sorted_cam_pos;
	bool            needs_resort;

	// Camera state (arc-ball style)
	float           cam_yaw;
	float           cam_pitch;
	float           cam_distance;
	float3          cam_target;
	float           cam_yaw_vel;
	float           cam_pitch_vel;
	float           cam_distance_vel;
	float3          cam_target_vel;

	float           time;
} scene_gaussian_splat_t;

// Load Gaussian splat data from PLY file
static bool _load_splat_ply(scene_gaussian_splat_t* scene, const char* filename) {
	void*  data = NULL;
	size_t size = 0;

	if (!su_file_read(filename, &data, &size)) {
		su_log(su_log_warning, "gaussian_splat: Failed to read PLY file: %s", filename);
		return false;
	}

	ply_file_t ply = {0};
	if (!ply_read(data, size, &ply)) {
		su_log(su_log_warning, "gaussian_splat: Failed to parse PLY file: %s", filename);
		free(data);
		return false;
	}

	// Find vertex count
	int32_t vertex_count = 0;
	for (int32_t i = 0; i < ply.count; i++) {
		if (strcmp(ply.elements[i].name, PLY_ELEMENT_VERTICES) == 0) {
			vertex_count = ply.elements[i].count;
			break;
		}
	}

	if (vertex_count == 0) {
		su_log(su_log_warning, "gaussian_splat: No vertices found in PLY file");
		ply_free(&ply);
		free(data);
		return false;
	}

	su_log(su_log_info, "gaussian_splat: Loading %d splats from %s (packed: %zu bytes, unpacked: %zu bytes)",
	       vertex_count, filename, sizeof(gaussian_splat_t), sizeof(gaussian_splat_unpacked_t));

	// Allocate unpacked splat data for loading
	gaussian_splat_unpacked_t* splats_unpacked = calloc(vertex_count, sizeof(gaussian_splat_unpacked_t));
	if (!splats_unpacked) {
		ply_free(&ply);
		free(data);
		return false;
	}

	// Define the mapping from PLY properties to unpacked splat structure
	// Build a single map with all properties (14 basic + 45 f_rest) for one ply_convert call
	static float zero = 0.0f;
	static float one  = 1.0f;

	// Property name storage for f_rest_* (need stable pointers for ply_map_t)
	static char f_rest_names[45][16];
	static bool names_initialized = false;
	if (!names_initialized) {
		for (int32_t i = 0; i < 45; i++) {
			snprintf(f_rest_names[i], sizeof(f_rest_names[i]), "f_rest_%d", i);
		}
		names_initialized = true;
	}

	// Build complete property map: 14 basic properties + 45 f_rest properties
	// f_rest reorganization: PLY has f_rest_0..14 (R), f_rest_15..29 (G), f_rest_30..44 (B)
	// We need: sh_rest[i] = {f_rest_i, f_rest_{i+15}, f_rest_{i+30}} for i=0..14
	ply_map_t ply_map[14 + 45] = {
		// Basic properties (14)
		{ PLY_PROP_POSITION_X, ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, position) + 0,  &zero },
		{ PLY_PROP_POSITION_Y, ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, position) + 4,  &zero },
		{ PLY_PROP_POSITION_Z, ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, position) + 8,  &zero },
		{ "opacity",           ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, opacity),       &zero },
		{ "f_dc_0",            ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, sh_dc) + 0,     &zero },
		{ "f_dc_1",            ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, sh_dc) + 4,     &zero },
		{ "f_dc_2",            ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, sh_dc) + 8,     &zero },
		{ "scale_0",           ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, scale) + 0,     &zero },
		{ "scale_1",           ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, scale) + 4,     &zero },
		{ "scale_2",           ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, scale) + 8,     &zero },
		{ "rot_0",             ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, rotation) + 0,  &one  },
		{ "rot_1",             ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, rotation) + 4,  &zero },
		{ "rot_2",             ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, rotation) + 8,  &zero },
		{ "rot_3",             ply_prop_decimal, sizeof(float), offsetof(gaussian_splat_unpacked_t, rotation) + 12, &zero },
	};

	// Add f_rest properties (45 total, reorganized into 15 float4s)
	// sh_rest[i].x = f_rest_i, sh_rest[i].y = f_rest_{i+15}, sh_rest[i].z = f_rest_{i+30}
	for (int32_t i = 0; i < 15; i++) {
		uint16_t base_offset = offsetof(gaussian_splat_unpacked_t, sh_rest) + i * 16;
		ply_map[14 + i*3 + 0] = (ply_map_t){ f_rest_names[i],      ply_prop_decimal, sizeof(float), base_offset + 0, &zero };
		ply_map[14 + i*3 + 1] = (ply_map_t){ f_rest_names[i + 15], ply_prop_decimal, sizeof(float), base_offset + 4, &zero };
		ply_map[14 + i*3 + 2] = (ply_map_t){ f_rest_names[i + 30], ply_prop_decimal, sizeof(float), base_offset + 8, &zero };
	}

	// Single ply_convert call for all properties
	void*   out_data  = NULL;
	int32_t out_count = 0;
	ply_convert(&ply, PLY_ELEMENT_VERTICES, ply_map, 14 + 45,
	            sizeof(gaussian_splat_unpacked_t), &out_data, &out_count);

	if (out_data && out_count > 0) {
		memcpy(splats_unpacked, out_data, (size_t)out_count * sizeof(gaussian_splat_unpacked_t));
		free(out_data);
	}

	// Normalize quaternions and compute bounding box
	float3 bbox_min = { 1e10f,  1e10f,  1e10f};
	float3 bbox_max = {-1e10f, -1e10f, -1e10f};

	for (int32_t i = 0; i < vertex_count; i++) {
		// Apply coordinate transform for COLMAP/3DGS (Y-down, Z-forward) to sk_renderer (Y-up, Z-backward)
		// Position: flip Y and Z
		splats_unpacked[i].position.y = -splats_unpacked[i].position.y;
		splats_unpacked[i].position.z = -splats_unpacked[i].position.z;
		// Quaternion: for Y-Z flip, negate the y and z components
		// Storage: .x=w, .y=x, .z=y, .w=z
		splats_unpacked[i].rotation.z = -splats_unpacked[i].rotation.z;  // Negate y component
		splats_unpacked[i].rotation.w = -splats_unpacked[i].rotation.w;  // Negate z component

		// Normalize quaternion
		float4 q = splats_unpacked[i].rotation;
		float len = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
		if (len > 0.0001f) {
			splats_unpacked[i].rotation.x = q.x / len;
			splats_unpacked[i].rotation.y = q.y / len;
			splats_unpacked[i].rotation.z = q.z / len;
			splats_unpacked[i].rotation.w = q.w / len;
		}

		// Update bounding box
		float3 p = splats_unpacked[i].position;
		if (p.x < bbox_min.x) bbox_min.x = p.x;
		if (p.y < bbox_min.y) bbox_min.y = p.y;
		if (p.z < bbox_min.z) bbox_min.z = p.z;
		if (p.x > bbox_max.x) bbox_max.x = p.x;
		if (p.y > bbox_max.y) bbox_max.y = p.y;
		if (p.z > bbox_max.z) bbox_max.z = p.z;
	}

	// Pack splat data into compressed format
	gaussian_splat_t* splats = calloc(vertex_count, sizeof(gaussian_splat_t));
	if (!splats) {
		free(splats_unpacked);
		ply_free(&ply);
		free(data);
		return false;
	}

	// SH_C0 constant for preprocessing (matches shader and 3DGS reference)
	const float SH_C0 = 0.28209479177387814f;

	for (int32_t i = 0; i < vertex_count; i++) {
		gaussian_splat_unpacked_t* src = &splats_unpacked[i];
		gaussian_splat_t* dst = &splats[i];

		// Position: full precision
		dst->pos_x = src->position.x;
		dst->pos_y = src->position.y;
		dst->pos_z = src->position.z;

		// Rotation: smallest-3 quaternion (PLY stores as w,x,y,z)
		dst->rot_packed = pack_quat_smallest3(src->rotation);

		// Scale + opacity: half precision
		dst->scale_xy       = pack_halfs(src->scale.x, src->scale.y);
		dst->scale_z_opacity = pack_halfs(src->scale.z, src->opacity);

		// SH DC: preprocess like Aras does (color = f_dc * SH_C0 + 0.5)
		// Then convert from sRGB to linear space for proper rendering
		float dc_r = src->sh_dc.x * SH_C0 + 0.5f;
		float dc_g = src->sh_dc.y * SH_C0 + 0.5f;
		float dc_b = src->sh_dc.z * SH_C0 + 0.5f;
		// Clamp to [0,1] before gamma to avoid NaN from negative values
		dc_r = fmaxf(0.0f, fminf(1.0f, dc_r));
		dc_g = fmaxf(0.0f, fminf(1.0f, dc_g));
		dc_b = fmaxf(0.0f, fminf(1.0f, dc_b));
		// Convert sRGB to linear (3DGS colors are in sRGB/perceptual space)
		dc_r = powf(dc_r, 2.2f);
		dc_g = powf(dc_g, 2.2f);
		dc_b = powf(dc_b, 2.2f);
		dst->sh_dc_rg    = pack_halfs(dc_r, dc_g);
		dst->sh_dc_b_pad = pack_halfs(dc_b, 0.0f);

		// SH rest: 45 floats -> 23 uint32s (45 halfs, padded to 46)
		float sh_flat[46];
		for (int32_t j = 0; j < 15; j++) {
			sh_flat[j * 3 + 0] = src->sh_rest[j].x;
			sh_flat[j * 3 + 1] = src->sh_rest[j].y;
			sh_flat[j * 3 + 2] = src->sh_rest[j].z;
		}
		sh_flat[45] = 0.0f;  // Padding

		for (int32_t j = 0; j < 23; j++) {
			dst->sh_rest[j] = pack_halfs(sh_flat[j * 2], sh_flat[j * 2 + 1]);
		}
	}

	free(splats_unpacked);

	// Set camera target to center of bounding box
	scene->cam_target = (float3){0,0,0};/* (float3){
		(bbox_min.x + bbox_max.x) * 0.5f,
		(bbox_min.y + bbox_max.y) * 0.5f,
		(bbox_min.z + bbox_max.z) * 0.5f
	};*/

	// Set camera distance based on bounding box size
	float3 bbox_size = {
		bbox_max.x - bbox_min.x,
		bbox_max.y - bbox_min.y,
		bbox_max.z - bbox_min.z
	};
	float max_dim = fmaxf(fmaxf(bbox_size.x, bbox_size.y), bbox_size.z);

	su_log(su_log_info, "gaussian_splat: Bounds [%.2f,%.2f,%.2f] - [%.2f,%.2f,%.2f], size %.2f",
	       bbox_min.x, bbox_min.y, bbox_min.z,
	       bbox_max.x, bbox_max.y, bbox_max.z, max_dim);

	// Create GPU buffers
	scene->splat_count = vertex_count;

	skr_buffer_create(splats, vertex_count, sizeof(gaussian_splat_t),
	                  skr_buffer_type_storage, skr_use_compute_read, &scene->splat_buffer);
	skr_buffer_set_name(&scene->splat_buffer, "gaussian_splat_data");

	// Calculate thread blocks for radix sort
	scene->thread_blocks = (vertex_count + RADIX_PART_SIZE - 1) / RADIX_PART_SIZE;
	if (scene->thread_blocks < 1) scene->thread_blocks = 1;

	// Create radix sort buffers
	uint32_t* zeros = calloc(vertex_count, sizeof(uint32_t));

	// Keys A/B (uint representation of depths)
	skr_buffer_create(zeros, vertex_count, sizeof(uint32_t),
	                  skr_buffer_type_storage, skr_use_compute_readwrite, &scene->sort_keys_a);
	skr_buffer_set_name(&scene->sort_keys_a, "radix_keys_a");

	skr_buffer_create(zeros, vertex_count, sizeof(uint32_t),
	                  skr_buffer_type_storage, skr_use_compute_readwrite, &scene->sort_keys_b);
	skr_buffer_set_name(&scene->sort_keys_b, "radix_keys_b");

	// Payloads A/B (splat indices)
	skr_buffer_create(zeros, vertex_count, sizeof(uint32_t),
	                  skr_buffer_type_storage, skr_use_compute_readwrite, &scene->sort_payload_a);
	skr_buffer_set_name(&scene->sort_payload_a, "radix_payload_a");

	skr_buffer_create(zeros, vertex_count, sizeof(uint32_t),
	                  skr_buffer_type_storage, skr_use_compute_readwrite, &scene->sort_payload_b);
	skr_buffer_set_name(&scene->sort_payload_b, "radix_payload_b");

	// Global histogram (RADIX * 4 = 1024 entries for 4 radix passes)
	uint32_t* global_hist = calloc(RADIX_BINS * 4, sizeof(uint32_t));
	skr_buffer_create(global_hist, RADIX_BINS * 4, sizeof(uint32_t),
	                  skr_buffer_type_storage, skr_use_compute_readwrite, &scene->global_hist);
	skr_buffer_set_name(&scene->global_hist, "radix_global_hist");
	free(global_hist);

	// Per-partition histograms (RADIX * thread_blocks)
	uint32_t pass_hist_size = RADIX_BINS * scene->thread_blocks;
	uint32_t* pass_hist = calloc(pass_hist_size, sizeof(uint32_t));
	skr_buffer_create(pass_hist, pass_hist_size, sizeof(uint32_t),
	                  skr_buffer_type_storage, skr_use_compute_readwrite, &scene->pass_hist);
	skr_buffer_set_name(&scene->pass_hist, "radix_pass_hist");
	free(pass_hist);

	// Final sorted indices for rendering
	skr_buffer_create(zeros, vertex_count, sizeof(uint32_t),
	                  skr_buffer_type_storage, skr_use_compute_readwrite, &scene->sort_indices);
	skr_buffer_set_name(&scene->sort_indices, "sort_indices_render");

	free(zeros);
	free(splats);

	ply_free(&ply);
	free(data);

	return true;
}

static scene_t* _scene_gaussian_splat_create(void) {
	scene_gaussian_splat_t* scene = calloc(1, sizeof(scene_gaussian_splat_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_gaussian_splat_t);

	// Default parameters
	scene->splat_scale   = 1.0f;
	scene->opacity_scale = 1.0f;
	scene->sh_degree     = 3;
	scene->max_radius    = 256.0f;  // Cap splat size to prevent massive overdraw
	scene->enable_sort   = true;
	scene->time          = 0.0f;

	// Camera defaults (will be updated when PLY loads)
	scene->cam_yaw      = 0.0f;
	scene->cam_pitch    = 0.0f;
	scene->cam_distance = 5.0f;
	scene->cam_target   = (float3){0, 0, 0};

	
	// Load PLY file
	//scene->ply_path = strdup("/home/koujaku/Downloads/Temple.ply"); // https://superspl.at/view?id=4653e2b9
	scene->ply_path = strdup("test_cube.ply");
	if (!_load_splat_ply(scene, scene->ply_path)) {
		su_log(su_log_warning, "gaussian_splat: Failed to load default PLY, scene will be empty");
	}

	// Create a unit quad mesh for rendering splats
	// Each splat is rendered as a screen-aligned quad
	su_vertex_t quad_verts[] = {
		{ .position = {-1, -1, 0}, .normal = {0, 0, 1}, .uv = {0, 0}, .color = 0xFFFFFFFF },
		{ .position = { 1, -1, 0}, .normal = {0, 0, 1}, .uv = {1, 0}, .color = 0xFFFFFFFF },
		{ .position = { 1,  1, 0}, .normal = {0, 0, 1}, .uv = {1, 1}, .color = 0xFFFFFFFF },
		{ .position = {-1,  1, 0}, .normal = {0, 0, 1}, .uv = {0, 1}, .color = 0xFFFFFFFF },
	};
	uint16_t quad_indices[] = { 0, 1, 2, 0, 2, 3 };
	skr_mesh_create(&su_vertex_type, skr_index_fmt_u16, quad_verts, 4, quad_indices, 6, &scene->quad_mesh);
	skr_mesh_set_name(&scene->quad_mesh, "gaussian_quad");

	// Load render shader
	scene->render_shader = su_shader_load("shaders/gaussian_splat.hlsl.sks", "gaussian_splat");
	if (!skr_shader_is_valid(&scene->render_shader)) {
		su_log(su_log_warning, "gaussian_splat: Failed to load render shader");
	}

	// Create material with alpha blending
	// Using Aras's front-to-back "under" operator: Blend OneMinusDstAlpha One
	skr_blend_state_t blend_front_to_back = {
		.src_color_factor = skr_blend_one_minus_dst_alpha,
		.dst_color_factor = skr_blend_one,
		.color_op         = skr_blend_op_add,
		.src_alpha_factor = skr_blend_one_minus_dst_alpha,
		.dst_alpha_factor = skr_blend_one,
		.alpha_op         = skr_blend_op_add,
	};
	skr_material_create((skr_material_info_t){
		.shader       = &scene->render_shader,
		.cull         = skr_cull_none,
		.write_mask   = skr_write_rgba,  // No depth write for proper alpha blending
		.depth_test   = skr_compare_less_or_eq,
		.blend_state  = blend_front_to_back,
		.queue_offset = 100,  // Render after opaque objects
	}, &scene->render_material);

	// Load GPU sort compute shaders (GPUSorting library)
	scene->sort_init_shader     = su_shader_load("shaders/gpu_sort_init.hlsl.sks", "gpu_sort_init");
	scene->sort_upsweep_shader  = su_shader_load("shaders/gpu_sort_upsweep.hlsl.sks", "gpu_sort_upsweep");
	scene->sort_scan_shader     = su_shader_load("shaders/gpu_sort_scan.hlsl.sks", "gpu_sort_scan");
	scene->sort_downsweep_shader = su_shader_load("shaders/gpu_sort_downsweep.hlsl.sks", "gpu_sort_downsweep");

	bool sort_shaders_valid =
		skr_shader_is_valid(&scene->sort_init_shader) &&
		skr_shader_is_valid(&scene->sort_upsweep_shader) &&
		skr_shader_is_valid(&scene->sort_scan_shader) &&
		skr_shader_is_valid(&scene->sort_downsweep_shader);

	if (sort_shaders_valid) {
		// Create compute pipelines
		skr_compute_create(&scene->sort_init_shader, &scene->sort_init);
		skr_compute_create(&scene->sort_upsweep_shader, &scene->sort_upsweep);
		skr_compute_create(&scene->sort_scan_shader, &scene->sort_scan);
		skr_compute_create(&scene->sort_downsweep_shader, &scene->sort_downsweep);

		// Bind all buffers to all shaders (even if not all used, they're declared)
		// Init kernel buffers
		skr_compute_set_buffer(&scene->sort_init, "splats",        &scene->splat_buffer);
		skr_compute_set_buffer(&scene->sort_init, "b_sort",        &scene->sort_keys_a);
		skr_compute_set_buffer(&scene->sort_init, "b_alt",         &scene->sort_keys_b);
		skr_compute_set_buffer(&scene->sort_init, "b_sortPayload", &scene->sort_payload_a);
		skr_compute_set_buffer(&scene->sort_init, "b_altPayload",  &scene->sort_payload_b);
		skr_compute_set_buffer(&scene->sort_init, "b_globalHist",  &scene->global_hist);
		skr_compute_set_buffer(&scene->sort_init, "b_passHist",    &scene->pass_hist);

		// Upsweep kernel buffers (b_sort set dynamically for ping-pong)
		skr_compute_set_buffer(&scene->sort_upsweep, "b_sort",        &scene->sort_keys_a);
		skr_compute_set_buffer(&scene->sort_upsweep, "b_alt",         &scene->sort_keys_b);
		skr_compute_set_buffer(&scene->sort_upsweep, "b_sortPayload", &scene->sort_payload_a);
		skr_compute_set_buffer(&scene->sort_upsweep, "b_altPayload",  &scene->sort_payload_b);
		skr_compute_set_buffer(&scene->sort_upsweep, "b_passHist",    &scene->pass_hist);
		skr_compute_set_buffer(&scene->sort_upsweep, "b_globalHist",  &scene->global_hist);

		// Scan kernel buffers
		skr_compute_set_buffer(&scene->sort_scan, "b_sort",        &scene->sort_keys_a);
		skr_compute_set_buffer(&scene->sort_scan, "b_alt",         &scene->sort_keys_b);
		skr_compute_set_buffer(&scene->sort_scan, "b_sortPayload", &scene->sort_payload_a);
		skr_compute_set_buffer(&scene->sort_scan, "b_altPayload",  &scene->sort_payload_b);
		skr_compute_set_buffer(&scene->sort_scan, "b_passHist",    &scene->pass_hist);
		skr_compute_set_buffer(&scene->sort_scan, "b_globalHist",  &scene->global_hist);

		// Downsweep kernel buffers (set dynamically for ping-pong)
		skr_compute_set_buffer(&scene->sort_downsweep, "b_sort",        &scene->sort_keys_a);
		skr_compute_set_buffer(&scene->sort_downsweep, "b_alt",         &scene->sort_keys_b);
		skr_compute_set_buffer(&scene->sort_downsweep, "b_sortPayload", &scene->sort_payload_a);
		skr_compute_set_buffer(&scene->sort_downsweep, "b_altPayload",  &scene->sort_payload_b);
		skr_compute_set_buffer(&scene->sort_downsweep, "b_passHist",    &scene->pass_hist);
		skr_compute_set_buffer(&scene->sort_downsweep, "b_globalHist",  &scene->global_hist);

		su_log(su_log_info, "gaussian_splat: GPU sort shaders loaded successfully");
	} else {
		su_log(su_log_warning, "gaussian_splat: Failed to load GPU sort shaders!");
	}

	su_log(su_log_info, "gaussian_splat: Scene created with %u splats", scene->splat_count);
	su_log(su_log_info, "gaussian_splat: Camera at distance %.2f, target (%.2f, %.2f, %.2f)",
	       scene->cam_distance, scene->cam_target.x, scene->cam_target.y, scene->cam_target.z);

	return (scene_t*)scene;
}

// Forward declarations
static void _run_sort_compute(scene_gaussian_splat_t* scene);

static void _scene_gaussian_splat_destroy(scene_t* base) {
	scene_gaussian_splat_t* scene = (scene_gaussian_splat_t*)base;

	// Splat data
	skr_buffer_destroy(&scene->splat_buffer);

	// Radix sort buffers
	skr_buffer_destroy(&scene->sort_keys_a);
	skr_buffer_destroy(&scene->sort_keys_b);
	skr_buffer_destroy(&scene->sort_payload_a);
	skr_buffer_destroy(&scene->sort_payload_b);
	skr_buffer_destroy(&scene->global_hist);
	skr_buffer_destroy(&scene->pass_hist);
	skr_buffer_destroy(&scene->sort_indices);

	// Rendering
	skr_mesh_destroy(&scene->quad_mesh);
	skr_material_destroy(&scene->render_material);
	skr_shader_destroy(&scene->render_shader);

	// GPU Sort compute
	skr_compute_destroy(&scene->sort_init);
	skr_compute_destroy(&scene->sort_upsweep);
	skr_compute_destroy(&scene->sort_scan);
	skr_compute_destroy(&scene->sort_downsweep);
	skr_shader_destroy(&scene->sort_init_shader);
	skr_shader_destroy(&scene->sort_upsweep_shader);
	skr_shader_destroy(&scene->sort_scan_shader);
	skr_shader_destroy(&scene->sort_downsweep_shader);

	if (scene->ply_path) free(scene->ply_path);

	free(scene);
}

static void _scene_gaussian_splat_update(scene_t* base, float delta_time) {
	scene_gaussian_splat_t* scene = (scene_gaussian_splat_t*)base;
	scene->time += delta_time;

	// Run compute shader for sorting (must be outside render pass)
	_run_sort_compute(scene);

	// Camera control - hybrid orbit + fly camera
	const float rotate_sensitivity = 0.003f;
	const float zoom_sensitivity   = 0.2f;
	const float velocity_damping   = 0.0001f;
	const float pitch_limit        = 1.5f;
	const float min_distance       = 0.1f;
	const float max_distance       = 100.0f;
	const float move_speed         = 5.0f;  // Units per second

	ImGuiIO* io = igGetIO();

	// Compute camera vectors for movement
	float cos_pitch = cosf(scene->cam_pitch);
	float sin_pitch = sinf(scene->cam_pitch);
	float cos_yaw   = cosf(scene->cam_yaw);
	float sin_yaw   = sinf(scene->cam_yaw);

	// Forward is from camera toward target (opposite of orbit direction)
	float3 forward = { -cos_pitch * sin_yaw, -sin_pitch, -cos_pitch * cos_yaw };
	float3 right   = {  cos_yaw, 0.0f, -sin_yaw };
	// Camera up = cross(right, forward)
	float3 up      = { -sin_yaw * sin_pitch, cos_pitch, -cos_yaw * sin_pitch };

	// WASD + QE fly movement (always active when not typing in UI)
	if (!io->WantCaptureKeyboard) {
		float move_delta = move_speed * delta_time;

		// Check for shift to move faster
		if (igIsKeyDown_Nil(ImGuiKey_LeftShift) || igIsKeyDown_Nil(ImGuiKey_RightShift)) {
			move_delta *= 3.0f;
		}

		if (igIsKeyDown_Nil(ImGuiKey_W)) {
			scene->cam_target.x += forward.x * move_delta;
			scene->cam_target.y += forward.y * move_delta;
			scene->cam_target.z += forward.z * move_delta;
		}
		if (igIsKeyDown_Nil(ImGuiKey_S)) {
			scene->cam_target.x -= forward.x * move_delta;
			scene->cam_target.y -= forward.y * move_delta;
			scene->cam_target.z -= forward.z * move_delta;
		}
		if (igIsKeyDown_Nil(ImGuiKey_A)) {
			scene->cam_target.x -= right.x * move_delta;
			scene->cam_target.z -= right.z * move_delta;
		}
		if (igIsKeyDown_Nil(ImGuiKey_D)) {
			scene->cam_target.x += right.x * move_delta;
			scene->cam_target.z += right.z * move_delta;
		}
		if (igIsKeyDown_Nil(ImGuiKey_E)) {
			scene->cam_target.x += up.x * move_delta;
			scene->cam_target.y += up.y * move_delta;
			scene->cam_target.z += up.z * move_delta;
		}
		if (igIsKeyDown_Nil(ImGuiKey_Q)) {
			scene->cam_target.x -= up.x * move_delta;
			scene->cam_target.y -= up.y * move_delta;
			scene->cam_target.z -= up.z * move_delta;
		}
	}

	if (!io->WantCaptureMouse) {
		// Left mouse: arc rotate (orbit around target)
		if (io->MouseDown[0]) {
			scene->cam_yaw_vel   -= io->MouseDelta.x * rotate_sensitivity;
			scene->cam_pitch_vel += io->MouseDelta.y * rotate_sensitivity;
		}

		// Right mouse: mouse look (same rotation, feels like FPS when combined with WASD)
		if (io->MouseDown[1]) {
			scene->cam_yaw_vel   -= io->MouseDelta.x * rotate_sensitivity;
			scene->cam_pitch_vel += io->MouseDelta.y * rotate_sensitivity;
		}

		// Scroll wheel: zoom
		if (io->MouseWheel != 0.0f) {
			scene->cam_distance_vel -= io->MouseWheel * zoom_sensitivity * scene->cam_distance * 0.1f;
		}
	}

	// Apply velocities
	scene->cam_yaw      += scene->cam_yaw_vel;
	scene->cam_pitch    += scene->cam_pitch_vel;
	scene->cam_distance += scene->cam_distance_vel;
	scene->cam_target.x += scene->cam_target_vel.x;
	scene->cam_target.y += scene->cam_target_vel.y;
	scene->cam_target.z += scene->cam_target_vel.z;

	// Clamp
	if (scene->cam_pitch >  pitch_limit) scene->cam_pitch =  pitch_limit;
	if (scene->cam_pitch < -pitch_limit) scene->cam_pitch = -pitch_limit;
	if (scene->cam_distance < min_distance) scene->cam_distance = min_distance;
	if (scene->cam_distance > max_distance) scene->cam_distance = max_distance;

	// Damping
	float damping = powf(velocity_damping, delta_time);
	scene->cam_yaw_vel      *= damping;
	scene->cam_pitch_vel    *= damping;
	scene->cam_distance_vel *= damping;
	scene->cam_target_vel.x *= damping;
	scene->cam_target_vel.y *= damping;
	scene->cam_target_vel.z *= damping;
}

// Check if two float3s are approximately equal
static bool _float3_approx_equal(float3 a, float3 b, float epsilon) {
	float dx = a.x - b.x;
	float dy = a.y - b.y;
	float dz = a.z - b.z;
	return (dx > -epsilon && dx < epsilon) &&
	       (dy > -epsilon && dy < epsilon) &&
	       (dz > -epsilon && dz < epsilon);
}

// Run GPU radix sort using GPUSorting library (Thomas Smith)
// 8-bit LSD radix sort with 4 passes (one per byte)
// Uses wave intrinsics for correct, stable sorting
static void _run_gpu_sort(scene_gaussian_splat_t* scene, float3 cam_pos) {
	if (!skr_compute_is_valid(&scene->sort_init) ||
	    !skr_compute_is_valid(&scene->sort_upsweep) ||
	    !skr_compute_is_valid(&scene->sort_scan) ||
	    !skr_compute_is_valid(&scene->sort_downsweep)) {
		return;
	}

	// Check if camera moved
	bool camera_moved = !_float3_approx_equal(cam_pos, scene->last_sorted_cam_pos, 0.0001f);
	if (camera_moved) {
		scene->needs_resort = true;
	}

	// Skip if no resort needed and already sorted
	if (!scene->needs_resort && scene->initial_sort_complete) {
		return;
	}

	uint32_t dispatch_splats = (scene->splat_count + 255) / 256;

	// === INIT PHASE ===
	// Pass 0: Clear global histogram
	skr_compute_set_param(&scene->sort_init, "e_numKeys",     sksc_shader_var_uint,  1, &scene->splat_count);
	skr_compute_set_param(&scene->sort_init, "e_threadBlocks", sksc_shader_var_uint, 1, &scene->thread_blocks);
	skr_compute_set_param(&scene->sort_init, "e_initPass",    sksc_shader_var_uint,  1, &(uint32_t){0});
	skr_compute_execute(&scene->sort_init, 4, 1, 1);  // 4 workgroups * 256 = 1024 threads

	// Pass 1: Compute depths and initialize keys/payloads
	skr_compute_set_param(&scene->sort_init, "e_camPos",   sksc_shader_var_float, 3, &cam_pos);
	skr_compute_set_param(&scene->sort_init, "e_initPass", sksc_shader_var_uint,  1, &(uint32_t){1});
	skr_compute_execute(&scene->sort_init, dispatch_splats, 1, 1);

	// Set common parameters for sorting kernels
	skr_compute_set_param(&scene->sort_upsweep,   "e_numKeys",      sksc_shader_var_uint, 1, &scene->splat_count);
	skr_compute_set_param(&scene->sort_upsweep,   "e_threadBlocks", sksc_shader_var_uint, 1, &scene->thread_blocks);
	skr_compute_set_param(&scene->sort_scan,      "e_numKeys",      sksc_shader_var_uint, 1, &scene->splat_count);
	skr_compute_set_param(&scene->sort_scan,      "e_threadBlocks", sksc_shader_var_uint, 1, &scene->thread_blocks);
	skr_compute_set_param(&scene->sort_downsweep, "e_numKeys",      sksc_shader_var_uint, 1, &scene->splat_count);
	skr_compute_set_param(&scene->sort_downsweep, "e_threadBlocks", sksc_shader_var_uint, 1, &scene->thread_blocks);

	// === SORT PHASE: 4 radix passes (one per byte) ===
	// After 4 passes (even number), result is back in original buffers
	bool is_even = true;
	for (uint32_t radix_shift = 0; radix_shift < 32; radix_shift += 8) {
		// Set radix shift for this pass
		skr_compute_set_param(&scene->sort_upsweep,   "e_radixShift", sksc_shader_var_uint, 1, &radix_shift);
		skr_compute_set_param(&scene->sort_scan,      "e_radixShift", sksc_shader_var_uint, 1, &radix_shift);
		skr_compute_set_param(&scene->sort_downsweep, "e_radixShift", sksc_shader_var_uint, 1, &radix_shift);

		// Set buffer bindings for ping-pong
		if (is_even) {
			skr_compute_set_buffer(&scene->sort_upsweep,   "b_sort",        &scene->sort_keys_a);
			skr_compute_set_buffer(&scene->sort_downsweep, "b_sort",        &scene->sort_keys_a);
			skr_compute_set_buffer(&scene->sort_downsweep, "b_alt",         &scene->sort_keys_b);
			skr_compute_set_buffer(&scene->sort_downsweep, "b_sortPayload", &scene->sort_payload_a);
			skr_compute_set_buffer(&scene->sort_downsweep, "b_altPayload",  &scene->sort_payload_b);
		} else {
			skr_compute_set_buffer(&scene->sort_upsweep,   "b_sort",        &scene->sort_keys_b);
			skr_compute_set_buffer(&scene->sort_downsweep, "b_sort",        &scene->sort_keys_b);
			skr_compute_set_buffer(&scene->sort_downsweep, "b_alt",         &scene->sort_keys_a);
			skr_compute_set_buffer(&scene->sort_downsweep, "b_sortPayload", &scene->sort_payload_b);
			skr_compute_set_buffer(&scene->sort_downsweep, "b_altPayload",  &scene->sort_payload_a);
		}

		// Upsweep: build per-partition histograms
		skr_compute_execute(&scene->sort_upsweep, scene->thread_blocks, 1, 1);

		// Global sum: convert globalHist from counts to exclusive prefix sums
		// Uses e_radixShift to determine which 256-entry section to process
		skr_compute_set_param(&scene->sort_init, "e_initPass", sksc_shader_var_uint, 1, &(uint32_t){2});
		skr_compute_set_param(&scene->sort_init, "e_radixShift", sksc_shader_var_uint, 1, &radix_shift);
		skr_compute_execute(&scene->sort_init, 1, 1, 1);

		// Scan: exclusive prefix sum over partition histograms (256 workgroups, one per digit)
		skr_compute_execute(&scene->sort_scan, 256, 1, 1);

		// Downsweep: rank keys and scatter to sorted positions
		skr_compute_execute(&scene->sort_downsweep, scene->thread_blocks, 1, 1);

		is_even = !is_even;
	}

	// After 4 passes, sorted payloads are in sort_payload_a
	// Copy to sort_indices for rendering
	// (We could optimize this away by having the renderer read from sort_payload_a directly)
	// For now, we use sort_payload_a as the render buffer

	scene->initial_sort_complete = true;
	scene->last_sorted_cam_pos = cam_pos;
	scene->needs_resort = false;
	scene->sort_buffers_swapped = false;
}

// Run sort compute shader - called from update to ensure compute runs before render
static void _run_sort_compute(scene_gaussian_splat_t* scene) {
	if (scene->splat_count == 0) return;

	// Compute camera position for distance² sorting
	float cos_pitch = cosf(scene->cam_pitch);
	float sin_pitch = sinf(scene->cam_pitch);
	float cos_yaw   = cosf(scene->cam_yaw);
	float sin_yaw   = sinf(scene->cam_yaw);

	float3 cam_pos = {
		scene->cam_target.x + scene->cam_distance * cos_pitch * sin_yaw,
		scene->cam_target.y + scene->cam_distance * sin_pitch,
		scene->cam_target.z + scene->cam_distance * cos_pitch * cos_yaw
	};

	_run_gpu_sort(scene, cam_pos);
}

static void _scene_gaussian_splat_render(scene_t* base, int32_t width, int32_t height,
                                          skr_render_list_t* ref_render_list,
                                          su_system_buffer_t* ref_system_buffer) {
	scene_gaussian_splat_t* scene = (scene_gaussian_splat_t*)base;

	if (scene->splat_count == 0) return;

	// Set shader parameters
	float2 screen_size = { (float)width, (float)height };
	skr_material_set_param(&scene->render_material, "splat_scale",   sksc_shader_var_float, 1, &scene->splat_scale);
	skr_material_set_param(&scene->render_material, "opacity_scale", sksc_shader_var_float, 1, &scene->opacity_scale);
	skr_material_set_param(&scene->render_material, "splat_count",   sksc_shader_var_uint,  1, &scene->splat_count);
	skr_material_set_param(&scene->render_material, "sh_degree",     sksc_shader_var_float, 1, &(float){(float)scene->sh_degree});
	skr_material_set_param(&scene->render_material, "screen_size",   sksc_shader_var_float, 2, &screen_size);
	skr_material_set_param(&scene->render_material, "max_radius",    sksc_shader_var_float, 1, &scene->max_radius);

	// Bind buffers
	// After 4 radix passes (even number), sorted indices are in sort_payload_a
	skr_material_set_buffer(&scene->render_material, "splats", &scene->splat_buffer);
	skr_material_set_buffer(&scene->render_material, "sort_indices", &scene->sort_payload_a);

	// Render all splats as instanced quads
	skr_render_list_add(ref_render_list, &scene->quad_mesh, &scene->render_material, NULL, 0, scene->splat_count);
}

static bool _scene_gaussian_splat_get_camera(scene_t* base, scene_camera_t* out_camera) {
	scene_gaussian_splat_t* scene = (scene_gaussian_splat_t*)base;

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
	out_camera->up     = (float3){0.0f, 1.0f, 0.0f};

	return true;
}

static void _scene_gaussian_splat_render_ui(scene_t* base) {
	scene_gaussian_splat_t* scene = (scene_gaussian_splat_t*)base;

	igText("Gaussian Splatting");
	igSeparator();

	igText("Splats: %u (partitions: %u)", scene->splat_count, scene->thread_blocks);
	igSliderFloat("Splat Scale", &scene->splat_scale, 0.1f, 5.0f, "%.2f", 0);
	igSliderFloat("Opacity", &scene->opacity_scale, 0.1f, 2.0f, "%.2f", 0);
	igSliderFloat("Max Radius", &scene->max_radius, 0.0f, 1024.0f, "%.0f px", 0);
	igSliderInt("SH Degree", &scene->sh_degree, 0, 3, "%d", 0);

	igSeparator();
	igText("PLY: %s", scene->ply_path ? scene->ply_path : "(none)");
	if (su_file_dialog_supported()) {
		if (igButton("Load PLY...", (ImVec2){0, 0})) {
			char* path = su_file_dialog_open("Select Gaussian Splat", "PLY Files", "ply");
			if (path) {
				// Destroy old buffers
				skr_buffer_destroy(&scene->splat_buffer);
				skr_buffer_destroy(&scene->sort_keys_a);
				skr_buffer_destroy(&scene->sort_keys_b);
				skr_buffer_destroy(&scene->sort_payload_a);
				skr_buffer_destroy(&scene->sort_payload_b);
				skr_buffer_destroy(&scene->global_hist);
				skr_buffer_destroy(&scene->pass_hist);
				skr_buffer_destroy(&scene->sort_indices);

				scene->splat_count = 0;
				scene->sort_initialized = false;
				scene->initial_sort_complete = false;
				scene->needs_resort = true;

				// Load new file
				if (_load_splat_ply(scene, path)) {
					if (scene->ply_path) free(scene->ply_path);
					scene->ply_path = path;

					// Update compute bindings for all 4 sort kernels (bind all buffers to all)
					if (skr_compute_is_valid(&scene->sort_init)) {
						skr_compute_set_buffer(&scene->sort_init, "splats",        &scene->splat_buffer);
						skr_compute_set_buffer(&scene->sort_init, "b_sort",        &scene->sort_keys_a);
						skr_compute_set_buffer(&scene->sort_init, "b_alt",         &scene->sort_keys_b);
						skr_compute_set_buffer(&scene->sort_init, "b_sortPayload", &scene->sort_payload_a);
						skr_compute_set_buffer(&scene->sort_init, "b_altPayload",  &scene->sort_payload_b);
						skr_compute_set_buffer(&scene->sort_init, "b_globalHist",  &scene->global_hist);
						skr_compute_set_buffer(&scene->sort_init, "b_passHist",    &scene->pass_hist);
					}
					if (skr_compute_is_valid(&scene->sort_upsweep)) {
						skr_compute_set_buffer(&scene->sort_upsweep, "b_sort",        &scene->sort_keys_a);
						skr_compute_set_buffer(&scene->sort_upsweep, "b_alt",         &scene->sort_keys_b);
						skr_compute_set_buffer(&scene->sort_upsweep, "b_sortPayload", &scene->sort_payload_a);
						skr_compute_set_buffer(&scene->sort_upsweep, "b_altPayload",  &scene->sort_payload_b);
						skr_compute_set_buffer(&scene->sort_upsweep, "b_passHist",    &scene->pass_hist);
						skr_compute_set_buffer(&scene->sort_upsweep, "b_globalHist",  &scene->global_hist);
					}
					if (skr_compute_is_valid(&scene->sort_scan)) {
						skr_compute_set_buffer(&scene->sort_scan, "b_sort",        &scene->sort_keys_a);
						skr_compute_set_buffer(&scene->sort_scan, "b_alt",         &scene->sort_keys_b);
						skr_compute_set_buffer(&scene->sort_scan, "b_sortPayload", &scene->sort_payload_a);
						skr_compute_set_buffer(&scene->sort_scan, "b_altPayload",  &scene->sort_payload_b);
						skr_compute_set_buffer(&scene->sort_scan, "b_passHist",    &scene->pass_hist);
						skr_compute_set_buffer(&scene->sort_scan, "b_globalHist",  &scene->global_hist);
					}
					if (skr_compute_is_valid(&scene->sort_downsweep)) {
						skr_compute_set_buffer(&scene->sort_downsweep, "b_sort",        &scene->sort_keys_a);
						skr_compute_set_buffer(&scene->sort_downsweep, "b_alt",         &scene->sort_keys_b);
						skr_compute_set_buffer(&scene->sort_downsweep, "b_sortPayload", &scene->sort_payload_a);
						skr_compute_set_buffer(&scene->sort_downsweep, "b_altPayload",  &scene->sort_payload_b);
						skr_compute_set_buffer(&scene->sort_downsweep, "b_passHist",    &scene->pass_hist);
						skr_compute_set_buffer(&scene->sort_downsweep, "b_globalHist",  &scene->global_hist);
					}
				} else {
					free(path);
				}
			}
		}
	}

	igSeparator();
	if (igButton("Reset Camera", (ImVec2){0, 0})) {
		scene->cam_yaw          = 0.0f;
		scene->cam_pitch        = 0.0f;
		scene->cam_yaw_vel      = 0.0f;
		scene->cam_pitch_vel    = 0.0f;
		scene->cam_distance_vel = 0.0f;
		scene->cam_target_vel   = (float3){0, 0, 0};
	}

	igText("Camera Distance: %.1f", scene->cam_distance);
}

const scene_vtable_t scene_gaussian_splat_vtable = {
	.name       = "Gaussian Splat",
	.create     = _scene_gaussian_splat_create,
	.destroy    = _scene_gaussian_splat_destroy,
	.update     = _scene_gaussian_splat_update,
	.render     = _scene_gaussian_splat_render,
	.get_camera = _scene_gaussian_splat_get_camera,
	.render_ui  = _scene_gaussian_splat_render_ui,
};
