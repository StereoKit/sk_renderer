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
#include <float.h>

// Single-pass GPU particle system: the vertex shader both simulates and renders
// particles in one draw call, with no separate compute pass. Each particle is a
// billboard quad (6 verts), and the first vertex of each particle runs the
// physics simulation and writes the result to a ping-pong output buffer. All 6
// vertices then read the particle's position to expand a view-space quad.

#define PARTICLE_COUNT 250000

// Particle params: rendering colors + simulation parameters, all packed into
// the shader's $Global cbuffer via loose uniforms.
typedef struct {
	float3 color_slow;
	float  max_speed;
	float3 color_fast;
	float  sim_time;
	float  delta_time;
	float  damping;
	float  strength;
	float  _pad3;
} particle_params_t;

// Orbital particles scene - VS UAV single-pass particle simulation + rendering
typedef struct {
	scene_t           base;
	skr_mesh_t        particle_mesh;
	skr_shader_t      shader;
	skr_material_t    material;
	particle_params_t params;
	skr_buffer_t      particle_buffer_a;
	skr_buffer_t      particle_buffer_b;

	float   time;
	int32_t frame;
} scene_orbital_particles_t;

// Particle data
// Padded so float3s don't cross 16-byte boundaries, matching the shader's
// core-Vulkan-compatible layout.
typedef struct {
	float3 position;
	float  _pad0;
	float3 velocity;
	float  _pad1;
} particle_t;

// Helper function for random hash
static float _hash_f(int32_t aPosition, uint32_t aSeed) {
	const uint32_t BIT_NOISE1 = 0x68E31DA4;
	const uint32_t BIT_NOISE2 = 0xB5297A4D;
	const uint32_t BIT_NOISE3 = 0x1B56C4E9;

	uint32_t mangled = (uint32_t)aPosition;
	mangled *= BIT_NOISE1;
	mangled += aSeed;
	mangled ^= (mangled >> 8);
	mangled += BIT_NOISE2;
	mangled ^= (mangled << 8);
	mangled *= BIT_NOISE3;
	mangled ^= (mangled >> 8);
	return (float)mangled / (float)4294967295;
}

static scene_t* _scene_orbital_particles_create(void) {
	scene_orbital_particles_t* scene = calloc(1, sizeof(scene_orbital_particles_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_orbital_particles_t);
	scene->time      = 0.0f;
	scene->frame     = 0;

	// Null vertex buffer mesh: 6 verts per particle (billboard quad), all data from StructuredBuffer
	skr_mesh_create(NULL, skr_index_fmt_u32, NULL, PARTICLE_COUNT * 6, NULL, 0, &scene->particle_mesh);

	// Load shader (VS does both simulation + rendering)
	scene->shader = su_shader_load("shaders/orbital_particles.hlsl.sks", "orbital_particles_shader");
	skr_material_create((skr_material_info_t){
		.shader       = &scene->shader,
		.cull         = skr_cull_none,
		.write_mask   = skr_write_default,
		.depth_test   = skr_compare_less,
	}, &scene->material);

	// Initialize particles in a sphere
	particle_t* particles = malloc(PARTICLE_COUNT * sizeof(particle_t));
	for (int i = 0; i < PARTICLE_COUNT; i++) {
		float theta  = _hash_f(i, 0) * 3.14159f * 2.0f;
		float phi    = _hash_f(i, 1) * 3.14159f;
		float radius = _hash_f(i, 2) * 5.0f + 1.0f;

		particles[i].velocity = (float3){0, 0, 0};
		particles[i].position = (float3){
			sinf(phi) * cosf(theta) * radius,
			sinf(phi) * sinf(theta) * radius,
			cosf(phi) * radius
		};
	}

	// Create ping-pong particle buffers for VS UAV read/write
	skr_buffer_create(particles, PARTICLE_COUNT, sizeof(particle_t), skr_buffer_type_storage, skr_use_compute_readwrite, &scene->particle_buffer_a);
	skr_buffer_create(particles, PARTICLE_COUNT, sizeof(particle_t), skr_buffer_type_storage, skr_use_compute_readwrite, &scene->particle_buffer_b);
	free(particles);

	scene->params = (particle_params_t){
		.color_slow = {0.818f, 0.0100f, 0.0177f},
		.max_speed  = 5.0f,
		.color_fast = {0.955f, 0.758f, 0.0177f},
		.sim_time   = 0.0f,
		.delta_time = 0.0f,
		.damping    = 0.98f,
		.strength   = 4.0f,
		._pad3      = 0.0f,
	};

	return (scene_t*)scene;
}

static void _scene_orbital_particles_destroy(scene_t* base) {
	scene_orbital_particles_t* scene = (scene_orbital_particles_t*)base;

	skr_mesh_destroy    (&scene->particle_mesh);
	skr_material_destroy(&scene->material);
	skr_shader_destroy  (&scene->shader);
	skr_buffer_destroy  (&scene->particle_buffer_a);
	skr_buffer_destroy  (&scene->particle_buffer_b);

	free(scene);
}

static void _scene_orbital_particles_update(scene_t* base, float delta_time) {
	scene_orbital_particles_t* scene = (scene_orbital_particles_t*)base;
	scene->time            += delta_time;
	scene->params.sim_time  = scene->time;
	scene->params.delta_time = delta_time;
}

static void _scene_orbital_particles_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	scene_orbital_particles_t* scene = (scene_orbital_particles_t*)base;

	// Bind params (colors + simulation uniforms)
	skr_material_set_params(&scene->material, &scene->params, sizeof(scene->params));

	// Ping-pong: read from one buffer, VS writes to the other
	skr_buffer_t* read_buf  = (scene->frame % 2 == 0) ? &scene->particle_buffer_a : &scene->particle_buffer_b;
	skr_buffer_t* write_buf = (scene->frame % 2 == 0) ? &scene->particle_buffer_b : &scene->particle_buffer_a;
	skr_material_set_buffer(&scene->material, "particles",     read_buf);
	skr_material_set_buffer(&scene->material, "particles_out", write_buf);
	scene->frame++;

	skr_render_list_add(ref_render_list, &scene->particle_mesh, &scene->material, NULL, 0, 1);
}

const scene_vtable_t scene_orbital_particles_vtable = {
	.name       = "Orbital Particles",
	.create     = _scene_orbital_particles_create,
	.destroy    = _scene_orbital_particles_destroy,
	.update     = _scene_orbital_particles_update,
	.render     = _scene_orbital_particles_render,
	.get_camera = NULL,
};
