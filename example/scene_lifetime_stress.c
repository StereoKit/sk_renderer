// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "scene.h"
#include "tools/scene_util.h"
#include "app.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>

// Lifetime Stress Test Scene
// Tests various resource creation/destruction patterns to validate thread safety
// and deferred destruction systems.

#define MAX_EPHEMERAL_MATERIALS 32
#define MAX_THREAD_MATERIALS    8
#define STRESS_CUBE_COUNT       25
#define SORT_STRESS_MAT_COUNT   16
#define SORT_STRESS_MESH_COUNT  4
#define READBACK_SLOTS          4
#define READBACK_UINTS          256

typedef struct {
	skr_material_t material;
	skr_tex_t      texture;
	int32_t        frames_alive;
	int32_t        destroy_after_frames;
	bool           in_use;
} ephemeral_resource_t;

typedef struct {
	skr_material_t material;
	skr_tex_t      texture;
	bool           ready;
	bool           used;
} thread_resource_t;

typedef struct {
	scene_t base;

	// Permanent resources (for comparison/baseline rendering)
	skr_mesh_t     cube_mesh;
	skr_shader_t   shader;
	skr_material_t base_material;
	skr_tex_t      base_texture;

	// Test 1: Create-use-destroy in same frame
	uint32_t test1_count;

	// Test 2: Create one frame, destroy next
	ephemeral_resource_t ephemeral[MAX_EPHEMERAL_MATERIALS];
	uint32_t             ephemeral_next;

	// Test 3: Thread-created resources
	pthread_t          thread;
	thread_resource_t  thread_resources[MAX_THREAD_MATERIALS];
	pthread_mutex_t    thread_mutex;
	volatile bool      thread_running;
	volatile bool      thread_should_stop;
	uint32_t           thread_create_count;

	// Test 4: Rapid create/destroy cycles
	uint32_t rapid_cycle_count;
	uint32_t rapid_cycles_per_frame;

	// Test 5: Texture replacement stress
	skr_tex_t      replaceable_texture;
	skr_material_t replaceable_material;
	uint32_t       texture_replace_count;

	// Test 6: Sampler stress (different sampler settings)
	skr_tex_t      sampler_test_textures[8];
	uint32_t       sampler_test_count;

	// Test 7: True destroy-before-draw (validates crash behavior)
	uint32_t       test7_count;

	// Test 8: Sort stress (many draw items with diverse sort keys)
	skr_material_t sort_stress_materials[SORT_STRESS_MAT_COUNT];
	skr_mesh_t     sort_stress_meshes[SORT_STRESS_MESH_COUNT];
	int32_t        sort_stress_draw_count;
	bool           sort_stress_enabled;

	// Test 10: Worker-side destroy while the open frame references the texture
	skr_tex_t      worker_texs[2];
	skr_tex_t      worker_copy_dst;
	skr_material_t worker_material;
	int32_t        worker_cur;          // slot the material binds, main only
	int32_t        worker_retire;       // slot the worker should destroy, -1 if none (thread_mutex)
	bool           worker_next_ready;   // texs[worker_cur ^ 1] exists (thread_mutex)
	bool           worker_retire_done;  // (thread_mutex)
	uint32_t       worker_destroy_count;

	// Tests 11-15: the other destroy timings, see render and idle
	skr_shader_t   fill_shader;
	skr_buffer_t   fill_buffer;
	skr_tex_t      unreg_tex;             // drawn one frame, destroyed by an unregistered thread after it
	bool           unreg_tex_live;
	uint32_t       idle_destroy_count;    // Test 11
	uint32_t       exit_destroy_count;    // Test 12
	uint32_t       disown_count;          // Test 13
	uint32_t       unreg_destroy_count;   // Test 14
	uint32_t       compute_destroy_count; // Test 15

	// Tests 16-17: futures crossing threads
	skr_future_t   worker_future;         // the worker's latest submit (thread_mutex)
	uint32_t       exited_readback_count; // Test 16
	uint32_t       cross_wait_count;      // Test 17

	// Test 9: Buffer readback churn (snapshots against live destroys)
	skr_buffer_t          readback_buffers[READBACK_SLOTS];
	skr_buffer_readback_t readbacks       [READBACK_SLOTS];
	uint32_t              readback_seeds  [READBACK_SLOTS];
	bool                  readback_live   [READBACK_SLOTS];
	uint32_t              readback_started;
	uint32_t              readback_verified;
	uint32_t              readback_abandoned;
	uint32_t              readback_failures;

	// Statistics
	uint32_t frame_count;
	uint32_t total_creates;
	uint32_t total_destroys;
	uint32_t total_draws;
	float    rotation;
} scene_lifetime_stress_t;

// Thread function that creates resources
static void* _thread_create_resources(void* arg) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)arg;

	// Register this thread with sk_renderer command system
	skr_thread_init();

	while (!scene->thread_should_stop) {
		pthread_mutex_lock(&scene->thread_mutex);

		// Find an unused slot
		int32_t slot = -1;
		for (int32_t i = 0; i < MAX_THREAD_MATERIALS; i++) {
			if (!scene->thread_resources[i].ready && !scene->thread_resources[i].used) {
				slot = i;
				break;
			}
		}

		pthread_mutex_unlock(&scene->thread_mutex);

		if (slot >= 0) {
			// Create resources outside the lock
			thread_resource_t* res = &scene->thread_resources[slot];

			// Cull modes for pipeline variety
			skr_cull_ cull_modes[] = { skr_cull_back, skr_cull_front, skr_cull_none };

			// Create a unique colored texture
			uint32_t color = 0xFF000000 | ((slot * 37) << 16) | ((slot * 73) << 8) | (slot * 113);
			res->texture = su_tex_create_solid_color(color);
			skr_tex_set_name(&res->texture, "thread_tex");

			// Create material with varied pipeline settings
			skr_material_create((skr_material_info_t){
				.shader     = &scene->shader,
				.depth_test = skr_compare_less,
				.cull       = cull_modes[slot % 3],  // Vary cull mode based on slot
			}, &res->material);
			skr_material_set_tex(&res->material, "tex", &res->texture);

			// Mark as ready
			pthread_mutex_lock(&scene->thread_mutex);
			res->ready = true;
			scene->thread_create_count++;
			pthread_mutex_unlock(&scene->thread_mutex);
		}

		// Test 10: retire the old texture here, on the worker, while main's
		// open frame still references it. Then build the next replacement
		// into the slot that just freed up.
		pthread_mutex_lock(&scene->thread_mutex);
		int32_t retire = scene->worker_retire;
		pthread_mutex_unlock(&scene->thread_mutex);
		if (retire >= 0) {
			skr_cmd_begin();
			skr_tex_destroy(&scene->worker_texs[retire]);
			skr_cmd_end();
			// Cycle this thread's whole command ring, the point at which a
			// destroy tied to this thread's fences would run
			for (int32_t i = 0; i < 10; i++) {
				skr_tex_t churn = su_tex_create_solid_color(0xFF102030);
				skr_tex_destroy(&churn);
			}
			pthread_mutex_lock(&scene->thread_mutex);
			scene->worker_retire      = -1;
			scene->worker_retire_done = true;
			pthread_mutex_unlock(&scene->thread_mutex);
		}

		pthread_mutex_lock(&scene->thread_mutex);
		bool    next_ready = scene->worker_next_ready;
		int32_t next       = scene->worker_cur ^ 1;
		pthread_mutex_unlock(&scene->thread_mutex);
		if (!next_ready) {
			scene->worker_texs[next] = su_tex_create_solid_color(0xFF00FFFF ^ (scene->worker_destroy_count * 0x1234));
			skr_tex_set_name(&scene->worker_texs[next], "worker_tex");
			pthread_mutex_lock(&scene->thread_mutex);
			scene->worker_next_ready = true;
			scene->worker_future     = skr_future_get();  // Test 17 waits on this from main
			pthread_mutex_unlock(&scene->thread_mutex);
		}

		// Sleep a bit to avoid spinning
		struct timespec ts = { .tv_sec = 0, .tv_nsec = 10000000 }; // 10ms
		nanosleep(&ts, NULL);
	}

	// Unregister this thread from sk_renderer command system
	skr_thread_shutdown();

	return NULL;
}

// A thread that lives only to destroy one object: registers, destroys inside
// a scope, cycles its ring, and exits while main's frame may still use it.
// Unregistered skips all of that, a destroy from a thread skr never met.
typedef struct {
	skr_tex_t*     opt_tex;
	skr_compute_t* opt_compute;
	int32_t        churn;
	bool           unregistered;
	bool           leave_scope_open;  // exits without skr_cmd_end: an app bug the shutdown must survive
} destroyer_t;

static void* _destroyer_thread(void* arg) {
	destroyer_t* d = (destroyer_t*)arg;
	if (d->unregistered) {
		if (d->opt_tex) skr_tex_destroy(d->opt_tex);
		return NULL;
	}
	skr_thread_init();
	skr_cmd_begin();
	if (d->opt_tex)     skr_tex_destroy    (d->opt_tex);
	if (d->opt_compute) skr_compute_destroy(d->opt_compute);
	if (!d->leave_scope_open) skr_cmd_end();
	for (int32_t i = 0; i < d->churn; i++) {
		skr_tex_t churn = su_tex_create_solid_color(0xFF304050);
		skr_tex_destroy(&churn);
	}
	skr_thread_shutdown();
	return NULL;
}

static void _destroy_on_thread(destroyer_t d) {
	pthread_t thread;
	pthread_create(&thread, NULL, _destroyer_thread, &d);
	pthread_join(thread, NULL);
}

// Stamps a pending block with its own fence, then exits with that block still
// behind a newer unstamped one, so the stamp has to be disowned before the
// fence dies. Only meaningful while main has no buffer open.
static void* _disown_thread(void* arg) {
	(void)arg;
	skr_thread_init();
	skr_tex_t a = su_tex_create_solid_color(0xFF111111);
	skr_tex_destroy(&a);                                  // pending, unstamped
	skr_tex_t b = su_tex_create_solid_color(0xFF222222);  // this submit stamps a's blocks
	skr_tex_destroy(&b);                                  // newer and unstamped, so a's can't free yet
	skr_thread_shutdown();
	return NULL;
}

// Starts a readback on a thread that exits before anyone looks at the future,
// so the fence it waited on is gone by the time main checks
typedef struct {
	skr_buffer_t          buffer;
	skr_buffer_readback_t readback;
	bool                  ok;
} exited_readback_t;

static void* _exited_readback_thread(void* arg) {
	exited_readback_t* r = (exited_readback_t*)arg;
	skr_thread_init();
	uint32_t data[READBACK_UINTS];
	for (uint32_t i = 0; i < READBACK_UINTS; i++) data[i] = 0xC0DE0000 + i;
	r->ok = skr_buffer_create(data, READBACK_UINTS, sizeof(uint32_t), skr_buffer_type_storage, skr_use_compute_read, &r->buffer) == skr_err_success
	     && skr_buffer_readback(&r->buffer, &r->readback) == skr_err_success;
	skr_thread_shutdown();
	return NULL;
}

// Hands texs[cur] to the worker to destroy and waits for it to cycle its
// ring. With copy, the open frame references the texture first.
static bool _worker_swap(scene_lifetime_stress_t* scene, bool copy) {
	pthread_mutex_lock(&scene->thread_mutex);
	bool ready = scene->worker_next_ready;
	pthread_mutex_unlock(&scene->thread_mutex);
	if (!ready) return false;

	int32_t cur  = scene->worker_cur;
	int32_t next = cur ^ 1;
	if (copy) skr_tex_copy(&scene->worker_texs[cur], &scene->worker_copy_dst, 0, 0, 0, 0, 1);
	skr_material_set_tex(&scene->worker_material, "tex", &scene->worker_texs[next]);
	scene->worker_cur = next;

	pthread_mutex_lock(&scene->thread_mutex);
	scene->worker_retire      = cur;
	scene->worker_retire_done = false;
	scene->worker_next_ready  = false;
	pthread_mutex_unlock(&scene->thread_mutex);
	for (bool done = false; !done; ) {
		pthread_mutex_lock(&scene->thread_mutex);
		done = scene->worker_retire_done;
		pthread_mutex_unlock(&scene->thread_mutex);
		if (!done) nanosleep(&(struct timespec){ .tv_nsec = 500000 }, NULL);
	}
	return true;
}

#define FILL_COUNT 1024  // uints written by buffer_fill.hlsl

static scene_t* _scene_lifetime_stress_create(void) {
	scene_lifetime_stress_t* scene = calloc(1, sizeof(scene_lifetime_stress_t));
	if (!scene) return NULL;

	scene->base.size = sizeof(scene_lifetime_stress_t);
	scene->rotation = 0.0f;
	scene->rapid_cycles_per_frame = 5;

	// Create base resources
	scene->cube_mesh = su_mesh_create_cube(0.8f, NULL);
	skr_mesh_set_name(&scene->cube_mesh, "stress_cube");

	scene->base_texture = su_tex_create_checkerboard(64, 8, 0xFFFFFFFF, 0xFF4444FF, true);
	skr_tex_set_name(&scene->base_texture, "stress_base_tex");

	scene->shader = su_shader_load("shaders/test.hlsl.sks", "stress_shader");

	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.depth_test = skr_compare_less,
	}, &scene->base_material);
	skr_material_set_tex(&scene->base_material, "tex", &scene->base_texture);

	// Create replaceable texture/material for Test 5
	scene->replaceable_texture = su_tex_create_solid_color(0xFF00FF00);
	skr_tex_set_name(&scene->replaceable_texture, "replaceable_tex");

	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.depth_test = skr_compare_less,
	}, &scene->replaceable_material);
	skr_material_set_tex(&scene->replaceable_material, "tex", &scene->replaceable_texture);

	// Create sampler test textures with different sampler settings
	for (int32_t i = 0; i < 8; i++) {
		uint32_t color = 0xFF000000 | (i * 32);
		skr_tex_sampler_t sampler = {
			.sample     = (i % 2 == 0) ? skr_tex_sample_linear : skr_tex_sample_point,
			.address    = (i % 3 == 0) ? skr_tex_address_wrap : skr_tex_address_clamp,
			.anisotropy = (i % 4) + 1,
		};

		skr_tex_create(skr_tex_fmt_rgba32_linear, skr_tex_flags_dynamic | skr_tex_flags_uninitialized, sampler, (skr_vec3i_t){4, 4, 1}, 1, 1, NULL, &scene->sampler_test_textures[i]);

		// Fill with color
		uint32_t pixels[16];
		for (int32_t j = 0; j < 16; j++) pixels[j] = color | ((j * 16) << 8);
		skr_tex_set_data(&scene->sampler_test_textures[i], &(skr_tex_data_t){.data = pixels, .mip_count = 1, .layer_count = 1});
	}

	// Test 10: the worker swaps and destroys these while main draws them
	scene->worker_texs[0]  = su_tex_create_solid_color(0xFF00FFFF);
	scene->worker_copy_dst = su_tex_create_solid_color(0xFF000000);
	skr_tex_set_name(&scene->worker_texs[0],  "worker_tex");
	skr_tex_set_name(&scene->worker_copy_dst, "worker_copy_dst");
	skr_material_create((skr_material_info_t){
		.shader     = &scene->shader,
		.depth_test = skr_compare_less,
	}, &scene->worker_material);
	skr_material_set_tex(&scene->worker_material, "tex", &scene->worker_texs[0]);
	scene->worker_retire = -1;

	// Test 15: a compute pipeline the frame dispatches before a thread destroys it
	scene->fill_shader = su_shader_load("shaders/buffer_fill.hlsl.sks", "buffer_fill");
	skr_buffer_create(NULL, FILL_COUNT, sizeof(uint32_t), skr_buffer_type_storage, skr_use_compute_write | skr_use_uninitialized, &scene->fill_buffer);
	skr_buffer_set_name(&scene->fill_buffer, "stress_fill_buffer");

	// Initialize thread resources
	pthread_mutex_init(&scene->thread_mutex, NULL);
	scene->thread_running = true;
	scene->thread_should_stop = false;
	pthread_create(&scene->thread, NULL, _thread_create_resources, scene);

	// Test 8: Sort stress - create diverse materials and meshes
	scene->sort_stress_draw_count = 2000;
	scene->sort_stress_enabled    = true;

	// 4 different mesh geometries for VkBuffer diversity
	scene->sort_stress_meshes[0] = scene->cube_mesh; // share the cube
	scene->sort_stress_meshes[1] = su_mesh_create_sphere(8, 6, 0.4f, (skr_vec4_t){1,1,1,1});
	scene->sort_stress_meshes[2] = su_mesh_create_pyramid(0.6f, 0.8f, (skr_vec4_t){1,1,1,1});
	scene->sort_stress_meshes[3] = su_mesh_create_quad(0.6f, 0.6f, (skr_vec3_t){0,0,1}, false, (skr_vec4_t){1,1,1,1});

	// 16 materials with varied pipeline settings for diverse sort keys
	skr_cull_    sort_culls[]  = { skr_cull_back, skr_cull_front, skr_cull_none, skr_cull_back };
	skr_compare_ sort_depths[] = { skr_compare_less, skr_compare_less_or_eq };
	int16_t      sort_queues[] = { -5, 0, 5, 10 };
	for (int32_t i = 0; i < SORT_STRESS_MAT_COUNT; i++) {
		skr_material_create((skr_material_info_t){
			.shader       = &scene->shader,
			.depth_test   = sort_depths[i % 2],
			.cull         = sort_culls[i % 4],
			.queue_offset = sort_queues[i / 4],
		}, &scene->sort_stress_materials[i]);
		skr_material_set_tex(&scene->sort_stress_materials[i], "tex", &scene->base_texture);
	}

	su_log(su_log_info, "Lifetime stress test scene created");

	return (scene_t*)scene;
}

static void _scene_lifetime_stress_destroy(scene_t* base) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;

	// Stop thread
	scene->thread_should_stop = true;
	pthread_join(scene->thread, NULL);
	pthread_mutex_destroy(&scene->thread_mutex);

	// Destroy thread resources
	for (int32_t i = 0; i < MAX_THREAD_MATERIALS; i++) {
		if (scene->thread_resources[i].ready || scene->thread_resources[i].used) {
			skr_material_destroy(&scene->thread_resources[i].material);
			skr_tex_destroy(&scene->thread_resources[i].texture);
		}
	}

	// Destroy ephemeral resources
	for (int32_t i = 0; i < MAX_EPHEMERAL_MATERIALS; i++) {
		if (scene->ephemeral[i].in_use) {
			skr_material_destroy(&scene->ephemeral[i].material);
			skr_tex_destroy(&scene->ephemeral[i].texture);
		}
	}

	// Destroy sampler test textures
	for (int32_t i = 0; i < 8; i++) {
		skr_tex_destroy(&scene->sampler_test_textures[i]);
	}

	// Destroy replaceable resources
	skr_material_destroy(&scene->replaceable_material);
	skr_tex_destroy(&scene->replaceable_texture);

	// Destroy worker swap resources; the thread is joined, so the flags are settled
	skr_material_destroy(&scene->worker_material);
	skr_tex_destroy(&scene->worker_texs[scene->worker_cur]);
	if (scene->worker_next_ready) skr_tex_destroy(&scene->worker_texs[scene->worker_cur ^ 1]);
	skr_tex_destroy(&scene->worker_copy_dst);
	if (scene->unreg_tex_live) skr_tex_destroy(&scene->unreg_tex);
	skr_buffer_destroy(&scene->fill_buffer);
	skr_shader_destroy(&scene->fill_shader);

	// Destroy in-flight readbacks, then their buffers
	for (int32_t i = 0; i < READBACK_SLOTS; i++) {
		if (scene->readback_live[i]) {
			skr_buffer_readback_destroy(&scene->readbacks[i]);
			skr_buffer_destroy(&scene->readback_buffers[i]);
		}
	}

	// Destroy sort stress resources
	for (int32_t i = 0; i < SORT_STRESS_MAT_COUNT; i++)
		skr_material_destroy(&scene->sort_stress_materials[i]);
	// meshes[0] is shared cube_mesh, destroyed below
	for (int32_t i = 1; i < SORT_STRESS_MESH_COUNT; i++)
		skr_mesh_destroy(&scene->sort_stress_meshes[i]);

	// Destroy base resources
	skr_material_destroy(&scene->base_material);
	skr_tex_destroy(&scene->base_texture);
	skr_shader_destroy(&scene->shader);
	skr_mesh_destroy(&scene->cube_mesh);

	su_log(su_log_info, "Lifetime stress test: %u creates, %u destroys, %u draws over %u frames",
		scene->total_creates, scene->total_destroys, scene->total_draws, scene->frame_count);
	su_log(su_log_info, "Lifetime stress threads: %u worker in-frame, %u worker idle, %u thread-exit, %u disown, %u unregistered, %u compute, %u exited-thread futures, %u cross-thread waits",
		scene->worker_destroy_count, scene->idle_destroy_count, scene->exit_destroy_count, scene->disown_count, scene->unreg_destroy_count, scene->compute_destroy_count,
		scene->exited_readback_count, scene->cross_wait_count);
	su_log(scene->readback_failures > 0 ? su_log_warning : su_log_info,
		"Lifetime stress readbacks: %u started, %u verified, %u abandoned, %u failures",
		scene->readback_started, scene->readback_verified, scene->readback_abandoned, scene->readback_failures);

	free(scene);
}

static void _scene_lifetime_stress_update(scene_t* base, float dt) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;
	scene->rotation += dt * 0.5f;
	scene->frame_count++;

	// Test 2: Age ephemeral resources and destroy when ready
	for (int32_t i = 0; i < MAX_EPHEMERAL_MATERIALS; i++) {
		ephemeral_resource_t* eph = &scene->ephemeral[i];
		if (eph->in_use) {
			eph->frames_alive++;
			if (eph->frames_alive >= eph->destroy_after_frames) {
				skr_material_destroy(&eph->material);
				skr_tex_destroy(&eph->texture);
				eph->in_use = false;
				scene->total_destroys++;
			}
		}
	}

	// Test 3: Check for used thread resources and destroy them
	pthread_mutex_lock(&scene->thread_mutex);
	for (int32_t i = 0; i < MAX_THREAD_MATERIALS; i++) {
		thread_resource_t* res = &scene->thread_resources[i];
		if (res->used && !res->ready) {
			// Was used and is no longer marked ready, destroy it
			skr_material_destroy(&res->material);
			skr_tex_destroy(&res->texture);
			res->used = false;
			scene->total_destroys++;
		}
	}
	pthread_mutex_unlock(&scene->thread_mutex);

	// Test 9: Buffer readback churn
	{
		uint32_t slot = scene->frame_count % READBACK_SLOTS;

		// Retire the slot's previous round: verify when complete, abandon
		// mid-flight otherwise - both must clean up without touching freed memory
		if (scene->readback_live[slot]) {
			if (skr_future_check(&scene->readbacks[slot].future)) {
				const uint32_t* values = (const uint32_t*)scene->readbacks[slot].data;
				bool ok = scene->readbacks[slot].size == READBACK_UINTS * sizeof(uint32_t);
				for (uint32_t i = 0; ok && i < READBACK_UINTS; i++)
					ok = values[i] == scene->readback_seeds[slot] + i;
				if (ok) scene->readback_verified++;
				else    scene->readback_failures++;
			} else {
				scene->readback_abandoned++;
			}
			skr_buffer_readback_destroy(&scene->readbacks[slot]);
			skr_buffer_destroy(&scene->readback_buffers[slot]);
			scene->readback_live[slot] = false;
			scene->total_destroys++;
		}

		// Refill: upload a seeded pattern and snapshot it in the same frame
		uint32_t seed = scene->frame_count * 2654435761u;
		uint32_t data[READBACK_UINTS];
		for (uint32_t i = 0; i < READBACK_UINTS; i++) data[i] = seed + i;
		if (skr_buffer_create(data, READBACK_UINTS, sizeof(uint32_t), skr_buffer_type_storage, skr_use_compute_read, &scene->readback_buffers[slot]) == skr_err_success) {
			if (skr_buffer_readback(&scene->readback_buffers[slot], &scene->readbacks[slot]) == skr_err_success) {
				scene->readback_seeds[slot] = seed;
				scene->readback_live [slot] = true;
				scene->readback_started++;
				scene->total_creates++;
			} else {
				skr_buffer_destroy(&scene->readback_buffers[slot]);
			}
		}

		// Same-frame create+readback+destroy: the abandon-while-pending path,
		// with the source buffer destroyed right behind it
		if (scene->frame_count % 7 == 0) {
			skr_buffer_t          burst_buffer;
			skr_buffer_readback_t burst;
			if (skr_buffer_create(data, READBACK_UINTS, sizeof(uint32_t), skr_buffer_type_storage, skr_use_compute_read, &burst_buffer) == skr_err_success) {
				if (skr_buffer_readback(&burst_buffer, &burst) == skr_err_success) {
					skr_buffer_readback_destroy(&burst);
					scene->readback_abandoned++;
				}
				skr_buffer_destroy(&burst_buffer);
			}
		}
	}
}

static void _scene_lifetime_stress_render(scene_t* base, int32_t width, int32_t height, skr_render_list_t* ref_render_list, su_system_buffer_t* ref_system_buffer) {
	(void)width; (void)height; (void)ref_system_buffer;
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;

	float4x4 transforms[STRESS_CUBE_COUNT];
	int32_t  draw_idx = 0;
	float3   unit_scale = {1, 1, 1};

	// Cull modes to cycle through for pipeline variety
	skr_cull_ cull_modes[] = { skr_cull_back, skr_cull_front, skr_cull_none };

	// === TEST 1: Create-use-destroy in same frame ===
	// Note: Resources are added to the ephemeral pool with 1-frame lifetime.
	// This ensures destruction happens after skr_renderer_draw processes the list.
	for (int32_t i = 0; i < 3; i++) {
		// Find an empty ephemeral slot
		int32_t slot = -1;
		for (int32_t j = 0; j < MAX_EPHEMERAL_MATERIALS; j++) {
			if (!scene->ephemeral[j].in_use) {
				slot = j;
				break;
			}
		}
		if (slot < 0) continue; // No slots available

		ephemeral_resource_t* eph = &scene->ephemeral[slot];

		// Create temporary texture and material with varied pipeline settings
		uint32_t color = 0xFFFF0000 | ((scene->frame_count * 17 + i * 73) & 0xFFFF);
		eph->texture = su_tex_create_solid_color(color);

		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.depth_test = skr_compare_less,
			.cull       = cull_modes[i % 3],  // Vary cull mode for different pipelines
		}, &eph->material);
		skr_material_set_tex(&eph->material, "tex", &eph->texture);

		eph->frames_alive = 0;
		eph->destroy_after_frames = 1; // Destroy next frame's update (after this frame's render)
		eph->in_use = true;

		scene->total_creates++;

		// Add to render list
		float x = -4.0f + i * 1.5f;
		float y = 2.0f;
		transforms[draw_idx] = float4x4_trs(
			(float3){x, y, 0},
			float4_quat_from_euler((float3){0, scene->rotation + i * 0.5f, 0}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &eph->material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;
		scene->total_draws++;
		scene->test1_count++;
	}

	// === TEST 2: Create resources to destroy next frame ===
	if (scene->frame_count % 3 == 0) {
		int32_t slot = scene->ephemeral_next % MAX_EPHEMERAL_MATERIALS;
		ephemeral_resource_t* eph = &scene->ephemeral[slot];

		// Destroy old if exists
		if (eph->in_use) {
			skr_material_destroy(&eph->material);
			skr_tex_destroy(&eph->texture);
			scene->total_destroys++;
		}

		// Create new with varied pipeline settings
		uint32_t color = 0xFF00FF00 | ((scene->frame_count * 31) & 0xFF00);
		eph->texture = su_tex_create_solid_color(color);

		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.depth_test = skr_compare_less,
			.cull       = cull_modes[slot % 3],  // Vary cull mode based on slot
		}, &eph->material);
		skr_material_set_tex(&eph->material, "tex", &eph->texture);

		eph->frames_alive = 0;
		eph->destroy_after_frames = 2 + (scene->frame_count % 5); // Destroy after 2-6 frames
		eph->in_use = true;
		scene->ephemeral_next++;
		scene->total_creates++;
	}

	// Draw all active ephemeral resources
	for (int32_t i = 0; i < MAX_EPHEMERAL_MATERIALS && draw_idx < STRESS_CUBE_COUNT; i++) {
		ephemeral_resource_t* eph = &scene->ephemeral[i];
		if (eph->in_use) {
			float x = -3.0f + (i % 8) * 1.0f;
			float y = 0.0f;
			float z = (float)(i / 8) * 1.5f;
			transforms[draw_idx] = float4x4_trs(
				(float3){x, y, z},
				float4_quat_from_euler((float3){0, scene->rotation * 0.5f + i * 0.3f, 0}),
				unit_scale);
			skr_render_list_add(ref_render_list, &scene->cube_mesh, &eph->material, &transforms[draw_idx], sizeof(float4x4), 1);
			draw_idx++;
			scene->total_draws++;
		}
	}

	// === TEST 3: Use thread-created resources ===
	pthread_mutex_lock(&scene->thread_mutex);
	for (int32_t i = 0; i < MAX_THREAD_MATERIALS && draw_idx < STRESS_CUBE_COUNT; i++) {
		thread_resource_t* res = &scene->thread_resources[i];
		if (res->ready) {
			float x = 3.0f + (i % 4) * 1.0f;
			float y = -1.5f;
			float z = (float)(i / 4) * 1.5f;
			transforms[draw_idx] = float4x4_trs(
				(float3){x, y, z},
				float4_quat_from_euler((float3){scene->rotation + i * 0.4f, 0, 0}),
				unit_scale);
			skr_render_list_add(ref_render_list, &scene->cube_mesh, &res->material, &transforms[draw_idx], sizeof(float4x4), 1);
			draw_idx++;
			scene->total_draws++;

			// Mark as used, will be destroyed in update and recreated by thread
			if (scene->frame_count % 10 == (uint32_t)i) {
				res->ready = false;
				res->used = true;
			}
		}
	}
	pthread_mutex_unlock(&scene->thread_mutex);

	// === TEST 4: Rapid create/destroy cycles ===
	for (uint32_t cycle = 0; cycle < scene->rapid_cycles_per_frame; cycle++) {
		// Create with varied pipeline settings
		skr_tex_t rapid_tex = su_tex_create_solid_color(0xFFFF00FF);
		skr_material_t rapid_mat;
		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.depth_test = skr_compare_less,
			.cull       = cull_modes[cycle % 3],  // Vary cull mode
		}, &rapid_mat);
		skr_material_set_tex(&rapid_mat, "tex", &rapid_tex);
		scene->total_creates++;

		// Immediately destroy without using
		skr_material_destroy(&rapid_mat);
		skr_tex_destroy(&rapid_tex);
		scene->total_destroys++;
		scene->rapid_cycle_count++;
	}

	// === TEST 5: Texture replacement ===
	if (scene->frame_count % 5 == 0) {
		// Replace the texture with a new one
		skr_tex_t old_tex = scene->replaceable_texture;

		uint32_t new_color = 0xFF000000 | (scene->frame_count * 12345);
		scene->replaceable_texture = su_tex_create_solid_color(new_color);
		skr_tex_set_name(&scene->replaceable_texture, "replaceable_tex_new");

		// Update material to use new texture
		skr_material_set_tex(&scene->replaceable_material, "tex", &scene->replaceable_texture);

		// Destroy old texture
		skr_tex_destroy(&old_tex);
		scene->texture_replace_count++;
		scene->total_creates++;
		scene->total_destroys++;
	}

	// Draw replaceable material cube
	if (draw_idx < STRESS_CUBE_COUNT) {
		transforms[draw_idx] = float4x4_trs(
			(float3){0, -2.5f, 0},
			float4_quat_from_euler((float3){0, 0, scene->rotation * 2.0f}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &scene->replaceable_material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;
		scene->total_draws++;
	}

	// === TEST 10: Worker-side destroy while this frame references the texture ===
	// The copy puts the current texture in the open frame. The worker then
	// destroys it and cycles its ring before this frame submits, so the
	// destroy must retire on main's fence, not the worker's. Every other
	// time, a flush then submits the copy mid-frame with that destroy pending.
	if (scene->frame_count % 15 == 0 && _worker_swap(scene, true)) {
		if (scene->frame_count % 30 == 0) skr_cmd_flush();
		scene->worker_destroy_count++;
		scene->total_destroys++;
	}

	// === TEST 12: Thread exit while this frame references the texture ===
	// Every other time the thread exits with its scope still open
	if (scene->frame_count % 20 == 10) {
		skr_tex_t doomed = su_tex_create_solid_color(0xFFABCDEF);
		skr_tex_copy(&doomed, &scene->worker_copy_dst, 0, 0, 0, 0, 1);
		_destroy_on_thread((destroyer_t){ .opt_tex = &doomed, .churn = 10, .leave_scope_open = scene->frame_count % 40 == 10 });
		scene->exit_destroy_count++;
		scene->total_creates++;
		scene->total_destroys++;
	}

	// === TEST 14, first half: a texture this frame references ===
	// Destroyed after the frame submits, by a thread skr never met (idle)
	if (scene->frame_count % 20 == 0 && !scene->unreg_tex_live) {
		scene->unreg_tex = su_tex_create_solid_color(0xFF0F0F0F);
		skr_tex_copy(&scene->unreg_tex, &scene->worker_copy_dst, 0, 0, 0, 0, 1);
		scene->unreg_tex_live = true;
		scene->total_creates++;
	}

	// === TEST 15: Pipeline destroyed on a thread while this frame dispatches it ===
	if (scene->frame_count % 20 == 15) {
		skr_compute_t doomed_compute;
		skr_compute_create(&scene->fill_shader, (skr_compute_info_t){0}, &doomed_compute);
		skr_compute_set_buffer(&doomed_compute, "output", &scene->fill_buffer);
		skr_compute_set_param (&doomed_compute, "count", sksc_shader_var_uint, 1, &(uint32_t){FILL_COUNT});
		skr_compute_execute   (&doomed_compute, (FILL_COUNT + 63) / 64, 1, 1); // 64 = [numthreads] in buffer_fill.hlsl
		_destroy_on_thread((destroyer_t){ .opt_compute = &doomed_compute, .churn = 10 });
		scene->compute_destroy_count++;
		scene->total_creates++;
		scene->total_destroys++;
	}

	if (draw_idx < STRESS_CUBE_COUNT) {
		transforms[draw_idx] = float4x4_trs(
			(float3){2.0f, -2.5f, 0},
			float4_quat_from_euler((float3){scene->rotation, 0, 0}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &scene->worker_material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;
		scene->total_draws++;
	}

	// === TEST 6: Sampler cache stress ===
	// Create materials with different sampler settings each frame to stress the cache
	if (scene->frame_count % 2 == 0) {
		int32_t sampler_idx = scene->frame_count % 8;

		// Modify sampler settings on existing texture
		skr_tex_sampler_t new_sampler = {
			.sample     = (scene->frame_count % 2 == 0) ? skr_tex_sample_linear : skr_tex_sample_point,
			.address    = (scene->frame_count % 3 == 0) ? skr_tex_address_wrap : skr_tex_address_clamp,
			.anisotropy = (scene->frame_count % 4) + 1,
		};
		skr_tex_set_sampler(&scene->sampler_test_textures[sampler_idx], new_sampler);
		scene->sampler_test_count++;
	}

	// === TEST 7: True destroy-before-draw ===
	// This test validates that materials destroyed before render list processing
	// are handled correctly. Currently this crashes - we'll fix this next.
	if (draw_idx < STRESS_CUBE_COUNT) {
		// Create a material
		skr_material_t doomed_material;
		skr_material_create((skr_material_info_t){
			.shader     = &scene->shader,
			.depth_test = skr_compare_less,
			.cull       = skr_cull_back,
		}, &doomed_material);
		skr_material_set_tex(&doomed_material, "tex", &scene->base_texture);

		// Add it to the render list - stores a POINTER to doomed_material
		transforms[draw_idx] = float4x4_trs(
			(float3){0, 3.0f, 0},
			float4_quat_from_euler((float3){0, scene->rotation * 3.0f, 0}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &doomed_material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;

		// IMMEDIATELY destroy the material - render list still has pointer to freed memory!
		skr_material_destroy(&doomed_material);

		scene->test7_count++;
		scene->total_creates++;
		scene->total_destroys++;
		scene->total_draws++;
		// When skr_renderer_draw processes this render list, it will try to
		// access doomed_material.param_buffer which is now freed -> CRASH
	}

	// === TEST 8: Sort stress - many draw items with diverse sort keys ===
	if (scene->sort_stress_enabled && scene->sort_stress_draw_count > 0) {
		int32_t count = scene->sort_stress_draw_count;
		int32_t cols  = 50;
		float   spacing = 1.2f;
		for (int32_t i = 0; i < count; i++) {
			float x = (float)(i % cols) * spacing - (cols * spacing * 0.5f);
			float z = (float)(i / cols) * spacing - 10.0f;
			float4x4 world = float4x4_trs(
				(float3){x, -2.0f, z},
				float4_quat_from_euler((float3){0, scene->rotation + i * 0.01f, 0}),
				(float3){0.4f, 0.4f, 0.4f});
			skr_render_list_add(ref_render_list,
				&scene->sort_stress_meshes[i % SORT_STRESS_MESH_COUNT],
				&scene->sort_stress_materials[i % SORT_STRESS_MAT_COUNT],
				&world, sizeof(float4x4), 1);
		}
		scene->total_draws += count;
	}

	// Draw base cubes in a grid
	for (int32_t i = 0; i < 5 && draw_idx < STRESS_CUBE_COUNT; i++) {
		float x = -2.0f + i * 1.0f;
		float y = 1.0f;
		transforms[draw_idx] = float4x4_trs(
			(float3){x, y, -3.0f},
			float4_quat_from_euler((float3){0, scene->rotation + i * 0.2f, 0}),
			unit_scale);
		skr_render_list_add(ref_render_list, &scene->cube_mesh, &scene->base_material, &transforms[draw_idx], sizeof(float4x4), 1);
		draw_idx++;
		scene->total_draws++;
	}
}

// Between frames: this thread has no command buffer open, so a worker's own
// submit is the first fence that covers anything pending
static void _scene_lifetime_stress_idle(scene_t* base) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;

	// === TEST 11: Worker-side destroy while main is idle ===
	// The previous frame drew the texture and has submitted, so the worker
	// stamps the destroy with its own fence and frees it on a later churn
	if (scene->frame_count % 15 == 7 && _worker_swap(scene, false)) {
		scene->idle_destroy_count++;
		scene->total_destroys++;
	}

	// === TEST 13: A thread exits with a stamped block still pending ===
	if (scene->frame_count % 20 == 5) {
		pthread_t thread;
		pthread_create(&thread, NULL, _disown_thread, NULL);
		pthread_join(thread, NULL);
		scene->disown_count++;
		scene->total_creates  += 2;
		scene->total_destroys += 2;
	}

	// === TEST 14, second half: an unregistered thread destroys it ===
	if (scene->unreg_tex_live && scene->frame_count % 20 == 0) {
		_destroy_on_thread((destroyer_t){ .opt_tex = &scene->unreg_tex, .unregistered = true });
		scene->unreg_tex_live = false;
		scene->unreg_destroy_count++;
		scene->total_destroys++;
	}

	// === TEST 16: Wait on a future whose thread has exited ===
	if (scene->frame_count % 20 == 12) {
		exited_readback_t r = {0};
		pthread_t thread;
		pthread_create(&thread, NULL, _exited_readback_thread, &r);
		pthread_join(thread, NULL);
		if (r.ok) {
			skr_future_wait(&r.readback.future);
			const uint32_t* values = (const uint32_t*)r.readback.data;
			bool match = skr_future_check(&r.readback.future);
			for (uint32_t i = 0; match && i < READBACK_UINTS; i++)
				match = values[i] == 0xC0DE0000 + i;
			if (match) scene->exited_readback_count++;
			else       scene->readback_failures++;
			skr_buffer_readback_destroy(&r.readback);
			skr_buffer_destroy(&r.buffer);
			scene->total_creates++;
			scene->total_destroys++;
		}
	}

	// === TEST 17: Wait on a live worker's future while it keeps submitting ===
	if (scene->frame_count % 5 == 3) {
		pthread_mutex_lock(&scene->thread_mutex);
		skr_future_t future = scene->worker_future;
		pthread_mutex_unlock(&scene->thread_mutex);
		skr_future_wait(&future);
		if (skr_future_check(&future)) scene->cross_wait_count++;
		else                           scene->readback_failures++;
	}
}

static void _scene_lifetime_stress_render_ui(scene_t* base) {
	scene_lifetime_stress_t* scene = (scene_lifetime_stress_t*)base;

	igText("Test 1 - Same-frame create/destroy: %u", scene->test1_count);
	igText("Test 2 - Multi-frame ephemeral: %u", scene->ephemeral_next);

	int32_t active_ephemeral = 0;
	for (int32_t i = 0; i < MAX_EPHEMERAL_MATERIALS; i++) {
		if (scene->ephemeral[i].in_use) active_ephemeral++;
	}
	igText("  Active ephemeral: %d", active_ephemeral);

	pthread_mutex_lock(&scene->thread_mutex);
	igText("Test 3 - Thread-created: %u", scene->thread_create_count);
	int32_t ready_count = 0;
	for (int32_t i = 0; i < MAX_THREAD_MATERIALS; i++) {
		if (scene->thread_resources[i].ready) ready_count++;
	}
	igText("  Ready to use: %d", ready_count);
	pthread_mutex_unlock(&scene->thread_mutex);

	igText("Test 4 - Rapid cycles: %u", scene->rapid_cycle_count);

	int cycles = (int)scene->rapid_cycles_per_frame;
	if (igSliderInt("Cycles/frame", &cycles, 0, 50, "%d", 0)) {
		scene->rapid_cycles_per_frame = (uint32_t)cycles;
	}

	igText("Test 5 - Texture replacements: %u", scene->texture_replace_count);
	igText("Test 6 - Sampler changes: %u", scene->sampler_test_count);

	igText("Test 7 - Destroy before draw: %u", scene->test7_count);
	igText("Test 10 - Worker destroys, frame open: %u", scene->worker_destroy_count);
	igText("Test 11 - Worker destroys, main idle: %u", scene->idle_destroy_count);
	igText("Test 12 - Thread-exit destroys: %u",       scene->exit_destroy_count);
	igText("Test 13 - Disowned stamps: %u",            scene->disown_count);
	igText("Test 14 - Unregistered-thread destroys: %u", scene->unreg_destroy_count);
	igText("Test 15 - Compute destroyed in-frame: %u", scene->compute_destroy_count);
	igText("Test 16 - Exited-thread futures: %u",     scene->exited_readback_count);
	igText("Test 17 - Cross-thread waits: %u",        scene->cross_wait_count);

	igSeparator();
	igText("Test 8 - Sort stress:");
	igCheckbox("Enabled##sort_stress", &scene->sort_stress_enabled);
	int sort_count = scene->sort_stress_draw_count;
	if (igSliderInt("Draw items", &sort_count, 0, 5000, "%d", 0)) {
		scene->sort_stress_draw_count = sort_count;
	}

	igSeparator();
	igText("Test 9 - Buffer readback churn:");
	igText("  Started: %u  Verified: %u  Abandoned: %u",
		scene->readback_started, scene->readback_verified, scene->readback_abandoned);
	if (scene->readback_failures > 0)
		igTextColored((ImVec4){1.0f, 0.4f, 0.4f, 1.0f}, "  FAILURES: %u", scene->readback_failures);

	igSeparator();
	igText("Totals:");
	igText("  Creates:  %u", scene->total_creates);
	igText("  Destroys: %u", scene->total_destroys);
	igText("  Draws:    %u", scene->total_draws);

	float creates_per_frame = scene->frame_count > 0 ? (float)scene->total_creates / scene->frame_count : 0;
	float destroys_per_frame = scene->frame_count > 0 ? (float)scene->total_destroys / scene->frame_count : 0;
	igText("  Creates/frame:  %.1f", creates_per_frame);
	igText("  Destroys/frame: %.1f", destroys_per_frame);
}

const scene_vtable_t scene_lifetime_stress_vtable = {
	.name      = "Lifetime Stress",
	.create    = _scene_lifetime_stress_create,
	.destroy   = _scene_lifetime_stress_destroy,
	.update    = _scene_lifetime_stress_update,
	.render    = _scene_lifetime_stress_render,
	.render_ui = _scene_lifetime_stress_render_ui,
	.idle      = _scene_lifetime_stress_idle,
};
