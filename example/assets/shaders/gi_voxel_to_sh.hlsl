//--name = gi_voxel_to_sh

// Compute shader to accumulate SH probes from voxel radiance via ray marching.
// Each frame, casts Fibonacci sphere rays per probe through the 3D voxel texture.
// First opaque voxel hit provides radiance; rays exiting the grid sample the
// environment cubemap. Results project into SH via exponential moving average.
//
// Directions: Fibonacci spiral (low-discrepancy) with per-frame random rotation
// (Rodrigues). All probes share the same directions each frame (DDGI-style) for
// better cache coherence. Much less temporal flickering than white noise.
//
// Voxel data stored as RGB10A2 in a 3D texture. Occupancy: alpha > 0.
// Cascaded: rays march through cascade 0 first, then 1, then 2 (fine-to-coarse).

#include "gi_voxel.hlsli"

Texture3D<float4>               voxel_tex     : register(t0);
TextureCube<float4>              env_cubemap   : register(t1);
SamplerState                     env_cubemap_s : register(s1);
RWStructuredBuffer<SHProbe>      sh_probes     : register(u0);

uint  grid_size;
uint  frame_seed;
float sh_decay;    // EMA decay (0.98 = ~50 frame window, 0.99 = ~100)
uint  ray_count;   // total rays per probe per frame
float env_mip;     // cubemap mip level for environment fallback (higher = blurrier)
float env_strength; // multiplier for environment cubemap contribution

#define MAX_RADIANCE  4.0

// Ray march method: 1 = uniform stepping (fast, ~2.4x fewer ALU per step),
//                   0 = DDA (exact voxel traversal, no missed voxels)
// Uniform stepping may skip diagonal voxel corners (~29% miss rate at 45 deg,
// ~42% on body diagonal) but SH probes average over many random rays with EMA,
// making this statistically invisible.
#define RAY_MARCH_UNIFORM 1

#if !RAY_MARCH_UNIFORM
// Single DDA step: advances voxel_pos and t_max along the nearest axis.
// Generates ~18 ALU per step on RDNA3 (3 comparisons + SGPR boolean chain
// + 6 conditional moves + 6 adds). The SGPR dependency chain
// (cmp -> s_and -> s_and_not1 -> s_or) limits VALU throughput.
void _dda_step(inout int3 voxel_pos, inout float3 t_max, float3 t_delta, int3 step_dir) {
	bool sel_x = (t_max.x <= t_max.y) && (t_max.x <= t_max.z);
	bool sel_y = !sel_x && (t_max.y <= t_max.z);

	voxel_pos += int3(sel_x ? step_dir.x : 0,
	                  sel_y ? step_dir.y : 0,
	                  (!sel_x && !sel_y) ? step_dir.z : 0);
	t_max += float3(sel_x ? t_delta.x : 0.0,
	                sel_y ? t_delta.y : 0.0,
	                (!sel_x && !sel_y) ? t_delta.z : 0.0);
}
#endif

// Single-round integer hash.
uint _hash(uint x) {
	x ^= x >> 16;
	x *= 0x45d9f3bu;
	x ^= x >> 16;
	return x;
}

// Fibonacci sphere: low-discrepancy direction sampling (DDGI-style).
// Golden ratio spacing ensures near-uniform sphere coverage with N points.
// Much less temporal flickering than white noise because directions are
// well-distributed rather than random. Per-frame rotation (Rodrigues)
// ensures each frame samples different directions while keeping the
// uniform distribution property.
#define GOLDEN_RATIO 1.6180339887

float3 _fibonacci_dir(uint index, uint total) {
	float i = (float)index + 0.5;
	float n = (float)total;

	// Cylindrical equal-area: z linearly spaced, phi by golden ratio
	float z   = 1.0 - 2.0 * i / n;
	float r   = sqrt(max(0.0, 1.0 - z * z));
	float phi = 6.28318530718 * frac(i * (1.0 / GOLDEN_RATIO));
	return float3(r * cos(phi), r * sin(phi), z);
}

// Rodrigues rotation: rotate vector v around unit axis k by angle theta.
float3 _rodrigues(float3 v, float3 k, float s, float c) {
	return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	// Multi-pass dispatch: Z dimension is 4x grid size. Each pass touches every
	// probe with ray_count/4 rays. Pass 0 applies EMA decay + accumulates;
	// passes 1-3 just accumulate.
	uint  pass = id.z / GI_GRID;
	uint3 pid  = uint3(id.x, id.y, id.z % GI_GRID);

	// Probe world position (computed from cascade 0)
	float3 vol_size_0  = 1.0 / gi_cascades[0].volume_inv;
	float3 probe_world = gi_cascades[0].volume_min + (float3(pid) + 0.5) * GI_INV_GRID * vol_size_0;

	// Weight per ray: full sphere sampling (4*pi)
	float solid_angle = 12.56637; // 4*pi
	float w           = solid_angle * (1.0 - sh_decay) / (float)ray_count;

	// Per-frame rotation via Rodrigues: random axis + angle from frame_seed.
	// Computed once per wavefront (scalar ALU), applied to every Fibonacci direction.
	// All probes share the same rotated directions each frame (DDGI-style),
	// improving texture cache coherence since nearby probes march similar paths.
	uint   rot_h = _hash(frame_seed);
	float  rot_a = 6.28318530718 * float(rot_h) / 4294967295.0;
	float  rot_z = 1.0 - 2.0 * float(_hash(rot_h)) / 4294967295.0;
	float  rot_r = sqrt(max(0.0, 1.0 - rot_z * rot_z));
	float  rot_p = 6.28318530718 * float(_hash(rot_h + 1u)) / 4294967295.0;
	float3 rot_axis = float3(rot_r * cos(rot_p), rot_r * sin(rot_p), rot_z);
	float  rot_sin, rot_cos;
	sincos(rot_a, rot_sin, rot_cos);

	// Each pass does a quarter of the rays
	uint rays_per_pass = ray_count >> 2;

	float4 total_r = float4(0, 0, 0, 0);
	float4 total_g = float4(0, 0, 0, 0);
	float4 total_b = float4(0, 0, 0, 0);

	for (uint ray = 0; ray < rays_per_pass; ray++) {
		// Fibonacci sphere: low-discrepancy directions with per-frame rotation.
		// Interleave across passes (ray*4+pass) so each pass covers the full
		// sphere uniformly, rather than clustering rays in one hemisphere.
		uint ray_idx = ray * 4 + pass;
		float3 dir = _rodrigues(_fibonacci_dir(ray_idx, ray_count), rot_axis, rot_sin, rot_cos);

		// Ray march through cascaded voxel volumes: fine-to-coarse
		float3 radiance = float3(0, 0, 0);
		bool   hit      = false;

#if RAY_MARCH_UNIFORM
		// Uniform stepping with 4-wide prefetch.
		// Per iteration: 12 float-to-int + 12 float-add + 4 loads + 4 alpha checks
		// All ALU ops are independent (no dependency chains), enabling full VALU
		// throughput. Compare to DDA's ~68 ALU with 4-deep SGPR dependency chain.
		for (uint cascade = 0; cascade < GI_CASCADE_COUNT && !hit; cascade++) {
			float3 origin = (probe_world - gi_cascades[cascade].volume_min) * gi_cascades[cascade].volume_inv;
			uint start_s = 1;

			// For cascade > 0: skip inner region covered by previous cascade
			if (cascade > 0) {
				float3 t_lo  = (float3(0.25, 0.25, 0.25) - origin) / dir;
				float3 t_hi  = (float3(0.75, 0.75, 0.75) - origin) / dir;
				float3 t_far = max(t_lo, t_hi);
				float  t_skip = min(t_far.x, min(t_far.y, t_far.z));
				if (t_skip > 0) start_s = max(1u, (uint)(t_skip * GI_VOXEL_RES) + 1u);
			}

			if (any(origin < 0) || any(origin >= 1.0)) continue;

			// Pre-compute exit distance via ray-AABB intersection (~6 ALU).
			// Division by zero gives +-inf (IEEE 754), correctly handled by
			// max/min selection below.
			float3 t_exit_lo = -origin / dir;
			float3 t_exit_hi = (1.0 - origin) / dir;
			float3 t_exit    = max(t_exit_lo, t_exit_hi);
			uint max_steps   = min((uint)GI_VOXEL_RES,
			                       (uint)(min(t_exit.x, min(t_exit.y, t_exit.z)) * (float)GI_VOXEL_RES) + 1u);

			uint z_offset = cascade * GI_VOXEL_RES;

			// Track 4 positions in voxel-space [0, VOXEL_RES) to avoid
			// per-step multiply. All 4 are independent — no dependency chain.
			float3 orig_v = origin * (float)GI_VOXEL_RES;
			float3 step4  = dir * 4.0;
			float3 p0v    = orig_v + dir * (float)start_s;
			float3 p1v    = p0v + dir;
			float3 p2v    = p1v + dir;
			float3 p3v    = p2v + dir;

			for (uint s = start_s; s < max_steps; s += 4) {
				// 12 independent float-to-int conversions
				int3 p0 = int3(p0v);
				int3 p1 = int3(p1v);
				int3 p2 = int3(p2v);
				int3 p3 = int3(p3v);

				// 4 grouped loads (memory-level parallelism via s_clause)
				float4 v0 = voxel_tex.Load(int4(p0.x, p0.y, p0.z + z_offset, 0));
				float4 v1 = voxel_tex.Load(int4(p1.x, p1.y, p1.z + z_offset, 0));
				float4 v2 = voxel_tex.Load(int4(p2.x, p2.y, p2.z + z_offset, 0));
				float4 v3 = voxel_tex.Load(int4(p3.x, p3.y, p3.z + z_offset, 0));

				// First hit wins (OOB loads return 0, alpha check handles bounds)
				if (v0.a > 0) { radiance = v0.rgb; hit = true; break; }
				if (v1.a > 0) { radiance = v1.rgb; hit = true; break; }
				if (v2.a > 0) { radiance = v2.rgb; hit = true; break; }
				if (v3.a > 0) { radiance = v3.rgb; hit = true; break; }

				// 12 independent float adds (advance all 4 positions by 4 steps)
				p0v += step4;
				p1v += step4;
				p2v += step4;
				p3v += step4;
			}
		}
#else
		// DDA path: exact voxel traversal via 4-wide unrolled DDA steps.
		// Visits every voxel the ray passes through (no misses), but the
		// axis-selection comparison chain (~18 ALU per step, 4-deep SGPR
		// dependency) dominates execution time on RDNA3.
		float3 abs_dir = abs(dir);
		bool   pos_x = dir.x >= 0, pos_y = dir.y >= 0, pos_z = dir.z >= 0;
		int3   step_dir = int3(pos_x ? 1 : -1, pos_y ? 1 : -1, pos_z ? 1 : -1);

		float3 t_delta;
		t_delta.x = abs_dir.x > 0.0001 ? GI_INV_VOXEL_RES / abs_dir.x : 1e30;
		t_delta.y = abs_dir.y > 0.0001 ? GI_INV_VOXEL_RES / abs_dir.y : 1e30;
		t_delta.z = abs_dir.z > 0.0001 ? GI_INV_VOXEL_RES / abs_dir.z : 1e30;

		for (uint cascade = 0; cascade < GI_CASCADE_COUNT && !hit; cascade++) {
			float3 origin = (probe_world - gi_cascades[cascade].volume_min) * gi_cascades[cascade].volume_inv;

			if (cascade > 0) {
				float3 t_lo  = (float3(0.25, 0.25, 0.25) - origin) / dir;
				float3 t_hi  = (float3(0.75, 0.75, 0.75) - origin) / dir;
				float3 t_far = max(t_lo, t_hi);
				float  t_skip = min(t_far.x, min(t_far.y, t_far.z));
				if (t_skip > 0) origin = origin + dir * (t_skip + 0.001);
			}

			if (any(origin < 0) || any(origin >= 1.0)) continue;

			int3 voxel_pos = clamp(int3(origin * (float)GI_VOXEL_RES),
			                       int3(0, 0, 0),
			                       int3(GI_VOXEL_RES - 1, GI_VOXEL_RES - 1, GI_VOXEL_RES - 1));

			float3 t_max;
			t_max.x = abs_dir.x > 0.0001 ? ((pos_x ? (voxel_pos.x + 1) : voxel_pos.x) * GI_INV_VOXEL_RES - origin.x) / dir.x : 1e30;
			t_max.y = abs_dir.y > 0.0001 ? ((pos_y ? (voxel_pos.y + 1) : voxel_pos.y) * GI_INV_VOXEL_RES - origin.y) / dir.y : 1e30;
			t_max.z = abs_dir.z > 0.0001 ? ((pos_z ? (voxel_pos.z + 1) : voxel_pos.z) * GI_INV_VOXEL_RES - origin.z) / dir.z : 1e30;

			uint z_offset = cascade * GI_VOXEL_RES;

			for (uint dda_step = 0; dda_step < GI_VOXEL_RES * 3u; dda_step += 4) {
				_dda_step(voxel_pos, t_max, t_delta, step_dir); int3 p0 = voxel_pos;
				_dda_step(voxel_pos, t_max, t_delta, step_dir); int3 p1 = voxel_pos;
				_dda_step(voxel_pos, t_max, t_delta, step_dir); int3 p2 = voxel_pos;
				_dda_step(voxel_pos, t_max, t_delta, step_dir); int3 p3 = voxel_pos;

				float4 v0 = voxel_tex.Load(int4(p0.x, p0.y, p0.z + z_offset, 0));
				float4 v1 = voxel_tex.Load(int4(p1.x, p1.y, p1.z + z_offset, 0));
				float4 v2 = voxel_tex.Load(int4(p2.x, p2.y, p2.z + z_offset, 0));
				float4 v3 = voxel_tex.Load(int4(p3.x, p3.y, p3.z + z_offset, 0));

				if (v0.a > 0) { radiance = v0.rgb; hit = true; break; }
				if (v1.a > 0) { radiance = v1.rgb; hit = true; break; }
				if (v2.a > 0) { radiance = v2.rgb; hit = true; break; }
				if (v3.a > 0) { radiance = v3.rgb; hit = true; break; }

				if (((uint)p3.x | (uint)p3.y | (uint)p3.z) >= GI_VOXEL_RES) break;
			}
		}
#endif

		// If ray exited all cascades without hitting geometry, sample environment
		if (!hit) {
			radiance = env_cubemap.SampleLevel(env_cubemap_s, dir, env_mip).rgb * env_strength;
		}

		radiance = min(radiance, MAX_RADIANCE);

		// SH projection for this ray's direction
		float4 sh = float4(
			0.28209 * w,
			0.48860 * dir.y * w,
			0.48860 * dir.z * w,
			0.48860 * dir.x * w
		);

		total_r += sh * radiance.r;
		total_g += sh * radiance.g;
		total_b += sh * radiance.b;
	}

	// Write SH: pass 0 applies EMA decay + accumulates, passes 1-3 just accumulate
	uint idx = voxel_index(pid);
	if (pass == 0) {
		sh_probes[idx].r = sh_pack(sh_unpack(sh_probes[idx].r) * sh_decay + total_r);
		sh_probes[idx].g = sh_pack(sh_unpack(sh_probes[idx].g) * sh_decay + total_g);
		sh_probes[idx].b = sh_pack(sh_unpack(sh_probes[idx].b) * sh_decay + total_b);
	} else {
		sh_probes[idx].r = sh_pack(sh_unpack(sh_probes[idx].r) + total_r);
		sh_probes[idx].g = sh_pack(sh_unpack(sh_probes[idx].g) + total_g);
		sh_probes[idx].b = sh_pack(sh_unpack(sh_probes[idx].b) + total_b);
	}
}
