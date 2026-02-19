//--name = gi_voxel_to_sh

// Compute shader to accumulate SH probes from voxel radiance via ray marching.
// Each frame, casts Fibonacci sphere rays per probe through the 3D voxel texture.
// First opaque voxel hit provides radiance; rays exiting the grid sample the
// environment cubemap. Results project into SH via sliding window average.
//
// Accumulation: ring buffer of per-frame SH snapshots. Each frame writes one
// slot and recomputes the average from all history entries. Each frame has
// equal 1/N weight with bounded error (no EMA tail).
//
// Directions: Fibonacci sphere with temporal stratification. Total pool =
// history_size * ray_count directions. Each frame samples a different stratum
// so over one full window every direction is sampled exactly once. Per-probe
// angular offset decorrelates neighboring probes.
//
// Voxel data stored as RGB10A2 in a 3D texture. Occupancy: alpha > 0.
// Cascaded: rays march through cascade 0 first, then 1, then 2 (fine-to-coarse).

#include "gi_voxel.hlsli"

Texture3D<float4>               voxel_tex     : register(t0);
TextureCube<float4>              env_cubemap   : register(t1);
SamplerState                     env_cubemap_s : register(s1);
RWStructuredBuffer<SHProbe>      sh_probes     : register(u0);
RWStructuredBuffer<SHProbe>      sh_history    : register(u1);

uint  grid_size;
uint  frame_seed;
uint  history_index; // = frame_seed % history_size
uint  history_size;  // sliding window size (default 32)
uint  ray_count;     // total rays per probe per frame
float env_mip;       // cubemap mip level for environment fallback (higher = blurrier)
float env_strength;  // multiplier for environment cubemap contribution

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

// Fibonacci sphere with temporal stratification and per-probe offset.
//
// Total direction pool = history_size * ray_count. Each frame samples a
// different stratum (stride = history_size), so over one full window every
// direction is sampled exactly once. Per-probe angular offset decorrelates
// neighboring probes so they don't flicker in unison.
#define GOLDEN_RATIO 1.6180339887

float3 _fibonacci_dir(uint index, uint total, float offset) {
	float i = (float)index + 0.5;
	float n = (float)total;

	// Cylindrical equal-area: z linearly spaced, phi by golden ratio + offset
	float z   = 1.0 - 2.0 * i / n;
	float r   = sqrt(max(0.0, 1.0 - z * z));
	float phi = 6.28318530718 * frac(i * (1.0 / GOLDEN_RATIO) + offset);
	return float3(r * cos(phi), r * sin(phi), z);
}

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	// One thread per probe. All rays computed in registers, single history write.
	uint3 pid = id;

	// Probe world position (computed from cascade 0)
	float3 vol_size_0  = 1.0 / gi_cascades[0].volume_inv;
	float3 probe_world = gi_cascades[0].volume_min + (float3(pid) + 0.5) * GI_INV_GRID * vol_size_0;

	// Weight per ray: full sphere sampling (4*pi), equal weight per frame
	float w = 12.56637 / (float)ray_count;

	// Per-probe angular offset: hash probe position to rotate the Fibonacci
	// spiral uniquely per probe. Decorrelates neighboring probes so they don't
	// flicker in unison. Deterministic — same probe always gets the same offset.
	uint  probe_hash  = _hash(pid.x + _hash(pid.y + _hash(pid.z)));
	float probe_offset = float(probe_hash) / 4294967295.0;

	// Temporal stratification: total pool = history_size * ray_count directions.
	// This frame samples indices {history_index, history_index + history_size, ...}
	// so over one full window, every direction in the pool is sampled exactly once.
	uint ray_total = history_size * ray_count;

	float4 total_r = float4(0, 0, 0, 0);
	float4 total_g = float4(0, 0, 0, 0);
	float4 total_b = float4(0, 0, 0, 0);

	for (uint ray = 0; ray < ray_count; ray++) {
		uint ray_idx = history_index + ray * history_size;
		float3 dir = _fibonacci_dir(ray_idx, ray_total, probe_offset);

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

	// Write this frame's SH to the ring buffer
	uint idx       = voxel_index(pid);
	uint hist_slot = idx * history_size + history_index;
	SHProbe new_entry;
	new_entry.r = sh_pack(total_r);
	new_entry.g = sh_pack(total_g);
	new_entry.b = sh_pack(total_b);
	sh_history[hist_slot] = new_entry;

	// Recompute average from all history entries
	float4 sum_r = float4(0, 0, 0, 0);
	float4 sum_g = float4(0, 0, 0, 0);
	float4 sum_b = float4(0, 0, 0, 0);
	uint base = idx * history_size;
	for (uint i = 0; i < history_size; i++) {
		SHProbe entry = sh_history[base + i];
		sum_r += sh_unpack(entry.r);
		sum_g += sh_unpack(entry.g);
		sum_b += sh_unpack(entry.b);
	}
	float inv_n = 1.0 / (float)history_size;
	sh_probes[idx].r = sh_pack(sum_r * inv_n);
	sh_probes[idx].g = sh_pack(sum_g * inv_n);
	sh_probes[idx].b = sh_pack(sum_b * inv_n);
}
