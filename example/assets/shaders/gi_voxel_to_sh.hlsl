//--name = gi_voxel_to_sh

// Compute shader to accumulate SH probes from voxel radiance via ray marching.
// Each frame, casts random rays per probe through the 3D voxel texture.
// First opaque voxel hit provides radiance; rays exiting the grid sample the
// environment cubemap. Results project into SH via exponential moving average.
//
// Voxel data stored as RGB10A2 in a 3D texture. Occupancy: alpha > 0.

#include "gi_voxel.hlsli"

Texture3D<float4>               voxel_tex     : register(t0);
TextureCube<float4>              env_cubemap   : register(t1);
SamplerState                     env_cubemap_s : register(s1);
RWStructuredBuffer<SHProbe>      sh_probes     : register(u0);

uint  grid_size;
uint  frame_seed;
float sh_decay;    // EMA decay (0.98 = ~50 frame window, 0.99 = ~100)
uint  ray_count;   // total rays per probe per frame (>= 4, split across 4 passes)
float env_mip;     // cubemap mip level for environment fallback (higher = blurrier)
float env_strength; // multiplier for environment cubemap contribution

#define MAX_RADIANCE  4.0

#define RAY_MARCH_DDA

// Single DDA step: advances voxel_pos and t_max along the nearest axis.
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

// Integer hash for per-probe random direction
uint _hash(uint x) {
	x ^= x >> 16;
	x *= 0x45d9f3bu;
	x ^= x >> 16;
	x *= 0x45d9f3bu;
	x ^= x >> 16;
	return x;
}

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	// Multi-pass dispatch: Z dimension is 4x grid size. Each pass touches every
	// probe with ray_count/4 rays. Pass 0 applies EMA decay + accumulates;
	// passes 1-3 just accumulate.
	uint  pass = id.z / GI_GRID;
	uint3 pid  = uint3(id.x, id.y, id.z % GI_GRID);

	uint probe_hash = _hash(pid.x + _hash(pid.y + _hash(pid.z)));

	float3 origin = (float3(pid) + 0.5) * GI_INV_GRID;

	// Weight per ray: full sphere sampling (4*pi)
	float solid_angle = 12.56637; // 4*pi
	float w           = solid_angle * (1.0 - sh_decay) / (float)ray_count;

	// Each pass does a quarter of the rays
	uint rays_per_pass = ray_count >> 2;
	uint ray_base      = pass * rays_per_pass;

	float4 total_r = float4(0, 0, 0, 0);
	float4 total_g = float4(0, 0, 0, 0);
	float4 total_b = float4(0, 0, 0, 0);

	for (uint ray = 0; ray < rays_per_pass; ray++) {
		// White noise: hash(probe, frame, ray_idx) for fully random directions
		uint ray_idx = ray_base + ray;
		uint h   = _hash(probe_hash + _hash(frame_seed * ray_count + ray_idx));
		float u1 = float(h) / 4294967295.0;
		float u2 = float(_hash(h)) / 4294967295.0;

		float z   = 1.0 - 2.0 * u1;
		float r   = sqrt(max(0.0, 1.0 - z * z));
		float phi = 6.28318530718 * u2;
		float3 dir = float3(r * cos(phi), r * sin(phi), z);

		// Ray march through voxel volume: first hit wins
		float3 radiance = float3(0, 0, 0);
		bool   hit      = false;

#ifdef RAY_MARCH_DDA
		// DDA with linear index stepping.
		float3 abs_dir = abs(dir);
		bool pos_x = dir.x >= 0;
		bool pos_y = dir.y >= 0;
		bool pos_z = dir.z >= 0;
		int3 step_dir = int3(pos_x ? 1 : -1, pos_y ? 1 : -1, pos_z ? 1 : -1);

		int3 voxel_pos = int3(origin * (float)GI_VOXEL_RES);

		// Distance to next voxel boundary along each axis (in t units)
		float3 t_max;
		t_max.x = abs_dir.x > 0.0001 ? ((pos_x ? (voxel_pos.x + 1) : voxel_pos.x) * GI_INV_VOXEL_RES - origin.x) / dir.x : 1e30;
		t_max.y = abs_dir.y > 0.0001 ? ((pos_y ? (voxel_pos.y + 1) : voxel_pos.y) * GI_INV_VOXEL_RES - origin.y) / dir.y : 1e30;
		t_max.z = abs_dir.z > 0.0001 ? ((pos_z ? (voxel_pos.z + 1) : voxel_pos.z) * GI_INV_VOXEL_RES - origin.z) / dir.z : 1e30;

		float3 t_delta;
		t_delta.x = abs_dir.x > 0.0001 ? GI_INV_VOXEL_RES / abs_dir.x : 1e30;
		t_delta.y = abs_dir.y > 0.0001 ? GI_INV_VOXEL_RES / abs_dir.y : 1e30;
		t_delta.z = abs_dir.z > 0.0001 ? GI_INV_VOXEL_RES / abs_dir.z : 1e30;

		for (uint dda_step = 0; dda_step < GI_VOXEL_RES * 3u; dda_step += 4) {
			// Advance 4 DDA steps (pure ALU, no memory access)
			_dda_step(voxel_pos, t_max, t_delta, step_dir); int3 p0 = voxel_pos;
			_dda_step(voxel_pos, t_max, t_delta, step_dir); int3 p1 = voxel_pos;
			_dda_step(voxel_pos, t_max, t_delta, step_dir); int3 p2 = voxel_pos;
			_dda_step(voxel_pos, t_max, t_delta, step_dir); int3 p3 = voxel_pos;

			// Issue 4 loads back-to-back (texture unit can prefetch)
			float4 v0 = voxel_tex.Load(int4(p0, 0));
			float4 v1 = voxel_tex.Load(int4(p1, 0));
			float4 v2 = voxel_tex.Load(int4(p2, 0));
			float4 v3 = voxel_tex.Load(int4(p3, 0));

			// First hit wins (OOB loads return 0, so alpha check handles bounds)
			if (v0.a > 0) { radiance = v0.rgb; hit = true; break; }
			if (v1.a > 0) { radiance = v1.rgb; hit = true; break; }
			if (v2.a > 0) { radiance = v2.rgb; hit = true; break; }
			if (v3.a > 0) { radiance = v3.rgb; hit = true; break; }

			if (((uint)p3.x | (uint)p3.y | (uint)p3.z) >= GI_VOXEL_RES) break;
		}
#else
		for (int s = 1; s < GI_VOXEL_RES; s++) {
			float3 pos = origin + dir * (s * GI_INV_VOXEL_RES);

			if (pos.x < 0 || pos.x >= 1.0 ||
			    pos.y < 0 || pos.y >= 1.0 ||
			    pos.z < 0 || pos.z >= 1.0) break;

			int3 texel_pos = clamp(int3(pos * (float)GI_VOXEL_RES), int3(0,0,0),
			                       int3(GI_VOXEL_RES-1, GI_VOXEL_RES-1, GI_VOXEL_RES-1));

			float4 voxel = voxel_tex.Load(int4(texel_pos, 0));
			if (voxel.a > 0) {
				radiance = voxel.rgb;
				hit = true;
				break;
			}
		}
#endif

		// If ray exited the grid without hitting geometry, sample environment
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
