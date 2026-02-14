//--name = gi_voxel_to_sh

// Compute shader to accumulate SH probes from voxel radiance via ray marching.
// Each frame, casts low-discrepancy rays per probe through the voxel grid.
// First opaque voxel hit provides radiance from the specific face the ray enters;
// rays exiting the grid sample the environment cubemap at a low mip. Results
// project into SH via exponential moving average. White noise with per-probe
// hash gives uniform sphere coverage.
//
// Per-face color stored as RGB5A1 in a StructuredBuffer<Voxel> (6 faces per voxel).
// Occupancy derived from alpha bits (A=1 means occupied).
// A ray samples the color of the face it enters through (dominant axis).

#include "gi_voxel.hlsli"

StructuredBuffer<Voxel>      voxel_buf     : register(t0);
TextureCube<float4>          env_cubemap   : register(t1);
SamplerState                 env_cubemap_s : register(s1);
RWStructuredBuffer<SHProbe>  sh_probes     : register(u0);

uint  grid_size;
uint  frame_seed;
float sh_decay;    // EMA decay (0.98 = ~50 frame window, 0.99 = ~100)
uint  ray_count;   // total rays per probe per frame (>= 4, split across 4 passes)
float env_mip;     // cubemap mip level for environment fallback (higher = blurrier)
float env_strength; // multiplier for environment cubemap contribution

#define MAX_RADIANCE  4.0

// Uncomment to use DDA traversal instead of fixed-step ray march.
// DDA visits every voxel the ray passes through (no skipping), using linear
// index stride adds (~2 ALU per step). Fixed-step recomputes voxel_index each
// step (~5 ALU) but takes fewer steps on diagonal rays.
#define RAY_MARCH_DDA

// Integer hash for per-probe random direction
uint _hash(uint x) {
	x ^= x >> 16;
	x *= 0x45d9f3bu;
	x ^= x >> 16;
	x *= 0x45d9f3bu;
	x ^= x >> 16;
	return x;
}

// Build a 6-bit mask of which voxel faces a ray would enter through,
// based on ray direction. Computed once per ray, used for all steps.
uint _ray_entry_mask(float3 dir) {
	uint m = 0;
	if (dir.x > 0) m |= 2u;  // ray going +X enters through -X face
	if (dir.x < 0) m |= 1u;  // ray going -X enters through +X face
	if (dir.y > 0) m |= 8u;  // ray going +Y enters through -Y face
	if (dir.y < 0) m |= 4u;  // ray going -Y enters through +Y face
	if (dir.z > 0) m |= 32u; // ray going +Z enters through -Z face
	if (dir.z < 0) m |= 16u; // ray going -Z enters through +Z face
	return m;
}

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	// Multi-pass dispatch: Z dimension is 4x grid size. Each pass touches every
	// probe with ray_count/4 rays. Pass 0 applies EMA decay + accumulates;
	// passes 1-3 just accumulate. Thread group scheduling ensures each pass
	// completes before the next starts (512 groups of separation per pass).
	uint  pass = id.z >> 5;                          // 0-3
	uint3 pid  = uint3(id.x, id.y, id.z & (GI_GRID - 1)); // probe coordinates

	uint probe_hash = _hash(pid.x + _hash(pid.y + _hash(pid.z)));

	float3 origin = (float3(pid) + 0.5) * GI_INV_GRID;

	// Load origin voxel and derive face mask from alpha bits
	uint  origin_lidx      = voxel_index(pid);
	Voxel origin_voxel     = voxel_buf[origin_lidx];
	uint  origin_face_mask = voxel_face_mask(origin_voxel);

	// If the origin voxel has a clear outward normal, sample only that
	// hemisphere. Otherwise (empty or opposing faces), sample full sphere.
	float3 face_normal     = voxel_face_normal(origin_face_mask);
	float  normal_len2     = dot(face_normal, face_normal);
	bool   use_hemisphere  = normal_len2 > 0.01;
	if (use_hemisphere) face_normal *= rsqrt(normal_len2);

	// Weight per ray: uses full ray_count so both passes sum to correct total
	float solid_angle = use_hemisphere ? 6.28318 : 12.56637; // 2*pi or 4*pi
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

		// Flip ray into outward hemisphere if origin has a clear surface normal
		if (use_hemisphere && dot(dir, face_normal) < 0)
			dir = -dir;

		// Precompute which faces this ray would enter through
		uint entry_mask = _ray_entry_mask(dir);

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

		// Linear index strides per axis (constant for grid=32)
		int stride_x = pos_x ? 1 : -1;
		int stride_y = pos_y ? GI_GRID : -GI_GRID;
		int stride_z = pos_z ? GI_GRID2 : -GI_GRID2;

		int3 voxel_pos = int3(pid);
		uint lidx      = origin_lidx;

		// Distance to next voxel boundary along each axis (in t units)
		float3 t_max;
		t_max.x = abs_dir.x > 0.0001 ? ((pos_x ? (voxel_pos.x + 1) : voxel_pos.x) * GI_INV_GRID - origin.x) / dir.x : 1e30;
		t_max.y = abs_dir.y > 0.0001 ? ((pos_y ? (voxel_pos.y + 1) : voxel_pos.y) * GI_INV_GRID - origin.y) / dir.y : 1e30;
		t_max.z = abs_dir.z > 0.0001 ? ((pos_z ? (voxel_pos.z + 1) : voxel_pos.z) * GI_INV_GRID - origin.z) / dir.z : 1e30;

		float3 t_delta;
		t_delta.x = abs_dir.x > 0.0001 ? GI_INV_GRID / abs_dir.x : 1e30;
		t_delta.y = abs_dir.y > 0.0001 ? GI_INV_GRID / abs_dir.y : 1e30;
		t_delta.z = abs_dir.z > 0.0001 ? GI_INV_GRID / abs_dir.z : 1e30;

		for (uint dda_step = 0; dda_step < GI_GRID * 3u; dda_step++) {
			// Axis selection (branchless -> v_cmp + v_cndmask)
			bool sel_x = (t_max.x <= t_max.y) && (t_max.x <= t_max.z);
			bool sel_y = !sel_x && (t_max.y <= t_max.z);

			// Linear index step: one v_cndmask + one v_add
			lidx += (uint)(sel_x ? stride_x : (sel_y ? stride_y : stride_z));

			// Branchless position + t_max update
			voxel_pos += int3(sel_x ? step_dir.x : 0,
			                  sel_y ? step_dir.y : 0,
			                  (!sel_x && !sel_y) ? step_dir.z : 0);
			t_max += float3(sel_x ? t_delta.x : 0.0,
			                sel_y ? t_delta.y : 0.0,
			                (!sel_x && !sel_y) ? t_delta.z : 0.0);

			// Bounds check: negative ints cast to huge uints (bit 31), values
			// >= 32 have bit 5+. OR combines, one compare catches all 6 cases.
			if (((uint)voxel_pos.x | (uint)voxel_pos.y | (uint)voxel_pos.z) >= GI_GRID) break;

			// Load voxel and derive face mask from alpha bits
			Voxel v = voxel_buf[lidx];
			uint face_mask = voxel_face_mask(v);
			if (face_mask == 0) continue;

			if ((face_mask & entry_mask) == 0) {
				hit = true;
				break;
			}

			uint face_idx = entry_mask_to_face(entry_mask, dir);
			if (!(face_mask & (1u << face_idx))) {
				uint matching = face_mask & entry_mask;
				face_idx = firstbitlow(matching);
			}
			radiance = unpack_rgba8_color(voxel_get_face(v, face_idx));
			hit = true;
			break;
		}
#else
		// Fixed-step march: step one voxel-width along the ray direction.
		// Recomputes voxel_index each step (~5 ALU) but takes at most
		// grid_size steps even on diagonal rays.
		for (int s = 1; s < GI_GRID; s++) {
			float3 pos = origin + dir * (s * GI_INV_GRID);

			if (pos.x < 0 || pos.x >= 1.0 ||
			    pos.y < 0 || pos.y >= 1.0 ||
			    pos.z < 0 || pos.z >= 1.0) break;

			int3 texel_pos = clamp(int3(pos * (float)GI_GRID), int3(0,0,0),
			                       int3(GI_GRID-1, GI_GRID-1, GI_GRID-1));
			uint idx = voxel_index(uint3(texel_pos));

			Voxel v = voxel_buf[idx];
			uint face_mask = voxel_face_mask(v);
			if (face_mask == 0) continue;

			if ((face_mask & entry_mask) == 0) {
				hit = true;
				break;
			}

			uint face_idx = entry_mask_to_face(entry_mask, dir);
			if (!(face_mask & (1u << face_idx))) {
				uint matching = face_mask & entry_mask;
				face_idx = firstbitlow(matching);
			}
			radiance = unpack_rgba8_color(voxel_get_face(v, face_idx));
			hit = true;
			break;
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

	// Write SH: pass 0 applies EMA decay + accumulates, pass 1 just accumulates
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
