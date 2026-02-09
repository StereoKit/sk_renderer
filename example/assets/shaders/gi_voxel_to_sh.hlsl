//--name = gi_voxel_to_sh

// Compute shader to accumulate SH probes from voxel radiance via ray marching.
// Each frame, casts low-discrepancy rays per probe through the voxel grid.
// First opaque voxel hit provides radiance; rays exiting the grid sample the
// environment cubemap at a low mip. Results project into SH via exponential
// moving average. Hammersley-like sampling with per-probe Cranley-Patterson
// rotation gives fast, uniform sphere coverage with minimal fireflies.
//
// Backface culling uses a 6-bit face mask packed in voxel alpha (set during
// voxelization from capture direction). Bit encoding:
//   bit 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
// A ray is blocked only if the voxel has a face on the side the ray enters.
// Rays from surface probes that would exit through local geometry are skipped.

Texture3D<float4>   voxel_tex    : register(t0);
SamplerState        voxel_tex_s  : register(s0);
TextureCube<float4> env_cubemap  : register(t1);
SamplerState        env_cubemap_s: register(s1);
RWTexture3D<float4> sh_r         : register(u0);
RWTexture3D<float4> sh_g         : register(u1);
RWTexture3D<float4> sh_b         : register(u2);

uint  grid_size;
uint  frame_seed;
float sh_decay;    // EMA decay (0.98 = ~50 frame window, 0.99 = ~100)
uint  ray_count;   // rays per probe per frame (1-8)
float env_mip;     // cubemap mip level for environment fallback (higher = blurrier)
float env_strength; // multiplier for environment cubemap contribution

#define MAX_RADIANCE  4.0
#define GOLDEN_RATIO  0.6180339887

// Integer hash for per-probe random offset
uint _hash(uint x) {
	x ^= x >> 16;
	x *= 0x45d9f3bu;
	x ^= x >> 16;
	x *= 0x45d9f3bu;
	x ^= x >> 16;
	return x;
}

// Van der Corput radical inverse (bit reversal) for low-discrepancy sampling.
// Pure bitwise ops — very cheap.
float _radical_inverse(uint bits) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10;
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
	if (id.x >= grid_size || id.y >= grid_size || id.z >= grid_size) return;

	// Per-probe random offset (Cranley-Patterson rotation for decorrelation)
	uint  probe_hash = _hash(id.x + _hash(id.y + _hash(id.z)));
	float offset1    = float(probe_hash) / 4294967295.0;
	float offset2    = float(_hash(probe_hash)) / 4294967295.0;

	float  inv_grid = 1.0 / (float)grid_size;
	float3 origin   = (float3(id) + 0.5) * inv_grid;

	// Weight per ray: 4*pi*(1-decay)/ray_count for correct steady-state integral
	float w = 12.56637 * (1.0 - sh_decay) / (float)ray_count;

	// Load origin voxel face mask once per probe (for exit test)
	uint origin_face_mask = uint(voxel_tex.Load(int4(id, 0)).a + 0.5);

	float4 total_r = float4(0, 0, 0, 0);
	float4 total_g = float4(0, 0, 0, 0);
	float4 total_b = float4(0, 0, 0, 0);

	for (uint ray = 0; ray < ray_count; ray++) {
		uint sample_idx = frame_seed * ray_count + ray;

		// Low-discrepancy direction: radical inverse + golden ratio, per-probe rotated
		float u1 = frac(_radical_inverse(sample_idx) + offset1);
		float u2 = frac(float(sample_idx) * GOLDEN_RATIO + offset2);

		float z   = 1.0 - 2.0 * u1;
		float r   = sqrt(max(0.0, 1.0 - z * z));
		float phi = 6.28318530718 * u2;
		float3 dir = float3(r * cos(phi), r * sin(phi), z);

		// Precompute which faces this ray would enter through
		uint entry_mask = _ray_entry_mask(dir);

		// Skip rays that go behind local geometry: if the origin voxel has a
		// face matching the ray's entry direction, the ray is heading into the
		// solid side of the surface.
		if (origin_face_mask != 0 && (origin_face_mask & entry_mask) != 0)
			continue;

		// Ray march through voxel volume: first hit wins
		float3 radiance = float3(0, 0, 0);
		bool   hit      = false;

		for (int s = 1; s < (int)grid_size; s++) {
			float3 pos = origin + dir * (s * inv_grid);

			if (pos.x < 0 || pos.x >= 1.0 ||
			    pos.y < 0 || pos.y >= 1.0 ||
			    pos.z < 0 || pos.z >= 1.0) break;

			// Use Load() for integer-exact face mask from alpha
			int3 texel_pos = clamp(int3(pos * (float)grid_size), int3(0,0,0),
			                       int3(grid_size-1, grid_size-1, grid_size-1));
			float4 voxel = voxel_tex.Load(int4(texel_pos, 0));

			uint face_mask = uint(voxel.a + 0.5);
			if (face_mask == 0) continue; // empty voxel
			if ((face_mask & entry_mask) == 0) continue; // no face on ray's entry side

			uint count = countbits(face_mask);
			radiance   = voxel.rgb / (float)count;
			hit        = true;
			break;
		}

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

	// Exponential moving average (single decay per frame, both rays summed)
	sh_r[id] = sh_r[id] * sh_decay + total_r;
	sh_g[id] = sh_g[id] * sh_decay + total_g;
	sh_b[id] = sh_b[id] * sh_decay + total_b;
}
