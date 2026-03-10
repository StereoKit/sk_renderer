//--name = texture3d_raymarch

#include "common.hlsli"

struct Inst {
	float4x4 world;
	float4x4 world_inv;
};
StructuredBuffer<Inst> inst : register(t2, space0);

struct vsIn {
	float3 pos  : SV_POSITION;
	float3 norm : NORMAL;
	float2 uv   : TEXCOORD0;
	float4 color: COLOR0;
};
struct psIn {
	float4 pos       : SV_POSITION;
	float3 local_pos : TEXCOORD0;
	float3 cam_local : TEXCOORD1;
};

Texture3D    tex         : register(t0);
SamplerState tex_sampler : register(s0);

psIn vs(vsIn input, skr_ids_t ids) {
	float4x4 world     = inst[ids.inst].world;
	float4x4 world_inv = inst[ids.inst].world_inv;

	// Transform camera position to local space using precomputed inverse
	float3 cam_world = cam_pos[ids.view].xyz;
	float3 cam_local = mul(float4(cam_world, 1), world_inv).xyz;

	psIn output;
	float4 world_position = mul(float4(input.pos, 1), world);
	output.pos       = mul(world_position, viewproj[ids.view]);
	output.local_pos = input.pos;
	output.cam_local = cam_local;
	return output;
}

// Ray-box intersection for unit cube centered at origin
// Returns (tmin, tmax) - entry and exit distances along ray
float2 ray_box_intersect(float3 ray_origin, float3 ray_dir, float3 box_min, float3 box_max) {
	float3 inv_dir = 1.0 / ray_dir;
	float3 t0      = (box_min - ray_origin) * inv_dir;
	float3 t1      = (box_max - ray_origin) * inv_dir;
	float3 tmin3   = min(t0, t1);
	float3 tmax3   = max(t0, t1);
	float  tmin    = max(max(tmin3.x, tmin3.y), tmin3.z);
	float  tmax    = min(min(tmax3.x, tmax3.y), tmax3.z);
	return float2(tmin, tmax);
}

float4 ps(psIn input) : SV_TARGET {
	// Ray direction computed per-pixel (from camera through this fragment)
	float3 ray_dir = normalize(input.local_pos - input.cam_local);

	// Ray-box intersection for the cube (local space, cube is -0.5 to 0.5)
	float3 box_min = float3(-0.5, -0.5, -0.5);
	float3 box_max = float3( 0.5,  0.5,  0.5);
	float2 t       = ray_box_intersect(input.cam_local, ray_dir, box_min, box_max);

	// Clamp to start at current fragment position (handles camera inside cube)
	float t_start = max(t.x, 0.0);
	float t_end   = t.y;

	if (t_end <= t_start) discard;

	// Raymarching parameters
	const int   MAX_STEPS = 64;
	const float step_size = (t_end - t_start) / float(MAX_STEPS);

	// Front-to-back compositing
	float4 accum   = float4(0, 0, 0, 0);
	float3 ray_pos = input.cam_local + ray_dir * t_start;

	for (int i = 0; i < MAX_STEPS && accum.a < 0.95; i++) {
		// Convert local position (-0.5 to 0.5) to UVW (0 to 1)
		float3 uvw = ray_pos + 0.5;

		// Sample the 3D texture
		float4 sample_color = tex.SampleLevel(tex_sampler, uvw, 0);

		// Front-to-back compositing with step-adjusted alpha
		float  alpha = sample_color.a * step_size * 4.0;
		float3 color = sample_color.rgb;
		accum.rgb   += (1.0 - accum.a) * alpha * color;
		accum.a     += (1.0 - accum.a) * alpha;

		ray_pos += ray_dir * step_size;
	}

	if (accum.a < 0.01) discard;

	return float4(accum.rgb, accum.a);
}
