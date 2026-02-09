//--name = orbital_particles

#include "common.hlsli"

// Rendering params
float3 color_slow;
float  max_speed;
float3 color_fast;
float  sim_time;
float  delta_time;
float  damping;
float  strength;
float  _pad3;

struct Particle {
	float3 position;
	float3 velocity;
};
StructuredBuffer  <Particle> particles     : register(t3, space0);
RWStructuredBuffer<Particle> particles_out : register(u2, space0);

struct psIn {
	float4 pos   : SV_POSITION;
	float3 color : COLOR0;
	uint   layer : SV_RenderTargetArrayIndex;
};

psIn vs(uint vertex_id : SV_VertexID, uint view_idx : SV_InstanceID) {
	const float size = 0.01;

	uint particle_idx = vertex_id / 6;
	uint corner       = vertex_id % 6;

	Particle p = particles[particle_idx];

	// Simulate on first vertex of first view only
	if (corner == 0 && view_idx == 0) {
		// Moving attractors
		float3 attractors[3];
		attractors[0] = float3(cos(sim_time * 0.5) * 2.0, sin(sim_time * 0.7) * 1.5, sin(sim_time * 0.5) * 2.0);
		attractors[1] = float3(sin(sim_time * 0.6) * 2.5, cos(sim_time * 0.4) * 2.0, cos(sim_time * 0.6) * 1.5);
		attractors[2] = float3(cos(sim_time * 0.4) * 1.5, sin(sim_time * 0.8) * 2.5, sin(sim_time * 0.3) * 2.0);

		// Find nearest 2 attractors
		float dist1 = 1e10, dist2 = 1e10;
		int   idx1  = 0,    idx2  = 1;
		for (int a = 0; a < 3; a++) {
			float dist = length(attractors[a] - p.position);
			if (dist < dist1) {
				dist2 = dist1; idx2 = idx1;
				dist1 = dist;  idx1 = a;
			} else if (dist < dist2) {
				dist2 = dist;  idx2 = a;
			}
		}

		// Gravitational force from nearest 2
		float3 force = float3(0, 0, 0);
		for (int a = 0; a < 2; a++) {
			int    idx     = (a == 0) ? idx1 : idx2;
			float3 diff    = attractors[idx] - p.position;
			float  dist_sq = dot(diff, diff) + 0.1;
			float  dist    = sqrt(dist_sq);
			force += (diff / dist) * (strength / dist_sq);
		}

		// Integrate velocity and position
		p.velocity += force * delta_time;
		p.velocity *= damping;

		float speed = length(p.velocity);
		if (speed > max_speed)
			p.velocity *= max_speed / speed;

		p.position += p.velocity * delta_time;

		particles_out[particle_idx] = p;
	}

	// Billboard quad: 2 triangles, 6 vertices
	float2 offsets[6] = {
		float2(-1, -1), // tri 0
		float2( 1, -1),
		float2( 1,  1),
		float2(-1, -1), // tri 1
		float2( 1,  1),
		float2(-1,  1),
	};

	// Transform center to view space, expand there (+X = right, +Y = up), then project
	float4 view_pos = mul(float4(p.position, 1), view[view_idx]);
	view_pos.xy += offsets[corner] * size;

	// Color based on particle speed
	float  speed_val  = length(p.velocity);
	float  speed_t    = saturate(speed_val / max_speed) * 2;
	float3 base_color = lerp(color_slow, color_fast, speed_t);

	psIn output;
	output.pos   = mul(view_pos, projection[view_idx]);
	output.color = base_color;
	output.layer = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	return float4(input.color, 1);
}
