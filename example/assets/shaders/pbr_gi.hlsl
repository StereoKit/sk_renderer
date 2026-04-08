//--name = pbr_gi

#include "common.hlsli"
#include "gi_voxel.hlsli"

///////////////////////////////////////////
// Material Parameters
///////////////////////////////////////////

//--color:color           = 1,1,1,1
//--emission_factor:color = 0,0,0,0
//--tex_trans             = 0,0,1,1

float4 color;
float4 emission_factor;
float4 tex_trans;

///////////////////////////////////////////
// Instance Data
///////////////////////////////////////////

struct Inst {
	float4x4 world;
};
 StructuredBuffer<Inst> inst : register(t2);

///////////////////////////////////////////
// Shadow Buffer (b13)
///////////////////////////////////////////

cbuffer ShadowBuffer : register(b13, space0) {
	float4x4 shadow_transform;
	float3   light_direction;
	float    shadow_bias;
	float3   light_color;
	float    shadow_pixel_size;
};

///////////////////////////////////////////
// Textures
///////////////////////////////////////////

Texture2D    albedo_tex   : register(t1);
SamplerState albedo_tex_s : register(s1);

Texture2D    emission_tex   : register(t0);
SamplerState emission_tex_s : register(s0);

// GI SH probe buffer (one SHProbe per grid cell)
StructuredBuffer<SHProbe> gi_sh_probes : register(t6);

// Shadow map (t14/s14)
Texture2D              shadow_map         : register(t14);
SamplerComparisonState shadow_map_sampler : register(s14);

///////////////////////////////////////////
// Vertex/Pixel Shader I/O
///////////////////////////////////////////

struct vsIn {
	float3 pos   : SV_POSITION;
	float3 norm  : NORMAL0;
	float2 uv    : TEXCOORD0;
	float4 color : COLOR0;
};

struct psIn {
	float4 pos       : SV_POSITION;
	float3 normal    : NORMAL0;
	float2 uv        : TEXCOORD0;
	float4 color     : COLOR0;
	float3 world_pos : TEXCOORD1;
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, skr_ids_t ids) {
	psIn output;

	float3 world_pos = mul(float4(input.pos, 1), inst[ids.inst].world).xyz;
	float3 normal    = normalize(mul(float4(input.norm, 0), inst[ids.inst].world).xyz);
	output.pos       = mul(float4(world_pos, 1), viewproj[ids.view]);
	output.normal    = normal;
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;
	output.world_pos = world_pos;

	return output;
}

///////////////////////////////////////////
// Shadow Sampling
///////////////////////////////////////////

float shadow_factor_pcf4(float3 uv, float scale) {
	float r = shadow_pixel_size * scale * 0.5;
	return (
		shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + float2(-r, -r), uv.z) +
		shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + float2( r, -r), uv.z) +
		shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + float2(-r,  r), uv.z) +
		shadow_map.SampleCmpLevelZero(shadow_map_sampler, uv.xy + float2( r,  r), uv.z)
	) * 0.25;
}

///////////////////////////////////////////
// GI Probe Sampling (tetrahedral interpolation)
///////////////////////////////////////////

void gi_sample_probes(float3 world_pos, uint cascade, out float4 shr, out float4 shg, out float4 shb) {
	float3 uvw      = saturate((world_pos - gi_cascades[cascade].volume_min) * gi_cascades[cascade].volume_inv);
	float3 grid_pos = uvw * (float)GI_GRID - 0.5;
	int3   base     = int3(floor(grid_pos));
	float3 f        = grid_pos - float3(base);
	int3   max_idx  = int3(GI_GRID - 1, GI_GRID - 1, GI_GRID - 1);

	uint3 p0 = uint3(clamp(base,     int3(0, 0, 0), max_idx));
	uint3 p1 = uint3(clamp(base + 1, int3(0, 0, 0), max_idx));

	uint  idx0 = probe_index_scrolled(p0, cascade);
	uint  idx3 = probe_index_scrolled(p1, cascade);
	uint  idx1, idx2;
	float tw0, tw1, tw2, tw3;

	if (f.x >= f.y) {
		if (f.y >= f.z) {
			idx1 = probe_index_scrolled(uint3(p1.x, p0.y, p0.z), cascade);
			idx2 = probe_index_scrolled(uint3(p1.x, p1.y, p0.z), cascade);
			tw0 = 1-f.x; tw1 = f.x-f.y; tw2 = f.y-f.z; tw3 = f.z;
		} else if (f.x >= f.z) {
			idx1 = probe_index_scrolled(uint3(p1.x, p0.y, p0.z), cascade);
			idx2 = probe_index_scrolled(uint3(p1.x, p0.y, p1.z), cascade);
			tw0 = 1-f.x; tw1 = f.x-f.z; tw2 = f.z-f.y; tw3 = f.y;
		} else {
			idx1 = probe_index_scrolled(uint3(p0.x, p0.y, p1.z), cascade);
			idx2 = probe_index_scrolled(uint3(p1.x, p0.y, p1.z), cascade);
			tw0 = 1-f.z; tw1 = f.z-f.x; tw2 = f.x-f.y; tw3 = f.y;
		}
	} else {
		if (f.x >= f.z) {
			idx1 = probe_index_scrolled(uint3(p0.x, p1.y, p0.z), cascade);
			idx2 = probe_index_scrolled(uint3(p1.x, p1.y, p0.z), cascade);
			tw0 = 1-f.y; tw1 = f.y-f.x; tw2 = f.x-f.z; tw3 = f.z;
		} else if (f.y >= f.z) {
			idx1 = probe_index_scrolled(uint3(p0.x, p1.y, p0.z), cascade);
			idx2 = probe_index_scrolled(uint3(p0.x, p1.y, p1.z), cascade);
			tw0 = 1-f.y; tw1 = f.y-f.z; tw2 = f.z-f.x; tw3 = f.x;
		} else {
			idx1 = probe_index_scrolled(uint3(p0.x, p0.y, p1.z), cascade);
			idx2 = probe_index_scrolled(uint3(p0.x, p1.y, p1.z), cascade);
			tw0 = 1-f.z; tw1 = f.z-f.y; tw2 = f.y-f.x; tw3 = f.x;
		}
	}

	SHProbe sp0 = gi_sh_probes[idx0];
	SHProbe sp1 = gi_sh_probes[idx1];
	SHProbe sp2 = gi_sh_probes[idx2];
	SHProbe sp3 = gi_sh_probes[idx3];

	shr = sh_unpack(sp0.r)*tw0 + sh_unpack(sp1.r)*tw1 + sh_unpack(sp2.r)*tw2 + sh_unpack(sp3.r)*tw3;
	shg = sh_unpack(sp0.g)*tw0 + sh_unpack(sp1.g)*tw1 + sh_unpack(sp2.g)*tw2 + sh_unpack(sp3.g)*tw3;
	shb = sh_unpack(sp0.b)*tw0 + sh_unpack(sp1.b)*tw1 + sh_unpack(sp2.b)*tw2 + sh_unpack(sp3.b)*tw3;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	float4 albedo   = albedo_tex  .Sample(albedo_tex_s,   input.uv) * input.color;
	float3 emissive = emission_tex.Sample(emission_tex_s, input.uv).rgb * emission_factor.rgb;

	float3 normal    = normalize(input.normal);
	float3 world_pos = input.world_pos;

	// Shadow: issued early so samples overlap with SH probe work below.
	float  ndotl  = dot(normal, light_direction);
	float  shadow = 1.0;
	if (ndotl > 0.0) {
		float  slope = saturate(min(1.0, sqrt(1.0 - ndotl * ndotl) / max(0.001, ndotl)));
		float3 bias  = normal * (shadow_bias * slope);
		float4 spos  = mul(float4(world_pos + bias, 1), shadow_transform);
		shadow = shadow_factor_pcf4(float3(spos.xy * float2(0.5, -0.5) + 0.5, spos.z / spos.w), 1.0);
	}

	// GI cascade selection: find finest cascade, compute blend at boundary
	float  cascade_blend = 1.0;
	uint   gi_cascade    = GI_CASCADE_COUNT - 1;
	float  blend_margin  = 2.0 / (float)GI_GRID; // 2 probe cells

	for (uint c = 0; c < GI_CASCADE_COUNT - 1; c++) {
		float3 test_uvw = (world_pos - gi_cascades[c].volume_min) * gi_cascades[c].volume_inv;
		if (all(test_uvw >= 0) && all(test_uvw <= 1)) {
			gi_cascade    = c;
			float3 edge_d = min(test_uvw, 1.0 - test_uvw);
			cascade_blend = saturate(min(edge_d.x, min(edge_d.y, edge_d.z)) / blend_margin);
			break;
		}
	}

	// Sample finest cascade (tetrahedral interpolation)
	float4 shr, shg, shb;
	gi_sample_probes(world_pos, gi_cascade, shr, shg, shb);

	// Blend with coarser cascade in the transition zone
	if (cascade_blend < 1.0 && gi_cascade < GI_CASCADE_COUNT - 1) {
		float4 shr_c, shg_c, shb_c;
		gi_sample_probes(world_pos, gi_cascade + 1, shr_c, shg_c, shb_c);
		shr = lerp(shr_c, shr, cascade_blend);
		shg = lerp(shg_c, shg, cascade_blend);
		shb = lerp(shb_c, shb, cascade_blend);
	}

	float4 sh_basis   = float4(0.88623, 1.02333 * normal.yzx);
	float3 irradiance = max(float3(0, 0, 0), float3(
		dot(shr, sh_basis),
		dot(shg, sh_basis),
		dot(shb, sh_basis)
	)) * gi_intensity;

	float3 diffuse     = albedo.rgb * irradiance;
	float3 directional = albedo.rgb * light_color * saturate(ndotl) * shadow;

	return float4(diffuse + directional + emissive, albedo.a);
}
