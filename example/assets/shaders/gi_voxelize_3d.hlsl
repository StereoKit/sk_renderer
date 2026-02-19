//--name = gi_voxelize_3d
//--color:color = 1,1,1,1
//--tex_trans   = 0,0,1,1

// Fragment shader voxelization. Renders 3 orthographic views (X, Y, Z axes)
// with no depth test and no backface culling. Each fragment computes its 3D
// voxel coordinate from interpolated world position and writes directly to a
// 3D texture via imageStore. Each view only processes triangles whose face
// normal is dominant for that axis, ensuring full coverage with no thin
// triangle gaps.

#include "common.hlsli"
#include "gi_voxel.hlsli"

float4 color;
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

Texture2D              shadow_map         : register(t14);
SamplerComparisonState shadow_map_sampler : register(s14);

///////////////////////////////////////////
// 3D Voxel Output (storage image)
///////////////////////////////////////////

RWTexture3D<unorm float4> voxel_tex : register(u0);

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
	float3 world_pos : TEXCOORD0;
	float3 normal    : NORMAL0;
	float2 uv        : TEXCOORD1;
	float4 color     : COLOR0;
	float3 shadow_uv : TEXCOORD2;
	nointerpolation uint view_axis : TEXCOORD3;
	uint   layer     : SV_RenderTargetArrayIndex;
};

///////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn output;

	uint view_idx = id % view_count;
	uint inst_idx = id / view_count;

	float3 world_pos = mul(float4(input.pos, 1), inst[inst_idx].world).xyz;
	output.pos       = mul(float4(world_pos, 1), viewproj[view_idx]);
	output.world_pos = world_pos;
	output.normal    = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;

	// Shadow map projection
	float  ndotl = dot(output.normal, light_direction);
	float  slope = saturate(min(1.0, sqrt(1.0 - ndotl * ndotl) / max(0.001, ndotl)));
	float3 bias  = output.normal * (shadow_bias * slope);
	float4 shadow_pos = mul(float4(world_pos + bias, 1), shadow_transform);
	output.shadow_uv  = float3(shadow_pos.xy * float2(0.5, -0.5) + 0.5, shadow_pos.z / shadow_pos.w);

	output.view_axis = view_idx; // 0=X, 1=Y, 2=Z
	output.layer     = view_idx;
	return output;
}

///////////////////////////////////////////
// Pixel Shader
///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	// Compute geometric face normal from screen-space derivatives
	float3 dp_dx = ddx(input.world_pos);
	float3 dp_dy = ddy(input.world_pos);
	float3 face_normal = normalize(cross(dp_dx, dp_dy));

	// Dominant axis selection: only process this triangle from the view
	// where its face normal has the largest component
	float3 abs_n = abs(face_normal);
	uint dominant;
	if      (abs_n.x >= abs_n.y && abs_n.x >= abs_n.z) dominant = 0;
	else if (abs_n.y >= abs_n.x && abs_n.y >= abs_n.z) dominant = 1;
	else                                               dominant = 2;

	if (dominant != input.view_axis) discard;

	// Compute 3D voxel coordinate in active cascade
	float3 uvw   = (input.world_pos - gi_cascades[gi_active_cascade].volume_min) * gi_cascades[gi_active_cascade].volume_inv;
	int3   texel = int3(uvw * (float)GI_VOXEL_RES);
	texel.z += gi_active_cascade * GI_VOXEL_RES; // offset into stacked texture

	// Bounds check
	if (any(texel < 0) || any(texel >= int3(GI_VOXEL_RES, GI_VOXEL_RES, GI_VOXEL_RES * GI_CASCADE_COUNT))) discard;

	// Albedo
	float4 albedo = albedo_tex.Sample(albedo_tex_s, input.uv) * input.color;

	// Shadow (single sample — voxels don't need PCF)
	float ndotl  = dot(input.normal, light_direction);
	float shadow = 1.0;
	if (ndotl > 0.0) {
		shadow = shadow_map.SampleCmpLevelZero(shadow_map_sampler, input.shadow_uv.xy, input.shadow_uv.z);
	}

	// Simple diffuse lighting
	float diffuse = saturate(ndotl) * shadow;
	float3 lit = albedo.rgb * light_color * diffuse;

	// Write to 3D voxel texture (keep brightest to reduce flickering)
	float4 val      = float4(lit, 1.0);
	float4 existing = voxel_tex[texel];
	if (dot(val, val) > dot(existing, existing))
		voxel_tex[texel] = val;

	// Dummy output (color writes disabled on this material)
	return float4(0, 0, 0, 0);
}
