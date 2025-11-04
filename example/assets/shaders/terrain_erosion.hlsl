//--name = terrain_erosion

#include "common.hlsli"

struct Inst {
	float4x4 world;
};

struct FlowData {
	float outflow_right;  // Absolute water amount flowing right
	float outflow_up;     // Absolute water amount flowing up
	float outflow_left;   // Absolute water amount flowing left
	float outflow_down;   // Absolute water amount flowing down
};

StructuredBuffer<Inst>     inst     : register(t2, space0);
StructuredBuffer<FlowData> flow_map : register(t5);

Texture2D    height_map         : register(t3);
SamplerState height_map_sampler : register(s3);
Texture2D    water_map          : register(t4);
SamplerState water_map_sampler  : register(s4);
Texture2D    debug_tex          : register(t6);
SamplerState debug_tex_sampler  : register(s6);

struct vsIn {
	float3 pos   : SV_POSITION;
	float3 norm  : NORMAL;
	float2 uv    : TEXCOORD0;
	float4 color : COLOR0;
};

struct psIn {
	float4 pos       : SV_POSITION;
	float2 uv        : TEXCOORD0;
	float3 world_pos : TEXCOORD1;
	float3 normal    : NORMAL;
	float  height    : TEXCOORD2;
	float  water     : TEXCOORD3;
	uint   layer     : SV_RenderTargetArrayIndex;
};

psIn vs(vsIn input, uint id : SV_InstanceID) {
	// Multi-view instancing: extract instance index and view index
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	psIn output;

	// Sample height from texture
	float height = height_map.SampleLevel(height_map_sampler, input.uv, 0).r;
	float water  = water_map .SampleLevel(water_map_sampler,  input.uv, 0).r;

	// Adjust vertex position based on height
	float3 displaced_pos = input.pos;
	displaced_pos.y = height;

	// Add water height if above threshold
	float water_threshold = 0.01;
	if (water > water_threshold) {
		//displaced_pos.y += water;
	}

	// Transform to world space
	output.world_pos = mul(float4(displaced_pos, 1), inst[inst_idx].world).xyz;
	output.pos = mul(float4(output.world_pos, 1), viewproj[view_idx]);
	output.uv = input.uv;
	output.height = height;
	output.water = water;
	output.layer = view_idx;

	// Calculate normal from height map using finite differences
	float texel_size = 1.0 / 256.0; // Assuming 256x256 texture
	float h_right = height_map.SampleLevel(height_map_sampler, input.uv + float2(texel_size, 0), 0).r;
	float h_left  = height_map.SampleLevel(height_map_sampler, input.uv + float2(-texel_size, 0), 0).r;
	float h_up    = height_map.SampleLevel(height_map_sampler, input.uv + float2(0, texel_size), 0).r;
	float h_down  = height_map.SampleLevel(height_map_sampler, input.uv + float2(0, -texel_size), 0).r;

	// Calculate tangent vectors in world space (accounting for terrain scale)
	float4x4 world_mat = inst[inst_idx].world;
	float scale_x = length(world_mat[0].xyz);
	float scale_y = length(world_mat[1].xyz);
	float scale_z = length(world_mat[2].xyz);

	float3 tangent_x = float3(texel_size * scale_x, (h_right - h_left) * scale_y, 0);
	float3 tangent_z = float3(0, (h_up - h_down) * scale_y, texel_size * scale_z);

	output.normal = normalize(cross(tangent_z, tangent_x));

	return output;
}

float4 ps(psIn input) : SV_TARGET {
	const float3 light_dir = normalize(float3(1, 4, 2));

	// Calculate lighting (similar to test.hlsl)
	float3 normal   = input.normal;
	float3 ambient  = float3(0.05, 0.05, 0.1);

	// Calculate slope (steepness)
	float slope = 1.0 - input.normal.y;

	// Terrain color based on height and slope
	float3 terrain_color;

	// Color palette:
	// - Low areas: dark green/grass
	// - Mid areas: lighter green
	// - High areas: rocky brown
	// - Steep slopes: rocky gray

	float3 low_color    = float3(0.2, 0.4, 0.1);  // Dark green
	float3 mid_color    = float3(0.3, 0.6, 0.2);  // Light green
	float3 high_color   = float3(0.5, 0.4, 0.3);  // Brown
	float3 cliff_color  = float3(0.4, 0.4, 0.4);  // Gray rock

	// Blend based on height
	if (input.height < 0.3) {
		terrain_color = lerp(low_color, mid_color, input.height / 0.3);
	} else if (input.height < 0.6) {
		terrain_color = lerp(mid_color, high_color, (input.height - 0.3) / 0.3);
	} else {
		terrain_color = high_color;
	}

	// Blend in cliff color based on slope
	if (slope > 0.3) {
		float cliff_blend = saturate((slope - 0.3) / 0.3);
		terrain_color = lerp(terrain_color, cliff_color, cliff_blend);
	}

	// Water visualization
	float  water_threshold  = 0.02;
	float3 water_color      = float3(0.1, 0.3, 0.6);
	float3 water_color_deep = float3(0.05, 0.15, 0.3);
	if (input.water > water_threshold) {
		// Blue water color
		float water_blend = saturate((input.water-water_threshold)*10);
		terrain_color = lerp(water_color, water_color_deep, water_blend);
		normal = float3(0, 1, 0); // Flatten normal for water surface
	}
	else {
		float water_blend = input.water / water_threshold;
		water_blend = water_blend * water_blend;
		terrain_color = lerp(terrain_color, water_color, water_blend);
		normal = lerp(normal, float3(0, 1, 0), water_blend);
	}

	float  diffuse  = saturate(dot(normal, light_dir)) * 0.8;
	float3 lighting = ambient + diffuse;
	
	// Apply lighting
	float3 final_color = terrain_color * lighting;

	// Visualize flow magnitude in the red channel
	uint2 flow_pos = uint2(input.uv * 256);
	uint flow_index = flow_pos.x + flow_pos.y * 256;
	FlowData flow_data_struct = flow_map[flow_index];

	float4 flow_data = float4(
		flow_data_struct.outflow_right,
		flow_data_struct.outflow_up,
		flow_data_struct.outflow_left,
		flow_data_struct.outflow_down
	) * input.water;

	// Visualize flow direction as lines
	// Calculate direction from center of UV texel
	float2 px     = input.uv * 256;
	float2 px_off = frac(px) - float2(0.5, 0.5);
	float2 px_dir = -normalize(px_off);

	// Dot product with flow direction - high values when aligned
	float2 flow_dir   = float2(flow_data.x - flow_data.z, flow_data.y - flow_data.w);
	float  flow_mag   = length(flow_dir);
	float2 flow_dir_n = flow_dir/flow_mag;
	float  flow_alignment = saturate((dot(px_dir, flow_dir_n) - 0.95) / 0.05f);

	// Visualize alignment (brighter when flow aligns with radial direction)
	float3 flow_vis = float3(0, 0, 0);
	flow_vis = float3(1, 1, 1) * flow_alignment * saturate(flow_mag * 50.0);// * (saturate((saturate(flow_magnitude * 5.0) - length(px_off*2)) * 100) );

	// Debug visualization - uncomment to overlay debug data:
	float4 debug_data = debug_tex.Sample(debug_tex_sampler, input.uv);

	// Option 1: Show debug as overlay (blend with terrain)
	//final_color = lerp(final_color, debug_data.rgb, saturate(length(debug_data.rgb)));

	// Option 2: Replace terrain color entirely with debug data
	//return float4(debug_data.rgb + flow_vis*2,1);

	// Option 3: Show debug only where it's non-zero (overlay)
	//if (length(debug_data.rgb) > 0.01) {
	//	final_color = debug_data.rgb;
	//}

	// Option 4: Mix debug with flow visualization
	//final_color += debug_data.rgb;

	return float4(final_color+flow_vis, 1);

	//return float4(flow_magnitude.xxx*10, 1);

	return float4(final_color, 1.0);
}
