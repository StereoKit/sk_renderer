// Flow direction calculation compute shader
// Determines which direction water should flow based on terrain height

struct FlowData {
	float outflow_right;  // Absolute water amount flowing right
	float outflow_up;     // Absolute water amount flowing up
	float outflow_left;   // Absolute water amount flowing left
	float outflow_down;   // Absolute water amount flowing down
};

float    terrain_size;
float    timestep;
float    water_threshold;
float    rainfall_rate;      // Not used in flow compute, but needed for struct layout
float    evaporation_rate;   // Not used in flow compute, but needed for struct layout
float3   padding;

Texture2D           <float>    height_map : register(t1);
Texture2D           <float>    water_map  : register(t2);
RWStructuredBuffer  <FlowData> out_flow   : register(u3);

[numthreads(8, 8, 1)]
void cs(uint3 dispatchThreadID : SV_DispatchThreadID) {
	uint2 pos = dispatchThreadID.xy;
	uint size = (uint)terrain_size;

	if (pos.x >= size || pos.y >= size) return;

	// Get current terrain height and water depth
	float curr_height_terrain = height_map[pos].r;
	float curr_height_water   = water_map [pos].r;
	float curr_height_total   = curr_height_terrain + curr_height_water; // Total water surface height

	// Sample neighboring surface heights (terrain + water)
	float left_height_total, right_height_total, down_height_total, up_height_total;

	if (pos.x > 0) {
		float terrain = height_map[pos + int2(-1, 0)].r;
		float water   = water_map [pos + int2(-1, 0)].r;
		left_height_total = terrain + water;
	} else {
		left_height_total = curr_height_total;
	}

	if (pos.x < size - 1) {
		float terrain = height_map[pos + int2(1, 0)].r;
		float water   = water_map [pos + int2(1, 0)].r;
		right_height_total = terrain + water;
	} else {
		right_height_total = curr_height_total;
	}

	if (pos.y > 0) {
		float terrain = height_map[pos + int2(0, -1)].r;
		float water   = water_map [pos + int2(0, -1)].r;
		down_height_total = terrain + water;
	} else {
		down_height_total = curr_height_total;
	}

	if (pos.y < size - 1) {
		float terrain = height_map[pos + int2(0, 1)].r;
		float water   = water_map [pos + int2(0, 1)].r;
		up_height_total = terrain + water;
	} else {
		up_height_total = curr_height_total;
	}

	// Calculate surface height differences (negative means downhill)
	float diff_left  = left_height_total  - curr_height_total;
	float diff_right = right_height_total - curr_height_total;
	float diff_down  = down_height_total  - curr_height_total;
	float diff_up    = up_height_total    - curr_height_total;

	// Calculate total downhill slope
	float total_slope = 0.0;
	if (diff_left  < 0) total_slope += abs(diff_left);
	if (diff_right < 0) total_slope += abs(diff_right);
	if (diff_down  < 0) total_slope += abs(diff_down);
	if (diff_up    < 0) total_slope += abs(diff_up);

	// Calculate water output to each direction
	// Water is distributed proportionally to the slope in each direction
	float water_amount = curr_height_water;

	// Stability constraint: prevent oscillations by ensuring outflow doesn't create uphill gradients
	// After water flows out, our surface must remain >= lowest neighbor's surface. 
	float min_neighbor_height = min(min(left_height_total, right_height_total), min(down_height_total, up_height_total));
	float max_safe_outflow    = max(0.0, (curr_height_total - min_neighbor_height) * 0.25); // we could flow a bit more, but 1/4 * min_neighbor is the worst case scenario

	float max_flow = min(water_amount * timestep, max_safe_outflow);  // Max water that can flow this step

	// Outflow to each neighbor (absolute water amounts)
	float outflow_right = 0.0;
	float outflow_up    = 0.0;
	float outflow_left  = 0.0;
	float outflow_down  = 0.0;

	if (total_slope > 0.0 && water_amount > 0.0) {
		// Distribute water proportionally to downhill slopes
		outflow_right = (diff_right < 0) ? (abs(diff_right) / total_slope) * max_flow : 0.0;
		outflow_up    = (diff_up    < 0) ? (abs(diff_up)    / total_slope) * max_flow : 0.0;
		outflow_left  = (diff_left  < 0) ? (abs(diff_left)  / total_slope) * max_flow : 0.0;
		outflow_down  = (diff_down  < 0) ? (abs(diff_down)  / total_slope) * max_flow : 0.0;
	}

	// Store flow data in structured buffer
	uint index = pos.x + pos.y * size;
	FlowData flow_data;
	flow_data.outflow_right = outflow_right;
	flow_data.outflow_up    = outflow_up;
	flow_data.outflow_left  = outflow_left;
	flow_data.outflow_down  = outflow_down;

	out_flow[index] = flow_data;
}
