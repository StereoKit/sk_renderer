// Flow direction calculation compute shader
// Determines which direction water should flow based on terrain height

float    terrain_size;
float    timestep;
float    water_threshold;
uint     padding;

Texture2D    <float>  height_map : register(t1);
Texture2D    <float>  water_map  : register(t2);
RWTexture2D  <float4> out_flow   : register(u3);

[numthreads(8, 8, 1)]
void cs(uint3 dispatchThreadID : SV_DispatchThreadID) {
	uint2 pos = dispatchThreadID.xy;
	uint size = (uint)terrain_size;

	if (pos.x >= size || pos.y >= size) return;

	// Get current terrain height and water depth
	float center_terrain = height_map[pos].r;
	float center_water   = water_map [pos].r;
	float center_surface = center_terrain + center_water; // Total water surface height

	// Sample neighboring surface heights (terrain + water)
	float surface_left, surface_right, surface_down, surface_up;

	if (pos.x > 0) {
		float terrain = height_map[pos + int2(-1, 0)].r;
		float water   = water_map[ pos + int2(-1, 0)].r;
		surface_left = terrain + water;
	} else {
		surface_left = center_surface;
	}

	if (pos.x < size - 1) {
		float terrain = height_map[pos + int2(1, 0)].r;
		float water   = water_map [pos + int2(1, 0)].r;
		surface_right = terrain + water;
	} else {
		surface_right = center_surface;
	}

	if (pos.y > 0) {
		float terrain = height_map[pos + int2(0, -1)].r;
		float water   = water_map [pos + int2(0, -1)].r;
		surface_down = terrain + water;
	} else {
		surface_down = center_surface;
	}

	if (pos.y < size - 1) {
		float terrain = height_map[pos + int2(0, 1)].r;
		float water   = water_map [pos + int2(0, 1)].r;
		surface_up = terrain + water;
	} else {
		surface_up = center_surface;
	}

	// Calculate surface height differences (negative means downhill)
	float diff_left  = surface_left  - center_surface;
	float diff_right = surface_right - center_surface;
	float diff_down  = surface_down  - center_surface;
	float diff_up    = surface_up    - center_surface;

	// Calculate total downhill slope
	float total_slope = 0.0;
	if (diff_left  < 0) total_slope += abs(diff_left);
	if (diff_right < 0) total_slope += abs(diff_right);
	if (diff_down  < 0) total_slope += abs(diff_down);
	if (diff_up    < 0) total_slope += abs(diff_up);

	// Calculate water output to each direction
	// Water is distributed proportionally to the slope in each direction
	float water_amount = center_water;
	float max_flow = water_amount * timestep;  // Max water that can flow this step

	// Outflow to each neighbor (as percentage of cell's current water)
	float outflow_right = 0.0;
	float outflow_up    = 0.0;
	float outflow_left  = 0.0;
	float outflow_down  = 0.0;

	if (total_slope > 0.00001 && water_amount > 0.00001) {
		// Calculate percentage flowing to each direction
		float pct_left  = (diff_left  < 0) ? (abs(diff_left)  / total_slope) : 0.0;
		float pct_right = (diff_right < 0) ? (abs(diff_right) / total_slope) : 0.0;
		float pct_down  = (diff_down  < 0) ? (abs(diff_down)  / total_slope) : 0.0;
		float pct_up    = (diff_up    < 0) ? (abs(diff_up)    / total_slope) : 0.0;

		// Calculate total outflow as percentage of cell's water
		float total_outflow_pct = min(max_flow / water_amount, 1.0);

		// Distribute outflow to each direction proportionally
		outflow_right = pct_right * total_outflow_pct;
		outflow_up    = pct_up    * total_outflow_pct;
		outflow_left  = pct_left  * total_outflow_pct;
		outflow_down  = pct_down  * total_outflow_pct;
	}

	// Store flow data
	// Each component represents outflow to that neighbor as a percentage of this cell's water
	// .x = outflow to right
	// .y = outflow to up
	// .z = outflow to left
	// .w = outflow to down
	// Sum of all components = total_outflow / center_water
	out_flow[pos] = float4(outflow_right, outflow_up, outflow_left, outflow_down);
}
