// Water flow simulation compute shader
// Simulates water flowing based on flow directions
// Water flows from each cell to its neighbors, and receives water from neighbors

float    terrain_size;
float    timestep;
float    water_threshold;
uint     padding;

Texture2D    <float>  height_map : register(t1);
Texture2D    <float4> flow_map   : register(t2);
Texture2D    <float>  water_in   : register(t3);
RWTexture2D  <float>  water_out  : register(u4);

[numthreads(8, 8, 1)]
void cs(uint3 dispatchThreadID : SV_DispatchThreadID) {
	uint2 pos  = dispatchThreadID.xy;
	uint  size = (uint)terrain_size;

	if (pos.x >= size || pos.y >= size) return;

	// Get current water amount
	float current_water = water_in[pos].r;

	// Get flow direction for this cell
	float4 flow_data      = flow_map[pos];
	float2 flow_dir       = flow_data.xy;
	float  flow_magnitude = flow_data.z;

	// Calculate water output (how much water flows out of this cell)
	float water_output = 0.0;
	if (flow_magnitude > 0.000001) {
		// Water flows proportional to slope and current water amount
		water_output = min(current_water, flow_magnitude * timestep * 10.0);
	}

	// Check all 4 neighbors
	int2 neighbors[4] = {
		int2(-1,  0),  // Left
		int2( 1,  0),  // Right
		int2( 0, -1),  // Down
		int2( 0,  1)   // Up
	};

	// Calculate water input from neighboring cells
	float water_input = 0.0;
	for (int i = 0; i < 4; i++) {
		int2 neighbor_pos = (int2)pos + neighbors[i];

		// Check bounds
		if (neighbor_pos.x < 0 || neighbor_pos.x >= (int)size ||
		    neighbor_pos.y < 0 || neighbor_pos.y >= (int)size) {
			continue;
		}

		// Get neighbor's flow direction
		float4 neighbor_flow           = flow_map[neighbor_pos];
		float2 neighbor_flow_dir       = neighbor_flow.xy;
		float  neighbor_flow_magnitude = neighbor_flow.z;

		if (neighbor_flow_magnitude < 0.000001) continue;

		// Check if neighbor flows toward this cell
		// Calculate direction from neighbor to this cell
		float2 dir_to_us = normalize(float2(neighbors[i].x, neighbors[i].y) * -1.0);

		// Dot product tells us how much the neighbor flows toward us
		float flow_toward_us = dot(neighbor_flow_dir, dir_to_us);

		if (flow_toward_us > 0.5) {  // If neighbor flows mostly toward us
			float neighbor_water = water_in[neighbor_pos].r;
			float neighbor_output = min(neighbor_water, neighbor_flow_magnitude * timestep * 10.0);
			water_input += neighbor_output * flow_toward_us * 0.25;  // 0.25 since we have 4 neighbors
		}
	}

	// Add constant rainfall (small amount to keep simulation active)
	float rainfall = 0;// 0.001;

	// Calculate new water amount
	float new_water = current_water - water_output + water_input + rainfall;

	// Evaporation
	float evaporation = 0;// 0.0005;
	new_water = max(0.0, new_water - evaporation);

	// Clamp to reasonable range
	new_water = clamp(new_water, 0.0, 2.0);

	water_out[pos] = new_water;
}
