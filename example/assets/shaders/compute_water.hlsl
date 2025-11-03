// Water flow simulation compute shader
// Simulates water flowing based on flow directions
// Water flows from each cell to its neighbors, and receives water from neighbors

float    terrain_size;
float    timestep;
float    water_threshold;
uint     padding;

Texture2D    <float>  height_in     : register(t1);
Texture2D    <float4> flow_map     : register(t2);
Texture2D    <float>  water_in     : register(t3);
Texture2D    <float>  sediment_in  : register(t4);
RWTexture2D  <float>  height_out   : register(u5);
RWTexture2D  <float>  water_out    : register(u6);
RWTexture2D  <float>  sediment_out : register(u7);

[numthreads(8, 8, 1)]
void cs(uint3 dispatchThreadID : SV_DispatchThreadID) {
	uint2 pos  = dispatchThreadID.xy;
	uint  size = (uint)terrain_size;

	if (pos.x >= size || pos.y >= size) return;

	// Get current water amount

	float current_height   = height_in  [pos].r;
	float current_water    = water_in   [pos].r;
	float current_sediment = sediment_in[pos].r;

	// Get flow data for this cell (outflow to each neighbor as percentage of our water)
	float4 flow_data       = flow_map[pos];
	float  outflow_right   = flow_data.x;  // Percentage flowing to right neighbor
	float  outflow_up      = flow_data.y;  // Percentage flowing to up neighbor
	float  outflow_left    = flow_data.z;  // Percentage flowing to left neighbor
	float  outflow_down    = flow_data.w;  // Percentage flowing to down neighbor

	// Calculate total water output
	float total_outflow_pct = outflow_right + outflow_up + outflow_left + outflow_down;
	float water_output      = total_outflow_pct * current_water;
	float sediment_output   = total_outflow_pct * current_sediment;
	
	// Calculate water input from neighbors
	// Each neighbor's outflow_to_us is stored explicitly in their flow map
	float water_input    = 0.0;
	float sediment_input = 0.0;

	// Left neighbor flows right (toward us)
	if (pos.x > 0) {
		int2   left_pos          = pos + int2(-1, 0);
		float4 left_flow_data    = flow_map[left_pos];
		float  left_outflow_pct  = left_flow_data.x;  // Their outflow_right
		float  left_sediment     = sediment_in[left_pos].r;
		float  left_water        = water_in[left_pos].r;
		water_input    += left_outflow_pct * left_water;
		sediment_input += left_outflow_pct * left_sediment;
	}

	// Right neighbor flows left (toward us)
	if (pos.x < size - 1) {
		int2   right_pos         = pos + int2(1, 0);
		float4 right_flow_data   = flow_map[right_pos];
		float  right_outflow_pct = right_flow_data.z;  // Their outflow_left
		float  right_sediment    = sediment_in[right_pos].r;
		float  right_water       = water_in[right_pos].r;
		water_input    += right_outflow_pct * right_water;
		sediment_input += right_outflow_pct * right_sediment;
	}

	// Down neighbor flows up (toward us)
	if (pos.y > 0) {
		int2   down_pos         = pos + int2(0, -1);
		float4 down_flow_data   = flow_map[down_pos];
		float  down_outflow_pct = down_flow_data.y;  // Their outflow_up
		float  down_sediment    = sediment_in[down_pos].r;
		float  down_water       = water_in[down_pos].r;
		water_input    += down_outflow_pct * down_water;
		sediment_input += down_outflow_pct * down_sediment;
	}

	// Up neighbor flows down (toward us)
	if (pos.y < size - 1) {
		int2   up_pos         = pos + int2(0, 1);
		float4 up_flow_data   = flow_map[up_pos];
		float  up_outflow_pct = up_flow_data.w;  // Their outflow_down
		float  up_sediment    = sediment_in[up_pos].r;
		float  up_water       = water_in[up_pos].r;
		water_input    += up_outflow_pct * up_water;
		sediment_input += up_outflow_pct * up_sediment;
	}

	// Calculate new water amount (before rainfall)
	float new_water = current_water - water_output + water_input;

	// Height-dependent evaporation (faster at higher elevations)
	// Apply evaporation BEFORE rainfall so rain always has a chance to contribute
	float evaporation_base = 0.003;
	float evaporation = evaporation_base * (1.0 + (current_height + current_water) * 2.0);  // More evaporation at higher elevations
	new_water = max(0.0, new_water - evaporation);

	// Add constant rainfall (small amount to keep simulation active)
	// Applied AFTER evaporation so it always contributes to erosion
	float rainfall = 0.006;
	new_water += rainfall;

	// Clamp to reasonable range
	new_water = clamp(new_water, 0.0, 2.0);

	// === EROSION AND DEPOSITION ===

	// Update sediment from transport (what flows in vs what flows out)
	float transported_sediment = current_sediment - sediment_output + sediment_input;

	// Erosion parameters
	float erosion_rate    = 0.5f;   // How fast water picks up sediment
	float deposition_rate = 0.1f;   // How fast water drops sediment
	float sediment_capacity_factor = 1.0f;  // Max sediment proportional to flow * water

	// Calculate sediment capacity based on flow velocity and water amount (AFTER transport)
	float sediment_capacity = water_output * sediment_capacity_factor;

	// Erosion: pick up sediment where water flows fast and capacity isn't met
	float erosion_amount = 0.0;
	if (transported_sediment < sediment_capacity) {
		// Erode terrain proportional to flow speed and water amount
		erosion_amount = (sediment_capacity - transported_sediment) * erosion_rate;// * timestep;
		//erosion_amount = min(erosion_amount, current_height * 0.1); // Don't erode more than 10% of height
	}

	// Deposition: drop sediment where water slows down or capacity is exceeded
	float deposition_amount = 0.0;
	if (transported_sediment > sediment_capacity) {
		// Deposit excess sediment
		deposition_amount = (transported_sediment - sediment_capacity) * deposition_rate;// * timestep;
	}

	// Update height: subtract erosion, add deposition
	float new_height = current_height - erosion_amount + deposition_amount;
	new_height = max(0.0, new_height); // Don't go below zero

	// Update sediment: transported amount + erosion - deposition
	float new_sediment = transported_sediment + erosion_amount - deposition_amount;
	new_sediment = max(0.0, new_sediment);

	// Write outputs
	height_out  [pos] = new_height;
	water_out   [pos] = new_water;
	sediment_out[pos] = new_sediment;
}
