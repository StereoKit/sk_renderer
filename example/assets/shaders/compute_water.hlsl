// Water flow simulation compute shader
// Simulates water flowing based on flow directions
// Water flows from each cell to its neighbors, and receives water from neighbors

struct FlowData {
	float outflow_right;
	float outflow_up;
	float outflow_left;
	float outflow_down;
};

float    terrain_size;
float    timestep;
float    water_threshold;
float    rainfall_rate;
float    evaporation_rate;
float3   padding;

Texture2D          <float>    height_in    : register(t1);
StructuredBuffer   <FlowData> flow_map     : register(t2);
Texture2D          <float>    water_in     : register(t3);
Texture2D          <float>    sediment_in  : register(t4);
RWTexture2D        <float>    height_out   : register(u5);
RWTexture2D        <float>    water_out    : register(u6);
RWTexture2D        <float>    sediment_out : register(u7);
RWTexture2D        <float4>   debug_out    : register(u8);

[numthreads(8, 8, 1)]
void cs(uint3 dispatchThreadID : SV_DispatchThreadID) {
	uint2 pos  = dispatchThreadID.xy;
	uint  size = (uint)terrain_size;

	if (pos.x >= size || pos.y >= size) return;

	// Get current water amount

	float current_height   = height_in  [pos].r;
	float current_water    = water_in   [pos].r;
	float current_sediment = sediment_in[pos].r;

	// Get flow data for this cell (outflow to each neighbor as absolute amounts)
	uint     index         = pos.x + pos.y * size;
	FlowData flow_data     = flow_map[index];
	float    outflow_right = flow_data.outflow_right;  // Absolute water flowing to right neighbor
	float    outflow_up    = flow_data.outflow_up;     // Absolute water flowing to up neighbor
	float    outflow_left  = flow_data.outflow_left;   // Absolute water flowing to left neighbor
	float    outflow_down  = flow_data.outflow_down;   // Absolute water flowing to down neighbor

	// Calculate total water output
	float water_output = outflow_right + outflow_up + outflow_left + outflow_down;

	// Calculate sediment output proportional to water output
	float sediment_ratio  = (current_water > 0.00001) ? (water_output / current_water) : 0.0;
	float sediment_output = sediment_ratio * current_sediment;
	
	// Calculate water input from neighbors
	// Each neighbor's outflow_to_us is stored explicitly in their flow map
	float water_input    = 0.0;
	float sediment_input = 0.0;

	// Left neighbor flows right (toward us)
	if (pos.x > 0) {
		int2     left_pos        = pos + int2(-1, 0);
		uint     left_index      = left_pos.x + left_pos.y * size;
		FlowData left_flow_data  = flow_map[left_index];
		float    left_outflow    = left_flow_data.outflow_right;  // Absolute water flowing to us
		float    left_sediment   = sediment_in[left_pos].r;
		float    left_water      = water_in[left_pos].r;
		float    left_sed_ratio  = (left_water > 0.00001) ? (left_outflow / left_water) : 0.0;
		water_input    += left_outflow;
		sediment_input += left_sed_ratio * left_sediment;
	}

	// Right neighbor flows left (toward us)
	if (pos.x < size - 1) {
		int2     right_pos       = pos + int2(1, 0);
		uint     right_index     = right_pos.x + right_pos.y * size;
		FlowData right_flow_data = flow_map[right_index];
		float    right_outflow   = right_flow_data.outflow_left;  // Absolute water flowing to us
		float    right_sediment  = sediment_in[right_pos].r;
		float    right_water     = water_in[right_pos].r;
		float    right_sed_ratio = (right_water > 0.00001) ? (right_outflow / right_water) : 0.0;
		water_input    += right_outflow;
		sediment_input += right_sed_ratio * right_sediment;
	}

	// Down neighbor flows up (toward us)
	if (pos.y > 0) {
		int2     down_pos       = pos + int2(0, -1);
		uint     down_index     = down_pos.x + down_pos.y * size;
		FlowData down_flow_data = flow_map[down_index];
		float    down_outflow   = down_flow_data.outflow_up;  // Absolute water flowing to us
		float    down_sediment  = sediment_in[down_pos].r;
		float    down_water     = water_in[down_pos].r;
		float    down_sed_ratio = (down_water > 0.00001) ? (down_outflow / down_water) : 0.0;
		water_input    += down_outflow;
		sediment_input += down_sed_ratio * down_sediment;
	}

	// Up neighbor flows down (toward us)
	if (pos.y < size - 1) {
		int2     up_pos       = pos + int2(0, 1);
		uint     up_index     = up_pos.x + up_pos.y * size;
		FlowData up_flow_data = flow_map[up_index];
		float    up_outflow   = up_flow_data.outflow_down;  // Absolute water flowing to us
		float    up_sediment  = sediment_in[up_pos].r;
		float    up_water     = water_in[up_pos].r;
		float    up_sed_ratio = (up_water > 0.00001) ? (up_outflow / up_water) : 0.0;
		water_input    += up_outflow;
		sediment_input += up_sed_ratio * up_sediment;
	}

	// Calculate new water amount (before rainfall)
	float new_water = current_water - water_output + water_input;

	// Height-dependent evaporation (faster at higher elevations)
	// Apply evaporation BEFORE rainfall so rain always has a chance to contribute
	float evaporation = evaporation_rate * (1.0 + (current_height + current_water) * 2.0);  // More evaporation at higher elevations
	new_water = max(0.0, new_water - evaporation);

	// Add rainfall (toggled on/off via parameter)
	// Applied AFTER evaporation so it always contributes to erosion
	new_water += rainfall_rate;

	//if (pos.x == 100 && pos.y == 170)
	//	new_water += 0.04;

	// Clamp to reasonable range
	new_water = clamp(new_water, 0.0, 2.0);

	// === EROSION AND DEPOSITION ===

	// Update sediment from transport (what flows in vs what flows out)
	float transported_sediment = current_sediment - sediment_output + sediment_input;

	// Erosion parameters
	float erosion_rate             = 0.5f;   // How fast water picks up sediment
	float deposition_rate          = 0.5f;   // How fast water drops sediment
	float sediment_capacity_factor = 1.0f;   // Max sediment proportional to flow amount

	// Calculate sediment capacity based on water output (absolute amount)
	// Only flowing water can carry sediment, not stagnant pools
	float sediment_capacity = water_output * sediment_capacity_factor;

	// Erosion: pick up sediment where water flows fast and capacity isn't met
	float erosion_amount = 0.0;
	if (transported_sediment < sediment_capacity) {
		// Erode terrain proportional to capacity deficit
		erosion_amount = (sediment_capacity - transported_sediment) * erosion_rate;
		// CRITICAL: Can't erode more terrain than exists (prevents generating sediment from nowhere)
		erosion_amount = min(erosion_amount, current_height);
	}

	// Deposition: drop sediment where water slows down or capacity is exceeded
	float deposition_amount = 0.0;
	if (transported_sediment > sediment_capacity) {
		// Deposit excess sediment
		deposition_amount = max(0, (transported_sediment - sediment_capacity) * deposition_rate);// * timestep;
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

	// Debug visualization - uncomment what you want to see:
	// Example debug outputs (comment/uncomment as needed):

	// Option 1: Visualize erosion amount (red = erosion, green = deposition)
	//debug_out[pos] = float4(erosion_amount * 100.0, deposition_amount * 100.0, sediment_capacity, 1);
	debug_out[pos] = float4(water_output*10, 0, 0, 1);

	// Option 2: Visualize sediment capacity vs transported sediment
	//debug_out[pos] = float4(sediment_capacity * 5.0, transported_sediment * 5.0, 0, 1);

	// Option 3: Visualize water flow rate (outflow percentage)
	//debug_out[pos] = float4(total_outflow_pct, total_outflow_pct, total_outflow_pct, 1);

	// Option 4: Visualize slope factor (velocity estimate)
	//float slope_viz = sqrt(max(total_outflow_pct, 0.001));
	//debug_out[pos] = float4(slope_viz, slope_viz, slope_viz, 1);

	// Default: Clear to black (no visualization)
	//debug_out[pos] = float4(0, 0, 0, 1);
}
