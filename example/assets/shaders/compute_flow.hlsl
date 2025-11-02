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
	float center_water   = water_map[pos].r;
	float center_surface = center_terrain + center_water; // Total water surface height

	// Sample neighboring surface heights (terrain + water)
	float surface_left, surface_right, surface_down, surface_up;

	if (pos.x > 0) {
		float terrain = height_map[pos + int2(-1, 0)].r;
		float water   = water_map[pos + int2(-1, 0)].r;
		surface_left = terrain + water;
	} else {
		surface_left = center_surface;
	}

	if (pos.x < size - 1) {
		float terrain = height_map[pos + int2(1, 0)].r;
		float water   = water_map[pos + int2(1, 0)].r;
		surface_right = terrain + water;
	} else {
		surface_right = center_surface;
	}

	if (pos.y > 0) {
		float terrain = height_map[pos + int2(0, -1)].r;
		float water   = water_map[pos + int2(0, -1)].r;
		surface_down = terrain + water;
	} else {
		surface_down = center_surface;
	}

	if (pos.y < size - 1) {
		float terrain = height_map[pos + int2(0, 1)].r;
		float water   = water_map[pos + int2(0, 1)].r;
		surface_up = terrain + water;
	} else {
		surface_up = center_surface;
	}

	// Calculate surface height differences (negative means downhill)
	float diff_left  = surface_left  - center_surface;
	float diff_right = surface_right - center_surface;
	float diff_down  = surface_down  - center_surface;
	float diff_up    = surface_up    - center_surface;

	// Flow direction vector (normalized)
	// Positive flow means water flows in that direction
	float2 flow_dir = float2(0, 0);

	// Only flow downhill (negative differences)
	if (diff_left  < 0) flow_dir.x -= abs(diff_left);
	if (diff_right < 0) flow_dir.x += abs(diff_right);
	if (diff_down  < 0) flow_dir.y -= abs(diff_down);
	if (diff_up    < 0) flow_dir.y += abs(diff_up);

	// Normalize flow direction
	float flow_magnitude = length(flow_dir);
	if (flow_magnitude > 0.000001) {
		flow_dir = flow_dir / flow_magnitude;
	}

	// Store flow direction and magnitude
	// .xy = flow direction (normalized)
	// .z = flow magnitude (slope)
	// .w = unused
	out_flow[pos] = float4(flow_dir.x, flow_dir.y, flow_magnitude, 0);
}
