//--name = gi_probe_scroll

// Compute shader that invalidates newly-entered probes after a toroidal scroll
// shift. Probes whose buffer position now maps to a different world cell are
// filled from the coarser cascade (or zeroed for the coarsest cascade).
// History buffer slots are also filled so the sliding window starts cleanly.

#include "gi_voxel.hlsli"

RWStructuredBuffer<SHProbe> sh_probes  : register(u0);
RWStructuredBuffer<SHProbe> sh_history : register(u1);

int  shift_x;       // grid cells shifted this frame (per axis)
int  shift_y;
int  shift_z;
uint cascade;       // which cascade was scrolled
uint history_size;   // sliding window size

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (any(id >= (uint3)GI_GRID)) return;

	int3 shift = int3(shift_x, shift_y, shift_z);

	// This buffer position's old grid coordinate (what world cell it previously held)
	int3 cur_scroll = int3(gi_cascades[cascade].scroll_x,
	                       gi_cascades[cascade].scroll_y,
	                       gi_cascades[cascade].scroll_z);
	int3 old_scroll = cur_scroll - shift;
	int3 old_grid   = (int3(id) - old_scroll + GI_GRID * 256) % GI_GRID;

	// Check if old data is still valid per axis.
	// For positive shift: old positions [0, shift) are now outside the grid.
	// For negative shift: old positions [GI_GRID+shift, GI_GRID) are outside.
	// For |shift| >= GI_GRID: ALL probes are invalid.
	bool valid_x, valid_y, valid_z;
	if (abs(shift.x) >= GI_GRID) { valid_x = false; }
	else if (shift.x >= 0) { valid_x = (old_grid.x >= shift.x); }
	else                    { valid_x = (old_grid.x < GI_GRID + shift.x); }

	if (abs(shift.y) >= GI_GRID) { valid_y = false; }
	else if (shift.y >= 0) { valid_y = (old_grid.y >= shift.y); }
	else                    { valid_y = (old_grid.y < GI_GRID + shift.y); }

	if (abs(shift.z) >= GI_GRID) { valid_z = false; }
	else if (shift.z >= 0) { valid_z = (old_grid.z >= shift.z); }
	else                    { valid_z = (old_grid.z < GI_GRID + shift.z); }

	if (valid_x && valid_y && valid_z) return; // existing data still good

	// This buffer slot now represents a new world cell. Compute its grid position.
	uint3 new_grid = uint3((int3(id) - cur_scroll + GI_GRID * 256) % GI_GRID);

	// Fill from coarser cascade or zero
	SHProbe fill;
	fill.r = uint2(0, 0);
	fill.g = uint2(0, 0);
	fill.b = uint2(0, 0);

	if (cascade < GI_CASCADE_COUNT - 1) {
		// Compute world position of this new probe
		float3 cell_sz   = 1.0 / (gi_cascades[cascade].volume_inv * (float)GI_GRID);
		float3 world_pos = gi_cascades[cascade].volume_min + (float3(new_grid) + 0.5) * cell_sz;

		// Check if it falls within the coarser cascade's volume
		uint cc = cascade + 1;
		float3 uvw = (world_pos - gi_cascades[cc].volume_min) * gi_cascades[cc].volume_inv;
		if (all(uvw >= 0) && all(uvw <= 1)) {
			uint3 cg = uint3(clamp(uvw * (float)GI_GRID, float3(0,0,0), float3(GI_GRID-1, GI_GRID-1, GI_GRID-1)));
			fill = sh_probes[probe_index_scrolled(cg, cc)];
		}
	}

	// Write probe and fill all history slots
	uint idx = cascade * GI_GRID3 + voxel_index(id);
	sh_probes[idx] = fill;

	uint hist_base = idx * history_size;
	for (uint h = 0; h < history_size; h++)
		sh_history[hist_base + h] = fill;
}
