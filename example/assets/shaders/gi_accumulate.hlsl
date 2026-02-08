//--name = gi_accumulate

// Compute shader to project orthographic captures into SH coefficients.
// Each dispatch processes one 32x32 capture from one axis-aligned direction,
// accumulating into the 3D SH textures.

Texture2D<float4>   capture_tex : register(t0);
RWTexture3D<float4> sh_r        : register(u0);
RWTexture3D<float4> sh_g        : register(u1);
RWTexture3D<float4> sh_b        : register(u2);

uint  axis;        // 0=X, 1=Y, 2=Z
uint  layer_index; // 0-31
float dir_sign;    // +1.0 or -1.0
uint  grid_size;   // 32
uint  flip_x;      // flip pixel X before mapping to texel
uint  flip_y;      // flip pixel Y before mapping to texel
float hysteresis;  // decay factor for existing SH (1.0 = pure add, <1.0 = temporal blend)

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x >= grid_size || id.y >= grid_size) return;

	float3 color = capture_tex[id.xy].rgb;

	// Correct for view-space axis flips caused by the lookat orientation.
	// The lookat right/up vectors flip between + and - directions,
	// so some pixel axes map in reverse. Flip them back here.
	uint px = flip_x ? (grid_size - 1 - id.x) : id.x;
	uint py = flip_y ? (grid_size - 1 - id.y) : id.y;

	// Map (pixel_x, pixel_y, layer_index) -> 3D texel based on capture axis
	uint3 texel;
	if      (axis == 0) texel = uint3(layer_index, py, px); // X-layer: image plane is ZY
	else if (axis == 1) texel = uint3(px, layer_index, py); // Y-layer: image plane is XZ
	else                texel = uint3(px, py, layer_index); // Z-layer: image plane is XY

	// SH basis for axis-aligned direction, scaled by integration weight (4*pi/6)
	// Y00  = 0.28209
	// Y1m  = 0.48860 * direction_component
	// Weight = 4*pi/6 = 2.09440
	float3 dir = float3(0, 0, 0);
	if      (axis == 0) dir.x = dir_sign;
	else if (axis == 1) dir.y = dir_sign;
	else                dir.z = dir_sign;

	float w = 2.09440;
	float4 sh_coeff = float4(
		0.28209 * w,
		0.48860 * dir.y * w,
		0.48860 * dir.z * w,
		0.48860 * dir.x * w
	);

	// Temporal accumulation: decay existing SH then add new contribution.
	// hysteresis=1.0 is pure additive (second direction in same frame),
	// hysteresis<1.0 decays old data (first direction, enables continuous updates).
	sh_r[texel] = sh_r[texel] * hysteresis + sh_coeff * color.r;
	sh_g[texel] = sh_g[texel] * hysteresis + sh_coeff * color.g;
	sh_b[texel] = sh_b[texel] * hysteresis + sh_coeff * color.b;
}
