//--name = gaussian_splat

#include "common.hlsli"

// Render parameters
float  splat_scale;      // Global scale multiplier for splats
float  sh_degree;        // SH degree to use (0-3)
uint   splat_count;      // Total number of splats
float  opacity_scale;    // Opacity multiplier
float2 screen_size;      // Screen resolution in pixels
float  max_radius;       // Max splat radius in pixels (0 = unlimited)

// Packed Gaussian splat data (124 bytes per splat, 59% smaller than unpacked)
// Uses half precision and smallest-3 quaternion encoding
struct GaussianSplatPacked {
	float  pos_x, pos_y, pos_z;  // 12 bytes - full precision position
	uint   rot_packed;           // 4 bytes  - smallest-3 quaternion (10.10.10.2)
	uint   scale_xy;             // 4 bytes  - scale.x | scale.y (half floats)
	uint   scale_z_opacity;      // 4 bytes  - scale.z | opacity (half floats)
	uint   sh_dc_rg;             // 4 bytes  - sh_dc.r | sh_dc.g (half floats)
	uint   sh_dc_b_pad;          // 4 bytes  - sh_dc.b | padding (half floats)
	uint   sh_rest[23];          // 92 bytes - 45 half floats packed
};

// Unpacked splat data (computed from packed)
struct GaussianSplat {
	float3 position;
	float  opacity;
	float3 sh_dc;
	float3 scale;
	float4 rotation;
	float3 sh_rest[15];
};

// Unpack two half floats from uint
float2 unpack_halfs(uint packed) {
	return float2(f16tof32(packed), f16tof32(packed >> 16));
}

// Unpack smallest-3 quaternion (10.10.10.2 format)
float4 unpack_quat_smallest3(uint packed) {
	// Extract components
	float a = ((packed      ) & 0x3FF) / 1023.0f;
	float b = ((packed >> 10) & 0x3FF) / 1023.0f;
	float c = ((packed >> 20) & 0x3FF) / 1023.0f;
	uint  idx = (packed >> 30) & 0x3;

	// Convert from 0-1 to -1/sqrt(2) to +1/sqrt(2)
	const float scale = 1.41421356237f;  // sqrt(2)
	float3 three = float3(a, b, c) * 2.0f - 1.0f;
	three *= scale * 0.5f;

	// Reconstruct 4th component
	float w = sqrt(max(0.0f, 1.0f - dot(three, three)));

	// Reorder based on which component was largest
	float4 q;
	if (idx == 0)      q = float4(w, three.x, three.y, three.z);
	else if (idx == 1) q = float4(three.x, w, three.y, three.z);
	else if (idx == 2) q = float4(three.x, three.y, w, three.z);
	else               q = float4(three.x, three.y, three.z, w);
	return q;
}

// Unpack a packed splat into usable format
GaussianSplat unpack_splat(GaussianSplatPacked p) {
	GaussianSplat s;

	s.position = float3(p.pos_x, p.pos_y, p.pos_z);
	s.rotation = unpack_quat_smallest3(p.rot_packed);

	float2 scale_xy = unpack_halfs(p.scale_xy);
	float2 scale_z_opacity = unpack_halfs(p.scale_z_opacity);
	s.scale = float3(scale_xy.x, scale_xy.y, scale_z_opacity.x);
	s.opacity = scale_z_opacity.y;

	float2 sh_dc_rg = unpack_halfs(p.sh_dc_rg);
	float2 sh_dc_b_pad = unpack_halfs(p.sh_dc_b_pad);
	s.sh_dc = float3(sh_dc_rg.x, sh_dc_rg.y, sh_dc_b_pad.x);

	// Unpack SH rest: 45 halfs from 23 uints
	[unroll]
	for (int i = 0; i < 15; i++) {
		int base = i * 3;
		int uint_idx0 = base / 2;
		int uint_idx1 = (base + 1) / 2;
		int uint_idx2 = (base + 2) / 2;

		float2 pair0 = unpack_halfs(p.sh_rest[uint_idx0]);
		float2 pair1 = unpack_halfs(p.sh_rest[uint_idx1]);
		float2 pair2 = unpack_halfs(p.sh_rest[uint_idx2]);

		s.sh_rest[i].x = (base % 2 == 0) ? pair0.x : pair0.y;
		s.sh_rest[i].y = ((base + 1) % 2 == 0) ? pair1.x : pair1.y;
		s.sh_rest[i].z = ((base + 2) % 2 == 0) ? pair2.x : pair2.y;
	}

	return s;
}

StructuredBuffer<GaussianSplatPacked> splats : register(t3, space0);
StructuredBuffer<uint>          sort_indices : register(t4, space0);

struct vsIn {
	float3 pos   : SV_POSITION;  // Quad vertex position (-1 to 1)
	float3 norm  : NORMAL;
	float2 uv    : TEXCOORD0;
	float4 color : COLOR0;
};

struct psIn {
	float4 pos        : SV_POSITION;
	float2 uv         : TEXCOORD0;     // Pixel offset from center
	float3 color      : COLOR0;
	float  opacity    : COLOR1;
	float3 conic      : TEXCOORD1;     // Inverse covariance (xx, xy, yy)
};

// Spherical harmonics constants
static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2_0 = 1.0925484305920792f;
static const float SH_C2_1 = -1.0925484305920792f;
static const float SH_C2_2 = 0.31539156525252005f;
static const float SH_C2_3 = -1.0925484305920792f;
static const float SH_C2_4 = 0.5462742152960396f;
static const float SH_C3_0 = -0.5900435899266435f;
static const float SH_C3_1 = 2.890611442640554f;
static const float SH_C3_2 = -0.4570457994644658f;
static const float SH_C3_3 = 0.3731763325901154f;
static const float SH_C3_4 = -0.4570457994644658f;
static const float SH_C3_5 = 1.445305721320277f;
static const float SH_C3_6 = -0.5900435899266435f;

// Evaluate spherical harmonics for a given direction
// Note: sh_dc is already preprocessed at load time as (f_dc * SH_C0 + 0.5)
// So DC term is used directly, higher orders add to it
float3 eval_sh(GaussianSplat splat, float3 dir) {
	// Start with preprocessed DC term (already has SH_C0 * f_dc + 0.5 applied)
	float3 result = splat.sh_dc;

	if (sh_degree < 1) return result;

	// Order 1
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	result += SH_C1 * (-y * splat.sh_rest[0] + z * splat.sh_rest[1] - x * splat.sh_rest[2]);

	if (sh_degree < 2) return result;

	// Order 2
	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, yz = y * z, xz = x * z;

	result += SH_C2_0 * xy * splat.sh_rest[3];
	result += SH_C2_1 * yz * splat.sh_rest[4];
	result += SH_C2_2 * (2.0f * zz - xx - yy) * splat.sh_rest[5];
	result += SH_C2_3 * xz * splat.sh_rest[6];
	result += SH_C2_4 * (xx - yy) * splat.sh_rest[7];

	if (sh_degree < 3) return result;

	// Order 3
	result += SH_C3_0 * y * (3.0f * xx - yy) * splat.sh_rest[8];
	result += SH_C3_1 * xy * z * splat.sh_rest[9];
	result += SH_C3_2 * y * (4.0f * zz - xx - yy) * splat.sh_rest[10];
	result += SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * splat.sh_rest[11];
	result += SH_C3_4 * x * (4.0f * zz - xx - yy) * splat.sh_rest[12];
	result += SH_C3_5 * z * (xx - yy) * splat.sh_rest[13];
	result += SH_C3_6 * x * (xx - 3.0f * yy) * splat.sh_rest[14];

	return result;
}

// Build rotation matrix from quaternion
// Input quaternion is in (w, x, y, z) order from PLY (rot_0, rot_1, rot_2, rot_3)
float3x3 quat_to_matrix(float4 q) {
	float w = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	return float3x3(
		1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - w * z), 2.0f * (x * z + w * y),
		2.0f * (x * y + w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - w * x),
		2.0f * (x * z - w * y), 2.0f * (y * z + w * x), 1.0f - 2.0f * (x * x + y * y)
	);
}

// Compute 2D covariance matrix from 3D Gaussian (in pixel space, following original 3DGS)
// Returns clip-space position to avoid redundant viewproj multiply
void compute_cov2d(float3 mean, float3 scale, float4 rotation, uint view_idx,
                   out float3 cov2d, out float4 clip_pos) {
	// Transform to clip and view space
	float4x4 V = view[view_idx];
	float4 p_view = mul(float4(mean, 1), V);
	clip_pos = mul(float4(mean, 1), viewproj[view_idx]);

	// Check if behind camera (right-handed: objects in front have negative Z)
	float view_z = -p_view.z;  // Flip to positive-forward convention
	if (view_z <= 0.1f) {
		cov2d = float3(0, 0, 0);
		clip_pos = float4(-1000, -1000, 0, 1);
		return;
	}

	// Build 3D covariance: Sigma = R * S² * R^T
	// Instead of building matrices, compute M = R * S directly by scaling R columns
	float3x3 R = quat_to_matrix(rotation);
	float3 s = exp(scale) * splat_scale;
	float3 s2 = s * s;  // Variance (σ²) along each axis

	// Sigma[i][j] = sum_k (R[i][k] * s2[k] * R[j][k])
	// Vectorized: Sigma row i = R_row_i * s2 dot R_rows
	float3 Rs0 = R[0] * s2;  // R row 0 scaled by s²
	float3 Rs1 = R[1] * s2;
	float3 Rs2 = R[2] * s2;
	float3x3 Sigma = float3x3(
		dot(Rs0, R[0]), dot(Rs0, R[1]), dot(Rs0, R[2]),
		dot(Rs1, R[0]), dot(Rs1, R[1]), dot(Rs1, R[2]),
		dot(Rs2, R[0]), dot(Rs2, R[1]), dot(Rs2, R[2])
	);

	// Focal lengths in pixels
	float2 focal_ndc = float2(projection[view_idx][0][0], projection[view_idx][1][1]);
	float2 focal = focal_ndc * screen_size * 0.5f;

	// Clamp view-space position to avoid extreme distortion at edges
	float2 tan_fov = 1.0f / focal_ndc;
	float2 lim = 1.3f * tan_fov;
	float2 t = clamp(p_view.xy / view_z, -lim, lim);
	float2 xy = t * view_z;

	// Jacobian of perspective projection (sparse - only 2 rows matter, third row is zero)
	// J = | f.x/z    0     f.x*x/z² |
	//     |   0    f.y/z   f.y*y/z² |
	// Third column: positive because view_z = -p_view.z (chain rule sign flip)
	float z_inv = 1.0f / view_z;
	float z_inv2 = z_inv * z_inv;
	float2 j_diag = focal * z_inv;           // J[0][0], J[1][1]
	float2 j_z = focal * xy * z_inv2;        // J[0][2], J[1][2]

	// W = transpose of view rotation (world->view rotation)
	// T = J * W, but J has special structure, so compute T rows directly:
	// T[0] = J[0][0] * W[0] + J[0][2] * W[2] = j_diag.x * W_col0 + j_z.x * W_col2
	// T[1] = J[1][1] * W[1] + J[1][2] * W[2] = j_diag.y * W_col1 + j_z.y * W_col2
	float3x3 W = transpose((float3x3)V);
	float3 T0 = j_diag.x * W[0] + j_z.x * W[2];
	float3 T1 = j_diag.y * W[1] + j_z.y * W[2];

	// cov2D = T * Sigma * T^T (only need 2x2 upper-left)
	// cov[0][0] = T0 · (Sigma * T0), cov[0][1] = T0 · (Sigma * T1), cov[1][1] = T1 · (Sigma * T1)
	float3 ST0 = mul(Sigma, T0);
	float3 ST1 = mul(Sigma, T1);

	// Low-pass filter of 0.3 ensures minimum ~1 pixel splat size
	cov2d = float3(dot(T0, ST0) + 0.3f, dot(T0, ST1), dot(T1, ST1) + 0.3f);
}

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn output = (psIn)0;

	// Multi-view instancing
	uint splat_idx = id / view_count;
	uint view_idx  = id % view_count;

	if (splat_idx >= splat_count) {
		output.pos = float4(-1000, -1000, 0, 1);
		return output;
	}

	// Get sorted splat index and unpack
	uint sorted_idx = sort_indices[splat_idx];
	GaussianSplat splat = unpack_splat(splats[sorted_idx]);

	// Compute 2D covariance and clip position (reused for depth)
	float3 cov2d;
	float4 clip_pos;
	compute_cov2d(splat.position, splat.scale, splat.rotation, view_idx, cov2d, clip_pos);

	// Invert 2x2 covariance to get conic
	float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
	if (det <= 0.0f) {
		output.pos = float4(-1000, -1000, 0, 1);
		return output;
	}
	float det_inv = 1.0f / det;
	float3 conic = float3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

	// Compute eigenvalues of 2D covariance (ellipse axis lengths squared)
	float mid = 0.5f * (cov2d.x + cov2d.z);
	float disc = max(0.0f, mid * mid - det);
	float sqrt_disc = sqrt(disc);
	float lambda1 = mid + sqrt_disc;
	float lambda2 = max(0.3f, mid - sqrt_disc);

	// Axis lengths in pixels (3 sigma covers 99.7%)
	float major_len = 3.0f * sqrt(lambda1);
	float minor_len = 3.0f * sqrt(lambda2);

	// Compute eigenvector for lambda1 (major axis direction)
	float2 major_axis = (abs(cov2d.y) > 1e-6f)
		? normalize(float2(cov2d.y, lambda1 - cov2d.x))
		: ((cov2d.x >= cov2d.z) ? float2(1, 0) : float2(0, 1));
	float2 minor_axis = float2(-major_axis.y, major_axis.x);

	// Cap axis lengths to prevent massive overdraw
	float2 orig_lens = float2(major_len, minor_len);
	if (max_radius > 0.0f) {
		major_len = min(major_len, max_radius);
		minor_len = min(minor_len, max_radius);
	}

	// Scale conic when capping to preserve Gaussian falloff at quad edge
	// R * S² * R^T * conic where R rotates to eigenvector basis, S² scales by cap ratio squared
	float2 cap_scale = orig_lens / float2(major_len, minor_len);  // >= 1 when capped
	float2 s2 = cap_scale * cap_scale;
	if (s2.x != 1.0f || s2.y != 1.0f) {
		// Rotation matrix from screen space to eigenvector basis
		float2x2 Rot = float2x2(major_axis.x, -major_axis.y,
		                        major_axis.y,  major_axis.x);
		float2x2 Scale = float2x2(s2.x, 0, 0, s2.y);
		float2x2 conic_mat = float2x2(conic.x, conic.y, conic.y, conic.z);
		float2x2 RSR = mul(Rot, mul(Scale, transpose(Rot)));
		float2x2 result = mul(RSR, conic_mat);
		conic = float3(result[0][0], result[0][1], result[1][1]);
	}

	// Transform quad vertex by ellipse axes (oriented rectangle)
	float2 quad_offset_pixels = input.pos.x * major_axis * major_len +
	                            input.pos.y * minor_axis * minor_len;

	// Convert to clip space offset and add to center
	float2 quad_offset_ndc = clamp(quad_offset_pixels / (screen_size * 0.5f), -1.0f, 1.0f);
	float2 screen_center = clip_pos.xy / clip_pos.w;

	output.pos = float4(screen_center + quad_offset_ndc, clip_pos.z / clip_pos.w, 1);
	output.uv = quad_offset_pixels;

	// Compute color from spherical harmonics
	// Negate direction to match 3DGS convention (affects odd SH bands 1 and 3)
	float3 view_dir = normalize(cam_pos[view_idx].xyz - splat.position);
	// DC is preprocessed with +0.5, so no additional offset needed
	// Use max(0) like Aras to clamp negative values only (allows HDR)
	float3 sh_color = max(0.0f, eval_sh(splat, view_dir));

	output.color = sh_color;

	// Sigmoid activation for opacity
	output.opacity = 1.0f / (1.0f + exp(-splat.opacity)) * opacity_scale;

	output.conic = conic;

	return output;
}

float4 ps(psIn input) : SV_TARGET {
	// Compute Gaussian falloff: power = -0.5 * d^T * conic * d
	float2 d = input.uv;
	float power = -0.5f * (input.conic.x * d.x * d.x +
	                       input.conic.z * d.y * d.y +
	                       2.0f * input.conic.y * d.x * d.y);

	if (power > 0.0f) discard;

	float alpha = min(0.99f, input.opacity * exp(power));
	if (alpha < 1.0f / 255.0f) discard;

	// Colors are already in linear space (converted at load time)
	// sRGB framebuffer will convert to sRGB for correct display
	return float4(input.color * alpha, alpha);
}
