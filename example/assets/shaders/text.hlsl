//--name = text_vector

// GPU-evaluated vector text rendering using Slug algorithm.
// Evaluates quadratic Bezier curves directly in the fragment shader
// using sign-based root classification and dual-ray coverage for
// resolution-independent, artifact-free text at any scale or angle.
//
// Based on: Eric Lengyel, "GPU-Centered Font Rendering Directly from
// Glyph Outlines", JCGT 2017. Reference code released under MIT License.

#include "common.hlsli"

///////////////////////////////////////////////////////////////////////////////
// GPU Buffer Structures (must match C exactly)
///////////////////////////////////////////////////////////////////////////////

struct Curve {
	float2 p0;      // Start point
	float2 p1;      // Control point (off-curve)
	float2 p2;      // End point
};

#define H_BAND_COUNT 16   // Must match TEXT_H_BAND_COUNT in C
#define V_BAND_COUNT 16   // Must match TEXT_V_BAND_COUNT in C

struct Glyph {
	uint   curve_start;     // Base index into curves array
	uint   curve_count;     // Total curves for this glyph (all bands)
	float2 bounds_min;      // Glyph bounding box
	float2 bounds_max;
	float  advance;         // Horizontal advance
	float  lsb;             // Left side bearing
	uint   h_bands[H_BAND_COUNT]; // Horizontal bands: packed (offset << 16) | count
	uint   v_bands[V_BAND_COUNT]; // Vertical bands:   packed (offset << 16) | count
};

struct Instance {
	float3 pos;             // World position
	uint   glyph_index;     // Index into glyphs array
	float3 right;           // X axis * scale
	uint   color;           // Packed RGBA8 (0xAABBGGRR)
	float3 up;              // Y axis * scale
	uint   _pad;
};

///////////////////////////////////////////////////////////////////////////////
// Buffers
///////////////////////////////////////////////////////////////////////////////

StructuredBuffer<Instance> inst   : register(t2, space0);
StructuredBuffer<Curve>    curves : register(t3);
StructuredBuffer<Glyph>    glyphs : register(t4);

///////////////////////////////////////////////////////////////////////////////
// Vertex Shader
///////////////////////////////////////////////////////////////////////////////

struct vsIn {
	float2 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

struct psIn {
	float4 pos       : SV_POSITION;
	float2 glyph_pos : TEXCOORD0;   // Position in glyph space (em-space)
	nointerpolation uint   glyph_idx   : TEXCOORD1;
	nointerpolation float4 band_xform  : TEXCOORD2; // (v_scale, h_scale, v_offset, h_offset)
	nointerpolation uint2  band_max    : TEXCOORD3; // (v_max, h_max)
	nointerpolation uint   curve_start : TEXCOORD4;
	float3 color     : COLOR0;
};

float3 unpack_color(uint packed) {
	float r = float((packed >>  0) & 0xFF) / 255.0;
	float g = float((packed >>  8) & 0xFF) / 255.0;
	float b = float((packed >> 16) & 0xFF) / 255.0;
	return float3(r, g, b);
}

psIn vs(vsIn input, skr_ids_t ids) {
	Instance instance = inst[ids.inst];
	Glyph    glyph    = glyphs[instance.glyph_index];

	float2 glyph_size = glyph.bounds_max - glyph.bounds_min;

	// Dynamic dilation: expand quad by exactly 0.5 pixels in glyph space.
	// Project the instance origin and its right/up basis vectors to screen
	// pixels, then invert to get glyph-space size of half a pixel.
	float4 center_clip = mul(float4(instance.pos, 1), viewproj[ids.view]);
	float4 right_clip  = mul(float4(instance.pos + instance.right, 1), viewproj[ids.view]);
	float4 up_clip     = mul(float4(instance.pos + instance.up,    1), viewproj[ids.view]);

	float2 center_px = center_clip.xy / center_clip.w * screen_size.xy * 0.5;
	float2 right_px  = right_clip.xy  / right_clip.w  * screen_size.xy * 0.5;
	float2 up_px     = up_clip.xy     / up_clip.w     * screen_size.xy * 0.5;

	float pix_per_glyph_x = length(right_px - center_px);
	float pix_per_glyph_y = length(up_px    - center_px);

	// Clamp to avoid extreme expansion for sub-pixel glyphs
	float2 expand = float2(
		0.5 / max(pix_per_glyph_x, 0.5),
		0.5 / max(pix_per_glyph_y, 0.5)
	);

	// Transform quad vertex to glyph bounds with dynamic expansion
	float2 local_pos = (glyph.bounds_min - expand) + input.uv * (glyph_size + expand * 2.0);

	// Transform to world space
	float3 world_pos = instance.pos
	                 + local_pos.x * instance.right
	                 + local_pos.y * instance.up;

	// Precompute band transform: pos * scale + offset → band index
	// Avoids per-pixel reciprocals (Slug passes this as vertex attribute)
	float2 inv_size = float2(
		V_BAND_COUNT / max(glyph_size.x, 1e-6),
		H_BAND_COUNT / max(glyph_size.y, 1e-6)
	);

	psIn output;
	output.pos         = mul(float4(world_pos, 1), viewproj[ids.view]);
	output.glyph_pos   = local_pos;
	output.glyph_idx   = instance.glyph_index;
	output.band_xform  = float4(inv_size, -inv_size * glyph.bounds_min);
	output.band_max    = uint2(V_BAND_COUNT - 1, H_BAND_COUNT - 1);
	output.curve_start = glyph.curve_start;
	output.color       = unpack_color(instance.color);

	return output;
}

///////////////////////////////////////////////////////////////////////////////
// Slug Algorithm — Root Classification and Coverage
///////////////////////////////////////////////////////////////////////////////

// Classify a quadratic Bezier curve by the signs of its three control point
// y-coordinates (relative to the sample point). Returns a 2-bit code packed
// into bits 0 and 8 indicating which roots contribute to the winding number.
// This is unconditionally robust — no epsilon comparisons, no t-range checks.
// Reference: Lengyel 2017, Table 1 and Equation 2.
uint CalcRootCode(float y1, float y2, float y3) {
	uint i1 = asuint(y1) >> 31U;
	uint i2 = asuint(y2) >> 30U;
	uint i3 = asuint(y3) >> 29U;

	uint shift = (i2 & 2U) | (i1 & ~2U);
	shift = (i3 & 4U) | (shift & ~4U);

	return ((0x2E74U >> shift) & 0x0101U);
}

// Solve for the x-coordinates where a quadratic Bezier crosses y = 0.
// The polynomial is: a*t^2 - 2*b*t + c, where a = p1 - 2*p2 + p3,
// b = p1 - p2, c = p1. Returns (x at t1, x at t2).
float2 SolveHorizPoly(float4 p12, float2 p3) {
	float2 a = p12.xy - p12.zw * 2.0 + p3;
	float2 b = p12.xy - p12.zw;
	float ra = 1.0 / a.y;
	float rb = 0.5 / b.y;

	float d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
	float t1 = (b.y - d) * ra;
	float t2 = (b.y + d) * ra;

	// Nearly linear case: solve -2*b*t + c = 0
	if (abs(a.y) < 1.0 / 65536.0) t1 = t2 = p12.y * rb;

	return float2(
		(a.x * t1 - b.x * 2.0) * t1 + p12.x,
		(a.x * t2 - b.x * 2.0) * t2 + p12.x
	);
}

// Solve for the y-coordinates where a quadratic Bezier crosses x = 0.
float2 SolveVertPoly(float4 p12, float2 p3) {
	float2 a = p12.xy - p12.zw * 2.0 + p3;
	float2 b = p12.xy - p12.zw;
	float ra = 1.0 / a.x;
	float rb = 0.5 / b.x;

	float d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
	float t1 = (b.x - d) * ra;
	float t2 = (b.x + d) * ra;

	if (abs(a.x) < 1.0 / 65536.0) t1 = t2 = p12.x * rb;

	return float2(
		(a.y * t1 - b.y * 2.0) * t1 + p12.y,
		(a.y * t2 - b.y * 2.0) * t2 + p12.y
	);
}

// Combine horizontal and vertical ray coverage using proximity-based weights.
// Each ray's weight reflects how close the intersection is to the pixel center.
// The max() fallback ensures robustness when one ray has very low weight.
float CalcCoverage(float xcov, float ycov, float xwgt, float ywgt) {
	return max(
		abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0),
		min(abs(xcov), abs(ycov))
	);
}

///////////////////////////////////////////////////////////////////////////////
// Fragment Shader
///////////////////////////////////////////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	Glyph  glyph = glyphs[input.glyph_idx];
	float2 pos   = input.glyph_pos;

	// Pixel dimensions in glyph space, computed independently per axis.
	float2 emsPerPixel = fwidth(pos);
	float2 pixelsPerEm = 1.0 / emsPerPixel;

	// Band indices via precomputed scale+offset (no per-pixel reciprocals)
	int2 bandIndex = clamp(int2(pos * input.band_xform.xy + input.band_xform.zw),
	                       int2(0, 0), int2(input.band_max));

	// _______________________________________________________________
	// Horizontal ray: fire in +x direction, accumulate coverage (xcov)
	// Curves sorted by descending max-x for early-out.
	// _______________________________________________________________
	float xcov = 0.0;
	float xwgt = 0.0;

	uint h_data   = glyph.h_bands[bandIndex.y];
	uint h_offset = h_data >> 16;
	uint h_count  = h_data & 0xFFFF;
	uint h_start  = input.curve_start + h_offset;

	for (uint i = 0; i < h_count; i++) {
		Curve  c   = curves[h_start + i];
		float4 p12 = float4(c.p0, c.p1) - float4(pos, pos);
		float2 p3  = c.p2 - pos;

		// Early-out: curve's max x is past the pixel (sorted descending)
		if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) break;

		uint code = CalcRootCode(p12.y, p12.w, p3.y);
		if (code != 0U) {
			float2 r = SolveHorizPoly(p12, p3) * pixelsPerEm.x;

			if ((code & 1U) != 0U) {
				xcov += saturate(r.x + 0.5);
				xwgt  = max(xwgt, saturate(1.0 - abs(r.x) * 2.0));
			}
			if (code > 1U) {
				xcov -= saturate(r.y + 0.5);
				xwgt  = max(xwgt, saturate(1.0 - abs(r.y) * 2.0));
			}
		}
	}

	// _______________________________________________________________
	// Vertical ray: fire in +y direction, accumulate coverage (ycov)
	// Curves sorted by descending max-y for early-out.
	// _______________________________________________________________
	float ycov = 0.0;
	float ywgt = 0.0;

	uint v_data   = glyph.v_bands[bandIndex.x];
	uint v_offset = v_data >> 16;
	uint v_count  = v_data & 0xFFFF;
	uint v_start  = input.curve_start + v_offset;

	for (uint j = 0; j < v_count; j++) {
		Curve  c   = curves[v_start + j];
		float4 p12 = float4(c.p0, c.p1) - float4(pos, pos);
		float2 p3  = c.p2 - pos;

		// Early-out: curve's max y is past the pixel (sorted descending)
		if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) break;

		uint code = CalcRootCode(p12.x, p12.z, p3.x);
		if (code != 0U) {
			float2 r = SolveVertPoly(p12, p3) * pixelsPerEm.y;

			if ((code & 1U) != 0U) {
				ycov -= saturate(r.x + 0.5);
				ywgt  = max(ywgt, saturate(1.0 - abs(r.x) * 2.0));
			}
			if (code > 1U) {
				ycov += saturate(r.y + 0.5);
				ywgt  = max(ywgt, saturate(1.0 - abs(r.y) * 2.0));
			}
		}
	}

	// Combine dual-ray coverage
	float coverage = CalcCoverage(xcov, ycov, xwgt, ywgt);
	coverage = saturate(coverage);

	if (coverage < 0.01) discard;

	return float4(input.color, coverage);
}
