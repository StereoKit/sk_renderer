//--name = text_vector

// GPU-evaluated vector text rendering
// Evaluates quadratic Bezier curves directly in the fragment shader
// for resolution-independent text at any scale or angle.

#include "common.hlsli"

///////////////////////////////////////////////////////////////////////////////
// GPU Buffer Structures (must match C exactly)
///////////////////////////////////////////////////////////////////////////////

struct Curve {
	float2 p0;      // Start point
	float2 p1;      // Control point
	float2 p2;      // End point
	float  x_min;   // Curve AABB (precomputed)
	float  x_max;
	float  y_min;
	float  y_max;
};

#define BAND_COUNT 32    // Must match TEXT_BAND_COUNT in C

struct Glyph {
	uint   curve_start;     // Base index into curves array
	uint   curve_count;     // Total number of curves for this glyph
	float2 bounds_min;      // Glyph bounding box
	float2 bounds_max;
	float  advance;         // Horizontal advance
	float  lsb;             // Left side bearing
	uint   bands[BAND_COUNT]; // Packed (offset << 16) | count per band
};

struct Instance {
	float3 pos;             // World position - 12 bytes, offset 0
	uint   glyph_index;     // Index into glyphs array - 4 bytes, offset 12
	float3 right;           // X axis * scale - 12 bytes, offset 16
	uint   color;           // Packed RGBA8 (0xAABBGGRR) - 4 bytes, offset 28
	float3 up;              // Y axis * scale - 12 bytes, offset 32
	uint   _pad;            // Padding - 4 bytes, offset 44
};                          // 48 bytes total

///////////////////////////////////////////////////////////////////////////////
// Buffers
///////////////////////////////////////////////////////////////////////////////

// Instance data - passed via skr_render_list_add (automatic instancing)
StructuredBuffer<Instance> inst : register(t2, space0);

// Font data - bound via skr_material_set_buffer
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
	float2 glyph_uv  : TEXCOORD0;   // UV in glyph bounds [0,1]
	float2 glyph_pos : TEXCOORD1;   // Position in glyph space
	nointerpolation uint glyph_idx : TEXCOORD2;
	float3 color     : COLOR0;
};

// Unpack RGBA8 color from uint (0xAABBGGRR format)
float3 unpack_color(uint packed) {
	float r = float((packed >>  0) & 0xFF) / 255.0;
	float g = float((packed >>  8) & 0xFF) / 255.0;
	float b = float((packed >> 16) & 0xFF) / 255.0;
	return float3(r, g, b);
}

psIn vs(vsIn input, skr_ids_t ids) {
	Instance instance = inst[ids.inst];
	Glyph glyph = glyphs[instance.glyph_index];

	// Transform quad vertex to glyph bounds with small expansion for edge pixels
	// This ensures pixels at curve boundaries have room for anti-aliasing
	float2 glyph_size = glyph.bounds_max - glyph.bounds_min;
	float2 expand     = glyph_size * 0.02; // 2% expansion on each side
	float2 local_pos  = (glyph.bounds_min - expand) + input.uv * (glyph_size + expand * 2.0);

	// Transform to world space using position + right/up vectors
	// This is simpler and faster than full matrix multiply for 2D glyphs
	float3 world_pos = instance.pos
	                 + local_pos.x * instance.right
	                 + local_pos.y * instance.up;

	psIn output;
	output.pos       = mul(float4(world_pos, 1), viewproj[ids.view]);
	output.glyph_uv  = input.uv;
	output.glyph_pos = local_pos;
	output.glyph_idx = instance.glyph_index;
	output.color     = unpack_color(instance.color);

	return output;
}

///////////////////////////////////////////////////////////////////////////////
// Curve Evaluation Functions
///////////////////////////////////////////////////////////////////////////////

// Evaluate only X coordinate of quadratic Bezier at parameter t
float bezier_x(float x0, float x1, float x2, float t) {
	float it = 1.0 - t;
	return it * it * x0 + 2.0 * it * t * x1 + t * t * x2;
}

// Calculate coverage contribution from a single MONOTONIC curve.
// Curves are preprocessed to be monotonic in Y, meaning each curve
// only goes up OR down, never both. This simplifies the math significantly.
//
// Returns coverage contribution: positive for exit, negative for entry.
// Based on Sebastian Lague's approach.
float curve_coverage(float2 pos, Curve c, float inv_pixel_size) {
	// Translate curve so pos is at origin
	float2 p0 = c.p0 - pos;
	float2 p2 = c.p2 - pos;

	// Determine curve direction (is it going down in Y?)
	// Downward = exiting shape (winding +), Upward = entering shape (winding -)
	bool is_downward = p0.y > p2.y;

	// Skip curves entirely above or below the ray using asymmetric comparisons.
	// This handles shared endpoints correctly: the "exiting" endpoint gets
	// the inclusive boundary to avoid double-counting.
	if (is_downward) {
		if (p0.y < 0.0 && p2.y <= 0.0) return 0.0;  // Both below
		if (p0.y > 0.0 && p2.y >= 0.0) return 0.0;  // Both above
	} else {
		if (p0.y <= 0.0 && p2.y < 0.0) return 0.0;  // Both below
		if (p0.y >= 0.0 && p2.y > 0.0) return 0.0;  // Both above
	}

	// Quick X rejection: curve entirely to left of ray
	if (c.x_max < pos.x) return 0.0;

	float2 p1 = c.p1 - pos;

	// Quadratic coefficients for y(t) = a*t^2 + b*t + c
	float a = p0.y - 2.0 * p1.y + p2.y;
	float b = 2.0 * (p1.y - p0.y);
	float c_coef = p0.y;

	// Find intersection with y=0
	float t = 0.0;
	const float epsilon = 1e-4;

	if (abs(a) < 1e-6) {
		// Linear case
		if (abs(b) > 1e-6) {
			t = -c_coef / b;
		}
	} else {
		// Quadratic case - with monotonic curves, at most one root in [0,1]
		float discriminant = b * b - 4.0 * a * c_coef;
		if (discriminant >= -epsilon) {
			float sqrt_d = sqrt(max(0.0, discriminant));
			float inv_2a = 0.5 / a;
			float t1 = (-b - sqrt_d) * inv_2a;
			float t2 = (-b + sqrt_d) * inv_2a;

			// Pick the root that's in [0, 1]
			if (t1 >= -epsilon && t1 <= 1.0 + epsilon) t = t1;
			else if (t2 >= -epsilon && t2 <= 1.0 + epsilon) t = t2;
			else return 0.0;
		} else {
			return 0.0;
		}
	}

	// Check if intersection is valid
	t = saturate(t);

	// Calculate X position of intersection
	float intersect_x = a * t * t + b * t + c_coef;  // Wait, this is Y, need X
	float ax = p0.x - 2.0 * p1.x + p2.x;
	float bx = 2.0 * (p1.x - p0.x);
	intersect_x = ax * t * t + bx * t + p0.x;

	// Calculate coverage: 0 at left edge of pixel, 1 at right edge
	// This gives smooth AA based on where the intersection falls within the pixel
	float coverage = saturate(0.5 + intersect_x * inv_pixel_size);

	// Sign based on direction: downward = exit = positive, upward = entry = negative
	return is_downward ? coverage : -coverage;
}

// Analytical squared distance to quadratic Bezier curve (Inigo Quilez method)
// Only called for edge pixels where winding-based AA isn't sufficient
float curve_distance_sq(float2 pos, Curve c) {
	float2 A = c.p0;
	float2 B = c.p1;
	float2 C = c.p2;

	float2 a = B - A;
	float2 b = A - 2.0 * B + C;
	float2 cc = a * 2.0;
	float2 d = A - pos;

	float bb = dot(b, b);
	if (bb < 1e-8) {
		float2 ac = C - A;
		float t = clamp(dot(pos - A, ac) / dot(ac, ac), 0.0, 1.0);
		float2 diff = pos - (A + ac * t);
		return dot(diff, diff);
	}

	float kk = 1.0 / bb;
	float kx = kk * dot(a, b);
	float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
	float kz = kk * dot(d, a);

	float p  = ky - kx * kx;
	float p3 = p * p * p;
	float q  = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
	float h  = q * q + 4.0 * p3;

	float res;
	if (h >= 0.0) {
		h = sqrt(h);
		float2 x  = (float2(h, -h) - q) / 2.0;
		float2 uv = sign(x) * pow(abs(x), 1.0 / 3.0);
		float  t  = clamp(uv.x + uv.y - kx, 0.0, 1.0);
		float2 dd = d + (cc + b * t) * t;
		res = dot(dd, dd);
	} else {
		float z = sqrt(-p);
		float v = acos(q / (p * z * 2.0)) / 3.0;
		float m = cos(v);
		float n = sin(v) * 1.732050808;
		float3 t = clamp(float3(m + m, -n - m, n - m) * z - kx, 0.0, 1.0);
		float2 d1 = d + (cc + b * t.x) * t.x;
		float2 d2 = d + (cc + b * t.y) * t.y;
		res = min(dot(d1, d1), dot(d2, d2));
	}
	return res;
}

///////////////////////////////////////////////////////////////////////////////
// Fragment Shader
///////////////////////////////////////////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	Glyph  glyph = glyphs[input.glyph_idx];
	float2 pos   = input.glyph_pos;

	// Compute pixel size in glyph space for anti-aliasing
	float pixel_size = length(fwidth(pos));
	float inv_pixel_size = 1.0 / pixel_size;

	// Determine which band this pixel falls into
	float glyph_height = glyph.bounds_max.y - glyph.bounds_min.y;
	float normalized_y = (pos.y - glyph.bounds_min.y) / max(glyph_height, 1e-6);
	uint  band_idx     = clamp((uint)(normalized_y * BAND_COUNT), 0, BAND_COUNT - 1);

	// Unpack band data
	uint band_data   = glyph.bands[band_idx];
	uint band_offset = band_data >> 16;
	uint band_count  = band_data & 0xFFFF;
	uint curve_start = glyph.curve_start + band_offset;

	// Calculate winding using coverage (handles monotonic curves correctly)
	float coverage = 0.0;
	for (uint i = 0; i < band_count; i++) {
		Curve c = curves[curve_start + i];
		coverage += curve_coverage(pos, c, inv_pixel_size);
	}
	coverage = saturate(coverage);

	// Early out for clearly outside pixels
	if (coverage < 0.01) {
		discard;
	}

	// For solidly inside pixels, skip SDF - just use full opacity
	// This prevents internal curves from causing AA bleeding
	if (coverage > 0.99) {
		return float4(input.color, 1.0);
	}

	// Edge pixel: use SDF for smooth anti-aliasing
	bool is_inside = coverage > 0.5;

	// Find minimum squared distance to any curve in this band
	float min_dist_sq = 1e10;
	for (uint j = 0; j < band_count; j++) {
		Curve c = curves[curve_start + j];
		// Quick AABB rejection - expand by 1 pixel for edge detection
		if (pos.x >= c.x_min - pixel_size && pos.x <= c.x_max + pixel_size &&
		    pos.y >= c.y_min - pixel_size && pos.y <= c.y_max + pixel_size) {
			float dist_sq = curve_distance_sq(pos, c);
			min_dist_sq = min(min_dist_sq, dist_sq);
		}
	}

	float dist = sqrt(min_dist_sq);

	// Use signed distance for anti-aliasing
	// Sign comes from coverage (handles monotonic curves correctly)
	float signed_dist = is_inside ? -dist : dist;
	float alpha = saturate(0.5 - signed_dist * inv_pixel_size);

	if (alpha < 0.01) {
		discard;
	}

	return float4(input.color, alpha);
}
