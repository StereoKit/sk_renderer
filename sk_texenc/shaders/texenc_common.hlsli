// IEC 61966-2-1 transfer function. Used by the SRGB_ENCODE pipeline variants
// to gamma-encode linear-light sources (float or sRGB-view textures, which
// Load as linear) so endpoints quantize in the space the *_srgb output
// format decodes from.
float3 linear_to_srgb(float3 c) {
	c = max(c, 0.0);
	float3 lo = c * 12.92;
	float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
	return lerp(lo, hi, step(0.0031308, c));
}

// Axis-fit channel weights. Color leans on green the way the eye does (R2 G4
// B1 A2), while LINEAR_DATA weighs channels evenly since each one is its own
// signal. Encoders that use these declare LINEAR_DATA before including.
#define PW_R     (LINEAR_DATA ? 1.0 : 2.0)
#define PW_G     (LINEAR_DATA ? 1.0 : 4.0)
#define PW_A     (LINEAR_DATA ? 1.0 : 2.0)
#define PW_R_INV (LINEAR_DATA ? 1.0 : 0.5)
#define PW_G_INV (LINEAR_DATA ? 1.0 : 0.25)
#define PW_A_INV (LINEAR_DATA ? 1.0 : 0.5)
