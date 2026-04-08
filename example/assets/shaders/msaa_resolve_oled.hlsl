//--name = msaa_resolve_oled
//
// Wide-kernel OLED subpixel-aware MSAA resolve for non-stripe layouts.
// Samples center pixel + 2 nearest from each edge neighbor (12 total,
// same cross pattern as msaa_resolve_wide), weighted per channel by
// Gaussian distance to each physical subpixel center.
//
// Subpixel layout, MSAA sample positions, and neighbor sampling pattern.
// 12 samples total: 4 from center pixel, 2 nearest from each edge neighbor.
// Samples marked * are the ones we load. (uppercase) = dominant channel.
//
// -0.5          0.0              0.5              1.0          1.5
//  .             .                .                .             .  
//. . . . . . . . . . . . . . . . ... . . . . . . . . . . . . . . . . -0.5
//  .             . *2(G)          .                . *           .
//  .             .                .                .             .
//  .             .                .                .             .
//  .             .                .                .             .
//  .  *          .                .  *3(G)         .             .
//. . . . . . . . =================================== . . . . . . . . -0.0
//  .             |      *0(G)     .                |      *0(g)  .
//  .             |                .                |             .
//  .             |                .                |             .
//  .       *1(R) |                .           *1(g)|             .
//  .             |                .                |             .
//. . . . . . . . |---------------------------------| . . . . . . . .  0.5
//  .             | *2(R)          |                | *2(B)       .
//  .             |                |                |             .
//  .             |                |                |             .
//  .             |                |                |             .
//  .  *3(R)      |                |  *3(B)         |             .
//. . . . . . . . =================================== . . . . . . . .  1.0
//  .             .      *0(R)     .                .      *      .
//  .             .                .                .             .
//  .             .                .                .             .
//  .        *    .                .          *1(B) .             .
//  .             .                .                .             .  
//. . . . . . . . . . . . . . . . .. . . . . . . . .. . . . . . . . . . 1.5
//  .             .                .                .             .  
//
// Weights: w = exp(-d^2 / 2*sigma^2), sigma = 0.4, normalized per channel.

Texture2DMS<float4, 4> msaa_color : register(t0);

// Center pixel weights [s0, s1, s2, s3] per channel
//                            s0      s1      s2      s3
static const float4 WC_R = { 0.2811, 0.1901, 0.9070, 0.6136 };
static const float4 WC_G = { 0.9070, 0.6136, 0.4153, 0.2811 };
static const float4 WC_B = { 0.1901, 0.6136, 0.2811, 0.9070 };

// Neighbor weights {R, G, B} — same sample pairs as msaa_resolve_wide
//                                    R       G       B
static const float3 W_RIGHT_NEAR = { 0.0869, 0.1901, 0.6136 }; // s2 at (1.125, 0.625)
static const float3 W_RIGHT_FAR  = { 0.0057, 0.0869, 0.0869 }; // s0 at (1.375, 0.125)
static const float3 W_LEFT_NEAR  = { 0.4153, 0.2811, 0.0588 }; // s1 at (-0.125, 0.375)
static const float3 W_LEFT_FAR   = { 0.2811, 0.0270, 0.0183 }; // s3 at (-0.375, 0.875)
static const float3 W_DOWN_NEAR  = { 0.6136, 0.0869, 0.4153 }; // s0 at (0.375, 1.125)
static const float3 W_DOWN_FAR   = { 0.0869, 0.0124, 0.2811 }; // s1 at (0.875, 1.375)
static const float3 W_UP_NEAR    = { 0.0588, 0.6136, 0.0869 }; // s3 at (0.625, -0.125)
static const float3 W_UP_FAR     = { 0.0183, 0.1901, 0.0057 }; // s2 at (0.125, -0.375)

static const float3 W_TOTAL_INV = 1.0 / float3(3.5584, 3.7051, 3.5584);

struct psIn {
	float4 pos : SV_POSITION;
};

psIn vs(uint id : SV_VertexID) {
	psIn o;
	float2 uv = float2(id & 2, (id << 1) & 2);
	o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	int2 p = int2(input.pos.xy);

	// Center pixel
	float4 s0 = msaa_color.Load(p, 0);
	float4 s1 = msaa_color.Load(p, 1);
	float4 s2 = msaa_color.Load(p, 2);
	float4 s3 = msaa_color.Load(p, 3);

	float3 accum = float3(
		dot(float4(s0.r, s1.r, s2.r, s3.r), WC_R),
		dot(float4(s0.g, s1.g, s2.g, s3.g), WC_G),
		dot(float4(s0.b, s1.b, s2.b, s3.b), WC_B));

	// Right (+1,0): samples 2 (near) and 0 (far)
	int2 r = p + int2(1, 0);
	accum += msaa_color.Load(r, 2).rgb * W_RIGHT_NEAR + msaa_color.Load(r, 0).rgb * W_RIGHT_FAR;

	// Left (-1,0): samples 1 (near) and 3 (far)
	int2 l = p - int2(1, 0);
	accum += msaa_color.Load(l, 1).rgb * W_LEFT_NEAR + msaa_color.Load(l, 3).rgb * W_LEFT_FAR;

	// Down (0,+1): samples 0 (near) and 1 (far)
	int2 d = p + int2(0, 1);
	accum += msaa_color.Load(d, 0).rgb * W_DOWN_NEAR + msaa_color.Load(d, 1).rgb * W_DOWN_FAR;

	// Up (0,-1): samples 3 (near) and 2 (far)
	int2 u = p - int2(0, 1);
	accum += msaa_color.Load(u, 3).rgb * W_UP_NEAR + msaa_color.Load(u, 2).rgb * W_UP_FAR;

	return float4(accum * W_TOTAL_INV, (s0.a + s1.a + s2.a + s3.a) * 0.25);
}
