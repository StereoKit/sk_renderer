//--name = tex_psnr

// GPU-side PSNR support: computes per-group partial sums of squared error
// between two same-size textures at mip 0. The CPU reduces the partials and
// finishes the PSNR math (tools/tex_psnr.c).
//
// Comparing through Texture2D.Load means the compressed texture goes through
// the hardware decoder — so this measures exactly what rendering will sample,
// for any format the GPU can decode (BC on desktop, ASTC on device).
//
// Error is measured in 8-bit sRGB-encoded RGB space to match the standard
// PSNR convention (and the old CPU tooling): both inputs Load as linear
// (sRGB and float formats alike), get gamma-encoded, and are rounded to
// 8-bit steps before differencing.

Texture2D<float4>         tex_ref     : register(t0);
Texture2D<float4>         tex_cmp     : register(t1);
RWStructuredBuffer<float> partial_sse : register(u2);

uint image_width;
uint image_height;
uint groups_x;

groupshared float g_sums[64];

float3 linear_to_srgb(float3 c) {
	// IEC 61966-2-1 transfer function
	c = max(c, 0.0);
	float3 lo = c * 12.92;
	float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
	return lerp(lo, hi, step(0.0031308, c));
}

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
	float sse = 0;
	if (id.x < image_width && id.y < image_height) {
		float3 a = tex_ref.Load(int3(id.xy, 0)).rgb;
		float3 b = tex_cmp.Load(int3(id.xy, 0)).rgb;
		a = round(saturate(linear_to_srgb(a)) * 255.0);
		b = round(saturate(linear_to_srgb(b)) * 255.0);
		float3 d = a - b;
		sse = dot(d, d);
	}

	uint li = gtid.y * 8 + gtid.x;
	g_sums[li] = sse;
	GroupMemoryBarrierWithGroupSync();

	[unroll] for (uint s = 32; s > 0; s >>= 1) {
		if (li < s) g_sums[li] += g_sums[li + s];
		GroupMemoryBarrierWithGroupSync();
	}

	if (li == 0)
		partial_sse[gid.y * groups_x + gid.x] = g_sums[0];
}
