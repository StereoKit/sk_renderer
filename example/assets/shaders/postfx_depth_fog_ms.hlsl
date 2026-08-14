//--name = postfx_depth_fog_ms
//--fog_density = 0.15
//--clip_near   = 0.1
//--clip_far    = 100

// The MSAA half of postfx_depth_fog. Declaring depth as SubpassInputMS reads
// the multisampled attachment directly, so the pass skips the on-tile depth
// resolve and the 1x transient it resolves into.
//
// Sample 0 is what the resolve would have produced anyway (SAMPLE_ZERO is the
// only resolve mode Vulkan mandates), so this renders identically to the
// non-MS shader and the two modes are a like-for-like A/B on the resolve cost.
// Reading samples 1..N-1 is the capability the resolve path can't offer.
//
// The type is baked into the SPIR-V, so this shader only runs in a pass that
// actually has MSAA. A 1x pass is an error, not a fallback.

float fog_density;
float clip_near;
float clip_far;

[[vk::input_attachment_index(0)]] SubpassInput<float4>  color;
[[vk::input_attachment_index(1)]] SubpassInputMS<float> depth;

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

psIn vs(uint id : SV_VertexID) {
	psIn o;
	float2 uv = float2(id & 2, (id << 1) & 2);
	o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
	o.uv  = uv;
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	float4 c = color.SubpassLoad();
	float  d = depth.SubpassLoad(0);

	// Linearize [0,1] projected depth to view-space distance, then fog it
	float  dist = (clip_near * clip_far) / (clip_far - d * (clip_far - clip_near));
	float  fog  = saturate(exp(-dist * fog_density));
	float3 fog_color = float3(0.35, 0.4, 0.5);

	return float4(lerp(fog_color, c.rgb, fog), c.a);
}
