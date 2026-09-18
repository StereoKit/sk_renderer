//--name = postfx_depth_fog
//--fog_density = 0.15
//--clip_near   = 0.1
//--clip_far    = 100

// Depth-read postfx: fog from the depth buffer, read pixel-local as an input
// attachment. Under MSAA, sk_renderer resolves depth on-tile first, so this
// same shader works at every MSAA setting.

float fog_density;
float clip_near;
float clip_far;

[[vk::input_attachment_index(0)]] SubpassInput<float4> color;
[[vk::input_attachment_index(1)]] SubpassInput<float>  depth;

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
	float  d = depth.SubpassLoad();

	// Linearize [0,1] projected depth to view-space distance, then fog it
	float  dist = (clip_near * clip_far) / (clip_far - d * (clip_far - clip_near));
	float  fog  = saturate(exp(-dist * fog_density));
	float3 fog_color = float3(0.35, 0.4, 0.5);

	return float4(lerp(fog_color, c.rgb, fog), c.a);
}
