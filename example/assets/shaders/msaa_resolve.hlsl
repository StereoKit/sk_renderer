//--name = msaa_resolve

[[vk::input_attachment_index(0)]] SubpassInputMS<float4> color;

struct psIn {
	float4 pos : SV_POSITION;
};

psIn vs(uint id : SV_VertexID) {
	psIn o;
	float2 uv = float2((id << 1) & 2, id & 2);
	o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	float4 c = (color.SubpassLoad(0)
	          + color.SubpassLoad(1)
	          + color.SubpassLoad(2)
	          + color.SubpassLoad(3)) * 0.25;
	float3 s   = sqrt(c.rgb);
	float3 inv = 1.0 - 2.0 * s + c.rgb;
	return float4(inv, c.a);
}
