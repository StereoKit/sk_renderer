//--name = upscale

// UV scale: maps [0,1] output UVs to the active viewport region of the
// source texture. When viewport fills the whole render target, this is (1,1).
float2 uv_scale;

Texture2D    src_tex   : register(t0);
SamplerState src_tex_s : register(s0);

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

psIn vs(uint id : SV_VertexID) {
	psIn o;
	o.uv  = float2(id & 2, (id << 1) & 2);
	o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	return src_tex.Sample(src_tex_s, input.uv * uv_scale);
}
