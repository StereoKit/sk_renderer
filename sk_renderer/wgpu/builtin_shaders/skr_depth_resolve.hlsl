//--name = skr_depth_resolve
// WebGPU-only helper: resolves a multisampled depth buffer to a single-sample
// r32f color target so postfx shaders can read scene depth. WebGPU render
// passes have no depth resolve, and postfx depth reads lower to plain texture
// loads there anyway (SVSL's WGSL emitter), so an r32f copy of sample 0 serves
// exactly as well as a real depth attachment — this mirrors the Vulkan
// backend's SAMPLE_ZERO on-tile depth resolve. The input attachment is
// deliberately NOT named "depth" so the runtime binds it as the stage's
// primary (color-slot) input.

[[vk::input_attachment_index(0)]] SubpassInputMS<float> source;

struct psIn {
	float4 pos : SV_POSITION;
};

psIn vs(uint id : SV_VertexID) {
	psIn o;
	float2 uv = float2(id & 2, (id << 1) & 2);
	o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
	return o;
}

float ps(psIn input) : SV_Target {
	return source.SubpassLoad(0);
}
