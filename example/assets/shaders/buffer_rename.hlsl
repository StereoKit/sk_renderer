// Copies a dynamic constant buffer's value into one output slot, so
// scene_tex_copy can check each dispatch read what was set just before it.

cbuffer RenameParams : register(b0) {
	uint  value;
	uint  slot;
	uint2 _pad;
};

RWStructuredBuffer<uint> output : register(u1);

[numthreads(1, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	output[slot] = value;
}
