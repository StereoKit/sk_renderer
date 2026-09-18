// Writes a deterministic pattern for scene_tex_copy's buffer readback check.

RWStructuredBuffer<uint> output : register(u1);

uint count;

[numthreads(64, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (id.x < count)
		output[id.x] = id.x * 2654435761u + 42u;
}
