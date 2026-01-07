//--name = test_duplicate_slot
// Test shader with multiple textures bound to the same slot
// This should produce an error or warning from skshaderc

#include "common.hlsli"

// Two textures bound to the same register - this is a conflict
Texture2D    tex_a : register(t0);
Texture2D    tex_b : register(t0);  // Same slot as tex_a!
SamplerState tex_s : register(s0);

struct vsIn {
    float3 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

struct psIn {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

StructuredBuffer<float4x4> inst : register(t2, space0);

psIn vs(vsIn input, uint id : SV_InstanceID) {
    psIn output;
    uint inst_idx = id / view_count;
    uint view_idx = id % view_count;

    float4 world_pos = mul(float4(input.pos, 1), inst[inst_idx]);
    output.pos = mul(world_pos, viewproj[view_idx]);
    output.uv  = input.uv;
    return output;
}

float4 ps(psIn input) : SV_TARGET {
    // Try to sample from both textures
    float4 a = tex_a.Sample(tex_s, input.uv);
    float4 b = tex_b.Sample(tex_s, input.uv);
    return a + b;
}
