/******************************************************************************
 * GPU Sort Init Kernel
 * Clears global histogram and computes initial sort keys from splat depths
 *
 * Pass 0: Clear global histogram (dispatch 4 workgroups)
 * Pass 1: Compute depths and initialize keys/payloads (dispatch splat_count/256)
 ******************************************************************************/

// Configuration
#define KEY_UINT
#define PAYLOAD_UINT
#define SHOULD_ASCEND
#define SORT_PAIRS

// Additional uniforms for init pass
uint   e_initPass;      // 0 = clear histogram, 1 = compute depths
float3 e_camPos;        // Camera position for depth calculation

// Packed Gaussian splat structure (must match C side)
struct GaussianSplatPacked {
    float  pos_x, pos_y, pos_z;
    uint   rot_packed;
    uint   scale_xy;
    uint   scale_z_opacity;
    uint   sh_dc_rg;
    uint   sh_dc_b_pad;
    uint   sh_rest[23];
};

StructuredBuffer<GaussianSplatPacked> splats : register(t0);

// Include core sort definitions (for buffer declarations)
#include "gpu_sort_common.hlsli"

// Convert float to sortable uint (Herf's radix tricks)
// Handles negative numbers correctly for sorting
uint FloatToSortableUint(float f)
{
    uint fu = asuint(f);
    uint mask = -((int)(fu >> 31)) | 0x80000000;
    return fu ^ mask;
}

// Compute distance squared from camera (for back-to-front sorting)
float ComputeDepth(float3 position)
{
    float3 d = position - e_camPos;
    return dot(d, d);  // distance squared
}

// Shared memory for exclusive prefix sum in pass 2
groupshared uint g_prefixSum[256];

[numthreads(256, 1, 1)]
void cs(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint idx = dtid.x;

    if (e_initPass == 0)
    {
        // Pass 0: Clear global histogram (RADIX * 4 = 1024 entries)
        if (idx < RADIX * RADIX_PASSES)
        {
            b_globalHist[idx] = 0;
        }
    }
    else if (e_initPass == 1)
    {
        // Pass 1: Compute depths and initialize keys/payloads
        if (idx < e_numKeys)
        {
            float3 pos = float3(splats[idx].pos_x, splats[idx].pos_y, splats[idx].pos_z);
            float depth = ComputeDepth(pos);

            // Front-to-back sorting (smaller distances first) for "under" operator
            // Use sortable uint for correct radix sort ordering
            b_sort[idx] = FloatToSortableUint(depth);
            b_sortPayload[idx] = idx;
        }
    }
    else if (e_initPass == 2)
    {
        // Pass 2: Convert globalHist from counts to exclusive prefix sums
        // Dispatch with 1 workgroup, e_radixShift determines which section to process
        // gtid.x is thread index (0-255)
        uint passOffset = GlobalHistOffset();  // e_radixShift << 5 = (radixShift/8) * 256

        // Load count into shared memory
        uint count = b_globalHist[passOffset + gtid.x];
        g_prefixSum[gtid.x] = count;
        GroupMemoryBarrierWithGroupSync();

        // Hillis-Steele inclusive prefix sum
        for (uint offset = 1; offset < 256; offset <<= 1)
        {
            uint temp = 0;
            if (gtid.x >= offset)
                temp = g_prefixSum[gtid.x - offset];
            GroupMemoryBarrierWithGroupSync();
            g_prefixSum[gtid.x] += temp;
            GroupMemoryBarrierWithGroupSync();
        }

        // Convert to exclusive (shift right, insert 0 at start)
        uint exclusive = (gtid.x == 0) ? 0 : g_prefixSum[gtid.x - 1];

        // Write back
        b_globalHist[passOffset + gtid.x] = exclusive;
    }
}
