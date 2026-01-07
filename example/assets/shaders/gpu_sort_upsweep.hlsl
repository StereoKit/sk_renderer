/******************************************************************************
 * GPU Sort Upsweep Kernel
 * Builds per-partition histograms and reduces to global histogram
 *
 * Dispatch: thread_blocks workgroups
 ******************************************************************************/

// Configuration
#define KEY_UINT
#define PAYLOAD_UINT
#define SHOULD_ASCEND
#define SORT_PAIRS

#include "gpu_sort_common.hlsli"

[numthreads(US_DIM, 1, 1)]
void cs(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Upsweep_Main(gtid.x, gid.x);
}
