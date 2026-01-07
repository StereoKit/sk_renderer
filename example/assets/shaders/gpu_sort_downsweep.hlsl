/******************************************************************************
 * GPU Sort Downsweep Kernel
 * Ranks keys using wave-level multi-split and scatters to sorted positions
 *
 * Dispatch: thread_blocks workgroups
 ******************************************************************************/

// Configuration
#define KEY_UINT
#define PAYLOAD_UINT
#define SHOULD_ASCEND
#define SORT_PAIRS

#include "gpu_sort_common.hlsli"

[numthreads(D_DIM, 1, 1)]
void cs(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Downsweep_Main(gtid.x, gid.x);
}
