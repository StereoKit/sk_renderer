/******************************************************************************
 * GPU Sort Scan Kernel
 * Exclusive prefix sum over partition histograms
 *
 * Dispatch: RADIX (256) workgroups - one per digit
 ******************************************************************************/

// Configuration
#define KEY_UINT
#define PAYLOAD_UINT
#define SHOULD_ASCEND
#define SORT_PAIRS

#include "gpu_sort_common.hlsli"

[numthreads(SCAN_DIM, 1, 1)]
void cs(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Scan_Main(gtid.x, gid.x);
}
