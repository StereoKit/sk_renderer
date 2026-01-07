/******************************************************************************
 * GPU Radix Sort for sk_renderer
 * Adapted from GPUSorting by Thomas Smith (b0nes164)
 * https://github.com/b0nes164/GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 5/17/2024
 *
 * 8-bit LSD radix sort with reduce-then-scan approach.
 * Uses wave intrinsics for stable, correct sorting.
 ******************************************************************************/

#ifndef GPU_SORT_COMMON_HLSLI
#define GPU_SORT_COMMON_HLSLI

//=============================================================================
// CONFIGURATION - These must be defined before including this file:
//   KEY_UINT, PAYLOAD_UINT, SHOULD_ASCEND, SORT_PAIRS, VULKAN
//=============================================================================

//=============================================================================
// CONSTANTS
//=============================================================================
#define KEYS_PER_THREAD     15U
#define D_DIM               256U        // Downsweep threads
#define US_DIM              128U        // Upsweep threads
#define SCAN_DIM            128U        // Scan threads
#define PART_SIZE           3840U       // D_DIM * KEYS_PER_THREAD
#define D_TOTAL_SMEM        4096U

#define RADIX               256U        // Number of digit bins
#define RADIX_MASK          255U        // Mask of digit bins
#define HALF_RADIX          128U        // For smaller waves where bit packing is necessary
#define HALF_MASK           127U
#define RADIX_LOG           8U          // log2(RADIX)
#define RADIX_PASSES        4U          // (Key width) / RADIX_LOG

//=============================================================================
// UNIFORMS (sk_renderer loose uniforms instead of cbuffer)
//=============================================================================
uint e_numKeys;
uint e_radixShift;
uint e_threadBlocks;

//=============================================================================
// BUFFERS
//=============================================================================
RWStructuredBuffer<uint> b_sort        : register(u0);
RWStructuredBuffer<uint> b_alt         : register(u1);
RWStructuredBuffer<uint> b_sortPayload : register(u2);
RWStructuredBuffer<uint> b_altPayload  : register(u3);
RWStructuredBuffer<uint> b_globalHist  : register(u4);
RWStructuredBuffer<uint> b_passHist    : register(u5);

//=============================================================================
// SHARED MEMORY
//=============================================================================
groupshared uint g_us[RADIX * 2];       // Upsweep shared memory
groupshared uint g_scan[SCAN_DIM];      // Scan shared memory
groupshared uint g_d[D_TOTAL_SMEM];     // Downsweep shared memory

//=============================================================================
// STRUCTURES
//=============================================================================
struct KeyStruct
{
    uint k[KEYS_PER_THREAD];
};

struct OffsetStruct
{
    uint o[KEYS_PER_THREAD];
};

struct DigitStruct
{
    uint d[KEYS_PER_THREAD];
};

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

// Due to a bug with SPIRV pre 1.6, we cannot use WaveGetLaneCount()
// Use fixed wave size - AMD RDNA (GFX10+) uses Wave32 in compute.
// TODO: Make this configurable via specialization constants.
#define FIXED_WAVE_SIZE 32

inline uint getWaveSize()
{
    return FIXED_WAVE_SIZE;
}

// Compute lane index from group thread ID to avoid multiple WaveGetLaneIndex() calls
// which cause duplicate SubgroupLocalInvocationId in SPIRV output.
inline uint getLaneIndex(uint gtid)
{
    return gtid % FIXED_WAVE_SIZE;
}

inline uint getWaveIndex(uint gtid, uint waveSize)
{
    return gtid / waveSize;
}

// Radix Tricks by Michael Herf - http://stereopsis.com/radix.html
inline uint FloatToUint(float f)
{
    uint mask = -((int)(asuint(f) >> 31)) | 0x80000000;
    return asuint(f) ^ mask;
}

inline float UintToFloat(uint u)
{
    uint mask = ((u >> 31) - 1) | 0x80000000;
    return asfloat(u ^ mask);
}

inline uint getWaveCountPass(uint waveSize)
{
    return D_DIM / waveSize;
}

inline uint ExtractDigit(uint key)
{
    return key >> e_radixShift & RADIX_MASK;
}

inline uint ExtractDigit(uint key, uint shift)
{
    return key >> shift & RADIX_MASK;
}

inline uint ExtractPackedIndex(uint key)
{
    return key >> (e_radixShift + 1) & HALF_MASK;
}

inline uint ExtractPackedShift(uint key)
{
    return (key >> e_radixShift & 1) ? 16 : 0;
}

inline uint ExtractPackedValue(uint packed, uint key)
{
    return packed >> ExtractPackedShift(key) & 0xffff;
}

inline uint SubPartSizeWGE16(uint waveSize)
{
    return KEYS_PER_THREAD * waveSize;
}

inline uint SharedOffsetWGE16(uint gtid, uint waveSize)
{
    return WaveGetLaneIndex() + getWaveIndex(gtid, waveSize) * SubPartSizeWGE16(waveSize);
}

inline uint SubPartSizeWLT16(uint waveSize, uint _serialIterations)
{
    return KEYS_PER_THREAD * waveSize * _serialIterations;
}

inline uint SharedOffsetWLT16(uint gtid, uint waveSize, uint _serialIterations)
{
    return WaveGetLaneIndex() +
        (getWaveIndex(gtid, waveSize) / _serialIterations * SubPartSizeWLT16(waveSize, _serialIterations)) +
        (getWaveIndex(gtid, waveSize) % _serialIterations * waveSize);
}

inline uint DeviceOffsetWGE16(uint gtid, uint waveSize, uint partIndex)
{
    return SharedOffsetWGE16(gtid, waveSize) + partIndex * PART_SIZE;
}

inline uint DeviceOffsetWLT16(uint gtid, uint waveSize, uint partIndex, uint serialIterations)
{
    return SharedOffsetWLT16(gtid, waveSize, serialIterations) + partIndex * PART_SIZE;
}

inline uint GlobalHistOffset()
{
    return e_radixShift << 5;
}

inline uint WaveHistsSizeWGE16(uint waveSize)
{
    return D_DIM / waveSize * RADIX;
}

inline uint WaveHistsSizeWLT16()
{
    return D_TOTAL_SMEM;
}

inline uint SerialIterations(uint waveSize)
{
    return (D_DIM / waveSize + 31) >> 5;
}

inline void ClearWaveHists(uint gtid, uint waveSize)
{
    const uint histsEnd = waveSize >= 16 ?
        WaveHistsSizeWGE16(waveSize) : WaveHistsSizeWLT16();
    for (uint i = gtid; i < histsEnd; i += D_DIM)
        g_d[i] = 0;
}

inline void LoadKey(inout uint key, uint index)
{
    key = b_sort[index];
}

inline void LoadDummyKey(inout uint key)
{
    key = 0xffffffff;
}

inline uint WaveFlagsWGE16(uint waveSize)
{
    return (waveSize & 31) ? (1U << waveSize) - 1 : 0xffffffff;
}

inline uint WaveFlagsWLT16(uint waveSize)
{
    return (1U << waveSize) - 1;
}

//=============================================================================
// KEY LOADING
//=============================================================================
inline KeyStruct LoadKeysWGE16(uint gtid, uint waveSize, uint partIndex)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, waveSize, partIndex);
        i < KEYS_PER_THREAD;
        ++i, t += waveSize)
    {
        LoadKey(keys.k[i], t);
    }
    return keys;
}

inline KeyStruct LoadKeysWLT16(uint gtid, uint waveSize, uint partIndex, uint serialIterations)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, waveSize, partIndex, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += waveSize * serialIterations)
    {
        LoadKey(keys.k[i], t);
    }
    return keys;
}

inline KeyStruct LoadKeysPartialWGE16(uint gtid, uint waveSize, uint partIndex)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, waveSize, partIndex);
        i < KEYS_PER_THREAD;
        ++i, t += waveSize)
    {
        if (t < e_numKeys)
            LoadKey(keys.k[i], t);
        else
            LoadDummyKey(keys.k[i]);
    }
    return keys;
}

inline KeyStruct LoadKeysPartialWLT16(uint gtid, uint waveSize, uint partIndex, uint serialIterations)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, waveSize, partIndex, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += waveSize * serialIterations)
    {
        if (t < e_numKeys)
            LoadKey(keys.k[i], t);
        else
            LoadDummyKey(keys.k[i]);
    }
    return keys;
}

//=============================================================================
// WAVE-LEVEL MULTI-SPLIT (RANKING)
//=============================================================================
inline void WarpLevelMultiSplitWGE16(uint key, inout uint4 waveFlags)
{
    [unroll]
    for (uint k = 0; k < RADIX_LOG; ++k)
    {
        const uint currentBit = 1 << k + e_radixShift;
        const bool t = (key & currentBit) != 0;
        GroupMemoryBarrierWithGroupSync();
        const uint4 ballot = WaveActiveBallot(t);
        if (t)
            waveFlags &= ballot;
        else
            waveFlags &= (~ballot);
    }
}

inline uint2 CountBitsWGE16(uint waveSize, uint ltMask, uint4 waveFlags)
{
    uint2 count = uint2(0, 0);

    for (uint wavePart = 0; wavePart < waveSize; wavePart += 32)
    {
        uint t = countbits(waveFlags[wavePart >> 5]);
        if (WaveGetLaneIndex() >= wavePart)
        {
            if (WaveGetLaneIndex() >= wavePart + 32)
                count.x += t;
            else
                count.x += countbits(waveFlags[wavePart >> 5] & ltMask);
        }
        count.y += t;
    }

    return count;
}

inline void WarpLevelMultiSplitWLT16(uint key, inout uint waveFlags)
{
    [unroll]
    for (uint k = 0; k < RADIX_LOG; ++k)
    {
        const bool t = key >> (k + e_radixShift) & 1;
        waveFlags &= (t ? 0 : 0xffffffff) ^ (uint)WaveActiveBallot(t);
    }
}

inline OffsetStruct RankKeysWGE16(uint waveSize, uint waveOffset, KeyStruct keys)
{
    OffsetStruct offsets;
    const uint initialFlags = WaveFlagsWGE16(waveSize);
    const uint ltMask = (1U << (WaveGetLaneIndex() & 31)) - 1;

    [unroll]
    for (uint i = 0; i < KEYS_PER_THREAD; ++i)
    {
        uint4 waveFlags = initialFlags;
        WarpLevelMultiSplitWGE16(keys.k[i], waveFlags);

        const uint index = ExtractDigit(keys.k[i]) + waveOffset;
        const uint2 bitCount = CountBitsWGE16(waveSize, ltMask, waveFlags);

        offsets.o[i] = g_d[index] + bitCount.x;
        GroupMemoryBarrierWithGroupSync();
        if (bitCount.x == 0)
            g_d[index] += bitCount.y;
        GroupMemoryBarrierWithGroupSync();
    }

    return offsets;
}

inline OffsetStruct RankKeysWLT16(uint waveSize, uint waveIndex, KeyStruct keys, uint serialIterations)
{
    OffsetStruct offsets;
    const uint ltMask = (1U << WaveGetLaneIndex()) - 1;
    const uint initialFlags = WaveFlagsWLT16(waveSize);

    [unroll]
    for (uint i = 0; i < KEYS_PER_THREAD; ++i)
    {
        uint waveFlags = initialFlags;
        WarpLevelMultiSplitWLT16(keys.k[i], waveFlags);

        const uint index = ExtractPackedIndex(keys.k[i]) +
            (waveIndex / serialIterations * HALF_RADIX);

        const uint peerBits = countbits(waveFlags & ltMask);
        for (uint k = 0; k < serialIterations; ++k)
        {
            if (waveIndex % serialIterations == k)
                offsets.o[i] = ExtractPackedValue(g_d[index], keys.k[i]) + peerBits;

            GroupMemoryBarrierWithGroupSync();
            if (waveIndex % serialIterations == k && peerBits == 0)
            {
                InterlockedAdd(g_d[index],
                    countbits(waveFlags) << ExtractPackedShift(keys.k[i]));
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    return offsets;
}

//=============================================================================
// HISTOGRAM OPERATIONS
//=============================================================================
inline uint WaveHistInclusiveScanCircularShiftWGE16(uint gtid, uint waveSize)
{
    uint histReduction = g_d[gtid];
    for (uint i = gtid + RADIX; i < WaveHistsSizeWGE16(waveSize); i += RADIX)
    {
        histReduction += g_d[i];
        g_d[i] = histReduction - g_d[i];
    }
    return histReduction;
}

inline uint WaveHistInclusiveScanCircularShiftWLT16(uint gtid)
{
    uint histReduction = g_d[gtid];
    for (uint i = gtid + HALF_RADIX; i < WaveHistsSizeWLT16(); i += HALF_RADIX)
    {
        histReduction += g_d[i];
        g_d[i] = histReduction - g_d[i];
    }
    return histReduction;
}

inline void WaveHistReductionExclusiveScanWGE16(uint gtid, uint waveSize, uint histReduction)
{
    if (gtid < RADIX)
    {
        const uint laneMask = waveSize - 1;
        g_d[((WaveGetLaneIndex() + 1) & laneMask) + (gtid & ~laneMask)] = histReduction;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid < RADIX / waveSize)
    {
        g_d[gtid * waveSize] = WavePrefixSum(g_d[gtid * waveSize]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint t = WaveReadLaneAt(g_d[gtid], 0);
    if (gtid < RADIX && WaveGetLaneIndex())
        g_d[gtid] += t;
}

inline void WaveHistReductionExclusiveScanWLT16(uint gtid)
{
    uint shift = 1;
    for (uint j = RADIX >> 2; j > 0; j >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gtid < j)
        {
            g_d[((((gtid << 1) + 2) << shift) - 1) >> 1] +=
                g_d[((((gtid << 1) + 1) << shift) - 1) >> 1] & 0xffff0000;
        }
        shift++;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid == 0)
        g_d[HALF_RADIX - 1] &= 0xffff;

    for (uint j = 1; j < RADIX >> 1; j <<= 1)
    {
        --shift;
        GroupMemoryBarrierWithGroupSync();
        if (gtid < j)
        {
            const uint t = ((((gtid << 1) + 1) << shift) - 1) >> 1;
            const uint t2 = ((((gtid << 1) + 2) << shift) - 1) >> 1;
            const uint t3 = g_d[t];
            g_d[t] = (g_d[t] & 0xffff) | (g_d[t2] & 0xffff0000);
            g_d[t2] += t3 & 0xffff0000;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (gtid < HALF_RADIX)
    {
        const uint t = g_d[gtid];
        g_d[gtid] = (t >> 16) + (t << 16) + (t & 0xffff0000);
    }
}

inline void UpdateOffsetsWGE16(uint gtid, uint waveSize, inout OffsetStruct offsets, KeyStruct keys)
{
    if (gtid >= waveSize)
    {
        const uint t = getWaveIndex(gtid, waveSize) * RADIX;
        [unroll]
        for (uint i = 0; i < KEYS_PER_THREAD; ++i)
        {
            const uint t2 = ExtractDigit(keys.k[i]);
            offsets.o[i] += g_d[t2 + t] + g_d[t2];
        }
    }
    else
    {
        [unroll]
        for (uint i = 0; i < KEYS_PER_THREAD; ++i)
            offsets.o[i] += g_d[ExtractDigit(keys.k[i])];
    }
}

inline void UpdateOffsetsWLT16(uint gtid, uint waveSize, uint serialIterations, inout OffsetStruct offsets, KeyStruct keys)
{
    if (gtid >= waveSize * serialIterations)
    {
        const uint t = getWaveIndex(gtid, waveSize) / serialIterations * HALF_RADIX;
        [unroll]
        for (uint i = 0; i < KEYS_PER_THREAD; ++i)
        {
            const uint t2 = ExtractPackedIndex(keys.k[i]);
            offsets.o[i] += ExtractPackedValue(g_d[t2 + t] + g_d[t2], keys.k[i]);
        }
    }
    else
    {
        [unroll]
        for (uint i = 0; i < KEYS_PER_THREAD; ++i)
            offsets.o[i] += ExtractPackedValue(g_d[ExtractPackedIndex(keys.k[i])], keys.k[i]);
    }
}

//=============================================================================
// SCATTERING
//=============================================================================
inline void ScatterKeysShared(OffsetStruct offsets, KeyStruct keys)
{
    [unroll]
    for (uint i = 0; i < KEYS_PER_THREAD; ++i)
        g_d[offsets.o[i]] = keys.k[i];
}

inline uint DescendingIndex(uint deviceIndex)
{
    return e_numKeys - deviceIndex - 1;
}

inline void WriteKey(uint deviceIndex, uint groupSharedIndex)
{
    b_alt[deviceIndex] = g_d[groupSharedIndex];
}

inline void LoadPayload(inout uint payload, uint deviceIndex)
{
    payload = b_sortPayload[deviceIndex];
}

inline void ScatterPayloadsShared(OffsetStruct offsets, KeyStruct payloads)
{
    ScatterKeysShared(offsets, payloads);
}

inline void WritePayload(uint deviceIndex, uint groupSharedIndex)
{
    b_altPayload[deviceIndex] = g_d[groupSharedIndex];
}

inline void LoadThreadBlockReductions(uint gtid, uint gid, uint exclusiveHistReduction)
{
    if (gtid < RADIX)
    {
        g_d[gtid + PART_SIZE] = b_globalHist[gtid + GlobalHistOffset()] +
            b_passHist[gtid * e_threadBlocks + gid] - exclusiveHistReduction;
    }
}

//=============================================================================
// SCATTERING: FULL PARTITIONS
//=============================================================================
inline void ScatterPairsKeyPhaseAscending(uint gtid, inout DigitStruct digits)
{
    [unroll]
    for (uint i = 0, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        digits.d[i] = ExtractDigit(g_d[t]);
        WriteKey(g_d[digits.d[i] + PART_SIZE] + t, t);
    }
}

inline void ScatterPairsKeyPhaseDescending(uint gtid, inout DigitStruct digits)
{
    if (e_radixShift == 24)
    {
        [unroll]
        for (uint i = 0, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
        {
            digits.d[i] = ExtractDigit(g_d[t]);
            WriteKey(DescendingIndex(g_d[digits.d[i] + PART_SIZE] + t), t);
        }
    }
    else
    {
        ScatterPairsKeyPhaseAscending(gtid, digits);
    }
}

inline void LoadPayloadsWGE16(uint gtid, uint waveSize, uint partIndex, inout KeyStruct payloads)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, waveSize, partIndex);
        i < KEYS_PER_THREAD;
        ++i, t += waveSize)
    {
        LoadPayload(payloads.k[i], t);
    }
}

inline void LoadPayloadsWLT16(uint gtid, uint waveSize, uint partIndex, uint serialIterations, inout KeyStruct payloads)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, waveSize, partIndex, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += waveSize * serialIterations)
    {
        LoadPayload(payloads.k[i], t);
    }
}

inline void ScatterPayloadsAscending(uint gtid, DigitStruct digits)
{
    [unroll]
    for (uint i = 0, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
        WritePayload(g_d[digits.d[i] + PART_SIZE] + t, t);
}

inline void ScatterPayloadsDescending(uint gtid, DigitStruct digits)
{
    if (e_radixShift == 24)
    {
        [unroll]
        for (uint i = 0, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
            WritePayload(DescendingIndex(g_d[digits.d[i] + PART_SIZE] + t), t);
    }
    else
    {
        ScatterPayloadsAscending(gtid, digits);
    }
}

inline void ScatterPairsDevice(uint gtid, uint waveSize, uint partIndex, OffsetStruct offsets)
{
    DigitStruct digits;
#if defined(SHOULD_ASCEND)
    ScatterPairsKeyPhaseAscending(gtid, digits);
#else
    ScatterPairsKeyPhaseDescending(gtid, digits);
#endif
    GroupMemoryBarrierWithGroupSync();

    KeyStruct payloads;
    if (waveSize >= 16)
        LoadPayloadsWGE16(gtid, waveSize, partIndex, payloads);
    else
        LoadPayloadsWLT16(gtid, waveSize, partIndex, SerialIterations(waveSize), payloads);
    ScatterPayloadsShared(offsets, payloads);
    GroupMemoryBarrierWithGroupSync();

#if defined(SHOULD_ASCEND)
    ScatterPayloadsAscending(gtid, digits);
#else
    ScatterPayloadsDescending(gtid, digits);
#endif
}

inline void ScatterDevice(uint gtid, uint waveSize, uint partIndex, OffsetStruct offsets)
{
#if defined(SORT_PAIRS)
    ScatterPairsDevice(gtid, waveSize, partIndex, offsets);
#else
    // Keys only not used in gaussian splat sorting
#endif
}

//=============================================================================
// SCATTERING: PARTIAL PARTITIONS
//=============================================================================
inline void ScatterPairsKeyPhaseAscendingPartial(uint gtid, uint finalPartSize, inout DigitStruct digits)
{
    [unroll]
    for (uint i = 0, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        if (t < finalPartSize)
        {
            digits.d[i] = ExtractDigit(g_d[t]);
            WriteKey(g_d[digits.d[i] + PART_SIZE] + t, t);
        }
    }
}

inline void ScatterPairsKeyPhaseDescendingPartial(uint gtid, uint finalPartSize, inout DigitStruct digits)
{
    if (e_radixShift == 24)
    {
        [unroll]
        for (uint i = 0, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
        {
            if (t < finalPartSize)
            {
                digits.d[i] = ExtractDigit(g_d[t]);
                WriteKey(DescendingIndex(g_d[digits.d[i] + PART_SIZE] + t), t);
            }
        }
    }
    else
    {
        ScatterPairsKeyPhaseAscendingPartial(gtid, finalPartSize, digits);
    }
}

inline void LoadPayloadsPartialWGE16(uint gtid, uint waveSize, uint partIndex, inout KeyStruct payloads)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, waveSize, partIndex);
        i < KEYS_PER_THREAD;
        ++i, t += waveSize)
    {
        if (t < e_numKeys)
            LoadPayload(payloads.k[i], t);
    }
}

inline void LoadPayloadsPartialWLT16(uint gtid, uint waveSize, uint partIndex, uint serialIterations, inout KeyStruct payloads)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, waveSize, partIndex, serialIterations);
        i < KEYS_PER_THREAD;
        ++i, t += waveSize * serialIterations)
    {
        if (t < e_numKeys)
            LoadPayload(payloads.k[i], t);
    }
}

inline void ScatterPayloadsAscendingPartial(uint gtid, uint finalPartSize, DigitStruct digits)
{
    [unroll]
    for (uint i = 0, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        if (t < finalPartSize)
            WritePayload(g_d[digits.d[i] + PART_SIZE] + t, t);
    }
}

inline void ScatterPayloadsDescendingPartial(uint gtid, uint finalPartSize, DigitStruct digits)
{
    if (e_radixShift == 24)
    {
        [unroll]
        for (uint i = 0, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
        {
            if (t < finalPartSize)
                WritePayload(DescendingIndex(g_d[digits.d[i] + PART_SIZE] + t), t);
        }
    }
    else
    {
        ScatterPayloadsAscendingPartial(gtid, finalPartSize, digits);
    }
}

inline void ScatterPairsDevicePartial(uint gtid, uint waveSize, uint partIndex, OffsetStruct offsets)
{
    DigitStruct digits;
    const uint finalPartSize = e_numKeys - partIndex * PART_SIZE;
#if defined(SHOULD_ASCEND)
    ScatterPairsKeyPhaseAscendingPartial(gtid, finalPartSize, digits);
#else
    ScatterPairsKeyPhaseDescendingPartial(gtid, finalPartSize, digits);
#endif
    GroupMemoryBarrierWithGroupSync();

    KeyStruct payloads;
    if (waveSize >= 16)
        LoadPayloadsPartialWGE16(gtid, waveSize, partIndex, payloads);
    else
        LoadPayloadsPartialWLT16(gtid, waveSize, partIndex, SerialIterations(waveSize), payloads);
    ScatterPayloadsShared(offsets, payloads);
    GroupMemoryBarrierWithGroupSync();

#if defined(SHOULD_ASCEND)
    ScatterPayloadsAscendingPartial(gtid, finalPartSize, digits);
#else
    ScatterPayloadsDescendingPartial(gtid, finalPartSize, digits);
#endif
}

inline void ScatterDevicePartial(uint gtid, uint waveSize, uint partIndex, OffsetStruct offsets)
{
#if defined(SORT_PAIRS)
    ScatterPairsDevicePartial(gtid, waveSize, partIndex, offsets);
#else
    // Keys only not used
#endif
}

//=============================================================================
// UPSWEEP KERNEL FUNCTIONS
//=============================================================================
inline void HistogramDigitCounts(uint gtid, uint gid)
{
    const uint histOffset = gtid / 64 * RADIX;
    const uint partitionEnd = gid == e_threadBlocks - 1 ?
        e_numKeys : (gid + 1) * PART_SIZE;
    for (uint i = gtid + gid * PART_SIZE; i < partitionEnd; i += US_DIM)
    {
        InterlockedAdd(g_us[ExtractDigit(b_sort[i]) + histOffset], 1);
    }
}

inline void ReduceWriteDigitCounts(uint gtid, uint gid)
{
    for (uint i = gtid; i < RADIX; i += US_DIM)
    {
        g_us[i] += g_us[i + RADIX];
        b_passHist[i * e_threadBlocks + gid] = g_us[i];
        g_us[i] += WavePrefixSum(g_us[i]);
    }
}

inline void GlobalHistExclusiveScanWGE16(uint gtid, uint waveSize)
{
    GroupMemoryBarrierWithGroupSync();

    if (gtid < (RADIX / waveSize))
    {
        g_us[(gtid + 1) * waveSize - 1] +=
            WavePrefixSum(g_us[(gtid + 1) * waveSize - 1]);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint globalHistOffset = GlobalHistOffset();
    const uint laneMask = waveSize - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
    for (uint i = gtid; i < RADIX; i += US_DIM)
    {
        const uint index = circularLaneShift + (i & ~laneMask);
        uint t = WaveGetLaneIndex() != laneMask ? g_us[i] : 0;
        if (i >= waveSize)
            t += WaveReadLaneAt(g_us[i - 1], 0);
        InterlockedAdd(b_globalHist[index + globalHistOffset], t);
    }
}

inline void GlobalHistExclusiveScanWLT16(uint gtid, uint waveSize)
{
    const uint globalHistOffset = GlobalHistOffset();
    if (gtid < waveSize)
    {
        const uint circularLaneShift = WaveGetLaneIndex() + 1 & waveSize - 1;
        InterlockedAdd(b_globalHist[circularLaneShift + globalHistOffset],
            circularLaneShift ? g_us[gtid] : 0);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint laneLog = countbits(waveSize - 1);
    uint offset = laneLog;
    uint j = waveSize;
    for (; j < (RADIX >> 1); j <<= laneLog)
    {
        if (gtid < (RADIX >> offset))
        {
            g_us[((gtid + 1) << offset) - 1] +=
                WavePrefixSum(g_us[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint i = gtid + j; i < RADIX; i += US_DIM)
        {
            if ((i & ((j << laneLog) - 1)) >= j)
            {
                if (i < (j << laneLog))
                {
                    InterlockedAdd(b_globalHist[i + globalHistOffset],
                        WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0) +
                        ((i & (j - 1)) ? g_us[i - 1] : 0));
                }
                else
                {
                    if ((i + 1) & (j - 1))
                    {
                        g_us[i] +=
                            WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0);
                    }
                }
            }
        }
        offset += laneLog;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = gtid + j; i < RADIX; i += US_DIM)
    {
        InterlockedAdd(b_globalHist[i + globalHistOffset],
            WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0) +
            ((i & (j - 1)) ? g_us[i - 1] : 0));
    }
}

// Simple upsweep without wave intrinsics
// Builds per-partition histogram and atomically adds to global histogram
inline void Upsweep_Main_Simple(uint gtid, uint gid)
{
    // Clear shared histogram (US_DIM=128, RADIX=256, so each thread clears 2 entries)
    for (uint d = gtid; d < RADIX; d += US_DIM)
        g_us[d] = 0;
    GroupMemoryBarrierWithGroupSync();

    // Count digit occurrences in this partition
    uint partStart = gid * PART_SIZE;
    uint partEnd = gid == e_threadBlocks - 1 ? e_numKeys : (gid + 1) * PART_SIZE;

    for (uint i = partStart + gtid; i < partEnd; i += US_DIM)
    {
        uint digit = ExtractDigit(b_sort[i]);
        InterlockedAdd(g_us[digit], 1);
    }
    GroupMemoryBarrierWithGroupSync();

    // Write partition histogram and atomically add to global histogram
    // Each thread handles multiple digits (256/128 = 2 per thread)
    for (uint d = gtid; d < RADIX; d += US_DIM)
    {
        uint count = g_us[d];
        b_passHist[d * e_threadBlocks + gid] = count;
        InterlockedAdd(b_globalHist[d + GlobalHistOffset()], count);
    }
}

inline void Upsweep_Main(uint gtid, uint gid)
{
    // Use simple version to avoid wave intrinsic SPIRV issues
    Upsweep_Main_Simple(gtid, gid);
}

//=============================================================================
// SCAN KERNEL FUNCTIONS
//=============================================================================
inline void ExclusiveThreadBlockScanFullWGE16(
    uint gtid,
    uint laneMask,
    uint circularLaneShift,
    uint partEnd,
    uint deviceOffset,
    uint waveSize,
    inout uint reduction)
{
    for (uint i = gtid; i < partEnd; i += SCAN_DIM)
    {
        g_scan[gtid] = b_passHist[i + deviceOffset];
        g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
        GroupMemoryBarrierWithGroupSync();

        if (gtid < SCAN_DIM / waveSize)
        {
            g_scan[(gtid + 1) * waveSize - 1] +=
                WavePrefixSum(g_scan[(gtid + 1) * waveSize - 1]);
        }
        GroupMemoryBarrierWithGroupSync();

        uint t = (WaveGetLaneIndex() != laneMask ? g_scan[gtid] : 0) + reduction;
        if (gtid >= waveSize)
            t += WaveReadLaneAt(g_scan[gtid - 1], 0);
        b_passHist[circularLaneShift + (i & ~laneMask) + deviceOffset] = t;

        reduction += g_scan[SCAN_DIM - 1];
        GroupMemoryBarrierWithGroupSync();
    }
}

inline void ExclusiveThreadBlockScanPartialWGE16(
    uint gtid,
    uint laneMask,
    uint circularLaneShift,
    uint partEnd,
    uint deviceOffset,
    uint waveSize,
    uint reduction)
{
    uint i = gtid + partEnd;
    if (i < e_threadBlocks)
        g_scan[gtid] = b_passHist[deviceOffset + i];
    g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
    GroupMemoryBarrierWithGroupSync();

    if (gtid < SCAN_DIM / waveSize)
    {
        g_scan[(gtid + 1) * waveSize - 1] +=
            WavePrefixSum(g_scan[(gtid + 1) * waveSize - 1]);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint index = circularLaneShift + (i & ~laneMask);
    if (index < e_threadBlocks)
    {
        uint t = (WaveGetLaneIndex() != laneMask ? g_scan[gtid] : 0) + reduction;
        if (gtid >= waveSize)
            t += g_scan[(gtid & ~laneMask) - 1];
        b_passHist[index + deviceOffset] = t;
    }
}

inline void ExclusiveThreadBlockScanWGE16(uint gtid, uint gid, uint waveSize)
{
    uint reduction = 0;
    const uint laneMask = waveSize - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
    const uint partionsEnd = e_threadBlocks / SCAN_DIM * SCAN_DIM;
    const uint deviceOffset = gid * e_threadBlocks;

    ExclusiveThreadBlockScanFullWGE16(
        gtid, laneMask, circularLaneShift,
        partionsEnd, deviceOffset, waveSize, reduction);

    ExclusiveThreadBlockScanPartialWGE16(
        gtid, laneMask, circularLaneShift,
        partionsEnd, deviceOffset, waveSize, reduction);
}

inline void ExclusiveThreadBlockScanFullWLT16(
    uint gtid,
    uint partitions,
    uint deviceOffset,
    uint laneLog,
    uint circularLaneShift,
    uint waveSize,
    inout uint reduction)
{
    for (uint k = 0; k < partitions; ++k)
    {
        g_scan[gtid] = b_passHist[gtid + k * SCAN_DIM + deviceOffset];
        g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
        GroupMemoryBarrierWithGroupSync();
        if (gtid < waveSize)
        {
            b_passHist[circularLaneShift + k * SCAN_DIM + deviceOffset] =
                (circularLaneShift ? g_scan[gtid] : 0) + reduction;
        }

        uint offset = laneLog;
        uint j = waveSize;
        for (; j < (SCAN_DIM >> 1); j <<= laneLog)
        {
            if (gtid < (SCAN_DIM >> offset))
            {
                g_scan[((gtid + 1) << offset) - 1] +=
                    WavePrefixSum(g_scan[((gtid + 1) << offset) - 1]);
            }
            GroupMemoryBarrierWithGroupSync();

            if ((gtid & ((j << laneLog) - 1)) >= j)
            {
                if (gtid < (j << laneLog))
                {
                    b_passHist[gtid + k * SCAN_DIM + deviceOffset] =
                        WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0) +
                        ((gtid & (j - 1)) ? g_scan[gtid - 1] : 0) + reduction;
                }
                else
                {
                    if ((gtid + 1) & (j - 1))
                    {
                        g_scan[gtid] +=
                            WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0);
                    }
                }
            }
            offset += laneLog;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint i = gtid + j; i < SCAN_DIM; i += SCAN_DIM)
        {
            b_passHist[i + k * SCAN_DIM + deviceOffset] =
                WaveReadLaneAt(g_scan[((i >> offset) << offset) - 1], 0) +
                ((i & (j - 1)) ? g_scan[i - 1] : 0) + reduction;
        }

        reduction += WaveReadLaneAt(g_scan[SCAN_DIM - 1], 0) +
            WaveReadLaneAt(g_scan[(((SCAN_DIM - 1) >> offset) << offset) - 1], 0);
        GroupMemoryBarrierWithGroupSync();
    }
}

inline void ExclusiveThreadBlockScanPartialWLT16(
    uint gtid,
    uint partitions,
    uint deviceOffset,
    uint laneLog,
    uint circularLaneShift,
    uint waveSize,
    uint reduction)
{
    const uint finalPartSize = e_threadBlocks - partitions * SCAN_DIM;
    if (gtid < finalPartSize)
    {
        g_scan[gtid] = b_passHist[gtid + partitions * SCAN_DIM + deviceOffset];
        g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
    }
    GroupMemoryBarrierWithGroupSync();
    if (gtid < waveSize && circularLaneShift < finalPartSize)
    {
        b_passHist[circularLaneShift + partitions * SCAN_DIM + deviceOffset] =
            (circularLaneShift ? g_scan[gtid] : 0) + reduction;
    }

    uint offset = laneLog;
    for (uint j = waveSize; j < finalPartSize; j <<= laneLog)
    {
        if (gtid < (finalPartSize >> offset))
        {
            g_scan[((gtid + 1) << offset) - 1] +=
                WavePrefixSum(g_scan[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();

        if ((gtid & ((j << laneLog) - 1)) >= j && gtid < finalPartSize)
        {
            if (gtid < (j << laneLog))
            {
                b_passHist[gtid + partitions * SCAN_DIM + deviceOffset] =
                    WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0) +
                    ((gtid & (j - 1)) ? g_scan[gtid - 1] : 0) + reduction;
            }
            else
            {
                if ((gtid + 1) & (j - 1))
                {
                    g_scan[gtid] +=
                        WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0);
                }
            }
        }
        offset += laneLog;
    }
}

inline void ExclusiveThreadBlockScanWLT16(uint gtid, uint gid, uint waveSize)
{
    uint reduction = 0;
    const uint partitions = e_threadBlocks / SCAN_DIM;
    const uint deviceOffset = gid * e_threadBlocks;
    const uint laneLog = countbits(waveSize - 1);
    const uint circularLaneShift = WaveGetLaneIndex() + 1 & waveSize - 1;

    ExclusiveThreadBlockScanFullWLT16(
        gtid, partitions, deviceOffset, laneLog,
        circularLaneShift, waveSize, reduction);

    ExclusiveThreadBlockScanPartialWLT16(
        gtid, partitions, deviceOffset, laneLog,
        circularLaneShift, waveSize, reduction);
}

// Simple scan without wave intrinsics
// Computes exclusive prefix sum of partition histograms for one digit
// gid = digit index (0..255), gtid = thread index in workgroup
inline void Scan_Main_Simple(uint gtid, uint gid)
{
    // Each workgroup handles one digit
    // b_passHist layout: [digit * e_threadBlocks + partition] = count
    uint deviceOffset = gid * e_threadBlocks;

    // Process in chunks of SCAN_DIM
    uint runningSum = 0;
    for (uint chunk = 0; chunk < e_threadBlocks; chunk += SCAN_DIM)
    {
        uint idx = chunk + gtid;

        // Load value (0 if out of bounds)
        uint val = 0;
        if (idx < e_threadBlocks)
            val = b_passHist[deviceOffset + idx];

        // Store in shared memory
        g_scan[gtid] = val;
        GroupMemoryBarrierWithGroupSync();

        // Hillis-Steele inclusive scan in shared memory
        for (uint offset = 1; offset < SCAN_DIM; offset <<= 1)
        {
            uint temp = 0;
            if (gtid >= offset)
                temp = g_scan[gtid - offset];
            GroupMemoryBarrierWithGroupSync();
            g_scan[gtid] += temp;
            GroupMemoryBarrierWithGroupSync();
        }

        // Convert to exclusive scan and add running sum
        uint exclusiveVal = (gtid == 0) ? 0 : g_scan[gtid - 1];
        exclusiveVal += runningSum;

        // Write back
        if (idx < e_threadBlocks)
            b_passHist[deviceOffset + idx] = exclusiveVal;

        // Update running sum for next chunk (inclusive scan result of last element)
        runningSum += g_scan[SCAN_DIM - 1];
        GroupMemoryBarrierWithGroupSync();
    }
}

inline void Scan_Main(uint gtid, uint gid)
{
    // Use simple version to avoid wave intrinsic SPIRV issues
    Scan_Main_Simple(gtid, gid);
}

//=============================================================================
// DOWNSWEEP KERNEL FUNCTIONS
//=============================================================================

// Shared memory layout for stable downsweep:
// g_d[0..255]     = running digit counters (advanced by batch totals)
// g_d[256..511]   = batch base offsets for each digit
// g_d[512..767]   = batch keys (D_DIM elements)
// g_d[768..1023]  = batch payloads (D_DIM elements)
// g_d[1024..1279] = batch digit counts (RADIX elements)
#define BATCH_KEYS_OFFSET    512
#define BATCH_PAYLOAD_OFFSET 768
#define BATCH_COUNTS_OFFSET  1024

// Stable downsweep using shared memory for local ranking
// Guarantees stability by computing exact local ranks within each batch
inline void Downsweep_Main_Simple(uint gtid, uint gid)
{
    // Initialize running counters from global/partition histograms
    if (gtid < RADIX)
    {
        uint globalOffset = b_globalHist[gtid + GlobalHistOffset()];
        uint partitionOffset = b_passHist[gtid * e_threadBlocks + gid];
        g_d[gtid] = globalOffset + partitionOffset;  // Running offset for this digit
    }
    GroupMemoryBarrierWithGroupSync();

    // Compute partition bounds
    uint partStart = gid * PART_SIZE;
    uint partEnd = min(partStart + PART_SIZE, e_numKeys);

    // Process partition in batches of D_DIM elements
    for (uint batch = 0; batch < PART_SIZE && (partStart + batch) < partEnd; batch += D_DIM)
    {
        uint globalIdx = partStart + batch + gtid;
        bool valid = globalIdx < partEnd;

        // Load keys and payloads into shared memory
        uint key = 0;
        uint payload = 0;
        if (valid)
        {
            key = b_sort[globalIdx];
            payload = b_sortPayload[globalIdx];
        }
        g_d[BATCH_KEYS_OFFSET + gtid] = key;
        g_d[BATCH_PAYLOAD_OFFSET + gtid] = payload;

        // Clear batch digit counts
        if (gtid < RADIX)
            g_d[BATCH_COUNTS_OFFSET + gtid] = 0;
        GroupMemoryBarrierWithGroupSync();

        // Count occurrences of each digit in this batch (atomically)
        uint myDigit = ExtractDigit(key);
        if (valid)
            InterlockedAdd(g_d[BATCH_COUNTS_OFFSET + myDigit], 1);
        GroupMemoryBarrierWithGroupSync();

        // Save batch base offsets and advance running counters
        if (gtid < RADIX)
        {
            uint batchCount = g_d[BATCH_COUNTS_OFFSET + gtid];
            g_d[gtid + RADIX] = g_d[gtid];  // Batch base = current running offset
            g_d[gtid] += batchCount;         // Advance running offset
        }
        GroupMemoryBarrierWithGroupSync();

        // Each thread computes its stable local rank by counting
        // same-digit elements that appear before it in the batch
        if (valid)
        {
            uint batchSize = min(D_DIM, partEnd - (partStart + batch));
            uint localRank = 0;

            // Count elements with same digit that come before this thread
            for (uint i = 0; i < gtid; i++)
            {
                if (i < batchSize)
                {
                    uint otherDigit = ExtractDigit(g_d[BATCH_KEYS_OFFSET + i]);
                    if (otherDigit == myDigit)
                        localRank++;
                }
            }

            // Compute final destination
            uint destIdx = g_d[myDigit + RADIX] + localRank;

#if defined(SHOULD_ASCEND)
            b_alt[destIdx] = key;
            b_altPayload[destIdx] = payload;
#else
            if (e_radixShift == 24)
                destIdx = e_numKeys - destIdx - 1;
            b_alt[destIdx] = key;
            b_altPayload[destIdx] = payload;
#endif
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

inline void Downsweep_Main(uint gtid, uint gid)
{
    // Use the simple atomic-based version to avoid wave intrinsic SPIRV issues
    Downsweep_Main_Simple(gtid, gid);
}

#endif // GPU_SORT_COMMON_HLSLI
