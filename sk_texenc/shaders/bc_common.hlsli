// Shared helpers for the BC-family GPU encoders (bc1/bc6h/bc7_compress.hlsl).
// The ASTC analog is astc_common.hlsli.

#include "texenc_common.hlsli"

///////////////////////////////////////////////////////////////////////////////
// 128-bit block packing (BC6H / BC7)
///////////////////////////////////////////////////////////////////////////////

// 4-bit index weights in 1/64ths — rounded k*64/15 so indices 0 and 15
// reproduce the endpoints exactly. Shared by BC6H mode 11 and BC7 mode 6;
// interpolation is (a*(64-w) + b*w + 32) >> 6.
static const float W4[16] = {
	 0.0 / 64.0,  4.0 / 64.0,  9.0 / 64.0, 13.0 / 64.0,
	17.0 / 64.0, 21.0 / 64.0, 26.0 / 64.0, 30.0 / 64.0,
	34.0 / 64.0, 38.0 / 64.0, 43.0 / 64.0, 47.0 / 64.0,
	51.0 / 64.0, 55.0 / 64.0, 60.0 / 64.0, 64.0 / 64.0,
};

// OR count bits of value into a 128-bit block at a bit offset (LSB-first,
// matching the D3D bitstream). count <= 24 so a value spans at most 2 dwords.
void write_bits(inout uint4 b, uint offset, uint count, uint value) {
	value &= (1u << count) - 1u;
	uint word = offset >> 5u;
	uint bit  = offset & 31u;
	b[word] |= value << bit;
	if (bit + count > 32u)
		b[word + 1u] |= value >> (32u - bit);
}
