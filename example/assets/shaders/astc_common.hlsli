// ASTC LDR compression helpers — fast path
//
// Design: single partition, CEM 8 (LDR RGB direct), several fixed weight-grid
// configurations. All fields use pure binary packing (no trit/quint BISE) so
// the block can be assembled with simple shifts and a bit reversal.
//
// Common block layout (128 bits total):
//   bits   0-10 : block mode       (11 bits)
//   bits  11-12 : partition cnt -1 ( 2 bits, = 0)
//   bits  13-16 : CEM              ( 4 bits, = 8)
//   bits  17+   : endpoints        (N × bits_per_value, growing upward)
//   bits  ...   : unused           (gap)
//   bits ...-127: weights          (M × bits_per_weight, bit-reversed from
//                                   the top so block[127 - k] = natural[k])
//
// The three supported configurations:
//
//   Mode 0x053 — 4x4 weight grid, 3-bit weights (8 levels), 8-bit endpoints.
//     Used by the ASTC 4x4 block (per-pixel) and the ASTC 6x6 block with a
//     downsampled weight grid.
//     Bit budget: 17 + 48 + 48 = 113 bits (15 unused).
//
//   Mode 0x108 — 6x6 weight grid, 2-bit weights (4 levels), 6-bit endpoints.
//     Used by the ASTC 6x6 block with per-pixel weights. Trades endpoint
//     precision for per-pixel spatial detail — better for sharp edges.
//     Bit budget: 17 + 36 + 72 = 125 bits (3 unused).
//
// Block mode derivation (from the ASTC spec / mesa's decoder):
//
//   Standard sub-modes (bits[1:0] ≠ 00):
//     bits[1:0]   = wt_range low 2 bits
//     bits[3:2]   = sub-mode (00 → wt_w=b+4, wt_h=a+2)
//     bit  4      = wt_range middle bit
//     bits[6:5]   = A
//     bits[8:7]   = B
//     bit  9      = H (wt_range high bit / high precision)
//     bit 10      = D (dual plane)
//   For wt_range=7 (8 levels, 3 bits), sub-mode=00, A=2, B=0, H=0, D=0:
//     bits → 0 0 0 0 1 0 1 0 0 1 1  =  0x053
//
//   Alternate sub-modes (bits[1:0] = 00): needed for 6x6 weight grids.
//     Pattern "BB10AARRR00" (bit 10…0):
//       bits[1:0] = 00 (literal)
//       bits[4:2] = R (see below)
//       bits[6:5] = A, where wt_w = A + 6
//       bits[8:7] = 10 (literal, identifies this sub-mode)
//       bits[10:9]= B, where wt_h = B + 6
//       dual_plane and high_prec are forced to 0 in this mode.
//       wt_range encoding uses bits 1..4 per:
//         wt_range = get_bits(1,3) | get_bits(4,1)
//         (bit 1 is always 0 here, so wt_range[0]=bit4, wt_range[1]=bit2, wt_range[2]=bit3)
//   For wt_range=4 (4 levels, 2 bits), A=0 (wt_w=6), B=0 (wt_h=6):
//     bits → 0 0 1 0 0 0 0 1 0 0 0  =  0x108

static const uint ASTC_BLOCK_MODE_3x3_R3      = 0x1BFu; // 3x3 grid, 3-bit weights
static const uint ASTC_BLOCK_MODE_3x3_R5      = 0x3BFu; // 3x3 grid, 5-bit weights (32 lvl, H=1 + same R as R3)
static const uint ASTC_BLOCK_MODE_4x4_R2      = 0x042u; // 4x4 grid, 2-bit weights
static const uint ASTC_BLOCK_MODE_4x4_R3      = 0x053u; // 4x4 grid, 3-bit weights
static const uint ASTC_BLOCK_MODE_4x4_R6_TRIT = 0x043u; // 4x4 grid, trit+1bit (6 lvl)
static const uint ASTC_BLOCK_MODE_4x4_R12_TRIT= 0x251u; // 4x4 grid, trit+2bit (12 lvl, H=1)
static const uint ASTC_BLOCK_MODE_5x4_R3      = 0x0D3u; // 5x4 grid, 3-bit weights
static const uint ASTC_BLOCK_MODE_5x5_R2      = 0x0E2u; // 5x5 grid, 2-bit weights
// 6x5 grid, 2-bit weights. Same sub-mode 00 as 5x5 (wt_h = a+2 = 5 → a=3),
// just bumps b from 1 to 2 to get wt_w = b+4 = 6 — i.e. flip bit 7 (off)
// and set bit 8 (on) relative to 5x5_R2. Used by the 8x8 HDR encoder for
// asymmetric horizontal detail.
static const uint ASTC_BLOCK_MODE_6x5_R2      = 0x162u; // 6x5 grid, 2-bit weights
static const uint ASTC_BLOCK_MODE_6x6_R2      = 0x108u; // 6x6 grid, 2-bit weights
// Dual-plane variant of 3x3+R2 (D=1 in bit 10). Decoder reads 2 weight values
// per grid point (primary for the main color channels, secondary for the
// channel selected by the 2-bit CCS field).
static const uint ASTC_BLOCK_MODE_3x3_R2_DUAL = 0x5AEu; // 3x3 grid, 2-bit weights, dual-plane
static const uint ASTC_CEM_LDR_RGB            = 8u;
static const uint ASTC_CEM_LDR_RGBA           = 12u;
// HDR RGB direct, 6 endpoint values. Decoder reconstructs the 6 bytes as
// (a, c, b0, b1, d0, d1) — a custom log-FP encoding, NOT direct RGB. See
// astc_quantize_hdr_rgb / astc_pack_hdr_rgb_endpoints below.
static const uint ASTC_CEM_HDR_RGB            = 11u;
// Color component selector: which channel uses the secondary weight plane.
// 0=R, 1=G, 2=B, 3=A. For RGBA-with-alpha-cutout we want alpha as secondary.
static const uint ASTC_CCS_ALPHA              = 3u;

// Sparse bit-write helper. value must already be masked to <count> bits.
// Writes <count> bits of <value> at bit position <pos> inside the 128-bit
// block. The block is stored little-endian across the uint4 lanes: lane.x
// holds bits 0-31, lane.y 32-63, lane.z 64-95, lane.w 96-127.
void astc_block_write_bits(inout uint4 block, uint pos, uint count, uint value) {
	uint word  = pos >> 5;
	uint shift = pos & 31u;
	uint mask  = (count >= 32u) ? 0xFFFFFFFFu : ((1u << count) - 1u);
	value &= mask;

	block[word] |= value << shift;
	// Straddle into the next word when the field crosses a 32-bit boundary
	if (shift + count > 32u) {
		block[word + 1u] |= value >> (32u - shift);
	}
}

// Reverse the low 3 bits: abc → cba. Used for 3-bit weight placement.
uint astc_reverse3(uint v) {
	return ((v & 1u) << 2) | (v & 2u) | ((v & 4u) >> 2);
}

// Reverse the low 2 bits: ab → ba. Used for 2-bit weight placement.
uint astc_reverse2(uint v) {
	return ((v & 1u) << 1) | ((v >> 1) & 1u);
}

// Reverse the low 5 bits: abcde → edcba. Used for 5-bit weight placement
// (32 weight levels, range index 13 with H=1 + R=7).
uint astc_reverse5(uint v) {
	return ((v & 1u) << 4) | ((v & 2u) << 2) | (v & 4u) | ((v & 8u) >> 2) | ((v & 16u) >> 4);
}

// Write a 3-bit weight into its bit-reversed slot at the top of the block.
// For weight index i (in a 16-weight grid), natural bits 3i..3i+2 land at
// block bits (125-3i)..(127-3i) after the reversal. Placing astc_reverse3(w)
// at bit (125-3i) in normal low-to-high order accomplishes both the reorder
// and the bit reversal.
void astc_write_weight_3bit(inout uint4 block, uint weight_index, uint weight_value) {
	uint pos = 125u - 3u * weight_index;
	astc_block_write_bits(block, pos, 3u, astc_reverse3(weight_value));
}

// Write a 2-bit weight into its bit-reversed slot at the top of the block.
// For weight index i (in a 36-weight grid), natural bits 2i..2i+1 land at
// block bits (126-2i)..(127-2i) after the reversal.
void astc_write_weight_2bit(inout uint4 block, uint weight_index, uint weight_value) {
	uint pos = 126u - 2u * weight_index;
	astc_block_write_bits(block, pos, 2u, astc_reverse2(weight_value));
}

// Write a 5-bit weight (32 levels, range index 13) into its bit-reversed
// slot at the top of the block. For weight index i, natural bits 5i..5i+4
// land at block bits (123-5i)..(127-5i). Used by 3x3 + 5-bit weight modes
// (e.g. HDR 8x8 Mode B smooth-gradient).
void astc_write_weight_5bit(inout uint4 block, uint weight_index, uint weight_value) {
	uint pos = 123u - 5u * weight_index;
	astc_block_write_bits(block, pos, 5u, astc_reverse5(weight_value));
}

// Write an interleaved primary/secondary 2-bit weight pair for dual-plane
// mode. The natural weight stream stores them as (w0_pri, w0_sec, w1_pri,
// w1_sec, ...) — 4 bits per grid point. After bit-reversal at the top of
// the block:
//   primary  weight i bits land at (127-4i) and (126-4i)   (LSB at 127-4i)
//   secondary weight i bits land at (125-4i) and (124-4i)  (LSB at 125-4i)
// Writing astc_reverse2(value) at position (126-4i) places primary correctly,
// likewise position (124-4i) places secondary correctly.
void astc_write_dual_weight_2bit(inout uint4 block, uint weight_index, uint w_primary, uint w_secondary) {
	uint pos_pri = 126u - 4u * weight_index;
	uint pos_sec = 124u - 4u * weight_index;
	astc_block_write_bits(block, pos_pri, 2u, astc_reverse2(w_primary));
	astc_block_write_bits(block, pos_sec, 2u, astc_reverse2(w_secondary));
}

// Write the 2-bit Color Component Selector (CCS) for dual-plane mode. The
// CCS lives at bit position (128 - weight_bits - 2), i.e. just below the
// bit-reversed weight area. NOT bit-reversed itself.
//   For 3x3 grid + 2-bit weights + dual-plane: weight_bits = 36, CCS at 90.
void astc_write_ccs(inout uint4 block, uint ccs_pos, uint ccs_value) {
	astc_block_write_bits(block, ccs_pos, 2u, ccs_value);
}

// Pack endpoints for single-partition CEM 8 (LDR RGB direct) at 8 bits per
// value. ASTC's BISE value order for CEM 8 is INTERLEAVED: v0..v5 maps to
// R0, R1, G0, G1, B0, B1 — not sequential per-endpoint. The decoder uses
// sum(v0,v2,v4) vs sum(v1,v3,v5) to decide blue-contract swap; caller should
// ensure e0 has the smaller channel sum than e1 to avoid that swap (which
// would invert R/B).
void astc_write_endpoints_rgb8(inout uint4 block, uint3 e0, uint3 e1) {
	astc_block_write_bits(block, 17u, 8u, e0.r);
	astc_block_write_bits(block, 25u, 8u, e1.r);
	astc_block_write_bits(block, 33u, 8u, e0.g);
	astc_block_write_bits(block, 41u, 8u, e1.g);
	astc_block_write_bits(block, 49u, 8u, e0.b);
	astc_block_write_bits(block, 57u, 8u, e1.b);
}

// CEM 12 (LDR RGBA direct) at 8 bits per value. Same interleaved order as
// CEM 8, with two additional values for alpha at the end:
// (v0..v7) = (R0, R1, G0, G1, B0, B1, A0, A1). Decoder still uses sum(v0,v2,v4)
// vs sum(v1,v3,v5) for blue-contract — alpha doesn't enter the swap decision.
void astc_write_endpoints_rgba8(inout uint4 block, uint3 e0_rgb, uint3 e1_rgb, uint a0, uint a1) {
	astc_block_write_bits(block, 17u, 8u, e0_rgb.r);
	astc_block_write_bits(block, 25u, 8u, e1_rgb.r);
	astc_block_write_bits(block, 33u, 8u, e0_rgb.g);
	astc_block_write_bits(block, 41u, 8u, e1_rgb.g);
	astc_block_write_bits(block, 49u, 8u, e0_rgb.b);
	astc_block_write_bits(block, 57u, 8u, e1_rgb.b);
	astc_block_write_bits(block, 65u, 8u, a0);
	astc_block_write_bits(block, 73u, 8u, a1);
}

// BISE quint packer — inverse of mesa's unpack_quint_block (texcompress_astc.cpp).
// Given 3 quint digits q0, q1, q2 ∈ [0, 4], produces a 7-bit pattern that the
// decoder unpacks back to (q0, q1, q2). The decoder has three cases keyed on
// the Q-bit pattern; this encoder mirrors them.
uint astc_pack_3quints(uint q0, uint q1, uint q2) {
	uint Q0, Q1, Q2, Q3, Q4, Q5, Q6;
	if (q0 == 4u && q1 == 4u) {
		// Case A: q0 = q1 = 4, q2 free.
		Q1 = 1u; Q2 = 1u; Q5 = 0u; Q6 = 0u;
		if (q2 == 4u) { Q0 = 1u; Q3 = 0u; Q4 = 0u; }
		else          { Q0 = 0u; Q3 = q2 & 1u; Q4 = (q2 >> 1) & 1u; }
	} else if (q2 == 4u) {
		// Case B: q2 = 4, (q0, q1) not both 4.
		Q1 = 1u; Q2 = 1u;
		uint C = (q1 == 4u) ? ((q0 << 3) | 5u) : ((q1 << 3) | q0);
		Q0 =  C        & 1u;
		Q5 = 1u - ((C >> 1) & 1u);
		Q6 = 1u - ((C >> 2) & 1u);
		Q3 = (C >> 3) & 1u;
		Q4 = (C >> 4) & 1u;
	} else {
		// Case C: q2 < 4.
		Q5 = q2 & 1u;
		Q6 = (q2 >> 1) & 1u;
		uint C = (q1 == 4u) ? ((q0 << 3) | 5u) : ((q1 << 3) | q0);
		Q0 =  C        & 1u;
		Q1 = (C >> 1) & 1u;
		Q2 = (C >> 2) & 1u;
		Q3 = (C >> 3) & 1u;
		Q4 = (C >> 4) & 1u;
	}
	// 7 Q-bits at their canonical positions within the BISE group: {Q0..Q6}
	// live at bit offsets {4, 5, 6, 11, 12, 17, 18} assuming n=4 binary bits
	// per value (range 80). Caller combines with the binary m0..m2 bits.
	return (Q0 << 4) | (Q1 << 5) | (Q2 << 6)
	     | (Q3 << 11) | (Q4 << 12)
	     | (Q5 << 17) | (Q6 << 18);
}

// Dequantize a range-80 (quint+4bit) encoded value back to 8-bit. Mirrors
// mesa's unquantise_colour_endpoints case (ce_quints=1, ce_bits=4). Needed
// for the multi-mode encoder's SSE computation, where we predict exactly
// what the decoder will reconstruct.
uint astc_dequant_r80(uint v) {
	uint t    = (v >> 1) & 7u;
	uint B    = (t >> 1) | (t << 6);
	uint D    = v >> 4;
	uint T    = D * 13u + B;
	uint Axor = (v & 1u) * 0x1FFu;
	T         = T ^ Axor;
	return ((v & 1u) * 0x80u) | (T >> 2);
}

// Inverse quant LUT for range 80 (quint + 4 bit). 256 entries: target 8-bit
// channel value → closest 7-bit encoded value that dequantizes to it.
// Precomputed offline (Python) by running mesa's dequant formula for all 80
// valid encoded values and picking the best v for each target. Replaces what
// would otherwise be an 80-iteration search per endpoint value per block.
static const uint astc_r80_quant_lut[256] = {
	 0u,  0u, 16u, 16u, 16u, 32u, 32u, 32u, 48u, 48u, 48u, 48u, 64u, 64u, 64u,  2u,
	 2u,  2u, 18u, 18u, 18u, 34u, 34u, 34u, 50u, 50u, 50u, 50u, 66u, 66u, 66u,  4u,
	 4u,  4u, 20u, 20u, 20u, 36u, 36u, 36u, 36u, 52u, 52u, 52u, 68u, 68u, 68u,  6u,
	 6u,  6u, 22u, 22u, 22u, 38u, 38u, 38u, 38u, 54u, 54u, 54u, 70u, 70u, 70u,  8u,
	 8u,  8u, 24u, 24u, 24u, 24u, 40u, 40u, 40u, 56u, 56u, 56u, 72u, 72u, 72u, 10u,
	10u, 10u, 26u, 26u, 26u, 26u, 42u, 42u, 42u, 58u, 58u, 58u, 74u, 74u, 74u, 12u,
	12u, 12u, 12u, 28u, 28u, 28u, 44u, 44u, 44u, 60u, 60u, 60u, 76u, 76u, 76u, 14u,
	14u, 14u, 14u, 30u, 30u, 30u, 46u, 46u, 46u, 62u, 62u, 62u, 78u, 78u, 78u, 78u,
	79u, 79u, 79u, 79u, 63u, 63u, 63u, 47u, 47u, 47u, 31u, 31u, 31u, 15u, 15u, 15u,
	15u, 77u, 77u, 77u, 61u, 61u, 61u, 45u, 45u, 45u, 29u, 29u, 29u, 13u, 13u, 13u,
	13u, 75u, 75u, 75u, 59u, 59u, 59u, 43u, 43u, 43u, 27u, 27u, 27u, 27u, 11u, 11u,
	11u, 73u, 73u, 73u, 57u, 57u, 57u, 41u, 41u, 41u, 25u, 25u, 25u, 25u,  9u,  9u,
	 9u, 71u, 71u, 71u, 55u, 55u, 55u, 39u, 39u, 39u, 39u, 23u, 23u, 23u,  7u,  7u,
	 7u, 69u, 69u, 69u, 53u, 53u, 53u, 37u, 37u, 37u, 37u, 21u, 21u, 21u,  5u,  5u,
	 5u, 67u, 67u, 67u, 51u, 51u, 51u, 51u, 35u, 35u, 35u, 19u, 19u, 19u,  3u,  3u,
	 3u, 65u, 65u, 65u, 49u, 49u, 49u, 49u, 33u, 33u, 33u, 17u, 17u, 17u,  1u,  1u
};

uint astc_quantize_to_r80(uint target) {
	return astc_r80_quant_lut[min(target, 255u)];
}

// Pack endpoints for single-partition CEM 8 at range 80 (1 quint + 4 bits
// per value). This is the configuration the decoder selects when there are
// 39 endpoint bits available — e.g. the 6x6 per-pixel block with 2-bit
// weights (17 header + 72 weights = 89 used, 39 left).
//
// Each 3-value BISE group takes 3·4 = 12 binary bits + 7 quint bits = 19 bits.
// Six values = two groups = 38 bits. Each 8-bit endpoint is quantized via
// astc_quantize_to_r80 to the closest representable value.
void astc_write_endpoints_rgb6(inout uint4 block, uint3 e0, uint3 e1) {
	// Closest-match quantization — the dequant formula is non-linear so we
	// can't do this with a simple shift or multiply.
	uint3 q0 = uint3(
		astc_quantize_to_r80(e0.r),
		astc_quantize_to_r80(e0.g),
		astc_quantize_to_r80(e0.b));
	uint3 q1 = uint3(
		astc_quantize_to_r80(e1.r),
		astc_quantize_to_r80(e1.g),
		astc_quantize_to_r80(e1.b));

	// CEM 8 value order: (v0..v5) = (R0, R1, G0, G1, B0, B1).
	uint v[6] = { q0.r, q1.r, q0.g, q1.g, q0.b, q1.b };

	// Group 0 (values 0,1,2) at block bit 17; Group 1 (values 3,4,5) at 36.
	// Within each 19-bit group the bits are laid out as:
	//   [0-3]   m0 (value 0 binary)
	//   [4-6]   Q0 Q1 Q2     (quint bits, from astc_pack_3quints)
	//   [7-10]  m1 (value 1 binary)
	//   [11-12] Q3 Q4
	//   [13-16] m2 (value 2 binary)
	//   [17-18] Q5 Q6
	// astc_pack_3quints returns the Q bits pre-shifted to those offsets, so
	// we OR the binary m0/m1/m2 at their offsets and write the whole 19-bit
	// value in a single pass.
	[unroll] for (uint g = 0; g < 2; g++) {
		uint base_bit = 17u + g * 19u;
		uint m0 = v[g*3 + 0] & 0xFu;
		uint m1 = v[g*3 + 1] & 0xFu;
		uint m2 = v[g*3 + 2] & 0xFu;
		uint q_a = v[g*3 + 0] >> 4;
		uint q_b = v[g*3 + 1] >> 4;
		uint q_c = v[g*3 + 2] >> 4;
		uint combined = m0 | (m1 << 7) | (m2 << 13) | astc_pack_3quints(q_a, q_b, q_c);
		astc_block_write_bits(block, base_bit, 19u, combined);
	}
}

// Headers for each supported configuration. Each writes block mode + single
// partition + CEM at the fixed positions.
void astc_write_header_4x4_rgb(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_4x4_R3);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_LDR_RGB);
}

void astc_write_header_6x6_rgb(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_6x6_R2);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_LDR_RGB);
}

void astc_write_header_3x3_rgba(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_3x3_R3);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_LDR_RGBA);
}

void astc_write_header_5x5_rgba(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_5x5_R2);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_LDR_RGBA);
}

// 4x4 grid + 2-bit weights for CEM 12 (RGBA) at 4x4 ASTC blocks. Per-pixel
// weights since the 4x4 grid matches the 4x4 block footprint exactly. Bit
// budget: 17 header + 32 weights + 64 endpoints (8 values × 8 bits, range
// 256 pure binary) = 113 bits used, 15 wasted.
void astc_write_header_4x4_rgba(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_4x4_R2);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_LDR_RGBA);
}

// 4x4 grid + trit+2bit weights (12 lvl) for CEM 8. Upgrades the R3 3-bit
// weight mode by 50% axial precision, keeping the same per-pixel grid and
// range-256 endpoints. Bit budget: 17 + 58 (weights) + 48 = 123 / 128.
void astc_write_header_4x4_rgb_r12(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_4x4_R12_TRIT);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_LDR_RGB);
}

// 4x4 grid + trit+1bit weights (6 lvl) for CEM 12. Upgrades the R2 2-bit
// weight mode by 50% axial precision, keeping per-pixel grid + range-256
// endpoints. Bit budget: 17 + 42 + 64 = 123 / 128.
void astc_write_header_4x4_rgba_r6(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_4x4_R6_TRIT);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_LDR_RGBA);
}

// Dual-plane RGBA header: 3x3 weight grid, 2-bit weights, CEM 12 (RGBA),
// dual-plane enabled (D=1). Caller must also write the 2-bit CCS at bit 90
// (immediately below the 36-bit weight area at the top of the block).
void astc_write_header_3x3_rgba_dual(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_3x3_R2_DUAL);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_LDR_RGBA);
}

// Decoder-matching grid-coord of each 6x6-block pixel within an NxN weight
// grid. Derived from mesa's Ds/gs fixed-point formula (not uniformly spaced
// due to the >>6 rounding). Using exact decoder coords makes the encoder's
// SSE estimate match what the decoder will actually reconstruct.
static const float ASTC_3X3_AXIS_POS[6] = { 0.0,  6.0/16.0, 13.0/16.0, 19.0/16.0, 26.0/16.0, 2.0 };
static const float ASTC_4X4_AXIS_POS[6] = { 0.0, 10.0/16.0, 19.0/16.0, 29.0/16.0, 38.0/16.0, 3.0 };
static const float ASTC_5X5_AXIS_POS[6] = { 0.0, 13.0/16.0, 26.0/16.0, 38.0/16.0, 51.0/16.0, 4.0 };

///////////////////////////////////////////////////////////////////////////////
// Range 192 (1 trit + 6 bit) endpoint encoding
//
// Used by CEM 12 (LDR RGBA direct) when the bit budget puts 8 endpoint values
// into ~61 bits — e.g. the 5x5 weight grid with 2-bit weights. The decoder
// picks range 192 (BISE: 1 trit + 6-bit binary) as the highest range that
// fits, so we have to encode that way regardless of intent.
//
// 192 valid encoded values map non-monotonically to 0-255 (alternating low/
// high halves), so we need a 256-entry inverse-quant LUT and a separate
// dequantize formula for the SSE side. Trits pack 5-at-a-time into 8 bits;
// for 8 values we use one full 5-trit group (38 bits) and one partial 3-trit
// group (23 bits) for a total of 61 bits.
///////////////////////////////////////////////////////////////////////////////

uint astc_dequant_r192(uint v) {
	uint B = ((v & 0x3Eu) << 3) | ((v >> 5) & 1u);
	uint D = v >> 6;
	uint T = D * 5u + B;
	uint Axor = (v & 1u) * 0x1FFu;
	T = (T ^ Axor) & 0x1FFu;
	return ((v & 1u) * 0x80u) | (T >> 2);
}

// 256 → range-192 inverse quant LUT (8-bit target → encoded value 0..191).
// Generated offline by enumerating astc_dequant_r192(v) for v in [0, 191] and
// picking, for each target, the v with smallest absolute error.
static const uint astc_r192_quant_lut[256] = {
	  0u,  64u, 128u,   2u,   2u,  66u, 130u,   4u,   4u,  68u, 132u,   6u,   6u,  70u, 134u,   8u,
	  8u,  72u, 136u,  10u,  10u,  74u, 138u,  12u,  12u,  76u, 140u,  14u,  14u,  78u, 142u,  16u,
	 16u,  80u, 144u,  18u,  18u,  82u, 146u,  20u,  20u,  84u, 148u,  22u,  22u,  86u, 150u,  24u,
	 24u,  88u, 152u,  26u,  26u,  90u, 154u,  28u,  28u,  92u, 156u,  30u,  30u,  94u, 158u,  32u,
	 32u,  96u, 160u,  34u,  34u,  98u, 162u,  36u,  36u, 100u, 164u,  38u,  38u, 102u, 166u,  40u,
	 40u, 104u, 168u,  42u,  42u, 106u, 170u,  44u,  44u, 108u, 172u,  46u,  46u, 110u, 174u,  48u,
	 48u, 112u, 176u,  50u,  50u, 114u, 178u,  52u,  52u, 116u, 180u,  54u,  54u, 118u, 182u,  56u,
	 56u, 120u, 184u,  58u,  58u, 122u, 186u,  60u,  60u, 124u, 188u,  62u,  62u, 126u, 190u, 190u,
	191u, 191u, 127u,  63u,  63u, 189u, 125u,  61u,  61u, 187u, 123u,  59u,  59u, 185u, 121u,  57u,
	 57u, 183u, 119u,  55u,  55u, 181u, 117u,  53u,  53u, 179u, 115u,  51u,  51u, 177u, 113u,  49u,
	 49u, 175u, 111u,  47u,  47u, 173u, 109u,  45u,  45u, 171u, 107u,  43u,  43u, 169u, 105u,  41u,
	 41u, 167u, 103u,  39u,  39u, 165u, 101u,  37u,  37u, 163u,  99u,  35u,  35u, 161u,  97u,  33u,
	 33u, 159u,  95u,  31u,  31u, 157u,  93u,  29u,  29u, 155u,  91u,  27u,  27u, 153u,  89u,  25u,
	 25u, 151u,  87u,  23u,  23u, 149u,  85u,  21u,  21u, 147u,  83u,  19u,  19u, 145u,  81u,  17u,
	 17u, 143u,  79u,  15u,  15u, 141u,  77u,  13u,  13u, 139u,  75u,  11u,  11u, 137u,  73u,   9u,
	  9u, 135u,  71u,   7u,   7u, 133u,  69u,   5u,   5u, 131u,  67u,   3u,   3u, 129u,  65u,   1u
};

// Trit BISE pack LUT: 5 trit digits → 8-bit T pattern that mesa's
// unpack_trit_block decodes back to those digits. Index = t0 + 3*t1 + 9*t2 +
// 27*t3 + 81*t4. Generated offline by brute-force inverse search.
static const uint astc_trit_pack_lut[243] = {
	0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x08, 0x09, 0x0A, 0x10, 0x11, 0x12, 0x14, 0x15, 0x16, 0x18,
	0x19, 0x1A, 0x03, 0x07, 0x0B, 0x13, 0x17, 0x1B, 0x0C, 0x0D, 0x0E, 0x20, 0x21, 0x22, 0x24, 0x25,
	0x26, 0x28, 0x29, 0x2A, 0x30, 0x31, 0x32, 0x34, 0x35, 0x36, 0x38, 0x39, 0x3A, 0x23, 0x27, 0x2B,
	0x33, 0x37, 0x3B, 0x2C, 0x2D, 0x2E, 0x40, 0x41, 0x42, 0x44, 0x45, 0x46, 0x48, 0x49, 0x4A, 0x50,
	0x51, 0x52, 0x54, 0x55, 0x56, 0x58, 0x59, 0x5A, 0x43, 0x47, 0x4B, 0x53, 0x57, 0x5B, 0x4C, 0x4D,
	0x4E, 0x80, 0x81, 0x82, 0x84, 0x85, 0x86, 0x88, 0x89, 0x8A, 0x90, 0x91, 0x92, 0x94, 0x95, 0x96,
	0x98, 0x99, 0x9A, 0x83, 0x87, 0x8B, 0x93, 0x97, 0x9B, 0x8C, 0x8D, 0x8E, 0xA0, 0xA1, 0xA2, 0xA4,
	0xA5, 0xA6, 0xA8, 0xA9, 0xAA, 0xB0, 0xB1, 0xB2, 0xB4, 0xB5, 0xB6, 0xB8, 0xB9, 0xBA, 0xA3, 0xA7,
	0xAB, 0xB3, 0xB7, 0xBB, 0xAC, 0xAD, 0xAE, 0xC0, 0xC1, 0xC2, 0xC4, 0xC5, 0xC6, 0xC8, 0xC9, 0xCA,
	0xD0, 0xD1, 0xD2, 0xD4, 0xD5, 0xD6, 0xD8, 0xD9, 0xDA, 0xC3, 0xC7, 0xCB, 0xD3, 0xD7, 0xDB, 0xCC,
	0xCD, 0xCE, 0x60, 0x61, 0x62, 0x64, 0x65, 0x66, 0x68, 0x69, 0x6A, 0x70, 0x71, 0x72, 0x74, 0x75,
	0x76, 0x78, 0x79, 0x7A, 0x63, 0x67, 0x6B, 0x73, 0x77, 0x7B, 0x6C, 0x6D, 0x6E, 0xE0, 0xE1, 0xE2,
	0xE4, 0xE5, 0xE6, 0xE8, 0xE9, 0xEA, 0xF0, 0xF1, 0xF2, 0xF4, 0xF5, 0xF6, 0xF8, 0xF9, 0xFA, 0xE3,
	0xE7, 0xEB, 0xF3, 0xF7, 0xFB, 0xEC, 0xED, 0xEE, 0x1C, 0x1D, 0x1E, 0x3C, 0x3D, 0x3E, 0x5C, 0x5D,
	0x5E, 0x9C, 0x9D, 0x9E, 0xBC, 0xBD, 0xBE, 0xDC, 0xDD, 0xDE, 0x1F, 0x3F, 0x5F, 0x9F, 0xBF, 0xDF,
	0x7C, 0x7D, 0x7E
};

// Pack a full 5-trit group (5 binary 6-bit values + 8 T-bits scattered
// across the trit positions). Total 38 bits; T bits are NOT contiguous, so
// each one is written individually at its specific offset within the group.
void astc_write_trit_group_full(inout uint4 block, uint base_bit, uint T_pattern,
                                uint m0, uint m1, uint m2, uint m3, uint m4) {
	astc_block_write_bits(block, base_bit +  0u, 6u, m0);
	astc_block_write_bits(block, base_bit +  6u, 1u,  T_pattern        & 1u);
	astc_block_write_bits(block, base_bit +  7u, 1u, (T_pattern >> 1u) & 1u);
	astc_block_write_bits(block, base_bit +  8u, 6u, m1);
	astc_block_write_bits(block, base_bit + 14u, 1u, (T_pattern >> 2u) & 1u);
	astc_block_write_bits(block, base_bit + 15u, 1u, (T_pattern >> 3u) & 1u);
	astc_block_write_bits(block, base_bit + 16u, 6u, m2);
	astc_block_write_bits(block, base_bit + 22u, 1u, (T_pattern >> 4u) & 1u);
	astc_block_write_bits(block, base_bit + 23u, 6u, m3);
	astc_block_write_bits(block, base_bit + 29u, 1u, (T_pattern >> 5u) & 1u);
	astc_block_write_bits(block, base_bit + 30u, 1u, (T_pattern >> 6u) & 1u);
	astc_block_write_bits(block, base_bit + 31u, 6u, m4);
	astc_block_write_bits(block, base_bit + 37u, 1u, (T_pattern >> 7u) & 1u);
}

// Pack a partial 3-trit group (3 binary + 5 lower T-bits, total 23 bits).
// T5, T6, T7 sit at offsets 29, 30, 37 within a full group — outside the
// partial group's 23-bit window. We rely on the LUT lookup using the partial
// indexing (idx in 0..26) which always returns T patterns with bits 5, 6, 7
// equal to 0, so we don't need to write them. Decoder reads them as 0 from
// past the partial group's data anyway.
void astc_write_trit_group_partial3(inout uint4 block, uint base_bit, uint T_pattern,
                                    uint m0, uint m1, uint m2) {
	astc_block_write_bits(block, base_bit +  0u, 6u, m0);
	astc_block_write_bits(block, base_bit +  6u, 1u,  T_pattern        & 1u);
	astc_block_write_bits(block, base_bit +  7u, 1u, (T_pattern >> 1u) & 1u);
	astc_block_write_bits(block, base_bit +  8u, 6u, m1);
	astc_block_write_bits(block, base_bit + 14u, 1u, (T_pattern >> 2u) & 1u);
	astc_block_write_bits(block, base_bit + 15u, 1u, (T_pattern >> 3u) & 1u);
	astc_block_write_bits(block, base_bit + 16u, 6u, m2);
	astc_block_write_bits(block, base_bit + 22u, 1u, (T_pattern >> 4u) & 1u);
}

// Pack a partial 1-trit group (1 binary + 2 T-bits, total 8 bits). LUT index
// covers only t0 ∈ [0, 2], so T_pattern uses only bits 0-1; bits 2+ stay 0.
void astc_write_trit_group_partial1(inout uint4 block, uint base_bit, uint T_pattern, uint m0) {
	astc_block_write_bits(block, base_bit + 0u, 6u, m0);
	astc_block_write_bits(block, base_bit + 6u, 1u,  T_pattern        & 1u);
	astc_block_write_bits(block, base_bit + 7u, 1u, (T_pattern >> 1u) & 1u);
}

// CEM 12 endpoints at range 192 (trit + 6-bit BISE) — 8 values in 61 bits,
// the configuration the decoder selects when 5x5 weight grid + 2-bit weights
// leaves 61 endpoint bits. Same value order as RGBA8: (R0, R1, G0, G1, B0,
// B1, A0, A1). Each 8-bit endpoint is mapped to its closest range-192
// encoded form via astc_r192_quant_lut, then packed as one full 5-trit group
// (38 bits at bit 17) plus one partial 3-trit group (23 bits at bit 55).
void astc_write_endpoints_rgba6trit(inout uint4 block, uint3 e0_rgb, uint3 e1_rgb, uint a0, uint a1) {
	uint v0 = astc_r192_quant_lut[min(e0_rgb.r, 255u)];
	uint v1 = astc_r192_quant_lut[min(e1_rgb.r, 255u)];
	uint v2 = astc_r192_quant_lut[min(e0_rgb.g, 255u)];
	uint v3 = astc_r192_quant_lut[min(e1_rgb.g, 255u)];
	uint v4 = astc_r192_quant_lut[min(e0_rgb.b, 255u)];
	uint v5 = astc_r192_quant_lut[min(e1_rgb.b, 255u)];
	uint v6 = astc_r192_quant_lut[min(a0, 255u)];
	uint v7 = astc_r192_quant_lut[min(a1, 255u)];

	// Each value = (trit_digit << 6) | binary_6bit.
	uint t0 = v0 >> 6, t1 = v1 >> 6, t2 = v2 >> 6, t3 = v3 >> 6;
	uint t4 = v4 >> 6, t5 = v5 >> 6, t6 = v6 >> 6, t7 = v7 >> 6;
	uint m0 = v0 & 63u, m1 = v1 & 63u, m2 = v2 & 63u, m3 = v3 & 63u;
	uint m4 = v4 & 63u, m5 = v5 & 63u, m6 = v6 & 63u, m7 = v7 & 63u;

	// Group 0: 5 values (R0, R1, G0, G1, B0).
	uint T_full = astc_trit_pack_lut[t0 + 3u*t1 + 9u*t2 + 27u*t3 + 81u*t4];
	astc_write_trit_group_full(block, 17u, T_full, m0, m1, m2, m3, m4);

	// Group 1 (partial): 3 values (B1, A0, A1). LUT index uses only the
	// first three trits — t3, t4 are implicitly 0 in this lookup, which
	// guarantees the returned T pattern has bits 5, 6, 7 equal to 0.
	uint T_part = astc_trit_pack_lut[t5 + 3u*t6 + 9u*t7];
	astc_write_trit_group_partial3(block, 55u, T_part, m5, m6, m7);
}

// Top-bit-preserving range-192 inverse quant LUTs. Same shape as
// astc_r192_quant_lut (target 8-bit value → encoded value 0..191), but
// constrained so the dequantized value has the same top 2 (or 4) bits as
// the input. Required for HDR endpoints (CEM 11) — the top bits of v[1..5]
// carry mode/majcomp info that the HDR decoder reads to interpret the rest
// of the bits, so a naive range-192 quant scrambles the decode.
//
// Generated offline by enumerating astc_dequant_r192(q) for q ∈ [0, 191],
// then for each input v picking the q whose dequantized form has matching
// top bits AND minimum |dequant(q) - v|. Both LUTs were verified to require
// no fallback (every 8-bit input has a top-bit-preserving range-192
// candidate) with max_err = 1, avg_err = 0.25 — basically lossless.

static const uint astc_r192_quant_lut_preserve_top2[256] = {
	  0u,  64u, 128u,   2u,   2u,  66u, 130u,   4u,   4u,  68u, 132u,   6u,   6u,  70u, 134u,   8u,
	  8u,  72u, 136u,  10u,  10u,  74u, 138u,  12u,  12u,  76u, 140u,  14u,  14u,  78u, 142u,  16u,
	 16u,  80u, 144u,  18u,  18u,  82u, 146u,  20u,  20u,  84u, 148u,  22u,  22u,  86u, 150u,  24u,
	 24u,  88u, 152u,  26u,  26u,  90u, 154u,  28u,  28u,  92u, 156u,  30u,  30u,  94u, 158u, 158u,
	 32u,  96u, 160u,  34u,  34u,  98u, 162u,  36u,  36u, 100u, 164u,  38u,  38u, 102u, 166u,  40u,
	 40u, 104u, 168u,  42u,  42u, 106u, 170u,  44u,  44u, 108u, 172u,  46u,  46u, 110u, 174u,  48u,
	 48u, 112u, 176u,  50u,  50u, 114u, 178u,  52u,  52u, 116u, 180u,  54u,  54u, 118u, 182u,  56u,
	 56u, 120u, 184u,  58u,  58u, 122u, 186u,  60u,  60u, 124u, 188u,  62u,  62u, 126u, 190u, 190u,
	191u, 191u, 127u,  63u,  63u, 189u, 125u,  61u,  61u, 187u, 123u,  59u,  59u, 185u, 121u,  57u,
	 57u, 183u, 119u,  55u,  55u, 181u, 117u,  53u,  53u, 179u, 115u,  51u,  51u, 177u, 113u,  49u,
	 49u, 175u, 111u,  47u,  47u, 173u, 109u,  45u,  45u, 171u, 107u,  43u,  43u, 169u, 105u,  41u,
	 41u, 167u, 103u,  39u,  39u, 165u, 101u,  37u,  37u, 163u,  99u,  35u,  35u, 161u,  97u,  33u,
	159u, 159u,  95u,  31u,  31u, 157u,  93u,  29u,  29u, 155u,  91u,  27u,  27u, 153u,  89u,  25u,
	 25u, 151u,  87u,  23u,  23u, 149u,  85u,  21u,  21u, 147u,  83u,  19u,  19u, 145u,  81u,  17u,
	 17u, 143u,  79u,  15u,  15u, 141u,  77u,  13u,  13u, 139u,  75u,  11u,  11u, 137u,  73u,   9u,
	  9u, 135u,  71u,   7u,   7u, 133u,  69u,   5u,   5u, 131u,  67u,   3u,   3u, 129u,  65u,   1u,
};

static const uint astc_r192_quant_lut_preserve_top4[256] = {
	  0u,  64u, 128u,   2u,   2u,  66u, 130u,   4u,   4u,  68u, 132u,   6u,   6u,  70u, 134u, 134u,
	  8u,  72u, 136u,  10u,  10u,  74u, 138u,  12u,  12u,  76u, 140u,  14u,  14u,  78u, 142u, 142u,
	 16u,  80u, 144u,  18u,  18u,  82u, 146u,  20u,  20u,  84u, 148u,  22u,  22u,  86u, 150u, 150u,
	 24u,  88u, 152u,  26u,  26u,  90u, 154u,  28u,  28u,  92u, 156u,  30u,  30u,  94u, 158u, 158u,
	 32u,  96u, 160u,  34u,  34u,  98u, 162u,  36u,  36u, 100u, 164u,  38u,  38u, 102u, 166u, 166u,
	 40u, 104u, 168u,  42u,  42u, 106u, 170u,  44u,  44u, 108u, 172u,  46u,  46u, 110u, 174u, 174u,
	 48u, 112u, 176u,  50u,  50u, 114u, 178u,  52u,  52u, 116u, 180u,  54u,  54u, 118u, 182u, 182u,
	 56u, 120u, 184u,  58u,  58u, 122u, 186u,  60u,  60u, 124u, 188u,  62u,  62u, 126u, 190u, 190u,
	191u, 191u, 127u,  63u,  63u, 189u, 125u,  61u,  61u, 187u, 123u,  59u,  59u, 185u, 121u,  57u,
	183u, 183u, 119u,  55u,  55u, 181u, 117u,  53u,  53u, 179u, 115u,  51u,  51u, 177u, 113u,  49u,
	175u, 175u, 111u,  47u,  47u, 173u, 109u,  45u,  45u, 171u, 107u,  43u,  43u, 169u, 105u,  41u,
	167u, 167u, 103u,  39u,  39u, 165u, 101u,  37u,  37u, 163u,  99u,  35u,  35u, 161u,  97u,  33u,
	159u, 159u,  95u,  31u,  31u, 157u,  93u,  29u,  29u, 155u,  91u,  27u,  27u, 153u,  89u,  25u,
	151u, 151u,  87u,  23u,  23u, 149u,  85u,  21u,  21u, 147u,  83u,  19u,  19u, 145u,  81u,  17u,
	143u, 143u,  79u,  15u,  15u, 141u,  77u,  13u,  13u, 139u,  75u,  11u,  11u, 137u,  73u,   9u,
	135u, 135u,  71u,   7u,   7u, 133u,  69u,   5u,   5u, 131u,  67u,   3u,   3u, 129u,  65u,   1u,
};

// HDR endpoints (CEM 11) at range 192. Same packing layout as range-256 v6
// — six bytes (a, c, b0, b1, d0, d1) — but quantized through the bit-
// preserving LUTs so the mode/majcomp bits in v[1..5] high positions survive
// the dequant roundtrip. Used by 8x4 weight grid in 8x8 blocks (60 wt bits
// + 17 hdr + 46 ep = 123 used). v0 (a) uses no preservation since a's bits
// are entirely value bits in all HDR modes.
//
// Bit layout: 1 full 5-trit group (38 bits at bit 17) + 1 partial 1-trit
// group (8 bits at bit 55).
void astc_write_endpoints_v6_r192_hdr(inout uint4 block, uint v0, uint v1, uint v2, uint v3, uint v4, uint v5) {
	uint q0 = astc_r192_quant_lut              [v0];   // a — pure value
	uint q1 = astc_r192_quant_lut_preserve_top2[v1];   // c  — top 2 = mode + a_bit_8
	uint q2 = astc_r192_quant_lut_preserve_top2[v2];   // b0 — top 2 = mode + scattered
	uint q3 = astc_r192_quant_lut_preserve_top2[v3];   // b1 — top 2 = mode + scattered
	uint q4 = astc_r192_quant_lut_preserve_top4[v4];   // d0 — top 4 = majcomp + scattered + d0 high bit
	uint q5 = astc_r192_quant_lut_preserve_top4[v5];   // d1 — top 4 = majcomp + scattered + d1 high bit

	// Each value = (trit_digit << 6) | binary_6bit.
	uint t0 = q0 >> 6, t1 = q1 >> 6, t2 = q2 >> 6, t3 = q3 >> 6, t4 = q4 >> 6, t5 = q5 >> 6;
	uint m0 = q0 & 63u, m1 = q1 & 63u, m2 = q2 & 63u, m3 = q3 & 63u, m4 = q4 & 63u, m5 = q5 & 63u;

	// Group 0: 5 values (a, c, b0, b1, d0).
	uint T_full = astc_trit_pack_lut[t0 + 3u*t1 + 9u*t2 + 27u*t3 + 81u*t4];
	astc_write_trit_group_full(block, 17u, T_full, m0, m1, m2, m3, m4);

	// Group 1 (partial): 1 value (d1). LUT index covers t0 ∈ [0, 2] only,
	// so the T pattern uses just bits 0-1 — exactly the partial1 layout.
	uint T_part = astc_trit_pack_lut[t5];
	astc_write_trit_group_partial1(block, 55u, T_part, m5);
}

///////////////////////////////////////////////////////////////////////////////
// HDR RGB direct (CEM 11) — 8x8 block with 4x4 weight grid
//
// Operates in LNS (Logarithmic Number System) space: a 16-bit code (0..65535)
// representing a log-FP encoding of an FP16 value. Linear FP source → LNS via
// astc_float_to_lns. The encoder picks the highest-precision of 8 bit-layout
// modes that fits the bbox; each mode trades range for precision in different
// places (a/b/c/d field widths). All 8 produce CEM 11 endpoints, decoded by
// hardware to FP16 RGB pairs.
//
// Header writers for the 8x8 HDR encoder's two block modes. Both use CEM 11
// (HDR RGB direct), single partition, no dual plane. Endpoints fit 6 bytes
// at range 256 (8-bit binary, no BISE) in both modes; only the weight grid
// + precision differ.
//
// Mode A — 6x5 + 2-bit:  17 hdr + 48 ep + 60 wts = 125 bits (3 wasted)
//   Best for detail/edges/textured content (more spatial points).
// Mode B — 4x4 + 3-bit:  17 hdr + 48 ep + 48 wts = 113 bits (15 wasted)
//   Best for smooth gradients (8 weight levels = finer along-axis steps).
// Mode C — 8x4 + 2-bit:  17 hdr + 46 ep + 64 wts = 127 bits (1 wasted)
//   Best for horizontal-detail content (8 X-points per row). Endpoints drop
//   to range 192 (BISE trit+6) — needs the bit-preserving HDR endpoint
//   writer above (astc_write_endpoints_v6_r192_hdr) to keep mode bits intact.
///////////////////////////////////////////////////////////////////////////////

// Block mode for 8x4 weight grid + 2-bit weights. Sub-mode 01 (wt_w = b+8,
// wt_h = a+2): for wt_w=8 → b=0, for wt_h=4 → a=2, R=2 (2-bit weights).
//   bits set: 1 (R[1]), 2 (sub[0]), 6 (A[1]) → 0x46
static const uint ASTC_BLOCK_MODE_8x4_R2 = 0x046u;

void astc_write_header_8x8_hdr_rgb_6x5(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_6x5_R2);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_HDR_RGB);
}

void astc_write_header_8x8_hdr_rgb_4x4(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_4x4_R3);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_HDR_RGB);
}

// 4x4 grid + trit+2bit (12 lvl) weights, CEM 11 HDR RGB. +50% axial
// precision over the 3-bit (8 lvl) version at the same 4x4 grid +
// range-256 endpoints. Bit budget: 17 + 58 + 48 = 123 / 128 used.
void astc_write_header_8x8_hdr_rgb_4x4_r12(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_4x4_R12_TRIT);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_HDR_RGB);
}

void astc_write_header_8x8_hdr_rgb_5x4(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_5x4_R3);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_HDR_RGB);
}

void astc_write_header_8x8_hdr_rgb_3x3(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_3x3_R5);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_HDR_RGB);
}

void astc_write_header_8x8_hdr_rgb_8x4(inout uint4 block) {
	astc_block_write_bits(block,  0u, 11u, ASTC_BLOCK_MODE_8x4_R2);
	astc_block_write_bits(block, 13u,  4u, ASTC_CEM_HDR_RGB);
}

// Write 6 raw endpoint bytes at the standard CEM 8/11 positions (8-bit binary,
// range 256). For CEM 11 the bytes are (a, c, b0, b1, d0, d1) — order is
// CEM-specific so the caller passes already-packed values from
// astc_quantize_hdr_rgb. No interleaving like CEM 8 RGB; just six raw bytes.
void astc_write_endpoints_v6(inout uint4 block, uint v0, uint v1, uint v2, uint v3, uint v4, uint v5) {
	astc_block_write_bits(block, 17u, 8u, v0);
	astc_block_write_bits(block, 25u, 8u, v1);
	astc_block_write_bits(block, 33u, 8u, v2);
	astc_block_write_bits(block, 41u, 8u, v3);
	astc_block_write_bits(block, 49u, 8u, v4);
	astc_block_write_bits(block, 57u, 8u, v5);
}

// frexp equivalent: split a positive float into mantissa in [0.5, 1) and
// integer exponent. Mirrors astcenc's frexp() in vecmathlib (bit-fiddles the
// IEEE encoding rather than calling math intrinsics).
float astc_frexp(float a, out int exp_out) {
	uint ai = asuint(a);
	exp_out = int((ai >> 23u) & 0xFFu) - 126;
	uint manti = (ai & 0x807FFFFFu) | 0x3F000000u;
	return asfloat(manti);
}

// Convert a positive float to the 16-bit LNS code used by HDR ASTC encoding.
// Direct port of astcenc's float_to_lns. Output range [0, 65535] represents
// the log-FP value that the HDR endpoint encoder operates on.
//
// Roughly: piecewise-linear approximation of log2 with mantissa-based
// remapping, plus exponent contribution at 2048 LNS units per FP exponent.
float astc_float_to_lns(float a) {
	int exp_v;
	float mant = astc_frexp(a, exp_v);

	bool mask_underflow_nan = !(a > (1.0 / 67108864.0));
	bool mask_infinity      = a >= 65536.0;

	// For very small values (exp < -13) skip the mantissa remap and just
	// scale by 2^25 — keeps the LNS code monotonic near zero.
	bool exp_lt_m13 = exp_v < -13;
	float a1a = a * 33554432.0;
	float a1b = (mant - 0.5) * 4096.0;
	int   expb = exp_v + 14;
	a     = exp_lt_m13 ? a1a : a1b;
	exp_v = exp_lt_m13 ? 0   : expb;

	// 3-piece linear approximation in [0, 2048]: matches log2 to ~3 LSBs.
	bool a_lt_384  = a <  384.0;
	bool a_lt_1408 = a <= 1408.0;
	float a2a = a * (4.0 / 3.0);
	float a2b = a + 128.0;
	float a2c = (a + 512.0) * (4.0 / 5.0);
	a = a_lt_384 ? a2a : (a_lt_1408 ? a2b : a2c);

	a = a + (float(exp_v) * 2048.0) + 1.0;
	if (mask_infinity)      a = 65535.0;
	if (mask_underflow_nan) a = 0.0;
	return a;
}

// Round-to-nearest float→int. Matches astcenc's flt2int_rtn (with HLSL's int()
// cast handling negatives toward zero, we add 0.5 with sign).
int astc_flt2int_rtn(float v) {
	return int(v >= 0.0 ? (v + 0.5) : (v - 0.5));
}

// HDR endpoint encoding (CEM 11). Direct port of astcenc's quantize_hdr_rgb.
// Tries 8 bit-allocation modes from highest precision (mode 7) to lowest
// (mode 0), picking the first that accommodates the (color0, color1) span.
// Falls back to a flat 8-bit-per-channel layout if none fit, indicated by
// (majcomp = 3) in the output. Output is 6 raw bytes ready to write at
// endpoint positions v0..v5 with QUANT_256 (no BISE, pure 8-bit binary).
//
// Inputs are LNS-encoded color endpoints (0..65535 each channel).
void astc_quantize_hdr_rgb(float3 color0_in, float3 color1_in, out uint v0, out uint v1, out uint v2, out uint v3, out uint v4, out uint v5) {
	float3 color0 = clamp(color0_in, 0.0, 65535.0);
	float3 color1 = clamp(color1_in, 0.0, 65535.0);
	float3 color0_bak = color0;
	float3 color1_bak = color1;

	// Majority component: which of R/G/B is largest in color1. Encoder
	// reorganizes channels so the majority is "red", then signals the swap
	// via the majcomp bits in d0/d1.
	int majcomp;
	if (color1.r > color1.g && color1.r > color1.b) majcomp = 0;
	else if (color1.g > color1.b)                   majcomp = 1;
	else                                            majcomp = 2;

	if (majcomp == 1) {
		// Red↔Green swap: encoder will write the green channel as "a" and
		// signal the swap so the decoder undoes it before sampling.
		color0 = float3(color0_bak.g, color0_bak.r, color0_bak.b);
		color1 = float3(color1_bak.g, color1_bak.r, color1_bak.b);
	} else if (majcomp == 2) {
		// Red↔Blue swap.
		color0 = float3(color0_bak.b, color0_bak.g, color0_bak.r);
		color1 = float3(color1_bak.b, color1_bak.g, color1_bak.r);
	}

	float a_base  = clamp(color1.r, 0.0, 65535.0);
	float b0_base = a_base - color1.g;
	float b1_base = a_base - color1.b;
	float c_base  = a_base - color0.r;
	float d0_base = a_base - b0_base - c_base - color0.g;
	float d1_base = a_base - b1_base - c_base - color0.b;

	// mode_bits[mode] = (a_bits, b_bits, c_bits, d_bits)
	// mode_cutoffs[mode] = (b_cutoff, c_cutoff, d_cutoff, fits_marker)
	// mode_scales[mode] = LNS-to-mode scale; mode_rscales = inverse.
	// Higher modes = more a-bits but tighter cutoffs.
	static const float mode_cutoffs_b[8] = { 16384, 32768,  4096,  8192,  8192,  2048,  2048,  1024 };
	static const float mode_cutoffs_c[8] = {  8192,  8192,  8192,  8192,  2048,  8192,  2048,  2048 };
	static const float mode_cutoffs_d[8] = {  8192,  4096,  4096,  2048,   512,  1024,   256,   512 };
	static const float mode_scales [8] = {
		1.0/128.0, 1.0/128.0, 1.0/64.0, 1.0/64.0, 1.0/32.0, 1.0/32.0, 1.0/16.0, 1.0/16.0
	};
	static const float mode_rscales[8] = { 128.0, 128.0, 64.0, 64.0, 32.0, 32.0, 16.0, 16.0 };
	static const int   mode_b_bits [8] = { 7, 8, 6, 7, 8, 6, 7, 6 };
	static const int   mode_c_bits [8] = { 6, 6, 7, 7, 6, 8, 7, 7 };
	static const int   mode_d_bits [8] = { 7, 6, 7, 6, 5, 6, 5, 6 };

	// At QUANT_256 the quant/unquant calls in the reference encoder are
	// identity — we just keep the lower bits of each *_intval and merge in
	// the mode/majcomp bits. The recompute steps then collapse to feeding
	// the same intval back through, so we can write outputs directly without
	// rerunning the math.
	[unroll] for (int mode = 7; mode >= 0; mode--) {
		float b_cut = mode_cutoffs_b[mode];
		float c_cut = mode_cutoffs_c[mode];
		float d_cut = mode_cutoffs_d[mode];
		if (b0_base > b_cut || b1_base > b_cut || c_base > c_cut ||
		    abs(d0_base) > d_cut || abs(d1_base) > d_cut)
			continue;

		float ms  = mode_scales [mode];
		float mrs = mode_rscales[mode];
		int b_intcut = 1 << mode_b_bits[mode];
		int c_intcut = 1 << mode_c_bits[mode];
		int d_intcut = 1 << (mode_d_bits[mode] - 1);

		int   a_intval = astc_flt2int_rtn(a_base * ms);
		float a_fval   = float(a_intval) * mrs;

		float c_fval = clamp(a_fval - color0.r, 0.0, 65535.0);
		int   c_intval = astc_flt2int_rtn(c_fval * ms);
		if (c_intval >= c_intcut) continue;
		c_fval = float(c_intval) * mrs;

		float b0_fval = clamp(a_fval - color1.g, 0.0, 65535.0);
		float b1_fval = clamp(a_fval - color1.b, 0.0, 65535.0);
		int   b0_intval = astc_flt2int_rtn(b0_fval * ms);
		int   b1_intval = astc_flt2int_rtn(b1_fval * ms);
		if (b0_intval >= b_intcut || b1_intval >= b_intcut) continue;
		b0_fval = float(b0_intval) * mrs;
		b1_fval = float(b1_intval) * mrs;

		float d0_fval = clamp(a_fval - b0_fval - c_fval - color0.g, -65535.0, 65535.0);
		float d1_fval = clamp(a_fval - b1_fval - c_fval - color0.b, -65535.0, 65535.0);
		int   d0_intval = astc_flt2int_rtn(d0_fval * ms);
		int   d1_intval = astc_flt2int_rtn(d1_fval * ms);
		if (abs(d0_intval) >= d_intcut || abs(d1_intval) >= d_intcut) continue;

		// Pack the mode/majcomp bits into the unused upper bits of c, b0,
		// b1, d0, d1. Each mode uses a different scattering — see astcenc's
		// quantize_hdr_rgb for the reference table.
		int bit0 = 0, bit1 = 0, bit2 = 0, bit3 = 0, bit4 = 0, bit5 = 0;
		if (mode == 0 || mode == 1 || mode == 3 || mode == 4 || mode == 6)
			bit0 = (b0_intval >> 6) & 1;
		else
			bit0 = (a_intval  >> 9) & 1;

		if (mode == 0 || mode == 1 || mode == 3 || mode == 4 || mode == 6)
			bit1 = (b1_intval >> 6) & 1;
		else if (mode == 2)
			bit1 = (c_intval  >> 6) & 1;
		else
			bit1 = (a_intval  >> 10) & 1;

		if (mode == 0 || mode == 2)      bit2 = (d0_intval >> 6) & 1;
		else if (mode == 1 || mode == 4) bit2 = (b0_intval >> 7) & 1;
		else if (mode == 3)              bit2 = (a_intval  >> 9) & 1;
		else if (mode == 5)              bit2 = (c_intval  >> 7) & 1;
		else                             bit2 = (a_intval  >> 11) & 1;

		if (mode == 0 || mode == 2)      bit3 = (d1_intval >> 6) & 1;
		else if (mode == 1 || mode == 4) bit3 = (b1_intval >> 7) & 1;
		else                             bit3 = (c_intval  >> 6) & 1;

		if (mode == 4 || mode == 6) {
			bit4 = (a_intval  >> 9) & 1;
			bit5 = (a_intval  >> 10) & 1;
		} else {
			bit4 = (d0_intval >> 5) & 1;
			bit5 = (d1_intval >> 5) & 1;
		}

		uint a_lo  = uint(a_intval  & 0xFF);
		uint c_lo  = uint(c_intval  & 0x3F) | (uint(mode & 1) << 7) | (uint((a_intval >> 8) & 1) << 6);
		uint b0_lo = uint(b0_intval & 0x3F) | (uint(bit0) << 6) | (uint((mode >> 1) & 1) << 7);
		uint b1_lo = uint(b1_intval & 0x3F) | (uint(bit1) << 6) | (uint((mode >> 2) & 1) << 7);
		uint d0_lo = uint(d0_intval & 0x1F) | (uint(bit2) << 6) | (uint(bit4) << 5) | (uint(majcomp & 1) << 7);
		uint d1_lo = uint(d1_intval & 0x1F) | (uint(bit3) << 6) | (uint(bit5) << 5) | (uint((majcomp >> 1) & 1) << 7);

		v0 = a_lo;  v1 = c_lo;  v2 = b0_lo;  v3 = b1_lo;  v4 = d0_lo;  v5 = d1_lo;
		return;
	}

	// Fallback — flat representation: 8-bit R0/R1/G0/G1, 7-bit B0/B1 + majcomp=3
	// signal in the high bits of v4/v5. Lower precision but accommodates any
	// bbox the modes couldn't fit (typically light/dark ratio > 4×).
	float vals[6] = {
		clamp(color0_bak.r, 0.0, 65020.0),
		clamp(color1_bak.r, 0.0, 65020.0),
		clamp(color0_bak.g, 0.0, 65020.0),
		clamp(color1_bak.g, 0.0, 65020.0),
		clamp(color0_bak.b, 0.0, 65020.0),
		clamp(color1_bak.b, 0.0, 65020.0)
	};
	v0 = uint(astc_flt2int_rtn(vals[0] / 256.0)) & 0xFFu;
	v1 = uint(astc_flt2int_rtn(vals[1] / 256.0)) & 0xFFu;
	v2 = uint(astc_flt2int_rtn(vals[2] / 256.0)) & 0xFFu;
	v3 = uint(astc_flt2int_rtn(vals[3] / 256.0)) & 0xFFu;
	// vals[4..5] in [0, 65020]; /512 + 128 lands in [128, 254], bit 7 always
	// set. Decoder reads majcomp = v4_bit7 | (v5_bit7 << 1) — both set →
	// majcomp = 3 → blue fallback path. The low 7 bits carry the blue value
	// (decoder uses (v4 & 0x7F) << 9 to scale to fp16). At QUANT_256 the
	// reference encoder's retain_top_two_bits is identity.
	v4 = uint(astc_flt2int_rtn(vals[4] / 512.0) + 128) & 0xFFu;
	v5 = uint(astc_flt2int_rtn(vals[5] / 512.0) + 128) & 0xFFu;
}

///////////////////////////////////////////////////////////////////////////////
// BISE-encoded weights
//
// ASTC supports non-power-of-2 weight ranges via BISE (trits + binary bits).
// Useful configurations for squeezing extra axial precision out of the
// weight budget:
//
//   wt_range 6 (H=0): trit + 1-bit binary → 6 levels per weight
//   wt_range 3 (H=1): trit + 2-bit binary → 12 levels per weight
//
// Weights are packed in groups: 5 weights per full trit group (8 T-bits +
// 5·B binary bits), plus a partial tail group for the remainder. Unlike
// endpoints (which live at bit 17 growing up), weights live at the TOP of
// the block bit-reversed — so we count natural offsets DOWN from bit 127
// and write each bit/field at (127 - natural_offset).
//
// Block mode constants (standard sub-mode 00 with wt_range via
// (bits[1:0] << 1) | bit[4], + H bit at bit 9):
///////////////////////////////////////////////////////////////////////////////

// (Block mode constants for 4x4_R6_TRIT, 4x4_R12_TRIT, 5x4_R3, 3x3_R5 are
// declared at the top of this file alongside the other ASTC_BLOCK_MODE_*
// constants, so the header writers above can reference them.)

// Quantize a normalized weight target in [0, 1] to trit+1bit (6 levels) and
// return the encoded value v ∈ [0, 5]. Thresholds and LEVEL_TO_V derived
// from mesa's unquantise_weights for (wt_trits=1, wt_bits=1): unquantized
// levels land at {0, 12, 25, 38, 51, 63} / 63, and the trit encoding
// scrambles v-to-level (v∈[0,5] maps to level indices {0, 5, 2, 4, 1, 3}).
uint astc_quantize_weight_trit_1bit(float target) {
	uint level =
		uint(target >=  6.0 / 63.0) +
		uint(target >= 18.5 / 63.0) +
		uint(target >= 31.5 / 63.0) +
		uint(target >= 44.5 / 63.0) +
		uint(target >= 57.0 / 63.0);
	static const uint LEVEL_TO_V[6] = { 0u, 2u, 4u, 5u, 3u, 1u };
	return LEVEL_TO_V[level];
}

// Quantize a normalized weight target in [0, 1] to trit+2bit (12 levels)
// and return the encoded value v ∈ [0, 11]. Levels {0, 5, 11, 18, 24, 30,
// 33, 39, 45, 52, 58, 63} / 63.
uint astc_quantize_weight_trit_2bit(float target) {
	uint level =
		uint(target >=  2.5 / 63.0) +
		uint(target >=  8.0 / 63.0) +
		uint(target >= 14.5 / 63.0) +
		uint(target >= 21.0 / 63.0) +
		uint(target >= 27.0 / 63.0) +
		uint(target >= 31.5 / 63.0) +
		uint(target >= 36.0 / 63.0) +
		uint(target >= 42.0 / 63.0) +
		uint(target >= 48.5 / 63.0) +
		uint(target >= 55.0 / 63.0) +
		uint(target >= 60.5 / 63.0);
	static const uint LEVEL_TO_V[12] = { 0u, 4u, 8u, 2u, 6u, 10u, 11u, 7u, 3u, 9u, 5u, 1u };
	return LEVEL_TO_V[level];
}

// Full 5-weight trit+1bit group (13 bits total: 5·1 binary + 8 T-bits)
// written at weight-stream natural offset `base`. `T_pattern` comes from
// astc_trit_pack_lut indexed by (t0 + 3·t1 + 9·t2 + 27·t3 + 81·t4); m0..m4
// are the 1-bit binary parts (value bit 0 of each encoded weight).
void astc_write_weight_trit_1bit_full(inout uint4 block, uint base, uint T_pattern,
                                      uint m0, uint m1, uint m2, uint m3, uint m4) {
	// Layout (natural offsets from `base`):
	//   m0 @ 0, T0 @ 1, T1 @ 2, m1 @ 3, T2 @ 4, T3 @ 5, m2 @ 6, T4 @ 7,
	//   m3 @ 8, T5 @ 9, T6 @ 10, m4 @ 11, T7 @ 12
	astc_block_write_bits(block, 127u - base -  0u, 1u, m0 & 1u);
	astc_block_write_bits(block, 127u - base -  1u, 1u,  T_pattern        & 1u);
	astc_block_write_bits(block, 127u - base -  2u, 1u, (T_pattern >> 1u) & 1u);
	astc_block_write_bits(block, 127u - base -  3u, 1u, m1 & 1u);
	astc_block_write_bits(block, 127u - base -  4u, 1u, (T_pattern >> 2u) & 1u);
	astc_block_write_bits(block, 127u - base -  5u, 1u, (T_pattern >> 3u) & 1u);
	astc_block_write_bits(block, 127u - base -  6u, 1u, m2 & 1u);
	astc_block_write_bits(block, 127u - base -  7u, 1u, (T_pattern >> 4u) & 1u);
	astc_block_write_bits(block, 127u - base -  8u, 1u, m3 & 1u);
	astc_block_write_bits(block, 127u - base -  9u, 1u, (T_pattern >> 5u) & 1u);
	astc_block_write_bits(block, 127u - base - 10u, 1u, (T_pattern >> 6u) & 1u);
	astc_block_write_bits(block, 127u - base - 11u, 1u, m4 & 1u);
	astc_block_write_bits(block, 127u - base - 12u, 1u, (T_pattern >> 7u) & 1u);
}

// Partial 1-weight trit+1bit group (3 bits: m0 + T0 + T1) at natural offset
// `base`. Used for the 16th weight in a 16-weight grid after 3 full groups.
// LUT indexed by (t0) gives T pattern with bits 2-7 == 0, so only T0/T1
// matter for reconstruction.
void astc_write_weight_trit_1bit_partial1(inout uint4 block, uint base, uint T_pattern, uint m0) {
	astc_block_write_bits(block, 127u - base - 0u, 1u, m0 & 1u);
	astc_block_write_bits(block, 127u - base - 1u, 1u,  T_pattern        & 1u);
	astc_block_write_bits(block, 127u - base - 2u, 1u, (T_pattern >> 1u) & 1u);
}

// Full 5-weight trit+2bit group (18 bits total: 5·2 binary + 8 T-bits).
// Same T-bit positions as the trit+1bit case but m_i are 2-bit binary
// values, bit-reversed into their natural slot.
void astc_write_weight_trit_2bit_full(inout uint4 block, uint base, uint T_pattern,
                                      uint m0, uint m1, uint m2, uint m3, uint m4) {
	// Natural offsets (from `base`):
	//   m0 @ [0,1], T0 @ 2, T1 @ 3, m1 @ [4,5], T2 @ 6, T3 @ 7,
	//   m2 @ [8,9], T4 @ 10, m3 @ [11,12], T5 @ 13, T6 @ 14,
	//   m4 @ [15,16], T7 @ 17.
	// 2-bit values go at (127-base-offset-1) via astc_reverse2 so that bit 0
	// lands at the lower natural offset.
	astc_block_write_bits(block, 127u - base -  1u, 2u, astc_reverse2(m0));
	astc_block_write_bits(block, 127u - base -  2u, 1u,  T_pattern        & 1u);
	astc_block_write_bits(block, 127u - base -  3u, 1u, (T_pattern >> 1u) & 1u);
	astc_block_write_bits(block, 127u - base -  5u, 2u, astc_reverse2(m1));
	astc_block_write_bits(block, 127u - base -  6u, 1u, (T_pattern >> 2u) & 1u);
	astc_block_write_bits(block, 127u - base -  7u, 1u, (T_pattern >> 3u) & 1u);
	astc_block_write_bits(block, 127u - base -  9u, 2u, astc_reverse2(m2));
	astc_block_write_bits(block, 127u - base - 10u, 1u, (T_pattern >> 4u) & 1u);
	astc_block_write_bits(block, 127u - base - 12u, 2u, astc_reverse2(m3));
	astc_block_write_bits(block, 127u - base - 13u, 1u, (T_pattern >> 5u) & 1u);
	astc_block_write_bits(block, 127u - base - 14u, 1u, (T_pattern >> 6u) & 1u);
	astc_block_write_bits(block, 127u - base - 16u, 2u, astc_reverse2(m4));
	astc_block_write_bits(block, 127u - base - 17u, 1u, (T_pattern >> 7u) & 1u);
}

// Partial 1-weight trit+2bit group (4 bits: m0 [2 bits] + T0 + T1) at
// natural offset `base`.
void astc_write_weight_trit_2bit_partial1(inout uint4 block, uint base, uint T_pattern, uint m0) {
	astc_block_write_bits(block, 127u - base - 1u, 2u, astc_reverse2(m0));
	astc_block_write_bits(block, 127u - base - 2u, 1u,  T_pattern        & 1u);
	astc_block_write_bits(block, 127u - base - 3u, 1u, (T_pattern >> 1u) & 1u);
}
