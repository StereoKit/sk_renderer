//--name = bc1_compress

// BC1 (DXT1) GPU texture compression
//
// Each thread compresses one 4x4 block. Output is an RWStructuredBuffer of
// uint2 (8 bytes per block), laid out in row-major block order.
//
// Port of the CPU bounding-box encoder from tools/tex_compress.c.

Texture2D<float4>         source_tex    : register(t0);
RWStructuredBuffer<uint2> output_blocks : register(u1);

// Pipeline-specialized: the C side creates one compute per value, so the
// branches below fold away at pipeline-compile time.
[[vk::constant_id(0)]] const bool ENABLE_ALPHA = false;  // false = opaque only (4-color), true = punch-through alpha
// Encode linear-light input to sRGB before quantizing, for use with float or
// sRGB-view sources (which Load as linear). BC1's 5:6:5 endpoints band badly
// in the darks if fed linear values; quantizing in sRGB space matches how the
// hardware decodes a *_srgb output format back to linear at sample time.
[[vk::constant_id(1)]] const bool SRGB_ENCODE = false;

float3 linear_to_srgb(float3 c) {
	// IEC 61966-2-1 transfer function
	c = max(c, 0.0);
	float3 lo = c * 12.92;
	float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
	return lerp(lo, hi, step(0.0031308, c));
}

uint mip_level;
uint image_width;
uint image_height;
uint blocks_x;
uint buffer_offset;

///////////////////////////////////////////////////////////////////////////////
// BC1 block encoder
///////////////////////////////////////////////////////////////////////////////

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint bx = id.x;
	uint by = id.y;
	uint blocks_y = (image_height + 3) / 4;
	if (bx >= blocks_x || by >= blocks_y)
		return;

	// Load 4x4 block. Bounding box computed in float space to avoid
	// converting all 16 pixels to uint upfront.
	float3 pixels[16];
	float3 cmin = float3(1, 1, 1);
	float3 cmax = float3(0, 0, 0);

	// Pack transparency into a bitmask instead of bool[16] to save VGPRs
	uint trans_mask = 0;

	[unroll] for (uint py = 0; py < 4; py++) {
		[unroll] for (uint px = 0; px < 4; px++) {
			uint sx = min(bx * 4 + px, image_width  - 1);
			uint sy = min(by * 4 + py, image_height - 1);
			float4 texel = source_tex.Load(int3(sx, sy, mip_level));
			if (SRGB_ENCODE)
				texel.rgb = linear_to_srgb(texel.rgb);

			uint idx = py * 4 + px;
			pixels[idx] = texel.rgb;
			cmin = min(cmin, texel.rgb);
			cmax = max(cmax, texel.rgb);

			if (ENABLE_ALPHA && texel.a < 0.5)
				trans_mask |= (1u << idx);
		}
	}

	bool has_transparent = (trans_mask != 0);
	bool has_opaque      = (trans_mask != 0xFFFF);

	// Alpha path: fully transparent block early-out
	if (ENABLE_ALPHA && !has_opaque) {
		output_blocks[buffer_offset + by * blocks_x + bx] = uint2(0, 0xFFFFFFFF);
		return;
	}

	// Recompute bounding box excluding transparent pixels (branchless)
	if (ENABLE_ALPHA && has_transparent) {
		cmin = float3(1, 1, 1);
		cmax = float3(0, 0, 0);
		[unroll] for (uint i = 0; i < 16; i++) {
			float3 p = pixels[i];
			bool opaque = !((trans_mask >> i) & 1u);
			cmin = min(cmin, opaque ? p : float3(1, 1, 1));
			cmax = max(cmax, opaque ? p : float3(0, 0, 0));
		}
	}

	// Convert bounding box to [0,255] uint for quantization
	uint3 imin = uint3(cmin * 255.0 + 0.5);
	uint3 imax = uint3(cmax * 255.0 + 0.5);

	// Inset by 1/16 of range
	uint3 range = imax - imin;
	imin += range >> 4;
	imax -= range >> 4;

	// Convert endpoints to RGB565
	uint c0 = ((imax.r >> 3) << 11) | ((imax.g >> 2) << 5) | (imax.b >> 3);
	uint c1 = ((imin.r >> 3) << 11) | ((imin.g >> 2) << 5) | (imin.b >> 3);

	// Set up color mode
	if (ENABLE_ALPHA && has_transparent) {
		// 3-color + alpha: need c0 <= c1
		if (c0 > c1) { uint t = c0; c0 = c1; c1 = t; }
		if (c0 == c1 && c0 > 0) c0--;
	} else {
		// 4-color opaque: need c0 > c1
		if (c0 < c1) { uint t = c0; c0 = c1; c1 = t; }
		if (c0 == c1 && c0 < 0xFFFF) c0++;
	}

	// Expand endpoints back to RGB888 via bit replication
	uint3 color0, color1;
	{
		uint r0 = (c0 >> 11) & 0x1F; uint g0 = (c0 >> 5) & 0x3F; uint b0 = c0 & 0x1F;
		uint r1 = (c1 >> 11) & 0x1F; uint g1 = (c1 >> 5) & 0x3F; uint b1 = c1 & 0x1F;
		color0 = uint3((r0 << 3) | (r0 >> 2), (g0 << 2) | (g0 >> 4), (b0 << 3) | (b0 >> 2));
		color1 = uint3((r1 << 3) | (r1 >> 2), (g1 << 2) | (g1 >> 4), (b1 << 3) | (b1 >> 2));
	}

	// Perceptually weighted axis in float space: avoids per-pixel float->uint
	// conversion and replaces 4-cycle v_mul_lo_u32 with 1-cycle v_fma_f32.
	float3 faxis = float3(
		float(int(color1.r) - int(color0.r)) * 2.0,
		float(int(color1.g) - int(color0.g)) * 4.0,
		float(int(color1.b) - int(color0.b)));

	// axis_len_sq with matching weights, scaled by 1/255 to match [0,1] pixel space.
	// Projection gives proj_int/255, so thresholds must also be /255.
	float faxis_len_sq = (faxis.r * faxis.r * 0.5 + faxis.g * faxis.g * 0.25 + faxis.b * faxis.b) / 255.0;
	float fc0_proj     = dot(float3(color0) / 255.0, faxis);

	// Assign indices via projection onto c0-c1 axis.
	// Thresholds are folded to absorb the per-pixel "- fc0_proj" and "* scale",
	// so the inner loop is just: dot(pixel, faxis) >= threshold.
	uint indices = 0;

	if (faxis_len_sq < 1e-6) {
		// Degenerate: all same color — everything index 0, except transparent
		// pixels which need index 3. Indices are 2 bits per pixel, so spread
		// each trans_mask bit j out to bit 2j, then *3 fills both bits.
		if (ENABLE_ALPHA && has_transparent) {
			uint spread = trans_mask;
			spread = (spread | (spread << 8)) & 0x00FF00FFu;
			spread = (spread | (spread << 4)) & 0x0F0F0F0Fu;
			spread = (spread | (spread << 2)) & 0x33333333u;
			spread = (spread | (spread << 1)) & 0x55555555u;
			indices = spread * 3u;
		}
	} else if (ENABLE_ALPHA && has_transparent) {
		// 3-color + alpha mode
		// Original: (dot(p, faxis) - fc0_proj) * 4 >= faxis_len_sq * N
		// Folded:    dot(p, faxis) >= faxis_len_sq * N / 4 + fc0_proj
		float t1 = faxis_len_sq * 0.25 + fc0_proj;
		float t3 = faxis_len_sq * 0.75 + fc0_proj;

		[unroll] for (uint j = 0; j < 16; j++) {
			float d = dot(pixels[j], faxis);
			uint bucket = uint(d >= t1) + uint(d >= t3);
			uint color_idx = (0x18u >> (bucket * 2)) & 3u;
			uint is_trans = (trans_mask >> j) & 1u;
			uint final_idx = is_trans ? 3u : color_idx;
			indices |= (final_idx << (j * 2));
		}
	} else {
		// 4-color opaque mode
		// Original: (dot(p, faxis) - fc0_proj) * 6 >= faxis_len_sq * N
		// Folded:    dot(p, faxis) >= faxis_len_sq * N / 6 + fc0_proj
		float rcp6 = 1.0 / 6.0;
		float t1 = faxis_len_sq       * rcp6 + fc0_proj;
		float t3 = faxis_len_sq * 3.0 * rcp6 + fc0_proj;
		float t5 = faxis_len_sq * 5.0 * rcp6 + fc0_proj;

		[unroll] for (uint j = 0; j < 16; j++) {
			float d = dot(pixels[j], faxis);
			uint bucket = uint(d >= t1) + uint(d >= t3) + uint(d >= t5);
			indices |= (((0x78u >> (bucket * 2)) & 3u) << (j * 2));
		}
	}

	// Pack output: [c0_lo, c0_hi, c1_lo, c1_hi] [idx0..idx3]
	output_blocks[buffer_offset + by * blocks_x + bx] = uint2(c0 | (c1 << 16), indices);
}
