// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// Minimal Khronos .astc file container.
//
// The .astc format is a 16-byte header followed by raw ASTC blocks in
// row-major order. It is the native I/O format for astcenc, which makes
// it ideal for validating our encoder against astcenc's reference decoder:
//   astcenc-sse4.1 -dl ours.astc decoded.png
//
// Header layout (little-endian):
//   magic[4]   = 0x13, 0xAB, 0xA1, 0x5C
//   block_x    = block footprint in pixels (e.g. 6 for 6x6)
//   block_y    = block footprint in pixels
//   block_z    = 1 (we only write 2D)
//   dim_x[3]   = image width  in pixels (24-bit LE)
//   dim_y[3]   = image height in pixels
//   dim_z[3]   = 1 (we only write 2D)

// Write mip 0 of a compressed block stream to a .astc file.
// Returns true on success.
bool astc_write_file(const char* path,
                     int32_t     width,
                     int32_t     height,
                     int32_t     block_w,
                     int32_t     block_h,
                     const void* block_data,
                     size_t      data_size);
