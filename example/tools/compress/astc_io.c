// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "astc_io.h"

#include <stdio.h>

bool astc_write_file(const char* path, int32_t width, int32_t height, int32_t block_w, int32_t block_h, const void* block_data, size_t data_size) {
	if (!path || !block_data || data_size == 0) return false;

	uint8_t header[16];
	header[ 0] = 0x13; header[ 1] = 0xAB; header[ 2] = 0xA1; header[ 3] = 0x5C;
	header[ 4] = (uint8_t)block_w;
	header[ 5] = (uint8_t)block_h;
	header[ 6] = 1;
	header[ 7] = (uint8_t)( width        & 0xFF);
	header[ 8] = (uint8_t)((width  >> 8) & 0xFF);
	header[ 9] = (uint8_t)((width  >> 16)& 0xFF);
	header[10] = (uint8_t)( height       & 0xFF);
	header[11] = (uint8_t)((height >> 8) & 0xFF);
	header[12] = (uint8_t)((height >> 16)& 0xFF);
	header[13] = 1;
	header[14] = 0;
	header[15] = 0;

	FILE* f = fopen(path, "wb");
	if (!f) return false;

	bool ok = fwrite(header, 1, sizeof(header), f) == sizeof(header)
	       && fwrite(block_data, 1, data_size, f) == data_size;
	fclose(f);
	return ok;
}
