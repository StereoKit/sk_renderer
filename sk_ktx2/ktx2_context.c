// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// The caller-owned context: host services, and storage for the ETC1S conversion
// tables.
//
// Those tables are identical for every file and cost more to build than a small
// texture costs to transcode: 16x16 ETC1S to BC3 is 0.01 ms of block work behind
// 2.3 ms of table building, and a scene of small textures paid that repeatedly.
// Here they are built at most once, and only the ones the format reads.
//
// The storage is a byte array rather than the real types so `sk_ktx2.h` stays
// free of internals. Both are uint8_t arrays underneath, so alignment is moot.

#include "ktx2_internal.h"

#include <string.h>

#define KTX2_GRAY_BYTES (sizeof(ktx2_gray_fit_t) * KTX2_GRAY_FIT_COUNT)
#define KTX2_BC1_BYTES  (sizeof(ktx2_bc1_table_t))

// If this fires, a table grew and KTX2_TABLE_BYTES has to grow with it.
typedef char ktx2_table_storage_check[(KTX2_GRAY_BYTES + KTX2_BC1_BYTES <= KTX2_TABLE_BYTES) ? 1 : -1];

const ktx2_gray_fit_t* ktx2_context_gray(ktx2_context_t* ref_context) {
	ktx2_gray_fit_t* fits = (ktx2_gray_fit_t*)ref_context->tables;
	if ((ref_context->tables_built & KTX2_TABLE_GRAY) == 0) {
		ktx2_gray_fit_build(fits);
		ref_context->tables_built |= KTX2_TABLE_GRAY;
	}
	return fits;
}

const ktx2_bc1_table_t* ktx2_context_bc1(ktx2_context_t* ref_context) {
	ktx2_bc1_table_t* table = (ktx2_bc1_table_t*)(ref_context->tables + KTX2_GRAY_BYTES);
	if ((ref_context->tables_built & KTX2_TABLE_BC1) == 0) {
		ktx2_bc1_table_build(table);
		ref_context->tables_built |= KTX2_TABLE_BC1;
	}
	return table;
}

void ktx2_context_prepare(ktx2_context_t* ref_context) {
	ktx2_context_gray(ref_context);
	ktx2_context_bc1 (ref_context);
}
