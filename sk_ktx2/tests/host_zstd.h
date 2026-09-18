// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

// The host side of ktx2_context_t. This is the whole of what an engine supplies
// to read Zstd supercompressed UASTC, and it lives here rather than in the
// library so the "no compression code" property stays visible.
//
// Built against the system libzstd when CMake finds one; without it the Zstd
// paths report as skipped rather than silently passing.

#pragma once

#include "sk_ktx2.h"

#ifdef SK_KTX2_TEST_ZSTD

#include <zstd.h>

static size_t host_zstd_inflate(void* context, const void* src, size_t src_bytes,
                                void* out_dst, size_t dst_bytes) {
	(void)context;
	size_t got = ZSTD_decompress(out_dst, dst_bytes, src, src_bytes);
	return ZSTD_isError(got) ? 0 : got;
}

static ktx2_context_t k_host_context = { host_zstd_inflate, NULL, 0, { 0 } };
#define HOST_HAS_ZSTD 1

#else

static ktx2_context_t k_host_context = { NULL, NULL, 0, { 0 } };
#define HOST_HAS_ZSTD 0

#endif
