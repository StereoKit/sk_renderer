// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include "include/sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////
// Internal contract between the backend-independent sources (skr_log.c) and
// the active backend. Each backend implements these allocation wrappers,
// honoring the skr_settings_t allocator callbacks once initialized.
///////////////////////////////////////////////////////////////////////////////

void* _skr_malloc (size_t size);
void* _skr_calloc (size_t count, size_t size);
void* _skr_realloc(void* ptr, size_t size);
void  _skr_free   (void* ptr);
