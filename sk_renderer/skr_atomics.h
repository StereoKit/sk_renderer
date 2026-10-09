// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#pragma once

#include <stdint.h>
#include <stdbool.h>

///////////////////////////////////////////////////////////////////////////////
// Atomics shared by every backend. Registries coordinate writers with one
// mutex and publish to lock-free readers with release/acquire pointer
// stores. MSVC gates C11 <stdatomic.h> behind /experimental:c11atomics, so
// these use the intrinsics both compilers ship. Each name keeps one memory
// order everywhere: plain names are relaxed, others say their order.
///////////////////////////////////////////////////////////////////////////////

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
	// Single-threaded web build, nothing to synchronize
	#define _skr_atomic(T)                 T
	#define _skr_load_acquire(p)           (*(p))
	#define _skr_store_release(p, v)       (*(p) = (v))
	#define _skr_store_u32(p, v)           (*(p) = (v))
	#define _skr_load_u32(p)               (*(p))
	#define _skr_load_u32_acquire(p)       (*(p))
	#define _skr_exchange_u32(p, v)        _skr_st_exchange((uint32_t*)(p), (uint32_t)(v))
	#define _skr_add_u32(p, v)             ((*(p) += (v)) - (v))
	#define _skr_add_acq_rel(p, v)         ((*(p) += (v)) - (v))
	#define _skr_and_acq_rel(p, v)         _skr_st_and((uint32_t*)(p), (uint32_t)(v))
	#define _skr_add_u64(p, v)             ((*(p) += (v)) - (v))
	#define _skr_load_u64(p)               (*(p))
	#define _skr_cas_u64(p, expected, desired) (*(p) == (expected) ? (*(p) = (desired), true) : false)
	#define _skr_load_ptr(p)               (*(p))
	#define _skr_exchange_ptr(p, v)        _skr_st_exchange_ptr((void**)(p), (void*)(v))
	#define _skr_cas_ptr(p, expected, desired) (*(p) == (expected) ? (*(p) = (desired), true) : false)
	#define _skr_yield()                   ((void)0)
	static inline uint32_t _skr_st_exchange    (uint32_t* p, uint32_t v) { uint32_t o = *p; *p = v; return o; }
	static inline uint32_t _skr_st_and         (uint32_t* p, uint32_t v) { uint32_t o = *p; *p = o & v; return o; }
	static inline void*    _skr_st_exchange_ptr(void**    p, void*    v) { void*    o = *p; *p = v; return o; }
#elif defined(_MSC_VER)
	#include <intrin.h>
	#include <threads.h>
	#define _skr_atomic(T)                 T volatile
	#if defined(_M_ARM64) || defined(_M_ARM64EC)
		#define _skr_load_acquire(p)       ((void*)__ldar64((volatile __int64*)(p)))
		#define _skr_load_u32_acquire(p)   ((uint32_t)__ldar32((volatile unsigned __int32*)(p)))
	#else
		#define _skr_load_acquire(p)       (*(p)) // x86/x64 loads carry acquire already
		#define _skr_load_u32_acquire(p)   ((uint32_t)*(p))
	#endif
	// Interlocked operations are full barriers, so every ordering maps to them
	#define _skr_store_release(p, v)       _InterlockedExchangePointer((void* volatile*)(p), (void*)(v))
	#define _skr_store_u32(p, v)           _InterlockedExchange((volatile long*)(p), (long)(v))
	#define _skr_load_u32(p)               (*(p))
	#define _skr_exchange_u32(p, v)        ((uint32_t)_InterlockedExchange((volatile long*)(p), (long)(v)))
	#define _skr_add_u32(p, v)             _InterlockedExchangeAdd((volatile long*)(p), (long)(v))
	#define _skr_add_acq_rel(p, v)         _InterlockedExchangeAdd((volatile long*)(p), (long)(v))
	#define _skr_and_acq_rel(p, v)         ((uint32_t)_InterlockedAnd((volatile long*)(p), (long)(v)))
	#define _skr_add_u64(p, v)             ((uint64_t)_InterlockedExchangeAdd64((volatile __int64*)(p), (__int64)(v)))
	#define _skr_load_u64(p)               (*(p))
	#define _skr_cas_u64(p, expected, desired) (_InterlockedCompareExchange64((volatile __int64*)(p), (__int64)(desired), (__int64)(expected)) == (__int64)(expected))
	#define _skr_load_ptr(p)               (*(p))
	#define _skr_exchange_ptr(p, v)        _InterlockedExchangePointer((void* volatile*)(p), (void*)(v))
	#define _skr_cas_ptr(p, expected, desired) (_InterlockedCompareExchangePointer((void* volatile*)(p), (void*)(desired), (void*)(expected)) == (void*)(expected))
	#define _skr_yield()                   thrd_yield()
#else
	#include <threads.h>
	#define _skr_atomic(T)                 T
	#define _skr_load_acquire(p)           __atomic_load_n    ((p),      __ATOMIC_ACQUIRE)
	#define _skr_store_release(p, v)       __atomic_store_n   ((p), (v), __ATOMIC_RELEASE)
	#define _skr_store_u32(p, v)           __atomic_store_n   ((p), (v), __ATOMIC_RELAXED)
	#define _skr_load_u32(p)               __atomic_load_n    ((p),      __ATOMIC_RELAXED)
	#define _skr_load_u32_acquire(p)       __atomic_load_n    ((p),      __ATOMIC_ACQUIRE)
	#define _skr_exchange_u32(p, v)        __atomic_exchange_n((p), (v), __ATOMIC_RELAXED)
	#define _skr_add_u32(p, v)             __atomic_fetch_add ((p), (v), __ATOMIC_RELAXED)
	#define _skr_add_acq_rel(p, v)         __atomic_fetch_add ((p), (v), __ATOMIC_ACQ_REL)
	#define _skr_and_acq_rel(p, v)         __atomic_fetch_and ((p), (v), __ATOMIC_ACQ_REL)
	#define _skr_add_u64(p, v)             __atomic_fetch_add ((p), (v), __ATOMIC_RELAXED)
	#define _skr_load_u64(p)               __atomic_load_n    ((p),      __ATOMIC_RELAXED)
	#define _skr_cas_u64(p, expected, desired) __atomic_compare_exchange_n((p), &(expected), (desired), false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)
	#define _skr_load_ptr(p)               __atomic_load_n    ((p),      __ATOMIC_RELAXED)
	#define _skr_exchange_ptr(p, v)        __atomic_exchange_n((p), (v), __ATOMIC_ACQ_REL)
	#define _skr_cas_ptr(p, expected, desired) __atomic_compare_exchange_n((p), &(expected), (desired), false, __ATOMIC_RELEASE, __ATOMIC_RELAXED) // MSVC's doesn't write expected back, so callers reload it
	#define _skr_yield()                   thrd_yield()
#endif
