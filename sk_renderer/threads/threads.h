// skr_threads.h - Lightweight C11 threads fallback for platforms without native support

#pragma once

// Use pthread fallback for MinGW and platforms without C11 threads
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

	// thread_local support
#ifndef thread_local
	#if defined(_MSC_VER)
		#define thread_local __declspec(thread)
	#elif defined(__GNUC__) || defined(__clang__)
		#define thread_local __thread
	#else
		#error "thread_local not supported on this compiler"
	#endif
#endif


// Thread types
typedef pthread_t        thrd_t;
typedef pthread_mutex_t  mtx_t;
typedef int (*thrd_start_t)(void*);

// Thread return values
enum {
	thrd_success = 0,
	thrd_error   = 1,
	thrd_nomem   = 2,
	thrd_timedout = 3,
	thrd_busy    = 4
};

// Mutex types
enum {
	mtx_plain     = 0,
	mtx_timed     = 1,
	mtx_recursive = 2
};

// Thread functions
typedef struct {
	thrd_start_t func;
	void*        arg;
} _thrd_wrapper_args_t;

static void* _thrd_wrapper(void* arg) {
	_thrd_wrapper_args_t wrapper = *(_thrd_wrapper_args_t*)arg;
	free(arg);
	int result = wrapper.func(wrapper.arg);
	return (void*)(intptr_t)result;
}

static inline int thrd_create(thrd_t* thr, thrd_start_t func, void* arg) {
	_thrd_wrapper_args_t* wrapper = (_thrd_wrapper_args_t*)malloc(sizeof(_thrd_wrapper_args_t));
	if (!wrapper) return thrd_nomem;
	wrapper->func = func;
	wrapper->arg  = arg;
	int result = pthread_create(thr, NULL, _thrd_wrapper, wrapper);
	if (result != 0) {
		free(wrapper);
		return thrd_error;
	}
	return thrd_success;
}

static inline int thrd_join(thrd_t thr, int* res) {
	void* thread_result;
	int   result = pthread_join(thr, &thread_result);
	if (res) *res = (int)(intptr_t)thread_result;
	return (result == 0) ? thrd_success : thrd_error;
}

static inline thrd_t thrd_current(void) {
	return pthread_self();
}

static inline int thrd_equal(thrd_t a, thrd_t b) {
	return pthread_equal(a, b);
}

// Mutex functions
static inline int mtx_init(mtx_t* mtx, int type) {
	if (type & mtx_recursive) {
		pthread_mutexattr_t attr;
		if (pthread_mutexattr_init   (&attr)                          != 0) return thrd_error;
		if (pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) != 0) {
			pthread_mutexattr_destroy(&attr);
			return thrd_error;
		}
		int result = pthread_mutex_init(mtx, &attr);
		pthread_mutexattr_destroy(&attr);
		return (result == 0) ? thrd_success : thrd_error;
	}
	int result = pthread_mutex_init(mtx, NULL);
	return (result == 0) ? thrd_success : thrd_error;
}

static inline void mtx_destroy(mtx_t* mtx) {
	pthread_mutex_destroy(mtx);
}

static inline int mtx_lock(mtx_t* mtx) {
	int result = pthread_mutex_lock(mtx);
	return (result == 0) ? thrd_success : thrd_error;
}

static inline int mtx_trylock(mtx_t* mtx) {
	int result = pthread_mutex_trylock(mtx);
	if (result == 0    ) return thrd_success;
	if (result == EBUSY) return thrd_busy;
	return thrd_error;
}

static inline int mtx_unlock(mtx_t* mtx) {
	int result = pthread_mutex_unlock(mtx);
	return (result == 0) ? thrd_success : thrd_error;
}

static inline void thrd_yield(void) {
	sched_yield();
}

// Sleep function (C11 thrd_sleep compatibility)
static inline int thrd_sleep(const struct timespec* duration, struct timespec* remaining) {
	int result = nanosleep(duration, remaining);
	if (result == 0) return 0;  // Success
	if (errno == EINTR) return -1;  // Interrupted
	return -2;  // Error
}