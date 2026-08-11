// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "_skr_shared.h"

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif

///////////////////////////////////////////////////////////////////////////////

static void _default_log(skr_log_ level, const char *text) {
#ifdef __ANDROID__
	android_LogPriority priority = 
		level == skr_log_info     ? ANDROID_LOG_INFO  :
		level == skr_log_warning  ? ANDROID_LOG_WARN  :
		level == skr_log_critical ? ANDROID_LOG_ERROR : ANDROID_LOG_UNKNOWN;
	__android_log_write(priority, "sk_renderer", text);
#else
	const char *prefix = 
		level == skr_log_info     ? "[skr:info] " :
		level == skr_log_warning  ? "[skr:warn] " :
		level == skr_log_critical ? "[skr:crit] " : "[skr:unkn] ";
	
	if (level == skr_log_critical) {
		// breakpoint anchor for critical logs
	}
	printf("%s%s\n", prefix, text);
#endif
}
void (*_skr_log)(skr_log_ level, const char *text) = _default_log;
bool _skr_log_disabled = false;

///////////////////////////////////////////////////////////////////////////////

void skr_callback_log(void (*callback)(skr_log_ level, const char *text)) {
	_skr_log = callback;
}

///////////////////////////////////////////////////////////////////////////////

void skr_log(skr_log_ level, const char *text, ...) {
	if (_skr_log_disabled) return;
	if (!_skr_log) return;

	va_list args, copy;
	va_start(args, text);
	va_copy (copy, args);
	int32_t length = vsnprintf(NULL, 0, text, args) + 1;
	char*  buffer = (char*)_skr_malloc(sizeof(char) * length);
	vsnprintf(buffer, length, text, copy);

	_skr_log(level, buffer);

	_skr_free(buffer);
	va_end(args);
	va_end(copy);
}
