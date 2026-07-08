// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

// Optional SVSL (SPIR-V Shading Language) backend for skshaderc. Compiled in
// only when SKSHADERC_ENABLE_SVSL is set, which links libsvsl and defines
// SKSC_HAS_SVSL. SVSL emits the very same SKS v9 container skshaderc writes, so
// this path compiles the source, then loads SVSL's container bytes back through
// sksc_shader_file_load_memory() to hand the rest of the pipeline (info dump,
// header/skcs emit, sksc_build_file) an ordinary sksc_shader_file_t.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "sksc.h"
#include "_sksc.h"
#include "array.h"

#include <svsl/svsl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////

// The include callback runs during svsl_compile, which copies whatever it
// keeps; we hold each buffer we hand out until the compile returns, then free
// them all in one sweep.
struct svsl_include_ctx_t {
	const sksc_settings_t *settings;
	array_t<void*>         allocs;
};

///////////////////////////////////////////

static const char *_last_sep(const char *path) {
	const char *sep  = strrchr(path, '/');
	const char *back = strrchr(path, '\\');
	if (back && (!sep || back > sep)) sep = back;
	return sep;
}

///////////////////////////////////////////

static char *_read_file(const char *path, int32_t *out_len) {
	FILE *fp = fopen(path, "rb");
	if (!fp) return nullptr;
	fseek(fp, 0, SEEK_END);
	long size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	// ftell returns -1 for an unseekable stream, and on POSIX fopen("rb")
	// succeeds on a directory (Windows fails it above), so reject a bad size
	// before it wraps to a huge malloc/fread.
	if (size < 0) { fclose(fp); return nullptr; }
	char *data = (char*)malloc((size_t)size + 1);
	if (!data) { fclose(fp); return nullptr; }
	size_t got = fread(data, 1, (size_t)size, fp);
	fclose(fp);
	data[got] = '\0';
	if (out_len) *out_len = (int32_t)got;
	return data;
}

///////////////////////////////////////////

static void *_track(svsl_include_ctx_t *ctx, void *p) {
	if (p) ctx->allocs.add(p);
	return p;
}

///////////////////////////////////////////

static char *_dup_str(const char *s) {
	size_t n = strlen(s) + 1;
	char  *c = (char*)malloc(n);
	if (c) memcpy(c, s, n);
	return c;
}

///////////////////////////////////////////

// Resolve an #include the way skshaderc does: the requesting file's own folder
// first, then each -i include folder in order.
static svsl_include_src_t _svsl_include(void *user, const char *path, const char *requester) {
	svsl_include_ctx_t *ctx = (svsl_include_ctx_t*)user;
	char                full[2048];

	const char *slash = requester ? _last_sep(requester) : nullptr;
	if (slash) {
		snprintf(full, sizeof(full), "%.*s/%s", (int32_t)(slash - requester), requester, path);
		int32_t len;
		char   *content = _read_file(full, &len);
		if (content) return svsl_include_src_t{ (const char*)_track(ctx, content), len, (const char*)_track(ctx, _dup_str(full)) };
	}
	for (int32_t i = 0; i < ctx->settings->include_folder_ct; i++) {
		snprintf(full, sizeof(full), "%s/%s", ctx->settings->include_folders[i], path);
		int32_t len;
		char   *content = _read_file(full, &len);
		if (content) return svsl_include_src_t{ (const char*)_track(ctx, content), len, (const char*)_track(ctx, _dup_str(full)) };
	}
	return svsl_include_src_t{};
}

///////////////////////////////////////////

bool sksc_svsl_compile(const char *filename, const char *hlsl_text, const sksc_settings_t *settings, sksc_shader_file_t *out_file) {
	// skshaderc's optimize scale (0 none, 1 size, 2/3 perf) onto SVSL's three
	// levels; a debug build compiles unoptimized so the SPIR-V stays readable.
	svsl_opt_level_ opt_level;
	if      (settings->debug)         opt_level = svsl_opt_none;
	else if (settings->optimize <= 0) opt_level = svsl_opt_none;
	else if (settings->optimize == 1) opt_level = svsl_opt_default;
	else                              opt_level = svsl_opt_aggressive;

	// An empty entry name means the stage is disabled; SVSL then finds no
	// matching function and simply emits no stage (unless the source tags one
	// with an explicit [vertex]/[fragment]/[compute] attribute).
	svsl_include_ctx_t inc_ctx = { settings, {} };
	svsl_options_t     options = {};
	options.include_cb    = _svsl_include;
	options.include_user  = &inc_ctx;
	options.entry_vertex  = settings->vs_entrypoint[0] ? settings->vs_entrypoint : "";
	options.entry_pixel   = settings->ps_entrypoint[0] ? settings->ps_entrypoint : "";
	options.entry_compute = settings->cs_entrypoint[0] ? settings->cs_entrypoint : "";
	options.opt_level     = opt_level;

	svsl_source_t source = {};
	source.text     = hlsl_text;
	source.length   = -1;
	source.filename = filename;

	svsl_result_t *result = svsl_compile(&source, &options);
	inc_ctx.allocs.each([](void *&p) { free(p); });
	inc_ctx.allocs.free();

	if (!result) {
		sksc_log(sksc_log_level_err, "SVSL out of memory");
		return false;
	}

	// Surface SVSL's diagnostics through skshaderc's log so callers see line and
	// column information exactly as they do for the glslang pipeline.
	for (int32_t i = 0; i < result->diagnostic_count; i++) {
		const svsl_diag_t *d     = &result->diagnostics[i];
		sksc_log_level_    level = sksc_log_level_info;
		switch (d->severity) {
		case svsl_severity_error:   level = sksc_log_level_err;  break;
		case svsl_severity_warning: level = sksc_log_level_warn; break;
		case svsl_severity_porting: level = sksc_log_level_info; break;
		case svsl_severity_info:    level = sksc_log_level_info; break;
		}
		sksc_log_at(level, d->loc.line, d->loc.col, "%s", d->message);
	}

	if (!result->ok) {
		sksc_log(sksc_log_level_err, "SVSL compile failed");
		svsl_result_free(result);
		return false;
	}

	// SVSL emits an SKS v9 container byte-identical in layout to skshaderc's own
	// output, so load it back into the shared sksc_shader_file_t and let the
	// normal pipeline take it from here.
	svsl_bytes_t sks = svsl_result_sks(result);
	if (!sks.data || sks.size <= 0) {
		sksc_log(sksc_log_level_err, "SVSL produced no container output");
		svsl_result_free(result);
		return false;
	}

	sksc_result_ load = sksc_shader_file_load_memory(sks.data, (uint32_t)sks.size, out_file);
	svsl_result_free(result);

	if (load != sksc_result_success) {
		sksc_log(sksc_log_level_err, "Failed to read SVSL container output (%d)", (int)load);
		return false;
	}
	return true;
}
