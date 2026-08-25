// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

// Optional SVSL (SPIR-V Shading Language) backend for skshaderc. Compiled in
// only when SKSHADERC_ENABLE_SVSL is set, which links libsvsl and defines
// SKSC_HAS_SVSL. SVSL emits the very same SKS container skshaderc writes —
// version in lockstep via the libsvsl pin, including native WGSL stages for
// the '-t w' target (no Tint on this path) — so this compiles the source, then
// loads SVSL's container bytes back through sksc_shader_file_load_memory() to
// hand the rest of the pipeline (info dump, header/skcs emit, sksc_build_file)
// an ordinary sksc_shader_file_t.

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

static svsl_include_src_t _svsl_include(void *user, const char *path, const char *requester) {
	svsl_include_ctx_t *ctx = (svsl_include_ctx_t*)user;
	char                full[SKSC_PATH_MAX];

	if (!sksc_include_resolve(path, requester, ctx->settings, full, sizeof(full)))
		return svsl_include_src_t{};

	int32_t len;
	char   *content = sksc_file_read(full, &len);
	if (!content) return svsl_include_src_t{};

	return svsl_include_src_t{ (const char*)_track(ctx, content), len, (const char*)_track(ctx, _dup_str(full)) };
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

	// One language per container: the target is fixed before preprocessing so
	// TARGET_SPIRV/TARGET_WGSL can vary the source, and a container carries a
	// single reflection table.
	if (settings->target_langs[skr_shader_lang_spirv] && settings->target_langs[skr_shader_lang_wgsl]) {
		sksc_log(sksc_log_level_err, "'-t sw' isn't supported: a .sks carries one language. Compile '%s' twice, with '-t s' and '-t w'", filename);
		return false;
	}
	// SPIR-V always compiles internally for the reflection metadata, so '-t w'
	// still validates and reflects fully. WGSL emission is native (no Tint here),
	// and stages browser WebGPU can't express are skipped with a located warning.
	if (settings->target_langs[skr_shader_lang_spirv]) options.targets |= svsl_target_spirv;
	if (settings->target_langs[skr_shader_lang_wgsl]) {
		if (svsl_supports_wgsl()) options.targets |= svsl_target_wgsl;
		else sksc_log(sksc_log_level_warn, "The WGSL target ('-t w') needs an skshaderc whose embedded libsvsl was built with SVSL_ENABLE_WGSL; emitting SPIR-V only");
	}
	if (options.targets == 0) {
		sksc_log(sksc_log_level_err, "No emittable target language for '%s' (the SVSL backend can't produce the requested set)", filename);
		return false;
	}

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

	// SVSL emits an SKS container byte-identical in layout to skshaderc's own
	// output (version kept in lockstep via the libsvsl pin), so load it back
	// into the shared sksc_shader_file_t and let the normal pipeline take it
	// from here.
	svsl_bytes_t sks = svsl_result_sks(result);
	if (!sks.data || sks.size <= 0) {
		sksc_log(sksc_log_level_err, "SVSL produced no container output");
		svsl_result_free(result);
		return false;
	}

	// The container already carries every requested language: SPIR-V and/or
	// native WGSL stages plus the v12 standalone-sampler records (split
	// samplers at s+400, statically paired) — nothing to transpile or reflect.
	sksc_result_ load = sksc_shader_file_load_memory(sks.data, (uint32_t)sks.size, out_file);
	svsl_result_free(result);

	if (load != sksc_result_success) {
		sksc_log(sksc_log_level_err, "Failed to read SVSL container output (%d)", (int)load);
		return false;
	}
	return true;
}
