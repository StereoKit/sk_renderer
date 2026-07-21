#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#define NOMINMAX
///////////////////////////////////////////

#include "sksc.h"
#include "_sksc.h"

#include "array.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

///////////////////////////////////////////

void sksc_log_shader_info(const sksc_shader_file_t *file);

///////////////////////////////////////////

void sksc_init() {
#ifdef SKSC_HAS_GLSLANG
	sksc_glslang_init();
#endif
}

///////////////////////////////////////////

void sksc_shutdown() {
#ifdef SKSC_HAS_GLSLANG
	sksc_glslang_shutdown();
#endif
}

///////////////////////////////////////////

#ifdef SKSC_HAS_GLSLANG
// Repack all dynamically-allocated meta sub-arrays into a single contiguous
// block. After this, sksc_shader_meta_free() only needs free(meta->buffers).
// Only the glslang path builds meta piecewise; SVSL's container arrives already
// packed via sksc_shader_file_load_memory.
static void _sksc_meta_pack(sksc_shader_meta_t *meta) {
	uint32_t total_var_count     = 0;
	uint32_t total_defaults_size = 0;
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		total_var_count += meta->buffers[i].var_count;
		if (meta->buffers[i].defaults)
			total_defaults_size += meta->buffers[i].size;
	}

	size_t buffers_size = sizeof(sksc_shader_buffer_t       ) * meta->buffer_count;
	size_t res_size     = sizeof(sksc_shader_resource_t     ) * meta->resource_count;
	size_t vars_size    = sizeof(sksc_shader_var_t          ) * total_var_count;
	size_t vinputs_size = sizeof(skr_vert_component_t       ) * meta->vertex_input_count;
	size_t spec_size    = sizeof(sksc_shader_spec_constant_t) * meta->spec_constant_count;
	size_t sampler_size = sizeof(sksc_shader_sampler_t      ) * meta->sampler_count;
	size_t total_size   = buffers_size + res_size + vars_size + total_defaults_size + vinputs_size + spec_size + sampler_size;

	if (total_size == 0) return;

	uint8_t *block = (uint8_t *)malloc(total_size);
	memset(block, 0, total_size);

	// Copy top-level arrays
	sksc_shader_buffer_t        *new_buffers = (sksc_shader_buffer_t       *)(block);
	sksc_shader_resource_t      *new_res     = (sksc_shader_resource_t     *)(block + buffers_size);
	uint8_t                     *vars_cursor = block + buffers_size + res_size;
	uint8_t                     *def_cursor  = block + buffers_size + res_size + vars_size;
	skr_vert_component_t        *new_vinputs = (skr_vert_component_t       *)(block + buffers_size + res_size + vars_size + total_defaults_size);
	sksc_shader_spec_constant_t *new_specs   = (sksc_shader_spec_constant_t*)(block + buffers_size + res_size + vars_size + total_defaults_size + vinputs_size);
	sksc_shader_sampler_t       *new_samps   = (sksc_shader_sampler_t      *)(block + buffers_size + res_size + vars_size + total_defaults_size + vinputs_size + spec_size);

	if (buffers_size > 0) memcpy(new_buffers, meta->buffers,        buffers_size);
	if (res_size     > 0) memcpy(new_res,     meta->resources,      res_size);
	if (vinputs_size > 0) memcpy(new_vinputs, meta->vertex_inputs,  vinputs_size);
	if (spec_size    > 0) memcpy(new_specs,   meta->spec_constants, spec_size);
	if (sampler_size > 0) memcpy(new_samps,   meta->samplers,       sampler_size);

	// Copy per-buffer vars and defaults, update pointers
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		size_t vsize = sizeof(sksc_shader_var_t) * new_buffers[i].var_count;
		if (vsize > 0) {
			memcpy(vars_cursor, meta->buffers[i].vars, vsize);
			new_buffers[i].vars = (sksc_shader_var_t *)vars_cursor;
			vars_cursor += vsize;
		} else {
			new_buffers[i].vars = NULL;
		}
		if (meta->buffers[i].defaults) {
			memcpy(def_cursor, meta->buffers[i].defaults, meta->buffers[i].size);
			new_buffers[i].defaults = def_cursor;
			def_cursor += meta->buffers[i].size;
		}
	}

	// Free old individual allocations
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		free(meta->buffers[i].vars);
		free(meta->buffers[i].defaults);
	}
	free(meta->buffers);
	free(meta->resources);
	free(meta->vertex_inputs);
	free(meta->spec_constants);
	free(meta->samplers);

	// Point meta at the packed block
	meta->buffers        = new_buffers;
	meta->resources      = new_res;
	meta->vertex_inputs  = new_vinputs;
	meta->spec_constants = new_specs;
	meta->samplers       = new_samps;
	meta->samplers_owned = false;
}
#endif // SKSC_HAS_GLSLANG

// Case-insensitive check for a trailing ".svsl" on the source path.
static bool _sksc_has_svsl_ext(const char *filename) {
	if (!filename) return false;
	size_t len = strlen(filename);
	const char *ext = ".svsl";
	size_t elen = strlen(ext);
	if (len < elen) return false;
	const char *tail = filename + (len - elen);
	for (size_t i = 0; i < elen; i++)
		if (tolower((unsigned char)tail[i]) != ext[i]) return false;
	return true;
}

bool sksc_compile(const char *filename, const char *hlsl_text, sksc_settings_t *settings, sksc_shader_file_t *out_file) {
	*out_file = {};
	 out_file->meta = {};
	 out_file->meta.global_buffer_id = -1;

	// The SVSL backend is used when explicitly requested (-svsl) or when the
	// source file carries a .svsl extension.
	bool want_svsl = settings->use_svsl || _sksc_has_svsl_ext(filename);
#if defined(SKSC_HAS_SVSL) && !defined(SKSC_HAS_GLSLANG)
	// No glslang backend in this build, so SVSL handles everything.
	want_svsl = true;
#endif
	if (want_svsl) {
#ifdef SKSC_HAS_SVSL
		bool ok = sksc_svsl_compile(filename, hlsl_text, settings, out_file);
		if (ok && !settings->silent_info)
			sksc_log_shader_info(out_file);
		return ok;
#else
		sksc_log(sksc_log_level_err, "SVSL backend requested, but skshaderc was built without it (SKSHADERC_ENABLE_SVSL=OFF)");
		return false;
#endif
	}

#ifdef SKSC_HAS_GLSLANG
	array_t<sksc_shader_file_stage_t> stages       = {};
	array_t<sksc_meta_item_t>         var_meta     = sksc_meta_find_defaults(hlsl_text);
	array_t<sksc_ast_default_t>       ast_defaults = sksc_hlsl_find_initializers(hlsl_text);

	skr_stage_ compile_stages[3] = { skr_stage_vertex, skr_stage_pixel, skr_stage_compute };
	char*      entrypoints   [3] = { settings->vs_entrypoint, settings->ps_entrypoint, settings->cs_entrypoint };
	uint64_t   stage_features[3] = {};
	bool       stage_present [3] = {};

	for (size_t i = 0; i < sizeof(compile_stages)/sizeof(compile_stages[0]); i++) {
		if (entrypoints[i][0] == 0)
			continue;

		// Build SPIRV
		sksc_shader_file_stage_t spirv_stage  = {};
		compile_result_          spirv_result = sksc_hlsl_to_spirv(filename, hlsl_text, settings, compile_stages[i], NULL, 0, false, &spirv_stage);
		if (spirv_result == compile_result_fail) {
			sksc_log(sksc_log_level_err, "SPIRV compile failed");
			return false;
		} else if (spirv_result == compile_result_skip)
			continue;

		// Extract metadata from the SPIRV
		sksc_spirv_to_meta(&spirv_stage, &out_file->meta);
		stage_present [i] = true;
		stage_features[i] = spirv_stage.code_size >= 20
			? sksc_spirv_features((const uint32_t*)spirv_stage.code, spirv_stage.code_size / 4)
			: 0;

		// Add it as a stage in our sks file
		if (settings->target_langs[skr_shader_lang_spirv]) {
			stages.add(spirv_stage);
		}

		if (!settings->target_langs[skr_shader_lang_spirv])
			free(spirv_stage.code);
	}

	// WGSL stages for the WebGPU backend. Shaders using features browser
	// WebGPU has no counterpart for get no WGSL blob, with a warning —
	// multiview is NOT in that set, it lowers to the sk_view_index spec
	// constant (see sksc_wgsl.cpp).
	if (settings->target_langs[skr_shader_lang_wgsl]) {
		const uint64_t unsupported_mask = sksc_wgsl_unsupported_features();

		for (size_t i = 0; i < sizeof(compile_stages)/sizeof(compile_stages[0]); i++) {
			if (!stage_present[i])
				continue;
			if (stage_features[i] & unsupported_mask) {
				sksc_log(sksc_log_level_warn, "Skipping WGSL output for a stage: it uses features unavailable in browser WebGPU (subgroups/wave size/tile or QCOM image ops)");
				continue;
			}
			// SubpassInput shaders pass through _wgsl_subpass_to_texture, which
			// lowers them to plain textures fetched at the fragment coordinate

			sksc_shader_file_stage_t wgsl_stage  = {};
			compile_result_          wgsl_result = sksc_wgsl_compile_stage(filename, hlsl_text, settings, compile_stages[i], &out_file->meta, &wgsl_stage);
			if (wgsl_result == compile_result_fail) {
				sksc_log(sksc_log_level_err, "WGSL compile failed");
				return false;
			}
			if (wgsl_result == compile_result_success)
				stages.add(wgsl_stage);
		}
	}

	sksc_meta_assign_defaults(ast_defaults, var_meta, &out_file->meta);
	var_meta.free();
	ast_defaults.free();
	out_file->stage_count = (uint32_t)stages.count;
	out_file->stages      = stages.data;

	if (!settings->silent_info) {
		sksc_log_shader_info(out_file);
	}

	if (!sksc_meta_check_dup_buffers(&out_file->meta)) {
		sksc_log(sksc_log_level_err, "Found constant buffers re-using slot ids");
		return false;
	}

	const char *dup_name1, *dup_name2;
	uint32_t    dup_slot;
	if (!sksc_meta_check_dup_resources(&out_file->meta, &dup_name1, &dup_name2, &dup_slot)) {
		sksc_log(sksc_log_level_err, "Resources '%s' and '%s' are both bound to the same slot (t%u)", dup_name1, dup_name2, dup_slot);
		return false;
	}

	_sksc_meta_pack(&out_file->meta);
	return true;
#else
	(void)hlsl_text;
	sksc_log(sksc_log_level_err, "Compiling '%s' needs the glslang backend, but skshaderc was built without it (SKSHADERC_ENABLE_GLSLANG=OFF). Use the SVSL backend (-svsl, or a .svsl source file).", filename);
	return false;
#endif
}

///////////////////////////////////////////

// Helper for building info string
struct info_builder_t {
	char  *str;
	size_t len;
	size_t cap;

	void append(const char *fmt, ...) {
		va_list args, args_copy;
		va_start(args, fmt);
		va_copy(args_copy, args);

		int needed = vsnprintf(nullptr, 0, fmt, args);
		va_end(args);

		if (needed < 0) { va_end(args_copy); return; }

		size_t new_len = len + needed + 1; // +1 for newline
		if (new_len + 1 > cap) {
			cap = cap == 0 ? 1024 : cap * 2;
			while (new_len + 1 > cap) cap *= 2;
			str = (char*)realloc(str, cap);
		}

		vsnprintf(str + len, cap - len, fmt, args_copy);
		va_end(args_copy);
		len += needed;
		str[len++] = '\n';
		str[len] = '\0';
	}
};

///////////////////////////////////////////

char* sksc_shader_file_info(const sksc_shader_file_t *file) {
	if (!file) return nullptr;

	const sksc_shader_meta_t *meta = &file->meta;
	info_builder_t info = {};

	info.append(" ________________");

	// A quick summary of performance
	info.append("|--Performance--");
	if (meta->ops_vertex.total > 0 || meta->ops_pixel.total > 0)
		info.append("| Instructions |  all | tex | flow |");
	if (meta->ops_vertex.total > 0) {
		info.append("|       Vertex | %4d | %3d | %4d |",
			meta->ops_vertex.total,
			meta->ops_vertex.tex_read,
			meta->ops_vertex.dynamic_flow);
	}
	if (meta->ops_pixel.total > 0) {
		info.append("|        Pixel | %4d | %3d | %4d |",
			meta->ops_pixel.total,
			meta->ops_pixel.tex_read,
			meta->ops_pixel.dynamic_flow);
	}

	// List of all the buffers
	info.append("|--Buffer Info--");
	for (size_t i = 0; i < meta->buffer_count; i++) {
		sksc_shader_buffer_t *buff = &meta->buffers[i];
		info.append("|  %s - %u bytes%s", buff->name, buff->size, buff->defaults ? " (has defaults)" : "");
		for (size_t v = 0; v < buff->var_count; v++) {
			sksc_shader_var_t *var = &buff->vars[v];
			const char *type_str = var->type_name[0] ? var->type_name : "unknown";

			// Compute element size from type_name to get actual array dimension
			uint32_t element_size = var->type_count;
			if      (strcmp(type_str, "float4x4") == 0 || strcmp(type_str, "int4x4") == 0 || strcmp(type_str, "uint4x4") == 0) element_size = 16;
			else if (strcmp(type_str, "float3x3") == 0 || strcmp(type_str, "int3x3") == 0 || strcmp(type_str, "uint3x3") == 0) element_size = 9;
			else if (strcmp(type_str, "float4")   == 0 || strcmp(type_str, "int4")   == 0 || strcmp(type_str, "uint4")   == 0) element_size = 4;
			else if (strcmp(type_str, "float3")   == 0 || strcmp(type_str, "int3")   == 0 || strcmp(type_str, "uint3")   == 0) element_size = 3;
			else if (strcmp(type_str, "float2")   == 0 || strcmp(type_str, "int2")   == 0 || strcmp(type_str, "uint2")   == 0) element_size = 2;
			else if (strcmp(type_str, "float")    == 0 || strcmp(type_str, "int")    == 0 || strcmp(type_str, "uint")    == 0) element_size = 1;
			else if (strcmp(type_str, "double")   == 0 || strcmp(type_str, "bool")   == 0) element_size = 1;

			uint32_t array_dim = element_size > 0 ? var->type_count / element_size : 1;
			if (array_dim == 0) array_dim = 1;

			// Show default value if present
			char default_str[256] = "";
			if (buff->defaults != nullptr) {
				uint8_t *def_ptr = ((uint8_t *)buff->defaults) + var->offset;
				int32_t written = 0;
				written += snprintf(default_str + written, sizeof(default_str) - written, " = ");
				for (uint32_t c = 0; c < var->type_count; c++) {
					if (written >= (int32_t)sizeof(default_str) - 16) {
						written += snprintf(default_str + written, sizeof(default_str) - written, "...");
						break;
					}
					if (c > 0) written += snprintf(default_str + written, sizeof(default_str) - written, ", ");
					switch (var->type) {
					case sksc_shader_var_float:  written += snprintf(default_str + written, sizeof(default_str) - written, "%.3g", ((float*)def_ptr)[c]);   break;
					case sksc_shader_var_double: written += snprintf(default_str + written, sizeof(default_str) - written, "%.3g", ((double*)def_ptr)[c]);  break;
					case sksc_shader_var_int:    written += snprintf(default_str + written, sizeof(default_str) - written, "%d",   ((int32_t*)def_ptr)[c]); break;
					case sksc_shader_var_uint:   written += snprintf(default_str + written, sizeof(default_str) - written, "%u",   ((uint32_t*)def_ptr)[c]); break;
					case sksc_shader_var_uint8:  written += snprintf(default_str + written, sizeof(default_str) - written, "%u",   def_ptr[c]);             break;
					default: break;
					}
				}
			}
			if (array_dim > 1) {
				info.append("|    %-15s: +%-4u %5ub - %s[%u]%s", var->name, var->offset, var->size, type_str, array_dim, default_str);
			} else {
				info.append("|    %-15s: +%-4u %5ub - %s%s", var->name, var->offset, var->size, type_str, default_str);
			}
		}
	}

	// List specialization constants
	if (meta->spec_constant_count > 0) {
		info.append("|--Spec Constants--");
		for (uint32_t i = 0; i < meta->spec_constant_count; i++) {
			sksc_shader_spec_constant_t *spec = &meta->spec_constants[i];
			switch (spec->type) {
			case sksc_shader_var_float: info.append("|  [%u] %-15s: float = %.3g", spec->constant_id, spec->name, *(float   *)&spec->default_value); break;
			case sksc_shader_var_uint:  info.append("|  [%u] %-15s: uint  = %u",   spec->constant_id, spec->name, spec->default_value);              break;
			default:                    info.append("|  [%u] %-15s: int   = %d",   spec->constant_id, spec->name, *(int32_t *)&spec->default_value); break;
			}
		}
	}

	// Show the vertex shader's input format
	if (meta->vertex_input_count > 0) {
		info.append("|--Mesh Input--");
		for (int32_t i = 0; i < meta->vertex_input_count; i++) {
			const char *format;
			const char *semantic;
			switch (meta->vertex_inputs[i].format) {
				case skr_vertex_fmt_f32:  format = "float"; break;
				case skr_vertex_fmt_i32:  format = "int  "; break;
				case skr_vertex_fmt_ui32: format = "uint "; break;
				default: format = "NA"; break;
			}
			switch (meta->vertex_inputs[i].semantic) {
				case skr_semantic_binormal:     semantic = "BiNormal";     break;
				case skr_semantic_blendindices: semantic = "BlendIndices"; break;
				case skr_semantic_blendweight:  semantic = "BlendWeight";  break;
				case skr_semantic_color:        semantic = "Color";        break;
				case skr_semantic_normal:       semantic = "Normal";       break;
				case skr_semantic_position:     semantic = "Position";     break;
				case skr_semantic_psize:        semantic = "PSize";        break;
				case skr_semantic_tangent:      semantic = "Tangent";      break;
				case skr_semantic_texcoord:     semantic = "TexCoord";     break;
				default:                        semantic = "NA";           break;
			}
			info.append("|  loc %d : %s%d : %s%d", meta->vertex_inputs[i].location, format, meta->vertex_inputs[i].count, semantic, meta->vertex_inputs[i].semantic_slot);
		}
	}

	for (uint32_t s = 0; s < file->stage_count; s++) {
		const sksc_shader_file_stage_t* stage = &file->stages[s];

		const char *stage_name = "";
		switch (stage->stage) {
		case skr_stage_vertex:  stage_name = "Vertex";  break;
		case skr_stage_pixel:   stage_name = "Pixel";   break;
		case skr_stage_compute: stage_name = "Compute"; break;
		}
		info.append("|--%s Shader--", stage_name);
		for (uint32_t i = 0; i < meta->buffer_count; i++) {
			sksc_shader_buffer_t *buff = &meta->buffers[i];
			if (buff->bind.stage_bits & stage->stage) {
				char reg[16];
				snprintf(reg, sizeof(reg), "b%u/s%d", buff->bind.slot, buff->space);
				info.append("|  %-7s: %s", reg, buff->name);
			}
		}
		for (uint32_t i = 0; i < meta->resource_count; i++) {
			sksc_shader_resource_t *tex = &meta->resources[i];
			if (tex->bind.stage_bits & stage->stage) {
				bool is_storage_buffer = tex->bind.register_type == skr_register_read_buffer || tex->bind.register_type == skr_register_readwrite;
				char reg_char          = (tex->bind.register_type == skr_register_texture || tex->bind.register_type == skr_register_read_buffer) ? 't' : 'u';
				char reg[16];
				snprintf(reg, sizeof(reg), "%c%u", reg_char, tex->bind.slot);
				if (is_storage_buffer && tex->element_size > 0) {
					info.append("|  %-7s: %-17s %3ub/elem", reg, tex->name, tex->element_size);
				} else {
					info.append("|  %-7s: %s", reg, tex->name);
				}
			}
		}
	}
	info.append("|________________");

	return info.str;
}

///////////////////////////////////////////

void sksc_log_shader_info(const sksc_shader_file_t *file) {
	char *info = sksc_shader_file_info(file);
	if (!info) return;

	// Log each line separately
	char *line = info;
	while (*line) {
		char *end = strchr(line, '\n');
		if (end) *end = '\0';
		sksc_log(sksc_log_level_info, "%s", line);
		if (end) line = end + 1;
		else break;
	}
	free(info);
}

///////////////////////////////////////////

struct file_data_t {
	array_t<uint8_t> data;

	void write_fixed_str(const char *item, int32_t _Size) {
		size_t len = strlen(item);
		data.add_range((uint8_t*)item, (int32_t)(sizeof(char) * len));

		int32_t count = (int32_t)(_Size - len);
		if (_Size - len > 0) {
			while (data.count + count > data.capacity) { data.resize(data.capacity * 2 < 4 ? 4 : data.capacity * 2); }
		}
		memset(&data.data[data.count], 0, count);
		data.count += count;
	}
	template <typename T> 
	void write(T &item) { data.add_range((uint8_t*)&item, sizeof(T)); }
	void write(void *item, size_t size) { data.add_range((uint8_t*)item, (int32_t)size); }
};

///////////////////////////////////////////

// Features browser WebGPU has no counterpart for — stages using any of these
// get no WGSL blob. Multiview is NOT in the set; it lowers to the
// sk_view_index spec constant (see sksc_wgsl.cpp).
uint64_t sksc_wgsl_unsupported_features(void) {
	return (1ull << sksc_feature_bit_subgroups)            |
	       (1ull << sksc_feature_bit_wave_size)            |
	       (1ull << sksc_feature_bit_tile_image)           |
	       (1ull << sksc_feature_bit_qcom_sample_weighted) |
	       (1ull << sksc_feature_bit_qcom_box_filter)      |
	       (1ull << sksc_feature_bit_qcom_block_match)     |
	       (1ull << sksc_feature_bit_qcom_image_proc2)     |
	       (1ull << sksc_feature_bit_qcom_tile_shading);
}

// Device-feature mask: scans a SPIR-V blob's OpCapability/OpExtension
// declarations (raw opcode numbers, like the op counting in sksc_meta.cpp)
// and maps them onto sksc_feature_bit_ values. Declarations with no assigned
// bit set sksc_feature_bit_unknown so runtimes never silently under-check.
// Non-static: sksc_compile also uses this to gate the WGSL target.
uint64_t sksc_spirv_features(const uint32_t *words, uint32_t word_count) {
	// capability value → feature bit; one bit can cover several capabilities
	static const struct { uint32_t cap; uint8_t bit; } cap_bits[] = {
		{ 9,    sksc_feature_bit_float16 },          // Float16
		{ 4433, sksc_feature_bit_storage16 },        // StorageBuffer16BitAccess
		{ 4434, sksc_feature_bit_storage16 },        // UniformAndStorageBuffer16BitAccess
		{ 4435, sksc_feature_bit_storage16 },        // StoragePushConstant16
		{ 4448, sksc_feature_bit_storage8 },         // StorageBuffer8BitAccess
		{ 4449, sksc_feature_bit_storage8 },         // UniformAndStorageBuffer8BitAccess
		{ 4450, sksc_feature_bit_storage8 },         // StoragePushConstant8
		{ 49,   sksc_feature_bit_extended_formats }, // StorageImageExtendedFormats
		{ 61,   sksc_feature_bit_subgroups },        // GroupNonUniform
		{ 62,   sksc_feature_bit_subgroups },        // ...Vote
		{ 63,   sksc_feature_bit_subgroups },        // ...Arithmetic
		{ 64,   sksc_feature_bit_subgroups },        // ...Ballot
		{ 65,   sksc_feature_bit_subgroups },        // ...Shuffle
		{ 66,   sksc_feature_bit_subgroups },        // ...ShuffleRelative
		{ 67,   sksc_feature_bit_subgroups },        // ...Clustered
		{ 68,   sksc_feature_bit_subgroups },        // ...Quad
		{ 4439, sksc_feature_bit_multiview },        // MultiView
		{ 5379, sksc_feature_bit_demote },           // DemoteToHelperInvocationEXT
		{ 11,   sksc_feature_bit_int64 },            // Int64
		{ 10,   sksc_feature_bit_float64 },          // Float64
		{ 22,   sksc_feature_bit_int16 },            // Int16
		{ 39,   sksc_feature_bit_int8 },             // Int8
		{ 55,   sksc_feature_bit_formatless },       // StorageImageReadWithoutFormat
		{ 56,   sksc_feature_bit_formatless },       // StorageImageWriteWithoutFormat
		{ 4166, sksc_feature_bit_tile_image },       // TileImageColorReadAccessEXT
		{ 4167, sksc_feature_bit_tile_image },       // TileImageDepthReadAccessEXT
		{ 4168, sksc_feature_bit_tile_image },       // TileImageStencilReadAccessEXT
		{ 6033, sksc_feature_bit_float_atomics },    // AtomicFloat32AddEXT
		{ 5612, sksc_feature_bit_float_atomics },    // AtomicFloat32MinMaxEXT
		{ 4484, sksc_feature_bit_qcom_sample_weighted }, // TextureSampleWeightedQCOM
		{ 4485, sksc_feature_bit_qcom_box_filter },      // TextureBoxFilterQCOM
		{ 4486, sksc_feature_bit_qcom_block_match },     // TextureBlockMatchQCOM
		{ 4498, sksc_feature_bit_qcom_image_proc2 },     // TextureBlockMatch2QCOM
		{ 4495, sksc_feature_bit_qcom_tile_shading },    // TileShadingQCOM
	};
	// capabilities every Vulkan 1.1 runtime satisfies — no bit, never unknown
	static const uint32_t baseline[] = { 1, 50, 43, 44, 40, 51 };
	// Shader, ImageQuery, Sampled1D, Image1D, InputAttachment, DerivativeControl
	// bit 0xFF = the extension is known but sets no bit itself: its capabilities
	// carry the precise per-op bits (image processing splits into three)
	static const struct { const char *name; uint8_t bit; } ext_bits[] = {
		// Reflection-only decorations from glslang's HLSL front end; every
		// driver ignores them, so they gate nothing
		{ "SPV_GOOGLE_hlsl_functionality1",      0xFF },
		{ "SPV_GOOGLE_user_type",                0xFF },
		{ "SPV_GOOGLE_decorate_string",          0xFF },
		{ "SPV_KHR_8bit_storage",                sksc_feature_bit_storage8 },
		{ "SPV_EXT_demote_to_helper_invocation", sksc_feature_bit_demote },
		{ "SPV_EXT_shader_tile_image",           sksc_feature_bit_tile_image },
		{ "SPV_EXT_shader_atomic_float_add",     sksc_feature_bit_float_atomics },
		{ "SPV_EXT_shader_atomic_float_min_max", sksc_feature_bit_float_atomics },
		{ "SPV_QCOM_image_processing",           0xFF },
		{ "SPV_QCOM_image_processing2",          sksc_feature_bit_qcom_image_proc2 },
		{ "SPV_QCOM_tile_shading",               sksc_feature_bit_qcom_tile_shading },
	};

	uint64_t bits = 0;
	for (uint32_t i = 5; i < word_count; ) {
		uint32_t count  = words[i] >> 16;
		uint32_t opcode = words[i] & 0xFFFF;
		if (count == 0) break;

		if (opcode == 17) { // OpCapability
			uint32_t cap   = words[i + 1];
			bool     known = false;
			for (size_t k = 0; k < sizeof(cap_bits) / sizeof(cap_bits[0]); k++)
				if (cap_bits[k].cap == cap) { bits |= 1ull << cap_bits[k].bit; known = true; }
			for (size_t k = 0; k < sizeof(baseline) / sizeof(baseline[0]); k++)
				if (baseline[k] == cap) known = true;
			if (!known) bits |= 1ull << sksc_feature_bit_unknown;
		} else if (opcode == 10) { // OpExtension
			const char *name  = (const char *)&words[i + 1];
			bool        known = false;
			for (size_t k = 0; k < sizeof(ext_bits) / sizeof(ext_bits[0]); k++)
				if (strncmp(name, ext_bits[k].name, (size_t)(count - 1) * 4) == 0) {
					if (ext_bits[k].bit != 0xFF) bits |= 1ull << ext_bits[k].bit;
					known = true;
				}
			if (!known) bits |= 1ull << sksc_feature_bit_unknown;
		} else if (opcode == 60) { // OpImageTexelPointer: image atomic access
			bits |= 1ull << sksc_feature_bit_image_atomics;
		}
		i += count;
	}
	return bits;
}

///////////////////////////////////////////

void sksc_build_file(const sksc_shader_file_t *file, void **out_data, uint32_t *out_size) {
	file_data_t data = {};

	const char tag[8] = {'S','K','S','H','A','D','E','R'};
	uint16_t version = SKSC_FILE_VERSION;
	data.write(tag);
	data.write(version);

	uint64_t features = 0;
	for (uint32_t i = 0; i < file->stage_count; i++)
		if (file->stages[i].language == skr_shader_lang_spirv && file->stages[i].code_size >= 20)
			features |= sksc_spirv_features((const uint32_t *)file->stages[i].code,
			                                file->stages[i].code_size / 4);
	if (file->meta.wave_size > 0)
		features |= 1ull << sksc_feature_bit_wave_size;
	uint64_t features_reserved = 0;

	data.write(file->stage_count);
	data.write_fixed_str(file->meta.name, sizeof(file->meta.name));
	data.write(file->meta.buffer_count);
	data.write(file->meta.resource_count);
	data.write(file->meta.vertex_input_count);
	data.write(file->meta.spec_constant_count);
	data.write(file->meta.sampler_count); // v12
	data.write(features);
	data.write(features_reserved);

	data.write(file->meta.ops_vertex.total);
	data.write(file->meta.ops_vertex.tex_read);
	data.write(file->meta.ops_vertex.dynamic_flow);
	data.write(file->meta.ops_pixel.total);
	data.write(file->meta.ops_pixel.tex_read);
	data.write(file->meta.ops_pixel.dynamic_flow);
	data.write(file->meta.wave_size);
	data.write(file->meta.tile_apron[0]); // v11: //--apron (QCOM tile shading)
	data.write(file->meta.tile_apron[1]);

	for (uint32_t i = 0; i < file->meta.buffer_count; i++) {
		sksc_shader_buffer_t *buff = &file->meta.buffers[i];
		data.write_fixed_str(buff->name, sizeof(buff->name));
		data.write(buff->space);
		data.write(buff->bind);
		data.write(buff->size);
		data.write(buff->var_count);
		if (buff->defaults) {
			data.write(buff->size);
			data.write(buff->defaults, buff->size);
		} else {
			uint32_t zero = 0;
			data.write(zero);
		}

		for (uint32_t t = 0; t < buff->var_count; t++) {
			sksc_shader_var_t *var = &buff->vars[t];
			data.write_fixed_str(var->name,      sizeof(var->name));
			data.write_fixed_str(var->extra,     sizeof(var->extra));
			data.write_fixed_str(var->type_name, sizeof(var->type_name));
			data.write(var->offset);
			data.write(var->size);
			data.write(var->type);
			data.write(var->type_count);
		}
	}

	for (int32_t i = 0; i < file->meta.vertex_input_count; i++) {
		skr_vert_component_t *com = &file->meta.vertex_inputs[i];
		data.write(com->format);
		data.write(com->count);
		data.write(com->semantic);
		data.write(com->semantic_slot);
		data.write(com->location);
	}

	for (uint32_t i = 0; i < file->meta.resource_count; i++) {
		sksc_shader_resource_t *res = &file->meta.resources[i];
		data.write_fixed_str(res->name,  sizeof(res->name));
		data.write_fixed_str(res->value, sizeof(res->value));
		data.write_fixed_str(res->tags,  sizeof(res->tags));
		data.write(res->bind);
		data.write(res->element_size);
		uint16_t reserved = 0;
		data.write(res->shape);
		data.write(res->image_format);
		data.write(reserved);
	}

	for (uint32_t i = 0; i < file->meta.spec_constant_count; i++) {
		sksc_shader_spec_constant_t *spec = &file->meta.spec_constants[i];
		data.write_fixed_str(spec->name, sizeof(spec->name));
		data.write(spec->constant_id);
		data.write(spec->default_value);
		data.write(spec->type);
		data.write(spec->stage_bits);
	}

	for (uint32_t i = 0; i < file->meta.sampler_count; i++) { // v12
		sksc_shader_sampler_t *sampler = &file->meta.samplers[i];
		data.write_fixed_str(sampler->name, sizeof(sampler->name));
		data.write(sampler->slot);
		data.write(sampler->stage_bits);
		data.write(sampler->paired_slot);
	}

	for (uint32_t i = 0; i < file->stage_count; i++) {
		sksc_shader_file_stage_t *stage = &file->stages[i];
		data.write(stage->language);
		data.write(stage->stage);
		uint32_t stage_wave = stage->stage == skr_stage_compute ? file->meta.wave_size : 0;
		data.write(stage_wave);
		data.write(stage->code_size);
		data.write(stage->code, stage->code_size);
	}

	*out_data = data.data.data;
	*out_size = (uint32_t)data.data.count;
}
