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

///////////////////////////////////////////

void sksc_log_shader_info(const sksc_shader_file_t *file);

///////////////////////////////////////////

void sksc_init() {
	sksc_glslang_init();
}

///////////////////////////////////////////

void sksc_shutdown() {
	sksc_glslang_shutdown();
}

///////////////////////////////////////////

bool sksc_compile(const char *filename, const char *hlsl_text, sksc_settings_t *settings, sksc_shader_file_t *out_file) {
	*out_file = {};
	 out_file->meta = (sksc_shader_meta_t*)malloc(sizeof(sksc_shader_meta_t));
	*out_file->meta = {};
	 out_file->meta->global_buffer_id = -1;
	 out_file->meta->references = 1;

	array_t<sksc_shader_file_stage_t> stages       = {};
	array_t<sksc_meta_item_t>         var_meta     = sksc_meta_find_defaults(hlsl_text);
	array_t<sksc_ast_default_t>       ast_defaults = sksc_hlsl_find_initializers(hlsl_text);

	skr_stage_ compile_stages[3] = { skr_stage_vertex, skr_stage_pixel, skr_stage_compute };
	char*      entrypoints   [3] = { settings->vs_entrypoint, settings->ps_entrypoint, settings->cs_entrypoint };

	for (size_t i = 0; i < sizeof(compile_stages)/sizeof(compile_stages[0]); i++) {
		if (entrypoints[i][0] == 0)
			continue;

		// Build SPIRV
		sksc_shader_file_stage_t spirv_stage  = {};
		compile_result_          spirv_result = sksc_hlsl_to_spirv(filename, hlsl_text, settings, compile_stages[i], NULL, 0, &spirv_stage);
		if (spirv_result == compile_result_fail) {
			sksc_log(sksc_log_level_err, "SPIRV compile failed");
			return false;
		} else if (spirv_result == compile_result_skip)
			continue;

		// Extract metadata from the SPIRV
		sksc_spirv_to_meta(&spirv_stage, out_file->meta);

		// Add it as a stage in our sks file
		if (settings->target_langs[skr_shader_lang_spirv]) {
			stages.add(spirv_stage);
		}

		if (!settings->target_langs[skr_shader_lang_spirv])
			free(spirv_stage.code);
	}

	sksc_meta_assign_defaults(ast_defaults, var_meta, out_file->meta);
	var_meta.free();
	ast_defaults.free();
	out_file->stage_count = (uint32_t)stages.count;
	out_file->stages      = stages.data;

	if (!settings->silent_info) {
		sksc_log_shader_info(out_file);
	}

	if (!sksc_meta_check_dup_buffers(out_file->meta)) {
		sksc_log(sksc_log_level_err, "Found constant buffers re-using slot ids");
		return false;
	}

	const char *dup_name1, *dup_name2;
	uint32_t    dup_slot;
	if (!sksc_meta_check_dup_resources(out_file->meta, &dup_name1, &dup_name2, &dup_slot)) {
		sksc_log(sksc_log_level_err, "Resources '%s' and '%s' are both bound to the same slot (t%u)", dup_name1, dup_name2, dup_slot);
		return false;
	}

	return true;
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
	if (!file || !file->meta) return nullptr;

	const sksc_shader_meta_t *meta = file->meta;
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
			info.append("|  %s%d : %s%d", format, meta->vertex_inputs[i].count, semantic, meta->vertex_inputs[i].semantic_slot);
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

void sksc_build_file(const sksc_shader_file_t *file, void **out_data, uint32_t *out_size) {
	file_data_t data = {};

	const char tag[8] = {'S','K','S','H','A','D','E','R'};
	uint16_t version = 6;
	data.write(tag);
	data.write(version);

	data.write(file->stage_count);
	data.write_fixed_str(file->meta->name, sizeof(file->meta->name));
	data.write(file->meta->buffer_count);
	data.write(file->meta->resource_count);
	data.write(file->meta->vertex_input_count);

	data.write(file->meta->ops_vertex.total);
	data.write(file->meta->ops_vertex.tex_read);
	data.write(file->meta->ops_vertex.dynamic_flow);
	data.write(file->meta->ops_pixel.total);
	data.write(file->meta->ops_pixel.tex_read);
	data.write(file->meta->ops_pixel.dynamic_flow);

	for (uint32_t i = 0; i < file->meta->buffer_count; i++) {
		sksc_shader_buffer_t *buff = &file->meta->buffers[i];
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

	for (int32_t i = 0; i < file->meta->vertex_input_count; i++) {
		skr_vert_component_t *com = &file->meta->vertex_inputs[i];
		data.write(com->format);
		data.write(com->count);
		data.write(com->semantic);
		data.write(com->semantic_slot);
	}

	for (uint32_t i = 0; i < file->meta->resource_count; i++) {
		sksc_shader_resource_t *res = &file->meta->resources[i];
		data.write_fixed_str(res->name,  sizeof(res->name));
		data.write_fixed_str(res->value, sizeof(res->value));
		data.write_fixed_str(res->tags,  sizeof(res->tags));
		data.write(res->bind);
		data.write(res->element_size);
	}

	for (uint32_t i = 0; i < file->stage_count; i++) {
		sksc_shader_file_stage_t *stage = &file->stages[i];
		data.write(stage->language);
		data.write(stage->stage);
		data.write(stage->code_size);
		data.write(stage->code, stage->code_size);
	}

	*out_data = data.data.data;
	*out_size = (uint32_t)data.data.count;
}
