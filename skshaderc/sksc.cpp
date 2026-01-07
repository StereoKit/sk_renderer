#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
///////////////////////////////////////////

#include "sksc.h"
#include "_sksc.h"

#include "array.h"

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

void sksc_log_shader_info(const sksc_shader_file_t *file) {
	const sksc_shader_meta_t *meta = file->meta;
	
	sksc_log(sksc_log_level_info, " ________________");
	// Write out our reflection information

	// A quick summary of performance
	sksc_log(sksc_log_level_info, "|--Performance--");
	if (meta->ops_vertex.total > 0 || meta->ops_pixel.total > 0)
	sksc_log(sksc_log_level_info, "| Instructions |  all | tex | flow |");
	if (meta->ops_vertex.total > 0) {
		sksc_log(sksc_log_level_info, "|       Vertex | %4d | %3d | %4d |",
			meta->ops_vertex.total,
			meta->ops_vertex.tex_read,
			meta->ops_vertex.dynamic_flow);
	}
	if (meta->ops_pixel.total > 0) {
		sksc_log(sksc_log_level_info, "|        Pixel | %4d | %3d | %4d |",
			meta->ops_pixel.total,
			meta->ops_pixel.tex_read,
			meta->ops_pixel.dynamic_flow);
	}

	// List of all the buffers
	sksc_log(sksc_log_level_info, "|--Buffer Info--");
	for (size_t i = 0; i < meta->buffer_count; i++) {
		sksc_shader_buffer_t *buff = &meta->buffers[i];
		sksc_log(sksc_log_level_info, "|  %s - %u bytes%s", buff->name, buff->size, buff->defaults ? " (has defaults)" : "");
		for (size_t v = 0; v < buff->var_count; v++) {
			sksc_shader_var_t *var = &buff->vars[v];
			const char *type_str = var->type_name[0] ? var->type_name : "unknown";

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
			sksc_log(sksc_log_level_info, "|    %-15s: +%-4u %5ub - %s[%u]%s", var->name, var->offset, var->size, type_str, var->type_count, default_str);
		}
	}

	// Show the vertex shader's input format
	if (meta->vertex_input_count > 0) {
		sksc_log(sksc_log_level_info, "|--Mesh Input--");
		for (int32_t i=0; i<meta->vertex_input_count; i++) {
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
			sksc_log(sksc_log_level_info, "|  %s%d : %s%d", format, meta->vertex_inputs[i].count, semantic, meta->vertex_inputs[i].semantic_slot);
		}
	} 

	// Only log buffer binds for the stages of a single language. Doesn't
	// matter which.
	skr_shader_lang_ stage_lang = file->stage_count > 0 ? file->stages[0].language : skr_shader_lang_hlsl;
	for (uint32_t s = 0; s < file->stage_count; s++) {
		const sksc_shader_file_stage_t* stage = &file->stages[s];

		if (stage->language != stage_lang)
			continue;

		const char *stage_name = "";
		switch (stage->stage) {
		case skr_stage_vertex:  stage_name = "Vertex";  break;
		case skr_stage_pixel:   stage_name = "Pixel";   break;
		case skr_stage_compute: stage_name = "Compute"; break;
		}
		sksc_log(sksc_log_level_info, "|--%s Shader--", stage_name);
		for (uint32_t i = 0; i < meta->buffer_count; i++) {
			sksc_shader_buffer_t *buff = &meta->buffers[i];
			if (buff->bind.stage_bits & stage->stage) {
				sksc_log(sksc_log_level_info, "|  b%u/s%d : %s", buff->bind.slot, buff->space, buff->name);
			}
		}
		for (uint32_t i = 0; i < meta->resource_count; i++) {
			sksc_shader_resource_t *tex = &meta->resources[i];
			if (tex->bind.stage_bits & stage->stage) {
				bool is_storage_buffer = tex->bind.register_type == skr_register_read_buffer || tex->bind.register_type == skr_register_readwrite;
				char reg_char          = (tex->bind.register_type == skr_register_texture || tex->bind.register_type == skr_register_read_buffer) ? 't' : 'u';
				if (is_storage_buffer && tex->element_size > 0) {
					sksc_log(sksc_log_level_info, "|  %c%u : %s (%u bytes/element)", reg_char, tex->bind.slot, tex->name, tex->element_size);
				} else {
					sksc_log(sksc_log_level_info, "|  %c%u : %s", reg_char, tex->bind.slot, tex->name);
				}
			}
		}
	}
	sksc_log(sksc_log_level_info, "|________________");
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
	uint16_t version = 5;
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
	*out_size = data.data.count;
}
