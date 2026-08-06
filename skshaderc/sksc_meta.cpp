#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#define NOMINMAX

#include "_sksc.h"
#include "array.h"

#include <string.h>

//#include <spirv_hlsl.hpp>
#include <spirv_reflect.h>

///////////////////////////////////////////

void sksc_line_col (const char *from_text, const char *at, int32_t *out_line, int32_t *out_column);
int  strcmp_nocase (char const *a, char const *b);
void parse_semantic(const char* str, char* out_str, size_t out_size, int32_t* out_idx);

///////////////////////////////////////////
// HLSL Source Initializer Parser        //
///////////////////////////////////////////

static bool _is_identifier_char(char c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static bool _is_whitespace(char c) {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static const char *_skip_to_line_end(const char *c) {
	while (*c && *c != '\n') c++;
	return c;
}

static const char *_skip_block_comment(const char *c) {
	c += 2; // skip /*
	while (*c && !(*c == '*' && *(c+1) == '/')) c++;
	if (*c) c += 2; // skip */
	return c;
}

static const char *_skip_whitespace_and_comments(const char *c) {
	while (*c) {
		if (_is_whitespace(*c)) {
			c++;
		} else if (*c == '/' && *(c+1) == '/') {
			c = _skip_to_line_end(c);
		} else if (*c == '/' && *(c+1) == '*') {
			c = _skip_block_comment(c);
		} else {
			break;
		}
	}
	return c;
}

// Skip a balanced brace block { ... }
static const char *_skip_brace_block(const char *c) {
	if (*c != '{') return c;
	c++;
	int32_t depth = 1;
	while (*c && depth > 0) {
		if (*c == '{') depth++;
		else if (*c == '}') depth--;
		else if (*c == '/' && *(c+1) == '/') c = _skip_to_line_end(c) - 1;
		else if (*c == '/' && *(c+1) == '*') { c = _skip_block_comment(c) - 1; }
		c++;
	}
	return c;
}

static bool _is_identifier_start(char c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

// Parse numeric values from an initializer expression
// Handles: 1.0, float3(1,2,3), {1,2,3,4}, -0.5, etc.
static int32_t _parse_initializer_values(const char *start, const char *end, double *out_values, int32_t max_values) {
	int32_t count = 0;
	const char *c = start;

	while (c < end && count < max_values) {
		c = _skip_whitespace_and_comments(c);
		if (c >= end) break;

		// Handle true/false before identifier skip
		if (strncmp(c, "true", 4) == 0 && !_is_identifier_char(c[4])) {
			out_values[count++] = 1.0;
			c += 4;
			continue;
		}
		if (strncmp(c, "false", 5) == 0 && !_is_identifier_char(c[5])) {
			out_values[count++] = 0.0;
			c += 5;
			continue;
		}

		// Skip type constructors like float3(, float4x4(, etc
		// Only treat as identifier if it starts with letter/underscore, not digit
		if (_is_identifier_start(*c)) {
			while (c < end && _is_identifier_char(*c)) c++;
			c = _skip_whitespace_and_comments(c);
			continue;
		}

		// Skip opening parens/braces
		if (*c == '(' || *c == '{') { c++; continue; }
		// Skip closing parens/braces
		if (*c == ')' || *c == '}') { c++; continue; }
		// Skip commas
		if (*c == ',') { c++; continue; }

		// Try to parse a number
		if (*c == '-' || *c == '+' || *c == '.' || (*c >= '0' && *c <= '9')) {
			char *num_end;
			double val = strtod(c, &num_end);
			if (num_end > c) {
				out_values[count++] = val;
				c = num_end;
				// Skip 'f' suffix
				if (*c == 'f' || *c == 'F') c++;
				continue;
			}
		}

		// Unknown token, skip it
		c++;
	}
	return count;
}

array_t<sksc_ast_default_t> sksc_hlsl_find_initializers(const char *hlsl_text) {
	array_t<sksc_ast_default_t> result = {};
	const char *c = hlsl_text;

	// Known HLSL scalar/vector/matrix type prefixes
	static const char * const type_prefixes[] = {
		"float", "half", "double", "int", "uint", "bool", "min16float", "min10float", "min16int", "min12int", "min16uint"
	};
	static const int32_t num_prefixes = sizeof(type_prefixes) / sizeof(type_prefixes[0]);

	while (*c) {
		c = _skip_whitespace_and_comments(c);
		if (!*c) break;

		// Skip preprocessor directives
		if (*c == '#') {
			c = _skip_to_line_end(c);
			continue;
		}

		// Check for keywords that introduce blocks we should skip
		// struct, cbuffer, tbuffer, class, interface, namespace
		if (strncmp(c, "struct",    6) == 0 && !_is_identifier_char(c[6])) { c += 6; c = _skip_whitespace_and_comments(c); while (*c && *c != '{') c++; c = _skip_brace_block(c); continue; }
		if (strncmp(c, "cbuffer",   7) == 0 && !_is_identifier_char(c[7])) { c += 7; c = _skip_whitespace_and_comments(c); while (*c && *c != '{') c++; c = _skip_brace_block(c); continue; }
		if (strncmp(c, "tbuffer",   7) == 0 && !_is_identifier_char(c[7])) { c += 7; c = _skip_whitespace_and_comments(c); while (*c && *c != '{') c++; c = _skip_brace_block(c); continue; }
		if (strncmp(c, "class",     5) == 0 && !_is_identifier_char(c[5])) { c += 5; c = _skip_whitespace_and_comments(c); while (*c && *c != '{') c++; c = _skip_brace_block(c); continue; }
		if (strncmp(c, "interface", 9) == 0 && !_is_identifier_char(c[9])) { c += 9; c = _skip_whitespace_and_comments(c); while (*c && *c != '{') c++; c = _skip_brace_block(c); continue; }
		if (strncmp(c, "namespace", 9) == 0 && !_is_identifier_char(c[9])) { c += 9; c = _skip_whitespace_and_comments(c); while (*c && *c != '{') c++; c = _skip_brace_block(c); continue; }

		// Check for function definitions (has parens before brace)
		// Look ahead to see if this looks like a function
		const char *lookahead = c;
		while (*lookahead && _is_identifier_char(*lookahead)) lookahead++;
		lookahead = _skip_whitespace_and_comments(lookahead);
		// Skip array dimensions
		while (*lookahead == '[') { while (*lookahead && *lookahead != ']') lookahead++; if (*lookahead) lookahead++; lookahead = _skip_whitespace_and_comments(lookahead); }
		while (*lookahead && _is_identifier_char(*lookahead)) lookahead++;
		lookahead = _skip_whitespace_and_comments(lookahead);
		if (*lookahead == '(') {
			// This might be a function, skip to after the parens
			lookahead++;
			int32_t paren_depth = 1;
			while (*lookahead && paren_depth > 0) {
				if (*lookahead == '(') paren_depth++;
				else if (*lookahead == ')') paren_depth--;
				lookahead++;
			}
			lookahead = _skip_whitespace_and_comments(lookahead);
			// Skip semantics like : SV_Target
			if (*lookahead == ':') {
				lookahead++;
				lookahead = _skip_whitespace_and_comments(lookahead);
				while (*lookahead && _is_identifier_char(*lookahead)) lookahead++;
				lookahead = _skip_whitespace_and_comments(lookahead);
			}
			if (*lookahead == '{') {
				c = _skip_brace_block(lookahead);
				continue;
			}
		}

		// Check if this is a known type
		bool is_type = false;
		int32_t type_len = 0;
		for (int32_t i = 0; i < num_prefixes; i++) {
			size_t len = strlen(type_prefixes[i]);
			if (strncmp(c, type_prefixes[i], len) == 0) {
				// Check it's not part of a larger identifier
				char next = c[len];
				// Allow digits for float2, float3x3, etc, or end of type
				if (!_is_identifier_char(next) || (next >= '0' && next <= '9')) {
					is_type = true;
					type_len = (int32_t)len;
					break;
				}
			}
		}

		if (!is_type) {
			// Skip this identifier/token
			if (_is_identifier_char(*c)) {
				while (*c && _is_identifier_char(*c)) c++;
			} else {
				c++;
			}
			continue;
		}

		// We found a type, now parse: type name = initializer;
		c += type_len;
		// Skip dimension suffixes like 2, 3, 4, 2x2, 3x3, 4x4
		while (*c && ((*c >= '0' && *c <= '9') || *c == 'x')) c++;

		c = _skip_whitespace_and_comments(c);

		// Get variable name
		if (!_is_identifier_char(*c)) continue;
		const char *name_start = c;
		while (*c && _is_identifier_char(*c)) c++;
		const char *name_end = c;

		c = _skip_whitespace_and_comments(c);

		// Skip array dimensions
		while (*c == '[') {
			while (*c && *c != ']') c++;
			if (*c) c++;
			c = _skip_whitespace_and_comments(c);
		}

		// Skip semantics : SEMANTIC
		if (*c == ':') {
			c++;
			c = _skip_whitespace_and_comments(c);
			while (*c && _is_identifier_char(*c)) c++;
			c = _skip_whitespace_and_comments(c);
		}

		// Check for initializer
		if (*c != '=') {
			// No initializer, skip to semicolon
			while (*c && *c != ';') c++;
			if (*c) c++;
			continue;
		}
		c++; // skip =
		c = _skip_whitespace_and_comments(c);

		// Find end of initializer (semicolon)
		const char *init_start = c;
		int32_t brace_depth = 0;
		int32_t paren_depth = 0;
		while (*c && !(*c == ';' && brace_depth == 0 && paren_depth == 0)) {
			if (*c == '{') brace_depth++;
			else if (*c == '}') brace_depth--;
			else if (*c == '(') paren_depth++;
			else if (*c == ')') paren_depth--;
			c++;
		}
		const char *init_end = c;

		// Parse the initializer values
		sksc_ast_default_t def = {};
		size_t name_len = name_end - name_start;
		if (name_len >= sizeof(def.name)) name_len = sizeof(def.name) - 1;
		memcpy(def.name, name_start, name_len);
		def.name[name_len] = '\0';

		def.value_count = _parse_initializer_values(init_start, init_end, def.values, 16);

		if (def.value_count > 0) {
			result.add(def);
		}

		if (*c == ';') c++;
	}

	return result;
}

///////////////////////////////////////////
// SPIRV Specialization Constant Scanner  //
///////////////////////////////////////////

// SPIRV-Reflect doesn't expose spec constant types or defaults, so walk the
// words directly. Valid SPIRV puts OpName/OpDecorate before the constants
// section, so a single forward pass sees everything it needs.
static void _sksc_spirv_scan_spec_constants(const sksc_shader_file_stage_t *spirv_stage, sksc_shader_meta_t *ref_meta) {
	const uint32_t *words      = (const uint32_t *)spirv_stage->code;
	size_t          word_count = spirv_stage->code_size / sizeof(uint32_t);
	const size_t    SPIRV_HEADER_SIZE = 5;
	if (word_count <= SPIRV_HEADER_SIZE) return;

	// Per-id lookup tables, sized from the module's id bound (header word 3)
	uint32_t id_bound = words[3];
	if (id_bound == 0 || id_bound > 0x400000) return;

	enum type_kind_ { type_kind_none, type_kind_bool, type_kind_int, type_kind_uint, type_kind_float };
	uint8_t     *type_kinds = (uint8_t    *)calloc(id_bound, sizeof(uint8_t));
	int64_t     *spec_ids   = (int64_t    *)malloc(id_bound * sizeof(int64_t)); // -1 = not decorated with SpecId
	const char **names      = (const char**)calloc(id_bound, sizeof(const char*));
	for (uint32_t i = 0; i < id_bound; i++) spec_ids[i] = -1;

	array_t<sksc_shader_spec_constant_t> spec_list = {};
	spec_list.data     = ref_meta->spec_constants;
	spec_list.capacity = ref_meta->spec_constant_count;
	spec_list.count    = ref_meta->spec_constant_count;

	for (size_t i = SPIRV_HEADER_SIZE; i < word_count; ) {
		uint32_t op_words = words[i] >> 16;
		uint32_t opcode   = words[i] & 0xFFFF;
		if (op_words == 0 || i + op_words > word_count) break; // Malformed SPIRV

		switch (opcode) {
			case 5: // OpName: [target id, literal string...]
				if (op_words >= 3 && words[i+1] < id_bound)
					names[words[i+1]] = (const char *)&words[i+2];
				break;
			case 71: // OpDecorate: [target id, decoration, extra...] — SpecId is decoration 1
				if (op_words >= 4 && words[i+2] == 1 && words[i+1] < id_bound)
					spec_ids[words[i+1]] = (int64_t)words[i+3];
				break;
			case 20: // OpTypeBool: [result id]
				if (op_words >= 2 && words[i+1] < id_bound)
					type_kinds[words[i+1]] = type_kind_bool;
				break;
			case 21: // OpTypeInt: [result id, width, signedness] — only 32-bit supported
				if (op_words >= 4 && words[i+1] < id_bound && words[i+2] == 32)
					type_kinds[words[i+1]] = words[i+3] ? type_kind_int : type_kind_uint;
				break;
			case 22: // OpTypeFloat: [result id, width] — only 32-bit supported
				if (op_words >= 3 && words[i+1] < id_bound && words[i+2] == 32)
					type_kinds[words[i+1]] = type_kind_float;
				break;
			case 48:   // OpSpecConstantTrue:  [result type, result id]
			case 49:   // OpSpecConstantFalse: [result type, result id]
			case 50: { // OpSpecConstant:      [result type, result id, value literal...]
				if (op_words < 3) break;
				uint32_t type_id   = words[i+1];
				uint32_t result_id = words[i+2];
				if (type_id >= id_bound || result_id >= id_bound)  break;
				if (spec_ids  [result_id] < 0)                     break; // Not user-specializable (no SpecId)
				if (type_kinds[type_id  ] == type_kind_none)       break; // Unsupported (non-32-bit) type

				sksc_shader_spec_constant_t spec = {};
				spec.constant_id = (uint32_t)spec_ids[result_id];
				switch (opcode) {
					case 48: spec.default_value = 1;          break;
					case 49: spec.default_value = 0;          break;
					case 50: spec.default_value = words[i+3]; break;
				}
				switch (type_kinds[type_id]) {
					case type_kind_bool:  spec.type = sksc_shader_var_int;   break; // VkBool32
					case type_kind_int:   spec.type = sksc_shader_var_int;   break;
					case type_kind_uint:  spec.type = sksc_shader_var_uint;  break;
					case type_kind_float: spec.type = sksc_shader_var_float; break;
				}
				spec.stage_bits = spirv_stage->stage;
				if (names[result_id]) strncpy(spec.name, names[result_id], sizeof(spec.name) - 1);
				else                  snprintf(spec.name, sizeof(spec.name), "spec%u", spec.constant_id);

				// Merge with the same constant across other stages
				int64_t existing = spec_list.index_where([](const sksc_shader_spec_constant_t &s, void *data) {
					return s.constant_id == *(uint32_t *)data;
				}, &spec.constant_id);
				if (existing >= 0) {
					sksc_shader_spec_constant_t *prev = &spec_list[existing];
					if (prev->type != spec.type || strcmp(prev->name, spec.name) != 0) {
						sksc_log(sksc_log_level_warn, "Specialization constant id %u is declared differently across stages ('%s' vs '%s')", spec.constant_id, prev->name, spec.name);
					}
					prev->stage_bits |= spec.stage_bits;
				} else {
					spec_list.add(spec);
				}
			} break;
			default: break;
		}

		i += op_words;
	}

	ref_meta->spec_constants      = spec_list.data;
	ref_meta->spec_constant_count = (uint32_t)spec_list.count;

	free(type_kinds);
	free(spec_ids);
	free((void*)names);
}

///////////////////////////////////////////

// Encode spirv-reflect image traits into sksc_shader_resource_t.shape:
// bits 0-2 dimension (1 = 2D, 2 = 3D, 3 = cube, 4 = 1D), bit 3 arrayed,
// bit 4 multisampled, bit 5 depth-comparison
static uint8_t _sksc_image_shape(const SpvReflectImageTraits *image) {
	uint8_t shape;
	switch (image->dim) {
		case SpvDim1D:   shape = 4; break;
		case SpvDim3D:   shape = 2; break;
		case SpvDimCube: shape = 3; break;
		default:         shape = 1; break; // 2D, SubpassData, Rect
	}
	if (image->arrayed)    shape |= SKSC_SHAPE_ARRAYED;
	if (image->ms)         shape |= SKSC_SHAPE_MS;
	if (image->depth == 1) shape |= SKSC_SHAPE_COMPARISON;
	return shape;
}

bool sksc_spirv_to_meta(const sksc_shader_file_stage_t *spirv_stage, sksc_shader_meta_t *ref_meta) {
	// Create reflection data
	SpvReflectShaderModule module;
	SpvReflectResult       result = spvReflectCreateShaderModule(spirv_stage->code_size, spirv_stage->code, &module);
	
	if (result != SPV_REFLECT_RESULT_SUCCESS) {
		sksc_log(sksc_log_level_err, "[SPIRV-Reflect] Failed to create shader module: %d", result);
		return false;
	}

	array_t<sksc_shader_buffer_t> buffer_list = {};
	buffer_list.data      = ref_meta->buffers;
	buffer_list.capacity  = ref_meta->buffer_count;
	buffer_list.count     = ref_meta->buffer_count;
	array_t<sksc_shader_resource_t> resource_list = {};
	resource_list.data     = ref_meta->resources;
	resource_list.capacity = ref_meta->resource_count;
	resource_list.count    = ref_meta->resource_count;

	// Get descriptor bindings
	uint32_t binding_count = 0;
	result = spvReflectEnumerateDescriptorBindings(&module, &binding_count, nullptr);
	if (result != SPV_REFLECT_RESULT_SUCCESS) {
		spvReflectDestroyShaderModule(&module);
		return false;
	}

	SpvReflectDescriptorBinding** bindings = (SpvReflectDescriptorBinding**)malloc(sizeof(SpvReflectDescriptorBinding*) * binding_count);
	result = spvReflectEnumerateDescriptorBindings(&module, &binding_count, bindings);
	if (result != SPV_REFLECT_RESULT_SUCCESS) {
		spvReflectDestroyShaderModule(&module);
		return false;
	}

	// Process uniform buffers
	for (uint32_t i = 0; i < binding_count; i++) {
		SpvReflectDescriptorBinding* binding = bindings[i];
		
		if (binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
			// Find or create a buffer
			const char* buffer_name = binding->type_description->type_name
				? binding->type_description->type_name 
				: (binding->name ? binding->name : "");

			int64_t id = buffer_list.index_where([](const sksc_shader_buffer_t &buff, void *data) { 
				return strcmp(buff.name, (char*)data) == 0; 
			}, (void*)buffer_name);
			bool is_new = id == -1;
			if (is_new) id = buffer_list.add({});

			// Update the stage of this buffer
			sksc_shader_buffer_t *buff = &buffer_list[id];
			buff->bind.stage_bits |= spirv_stage->stage;

			// And skip the rest if we've already seen it
			if (!is_new) continue;

			SpvReflectTypeDescription* type_desc = binding->type_description;
			uint32_t count = type_desc->member_count;

			buff->size               = (uint32_t)(binding->block.size % 16 == 0 ? binding->block.size : (binding->block.size / 16 + 1) * 16);
			buff->space              = binding->set;
			buff->bind.slot          = binding->binding;
			buff->bind.stage_bits    = spirv_stage->stage;
			buff->bind.register_type = skr_register_constant;
			buff->var_count          = count;
			buff->vars               = (sksc_shader_var_t*)malloc(count * sizeof(sksc_shader_var_t));
			memset(buff->vars, 0, count * sizeof(sksc_shader_var_t));
			strncpy(buff->name, buffer_name, sizeof(buff->name));

			for (uint32_t m = 0; m < count; m++) {
				SpvReflectBlockVariable* member = &binding->block.members[m];
				
				uint32_t dimensions = member->array.dims_count;
				int32_t  dim_size   = 1;
				for (uint32_t d = 0; d < dimensions; d++) {
					dim_size = dim_size * member->array.dims[d];
				}
				
				const char* member_name = member->name ? member->name : "";
				strncpy(buff->vars[m].name, member_name, sizeof(buff->vars[m].name));
				buff->vars[m].offset     = member->offset;
				buff->vars[m].size       = member->size;

				uint32_t vec_size = member->type_description->traits.numeric.vector.component_count;
				uint32_t columns  = member->type_description->traits.numeric.matrix.column_count;
				if (vec_size == 0) vec_size = 1;
				if (columns  == 0) columns  = 1;

				buff->vars[m].type_count = dim_size * vec_size * columns;

				if (buff->vars[m].type_count == 0)
					buff->vars[m].type_count = 1;

				// Build type name - use SPIRV type_name for structs, construct for primitives
				const char* type_name = member->type_description->type_name;
				if (type_name) {
					strncpy(buff->vars[m].type_name, type_name, sizeof(buff->vars[m].type_name));
				} else {
					// Construct type name for primitive types
					const char* base_type = "unknown";
					switch (member->type_description->type_flags & 0xFF) {
						case SPV_REFLECT_TYPE_FLAG_INT:
							if (member->type_description->traits.numeric.scalar.signedness) {
								base_type = "int";
							} else {
								base_type = member->type_description->traits.numeric.scalar.width == 8 ? "uint8" : "uint";
							}
							break;
						case SPV_REFLECT_TYPE_FLAG_FLOAT:
							base_type = member->type_description->traits.numeric.scalar.width == 64 ? "double" : "float";
							break;
						case SPV_REFLECT_TYPE_FLAG_BOOL:
							base_type = "bool";
							break;
					}

					// Build: base, base2, base3, base4, or base4x4 for matrices
					if (columns > 1) {
						snprintf(buff->vars[m].type_name, sizeof(buff->vars[m].type_name), "%s%ux%u", base_type, vec_size, columns);
					} else if (vec_size > 1) {
						snprintf(buff->vars[m].type_name, sizeof(buff->vars[m].type_name), "%s%u", base_type, vec_size);
					} else {
						strncpy(buff->vars[m].type_name, base_type, sizeof(buff->vars[m].type_name));
					}
				}

				switch (member->type_description->type_flags & 0xFF) {
					case SPV_REFLECT_TYPE_FLAG_INT:
						if (member->type_description->traits.numeric.scalar.signedness) {
							buff->vars[m].type = sksc_shader_var_int;
						} else {
							if (member->type_description->traits.numeric.scalar.width == 8)
								buff->vars[m].type = sksc_shader_var_uint8;
							else
								buff->vars[m].type = sksc_shader_var_uint;
						}
						break;
					case SPV_REFLECT_TYPE_FLAG_FLOAT:
						if (member->type_description->traits.numeric.scalar.width == 64)
							buff->vars[m].type = sksc_shader_var_double;
						else
							buff->vars[m].type = sksc_shader_var_float;
						break;
					default:
						buff->vars[m].type = sksc_shader_var_none;
						break;
				}
			}
			
			if (strcmp(buff->name, "$Global") == 0) {
				ref_meta->global_buffer_id = (int32_t)id;
			}
		}
	}

	// Find textures (sampled images)
	for (uint32_t i = 0; i < binding_count; i++) {
		SpvReflectDescriptorBinding* binding = bindings[i];
		
		if (binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
		    binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
			const char* name = binding->name ? binding->name : "";
			int64_t id = resource_list.index_where([](const sksc_shader_resource_t &tex, void *data) {
				return strcmp(tex.name, (char*)data) == 0;
			}, (void*)name);
			if (id == -1)
				id = resource_list.add({});

			sksc_shader_resource_t *tex = &resource_list[id];
			tex->bind.slot          = binding->binding;
			tex->bind.stage_bits   |= spirv_stage->stage;
			tex->bind.register_type = skr_register_texture;
			tex->shape             |= _sksc_image_shape(&binding->image);
			strncpy(tex->name, name, sizeof(tex->name));
		}
	}

	// Look for storage images (RWTexture2D)
	for (uint32_t i = 0; i < binding_count; i++) {
		SpvReflectDescriptorBinding* binding = bindings[i];
		
		if (binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
			const char* name = binding->name ? binding->name : "";
			int64_t id = resource_list.index_where([](const sksc_shader_resource_t &tex, void *data) {
				return strcmp(tex.name, (char*)data) == 0;
			}, (void*)name);
			if (id == -1)
				id = resource_list.add({});

			sksc_shader_resource_t *tex = &resource_list[id];
			tex->bind.slot          = binding->binding;
			tex->bind.stage_bits   |= spirv_stage->stage;
			tex->bind.register_type = skr_register_readwrite_tex;
			tex->shape             |= _sksc_image_shape(&binding->image);
			tex->image_format       = (uint8_t)binding->image.image_format;
			strncpy(tex->name, name, sizeof(tex->name));
		}
	}

	// Look for storage buffers (RWStructuredBuffers and StructuredBuffers)
	for (uint32_t i = 0; i < binding_count; i++) {
		SpvReflectDescriptorBinding* binding = bindings[i];

		if (binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
			const char* name = binding->name ? binding->name : "";

			int64_t id = resource_list.index_where([](const sksc_shader_resource_t &tex, void *data) {
				return strcmp(tex.name, (char*)data) == 0;
			}, (void*)name);
			if (id == -1)
				id = resource_list.add({});

			sksc_shader_resource_t *tex = &resource_list[id];
			tex->bind.slot          = binding->binding;
			tex->bind.stage_bits   |= spirv_stage->stage;
			tex->bind.register_type = binding->resource_type == SPV_REFLECT_RESOURCE_FLAG_SRV ? skr_register_read_buffer : skr_register_readwrite;

			// For StructuredBuffer<T>, DXC wraps the runtime array in a block with
			// a single member named @data. The member's array.stride gives us the
			// properly aligned element size (accounts for HLSL struct padding).
			uint32_t element_size = 0;
			if (binding->block.member_count > 0 && binding->block.members != nullptr) {
				SpvReflectBlockVariable* member = &binding->block.members[0];

				// Prefer type_description's array stride (includes HLSL struct alignment padding)
				if      (member->type_description && member->type_description->traits.array.stride > 0) { element_size = member->type_description->traits.array.stride; }
				else if (member->array.stride > 0) { element_size = member->array.stride; } // Fall back to member's array stride
				else if (member->padded_size  > 0) { element_size = member->padded_size;  } // Fall back to padded_size
				else                               { element_size = member->size;         } // Fall back to size
				

				// For primitive types (float4, int, etc.), size may be 0.
				// Calculate from type traits: width * component_count / 8
				if (element_size == 0 && member->type_description) {
					SpvReflectTypeDescription* td = member->type_description;
					uint32_t width      = td->traits.numeric.scalar.width;
					uint32_t components = td->traits.numeric.vector.component_count;
					if (components == 0) components = 1;
					if (width > 0) {
						element_size = (width * components) / 8;
					}
				}
			}
			tex->element_size = element_size;

			strncpy(tex->name, name, sizeof(tex->name));
		}
	}

	// Look for input attachments (SubpassInput in HLSL)
	for (uint32_t i = 0; i < binding_count; i++) {
		SpvReflectDescriptorBinding* binding = bindings[i];

		if (binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
			const char* name = binding->name ? binding->name : "";
			int64_t id = resource_list.index_where([](const sksc_shader_resource_t &res, void *data) {
				return strcmp(res.name, (char*)data) == 0;
			}, (void*)name);
			if (id == -1)
				id = resource_list.add({});

			sksc_shader_resource_t *res = &resource_list[id];
			res->bind.slot          = binding->binding;
			res->bind.stage_bits   |= spirv_stage->stage;
			res->bind.register_type = skr_register_input_attachment;
			res->shape             |= _sksc_image_shape(&binding->image);
			strncpy(res->name, name, sizeof(res->name));
		}
	}

	free(bindings);

	// Get vertex input info
	if (spirv_stage->stage == skr_stage_vertex) {
		uint32_t input_count = 0;
		result = spvReflectEnumerateInputVariables(&module, &input_count, nullptr);
		if (result != SPV_REFLECT_RESULT_SUCCESS) {
			spvReflectDestroyShaderModule(&module);
			return false;
		}

		SpvReflectInterfaceVariable** inputs = (SpvReflectInterfaceVariable**)malloc(sizeof(SpvReflectInterfaceVariable*) * input_count);
		result = spvReflectEnumerateInputVariables(&module, &input_count, inputs);
		if (result != SPV_REFLECT_RESULT_SUCCESS) {
			spvReflectDestroyShaderModule(&module);
			return false;
		}

		ref_meta->vertex_input_count = 0;
		ref_meta->vertex_inputs = (skr_vert_component_t*)malloc(sizeof(skr_vert_component_t) * input_count);

		int32_t curr = 0;
		for (uint32_t i = 0; i < input_count; i++) {
			SpvReflectInterfaceVariable* input = inputs[i];
			
			// Skip built-ins
			if (input->built_in != (SpvBuiltIn)0xFFFFFFFF) {
				continue;
			}

			const char* semantic_str = input->semantic ? input->semantic : "";
			
			char    semantic[64];
			int32_t semantic_idx = 0;
			parse_semantic(semantic_str, semantic, sizeof(semantic), &semantic_idx);

			if (strlen(semantic) > 3 &&
				tolower(semantic[0]) == 's' &&
				tolower(semantic[1]) == 'v' &&
				tolower(semantic[2]) == '_' &&
				strcmp_nocase(semantic, "sv_position") != 0)
			{
				continue;
			}

			ref_meta->vertex_inputs[curr].semantic_slot = semantic_idx;
			// SPIR-V input location this attribute is decorated with. Read it
			// straight from reflection rather than inferring from array index:
			// enumeration order is not guaranteed to match location order, and
			// skipped built-ins/SV_ inputs would break any index-based scheme.
			ref_meta->vertex_inputs[curr].location = (uint8_t)input->location;
			if      (strcmp_nocase(semantic, "sv_position" ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_position;     }
			else if (strcmp_nocase(semantic, "binormal"    ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_binormal;     }
			else if (strcmp_nocase(semantic, "blendindices") == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_blendindices; }
			else if (strcmp_nocase(semantic, "blendweight" ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_blendweight;  }
			else if (strcmp_nocase(semantic, "color"       ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_color;        }
			else if (strcmp_nocase(semantic, "normal"      ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_normal;       }
			else if (strcmp_nocase(semantic, "position"    ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_position;     }
			else if (strcmp_nocase(semantic, "psize"       ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_psize;        }
			else if (strcmp_nocase(semantic, "tangent"     ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_tangent;      }
			else if (strcmp_nocase(semantic, "texcoord"    ) == 0) { ref_meta->vertex_inputs[curr].semantic = skr_semantic_texcoord;     }

			uint32_t vec_size = input->type_description->traits.numeric.vector.component_count;
			ref_meta->vertex_inputs[curr].count = vec_size > 0 ? vec_size : 1;
			
			switch (input->type_description->type_flags & 0xFF) {
				case SPV_REFLECT_TYPE_FLAG_FLOAT: ref_meta->vertex_inputs[curr].format = skr_vertex_fmt_f32;  break;
				case SPV_REFLECT_TYPE_FLAG_INT:
					if (input->type_description->traits.numeric.scalar.signedness)
						ref_meta->vertex_inputs[curr].format = skr_vertex_fmt_i32;
					else
						ref_meta->vertex_inputs[curr].format = skr_vertex_fmt_ui32;
					break;
				default: ref_meta->vertex_inputs[curr].format = skr_vertex_fmt_none; break;
			}
			curr += 1;
		}
		free(inputs);
		ref_meta->vertex_input_count = curr;
	}

	ref_meta->buffers        = buffer_list.data;
	ref_meta->buffer_count   = (uint32_t)buffer_list.count;
	ref_meta->resources      = resource_list.data;
	ref_meta->resource_count = (uint32_t)resource_list.count;

	// Count SPIRV instructions for performance metrics
	// We only count "executable" instructions, skipping metadata like:
	// - OpNop, OpSource, OpName, OpMemberName, OpString, OpLine (0-8)
	// - OpExtension, OpExtInstImport, OpMemoryModel, OpEntryPoint, OpExecutionMode, OpCapability (11-17)
	// - OpType* declarations (19-39)
	// - OpConstant* definitions (41-52)
	// - OpDecorate, OpMemberDecorate, OpDecorationGroup, etc. (71-76)
	// - OpVariable declarations (59)
	sksc_shader_ops_t ops = {};
	const uint32_t *spirv      = (const uint32_t *)spirv_stage->code;
	size_t          spirv_size = spirv_stage->code_size / sizeof(uint32_t);
	const size_t    SPIRV_HEADER_SIZE = 5;

	for (size_t i = SPIRV_HEADER_SIZE; i < spirv_size; ) {
		uint32_t word_count = spirv[i] >> 16;
		uint32_t opcode     = spirv[i] & 0xFFFF;

		if (word_count == 0) break; // Malformed SPIRV

		// Skip metadata/declaration opcodes
		bool is_metadata =
			(opcode <= 8)            || // OpNop, OpUndef, OpSource*, OpName, OpMemberName, OpString, OpLine, OpNoLine
			(opcode >= 11 && opcode <= 17) || // OpExtension, OpExtInstImport, OpMemoryModel, OpEntryPoint, OpExecutionMode, OpCapability, OpExecutionModeId
			(opcode >= 19 && opcode <= 39)  || // OpType* declarations
			(opcode >= 41 && opcode <= 52)  || // OpConstant* definitions
			(opcode == 59)           || // OpVariable
			(opcode >= 71 && opcode <= 76);    // OpDecorate, OpMemberDecorate, etc.

		if (!is_metadata) {
			ops.total++;

			// Texture sample/fetch/gather/read operations (opcodes 87-98)
			if (opcode >= 87 && opcode <= 98) {
				ops.tex_read++;
			}
			// Dynamic control flow: OpBranch(249), OpBranchConditional(250), OpSwitch(251)
			else if (opcode >= 249 && opcode <= 251) {
				ops.dynamic_flow++;
			}
		}

		i += word_count;
	}

	if (spirv_stage->stage == skr_stage_vertex) {
		ref_meta->ops_vertex = ops;
	} else if (spirv_stage->stage == skr_stage_pixel) {
		ref_meta->ops_pixel = ops;
	}

	_sksc_spirv_scan_spec_constants(spirv_stage, ref_meta);

	spvReflectDestroyShaderModule(&module);
	return true;
}

///////////////////////////////////////////

void parse_semantic(const char* str, char* out_str, size_t out_size, int32_t* out_idx) {
	const char *curr  = str;
	char*       write = out_str;
	char*       end   = out_str + out_size - 1;
	int         idx   = 0;
	while (*curr != 0) {
		if (*curr>='0' && *curr<='9') {
			idx  = idx * 10;
			idx += (*curr) - '0';
		} else if (write < end) {
			*write = *curr;
			write++;
		}
		curr++;
	}
	*write   = '\0';
	*out_idx = idx;
}

///////////////////////////////////////////

int strcmp_nocase(char const *a, char const *b) {
	for (;; a++, b++) {
		int d = tolower((unsigned char)*a) - tolower((unsigned char)*b);
		if (d != 0 || !*a)
			return d;
	}
}

///////////////////////////////////////////

int64_t mini(int64_t a, int64_t b) {return a<b?a:b;}
int64_t maxi(int64_t a, int64_t b) {return a>b?a:b;}

///////////////////////////////////////////

array_t<sksc_meta_item_t> sksc_meta_find_defaults(const char *hlsl_text) {
	// Searches for metadata in comments that look like this:
	//--name                 = unlit/test
	//--time: color          = 1,1,1,1
	//--tex: 2D, external    = white
	//--uv_scale: range(0,2) = 0.5
	// Where --name is a unique keyword indicating the shader's name, and
	// other elements follow the pattern of:
	// |indicator|param name|tag separator|tag string|default separator|comma separated default values
	//  --        time       :             color      =                 1,1,1,1
	// Metadata can be in // as well as /**/ comments

	array_t<sksc_meta_item_t> items = {};

	// This function will get each line of comment from the file
	const char *(*next_comment)(const char *src, const char **ref_end, bool *ref_state) = [](const char *src, const char **ref_end, bool *ref_state) {
		const char *c      = *ref_end == nullptr ? src : *ref_end;
		const char *result = nullptr;

		// If we're inside a /**/ block, continue from the previous line, we
		// just need to skip any newline characters at the end.
		if (*ref_state) {
			result = (*ref_end)+1;
			while (*result == '\n' || *result == '\r') result++;
		}
		
		// Search for the start of a comment, if we don't have one already.
		while (*c != '\0' && result == nullptr) {
			if (*c == '/' && (*(c+1) == '/' || *(c+1) == '*')) {
				result = (char*)(c+2);
				*ref_state = *(c + 1) == '*';
			}
			c++;
		}

		// Find the end of this comment line.
		c = result;
		while (c != nullptr && *c != '\0' && *c != '\n' && *c != '\r') {
			if (*ref_state && *c == '*' && *(c+1) == '/') {
				*ref_state = false;
				break;
			}
			c++;
		}
		*ref_end = c;

		return result;
	};

	// This function checks if the line is relevant for our metadata
	const char *(*is_relevant)(const char *start, const char *end) = [](const char *start, const char *end) {
		const char *c = start;
		while (c != end && (*c == ' ' || *c == '\t')) c++;

		return end - c > 1 && c[0] == '-' && c[1] == '-' 
			? &c[2] 
			: (char*)nullptr;
	};

	void (*trim_str)(const char **ref_start, const char **ref_end) = [] (const char **ref_start, const char **ref_end){
		while (**ref_start   == ' ' || **ref_start   == '\t') (*ref_start)++;
		while (*(*ref_end-1) == ' ' || *(*ref_end-1) == '\t') (*ref_end)--;
	};

	const char *(*index_of)(const char *start, const char *end, char ch) = [](const char *start, const char *end, char ch) {
		while (start != end) {
			if (*start == ch)
				return start;
			start++;
		}
		return (const char*)nullptr;
	};

	bool        in_comment  = false;
	const char *comment_end = nullptr;
	const char *comment     = next_comment(hlsl_text, &comment_end, &in_comment);
	while (comment) {
		comment = is_relevant(comment, comment_end);
		if (comment) {
			const char *tag_str   = index_of(comment, comment_end, ':');
			const char *value_str = index_of(comment, comment_end, '=');

			const char *name_start = comment;
			const char *name_end   = tag_str?tag_str:(value_str?value_str:comment_end);
			trim_str(&name_start, &name_end);
			char name[32];
			int64_t ct = name_end - name_start;
			memcpy(name, name_start, mini(sizeof(name), ct));
			name[ct] = '\0';

			char tag[64]; tag[0] = '\0';
			if (tag_str) {
				const char *tag_start = tag_str + 1;
				const char *tag_end   = value_str ? value_str : comment_end;
				trim_str(&tag_start, &tag_end);
				ct = maxi(0, tag_end - tag_start);
				memcpy(tag, tag_start, mini(sizeof(tag), ct));
				tag[ct] = '\0';
			}

			char value[512]; value[0] = '\0';
			if (value_str) {
				const char *value_start = value_str + 1;
				const char *value_end   = comment_end;
				trim_str(&value_start, &value_end);
				ct = maxi(0, value_end - value_start);
				memcpy(value, value_start, mini(sizeof(value), ct));
				value[ct] = '\0';
			}

			sksc_meta_item_t item = {};
			sksc_line_col(hlsl_text, comment, &item.row, &item.col);
			strncpy(item.name,  name,  sizeof(item.name));
			strncpy(item.tag,   tag,   sizeof(item.tag));
			strncpy(item.value, value, sizeof(item.value));
			items.add(item);

			if (tag[0] == '\0' && value[0] == '\0') {
				sksc_log_at(sksc_log_level_warn, item.row, item.col, "Shader var data for '%s' has no tag or value, missing a ':' or '='?", name);
			}
		}
		comment = next_comment(hlsl_text, &comment_end, &in_comment);
	}
	return items;
}

///////////////////////////////////////////

static void _sksc_write_var_default(sksc_shader_buffer_t *buff, sksc_shader_var_t *var, double *values, int32_t value_count) {
	if (buff->defaults == nullptr) {
		buff->defaults = malloc(buff->size);
		memset(buff->defaults, 0, buff->size);
	}

	uint8_t *write_at = ((uint8_t *)buff->defaults) + var->offset;
	int32_t  count    = value_count < var->type_count ? value_count : var->type_count;

	for (int32_t i = 0; i < count; i++) {
		double d = values[i];
		switch (var->type) {
		case sksc_shader_var_float:  { float    val = (float   )d; memcpy(write_at, &val, sizeof(val)); write_at += sizeof(val); } break;
		case sksc_shader_var_double: { double   val =           d; memcpy(write_at, &val, sizeof(val)); write_at += sizeof(val); } break;
		case sksc_shader_var_int:    { int32_t  val = (int32_t )d; memcpy(write_at, &val, sizeof(val)); write_at += sizeof(val); } break;
		case sksc_shader_var_uint:   { uint32_t val = (uint32_t)d; memcpy(write_at, &val, sizeof(val)); write_at += sizeof(val); } break;
		case sksc_shader_var_uint8:  { uint8_t  val = (uint8_t )d; memcpy(write_at, &val, sizeof(val)); write_at += sizeof(val); } break;
		default: break;
		}
	}
}

///////////////////////////////////////////

void sksc_meta_assign_defaults(array_t<sksc_ast_default_t> ast_defaults, array_t<sksc_meta_item_t> comment_overrides, sksc_shader_meta_t *ref_meta) {
	sksc_shader_buffer_t *buff = ref_meta->global_buffer_id == -1 ? nullptr : &ref_meta->buffers[ref_meta->global_buffer_id];

	// First, apply AST defaults (actual HLSL initializers)
	for (size_t i = 0; i < ast_defaults.count; i++) {
		sksc_ast_default_t *ast = &ast_defaults[i];

		for (size_t v = 0; buff && v < buff->var_count; v++) {
			if (strcmp(buff->vars[v].name, ast->name) != 0) continue;

			if (buff->vars[v].type == sksc_shader_var_none) continue;

			_sksc_write_var_default(buff, &buff->vars[v], ast->values, ast->value_count);
			break;
		}
	}

	// Then apply comment overrides (//--name: tag = value)
	// These can add extra metadata (tags) and override AST default values
	int32_t(*count_ch)(const char *str, char ch) = [](const char *str, char ch) {
		const char *c      = str;
		int32_t     result = 0;
		while (*c != '\0') {
			if (*c == ch) result++;
			c++;
		}
		return result;
	};

	for (size_t i = 0; i < comment_overrides.count; i++) {
		sksc_meta_item_t *item  = &comment_overrides[i];
		int32_t           found = 0;

		for (size_t v = 0; buff && v < buff->var_count; v++) {
			if (strcmp(buff->vars[v].name, item->name) != 0) continue;

			found += 1;
			strncpy(buff->vars[v].extra, item->tag, sizeof(buff->vars[v].extra));

			// If no value specified, keep the AST default (if any)
			if (item->value[0] == '\0') break;

			int32_t commas = count_ch(item->value, ',');

			if (buff->vars[v].type == sksc_shader_var_none) {
				sksc_log_at(sksc_log_level_warn, item->row, item->col, "Can't set default for --%s, unimplemented type", item->name);
			} else if (commas + 1 != buff->vars[v].type_count) {
				sksc_log_at(sksc_log_level_warn, item->row, item->col, "Default value for --%s has an incorrect number of arguments", item->name);
			} else {
				// Parse comment values into double array and write
				double values[16];
				int32_t value_count = 0;

				char *start = item->value;
				char *end   = strchr(start, ',');
				char  param[64];
				for (int32_t c = 0; c <= commas && value_count < 16; c++) {
					int32_t length = (int32_t)(end == nullptr ? mini(sizeof(param)-1, strlen(start)) : end - start);
					memcpy(param, start, mini(sizeof(param), length));
					param[length] = '\0';

					values[value_count++] = atof(param);

					if (end != nullptr) {
						start = end + 1;
						end   = strchr(start, ',');
					}
				}

				_sksc_write_var_default(buff, &buff->vars[v], values, value_count);
			}
			break;
		}

		for (size_t r = 0; r < ref_meta->resource_count; r++) {
			if (strcmp(ref_meta->resources[r].name, item->name) != 0) continue;
			found += 1;

			strncpy(ref_meta->resources[r].tags,  item->tag,   sizeof(ref_meta->resources[r].tags ));
			strncpy(ref_meta->resources[r].value, item->value, sizeof(ref_meta->resources[r].value));
			break;
		}

		if (strcmp(item->name, "name") == 0) {
			found += 1;
			strncpy(ref_meta->name, item->value, sizeof(ref_meta->name));
		}

		if (strcmp(item->name, "wave_size") == 0) {
			found += 1;
			int32_t size = atoi(item->value);
			if (size < 4 || size > 128 || (size & (size - 1)) != 0) {
				sksc_log_at(sksc_log_level_warn, item->row, item->col, "wave_size must be a power of two between 4 and 128, got %d", size);
			} else {
				ref_meta->wave_size = (uint32_t)size;
			}
		}

		if (found != 1) {
			sksc_log_at(sksc_log_level_warn, item->row, item->col, "Can't find shader var named '%s'", item->name);
		}
	}
}

///////////////////////////////////////////

bool sksc_meta_check_dup_buffers(const sksc_shader_meta_t *ref_meta) {
	for (size_t i = 0; i < ref_meta->buffer_count; i++) {
		for (size_t t = 0; t < ref_meta->buffer_count; t++) {
			if (i == t) continue;
			if (ref_meta->buffers[i].bind.slot == ref_meta->buffers[t].bind.slot &&
			    ref_meta->buffers[i].space     == ref_meta->buffers[t].space) {
				return false;
			}
		}
	}
	return true;
}

///////////////////////////////////////////

bool sksc_meta_check_dup_resources(const sksc_shader_meta_t *ref_meta, const char **out_name1, const char **out_name2, uint32_t *out_slot) {
	for (size_t i = 0; i < ref_meta->resource_count; i++) {
		for (size_t t = i + 1; t < ref_meta->resource_count; t++) {
			// Check if slots match and they're the same register type (both textures, both storage, etc.)
			if (ref_meta->resources[i].bind.slot          == ref_meta->resources[t].bind.slot &&
			    ref_meta->resources[i].bind.register_type == ref_meta->resources[t].bind.register_type) {
				if (out_name1) *out_name1 = ref_meta->resources[i].name;
				if (out_name2) *out_name2 = ref_meta->resources[t].name;
				if (out_slot)  *out_slot  = ref_meta->resources[i].bind.slot;
				return false;
			}
		}
	}
	return true;
}

///////////////////////////////////////////

void sksc_line_col(const char *from_text, const char *at, int32_t *out_line, int32_t *out_column) {
	if (out_line  ) *out_line   = -1;
	if (out_column) *out_column = -1;

	bool found = false;
	const char *curr = from_text;
	int32_t line = 0, col = 0;
	while (*curr != '\0') {
		if (*curr == '\n') { line++; col = 0; } 
		else if (*curr != '\r') col++;
		if (curr == at) {
			found = true;
			break;
		}
		curr++;
	}

	if (found) {
		if (out_line  ) *out_line   = line+1;
		if (out_column) *out_column = col;
	}
}