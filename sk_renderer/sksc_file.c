// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "sksc_file.h"
#include "smolv.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////

// FNV hash
uint64_t skr_hash(const char *string) {
	uint64_t hash = 14695981039346656037UL;
	while (*string != '\0') {
		hash = (hash ^ *string) * 1099511628211;
		string++;
	}
	return hash;
}

///////////////////////////////////////////////////////////////////////////////

bool sksc_shader_file_verify(const void *data, uint32_t size, uint16_t *out_version, char *out_name, uint32_t out_name_size) {
	const char    *prefix  = "SKSHADER";
	const uint8_t *bytes   = (uint8_t*)data;

	// check the magic bytes to see if this is a SKS shader file
	if (size < 10 || memcmp(bytes, prefix, 8) != 0)
		return false;

	// Grab the file version
	if (out_version)
		memcpy(out_version, &bytes[8], sizeof(uint16_t));

	// And grab the name of the shader
	if (out_name != NULL && out_name_size > 0) {
		memcpy(out_name, &bytes[14], out_name_size < 256 ? out_name_size : 256);
		out_name[out_name_size - 1] = '\0';
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////

// Returns false on a truncated or malformed meta section. Every read is
// bounds-checked, and the cursor is 64-bit so a corrupt count can't wrap past
// the check. Partial output is left for the caller's cleanup to free.
static bool _sksc_load_meta(const uint8_t *bytes, uint32_t size, uint32_t *ref_at, sksc_shader_meta_t *meta) {
	#define _SKSC_TAKE(count) if ((uint64_t)at + (uint64_t)(count) > size) return false
	uint64_t at = *ref_at;
	_SKSC_TAKE(sizeof(meta->name               )); memcpy( meta->name,                &bytes[at], sizeof(meta->name               )); at += sizeof(meta->name);
	_SKSC_TAKE(sizeof(meta->buffer_count       )); memcpy(&meta->buffer_count,        &bytes[at], sizeof(meta->buffer_count       )); at += sizeof(meta->buffer_count);
	_SKSC_TAKE(sizeof(meta->resource_count     )); memcpy(&meta->resource_count,      &bytes[at], sizeof(meta->resource_count     )); at += sizeof(meta->resource_count);
	_SKSC_TAKE(sizeof(meta->vertex_input_count )); memcpy(&meta->vertex_input_count,  &bytes[at], sizeof(meta->vertex_input_count )); at += sizeof(meta->vertex_input_count);
	_SKSC_TAKE(sizeof(meta->spec_constant_count)); memcpy(&meta->spec_constant_count, &bytes[at], sizeof(meta->spec_constant_count)); at += sizeof(meta->spec_constant_count);
	_SKSC_TAKE(sizeof(meta->sampler_count      )); memcpy(&meta->sampler_count,       &bytes[at], sizeof(meta->sampler_count      )); at += sizeof(meta->sampler_count);
	_SKSC_TAKE(sizeof(meta->features)); memcpy(&meta->features,            &bytes[at], sizeof(meta->features));            at += sizeof(meta->features);
	_SKSC_TAKE(sizeof(meta->features_reserved)); memcpy(&meta->features_reserved,   &bytes[at], sizeof(meta->features_reserved));   at += sizeof(meta->features_reserved);

	_SKSC_TAKE(sizeof(meta->ops_vertex.total)); memcpy(&meta->ops_vertex.total,        &bytes[at], sizeof(meta->ops_vertex.total));        at += sizeof(meta->ops_vertex.total);
	_SKSC_TAKE(sizeof(meta->ops_vertex.tex_read)); memcpy(&meta->ops_vertex.tex_read,     &bytes[at], sizeof(meta->ops_vertex.tex_read));     at += sizeof(meta->ops_vertex.tex_read);
	_SKSC_TAKE(sizeof(meta->ops_vertex.dynamic_flow)); memcpy(&meta->ops_vertex.dynamic_flow, &bytes[at], sizeof(meta->ops_vertex.dynamic_flow)); at += sizeof(meta->ops_vertex.dynamic_flow);
	_SKSC_TAKE(sizeof(meta->ops_pixel.total)); memcpy(&meta->ops_pixel.total,         &bytes[at], sizeof(meta->ops_pixel.total));         at += sizeof(meta->ops_pixel.total);
	_SKSC_TAKE(sizeof(meta->ops_pixel.tex_read)); memcpy(&meta->ops_pixel.tex_read,      &bytes[at], sizeof(meta->ops_pixel.tex_read));      at += sizeof(meta->ops_pixel.tex_read);
	_SKSC_TAKE(sizeof(meta->ops_pixel.dynamic_flow)); memcpy(&meta->ops_pixel.dynamic_flow,  &bytes[at], sizeof(meta->ops_pixel.dynamic_flow));  at += sizeof(meta->ops_pixel.dynamic_flow);
	_SKSC_TAKE(sizeof(meta->wave_size)); memcpy(&meta->wave_size,               &bytes[at], sizeof(meta->wave_size));               at += sizeof(meta->wave_size);
	_SKSC_TAKE(sizeof(meta->tile_apron)); memcpy(&meta->tile_apron,              &bytes[at], sizeof(meta->tile_apron));              at += sizeof(meta->tile_apron);

	// Every element costs at least a byte on disk, so a count past the file size
	// is corrupt. Cheaper to reject here than to try allocating for it.
	if (meta->buffer_count        > size || meta->resource_count > size ||
		(uint32_t)meta->vertex_input_count > size || meta->spec_constant_count > size ||
		meta->sampler_count       > size || meta->vertex_input_count < 0) return false;

	// --- Pass 1: scan buffer section to accumulate var/defaults totals ---
	uint64_t buffer_section_start = at;
	uint32_t total_var_count      = 0;
	uint64_t total_defaults_size  = 0;

	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		_SKSC_TAKE(sizeof(((sksc_shader_buffer_t*)0)->name)  +
		           sizeof(((sksc_shader_buffer_t*)0)->space) +
		           sizeof(((sksc_shader_buffer_t*)0)->bind));
		at += sizeof(((sksc_shader_buffer_t*)0)->name)  +
		      sizeof(((sksc_shader_buffer_t*)0)->space) +
		      sizeof(((sksc_shader_buffer_t*)0)->bind);

		uint32_t buffer_size, var_count, default_size;
		_SKSC_TAKE(sizeof(buffer_size)); memcpy(&buffer_size, &bytes[at], sizeof(buffer_size)); at += sizeof(buffer_size);
		_SKSC_TAKE(sizeof(var_count)); memcpy(&var_count,   &bytes[at], sizeof(var_count));   at += sizeof(var_count);
		_SKSC_TAKE(sizeof(default_size)); memcpy(&default_size,&bytes[at], sizeof(default_size));at += sizeof(default_size);

		total_var_count += var_count;
		if (default_size != 0) {
			if (default_size > buffer_size) default_size = buffer_size;
			// skshaderc always stores a buffer's defaults in full, so the space
			// they reserve can't legitimately outgrow the file holding them
			total_defaults_size += buffer_size;
			if (total_defaults_size > size) return false;
			_SKSC_TAKE(default_size);
			at += default_size;
		}
		_SKSC_TAKE((uint64_t)var_count * (
			sizeof(((sksc_shader_var_t*)0)->name)      + sizeof(((sksc_shader_var_t*)0)->extra)    +
			sizeof(((sksc_shader_var_t*)0)->type_name)  + sizeof(((sksc_shader_var_t*)0)->offset)   +
			sizeof(((sksc_shader_var_t*)0)->size)       + sizeof(((sksc_shader_var_t*)0)->type)     +
			sizeof(((sksc_shader_var_t*)0)->type_count)));
		at += (uint64_t)var_count * (
			sizeof(((sksc_shader_var_t*)0)->name)      + sizeof(((sksc_shader_var_t*)0)->extra)    +
			sizeof(((sksc_shader_var_t*)0)->type_name)  + sizeof(((sksc_shader_var_t*)0)->offset)   +
			sizeof(((sksc_shader_var_t*)0)->size)       + sizeof(((sksc_shader_var_t*)0)->type)     +
			sizeof(((sksc_shader_var_t*)0)->type_count));
	}

	// --- Single allocation for all sub-arrays ---
	size_t buffers_size  = sizeof(sksc_shader_buffer_t       ) * meta->buffer_count;
	size_t res_size      = sizeof(sksc_shader_resource_t     ) * meta->resource_count;
	size_t vars_size     = sizeof(sksc_shader_var_t          ) * total_var_count;
	size_t vinputs_size  = sizeof(skr_vert_component_t       ) * meta->vertex_input_count;
	size_t spec_size     = sizeof(sksc_shader_spec_constant_t) * meta->spec_constant_count;
	size_t sampler_size  = sizeof(sksc_shader_sampler_t      ) * meta->sampler_count;
	// The defaults are raw bytes of arbitrary length, so pad them out to keep the
	// structs carved after them aligned
	size_t defaults_size = ((size_t)total_defaults_size + 7) & ~(size_t)7;
	size_t total_size    = buffers_size + res_size + vars_size + defaults_size + vinputs_size + spec_size + sampler_size;

	if (total_size == 0) {
		meta->buffers        = NULL;
		meta->resources      = NULL;
		meta->vertex_inputs  = NULL;
		meta->spec_constants = NULL;
		meta->samplers       = NULL;
		*ref_at = (uint32_t)at;
		return true;
	}

	uint8_t *block = (uint8_t *)calloc(1, total_size);
	if (block == NULL) return false;

	meta->buffers        = (sksc_shader_buffer_t       *)(block);
	meta->resources      = (sksc_shader_resource_t     *)(block + buffers_size);
	meta->vertex_inputs  = (skr_vert_component_t       *)(block + buffers_size + res_size + vars_size + defaults_size);
	meta->spec_constants = (sksc_shader_spec_constant_t*)(block + buffers_size + res_size + vars_size + defaults_size + vinputs_size);
	meta->samplers       = (sksc_shader_sampler_t      *)(block + buffers_size + res_size + vars_size + defaults_size + vinputs_size + spec_size);
	uint8_t *vars_cursor     = block + buffers_size + res_size;
	uint8_t *defaults_cursor = block + buffers_size + res_size + vars_size;

	// --- Pass 2: rewind and read buffer data ---
	at = buffer_section_start;
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		sksc_shader_buffer_t *buffer = &meta->buffers[i];
		_SKSC_TAKE(sizeof(buffer->name)); memcpy( buffer->name,      &bytes[at], sizeof(buffer->name));      at += sizeof(buffer->name);
		_SKSC_TAKE(sizeof(buffer->space)); memcpy(&buffer->space,     &bytes[at], sizeof(buffer->space));     at += sizeof(buffer->space);
		_SKSC_TAKE(sizeof(buffer->bind)); memcpy(&buffer->bind,      &bytes[at], sizeof(buffer->bind));      at += sizeof(buffer->bind);
		_SKSC_TAKE(sizeof(buffer->size)); memcpy(&buffer->size,      &bytes[at], sizeof(buffer->size));      at += sizeof(buffer->size);
		_SKSC_TAKE(sizeof(buffer->var_count)); memcpy(&buffer->var_count, &bytes[at], sizeof(buffer->var_count)); at += sizeof(buffer->var_count);

		uint32_t default_size = 0;
		_SKSC_TAKE(sizeof(buffer->size)); memcpy(&default_size, &bytes[at], sizeof(buffer->size)); at += sizeof(buffer->size);
		buffer->defaults = NULL;
		if (default_size != 0) {
			if (default_size > buffer->size) default_size = buffer->size;
			buffer->defaults = defaults_cursor;
				_SKSC_TAKE(default_size);
			memcpy(buffer->defaults, &bytes[at], default_size); at += default_size;
			defaults_cursor += buffer->size;
		}

		if (buffer->var_count > 0) {
			buffer->vars = (sksc_shader_var_t *)vars_cursor;
			vars_cursor += sizeof(sksc_shader_var_t) * buffer->var_count;
		}
		buffer->name_hash = skr_hash(buffer->name);

		for (uint32_t t = 0; t < buffer->var_count; t++) {
			sksc_shader_var_t *var = &buffer->vars[t];
			_SKSC_TAKE(sizeof(var->name     )); memcpy( var->name,       &bytes[at], sizeof(var->name     )); at += sizeof(var->name     );
			_SKSC_TAKE(sizeof(var->extra    )); memcpy( var->extra,      &bytes[at], sizeof(var->extra    )); at += sizeof(var->extra    );
			_SKSC_TAKE(sizeof(var->type_name)); memcpy( var->type_name,  &bytes[at], sizeof(var->type_name)); at += sizeof(var->type_name);
			_SKSC_TAKE(sizeof(var->offset   )); memcpy(&var->offset,     &bytes[at], sizeof(var->offset   )); at += sizeof(var->offset   );
			_SKSC_TAKE(sizeof(var->size     )); memcpy(&var->size,       &bytes[at], sizeof(var->size     )); at += sizeof(var->size     );
			_SKSC_TAKE(sizeof(var->type     )); memcpy(&var->type,       &bytes[at], sizeof(var->type     )); at += sizeof(var->type     );
			_SKSC_TAKE(sizeof(var->type_count)); memcpy(&var->type_count, &bytes[at], sizeof(var->type_count)); at += sizeof(var->type_count);
			var->name_hash = skr_hash(var->name);
		}

		if (strcmp(buffer->name, "$Global") == 0)
			meta->global_buffer_id = (int32_t)i;
	}

	for (int32_t i = 0; i < meta->vertex_input_count; i++) {
		skr_vert_component_t *com = &meta->vertex_inputs[i];
		_SKSC_TAKE(sizeof(com->format       )); memcpy(&com->format,        &bytes[at], sizeof(com->format       )); at += sizeof(com->format);
		_SKSC_TAKE(sizeof(com->count        )); memcpy(&com->count,         &bytes[at], sizeof(com->count        )); at += sizeof(com->count);
		_SKSC_TAKE(sizeof(com->semantic     )); memcpy(&com->semantic,      &bytes[at], sizeof(com->semantic     )); at += sizeof(com->semantic);
		_SKSC_TAKE(sizeof(com->semantic_slot)); memcpy(&com->semantic_slot, &bytes[at], sizeof(com->semantic_slot)); at += sizeof(com->semantic_slot);
		_SKSC_TAKE(sizeof(com->location     )); memcpy(&com->location,      &bytes[at], sizeof(com->location     )); at += sizeof(com->location);
	}

	for (uint32_t i = 0; i < meta->resource_count; i++) {
		sksc_shader_resource_t *res = &meta->resources[i];
		_SKSC_TAKE(sizeof(res->name        )); memcpy( res->name,         &bytes[at], sizeof(res->name        )); at += sizeof(res->name        );
		_SKSC_TAKE(sizeof(res->value       )); memcpy( res->value,        &bytes[at], sizeof(res->value       )); at += sizeof(res->value       );
		_SKSC_TAKE(sizeof(res->tags        )); memcpy( res->tags,         &bytes[at], sizeof(res->tags        )); at += sizeof(res->tags        );
		_SKSC_TAKE(sizeof(res->bind        )); memcpy(&res->bind,         &bytes[at], sizeof(res->bind        )); at += sizeof(res->bind        );
		_SKSC_TAKE(sizeof(res->element_size)); memcpy(&res->element_size, &bytes[at], sizeof(res->element_size)); at += sizeof(res->element_size);
		_SKSC_TAKE(sizeof(res->shape       )); memcpy(&res->shape,        &bytes[at], sizeof(res->shape       )); at += sizeof(res->shape       );
		_SKSC_TAKE(sizeof(res->image_format)); memcpy(&res->image_format, &bytes[at], sizeof(res->image_format)); at += sizeof(res->image_format);
		_SKSC_TAKE(2);
		at += 2; // reserved
		res->name_hash = skr_hash(res->name);
	}

	for (uint32_t i = 0; i < meta->spec_constant_count; i++) {
		sksc_shader_spec_constant_t *spec = &meta->spec_constants[i];
		_SKSC_TAKE(sizeof(spec->name         )); memcpy( spec->name,          &bytes[at], sizeof(spec->name         )); at += sizeof(spec->name         );
		_SKSC_TAKE(sizeof(spec->constant_id  )); memcpy(&spec->constant_id,   &bytes[at], sizeof(spec->constant_id  )); at += sizeof(spec->constant_id  );
		_SKSC_TAKE(sizeof(spec->default_value)); memcpy(&spec->default_value, &bytes[at], sizeof(spec->default_value)); at += sizeof(spec->default_value);
		_SKSC_TAKE(sizeof(spec->type         )); memcpy(&spec->type,          &bytes[at], sizeof(spec->type         )); at += sizeof(spec->type         );
		_SKSC_TAKE(sizeof(spec->stage_bits   )); memcpy(&spec->stage_bits,    &bytes[at], sizeof(spec->stage_bits   )); at += sizeof(spec->stage_bits   );
		spec->name_hash = skr_hash(spec->name);
	}

	for (uint32_t i = 0; i < meta->sampler_count; i++) {
		sksc_shader_sampler_t *sampler = &meta->samplers[i];
		_SKSC_TAKE(sizeof(sampler->name       )); memcpy( sampler->name,        &bytes[at], sizeof(sampler->name       )); at += sizeof(sampler->name       );
		_SKSC_TAKE(sizeof(sampler->slot       )); memcpy(&sampler->slot,        &bytes[at], sizeof(sampler->slot       )); at += sizeof(sampler->slot       );
		_SKSC_TAKE(sizeof(sampler->stage_bits )); memcpy(&sampler->stage_bits,  &bytes[at], sizeof(sampler->stage_bits )); at += sizeof(sampler->stage_bits );
		_SKSC_TAKE(sizeof(sampler->paired_slot)); memcpy(&sampler->paired_slot, &bytes[at], sizeof(sampler->paired_slot)); at += sizeof(sampler->paired_slot);
		sampler->name_hash = skr_hash(sampler->name);
	}

	*ref_at = (uint32_t)at;
	return true;
	#undef _SKSC_TAKE
}

///////////////////////////////////////////////////////////////////////////////

sksc_result_ sksc_shader_file_load_memory(const void *data, uint32_t size, sksc_shader_file_t *out_file) {
	uint16_t file_version = 0;
	if (!sksc_shader_file_verify(data, size, &file_version, NULL, 0)) return sksc_result_bad_format;
	if (file_version != SKSC_FILE_VERSION)                            return sksc_result_old_version; // exact match only — no .sks back-compat by policy

	const uint8_t *bytes  = (uint8_t*)data;
	sksc_result_   result = sksc_result_corrupt_data;
	uint32_t       at     = 10;

	// Zeroed up front so the failure path can always destroy a partial load
	*out_file = (sksc_shader_file_t){0};
	out_file->meta.global_buffer_id = -1;

	// Every read past here is bounds-checked against the file, since a truncated
	// or malformed .sks otherwise walks straight off the end of the buffer.
	#define _SKSC_NEED(count) if ((uint64_t)at + (count) > size) goto fail

	// A stage costs at least its four header fields on disk, so a count past
	// that is corrupt and must not reach the allocator
	uint32_t stage_count = 0;
	_SKSC_NEED(sizeof(stage_count));
	memcpy(&stage_count, &bytes[at], sizeof(stage_count)); at += sizeof(stage_count);
	if ((uint64_t)stage_count * 16 > size) goto fail;

	if (stage_count > 0) {
		out_file->stages = (sksc_shader_file_stage_t*)calloc(stage_count, sizeof(sksc_shader_file_stage_t));
		if (out_file->stages == NULL) { result = sksc_result_out_of_memory; goto fail; }
		out_file->stage_count = stage_count;
	}

	if (!_sksc_load_meta(bytes, size, &at, &out_file->meta)) goto fail;

	for (uint32_t i = 0; i < out_file->stage_count; i++) {
		sksc_shader_file_stage_t *stage = &out_file->stages[i];
		_SKSC_NEED(sizeof(stage->language) + sizeof(stage->stage) + sizeof(stage->wave_size) + sizeof(stage->code_size));
		memcpy(&stage->language,  &bytes[at], sizeof(stage->language));  at += sizeof(stage->language);
		memcpy(&stage->stage,     &bytes[at], sizeof(stage->stage));     at += sizeof(stage->stage);
		memcpy(&stage->wave_size, &bytes[at], sizeof(stage->wave_size)); at += sizeof(stage->wave_size);
		memcpy(&stage->code_size, &bytes[at], sizeof(stage->code_size)); at += sizeof(stage->code_size);

		stage->code = NULL;
		if (stage->code_size > 0) {
			_SKSC_NEED(stage->code_size);

			// The blob's own magic says whether it's SMOL-V, so callers get SPIR-V
			// back either way. Anything else is refused rather than passed on.
			bool is_smolv = stage->language == skr_shader_lang_spirv && smolv_is_smolv(&bytes[at], stage->code_size);
			if (stage->language == skr_shader_lang_spirv && !is_smolv &&
				(stage->code_size < 4 || memcmp(&bytes[at], "\x03\x02\x23\x07", 4) != 0))
				goto fail;

			if (is_smolv) {
				uint32_t spirv_size = (uint32_t)smolv_decoded_size(&bytes[at], stage->code_size);
				if (spirv_size == 0) goto fail;

				void *spirv = malloc(spirv_size);
				if (spirv == NULL) { result = sksc_result_out_of_memory; goto fail; }
				stage->code = spirv;
				if (!smolv_decode(&bytes[at], stage->code_size, spirv, spirv_size)) goto fail;
				at += stage->code_size;

				stage->code_size = spirv_size;
			} else {
				stage->code = malloc(stage->code_size);
				if (stage->code == NULL) { result = sksc_result_out_of_memory; goto fail; }
				memcpy(stage->code, &bytes[at], stage->code_size); at += stage->code_size;
			}
		}
	}

	#undef _SKSC_NEED
	return sksc_result_success;

fail:
	sksc_shader_file_destroy(out_file);
	return result;
}

///////////////////////////////////////////////////////////////////////////////

void sksc_shader_file_destroy(sksc_shader_file_t *ref_file) {
	for (uint32_t i = 0; i < ref_file->stage_count; i++) {
		free(ref_file->stages[i].code);
	}
	free(ref_file->stages);
	sksc_shader_meta_free(&ref_file->meta);
	*ref_file = (sksc_shader_file_t){0};
}

///////////////////////////////////////////////////////////////////////////////
// skr_shader_meta_t
///////////////////////////////////////////////////////////////////////////////

skr_bind_t sksc_shader_meta_get_bind(const sksc_shader_meta_t *meta, const char *name) {
	if (name == NULL) return (skr_bind_t){0};
	uint64_t hash = skr_hash(name);
	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		if (meta->buffers[i].name_hash == hash)
			return meta->buffers[i].bind;
	}
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		if (meta->resources[i].name_hash == hash)
			return meta->resources[i].bind;
	}
	return (skr_bind_t){0};
}

///////////////////////////////////////////////////////////////////////////////

sksc_pass_inputs_t sksc_shader_meta_pass_inputs(const sksc_shader_meta_t *meta) {
	sksc_pass_inputs_t result = {0};
	for (uint32_t i = 0; i < meta->resource_count; i++) {
		const sksc_shader_resource_t* res = &meta->resources[i];
		bool is_color = strcmp(res->name, "color") == 0;
		bool is_depth = strcmp(res->name, "depth") == 0;
		if (res->bind.register_type == skr_register_input_attachment) {
			if      (is_color)              result.input_color = true;
			else if (is_depth && !result.input_depth) {
				result.input_depth    = true;
				result.input_depth_ms = (res->shape & SKSC_SHAPE_MS) != 0;
			}
		} else if (res->bind.register_type == skr_register_tile_sampled && is_color) {
			result.tile_color = true;
		}
	}
	return result;
}

///////////////////////////////////////////////////////////////////////////////

uint64_t sksc_shader_meta_missing_features(const sksc_shader_meta_t *meta, uint64_t enabled_features) {
	if (meta == NULL) return 0;
	// Every requirement bit the shader declared that the capability mask doesn't
	// advertise. The compiler sets sksc_feature_bit_unknown when it saw a
	// capability it couldn't classify; a capability mask must never claim that
	// bit, so an unverifiable requirement always surfaces here.
	return meta->features & ~enabled_features;
}

///////////////////////////////////////////////////////////////////////////////

int32_t sksc_shader_meta_get_var_count(const sksc_shader_meta_t *meta) {
	return meta->global_buffer_id != -1
		? (int32_t)meta->buffers[meta->global_buffer_id].var_count
		: 0;
}

///////////////////////////////////////////////////////////////////////////////

int32_t sksc_shader_meta_get_var_index(const sksc_shader_meta_t *meta, const char *name) {
	return sksc_shader_meta_get_var_index_h(meta, skr_hash(name));
}

///////////////////////////////////////////////////////////////////////////////

int32_t sksc_shader_meta_get_var_index_h(const sksc_shader_meta_t *meta, uint64_t name_hash) {
	if (meta->global_buffer_id == -1) return -1;

	sksc_shader_buffer_t *buffer = &meta->buffers[meta->global_buffer_id];
	for (uint32_t i = 0; i < buffer->var_count; i++) {
		if (buffer->vars[i].name_hash == name_hash) {
			return (int32_t)i;
		}
	}
	return -1;
}

///////////////////////////////////////////////////////////////////////////////

const sksc_shader_var_t *sksc_shader_meta_get_var_info(const sksc_shader_meta_t *meta, int32_t var_index) {
	if (meta->global_buffer_id == -1 || var_index == -1) return NULL;

	sksc_shader_buffer_t *buffer = &meta->buffers[meta->global_buffer_id];
	return &buffer->vars[var_index];
}

///////////////////////////////////////////////////////////////////////////////

void sksc_shader_meta_free(sksc_shader_meta_t *ref_meta) {
	if (!ref_meta) return;
	// All sub-arrays (resources, vertex_inputs, per-buffer vars/defaults)
	// are carved from the single allocation starting at buffers. The compiler
	// paths can grow samplers as a standalone allocation, marked by the flag.
	if (ref_meta->samplers_owned) free(ref_meta->samplers);
	free(ref_meta->buffers);
	*ref_meta = (sksc_shader_meta_t){0};
}
