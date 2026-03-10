// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#include "sksc_file.h"

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

	// check the first 5 bytes to see if this is a SKS shader file
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

static uint32_t _sksc_load_meta(const uint8_t *bytes, uint32_t at, sksc_shader_meta_t *meta) {
	memcpy( meta->name,               &bytes[at], sizeof(meta->name              )); at += sizeof(meta->name);
	memcpy(&meta->buffer_count,       &bytes[at], sizeof(meta->buffer_count      )); at += sizeof(meta->buffer_count);
	memcpy(&meta->resource_count,     &bytes[at], sizeof(meta->resource_count    )); at += sizeof(meta->resource_count);
	memcpy(&meta->vertex_input_count, &bytes[at], sizeof(meta->vertex_input_count)); at += sizeof(meta->vertex_input_count);
	meta->buffers       = (sksc_shader_buffer_t  *)malloc(sizeof(sksc_shader_buffer_t  ) * meta->buffer_count);
	meta->resources     = (sksc_shader_resource_t*)malloc(sizeof(sksc_shader_resource_t) * meta->resource_count);
	meta->vertex_inputs = (skr_vert_component_t  *)malloc(sizeof(skr_vert_component_t  ) * meta->vertex_input_count);
	memset(meta->buffers,       0, sizeof(sksc_shader_buffer_t  ) * meta->buffer_count);
	memset(meta->resources,     0, sizeof(sksc_shader_resource_t) * meta->resource_count);
	memset(meta->vertex_inputs, 0, sizeof(skr_vert_component_t ) * meta->vertex_input_count);

	memcpy(&meta->ops_vertex.total,        &bytes[at], sizeof(meta->ops_vertex.total));        at += sizeof(meta->ops_vertex.total);
	memcpy(&meta->ops_vertex.tex_read,     &bytes[at], sizeof(meta->ops_vertex.tex_read));     at += sizeof(meta->ops_vertex.tex_read);
	memcpy(&meta->ops_vertex.dynamic_flow, &bytes[at], sizeof(meta->ops_vertex.dynamic_flow)); at += sizeof(meta->ops_vertex.dynamic_flow);
	memcpy(&meta->ops_pixel.total,         &bytes[at], sizeof(meta->ops_pixel.total));         at += sizeof(meta->ops_pixel.total);
	memcpy(&meta->ops_pixel.tex_read,      &bytes[at], sizeof(meta->ops_pixel.tex_read));      at += sizeof(meta->ops_pixel.tex_read);
	memcpy(&meta->ops_pixel.dynamic_flow,  &bytes[at], sizeof(meta->ops_pixel.dynamic_flow));  at += sizeof(meta->ops_pixel.dynamic_flow);

	for (uint32_t i = 0; i < meta->buffer_count; i++) {
		sksc_shader_buffer_t *buffer = &meta->buffers[i];
		memcpy( buffer->name,      &bytes[at], sizeof(buffer->name));      at += sizeof(buffer->name);
		memcpy(&buffer->space,     &bytes[at], sizeof(buffer->space));     at += sizeof(buffer->space);
		memcpy(&buffer->bind,      &bytes[at], sizeof(buffer->bind));      at += sizeof(buffer->bind);
		memcpy(&buffer->size,      &bytes[at], sizeof(buffer->size));      at += sizeof(buffer->size);
		memcpy(&buffer->var_count, &bytes[at], sizeof(buffer->var_count)); at += sizeof(buffer->var_count);

		uint32_t default_size = 0;
		memcpy(&default_size, &bytes[at], sizeof(buffer->size)); at += sizeof(buffer->size);
		buffer->defaults = NULL;
		if (default_size != 0) {
			buffer->defaults = malloc(buffer->size);
			memcpy(buffer->defaults, &bytes[at], default_size); at += default_size;
		}
		buffer->vars = (sksc_shader_var_t*)malloc(sizeof(sksc_shader_var_t) * buffer->var_count);
		memset(buffer->vars, 0, sizeof(sksc_shader_var_t) * buffer->var_count);
		buffer->name_hash = skr_hash(buffer->name);

		for (uint32_t t = 0; t < buffer->var_count; t++) {
			sksc_shader_var_t *var = &buffer->vars[t];
			memcpy( var->name,       &bytes[at], sizeof(var->name     )); at += sizeof(var->name     );
			memcpy( var->extra,      &bytes[at], sizeof(var->extra    )); at += sizeof(var->extra    );
			memcpy( var->type_name,  &bytes[at], sizeof(var->type_name)); at += sizeof(var->type_name);
			memcpy(&var->offset,     &bytes[at], sizeof(var->offset   )); at += sizeof(var->offset   );
			memcpy(&var->size,       &bytes[at], sizeof(var->size     )); at += sizeof(var->size     );
			memcpy(&var->type,       &bytes[at], sizeof(var->type     )); at += sizeof(var->type     );
			memcpy(&var->type_count, &bytes[at], sizeof(var->type_count)); at += sizeof(var->type_count);
			var->name_hash = skr_hash(var->name);
		}

		if (strcmp(buffer->name, "$Global") == 0)
			meta->global_buffer_id = (int32_t)i;
	}

	for (int32_t i = 0; i < meta->vertex_input_count; i++) {
		skr_vert_component_t *com = &meta->vertex_inputs[i];
		memcpy(&com->format,        &bytes[at], sizeof(com->format       )); at += sizeof(com->format);
		memcpy(&com->count,         &bytes[at], sizeof(com->count        )); at += sizeof(com->count);
		memcpy(&com->semantic,      &bytes[at], sizeof(com->semantic     )); at += sizeof(com->semantic);
		memcpy(&com->semantic_slot, &bytes[at], sizeof(com->semantic_slot)); at += sizeof(com->semantic_slot);
	}

	for (uint32_t i = 0; i < meta->resource_count; i++) {
		sksc_shader_resource_t *res = &meta->resources[i];
		memcpy( res->name,         &bytes[at], sizeof(res->name        )); at += sizeof(res->name        );
		memcpy( res->value,        &bytes[at], sizeof(res->value       )); at += sizeof(res->value       );
		memcpy( res->tags,         &bytes[at], sizeof(res->tags        )); at += sizeof(res->tags        );
		memcpy(&res->bind,         &bytes[at], sizeof(res->bind        )); at += sizeof(res->bind        );
		memcpy(&res->element_size, &bytes[at], sizeof(res->element_size)); at += sizeof(res->element_size);
		res->name_hash = skr_hash(res->name);
	}

	return at;
}

///////////////////////////////////////////////////////////////////////////////

sksc_result_ sksc_shader_file_load_memory(const void *data, uint32_t size, sksc_shader_file_t *out_file) {
	uint16_t file_version = 0;
	if (!sksc_shader_file_verify(data, size, &file_version, NULL, 0)) return sksc_result_bad_format;
	if (file_version != 6)                                            return sksc_result_old_version;

	const uint8_t *bytes = (uint8_t*)data;
	uint32_t at = 10;

	memcpy(&out_file->stage_count, &bytes[at], sizeof(out_file->stage_count)); at += sizeof(out_file->stage_count);
	out_file->stages = (sksc_shader_file_stage_t*)malloc(sizeof(sksc_shader_file_stage_t) * out_file->stage_count);
	if (out_file->stages == NULL) return sksc_result_out_of_memory;

	out_file->meta = (sksc_shader_meta_t*)malloc(sizeof(sksc_shader_meta_t));
	if (out_file->meta == NULL) return sksc_result_out_of_memory;
	*out_file->meta = (sksc_shader_meta_t){0};
	out_file->meta->global_buffer_id = -1;
	sksc_shader_meta_reference(out_file->meta);
	at = _sksc_load_meta(bytes, at, out_file->meta);

	for (uint32_t i = 0; i < out_file->stage_count; i++) {
		sksc_shader_file_stage_t *stage = &out_file->stages[i];
		memcpy(&stage->language,  &bytes[at], sizeof(stage->language));  at += sizeof(stage->language);
		memcpy(&stage->stage,     &bytes[at], sizeof(stage->stage));     at += sizeof(stage->stage);
		memcpy(&stage->code_size, &bytes[at], sizeof(stage->code_size)); at += sizeof(stage->code_size);

		stage->code = NULL;
		if (stage->code_size > 0) {
			stage->code = malloc(stage->code_size);
			if (stage->code == NULL) return sksc_result_out_of_memory;
			memcpy(stage->code, &bytes[at], stage->code_size); at += stage->code_size;
		}
	}

	return sksc_result_success;
}

///////////////////////////////////////////////////////////////////////////////

void sksc_shader_file_destroy(sksc_shader_file_t *ref_file) {
	for (uint32_t i = 0; i < ref_file->stage_count; i++) {
		free(ref_file->stages[i].code);
	}
	free(ref_file->stages);
	sksc_shader_meta_release(ref_file->meta);
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

void sksc_shader_meta_reference(sksc_shader_meta_t *ref_meta) {
	ref_meta->references += 1;
}

///////////////////////////////////////////////////////////////////////////////

void sksc_shader_meta_release(sksc_shader_meta_t *ref_meta) {
	if (!ref_meta) return;
	ref_meta->references -= 1;
	if (ref_meta->references == 0) {
		for (uint32_t i = 0; i < ref_meta->buffer_count; i++) {
			free(ref_meta->buffers[i].vars);
			free(ref_meta->buffers[i].defaults);
		}
		free(ref_meta->buffers);
		free(ref_meta->resources);
		free(ref_meta->vertex_inputs);
		*ref_meta = (sksc_shader_meta_t){0};
	}
}
