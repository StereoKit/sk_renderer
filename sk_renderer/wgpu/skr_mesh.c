// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "_sk_renderer.h"

///////////////////////////////////////////////////////////////////////////////
// Vertex types. Formats and offsets are shader-independent, so attributes are
// built here; shaderLocation is patched per-shader at pipeline creation by
// matching semantics (same scheme as the Vulkan backend's pipeline cache).

static uint32_t _skr_vert_component_size(const skr_vert_component_t* c) {
	uint32_t elem = 4;
	switch (c->format) {
		case skr_vertex_fmt_f64: elem = 8; break;
		case skr_vertex_fmt_f32: case skr_vertex_fmt_i32: case skr_vertex_fmt_ui32:
		case skr_vertex_fmt_i32_normalized: case skr_vertex_fmt_ui32_normalized: elem = 4; break;
		case skr_vertex_fmt_f16: case skr_vertex_fmt_i16: case skr_vertex_fmt_ui16:
		case skr_vertex_fmt_i16_normalized: case skr_vertex_fmt_ui16_normalized: elem = 2; break;
		case skr_vertex_fmt_i8: case skr_vertex_fmt_ui8:
		case skr_vertex_fmt_i8_normalized: case skr_vertex_fmt_ui8_normalized: elem = 1; break;
		default: break;
	}
	return elem * c->count;
}

WGPUVertexFormat _skr_vert_format(const skr_vert_component_t* c) {
	// WebGPU vertex formats: 8/16-bit come in x2/x4 only, 32-bit in x1..x4
	uint8_t n = c->count;
	switch (c->format) {
		case skr_vertex_fmt_f32:
			return n == 1 ? WGPUVertexFormat_Float32 : n == 2 ? WGPUVertexFormat_Float32x2 : n == 3 ? WGPUVertexFormat_Float32x3 : WGPUVertexFormat_Float32x4;
		case skr_vertex_fmt_i32:
			return n == 1 ? WGPUVertexFormat_Sint32 : n == 2 ? WGPUVertexFormat_Sint32x2 : n == 3 ? WGPUVertexFormat_Sint32x3 : WGPUVertexFormat_Sint32x4;
		case skr_vertex_fmt_ui32:
			return n == 1 ? WGPUVertexFormat_Uint32 : n == 2 ? WGPUVertexFormat_Uint32x2 : n == 3 ? WGPUVertexFormat_Uint32x3 : WGPUVertexFormat_Uint32x4;
		case skr_vertex_fmt_f16:             return n <= 2 ? WGPUVertexFormat_Float16x2 : WGPUVertexFormat_Float16x4;
		case skr_vertex_fmt_i16:             return n <= 2 ? WGPUVertexFormat_Sint16x2  : WGPUVertexFormat_Sint16x4;
		case skr_vertex_fmt_ui16:            return n <= 2 ? WGPUVertexFormat_Uint16x2  : WGPUVertexFormat_Uint16x4;
		case skr_vertex_fmt_i16_normalized:  return n <= 2 ? WGPUVertexFormat_Snorm16x2 : WGPUVertexFormat_Snorm16x4;
		case skr_vertex_fmt_ui16_normalized: return n <= 2 ? WGPUVertexFormat_Unorm16x2 : WGPUVertexFormat_Unorm16x4;
		case skr_vertex_fmt_i8:              return n <= 2 ? WGPUVertexFormat_Sint8x2   : WGPUVertexFormat_Sint8x4;
		case skr_vertex_fmt_ui8:             return n <= 2 ? WGPUVertexFormat_Uint8x2   : WGPUVertexFormat_Uint8x4;
		case skr_vertex_fmt_i8_normalized:   return n <= 2 ? WGPUVertexFormat_Snorm8x2  : WGPUVertexFormat_Snorm8x4;
		case skr_vertex_fmt_ui8_normalized:  return n <= 2 ? WGPUVertexFormat_Unorm8x2  : WGPUVertexFormat_Unorm8x4;
		default:                             return WGPUVertexFormat_Float32;
	}
}

skr_err_ skr_vert_type_create(const skr_vert_component_t* items, int32_t item_count, skr_vert_type_t* out_type) {
	if (out_type == NULL) return skr_err_invalid_parameter;
	memset(out_type, 0, sizeof(*out_type));
	if (items == NULL || item_count <= 0) return skr_err_invalid_parameter;

	out_type->components      = (skr_vert_component_t*)_skr_malloc(sizeof(skr_vert_component_t) * item_count);
	out_type->component_count = (uint32_t)item_count;
	memcpy(out_type->components, items, sizeof(skr_vert_component_t) * item_count);

	uint32_t binding_count = 0;
	for (int32_t i = 0; i < item_count; i++)
		if (items[i].binding + 1u > binding_count) binding_count = items[i].binding + 1u;
	if (binding_count > SKR_MAX_VERTEX_BUFFERS) {
		_skr_free(out_type->components);
		memset(out_type, 0, sizeof(*out_type));
		return skr_err_invalid_parameter;
	}

	// Attributes carry format+offset; shaderLocation is set per-pipeline
	out_type->attributes = (WGPUVertexAttribute*)_skr_calloc(item_count, sizeof(WGPUVertexAttribute));
	out_type->bindings   = (WGPUVertexBufferLayout*)_skr_calloc(binding_count, sizeof(WGPUVertexBufferLayout));
	out_type->binding_count = binding_count;

	uint64_t offsets[SKR_MAX_VERTEX_BUFFERS] = {0};
	for (int32_t i = 0; i < item_count; i++) {
		uint8_t b = items[i].binding;
		out_type->attributes[i] = (WGPUVertexAttribute){
			.format         = _skr_vert_format(&items[i]),
			.offset         = offsets[b],
			.shaderLocation = 0, // patched at pipeline creation
		};
		offsets[b] += _skr_vert_component_size(&items[i]);
	}
	for (uint32_t b = 0; b < binding_count; b++) {
		out_type->bindings[b] = (WGPUVertexBufferLayout){
			.arrayStride    = offsets[b],
			.stepMode       = WGPUVertexStepMode_Vertex,
			.attributeCount = 0,    // per-binding attribute grouping happens at pipeline creation
			.attributes     = NULL,
		};
	}

	out_type->pipeline_idx = -1;
	return skr_err_success;
}

bool skr_vert_type_is_valid(const skr_vert_type_t* type) {
	return type != NULL && type->components != NULL;
}

void skr_vert_type_destroy(skr_vert_type_t* ref_type) {
	if (ref_type == NULL) return;
	_skr_free(ref_type->components);
	_skr_free(ref_type->attributes);
	_skr_free(ref_type->bindings);
	memset(ref_type, 0, sizeof(*ref_type));
}

///////////////////////////////////////////////////////////////////////////////
// Meshes

skr_err_ skr_mesh_create(const skr_vert_type_t* vert_type, skr_index_fmt_ ind_type, const void* opt_vert_data, uint32_t vert_count, const void* opt_ind_data, uint32_t ind_count, skr_mesh_t* out_mesh) {
	if (out_mesh == NULL) return skr_err_invalid_parameter;
	memset(out_mesh, 0, sizeof(*out_mesh));
	// A NULL vert_type is a vertex-pulling mesh: vert_count drives the draw
	// and all vertex data comes from shader-bound buffers
	if (vert_type != NULL && !skr_vert_type_is_valid(vert_type)) return skr_err_invalid_parameter;
	if (ind_type == skr_index_fmt_u8) {
		skr_log(skr_log_warning, "WebGPU has no 8-bit index format");
		return skr_err_unsupported;
	}

	out_mesh->vert_type       = vert_type;
	out_mesh->ind_format      = ind_type;
	out_mesh->ind_format_wgpu = ind_type == skr_index_fmt_u16 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
	if (vert_type == NULL) out_mesh->vert_count = vert_count;

	if (opt_vert_data != NULL && vert_count > 0) {
		skr_err_ err = skr_mesh_set_verts(out_mesh, opt_vert_data, vert_count);
		if (err != skr_err_success) { skr_mesh_destroy(out_mesh); return err; }
	}
	if (opt_ind_data != NULL && ind_count > 0) {
		skr_err_ err = skr_mesh_set_inds(out_mesh, opt_ind_data, ind_count);
		if (err != skr_err_success) { skr_mesh_destroy(out_mesh); return err; }
	}
	return skr_err_success;
}

bool skr_mesh_is_valid(const skr_mesh_t* mesh) {
	return mesh != NULL && (mesh->vert_type != NULL || mesh->vert_count > 0);
}

void skr_mesh_destroy(skr_mesh_t* ref_mesh) {
	if (ref_mesh == NULL) return;
	for (uint32_t i = 0; i < SKR_MAX_VERTEX_BUFFERS; i++)
		if (ref_mesh->vertex_buffer_owned & (1u << i))
			skr_buffer_destroy(&ref_mesh->vertex_buffers[i]);
	skr_buffer_destroy(&ref_mesh->index_buffer);
	memset(ref_mesh, 0, sizeof(*ref_mesh));
}

///////////////////////////////////////////////////////////////////////////////

uint32_t skr_mesh_get_vert_count(const skr_mesh_t* mesh) { return mesh ? mesh->vert_count : 0; }
uint32_t skr_mesh_get_ind_count (const skr_mesh_t* mesh) { return mesh ? mesh->ind_count  : 0; }

void skr_mesh_set_name(skr_mesh_t* ref_mesh, const char* name) {
	if (ref_mesh == NULL || name == NULL) return;
	for (uint32_t i = 0; i < ref_mesh->vertex_buffer_count; i++)
		skr_buffer_set_name(&ref_mesh->vertex_buffers[i], name);
	skr_buffer_set_name(&ref_mesh->index_buffer, name);
}

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_mesh_set_verts(skr_mesh_t* ref_mesh, const void* vert_data, uint32_t vert_count) {
	if (ref_mesh == NULL || vert_data == NULL || vert_count == 0) return skr_err_invalid_parameter;
	if (ref_mesh->vert_type == NULL || ref_mesh->vert_type->binding_count == 0) return skr_err_invalid_parameter; // vertex-pulling meshes have no CPU vertex stream

	// Interleaved data targets binding 0; multi-buffer meshes attach their
	// extra streams via skr_mesh_set_vertex_buffer
	uint32_t stride = (uint32_t)ref_mesh->vert_type->bindings[0].arrayStride;
	uint32_t size   = stride * vert_count;

	skr_buffer_t* buffer = &ref_mesh->vertex_buffers[0];
	if (buffer->buffer != NULL && (ref_mesh->vertex_buffer_owned & 1u) && size <= buffer->size) {
		skr_buffer_set(buffer, vert_data, size);
	} else {
		if (ref_mesh->vertex_buffer_owned & 1u) skr_buffer_destroy(buffer);
		skr_err_ err = skr_buffer_create(vert_data, vert_count, stride, skr_buffer_type_vertex, skr_use_static, buffer);
		if (err != skr_err_success) return err;
		ref_mesh->vertex_buffer_owned |= 1u;
	}
	if (ref_mesh->vertex_buffer_count < 1) ref_mesh->vertex_buffer_count = 1;
	ref_mesh->vert_count = vert_count;
	return skr_err_success;
}

skr_err_ skr_mesh_set_inds(skr_mesh_t* ref_mesh, const void* ind_data, uint32_t ind_count) {
	if (ref_mesh == NULL || ind_data == NULL || ind_count == 0) return skr_err_invalid_parameter;

	uint32_t stride = ref_mesh->ind_format == skr_index_fmt_u16 ? 2 : 4;
	uint32_t size   = stride * ind_count;

	if (ref_mesh->index_buffer.buffer != NULL && size <= ref_mesh->index_buffer.size) {
		skr_buffer_set(&ref_mesh->index_buffer, ind_data, size);
	} else {
		skr_buffer_destroy(&ref_mesh->index_buffer);
		skr_err_ err = skr_buffer_create(ind_data, ind_count, stride, skr_buffer_type_index, skr_use_static, &ref_mesh->index_buffer);
		if (err != skr_err_success) return err;
	}
	ref_mesh->ind_count = ind_count;
	return skr_err_success;
}

skr_err_ skr_mesh_set_data(skr_mesh_t* ref_mesh, const void* vert_data, uint32_t vert_count, const void* ind_data, uint32_t ind_count) {
	skr_err_ err = skr_mesh_set_verts(ref_mesh, vert_data, vert_count);
	if (err != skr_err_success) return err;
	return skr_mesh_set_inds(ref_mesh, ind_data, ind_count);
}

skr_err_ skr_mesh_set_vertex_buffer(skr_mesh_t* ref_mesh, uint32_t binding, const skr_buffer_t* buffer, uint32_t vert_count) {
	if (ref_mesh == NULL || buffer == NULL || binding >= SKR_MAX_VERTEX_BUFFERS) return skr_err_invalid_parameter;

	if (ref_mesh->vertex_buffer_owned & (1u << binding))
		skr_buffer_destroy(&ref_mesh->vertex_buffers[binding]);
	ref_mesh->vertex_buffers[binding] = *buffer; // external reference, not owned
	ref_mesh->vertex_buffer_owned    &= ~(1u << binding);
	if (binding + 1 > ref_mesh->vertex_buffer_count) ref_mesh->vertex_buffer_count = binding + 1;
	if (vert_count > 0) ref_mesh->vert_count = vert_count;
	return skr_err_success;
}

skr_buffer_t* skr_mesh_get_vertex_buffer(const skr_mesh_t* mesh, uint32_t binding) {
	if (mesh == NULL || binding >= mesh->vertex_buffer_count) return NULL;
	return (skr_buffer_t*)&mesh->vertex_buffers[binding];
}
