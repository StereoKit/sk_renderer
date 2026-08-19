// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "skr_pipeline.h"

///////////////////////////////////////////////////////////////////////////////

static WGPUComputePipeline _skr_compute_pipeline_create(const skr_shader_t* shader, WGPUPipelineLayout layout, const uint32_t spec_values[SKR_MAX_SPEC_CONSTANTS]) {
	const sksc_shader_meta_t* meta = &shader->meta;

	WGPUConstantEntry constants[SKR_MAX_SPEC_CONSTANTS];
	_skr_const_key_t  keys     [SKR_MAX_SPEC_CONSTANTS];
	uint32_t constant_count = _skr_stage_constants(meta, spec_values, skr_stage_compute, false, 0, constants, keys, SKR_MAX_SPEC_CONSTANTS);

	WGPUComputePipelineDescriptor desc = {
		.label  = { meta->name, strnlen(meta->name, sizeof(meta->name)) },
		.layout = layout,
		.compute = {
			.module        = shader->compute_stage.shader,
			.entryPoint    = { NULL, WGPU_STRLEN }, // single entry per module
			.constantCount = constant_count,
			.constants     = constants,
		},
	};
	return wgpuDeviceCreateComputePipeline(_skr_wgpu.device, &desc);
}

///////////////////////////////////////////////////////////////////////////////

skr_err_ skr_compute_create(const skr_shader_t* shader, skr_compute_info_t info, skr_compute_t* out_compute) {
	if (out_compute == NULL) return skr_err_invalid_parameter;
	memset(out_compute, 0, sizeof(*out_compute));
	if (shader == NULL || shader->compute_stage.shader == NULL) {
		skr_log(skr_log_critical, "Invalid shader or no compute stage");
		return skr_err_invalid_parameter;
	}

	const sksc_shader_meta_t* meta = &shader->meta;
	out_compute->shader = shader;

	out_compute->bind_group_layout = _skr_bind_layout_create(meta, skr_stage_compute);
	WGPUPipelineLayoutDescriptor layout_desc = {
		.bindGroupLayoutCount = 1,
		.bindGroupLayouts     = &out_compute->bind_group_layout,
	};
	out_compute->layout = wgpuDeviceCreatePipelineLayout(_skr_wgpu.device, &layout_desc);

	uint32_t spec_values[SKR_MAX_SPEC_CONSTANTS];
	_skr_resolve_spec_constants(meta, info.spec_constants, info.spec_constant_count, spec_values);
	out_compute->pipeline = _skr_compute_pipeline_create(shader, out_compute->layout, spec_values);
	if (out_compute->pipeline == NULL) {
		skr_compute_destroy(out_compute);
		return skr_err_device_error;
	}

	out_compute->bind_count = meta->resource_count + meta->buffer_count;
	out_compute->binds      = (skr_material_bind_t*)_skr_calloc(out_compute->bind_count > 0 ? out_compute->bind_count : 1, sizeof(skr_material_bind_t));
	for (uint32_t i = 0; i < meta->buffer_count;   i++) out_compute->binds[i].bind = meta->buffers[i].bind;
	for (uint32_t i = 0; i < meta->resource_count; i++) out_compute->binds[i + meta->buffer_count].bind = meta->resources[i].bind;

	if (meta->global_buffer_id >= 0) {
		sksc_shader_buffer_t* global_buffer = &meta->buffers[meta->global_buffer_id];
		out_compute->param_buffer_size = global_buffer->size;
		out_compute->param_buffer      = _skr_malloc(global_buffer->size);
		if (global_buffer->defaults) memcpy(out_compute->param_buffer, global_buffer->defaults, global_buffer->size);
		else                         memset(out_compute->param_buffer, 0, global_buffer->size);
	}
	return skr_err_success;
}

void skr_compute_set_pipeline(skr_compute_t* ref_compute, skr_compute_info_t info) {
	if (ref_compute == NULL || ref_compute->shader == NULL) return;
	uint32_t spec_values[SKR_MAX_SPEC_CONSTANTS];
	_skr_resolve_spec_constants(&ref_compute->shader->meta, info.spec_constants, info.spec_constant_count, spec_values);
	WGPUComputePipeline pipeline = _skr_compute_pipeline_create(ref_compute->shader, ref_compute->layout, spec_values);
	if (pipeline) {
		if (ref_compute->pipeline) wgpuComputePipelineRelease(ref_compute->pipeline);
		ref_compute->pipeline = pipeline;
	}
}

bool skr_compute_is_valid(const skr_compute_t* compute) {
	return compute != NULL && compute->pipeline != NULL;
}

void skr_compute_destroy(skr_compute_t* ref_compute) {
	if (ref_compute == NULL) return;
	if (ref_compute->pipeline)          wgpuComputePipelineRelease(ref_compute->pipeline);
	if (ref_compute->layout)            wgpuPipelineLayoutRelease(ref_compute->layout);
	if (ref_compute->bind_group_layout) wgpuBindGroupLayoutRelease(ref_compute->bind_group_layout);
	_skr_free(ref_compute->binds);
	_skr_free(ref_compute->param_buffer);
	memset(ref_compute, 0, sizeof(*ref_compute));
}

///////////////////////////////////////////////////////////////////////////////

static WGPUBindGroup _skr_compute_bind_group(skr_compute_t* compute, uint32_t* out_dyn_offsets, uint32_t* out_dyn_count) {
	_skr_draw_buffers_t db = {0};
	if (compute->param_buffer && compute->param_buffer_size > 0) {
		_skr_bump_uniform_write(compute->param_buffer, compute->param_buffer_size, &db.material_offset);
		db.material_size = compute->param_buffer_size;
	}
	*out_dyn_count = _skr_dynamic_offsets(&compute->shader->meta, (uint8_t)skr_stage_compute, &db, out_dyn_offsets);
	return _skr_build_bind_group_meta(&compute->shader->meta, compute->bind_group_layout, compute->binds, compute->bind_count);
}

void skr_compute_execute(skr_compute_t* ref_compute, uint32_t x, uint32_t y, uint32_t z) {
	if (!skr_compute_is_valid(ref_compute)) return;

	uint32_t dyn_offsets[3], dyn_count = 0;
	WGPUBindGroup bind_group = _skr_compute_bind_group(ref_compute, dyn_offsets, &dyn_count);
	if (bind_group == NULL) {
		skr_log(skr_log_critical, "Compute dispatch missing bindings in shader '%s'", ref_compute->shader->meta.name);
		return;
	}

	WGPUComputePassDescriptor pass_desc = {0};
	WGPUPassTimestampWrites   ts;
	pass_desc.timestampWrites = _skr_timer_pass_writes(&ts);
	WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(_skr_cmd_get(), &pass_desc);
	wgpuComputePassEncoderSetPipeline (pass, ref_compute->pipeline);
	wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, dyn_count, dyn_offsets);
	wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, z);
	wgpuComputePassEncoderEnd(pass);
	wgpuComputePassEncoderRelease(pass);
	wgpuBindGroupRelease(bind_group);
}

void skr_compute_execute_indirect(skr_compute_t* ref_compute, skr_buffer_t* indirect_args) {
	if (!skr_compute_is_valid(ref_compute) || indirect_args == NULL || indirect_args->buffer == NULL) return;

	uint32_t dyn_offsets[3], dyn_count = 0;
	WGPUBindGroup bind_group = _skr_compute_bind_group(ref_compute, dyn_offsets, &dyn_count);
	if (bind_group == NULL) {
		skr_log(skr_log_critical, "Compute dispatch missing bindings in shader '%s'", ref_compute->shader->meta.name);
		return;
	}

	WGPUComputePassDescriptor pass_desc = {0};
	WGPUPassTimestampWrites   ts;
	pass_desc.timestampWrites = _skr_timer_pass_writes(&ts);
	WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(_skr_cmd_get(), &pass_desc);
	wgpuComputePassEncoderSetPipeline (pass, ref_compute->pipeline);
	wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, dyn_count, dyn_offsets);
	wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, indirect_args->buffer, 0);
	wgpuComputePassEncoderEnd(pass);
	wgpuComputePassEncoderRelease(pass);
	wgpuBindGroupRelease(bind_group);
}

///////////////////////////////////////////////////////////////////////////////

skr_bind_t skr_compute_get_bind(const skr_compute_t* compute, const char* bind_name) {
	if (compute == NULL || compute->shader == NULL) { skr_bind_t empty = {0}; return empty; }
	return sksc_shader_meta_get_bind(&compute->shader->meta, bind_name);
}

void skr_compute_set_tex(skr_compute_t* ref_compute, const char* name, skr_tex_t* texture) {
	if (ref_compute == NULL || name == NULL || ref_compute->shader == NULL) return;
	skr_bind_t bind = sksc_shader_meta_get_bind(&ref_compute->shader->meta, name);
	if (bind.stage_bits == 0) return;
	for (uint32_t i = 0; i < ref_compute->bind_count; i++)
		if (ref_compute->binds[i].bind.slot == bind.slot && ref_compute->binds[i].bind.register_type == bind.register_type) {
			ref_compute->binds[i].texture = texture;
			return;
		}
}

void skr_compute_set_buffer(skr_compute_t* ref_compute, const char* name, skr_buffer_t* buffer) {
	if (ref_compute == NULL || name == NULL || ref_compute->shader == NULL) return;
	skr_bind_t bind = sksc_shader_meta_get_bind(&ref_compute->shader->meta, name);
	if (bind.stage_bits == 0) return;
	for (uint32_t i = 0; i < ref_compute->bind_count; i++)
		if (ref_compute->binds[i].bind.slot == bind.slot && ref_compute->binds[i].bind.register_type == bind.register_type) {
			ref_compute->binds[i].buffer = buffer;
			return;
		}
}

///////////////////////////////////////////////////////////////////////////////

void skr_compute_set_params(skr_compute_t* ref_compute, const void* data, uint32_t size) {
	if (ref_compute == NULL || ref_compute->param_buffer == NULL || data == NULL) return;
	if (size > ref_compute->param_buffer_size) size = ref_compute->param_buffer_size;
	memcpy(ref_compute->param_buffer, data, size);
}

void skr_compute_set_param(skr_compute_t* compute, const char* name, sksc_shader_var_ type, uint32_t count, const void* data) {
	if (compute == NULL || compute->param_buffer == NULL || name == NULL || data == NULL) return;
	const sksc_shader_meta_t* meta = &compute->shader->meta;
	int32_t idx = sksc_shader_meta_get_var_index(meta, name);
	if (idx < 0) return;
	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(meta, idx);
	if (var == NULL || var->type != (uint16_t)type) return;

	uint32_t elem = type == sksc_shader_var_double ? 8 : type == sksc_shader_var_uint8 ? 1 : 4;
	uint32_t size = elem * count;
	if (size > var->size) size = var->size;
	memcpy((uint8_t*)compute->param_buffer + var->offset, data, size);
}

void skr_compute_get_param(const skr_compute_t* compute, const char* name, sksc_shader_var_ type, uint32_t count, void* out_data) {
	if (compute == NULL || compute->param_buffer == NULL || name == NULL || out_data == NULL) return;
	const sksc_shader_meta_t* meta = &compute->shader->meta;
	int32_t idx = sksc_shader_meta_get_var_index(meta, name);
	if (idx < 0) return;
	const sksc_shader_var_t* var = sksc_shader_meta_get_var_info(meta, idx);
	if (var == NULL || var->type != (uint16_t)type) return;

	uint32_t elem = type == sksc_shader_var_double ? 8 : type == sksc_shader_var_uint8 ? 1 : 4;
	uint32_t size = elem * count;
	if (size > var->size) size = var->size;
	memcpy(out_data, (const uint8_t*)compute->param_buffer + var->offset, size);
}
