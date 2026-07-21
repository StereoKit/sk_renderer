// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2026 Nick Klingensmith

#include "skr_pipeline.h"

///////////////////////////////////////////////////////////////////////////////
// Registries
///////////////////////////////////////////////////////////////////////////////

typedef struct _skr_pipeline_entry_t {
	skr_pipeline_pass_key_t pass;
	int32_t                 vert_idx;
	WGPURenderPipeline      pipeline;
} _skr_pipeline_entry_t;

typedef struct _skr_material_slot_t {
	_skr_pipeline_material_key_t key;
	int32_t                      refcount;
	bool                         pending_destroy; // handles release at next frame begin
	WGPUBindGroupLayout          bind_layout;
	WGPUPipelineLayout           layout;
	_skr_pipeline_entry_t*       pipelines;
	int32_t                      pipeline_count;
	int32_t                      pipeline_capacity;
} _skr_material_slot_t;

typedef struct _skr_vertformat_slot_t {
	skr_vert_component_t* components;
	uint32_t              component_count;
	WGPUVertexAttribute*  attributes;      // format+offset per component
	uint64_t              strides[SKR_MAX_VERTEX_BUFFERS];
	uint32_t              binding_count;
} _skr_vertformat_slot_t;

// Slots are heap-allocated individually and never move, published into
// fixed-size tables with release stores — readers (the per-draw path) follow
// acquire loads and take no locks. The mutex only coordinates writers
// (register/unregister from any thread), a rare and already-heavyweight
// path. Unregistered slots release their WGPU handles at the next frame
// boundary, since draws recorded this frame may still use them; their table
// entries are then recycled, so the fixed capacity bounds live materials,
// not total creations.
#define _SKR_MATERIAL_SLOT_MAX   4096
#define _SKR_VERTFORMAT_SLOT_MAX 256

static _skr_atomic(_skr_material_slot_t*)   _materials[_SKR_MATERIAL_SLOT_MAX];
static _skr_atomic(int32_t)                 _material_count;
static _skr_atomic(_skr_vertformat_slot_t*) _vertformats[_SKR_VERTFORMAT_SLOT_MAX];
static _skr_atomic(int32_t)                 _vertformat_count;
static _skr_mtx_t                           _registry_mutex;
static bool                                 _registry_mutex_ready;

void _skr_pipeline_init(void) {
	if (_registry_mutex_ready) return;
	_skr_mtx_init(&_registry_mutex);
	_registry_mutex_ready = true;
}

static void _skr_material_slot_release(_skr_material_slot_t* mat) {
	for (int32_t p = 0; p < mat->pipeline_count; p++)
		if (mat->pipelines[p].pipeline) wgpuRenderPipelineRelease(mat->pipelines[p].pipeline);
	_skr_free(mat->pipelines);
	if (mat->layout)      wgpuPipelineLayoutRelease(mat->layout);
	if (mat->bind_layout) wgpuBindGroupLayoutRelease(mat->bind_layout);
	memset(&mat->key, 0, sizeof(mat->key));
	mat->pipelines = NULL; mat->pipeline_count = 0; mat->pipeline_capacity = 0;
	mat->layout = NULL; mat->bind_layout = NULL;
	mat->pending_destroy = false;
}

// Release WGPU handles of slots unregistered last frame; call at frame begin
void _skr_pipeline_registry_drain(void) {
	_skr_pipeline_init();
	_skr_mtx_lock(&_registry_mutex);
	int32_t count = _skr_load_acquire(&_material_count);
	for (int32_t i = 0; i < count; i++) {
		_skr_material_slot_t* mat = _skr_load_acquire(&_materials[i]);
		if (mat && mat->pending_destroy)
			_skr_material_slot_release(mat);
	}
	_skr_mtx_unlock(&_registry_mutex);
}

void _skr_pipeline_shutdown(void) {
	int32_t mat_count = _skr_load_acquire(&_material_count);
	for (int32_t i = 0; i < mat_count; i++) {
		_skr_material_slot_t* mat = _skr_load_acquire(&_materials[i]);
		if (mat) { _skr_material_slot_release(mat); _skr_free(mat); }
		_skr_store_release(&_materials[i], NULL);
	}
	int32_t vf_count = _skr_load_acquire(&_vertformat_count);
	for (int32_t i = 0; i < vf_count; i++) {
		_skr_vertformat_slot_t* vf = _skr_load_acquire(&_vertformats[i]);
		if (vf) { _skr_free(vf->components); _skr_free(vf->attributes); _skr_free(vf); }
		_skr_store_release(&_vertformats[i], NULL);
	}
	_skr_store_release(&_material_count,   0);
	_skr_store_release(&_vertformat_count, 0);
	if (_registry_mutex_ready) { _skr_mtx_destroy(&_registry_mutex); _registry_mutex_ready = false; }
}

///////////////////////////////////////////////////////////////////////////////
// Bind group layout from shader meta
///////////////////////////////////////////////////////////////////////////////

static WGPUTextureViewDimension _skr_shape_view_dim(uint8_t shape) {
	bool arrayed = (shape & SKSC_SHAPE_ARRAYED) != 0;
	switch (shape & SKSC_SHAPE_DIM_MASK) {
		case SKSC_SHAPE_DIM_3D:   return WGPUTextureViewDimension_3D;
		case SKSC_SHAPE_DIM_CUBE: return arrayed ? WGPUTextureViewDimension_CubeArray : WGPUTextureViewDimension_Cube;
		case SKSC_SHAPE_DIM_1D:   return WGPUTextureViewDimension_1D;
		default:                  return arrayed ? WGPUTextureViewDimension_2DArray : WGPUTextureViewDimension_2D;
	}
}

static WGPUTextureFormat _skr_spv_image_format(uint8_t spv_format) {
	switch (spv_format) {
		case 1:  return WGPUTextureFormat_RGBA32Float;
		case 2:  return WGPUTextureFormat_RGBA16Float;
		case 3:  return WGPUTextureFormat_R32Float;
		case 4:  return WGPUTextureFormat_RGBA8Unorm;
		case 5:  return WGPUTextureFormat_RGBA8Snorm;
		case 6:  return WGPUTextureFormat_RG32Float;
		case 21: return WGPUTextureFormat_RGBA32Sint;
		case 25: return WGPUTextureFormat_RG32Sint;
		case 22: return WGPUTextureFormat_RGBA16Sint;
		case 23: return WGPUTextureFormat_RGBA8Sint;
		case 24: return WGPUTextureFormat_R32Sint;
		case 30: return WGPUTextureFormat_RGBA32Uint;
		case 31: return WGPUTextureFormat_RGBA16Uint;
		case 32: return WGPUTextureFormat_RGBA8Uint;
		case 33: return WGPUTextureFormat_R32Uint;
		case 35: return WGPUTextureFormat_RG32Uint;
		default:
			skr_log(skr_log_warning, "Storage image format %d unmapped, assuming rgba8", spv_format);
			return WGPUTextureFormat_RGBA8Unorm;
	}
}

static WGPUShaderStage _skr_stage_visibility(uint8_t stage_bits) {
	WGPUShaderStage vis = WGPUShaderStage_None;
	if (stage_bits & skr_stage_vertex ) vis |= WGPUShaderStage_Vertex;
	if (stage_bits & skr_stage_pixel  ) vis |= WGPUShaderStage_Fragment;
	if (stage_bits & skr_stage_compute) vis |= WGPUShaderStage_Compute;
	return vis;
}

bool _skr_sampler_is_comparison(const sksc_shader_meta_t* meta, uint32_t paired_slot) {
	for (uint32_t r = 0; r < meta->resource_count; r++)
		if (meta->resources[r].bind.slot == paired_slot && (meta->resources[r].shape & SKSC_SHAPE_COMPARISON))
			return true;
	return false;
}

void _skr_resolve_spec_constants(const sksc_shader_meta_t* meta, const skr_spec_constant_t* specs, uint32_t spec_count, uint32_t out_values[SKR_MAX_SPEC_CONSTANTS]) {
	memset(out_values, 0, sizeof(uint32_t) * SKR_MAX_SPEC_CONSTANTS);
	for (uint32_t i = 0; i < meta->spec_constant_count && i < SKR_MAX_SPEC_CONSTANTS; i++)
		out_values[i] = meta->spec_constants[i].default_value;

	for (uint32_t s = 0; s < spec_count; s++) {
		if (specs[s].name == NULL) continue;
		uint64_t hash = skr_hash(specs[s].name);
		for (uint32_t i = 0; i < meta->spec_constant_count && i < SKR_MAX_SPEC_CONSTANTS; i++) {
			if (meta->spec_constants[i].name_hash != hash) continue;
			switch (meta->spec_constants[i].type) {
				case sksc_shader_var_float: { float    v = (float   )specs[s].value; memcpy(&out_values[i], &v, 4); } break;
				case sksc_shader_var_uint:  { uint32_t v = (uint32_t)specs[s].value; out_values[i] = v; }             break;
				default:                    { int32_t  v = (int32_t )specs[s].value; memcpy(&out_values[i], &v, 4); } break;
			}
			break;
		}
	}
}

WGPUBindGroupLayout _skr_bind_layout_create(const sksc_shader_meta_t* meta, uint8_t stage_mask) {
	WGPUBindGroupLayoutEntry entries[64];
	uint32_t                 entry_count = 0;

	uint32_t system_slot   = (uint32_t)_skr_wgpu.binds.system_slot;
	uint32_t material_slot = (uint32_t)_skr_wgpu.binds.material_slot;
	uint32_t instance_slot = SKSC_SLOT_TEXTURE + (uint32_t)_skr_wgpu.binds.instance_slot; // StructuredBuffers live in t registers

	for (uint32_t i = 0; i < meta->buffer_count && entry_count < 64; i++) {
		uint8_t stages = meta->buffers[i].bind.stage_bits & stage_mask;
		if (!stages) continue;
		uint32_t slot = meta->buffers[i].bind.slot;
		// System + material params come from per-frame bump allocations at a
		// new offset every frame. Dynamic offsets keep the bind group itself
		// stable, so one group per material serves every draw and frame.
		bool dynamic = slot == material_slot || slot == system_slot;
		entries[entry_count++] = (WGPUBindGroupLayoutEntry){
			.binding    = slot,
			.visibility = _skr_stage_visibility(stages),
			.buffer     = { .type = WGPUBufferBindingType_Uniform, .hasDynamicOffset = dynamic },
		};
	}

	for (uint32_t i = 0; i < meta->resource_count && entry_count < 64; i++) {
		const sksc_shader_resource_t* res = &meta->resources[i];
		uint8_t stages = res->bind.stage_bits & stage_mask;
		if (!stages) continue;

		WGPUBindGroupLayoutEntry entry = {
			.binding    = res->bind.slot,
			.visibility = _skr_stage_visibility(stages),
		};
		switch (res->bind.register_type) {
			case skr_register_texture:
				entry.texture = (WGPUTextureBindingLayout){
					.sampleType    = (res->shape & SKSC_SHAPE_COMPARISON) ? WGPUTextureSampleType_Depth : WGPUTextureSampleType_Float,
					.viewDimension = _skr_shape_view_dim(res->shape),
					.multisampled  = (res->shape & SKSC_SHAPE_MS) != 0,
				};
				if (entry.texture.multisampled)
					entry.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
				break;
			case skr_register_input_attachment:
				// Lowered to textureLoad-only textures by SVSL's WGSL emitter;
				// UnfilterableFloat accepts both color and depth format views
				entry.texture = (WGPUTextureBindingLayout){
					.sampleType    = WGPUTextureSampleType_UnfilterableFloat,
					.viewDimension = WGPUTextureViewDimension_2D,
					.multisampled  = (res->shape & SKSC_SHAPE_MS) != 0,
				};
				break;
			case skr_register_read_buffer:
				// Per-draw instance data streams through the storage bump
				// buffer; a dynamic offset selects each draw's slice
				entry.buffer = (WGPUBufferBindingLayout){
					.type             = WGPUBufferBindingType_ReadOnlyStorage,
					.hasDynamicOffset = res->bind.slot == instance_slot,
				};
				break;
			case skr_register_readwrite:
				entry.buffer = (WGPUBufferBindingLayout){ .type = WGPUBufferBindingType_Storage };
				break;
			case skr_register_readwrite_tex: {
				WGPUTextureFormat fmt = _skr_spv_image_format(res->image_format);
				// sksc records real usage in shape bits 7 (read) / 6 (write) and
				// narrows the WGSL declaration to match. No bits set means the
				// image is never accessed; write-only is the safe layout then.
				bool reads  = (res->shape & SKSC_SHAPE_READ)    != 0;
				bool writes = (res->shape & SKSC_SHAPE_WRITTEN) != 0;
				entry.storageTexture = (WGPUStorageTextureBindingLayout){
					.access        = reads && writes ? WGPUStorageTextureAccess_ReadWrite :
					                 reads           ? WGPUStorageTextureAccess_ReadOnly  :
					                                   WGPUStorageTextureAccess_WriteOnly,
					.format        = fmt,
					.viewDimension = _skr_shape_view_dim(res->shape),
				};
			} break;
			default:
				continue; // QCOM/tile registers never reach the WebGPU backend
		}
		entries[entry_count++] = entry;
	}

	for (uint32_t i = 0; i < meta->sampler_count && entry_count < 64; i++) {
		const sksc_shader_sampler_t* samp = &meta->samplers[i];
		uint8_t stages = samp->stage_bits & stage_mask;
		if (!stages) continue;

		// Comparison samplers are identified via the paired texture's shape
		bool comparison = _skr_sampler_is_comparison(meta, samp->paired_slot);

		entries[entry_count++] = (WGPUBindGroupLayoutEntry){
			.binding    = samp->slot,
			.visibility = _skr_stage_visibility(stages),
			.sampler    = { .type = comparison ? WGPUSamplerBindingType_Comparison : WGPUSamplerBindingType_Filtering },
		};
	}

	WGPUBindGroupLayoutDescriptor desc = {
		.entryCount = entry_count,
		.entries    = entries,
	};
	return wgpuDeviceCreateBindGroupLayout(_skr_wgpu.device, &desc);
}

///////////////////////////////////////////////////////////////////////////////
// Material registry
///////////////////////////////////////////////////////////////////////////////

int32_t _skr_pipeline_register_material(const _skr_pipeline_material_key_t* key) {
	_skr_pipeline_init();
	_skr_mtx_lock(&_registry_mutex);

	int32_t count = _skr_load_acquire(&_material_count);
	int32_t idx   = -1;
	for (int32_t i = 0; i < count; i++) {
		_skr_material_slot_t* slot = _skr_load_acquire(&_materials[i]);
		if (slot->refcount > 0 && !slot->pending_destroy && memcmp(&slot->key, key, sizeof(*key)) == 0) {
			slot->refcount++;
			_skr_mtx_unlock(&_registry_mutex);
			return i;
		}
		if (idx == -1 && slot->refcount == 0 && !slot->pending_destroy)
			idx = i;
	}

	_skr_material_slot_t* mat;
	if (idx >= 0) {
		// Recycled slot: safe to rewrite, no live index refers to it — its old
		// material was destroyed and drained a frame boundary ago
		mat = _skr_load_acquire(&_materials[idx]);
		memset(mat, 0, sizeof(*mat));
	} else {
		if (count >= _SKR_MATERIAL_SLOT_MAX) {
			skr_log(skr_log_critical, "Material registry is full (%d live pipeline states)", count);
			_skr_mtx_unlock(&_registry_mutex);
			return -1;
		}
		idx = count;
		mat = (_skr_material_slot_t*)_skr_calloc(1, sizeof(_skr_material_slot_t));
	}
	mat->key      = *key;
	mat->refcount = 1;

	uint8_t stage_mask = (uint8_t)(skr_stage_vertex | skr_stage_pixel);
	mat->bind_layout = _skr_bind_layout_create(&key->shader->meta, stage_mask);
	WGPUPipelineLayoutDescriptor layout_desc = {
		.bindGroupLayoutCount = 1,
		.bindGroupLayouts     = &mat->bind_layout,
	};
	mat->layout = wgpuDeviceCreatePipelineLayout(_skr_wgpu.device, &layout_desc);

	// Publish: slot contents land before the pointer, pointer before the count
	if (idx == count) {
		_skr_store_release(&_materials[idx], mat);
		_skr_store_release(&_material_count, count + 1);
	}
	_skr_mtx_unlock(&_registry_mutex);
	return idx;
}

void _skr_pipeline_unregister_material(int32_t material_idx) {
	if (material_idx < 0) return;
	_skr_pipeline_init();
	_skr_mtx_lock(&_registry_mutex);
	_skr_material_slot_t* mat = material_idx < _skr_load_acquire(&_material_count)
		? _skr_load_acquire(&_materials[material_idx]) : NULL;
	// Handles stay alive until the next frame boundary — draws recorded this
	// frame may still reference them (see _skr_pipeline_registry_drain)
	if (mat && mat->refcount > 0 && --mat->refcount == 0)
		mat->pending_destroy = true;
	_skr_mtx_unlock(&_registry_mutex);
}

// Lock-free: slots never move once published, and their WGPU handles survive
// until the frame boundary after destruction
static _skr_material_slot_t* _skr_material_slot(int32_t material_idx) {
	if (material_idx < 0 || material_idx >= _skr_load_acquire(&_material_count)) return NULL;
	return _skr_load_acquire(&_materials[material_idx]);
}

const _skr_pipeline_material_key_t* _skr_pipeline_get_key(int32_t material_idx) {
	_skr_material_slot_t* mat = _skr_material_slot(material_idx);
	return mat ? &mat->key : NULL;
}

WGPUBindGroupLayout _skr_pipeline_get_bind_layout(int32_t material_idx) {
	_skr_material_slot_t* mat = _skr_material_slot(material_idx);
	return mat ? mat->bind_layout : NULL;
}


///////////////////////////////////////////////////////////////////////////////
// Vertex format registry
///////////////////////////////////////////////////////////////////////////////

int32_t _skr_pipeline_register_vertformat(const skr_vert_type_t* vert_type) {
	_skr_pipeline_init();
	_skr_mtx_lock(&_registry_mutex);
	int32_t count = _skr_load_acquire(&_vertformat_count);
	for (int32_t i = 0; i < count; i++) {
		_skr_vertformat_slot_t* vf = _skr_load_acquire(&_vertformats[i]);
		if (vf->component_count == vert_type->component_count &&
		    memcmp(vf->components, vert_type->components, sizeof(skr_vert_component_t) * vert_type->component_count) == 0) {
			_skr_mtx_unlock(&_registry_mutex);
			return i;
		}
	}
	if (count >= _SKR_VERTFORMAT_SLOT_MAX) {
		skr_log(skr_log_critical, "Vertex format registry is full (%d formats)", count);
		_skr_mtx_unlock(&_registry_mutex);
		return -1;
	}
	_skr_vertformat_slot_t* slot = (_skr_vertformat_slot_t*)_skr_calloc(1, sizeof(_skr_vertformat_slot_t));
	slot->component_count = vert_type->component_count;
	slot->components      = (skr_vert_component_t*)_skr_malloc(sizeof(skr_vert_component_t) * vert_type->component_count);
	memcpy(slot->components, vert_type->components, sizeof(skr_vert_component_t) * vert_type->component_count);
	slot->attributes      = (WGPUVertexAttribute*)_skr_malloc(sizeof(WGPUVertexAttribute) * vert_type->component_count);
	memcpy(slot->attributes, vert_type->attributes, sizeof(WGPUVertexAttribute) * vert_type->component_count);
	slot->binding_count   = vert_type->binding_count;
	for (uint32_t b = 0; b < vert_type->binding_count; b++)
		slot->strides[b] = vert_type->bindings[b].arrayStride;
	_skr_store_release(&_vertformats[count], slot);
	_skr_store_release(&_vertformat_count,  count + 1);
	_skr_mtx_unlock(&_registry_mutex);
	return count;
}

///////////////////////////////////////////////////////////////////////////////
// Pipeline creation
///////////////////////////////////////////////////////////////////////////////

static WGPUBlendFactor _skr_blend_factor(skr_blend_factor_ f) {
	switch (f) {
		case skr_blend_zero:                     return WGPUBlendFactor_Zero;
		case skr_blend_one:                      return WGPUBlendFactor_One;
		case skr_blend_src_color:                return WGPUBlendFactor_Src;
		case skr_blend_one_minus_src_color:      return WGPUBlendFactor_OneMinusSrc;
		case skr_blend_dst_color:                return WGPUBlendFactor_Dst;
		case skr_blend_one_minus_dst_color:      return WGPUBlendFactor_OneMinusDst;
		case skr_blend_src_alpha:                return WGPUBlendFactor_SrcAlpha;
		case skr_blend_one_minus_src_alpha:      return WGPUBlendFactor_OneMinusSrcAlpha;
		case skr_blend_dst_alpha:                return WGPUBlendFactor_DstAlpha;
		case skr_blend_one_minus_dst_alpha:      return WGPUBlendFactor_OneMinusDstAlpha;
		case skr_blend_constant_color:           return WGPUBlendFactor_Constant;
		case skr_blend_one_minus_constant_color: return WGPUBlendFactor_OneMinusConstant;
		case skr_blend_constant_alpha:           return WGPUBlendFactor_Constant;
		case skr_blend_one_minus_constant_alpha: return WGPUBlendFactor_OneMinusConstant;
		case skr_blend_src_alpha_saturate:       return WGPUBlendFactor_SrcAlphaSaturated;
		case skr_blend_src1_color:               return WGPUBlendFactor_Src1;
		case skr_blend_one_minus_src1_color:     return WGPUBlendFactor_OneMinusSrc1;
		case skr_blend_src1_alpha:               return WGPUBlendFactor_Src1Alpha;
		case skr_blend_one_minus_src1_alpha:     return WGPUBlendFactor_OneMinusSrc1Alpha;
		default:                                 return WGPUBlendFactor_One;
	}
}

static WGPUBlendOperation _skr_blend_op(skr_blend_op_ op) {
	switch (op) {
		case skr_blend_op_subtract:         return WGPUBlendOperation_Subtract;
		case skr_blend_op_reverse_subtract: return WGPUBlendOperation_ReverseSubtract;
		case skr_blend_op_min:              return WGPUBlendOperation_Min;
		case skr_blend_op_max:              return WGPUBlendOperation_Max;
		default:                            return WGPUBlendOperation_Add;
	}
}

static WGPUCompareFunction _skr_compare_fn(skr_compare_ compare) {
	switch (compare) {
		case skr_compare_less:          return WGPUCompareFunction_Less;
		case skr_compare_less_or_eq:    return WGPUCompareFunction_LessEqual;
		case skr_compare_greater:       return WGPUCompareFunction_Greater;
		case skr_compare_greater_or_eq: return WGPUCompareFunction_GreaterEqual;
		case skr_compare_equal:         return WGPUCompareFunction_Equal;
		case skr_compare_not_equal:     return WGPUCompareFunction_NotEqual;
		case skr_compare_never:         return WGPUCompareFunction_Never;
		default:                        return WGPUCompareFunction_Always;
	}
}

static WGPUStencilOperation _skr_stencil_op(skr_stencil_op_ op) {
	switch (op) {
		case skr_stencil_op_zero:            return WGPUStencilOperation_Zero;
		case skr_stencil_op_replace:         return WGPUStencilOperation_Replace;
		case skr_stencil_op_increment_clamp: return WGPUStencilOperation_IncrementClamp;
		case skr_stencil_op_decrement_clamp: return WGPUStencilOperation_DecrementClamp;
		case skr_stencil_op_invert:          return WGPUStencilOperation_Invert;
		case skr_stencil_op_increment_wrap:  return WGPUStencilOperation_IncrementWrap;
		case skr_stencil_op_decrement_wrap:  return WGPUStencilOperation_DecrementWrap;
		default:                             return WGPUStencilOperation_Keep;
	}
}

// Constants for one stage: user spec constants declared in that stage, plus
// the sk_view_index override when the stage's WGSL declares it. Overrides
// with @id(N) are keyed by the decimal id string, per the WebGPU spec.
uint32_t _skr_stage_constants(const sksc_shader_meta_t* meta, const uint32_t* values, skr_stage_ stage, bool has_view_index, uint32_t view_index, WGPUConstantEntry* out, _skr_const_key_t* keys, uint32_t out_max) {
	uint32_t count = 0;
	for (uint32_t i = 0; i < meta->spec_constant_count && count < out_max; i++) {
		const sksc_shader_spec_constant_t* spec = &meta->spec_constants[i];
		if (!(spec->stage_bits & stage)) continue;
		double value;
		switch (spec->type) {
			case sksc_shader_var_float: { float f; memcpy(&f, &values[i], 4); value = f; } break;
			case sksc_shader_var_uint:  value = values[i]; break;
			default:                    { int32_t v; memcpy(&v, &values[i], 4); value = v; } break;
		}
		int len = snprintf(keys[count], sizeof(keys[count]), "%u", spec->constant_id);
		out[count] = (WGPUConstantEntry){ .key = { keys[count], (size_t)len }, .value = value };
		count++;
	}
	if (has_view_index && count < out_max) {
		int len = snprintf(keys[count], sizeof(keys[count]), "%u", (uint32_t)SKSC_WGSL_VIEW_INDEX_SPEC_ID);
		out[count] = (WGPUConstantEntry){ .key = { keys[count], (size_t)len }, .value = (double)view_index };
		count++;
	}
	return count;
}

// Draw thread only: appends to the slot's pipeline cache without taking the
// registry mutex — safe because pipelines are only requested while drawing
WGPURenderPipeline _skr_pipeline_get(int32_t material_idx, const skr_pipeline_pass_key_t* pass, int32_t vertformat_idx) {
	_skr_material_slot_t* mat = _skr_material_slot(material_idx);
	if (mat == NULL || mat->key.shader == NULL) return NULL;

	for (int32_t i = 0; i < mat->pipeline_count; i++) {
		if (mat->pipelines[i].vert_idx == vertformat_idx &&
		    memcmp(&mat->pipelines[i].pass, pass, sizeof(*pass)) == 0)
			return mat->pipelines[i].pipeline;
	}

	const _skr_pipeline_material_key_t* key    = &mat->key;
	const skr_shader_t*                 shader = key->shader;
	const sksc_shader_meta_t*           meta   = &shader->meta;

	// --- Vertex state: match mesh components to shader inputs by semantic ---
	WGPUVertexAttribute    attrs  [SKR_MAX_VERTEX_BUFFERS][32];
	WGPUVertexBufferLayout layouts[SKR_MAX_VERTEX_BUFFERS];
	uint32_t               attr_counts[SKR_MAX_VERTEX_BUFFERS] = {0};
	uint32_t               layout_count = 0;

	_skr_vertformat_slot_t* vf = vertformat_idx >= 0 && vertformat_idx < _skr_load_acquire(&_vertformat_count)
		? _skr_load_acquire(&_vertformats[vertformat_idx]) : NULL;
	if (vf && shader->vertex_stage.shader) {
		for (uint32_t c = 0; c < vf->component_count; c++) {
			const skr_vert_component_t* com = &vf->components[c];
			int32_t location = -1;
			for (int32_t s = 0; s < meta->vertex_input_count; s++) {
				if (meta->vertex_inputs[s].semantic == com->semantic &&
				    meta->vertex_inputs[s].semantic_slot == com->semantic_slot) { location = meta->vertex_inputs[s].location; break; }
			}
			if (location < 0) continue; // mesh supplies data the shader doesn't consume

			uint8_t b = com->binding;
			if (b < SKR_MAX_VERTEX_BUFFERS && attr_counts[b] < 32) {
				attrs[b][attr_counts[b]] = vf->attributes[c];
				attrs[b][attr_counts[b]].shaderLocation = (uint32_t)location;
				attr_counts[b]++;
			}
		}
		for (uint32_t b = 0; b < vf->binding_count && b < SKR_MAX_VERTEX_BUFFERS; b++) {
			layouts[layout_count++] = (WGPUVertexBufferLayout){
				.arrayStride    = vf->strides[b],
				.stepMode       = WGPUVertexStepMode_Vertex,
				.attributeCount = attr_counts[b],
				.attributes     = attrs[b],
			};
		}
	}

	// --- Stage constants ---
	WGPUConstantEntry vs_constants[SKR_MAX_SPEC_CONSTANTS + 1];
	WGPUConstantEntry fs_constants[SKR_MAX_SPEC_CONSTANTS + 1];
	_skr_const_key_t  vs_keys     [SKR_MAX_SPEC_CONSTANTS + 1];
	_skr_const_key_t  fs_keys     [SKR_MAX_SPEC_CONSTANTS + 1];
	uint32_t vs_const_count = _skr_stage_constants(meta, key->spec_constant_values, skr_stage_vertex, shader->vertex_stage.has_view_index, pass->view_index, vs_constants, vs_keys, SKR_MAX_SPEC_CONSTANTS + 1);
	uint32_t fs_const_count = _skr_stage_constants(meta, key->spec_constant_values, skr_stage_pixel,  shader->pixel_stage .has_view_index, pass->view_index, fs_constants, fs_keys, SKR_MAX_SPEC_CONSTANTS + 1);

	// --- Fragment state ---
	WGPUBlendState blend = {
		.color = { .operation = _skr_blend_op(key->blend_state.color_op), .srcFactor = _skr_blend_factor(key->blend_state.src_color_factor), .dstFactor = _skr_blend_factor(key->blend_state.dst_color_factor) },
		.alpha = { .operation = _skr_blend_op(key->blend_state.alpha_op), .srcFactor = _skr_blend_factor(key->blend_state.src_alpha_factor), .dstFactor = _skr_blend_factor(key->blend_state.dst_alpha_factor) },
	};
	bool blend_enabled = !(key->blend_state.src_color_factor == skr_blend_zero && key->blend_state.dst_color_factor == skr_blend_zero &&
	                       key->blend_state.src_alpha_factor == skr_blend_zero && key->blend_state.dst_alpha_factor == skr_blend_zero) &&
	                     !(key->blend_state.src_color_factor == skr_blend_one  && key->blend_state.dst_color_factor == skr_blend_zero &&
	                       key->blend_state.src_alpha_factor == skr_blend_one  && key->blend_state.dst_alpha_factor == skr_blend_zero);

	WGPUColorWriteMask write_mask = WGPUColorWriteMask_None;
	if (key->write_mask & skr_write_r) write_mask |= WGPUColorWriteMask_Red;
	if (key->write_mask & skr_write_g) write_mask |= WGPUColorWriteMask_Green;
	if (key->write_mask & skr_write_b) write_mask |= WGPUColorWriteMask_Blue;
	if (key->write_mask & skr_write_a) write_mask |= WGPUColorWriteMask_Alpha;

	WGPUColorTargetState target = {
		.format    = pass->color_format,
		.blend     = blend_enabled ? &blend : NULL,
		.writeMask = write_mask,
	};
	// {NULL, WGPU_STRLEN} is the nil string: auto-select the module's single
	// entry point (each WGSL stage was compiled separately)
	WGPUStringView auto_entry = { NULL, WGPU_STRLEN };

	WGPUFragmentState fragment = {
		.module        = shader->pixel_stage.shader,
		.entryPoint    = auto_entry,
		.constantCount = fs_const_count,
		.constants     = fs_constants,
		.targetCount   = 1,
		.targets       = &target,
	};

	// --- Depth/stencil ---
	// Stencil state must stay at defaults when the depth format has no
	// stencil aspect, or validation rejects the pipeline
	bool has_stencil = pass->depth_format == WGPUTextureFormat_Depth24PlusStencil8 ||
	                   pass->depth_format == WGPUTextureFormat_Depth32FloatStencil8;
	WGPUDepthStencilState depth_stencil = {
		.format              = pass->depth_format,
		.depthWriteEnabled   = (key->write_mask & skr_write_depth) ? WGPUOptionalBool_True : WGPUOptionalBool_False,
		.depthCompare        = key->depth_test == skr_compare_none ? WGPUCompareFunction_Always : _skr_compare_fn(key->depth_test),
		.stencilFront = {
			.compare     = key->stencil_front.compare == skr_compare_none ? WGPUCompareFunction_Always : _skr_compare_fn(key->stencil_front.compare),
			.failOp      = _skr_stencil_op(key->stencil_front.fail_op),
			.depthFailOp = _skr_stencil_op(key->stencil_front.depth_fail_op),
			.passOp      = _skr_stencil_op(key->stencil_front.pass_op),
		},
		.stencilBack = {
			.compare     = key->stencil_back.compare == skr_compare_none ? WGPUCompareFunction_Always : _skr_compare_fn(key->stencil_back.compare),
			.failOp      = _skr_stencil_op(key->stencil_back.fail_op),
			.depthFailOp = _skr_stencil_op(key->stencil_back.depth_fail_op),
			.passOp      = _skr_stencil_op(key->stencil_back.pass_op),
		},
		.stencilReadMask  = key->stencil_front.compare_mask ? key->stencil_front.compare_mask : 0xFFFFFFFF,
		.stencilWriteMask = key->stencil_front.write_mask   ? key->stencil_front.write_mask   : 0xFFFFFFFF,
	};
	if (!has_stencil) {
		WGPUStencilFaceState keep = { .compare = WGPUCompareFunction_Always, .failOp = WGPUStencilOperation_Keep, .depthFailOp = WGPUStencilOperation_Keep, .passOp = WGPUStencilOperation_Keep };
		depth_stencil.stencilFront    = keep;
		depth_stencil.stencilBack     = keep;
		depth_stencil.stencilReadMask = 0xFFFFFFFF;
		depth_stencil.stencilWriteMask= 0xFFFFFFFF;
	}

	WGPURenderPipelineDescriptor desc = {
		.label  = { meta->name, strnlen(meta->name, sizeof(meta->name)) },
		.layout = mat->layout,
		.vertex = {
			.module        = shader->vertex_stage.shader,
			.entryPoint    = auto_entry,
			.constantCount = vs_const_count,
			.constants     = vs_constants,
			.bufferCount   = layout_count,
			.buffers       = layouts,
		},
		.primitive = {
			.topology         = key->wireframe ? WGPUPrimitiveTopology_LineList : WGPUPrimitiveTopology_TriangleList,
			.frontFace        = WGPUFrontFace_CCW,
			.cullMode         = key->cull == skr_cull_none ? WGPUCullMode_None : key->cull == skr_cull_front ? WGPUCullMode_Front : WGPUCullMode_Back,
			.unclippedDepth   = key->depth_clamp && _skr_wgpu.feat_depth_clip,
		},
		.depthStencil = pass->depth_format != WGPUTextureFormat_Undefined ? &depth_stencil : NULL,
		.multisample = {
			.count                  = pass->sample_count > 0 ? pass->sample_count : 1,
			.mask                   = 0xFFFFFFFF,
			.alphaToCoverageEnabled = key->alpha_to_coverage,
		},
		.fragment = pass->color_format != WGPUTextureFormat_Undefined && shader->pixel_stage.shader ? &fragment : NULL,
	};

	WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(_skr_wgpu.device, &desc);
	if (pipeline == NULL) {
		skr_log(skr_log_critical, "Failed to create render pipeline for '%s'", meta->name);
		return NULL;
	}

	if (mat->pipeline_count == mat->pipeline_capacity) {
		mat->pipeline_capacity = mat->pipeline_capacity == 0 ? 4 : mat->pipeline_capacity * 2;
		mat->pipelines = (_skr_pipeline_entry_t*)_skr_realloc(mat->pipelines, sizeof(_skr_pipeline_entry_t) * mat->pipeline_capacity);
	}
	mat->pipelines[mat->pipeline_count++] = (_skr_pipeline_entry_t){ .pass = *pass, .vert_idx = vertformat_idx, .pipeline = pipeline };
	return pipeline;
}
