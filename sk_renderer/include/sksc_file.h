// SPDX-License-Identifier: MIT
// The authors below grant copyright rights under the MIT license:
// Copyright (c) 2025 Nick Klingensmith
// Copyright (c) 2025 Qualcomm Technologies, Inc.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Symbol visibility macros
#if defined(_WIN32) || defined(_WIN64)
	#ifdef SKR_BUILD_SHARED
		#define SKSC_API __declspec(dllexport)
	#else
		#define SKSC_API
	#endif
#elif defined(__GNUC__) || defined(__clang__)
	#define SKSC_API __attribute__((visibility("default")))
#else
	#define SKSC_API
#endif

///////////////////////////////////////////////////////////////////////////////

typedef enum {
	skr_vertex_fmt_none,
	skr_vertex_fmt_f64,
	skr_vertex_fmt_f32,
	skr_vertex_fmt_f16,
	skr_vertex_fmt_i32,
	skr_vertex_fmt_i16,
	skr_vertex_fmt_i8,
	skr_vertex_fmt_i32_normalized,
	skr_vertex_fmt_i16_normalized,
	skr_vertex_fmt_i8_normalized,
	skr_vertex_fmt_ui32,
	skr_vertex_fmt_ui16,
	skr_vertex_fmt_ui8,
	skr_vertex_fmt_ui32_normalized,
	skr_vertex_fmt_ui16_normalized,
	skr_vertex_fmt_ui8_normalized,
} skr_vertex_fmt_;

typedef enum {
	sksc_shader_var_none,
	sksc_shader_var_int,
	sksc_shader_var_uint,
	sksc_shader_var_uint8,
	sksc_shader_var_float,
	sksc_shader_var_double,
} sksc_shader_var_;

typedef enum {
	skr_semantic_none,
	skr_semantic_position,
	skr_semantic_texcoord,
	skr_semantic_normal,
	skr_semantic_binormal,
	skr_semantic_tangent,
	skr_semantic_color,
	skr_semantic_psize,
	skr_semantic_blendweight,
	skr_semantic_blendindices,
} skr_semantic_;

typedef enum {
	skr_stage_vertex  = 1 << 0,
	skr_stage_pixel   = 1 << 1,
	skr_stage_compute = 1 << 2,
} skr_stage_;

typedef enum {
	skr_register_default,
	skr_register_vertex,
	skr_register_index,
	skr_register_constant,
	skr_register_texture,
	skr_register_read_buffer,
	skr_register_readwrite,
	skr_register_readwrite_tex,
} skr_register_;

typedef enum {
	skr_shader_lang_hlsl,
	skr_shader_lang_spirv,
	skr_shader_lang_glsl,
	skr_shader_lang_glsl_es,
	skr_shader_lang_glsl_web,
} skr_shader_lang_;

typedef enum {
	sksc_result_unknown       =  0,
	sksc_result_success       =  1,
	sksc_result_out_of_memory = -1,
	sksc_result_bad_format    = -2,
	sksc_result_old_version   = -3,
	sksc_result_corrupt_data  = -4,
} sksc_result_;

typedef struct {
	skr_vertex_fmt_ format;
	uint8_t         count;
	skr_semantic_   semantic;
	uint8_t         semantic_slot;
	uint8_t         binding;
} skr_vert_component_t;

typedef struct {
	uint16_t slot;
	// of type skr_stage_
	uint8_t  stage_bits; 
	// of type skr_register_
	uint8_t  register_type;
} skr_bind_t;

typedef struct {
	char     name [32];
	uint64_t name_hash;
	char     extra[64];
	char     type_name[32]; // For struct types, the name of the struct (e.g., "circle_mask_t")
	uint32_t offset;
	uint32_t size;
	// of type sksc_shader_var_
	uint16_t type;
	uint16_t type_count;
} sksc_shader_var_t;

typedef struct {
	char              name[32];
	uint64_t          name_hash;
	skr_bind_t        bind;
	uint8_t           space;
	uint32_t          size;
	void             *defaults;
	uint32_t          var_count;
	sksc_shader_var_t *vars;
} sksc_shader_buffer_t;

typedef struct {
	char       name [32];
	uint64_t   name_hash;
	char       value[64];
	char       tags [64];
	skr_bind_t bind;
	uint32_t   element_size; // For StructuredBuffer<T>, the size of T in bytes
} sksc_shader_resource_t;

typedef struct {
	int32_t total;
	int32_t tex_read;
	int32_t dynamic_flow;
} sksc_shader_ops_t;

typedef struct {
	char                   name[256];
	uint32_t               buffer_count;
	sksc_shader_buffer_t   *buffers;
	uint32_t               resource_count;
	sksc_shader_resource_t *resources;
	int32_t                references;
	int32_t                global_buffer_id;
	skr_vert_component_t  *vertex_inputs;
	int32_t                vertex_input_count;
	sksc_shader_ops_t      ops_vertex;
	sksc_shader_ops_t      ops_pixel;
} sksc_shader_meta_t;

typedef struct {
	skr_shader_lang_ language;
	skr_stage_       stage;
	uint32_t         code_size;
	void            *code;
} sksc_shader_file_stage_t;

typedef struct {
	sksc_shader_meta_t       *meta;
	uint32_t                  stage_count;
	sksc_shader_file_stage_t *stages;
} sksc_shader_file_t;


///////////////////////////////////////////////////////////////////////////////

SKSC_API bool                     sksc_shader_file_verify         (const void *file_memory, uint32_t file_size, uint16_t *out_version, char *out_name, uint32_t out_name_size);
SKSC_API sksc_result_             sksc_shader_file_load_memory    (const void *file_memory, uint32_t file_size, sksc_shader_file_t *out_file);
SKSC_API void                     sksc_shader_file_destroy        (sksc_shader_file_t *ref_file);

SKSC_API skr_bind_t               sksc_shader_meta_get_bind       (const sksc_shader_meta_t*     meta, const char *name);
SKSC_API int32_t                  sksc_shader_meta_get_var_count  (const sksc_shader_meta_t*     meta);
SKSC_API int32_t                  sksc_shader_meta_get_var_index  (const sksc_shader_meta_t*     meta, const char *name);
SKSC_API int32_t                  sksc_shader_meta_get_var_index_h(const sksc_shader_meta_t*     meta, uint64_t name_hash);
SKSC_API const sksc_shader_var_t* sksc_shader_meta_get_var_info   (const sksc_shader_meta_t*     meta, int32_t  var_index);
SKSC_API void                     sksc_shader_meta_reference      (      sksc_shader_meta_t* ref_meta);
SKSC_API void                     sksc_shader_meta_release        (      sksc_shader_meta_t* ref_meta);

#ifdef __cplusplus
}
#endif
