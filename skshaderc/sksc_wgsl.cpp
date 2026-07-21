// WGSL stage generation for the WebGPU backend (docs/webgpu-backend-plan.md).
//
// Source HLSL/SVSL stays untouched: all WGSL divergence happens here as SPIR-V
// transforms on a second, split-sampler compile of the same source, followed
// by Tint's SPIR-V reader -> WGSL writer.
//
// The output contract (consumed by sk_renderer's wgpu backend):
// - Bindings are @group(0), numbered by the register-shift scheme: b -> 0+,
//   t -> 100+, u -> 200+, input attachments -> 300+, s -> 400+.
// - SV_ViewID / BuiltIn ViewIndex becomes a pipeline-overridable constant
//   (SPIR-V spec constant) named sk_view_index with reserved id
//   SKSC_WGSL_VIEW_INDEX_SPEC_ID and default 0; the runtime renders layered
//   targets one pass per layer, specializing the pipeline per view.
// - Standalone samplers are recorded in meta->samplers, each paired with the
//   texture resource it samples so the runtime can apply that texture's
//   sampler settings (matching Vulkan's combined-sampler semantics).

#include "_sksc.h"

#include <spirv_reflect.h>

#include <stdlib.h>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#ifdef SKSC_HAS_TINT
#include "src/tint/api/tint.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/writer/writer.h"
#endif

///////////////////////////////////////////
// SPIR-V word-level helpers             //
///////////////////////////////////////////

// SPIR-V opcodes and enums used below, by number to avoid a header dependency
static const uint32_t spv_op_extension     = 10;
static const uint32_t spv_op_entry_point   = 15;
static const uint32_t spv_op_capability    = 17;
static const uint32_t spv_op_name          = 5;
static const uint32_t spv_op_type_int      = 21;
static const uint32_t spv_op_type_pointer  = 32;
static const uint32_t spv_op_spec_constant = 50;
static const uint32_t spv_op_variable      = 59;
static const uint32_t spv_op_load          = 61;
static const uint32_t spv_op_sampled_image = 86;
static const uint32_t spv_op_decorate      = 71;
static const uint32_t spv_decoration_spec_id = 1;
static const uint32_t spv_decoration_builtin = 11;
static const uint32_t spv_decoration_binding = 33;
static const uint32_t spv_builtin_layer      = 9;
static const uint32_t spv_builtin_view_index = 4440;
static const uint32_t spv_capability_multiview = 4439;
static const uint32_t spv_storage_input      = 1;
static const uint32_t spv_storage_private    = 6;
static const size_t   spv_header_words       = 5;

static const uint32_t spv_op_type_float      = 22;
static const uint32_t spv_op_type_vector     = 23;
static const uint32_t spv_op_type_image      = 25;
static const uint32_t spv_op_constant        = 43;
static const uint32_t spv_op_function        = 54;
static const uint32_t spv_op_image_fetch     = 95;
static const uint32_t spv_op_image_read      = 98;
static const uint32_t spv_op_vector_shuffle  = 79;
static const uint32_t spv_op_convert_f_to_s  = 110;
static const uint32_t spv_dim_2d             = 1;
static const uint32_t spv_dim_subpass_data   = 6;
static const uint32_t spv_builtin_frag_coord = 15;
static const uint32_t spv_capability_input_attachment  = 40;
static const uint32_t spv_decoration_input_attach_idx  = 43;
static const uint32_t spv_storage_uniform_constant     = 0;

static const uint32_t spv_op_member_decorate      = 72;
static const uint32_t spv_op_module_processed     = 330;
static const uint32_t spv_op_image_write          = 99;
static const uint32_t spv_op_image_sparse_read    = 307;
static const uint32_t spv_op_image_texel_pointer  = 60;
static const uint32_t spv_decoration_non_writable = 24;
static const uint32_t spv_decoration_non_readable = 25;

// Word count of a null-terminated literal string starting at `at`
static uint32_t _spv_string_words(const uint32_t *words, uint32_t at, uint32_t end) {
	for (uint32_t i = at; i < end; i++) {
		const uint8_t *bytes = (const uint8_t *)&words[i];
		if (bytes[0] == 0 || bytes[1] == 0 || bytes[2] == 0 || bytes[3] == 0)
			return (i - at) + 1;
	}
	return end - at;
}

///////////////////////////////////////////
// ViewIndex -> sk_view_index rewrite    //
///////////////////////////////////////////

// Rewrites the BuiltIn ViewIndex input into a module-private variable
// initialized from a new spec constant (id SKSC_WGSL_VIEW_INDEX_SPEC_ID,
// default 0), and strips the MultiView capability. Keeping the variable and
// its loads intact means no use-site rewriting is needed. Returns false on a
// structural error worth failing the compile over.
static bool _wgsl_view_index_to_spec(std::vector<uint32_t> &spirv, skr_stage_ stage) {
	if (spirv.size() <= spv_header_words) return true;

	// --- Pass 1: find the ViewIndex variable and related instructions ---
	uint32_t view_var    = 0;
	bool     has_spec999 = false;
	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;

		if (opcode == spv_op_decorate && count >= 4 && spirv[i+2] == spv_decoration_builtin) {
			if (spirv[i+3] == spv_builtin_view_index)
				view_var = spirv[i+1];
			if (spirv[i+3] == spv_builtin_layer && stage == skr_stage_vertex) {
				sksc_log(sksc_log_level_err, "Vertex stage writes SV_RenderTargetArrayIndex — the instanced-stereo pattern cannot be expressed in WGSL. Use SV_ViewID instead.");
				return false;
			}
		}
		if (opcode == spv_op_decorate && count >= 4 &&
		    spirv[i+2] == spv_decoration_spec_id && spirv[i+3] == SKSC_WGSL_VIEW_INDEX_SPEC_ID)
			has_spec999 = true;
		i += count;
	}
	if (view_var == 0) return true; // No multiview use, nothing to do
	if (has_spec999) {
		sksc_log(sksc_log_level_err, "Spec constant id %d is reserved for sk_view_index when targeting WGSL, pick another [[vk::constant_id]]", SKSC_WGSL_VIEW_INDEX_SPEC_ID);
		return false;
	}

	// --- Pass 2: find the variable's types ---
	uint32_t ptr_type  = 0; // OpTypePointer Input uint used by the variable
	uint32_t uint_type = 0; // OpTypeInt 32 0
	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;
		if (opcode == spv_op_variable && count >= 4 && spirv[i+2] == view_var)
			ptr_type = spirv[i+1];
		i += count;
	}
	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;
		if (opcode == spv_op_type_pointer && count == 4 && spirv[i+1] == ptr_type)
			uint_type = spirv[i+3];
		i += count;
	}
	if (ptr_type == 0 || uint_type == 0) {
		sksc_log(sksc_log_level_err, "Couldn't locate the ViewIndex variable's type while preparing WGSL output");
		return false;
	}

	// --- Allocate new ids: the spec constant and a Private pointer type ---
	uint32_t spec_id     = spirv[3];
	uint32_t priv_ptr_id = spirv[3] + 1;
	spirv[3] += 2;

	// --- Pass 3: rebuild the module ---
	std::vector<uint32_t> out;
	out.reserve(spirv.size() + 16);
	out.insert(out.end(), spirv.begin(), spirv.begin() + spv_header_words);

	bool name_emitted = false, decor_emitted = false, const_emitted = false;
	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;

		// Strip the multiview capability + extension
		if (opcode == spv_op_capability && count == 2 && spirv[i+1] == spv_capability_multiview) { i += count; continue; }
		if (opcode == spv_op_extension && count >= 2 &&
		    strncmp((const char *)&spirv[i+1], "SPV_KHR_multi_view", (count-1)*4) == 0) { i += count; continue; }

		// Remove the variable from the entry point's interface list — it's no
		// longer an Input, and SPIR-V <1.4 interfaces list only Input/Output
		if (opcode == spv_op_entry_point) {
			uint32_t name_at    = (uint32_t)i + 3;
			uint32_t name_words = _spv_string_words(spirv.data(), name_at, (uint32_t)(i + count));
			out.push_back(spirv[i]); // patched below
			size_t inst_start = out.size() - 1;
			for (uint32_t w = 1; w < count; w++) {
				bool is_interface_id = (i + w) >= (size_t)(name_at + name_words);
				if (is_interface_id && spirv[i+w] == view_var) continue;
				out.push_back(spirv[i+w]);
			}
			uint32_t new_count = (uint32_t)(out.size() - inst_start);
			out[inst_start] = (new_count << 16) | opcode;
			i += count;
			continue;
		}

		// SPIR-V module layout: debug names (7b) precede OpModuleProcessed
		// (7c), which precedes annotations (8). Emit our OpName at the first
		// instruction that ends the names section, and the SpecId decoration
		// separately at the start of the annotations section.
		if (!name_emitted && (opcode == spv_op_module_processed || opcode == spv_op_decorate || opcode == spv_op_member_decorate)) {
			name_emitted = true;
			const char name[16] = "sk_view_index"; // 13 chars + NUL, padded to 4 words
			out.push_back((6u << 16) | spv_op_name);
			out.push_back(spec_id);
			const uint32_t *name_words = (const uint32_t *)name;
			out.insert(out.end(), name_words, name_words + 4);
		}
		if (!decor_emitted && (opcode == spv_op_decorate || opcode == spv_op_member_decorate)) {
			decor_emitted = true;
			// OpDecorate %spec SpecId SKSC_WGSL_VIEW_INDEX_SPEC_ID
			out.push_back((4u << 16) | spv_op_decorate);
			out.push_back(spec_id);
			out.push_back(spv_decoration_spec_id);
			out.push_back(SKSC_WGSL_VIEW_INDEX_SPEC_ID);
		}

		// Swap the variable to Private with the spec constant as initializer
		if (opcode == spv_op_variable && count >= 4 && spirv[i+2] == view_var) {
			out.push_back((5u << 16) | spv_op_variable);
			out.push_back(priv_ptr_id);
			out.push_back(view_var);
			out.push_back(spv_storage_private);
			out.push_back(spec_id);
			i += count;
			continue;
		}

		// Drop the variable's builtin-related decorations (BuiltIn, Flat, ...)
		if (opcode == spv_op_decorate && count >= 3 && spirv[i+1] == view_var) { i += count; continue; }

		out.insert(out.end(), spirv.begin() + i, spirv.begin() + i + count);

		// After the uint type: emit the spec constant and Private pointer type
		if (!const_emitted && opcode == spv_op_type_int && count == 4 &&
		    spirv[i+1] == uint_type && spirv[i+2] == 32 && spirv[i+3] == 0) {
			const_emitted = true;
			out.push_back((4u << 16) | spv_op_spec_constant);
			out.push_back(uint_type);
			out.push_back(spec_id);
			out.push_back(0); // default view index 0
			out.push_back((4u << 16) | spv_op_type_pointer);
			out.push_back(priv_ptr_id);
			out.push_back(spv_storage_private);
			out.push_back(uint_type);
		}

		i += count;
	}

	if (!const_emitted || !name_emitted || !decor_emitted) {
		sksc_log(sksc_log_level_err, "Failed to place sk_view_index while preparing WGSL output");
		return false;
	}

	spirv = std::move(out);
	return true;
}

///////////////////////////////////////////
// Sampler reflection + pairing          //
///////////////////////////////////////////

// Maps each standalone sampler binding to the texture binding it samples, by
// tracing OpSampledImage's operands back through OpLoad to global variables.
// Bindings here are post-remap (t+100, s+400).
static void _wgsl_pair_samplers(const std::vector<uint32_t> &spirv, std::unordered_map<uint32_t, uint32_t> *out_sampler_to_tex) {
	std::unordered_map<uint32_t, uint32_t> var_binding; // variable id -> binding
	std::unordered_map<uint32_t, uint32_t> load_var;    // load result id -> variable id

	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;

		if (opcode == spv_op_decorate && count >= 4 && spirv[i+2] == spv_decoration_binding)
			var_binding[spirv[i+1]] = spirv[i+3];
		else if (opcode == spv_op_load && count >= 4)
			load_var[spirv[i+2]] = spirv[i+3];
		else if (opcode == spv_op_sampled_image && count >= 5) {
			auto img  = load_var.find(spirv[i+3]);
			auto samp = load_var.find(spirv[i+4]);
			if (img != load_var.end() && samp != load_var.end()) {
				auto img_bind  = var_binding.find(img ->second);
				auto samp_bind = var_binding.find(samp->second);
				if (img_bind != var_binding.end() && samp_bind != var_binding.end()) {
					auto existing = out_sampler_to_tex->find(samp_bind->second);
					if (existing == out_sampler_to_tex->end()) {
						(*out_sampler_to_tex)[samp_bind->second] = img_bind->second;
					} else if (existing->second != img_bind->second) {
						sksc_log(sksc_log_level_warn, "A sampler samples multiple textures; on WebGPU it inherits sampler settings from the first one only");
					}
				}
			}
		}
		i += count;
	}
}

// Where a combined image sampler's synthesized WGSL sampler lands: the same
// s-register + 400 scheme the glslang split-sampler path uses
static uint32_t _wgsl_combined_sampler_slot(uint32_t combined_binding) {
	return combined_binding >= SKSC_SLOT_TEXTURE
		? SKSC_SLOT_SAMPLER + (combined_binding - SKSC_SLOT_TEXTURE)
		: SKSC_SLOT_SAMPLER + combined_binding;
}

// Reflects standalone SAMPLER bindings from the split-sampler SPIR-V into
// meta->samplers, and cross-checks that buffer/texture bindings match the
// merged compile's meta (the two compiles must agree or runtime binding by
// meta slot would silently break). COMBINED image samplers (the SVSL backend
// emits these) get a synthesized sampler record — Tint splits them when
// reading, and `out_combined` returns their bindings so the caller can tell
// Tint where to put the split-off sampler.
static bool _wgsl_reflect_samplers(const std::vector<uint32_t> &spirv, skr_stage_ stage, sksc_shader_meta_t *ref_meta, std::vector<uint32_t> *out_combined) {
	spv_reflect::ShaderModule module(spirv.size() * sizeof(uint32_t), spirv.data());
	if (module.GetResult() != SPV_REFLECT_RESULT_SUCCESS) {
		sksc_log(sksc_log_level_err, "[SPIRV-Reflect] Failed to reflect the WGSL stage variant");
		return false;
	}

	uint32_t count = 0;
	module.EnumerateDescriptorBindings(&count, nullptr);
	std::vector<SpvReflectDescriptorBinding*> bindings(count);
	module.EnumerateDescriptorBindings(&count, bindings.data());

	std::unordered_map<uint32_t, uint32_t> sampler_to_tex;
	_wgsl_pair_samplers(spirv, &sampler_to_tex);

	for (SpvReflectDescriptorBinding *binding : bindings) {
		if (binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER ||
		    binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
			bool combined = binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			char name[64];
			snprintf(name, sizeof(name), combined ? "%s_sampler" : "%s", binding->name ? binding->name : "");
			if (combined && out_combined)
				out_combined->push_back(binding->binding);

			// Find or add the sampler record, merging stages
			int32_t id = -1;
			for (uint32_t s = 0; s < ref_meta->sampler_count; s++)
				if (strcmp(ref_meta->samplers[s].name, name) == 0) { id = (int32_t)s; break; }
			if (id == -1) {
				ref_meta->samplers = (sksc_shader_sampler_t *)realloc(ref_meta->samplers, sizeof(sksc_shader_sampler_t) * (ref_meta->sampler_count + 1));
				ref_meta->samplers_owned = true;
				id = (int32_t)ref_meta->sampler_count;
				ref_meta->sampler_count += 1;
				memset(&ref_meta->samplers[id], 0, sizeof(sksc_shader_sampler_t));
				strncpy(ref_meta->samplers[id].name, name, sizeof(ref_meta->samplers[id].name) - 1);

				if (combined) {
					// Tint splits the pair on read; the sampler goes where
					// sampler_mappings routes it, paired with its own texture
					ref_meta->samplers[id].slot        = (uint16_t)_wgsl_combined_sampler_slot(binding->binding);
					ref_meta->samplers[id].paired_slot = (uint16_t)binding->binding;
				} else {
					ref_meta->samplers[id].slot        = (uint16_t)binding->binding;
					ref_meta->samplers[id].paired_slot = 0xFFFF;

					auto pair = sampler_to_tex.find(binding->binding);
					if (pair != sampler_to_tex.end())
						ref_meta->samplers[id].paired_slot = (uint16_t)pair->second;
					else
						sksc_log(sksc_log_level_warn, "Sampler '%s' isn't paired with any texture; WebGPU builds will bind default sampler settings for it", name);
				}
			}
			ref_meta->samplers[id].stage_bits |= stage;
		} else if (binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
		           binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
			// Cross-check against the merged compile's meta by name
			const char *name = binding->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER
				? (binding->type_description && binding->type_description->type_name ? binding->type_description->type_name : (binding->name ? binding->name : ""))
				: (binding->name ? binding->name : "");
			bool found = false, matches = false;
			for (uint32_t b = 0; b < ref_meta->buffer_count && !found; b++)
				if (strcmp(ref_meta->buffers[b].name, name) == 0) { found = true; matches = ref_meta->buffers[b].bind.slot == binding->binding; }
			for (uint32_t r = 0; r < ref_meta->resource_count && !found; r++)
				if (strcmp(ref_meta->resources[r].name, name) == 0) { found = true; matches = ref_meta->resources[r].bind.slot == binding->binding; }
			if (found && !matches) {
				sksc_log(sksc_log_level_err, "Binding for '%s' differs between the Vulkan and WGSL compiles; give it an explicit register()", name);
				return false;
			}
		}
	}
	return true;
}

///////////////////////////////////////////
// SubpassInput -> sampled texture       //
///////////////////////////////////////////

// WebGPU has no subpass inputs — the runtime lowers postfx chains into
// sequential render passes that bind the previous stage as a plain texture.
// This rewrites SubpassData images into 2D sampled textures fetched at the
// fragment's own pixel coordinate, which is exactly subpass-load semantics.
// Multisampled subpass inputs keep their Sample operand. Bindings stay at the
// input-attachment slots (300 + index) the register remap already assigned.
static bool _wgsl_subpass_to_texture(std::vector<uint32_t> &spirv, skr_stage_ stage) {
	if (stage != skr_stage_pixel || spirv.size() <= spv_header_words) return true;

	// --- Pass 1: subpass image types/vars, plus existing types we can reuse ---
	std::unordered_set<uint32_t> img_types, ptr_types, vars;
	uint32_t frag_coord = 0;
	uint32_t t_float = 0, t_v4f = 0, t_v2f = 0, t_int = 0, t_v2i = 0, t_fc_ptr = 0, c_int0 = 0;

	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;

		if      (opcode == spv_op_type_image && count >= 9 && spirv[i+3] == spv_dim_subpass_data) img_types.insert(spirv[i+1]);
		else if (opcode == spv_op_type_float  && count == 3 && spirv[i+2] == 32) t_float = spirv[i+1];
		else if (opcode == spv_op_type_int    && count == 4 && spirv[i+2] == 32 && spirv[i+3] == 1) t_int = spirv[i+1];
		else if (opcode == spv_op_type_vector && count == 4) {
			if (t_float && spirv[i+2] == t_float && spirv[i+3] == 4) t_v4f = spirv[i+1];
			if (t_float && spirv[i+2] == t_float && spirv[i+3] == 2) t_v2f = spirv[i+1];
			if (t_int   && spirv[i+2] == t_int   && spirv[i+3] == 2) t_v2i = spirv[i+1];
		}
		else if (opcode == spv_op_constant && count == 4 && t_int && spirv[i+1] == t_int && spirv[i+3] == 0) c_int0 = spirv[i+2];
		else if (opcode == spv_op_type_pointer && count == 4) {
			if (img_types.count(spirv[i+3]) && spirv[i+2] == spv_storage_uniform_constant) ptr_types.insert(spirv[i+1]);
			if (t_v4f && spirv[i+3] == t_v4f && spirv[i+2] == spv_storage_input) t_fc_ptr = spirv[i+1];
		}
		else if (opcode == spv_op_variable && count >= 4 && ptr_types.count(spirv[i+1])) vars.insert(spirv[i+2]);
		else if (opcode == spv_op_decorate && count >= 4 && spirv[i+2] == spv_decoration_builtin && spirv[i+3] == spv_builtin_frag_coord)
			frag_coord = spirv[i+1];
		i += count;
	}
	if (vars.empty()) return true;

	// --- Allocate ids for anything the module doesn't already have ---
	uint32_t next_id = spirv[3];
	bool need_float  = t_float  == 0; if (need_float)  t_float  = next_id++;
	bool need_v4f    = t_v4f    == 0; if (need_v4f)    t_v4f    = next_id++;
	bool need_v2f    = t_v2f    == 0; if (need_v2f)    t_v2f    = next_id++;
	bool need_int    = t_int    == 0; if (need_int)    t_int    = next_id++;
	bool need_v2i    = t_v2i    == 0; if (need_v2i)    t_v2i    = next_id++;
	bool need_int0   = c_int0   == 0; if (need_int0)   c_int0   = next_id++;
	bool need_fc     = frag_coord == 0;
	bool need_fc_ptr = need_fc && t_fc_ptr == 0;
	if (need_fc_ptr) t_fc_ptr   = next_id++;
	if (need_fc)     frag_coord = next_id++;

	// --- Pass 2: rebuild ---
	std::vector<uint32_t> out;
	out.reserve(spirv.size() + 64);
	out.insert(out.end(), spirv.begin(), spirv.begin() + spv_header_words);

	std::unordered_map<uint32_t, uint32_t> load_var; // OpLoad result -> subpass var
	bool decor_emitted = false, globals_emitted = false;
	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;

		// Strip the InputAttachment capability and index decorations
		if (opcode == spv_op_capability && count == 2 && spirv[i+1] == spv_capability_input_attachment) { i += count; continue; }
		if (opcode == spv_op_decorate && count >= 3 && spirv[i+2] == spv_decoration_input_attach_idx)   { i += count; continue; }

		// New FragCoord joins the entry point's interface
		if (opcode == spv_op_entry_point && need_fc) {
			out.push_back(((count + 1) << 16) | opcode);
			for (uint32_t w = 1; w < count; w++) out.push_back(spirv[i+w]);
			out.push_back(frag_coord);
			i += count;
			continue;
		}

		// FragCoord's BuiltIn decoration lands at the head of the annotations
		if (!decor_emitted && (opcode == spv_op_decorate || opcode == spv_op_member_decorate)) {
			decor_emitted = true;
			if (need_fc) {
				out.push_back((4u << 16) | spv_op_decorate);
				out.push_back(frag_coord);
				out.push_back(spv_decoration_builtin);
				out.push_back(spv_builtin_frag_coord);
			}
		}

		// Missing types/constants/the FragCoord variable go right before the
		// first function, at the tail of the globals section
		if (!globals_emitted && opcode == spv_op_function) {
			globals_emitted = true;
			if (need_float) { out.push_back((3u << 16) | spv_op_type_float);  out.push_back(t_float); out.push_back(32); }
			if (need_v4f)   { out.push_back((4u << 16) | spv_op_type_vector); out.push_back(t_v4f); out.push_back(t_float); out.push_back(4); }
			if (need_v2f)   { out.push_back((4u << 16) | spv_op_type_vector); out.push_back(t_v2f); out.push_back(t_float); out.push_back(2); }
			if (need_int)   { out.push_back((4u << 16) | spv_op_type_int);    out.push_back(t_int); out.push_back(32); out.push_back(1); }
			if (need_v2i)   { out.push_back((4u << 16) | spv_op_type_vector); out.push_back(t_v2i); out.push_back(t_int); out.push_back(2); }
			if (need_int0)  { out.push_back((4u << 16) | spv_op_constant);    out.push_back(t_int); out.push_back(c_int0); out.push_back(0); }
			if (need_fc_ptr){ out.push_back((4u << 16) | spv_op_type_pointer);out.push_back(t_fc_ptr); out.push_back(spv_storage_input); out.push_back(t_v4f); }
			if (need_fc)    { out.push_back((4u << 16) | spv_op_variable);    out.push_back(t_fc_ptr); out.push_back(frag_coord); out.push_back(spv_storage_input); }
		}

		// SubpassData image types become plain sampled 2D (MS bit persists)
		if (opcode == spv_op_type_image && count >= 9 && img_types.count(spirv[i+1])) {
			out.push_back(spirv[i]);
			out.push_back(spirv[i+1]);
			out.push_back(spirv[i+2]);
			out.push_back(spv_dim_2d);   // Dim
			out.push_back(spirv[i+4]);   // Depth
			out.push_back(spirv[i+5]);   // Arrayed
			out.push_back(spirv[i+6]);   // MS
			out.push_back(1);            // Sampled
			for (uint32_t w = 8; w < count; w++) out.push_back(spirv[i+w]);
			i += count;
			continue;
		}

		if (opcode == spv_op_load && count >= 4 && vars.count(spirv[i+3]))
			load_var[spirv[i+2]] = spirv[i+3];

		// Subpass reads become fetches at the fragment's own pixel
		if (opcode == spv_op_image_read && count >= 5 && load_var.count(spirv[i+3])) {
			uint32_t id_fc  = next_id++;
			uint32_t id_xy  = next_id++;
			uint32_t id_ixy = next_id++;
			out.push_back((4u << 16) | spv_op_load);           // %fc = OpLoad v4f frag_coord
			out.push_back(t_v4f); out.push_back(id_fc); out.push_back(frag_coord);
			out.push_back((7u << 16) | spv_op_vector_shuffle); // %xy = fc.xy
			out.push_back(t_v2f); out.push_back(id_xy); out.push_back(id_fc); out.push_back(id_fc); out.push_back(0); out.push_back(1);
			out.push_back((4u << 16) | spv_op_convert_f_to_s); // %ixy = int2(xy)
			out.push_back(t_v2i); out.push_back(id_ixy); out.push_back(id_xy);

			bool has_operands = count > 5; // e.g. Sample <id> on multisampled reads
			if (has_operands) {
				// Keep the original image-operand words (Sample idx)
				out.push_back((count << 16) | spv_op_image_fetch);
				out.push_back(spirv[i+1]); out.push_back(spirv[i+2]); out.push_back(spirv[i+3]);
				out.push_back(id_ixy);
				for (uint32_t w = 5; w < count; w++) out.push_back(spirv[i+w]);
			} else {
				// Add an explicit Lod 0 operand
				out.push_back((7u << 16) | spv_op_image_fetch);
				out.push_back(spirv[i+1]); out.push_back(spirv[i+2]); out.push_back(spirv[i+3]);
				out.push_back(id_ixy);
				out.push_back(0x2); // ImageOperands Lod
				out.push_back(c_int0);
			}
			i += count;
			continue;
		}

		out.insert(out.end(), spirv.begin() + i, spirv.begin() + i + count);
		i += count;
	}

	out[3] = next_id;
	spirv.swap(out);
	return true;
}

///////////////////////////////////////////
// Storage image access analysis        //
///////////////////////////////////////////

// glslang never decorates RWTexture bindings NonReadable/NonWritable, so
// Tint conservatively emits `read_write` access — which WebGPU only allows
// on r32 formats. Analyze how each storage image is really used, decorate
// the SPIR-V so Tint emits the narrowest access, and record the usage in
// meta (shape bit 6 = written, bit 7 = read; only set on storage images,
// where they can't collide with the QCOM sampler bit sampled textures use)
// so the runtime builds a matching bind group layout. Returns false when a
// storage image genuinely needs read_write on a format WebGPU can't — the
// stage should skip WGSL output.
static bool _wgsl_storage_image_access(std::vector<uint32_t> &spirv, sksc_shader_meta_t *ref_meta) {
	std::unordered_map<uint32_t, uint32_t> img_types;   // OpTypeImage with Sampled=2 -> SpvImageFormat
	std::unordered_map<uint32_t, uint32_t> ptr_types;   // pointer type -> image format
	std::unordered_map<uint32_t, uint32_t> var_format;  // variable id -> image format
	std::unordered_map<uint32_t, uint32_t> var_binding; // var id -> binding
	std::unordered_map<uint32_t, uint32_t> load_var;    // OpLoad result -> var id

	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;
		if      (opcode == spv_op_type_image   && count >= 9 && spirv[i+7] == 2) img_types[spirv[i+1]] = spirv[i+8];
		else if (opcode == spv_op_type_pointer && count == 4 && img_types.count(spirv[i+3])) ptr_types[spirv[i+1]] = img_types[spirv[i+3]];
		else if (opcode == spv_op_variable     && count >= 4 && ptr_types.count(spirv[i+1])) var_format[spirv[i+2]] = ptr_types[spirv[i+1]];
		else if (opcode == spv_op_decorate && count >= 4 && spirv[i+2] == spv_decoration_binding) var_binding[spirv[i+1]] = spirv[i+3];
		i += count;
	}
	if (var_format.empty()) return true;

	// Track usage through the ops glslang/spirv-opt emit for storage images.
	// Only scan function bodies — operands elsewhere (constants, types) hold
	// literals that could collide with variable ids.
	std::unordered_map<uint32_t, uint32_t> access; // var id -> bit 1 read, bit 2 write
	bool in_function = false;
	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;
		if (opcode == spv_op_function) in_function = true;
		if (!in_function) { i += count; continue; }

		switch (opcode) {
			case spv_op_load:
				if (count >= 4 && var_format.count(spirv[i+3])) load_var[spirv[i+2]] = spirv[i+3];
				break;
			case spv_op_image_read: case spv_op_image_sparse_read:
				if (count >= 5) { auto it = load_var.find(spirv[i+3]); if (it != load_var.end()) access[it->second] |= 1; }
				break;
			case spv_op_image_write:
				if (count >= 4) { auto it = load_var.find(spirv[i+1]); if (it != load_var.end()) access[it->second] |= 2; }
				break;
			case 103: case 104: case 105: case 106: case 107: // OpImageQuery* read no texels
				break;
			case spv_op_image_texel_pointer: // atomics
				if (count >= 5 && var_format.count(spirv[i+3])) access[spirv[i+3]] |= 3;
				break;
			default:
				// Uses this pass can't narrow (access chains into image
				// arrays, image passed to a function, aliases) stay read_write
				for (uint32_t w = 1; w < count; w++) {
					if (var_format.count(spirv[i+w])) { access[spirv[i+w]] |= 3; continue; }
					auto it = load_var.find(spirv[i+w]);
					if (it != load_var.end()) access[it->second] |= 3;
				}
				break;
		}
		i += count;
	}

	// WebGPU restricts read_write storage texture access to the r32 formats
	for (auto &vf : var_format) {
		uint32_t use = access.count(vf.first) ? access[vf.first] : 0;
		uint32_t fmt = vf.second; // SpvImageFormat: R32f = 3, R32i = 24, R32ui = 33
		if (use == 3 && fmt != 3 && fmt != 24 && fmt != 33) {
			uint32_t slot = var_binding.count(vf.first) ? var_binding[vf.first] : 0;
			const char *name = "";
			for (uint32_t r = 0; r < ref_meta->resource_count; r++)
				if (ref_meta->resources[r].bind.register_type == skr_register_readwrite_tex && ref_meta->resources[r].bind.slot == slot)
					name = ref_meta->resources[r].name;
			sksc_log(sksc_log_level_warn, "Skipping WGSL output for a stage: storage texture '%s' is both read and written, which WebGPU only allows on r32 formats (r32f/r32i/r32ui)", name);
			return false;
		}
	}

	// Decorate write-only images NonReadable (25), read-only ones NonWritable
	// (24), at the head of the annotations section
	size_t insert_at = spirv.size();
	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;
		if (opcode == spv_op_decorate || opcode == spv_op_member_decorate) { insert_at = i; break; }
		i += count;
	}
	std::vector<uint32_t> extra;
	for (auto &vf : var_format) {
		uint32_t var = vf.first;
		uint32_t use = access.count(var) ? access[var] : 0;
		if (use == 2) { extra.push_back((3u << 16) | spv_op_decorate); extra.push_back(var); extra.push_back(spv_decoration_non_readable); }
		if (use == 1) { extra.push_back((3u << 16) | spv_op_decorate); extra.push_back(var); extra.push_back(spv_decoration_non_writable); }

		auto bind = var_binding.find(var);
		if (bind == var_binding.end()) continue;
		for (uint32_t r = 0; r < ref_meta->resource_count; r++) {
			sksc_shader_resource_t *res = &ref_meta->resources[r];
			if (res->bind.register_type != skr_register_readwrite_tex || res->bind.slot != bind->second) continue;
			if (use & 1) res->shape |= 0x80;
			if (use & 2) res->shape |= 0x40;
		}
	}
	spirv.insert(spirv.begin() + insert_at, extra.begin(), extra.end());
	return true;
}

///////////////////////////////////////////
// Non-finite constant scan              //
///////////////////////////////////////////

// WGSL has no way to express NaN/Inf constants, and Tint asserts (aborting
// the process) when it meets one — so refuse them gracefully up front.
static bool _wgsl_has_nonfinite_constant(const std::vector<uint32_t> &spirv) {
	std::vector<uint32_t> f32_types;
	for (size_t i = spv_header_words; i < spirv.size(); ) {
		uint32_t count  = spirv[i] >> 16;
		uint32_t opcode = spirv[i] & 0xFFFF;
		if (count == 0 || i + count > spirv.size()) break;

		if (opcode == spv_op_type_float && count >= 3 && spirv[i+2] == 32)
			f32_types.push_back(spirv[i+1]);
		if (opcode == spv_op_constant && count >= 4) {
			for (uint32_t t : f32_types)
				if (t == spirv[i+1] && (spirv[i+3] & 0x7F800000u) == 0x7F800000u)
					return true;
		}
		i += count;
	}
	return false;
}

///////////////////////////////////////////
// Stage compilation                     //
///////////////////////////////////////////

compile_result_ sksc_wgsl_from_spirv(const uint32_t *words, uint32_t word_count, skr_stage_ type, sksc_shader_meta_t *ref_meta, sksc_shader_file_stage_t *out_stage) {
#ifndef SKSC_HAS_TINT
	(void)words; (void)word_count; (void)type; (void)ref_meta; (void)out_stage;
	sksc_log(sksc_log_level_err, "The WGSL target ('-t w') needs a skshaderc built with SKSHADERC_ENABLE_WGSL (Tint)");
	return compile_result_fail;
#else
	std::vector<uint32_t> spirv(words, words + word_count);

	if (_wgsl_has_nonfinite_constant(spirv)) {
		sksc_log(sksc_log_level_warn, "Skipping WGSL output for a stage: it contains a NaN/Inf float constant, which WGSL cannot represent. Consider computing it at runtime instead.");
		return compile_result_skip;
	}

	std::vector<uint32_t> combined_samplers;
	if (!_wgsl_view_index_to_spec(spirv, type)) return compile_result_fail;
	if (!_wgsl_subpass_to_texture(spirv, type)) return compile_result_fail;
	if (!_wgsl_reflect_samplers  (spirv, type, ref_meta, &combined_samplers)) return compile_result_fail;
	if (!_wgsl_storage_image_access(spirv, ref_meta)) return compile_result_skip;

	static bool tint_initialized = false;
	if (!tint_initialized) { tint::Initialize(); tint_initialized = true; }

	// Tint: SPIR-V -> IR -> WGSL. The reader validates the module, acting as
	// the build-time gate for the transforms above.
	tint::spirv::reader::Options reader_options;
	// Combined image samplers split on read; route the synthesized samplers
	// to the slots the meta reflection recorded for them
	for (uint32_t b : combined_samplers)
		reader_options.sampler_mappings[tint::BindingPoint{0, b}] = tint::BindingPoint{0, _wgsl_combined_sampler_slot(b)};
	auto ir = tint::spirv::reader::ReadIR(spirv, reader_options);
	if (ir != tint::Success) {
		sksc_log(sksc_log_level_warn, "Skipping WGSL output for a stage, Tint couldn't read its SPIR-V:\n%s", ir.Failure().reason.c_str());
		return compile_result_skip;
	}

	tint::wgsl::writer::Options writer_options;
	auto result = tint::wgsl::writer::WgslFromIR(ir.Get(), writer_options);
	if (result != tint::Success) {
		// Usually a construct WebGPU genuinely can't express (e.g. read_write
		// storage buffers in a vertex stage) — the .sks just carries no WGSL
		// blob, and the WebGPU runtime reports it clearly at shader load.
		sksc_log(sksc_log_level_warn, "Skipping WGSL output for a stage, it isn't expressible in WGSL:\n%s", result.Failure().reason.c_str());
		return compile_result_skip;
	}

	const std::string &wgsl = result.Get().wgsl;
	out_stage->language  = skr_shader_lang_wgsl;
	out_stage->stage     = type;
	out_stage->code_size = (uint32_t)wgsl.size() + 1;
	out_stage->code      = malloc(out_stage->code_size);
	memcpy(out_stage->code, wgsl.c_str(), out_stage->code_size);
	return compile_result_success;
#endif
}

compile_result_ sksc_wgsl_compile_stage(const char *filename, const char *hlsl, const sksc_settings_t *settings, skr_stage_ type, sksc_shader_meta_t *ref_meta, sksc_shader_file_stage_t *out_stage) {
#ifndef SKSC_HAS_TINT
	(void)filename; (void)hlsl; (void)settings; (void)type; (void)ref_meta; (void)out_stage;
	sksc_log(sksc_log_level_err, "The WGSL target ('-t w') needs a skshaderc built with SKSHADERC_ENABLE_WGSL (Tint)");
	return compile_result_fail;
#else
	// Second compile of the same source with samplers kept separate
	sksc_shader_file_stage_t spirv_stage  = {};
	compile_result_          spirv_result = sksc_hlsl_to_spirv(filename, hlsl, settings, type, NULL, 0, true, &spirv_stage);
	if (spirv_result != compile_result_success) return compile_result_fail;

	compile_result_ result = sksc_wgsl_from_spirv((const uint32_t *)spirv_stage.code, spirv_stage.code_size / 4, type, ref_meta, out_stage);
	free(spirv_stage.code);
	return result;
#endif
}
