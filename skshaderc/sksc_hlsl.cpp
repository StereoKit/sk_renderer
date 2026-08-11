#include "_sksc.h"

#define ENABLE_HLSL

#include <glslang/Public/ShaderLang.h>
#include "StandAlone/DirStackFileIncluder.h"
#include "SPIRV/GlslangToSpv.h"

#include <spirv-tools/optimizer.hpp>
#include <spirv_reflect.h>

///////////////////////////////////////////

void sksc_glslang_init() {
	glslang::InitializeProcess();
}

///////////////////////////////////////////

void sksc_glslang_shutdown() {
	glslang::FinalizeProcess();
}

///////////////////////////////////////////
// HLSL to SPIR-V                        //
///////////////////////////////////////////

class SkscIncluder : public DirStackFileIncluder {
public:
	virtual IncludeResult* includeSystem(const char* header_name, const char* includer_name, size_t inclusion_depth) override {
		recordLocalPath(includer_name, inclusion_depth);
		IncludeResult* result = readLocalPath(header_name);
		if (result) return result;
		return readSystemPath(header_name);
	}
};

///////////////////////////////////////////

bool parse_startswith(const char* a, const char* is) {
	while (*is != '\0') {
		if (*a == '\0' || *is != *a)
			return false;
		a++;
		is++;
	}
	return true;
}

///////////////////////////////////////////

bool parse_readint(const char* a, char separator, int32_t *out_int, const char** out_at) {
	const char *end = a;
	while (*end != '\0' && *end != '\n' && *end != separator) {
		end++;
	}
	if (*end != separator) return false;

	char* success = nullptr;
	*out_int = (int32_t)strtol(a, &success, 10);
	*out_at  = end+1;

	return success <= end;
}

///////////////////////////////////////////

const char* parse_glslang_error(const char* at) {
	const char* curr  = at;
	sksc_log_level_ level = sksc_log_level_err;
	if      (parse_startswith(at, "ERROR: "  )) { level = sksc_log_level_err;  curr += 7;}
	else if (parse_startswith(at, "WARNING: ")) { level = sksc_log_level_warn; curr += 9;}

	bool has_line = false;
	int32_t line = 0;
	int32_t col  = 0; // '(line)' format errors carry no column

	const char* numbers = curr;
	// Check for 'col:line:' format line numbers
	if (parse_readint(numbers, ':', &col,  &numbers) &&
		parse_readint(numbers, ':', &line, &numbers)) {
		has_line = true;
		curr = numbers + 1;
	}
	numbers = curr;
	// check for '(line)' format line numbers
	if (!has_line && *numbers == '(' && parse_readint(numbers+1, ')', &line, &numbers)) {
		has_line = true;
		curr = numbers + 1;
		if (*curr != '\0') curr++;
	}

	const char* start = curr;
	while (*curr != '\0' && *curr != '\n') {
		curr++;
	}
	if (curr - at > 1) 
		has_line 
			? sksc_log_at(level, line, col, "%.*s", curr - start, start)
			: sksc_log   (level,            "%.*s", curr - start, start);
	if (*curr == '\n') curr++;
	return *curr == '\0' ? nullptr : curr;
}

///////////////////////////////////////////

void log_shader_msgs(glslang::TShader *shader) {
	const char* info_log  = shader->getInfoLog();
	const char* debug_log = shader->getInfoDebugLog();
	while (info_log  != nullptr && *info_log  != '\0') { info_log  = parse_glslang_error(info_log ); }
	while (debug_log != nullptr && *debug_log != '\0') { debug_log = parse_glslang_error(debug_log); }
}

///////////////////////////////////////////

compile_result_ sksc_hlsl_to_spirv(const char *filename, const char *hlsl, const sksc_settings_t *settings, skr_stage_ type, sksc_shader_file_stage_t *out_stage) {
	TBuiltInResource default_resource = {};
	EShMessages      messages         = EShMsgDefault;
	EShMessages      messages_link    = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules | EShMsgDebugInfo);
	EShLanguage      stage = EShLangVertex;
	const char*      entry = "na";
	switch(type) {
		case skr_stage_vertex:  stage = EShLangVertex;   entry = settings->vs_entrypoint; break;
		case skr_stage_pixel:   stage = EShLangFragment; entry = settings->ps_entrypoint; break;
		case skr_stage_compute: stage = EShLangCompute;  entry = settings->cs_entrypoint; break;
		default: sksc_log(sksc_log_level_err, "Unknown shader stage %d", (int)type); return compile_result_fail;
	}

	// Create the shader and set options
	glslang::TShader shader(stage);
	const char* shader_strings[1] = { hlsl };
	shader.setEntryPoint      (entry);
	shader.setSourceEntryPoint(entry);
	shader.setEnvInput        (glslang::EShSourceHlsl, stage, glslang::EShClientVulkan, 100);
	shader.setEnvClient       (glslang::EShClientVulkan,      glslang::EShTargetVulkan_1_1);
	shader.setEnvTarget       (glslang::EShTargetSpv,         glslang::EShTargetSpv_1_3);
	shader.setEnvTargetHlslFunctionality1();
	shader.setTextureSamplerTransformMode(EShTexSampTransUpgradeTextureRemoveSampler);
	// RWTextures get SPIR-V format Unknown instead of one inferred from the
	// template (float4 to Rgba32f), like DXC; needs the WithoutFormat features.
	shader.setNoStorageFormat(true);
	bool debug = settings->debug;
	if (debug) {
		shader.setDebugInfo (true);
		shader.setSourceFile(filename);
		shader.addSourceText(hlsl, strlen(hlsl));
	}

	shader.setAutoMapBindings(true); // Necessary for shifts?
	shader.setShiftBinding    (glslang::EResUbo,     0);   // b registers (CBVs)
	shader.setShiftBinding    (glslang::EResTexture, 100); // t registers (textures)
	shader.setShiftBinding    (glslang::EResSampler, 100); // s registers (samplers)
	shader.setShiftBinding    (glslang::EResUav,     200); // u registers (UAVs)

	shader.setStrings         (shader_strings, 1);

	// Setup includer
	SkscIncluder includer;
	includer.pushExternalDirectory(settings->folder);
	for (int32_t i = 0; i < settings->include_folder_ct; i++) {
		includer.pushExternalDirectory(settings->include_folders[i]);
	}

	std::string preprocessed_glsl;
	if (!shader.preprocess(
		&default_resource,
		100,                // default version
		ECoreProfile,       // default profile
		false,              // don't force default version and profile
		false,              // not forward compatible
		messages,
		&preprocessed_glsl,
		includer)) {

		log_shader_msgs(&shader);
		return compile_result_fail;
	}

	// Set the preprocessed shader
	const char* preprocessed_strings[1] = { preprocessed_glsl.c_str() };
	shader.setStrings(preprocessed_strings, 1);

	// Parse the shader
	if (!shader.parse(&default_resource, 100, false, messages)) {
		log_shader_msgs(&shader);
		return compile_result_fail;
	}

	// Create and link program
	glslang::TProgram program;
	program.addShader(&shader);
	if (!program.link(messages_link)) {
		log_shader_msgs(&shader);
		return compile_result_fail;
	}

	// Check if we found an entry point
	const char *link_info = program.getInfoLog();
	if (link_info != nullptr) {
		if (strstr(link_info, "Entry point not found") != nullptr) {
			return compile_result_skip;
		}
	}

	// Generate SPIR-V
	glslang::TIntermediate* intermediate = program.getIntermediate(stage);
	if (!intermediate) {
		return compile_result_fail;
	}

	std::vector<unsigned int> spirv;
	spv::SpvBuildLogger logger;
	glslang::SpvOptions spvOptions;
	spvOptions.generateDebugInfo                = debug;
	spvOptions.emitNonSemanticShaderDebugInfo   = debug;
	spvOptions.emitNonSemanticShaderDebugSource = debug;
	// Enable glslang's built-in SPIRV optimizer which includes HLSL-specific
	// legalization passes (FixStorageClass, InterpolateFixup, CFGCleanup, etc.)
	spvOptions.disableOptimizer                 = debug || settings->optimize == 0;
	spvOptions.optimizeSize                     = settings->optimize == 1;
	glslang::GlslangToSpv(*intermediate, spirv, &logger, &spvOptions);

	// Log any SPIR-V generation messages
	std::string gen_messages = logger.getAllMessages();
	if (gen_messages.length() > 0) {
		sksc_log(sksc_log_level_info, gen_messages.c_str());
	}

	////////////////////////////////////////////////////////////////
	// In this section, we are shifting the bind registers by manually
	// inspecting and editing the SPIRV.
	//
	// Ideally we would do this with:
	// shader.setShiftBinding    (glslang::EResUbo,     0);
	// shader.setShiftBinding    (glslang::EResTexture, 100);
	// shader.setShiftBinding    (glslang::EResSampler, 100);
	// shader.setShiftBinding    (glslang::EResUav,     200);
	//
	// However, setShiftBinding wasn't working.

	spv_reflect::ShaderModule reflection(spirv.size() * sizeof(uint32_t), spirv.data());
	uint32_t binding_count = 0;
	reflection.EnumerateDescriptorBindings(&binding_count, nullptr);
	std::vector<SpvReflectDescriptorBinding*> bindings(binding_count);
	reflection.EnumerateDescriptorBindings(&binding_count, bindings.data());
	std::unordered_map<uint32_t, uint32_t> binding_remaps;

	// Find the binds
	for (SpvReflectDescriptorBinding* binding : bindings) {
		uint32_t old_binding = binding->binding;
		uint32_t new_binding = old_binding;
		
		switch (binding->descriptor_type) {
			case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
			case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
				// b registers - no shift
				new_binding = old_binding + 0;
				break;
				
			case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
				// s registers - merged into textures (+100)
				new_binding = old_binding + SKSC_SLOT_TEXTURE;
				break;

			case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
			case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
				// t registers
				new_binding = old_binding + SKSC_SLOT_TEXTURE;
				break;

			case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
				// Check if it's read-only (StructuredBuffer) or read-write (RWStructuredBuffer)
				if (binding->resource_type == SPV_REFLECT_RESOURCE_FLAG_SRV) {
					// StructuredBuffer - t register
					new_binding = old_binding + SKSC_SLOT_TEXTURE;
				} else {
					// RWStructuredBuffer - u register
					new_binding = old_binding + SKSC_SLOT_READWRITE;
				}
				break;
				
			case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
			case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
				// u registers
				new_binding = old_binding + SKSC_SLOT_READWRITE;
				break;

			case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
				// input attachments - slot 300 + the declared
				// [[vk::input_attachment_index(N)]]. glslang's auto-mapper can
				// give several unregistered SubpassInputs the same binding, so
				// the attachment index is the only collision-free identity.
				new_binding = SKSC_SLOT_INPUT_ATT + binding->input_attachment_index;
				break;
			default:break;
		}
		
		if (old_binding != new_binding) {
			binding_remaps[binding->spirv_id] = new_binding;
		}
	}

	// Now manually patch the SPIR-V to update bindings
	const size_t SPIRV_HEADER_SIZE = 5;
	for (size_t i = SPIRV_HEADER_SIZE; i < spirv.size(); ) {
		uint32_t word_count = spirv[i] >> 16;
		uint32_t opcode     = spirv[i] & 0xFFFF;

		// OpDecorate instruction (opcode 71)
		if (opcode == 71 && word_count >= 4) {
			uint32_t target_id = spirv[i + 1]; // The ID being decorated
			uint32_t decoration_type = spirv[i + 2];
			
			// Binding decoration (33)
			if (decoration_type == 33) {
				auto it = binding_remaps.find(target_id);
				if (it != binding_remaps.end()) {
					spirv[i + 3] = it->second; // Patch the binding!
				}
			}
		}
		
		i += word_count;
	}

	// Run additional SPIRV optimization passes after binding remaps.
	// glslang's optimizer handles HLSL-specific legalization, but we can
	// squeeze out a bit more with the full performance/size passes.
	if (!debug && settings->optimize > 0) {
		spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
		optimizer.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
			printf("SPIRV optimization error: %s\n", m);
		});

		if (settings->optimize == 1) {
			optimizer.RegisterSizePasses();
		} else {
			optimizer.RegisterPerformancePasses();
		}

		// Additional passes not included in the standard bundles
		optimizer.RegisterPass(spvtools::CreateStrengthReductionPass());
		optimizer.RegisterPass(spvtools::CreateCodeSinkingPass());
		optimizer.RegisterPass(spvtools::CreateLoopInvariantCodeMotionPass());
		optimizer.RegisterPass(spvtools::CreateLoopPeelingPass());
		optimizer.RegisterPass(spvtools::CreateLoopUnswitchPass());
		optimizer.RegisterPass(spvtools::CreateLocalRedundancyEliminationPass());
		optimizer.RegisterPass(spvtools::CreateReduceLoadSizePass());
		// Cleanup unused/duplicate data
		optimizer.RegisterPass(spvtools::CreateUnifyConstantPass());
		optimizer.RegisterPass(spvtools::CreateEliminateDeadConstantPass());
		optimizer.RegisterPass(spvtools::CreateDeadVariableEliminationPass());
		optimizer.RegisterPass(spvtools::CreateRemoveDuplicatesPass());
		optimizer.RegisterPass(spvtools::CreateCFGCleanupPass());
		// Final cleanup
		optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
		optimizer.RegisterPass(spvtools::CreateTrimCapabilitiesPass());
		optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

		std::vector<uint32_t> spirv_optimized;
		if (!optimizer.Run(spirv.data(), spirv.size(), &spirv_optimized)) {
			return compile_result_fail;
		}

		out_stage->code_size = (uint32_t)(spirv_optimized.size() * sizeof(unsigned int));
		out_stage->code      = malloc(out_stage->code_size);
		memcpy(out_stage->code, spirv_optimized.data(), out_stage->code_size);
	} else {
		out_stage->code_size = (uint32_t)(spirv.size() * sizeof(unsigned int));
		out_stage->code      = malloc(out_stage->code_size);
		memcpy(out_stage->code, spirv.data(), out_stage->code_size);
	}
	out_stage->language = skr_shader_lang_spirv;
	out_stage->stage    = type;

	return compile_result_success;
}