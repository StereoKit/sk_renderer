# sk_texenc: GPU block compression for sk_renderer textures. Added from
# sk_renderer's root after the sk_renderer target, and needs skshaderc for its
# embedded shaders.

# One encoder family per platform: x64 GPUs sample BC, ARM GPUs sample ASTC
# (Apple Silicon does both, and ASTC wins there). WebGPU can land on either,
# so it carries both.
set(_skr_texenc_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
	set(_skr_texenc_arch "${CMAKE_OSX_ARCHITECTURES}")
elseif(MSVC AND CMAKE_GENERATOR_PLATFORM)
	set(_skr_texenc_arch "${CMAKE_GENERATOR_PLATFORM}")
endif()
string(TOLOWER "${_skr_texenc_arch}" _skr_texenc_arch)

if(SKR_USE_WEBGPU)
	set(_skr_texenc_bc   ON)
	set(_skr_texenc_astc ON)
elseif(_skr_texenc_arch MATCHES "(^|;)(arm|aarch64)")
	set(_skr_texenc_bc   OFF)
	set(_skr_texenc_astc ON)
else()
	set(_skr_texenc_bc   ON)
	set(_skr_texenc_astc OFF)
endif()
option(SK_TEXENC_BC   "Embed the BC encoders (BC1, BC6H, BC7)"                 ${_skr_texenc_bc})
option(SK_TEXENC_ASTC "Embed the ASTC encoders (4x4, 6x6, 8x8 HDR)"            ${_skr_texenc_astc})
# For test applications: raw block encodes for readback and profiling, and
# every encoder embedded so formats the GPU can't sample still validate.
option(SK_TEXENC_DEBUG "Build sk_texenc's debug API and embed every encoder"   OFF)
message(STATUS "sk_texenc: BC ${SK_TEXENC_BC}, ASTC ${SK_TEXENC_ASTC}, debug ${SK_TEXENC_DEBUG}")

set(_skr_texenc_bc_on   ${SK_TEXENC_BC})
set(_skr_texenc_astc_on ${SK_TEXENC_ASTC})
if(SK_TEXENC_DEBUG)
	set(_skr_texenc_bc_on   ON)
	set(_skr_texenc_astc_on ON)
endif()
# WebGPU has no ASTC HDR format to sample
set(_skr_texenc_astc_hdr OFF)
if(SK_TEXENC_DEBUG OR (SK_TEXENC_ASTC AND NOT SKR_USE_WEBGPU))
	set(_skr_texenc_astc_hdr ON)
endif()

add_library(sk_texenc STATIC
	include/sk_texenc.h
	sk_texenc.c
)
set_target_properties(sk_texenc PROPERTIES
	C_STANDARD          11
	C_STANDARD_REQUIRED ON
	POSITION_INDEPENDENT_CODE ON)
target_include_directories(sk_texenc PUBLIC include)
target_link_libraries     (sk_texenc PUBLIC sk_renderer)
target_compile_definitions(sk_texenc PRIVATE
	SK_TEXENC_BC=$<BOOL:${_skr_texenc_bc_on}>
	SK_TEXENC_ASTC=$<BOOL:${_skr_texenc_astc_on}>
	SK_TEXENC_ASTC_HDR=$<BOOL:${_skr_texenc_astc_hdr}>)
if(SK_TEXENC_DEBUG)
	target_compile_definitions(sk_texenc PUBLIC SK_TEXENC_DEBUG)
endif()
if(COMMAND skr_target_warnings)
	skr_target_warnings(sk_texenc)
endif()
if(NOT MSVC)
	target_compile_options(sk_texenc PRIVATE -fvisibility=hidden)
endif()

set(_skr_texenc_shaders)
if(_skr_texenc_bc_on)
	list(APPEND _skr_texenc_shaders
		shaders/bc1_compress.hlsl
		shaders/bc6h_compress.hlsl
		shaders/bc7_compress.hlsl)
endif()
if(_skr_texenc_astc_on)
	list(APPEND _skr_texenc_shaders
		shaders/astc4x4_compress.hlsl
		shaders/astc6x6_compress.hlsl)
endif()
if(_skr_texenc_astc_hdr)
	list(APPEND _skr_texenc_shaders shaders/astc8x8hdr_compress.hlsl)
endif()
if(_skr_texenc_shaders)
	# Unoptimized encoders take Adreno's compiler minutes per pipeline, so even
	# Debug builds optimize them. -z zips them; the host supplies the inflate.
	skshaderc_compile_headers(sk_texenc ${CMAKE_BINARY_DIR}/sk_renderer_generated/texenc_shaders "${_skr_builtin_shader_args} -z"
		${_skr_texenc_shaders}
		OPTIMIZED
		INCLUDES
			shaders/texenc_common.hlsli
			shaders/bc_common.hlsli
			shaders/astc_common.hlsli)
endif()
