# The skshaderc cmake file exports a few useful items for
# building shaders in your own project! The first is the
# skshaderc_compile_headers macro, which can compile a list
# of shader files, and return a list with the same files
# and .h appended to them.
#
# Example:
# set(PROJECT_SHADERS_HLSL
#     compute_test.hlsl
#     cubemap.hlsl
#     test.hlsl )
# skshaderc_compile_headers(PROJECT_SHADERS_HLSL "-t xge" PROJECT_SHADERS_H)
#
# PROJECT_SHADERS_H now contains a list of .h files that
# contain shader binaries in arrays.
#
# skshaderc_compile_assets is similar, but is not specifically
# for header files. It's best if you're looking for side-by-side
# .sks files. It has no output list.
#
# skshaderc_compile_assets(PROJECT_SHADERS_HLSL "-t xge")
#
# If you wish to design your own shader compiling function,
# this file also exports SKSHADERC_EXE_PATH, which is a
# generator for the path to the skshaderc compiler executable
# file.

cmake_minimum_required(VERSION 3.10)
project(skshaderc VERSION 1.0 DESCRIPTION "Shader compiler for sk_renderer")

###############################################################################
# Cross-compilation detection
# When cross-compiling, we use a pre-built host binary to compile shaders,
# while still building skshaderc for the target platform.
###############################################################################

# Get host architecture
string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" _HOST_PROC)
# Get target architecture (MSVC uses CMAKE_GENERATOR_PLATFORM or architecture ID)
if(MSVC AND CMAKE_GENERATOR_PLATFORM)
	string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" _TARGET_PROC)
elseif(MSVC AND MSVC_CXX_ARCHITECTURE_ID)
	string(TOLOWER "${MSVC_CXX_ARCHITECTURE_ID}" _TARGET_PROC)
elseif(CMAKE_SYSTEM_PROCESSOR)
	string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _TARGET_PROC)
else()
	set(_TARGET_PROC "${_HOST_PROC}")
endif()

# Normalize arch names to x64|arm64
if(_HOST_PROC MATCHES "amd64|x64|x86_64")
	set(_HOST_ARCH "x64")
elseif(_HOST_PROC MATCHES "arm64|aarch64")
	set(_HOST_ARCH "arm64")
endif()
if(_TARGET_PROC MATCHES "amd64|x64|x86_64")
	set(_TARGET_ARCH "x64")
elseif(_TARGET_PROC MATCHES "arm64|aarch64")
	set(_TARGET_ARCH "arm64")
endif()

# Normalize system name to win32|macos|linux|android
string(TOLOWER "${CMAKE_HOST_SYSTEM_NAME}" _HOST_SYS)
string(TOLOWER "${CMAKE_SYSTEM_NAME}" _TARGET_SYS)
if(_TARGET_SYS STREQUAL "darwin")
	set(_TARGET_SYS "macos")
elseif(_TARGET_SYS STREQUAL "windows")
	set(_TARGET_SYS       "win32")
	set(_TARGET_EXTENSION ".exe" )
endif()
if(_HOST_SYS STREQUAL "darwin")
	set(_HOST_SYS "macos")
elseif(_HOST_SYS STREQUAL "windows")
	set(_HOST_SYS       "win32")
	set(_HOST_EXTENSION ".exe" )
endif()

if((NOT _HOST_SYS STREQUAL _TARGET_SYS) OR (NOT _HOST_ARCH STREQUAL _TARGET_ARCH))
	set(SKSHADERC_CROSS_COMPILING TRUE)
	# Special case: Unix->Windows can use Wine instead of pre-built binary
	if(CMAKE_HOST_UNIX AND _TARGET_SYS STREQUAL "win32")
		set(SKSHADERC_CROSS_COMPILING FALSE)
		set(WINE "wine" CACHE INTERNAL "")
		message(STATUS "Cross-compiling to Windows: will use Wine to run skshaderc.exe")
	endif()
else()
	set(SKSHADERC_CROSS_COMPILING FALSE)
endif()

# Create final shader paths
# Use CMAKE_SOURCE_DIR so binaries go to the top-level project's bin/,
# allowing consuming projects to accumulate skshaderc builds.
set(SKSHADERC_HOST_BIN      "${CMAKE_SOURCE_DIR}/bin/tools/${_HOST_SYS}_${_HOST_ARCH}/skshaderc${_HOST_EXTENSION}")
set(SKSHADERC_TARGET_FOLDER "${CMAKE_SOURCE_DIR}/bin/tools/${_TARGET_SYS}_${_TARGET_ARCH}")

# Check if host binary exists when cross-compiling (not needed for Wine case)
if(SKSHADERC_CROSS_COMPILING)
	if(NOT EXISTS "${SKSHADERC_HOST_BIN}")
		message(FATAL_ERROR
			"Cross-compiling (${_HOST_SYS}_${_HOST_ARCH} -> ${_TARGET_SYS}_${_TARGET_ARCH}) requires "
			"a host compatible skshaderc binary!\n"
			""
			"To fix, build first for the host platform, ${_HOST_SYS}_${_HOST_ARCH}:\n"
			""
			"    cmake -B build-${_HOST_SYS} . && cmake --build build-${_HOST_SYS}.\n"
			""
			"This will install the binary to ${SKSHADERC_HOST_BIN}.")
	else()
		message(STATUS "Cross-compiling (${_HOST_SYS}_${_HOST_ARCH} -> ${_TARGET_SYS}_${_TARGET_ARCH}) using host skshaderc from ${SKSHADERC_HOST_BIN}")
		# Ensure executable permission on Unix hosts
		if(CMAKE_HOST_UNIX)
			file(CHMOD "${SKSHADERC_HOST_BIN}" PERMISSIONS
				OWNER_EXECUTE OWNER_WRITE OWNER_READ
				GROUP_EXECUTE GROUP_READ
				WORLD_EXECUTE WORLD_READ)
		endif()
		# Web builds compile shaders to WGSL; a host tool from an SVSL-only
		# build config has no WGSL target and would emit unloadable .sks files
		if(EMSCRIPTEN)
			file(STRINGS "${SKSHADERC_HOST_BIN}" _skshaderc_tint_probe REGEX "Tint couldn't read" LIMIT_COUNT 1)
			if(NOT _skshaderc_tint_probe)
				message(FATAL_ERROR
					"The host skshaderc at ${SKSHADERC_HOST_BIN} was built without the WGSL "
					"target (SKSHADERC_ENABLE_WGSL), which web builds require.\n"
					"Build the native WebGPU config first — it installs a WGSL-capable tool:\n"
					"    cmake -B build-webgpu -DSKR_USE_WEBGPU=ON && cmake --build build-webgpu")
			endif()
		endif()
	endif()
elseif(NOT SKSHADERC_USE_WINE)
	message(STATUS "Native build: building skshaderc from source")
endif()

###############################################################################
# Skip building skshaderc binaries for certain platforms
# Android doesn't need the skshaderc binary - shaders are compiled using the
# host binary during the build, and the resulting .sks/.h files are used at runtime.
###############################################################################

if(ANDROID OR EMSCRIPTEN)
	set(SKSHADERC_SKIP_BUILD TRUE)
	message(STATUS "Cross target: skipping skshaderc build (using host binary for shader compilation)")
else()
	set(SKSHADERC_SKIP_BUILD FALSE)
endif()

###############################################################################
# Build skshaderc from source
# Skipped when SKSHADERC_SKIP_BUILD is set (e.g., Android cross-compilation)
###############################################################################

if(NOT SKSHADERC_SKIP_BUILD)

include(FetchContent)
set(VULKAN_VERSION vulkan-sdk-1.4.328.1) # Most of our dependencies here use the same version tag!

###############################################################################
# Static library: skshaderc_lib
###############################################################################

add_library(skshaderc_lib STATIC)

set_target_properties(skshaderc_lib PROPERTIES
	CXX_STANDARD 11
	CXX_STANDARD_REQUIRED ON)

# Optional glslang backend: HLSL -> SPIR-V via glslang, optimized with the
# SPIRV-Tools optimizer and reflected with SPIRV-Reflect. glslang also pulls in
# the SPIRV-Tools disassembler, which powers the -raw/-h SPIR-V text output; a
# build without glslang skips all of SPIRV-Headers/Tools/Reflect (well over 3MB
# of the binary) and instead writes raw .spv for -raw and omits SPIR-V comments
# in -h. SKSC_HAS_GLSLANG and SKSC_HAS_SPIRV_TOOLS gate this in code.
option(SKSHADERC_ENABLE_GLSLANG "Build skshaderc with the glslang (HLSL) shader compiler backend" ON)

# Optional WGSL (WebGPU) target: '-t w' output via Tint's SPIR-V reader and
# WGSL writer, built from the same pinned Dawn release the sk_renderer WebGPU
# backend uses (see docs/webgpu-backend-plan.md). Needs the glslang backend
# for the split-sampler compile it works from.
option(SKSHADERC_ENABLE_WGSL "Build skshaderc with the Tint-based WGSL (WebGPU) shader target" ON)
if(SKSHADERC_ENABLE_WGSL AND NOT SKSHADERC_ENABLE_GLSLANG)
	message(WARNING "SKSHADERC_ENABLE_WGSL requires SKSHADERC_ENABLE_GLSLANG; disabling the WGSL target")
	set(SKSHADERC_ENABLE_WGSL OFF)
endif()

if(SKSHADERC_ENABLE_GLSLANG)
	# Tint's SPIR-V reader builds against SPIRV-Tools internals, so when the
	# WGSL target is enabled, SPIRV-Headers/Tools follow the pinned Dawn
	# release's DEPS (v20260713.082812) instead of the Vulkan SDK tag.
	if(SKSHADERC_ENABLE_WGSL)
		set(SKSC_SPIRV_HEADERS_URL https://github.com/KhronosGroup/SPIRV-Headers/archive/29981f65241605e08b0ede4cfeb999fe3b723c6a.zip)
		set(SKSC_SPIRV_TOOLS_URL   https://github.com/KhronosGroup/SPIRV-Tools/archive/85f222ec591863853ea603ce190ea508c7255568.zip)
	else()
		set(SKSC_SPIRV_HEADERS_URL https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/${VULKAN_VERSION}.zip)
		set(SKSC_SPIRV_TOOLS_URL   https://github.com/KhronosGroup/SPIRV-Tools/archive/refs/tags/${VULKAN_VERSION}.zip)
	endif()

	# SPIRV-Headers underpins both SPIRV-Tools and glslang.
	message(STATUS "Fetching SPIRV-Headers...")
	FetchContent_Declare( SPIRV-Headers
		URL ${SKSC_SPIRV_HEADERS_URL}
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	)
	FetchContent_MakeAvailable(SPIRV-Headers)

	# SPIRV-Tools: core provides the disassembler (-raw/-h text output), and
	# SPIRV-Tools-opt the optimizer. A baseline amount of optimization is crucial
	# for meta compatability with HLSL compilers.
	message(STATUS "Fetching SPIRV-Tools...")
	FetchContent_Declare( SPIRV-Tools
		URL ${SKSC_SPIRV_TOOLS_URL}
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	)
	set(SPIRV_SKIP_TESTS       ON  CACHE BOOL "")
	set(SPIRV_SKIP_EXECUTABLES ON  CACHE BOOL "")
	set(SPIRV_WERROR           OFF CACHE BOOL "")
	FetchContent_MakeAvailable(SPIRV-Tools)
	target_link_libraries     (skshaderc_lib PUBLIC SPIRV-Tools SPIRV-Tools-opt)
	target_compile_definitions(skshaderc_lib PUBLIC SKSC_HAS_SPIRV_TOOLS)

	# For grabbing shader metadata, SPIRV-Cross can also do this, but Reflect is
	# lighter weight.
	message(STATUS "Fetching SPIRV-Reflect...")
	FetchContent_Declare( SPIRV-Reflect
		GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Reflect.git
		GIT_TAG        ${VULKAN_VERSION}
		GIT_SHALLOW    TRUE
		GIT_PROGRESS   TRUE
	)
	set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "")
	set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "")
	FetchContent_MakeAvailable(SPIRV-Reflect)
	target_link_libraries     (skshaderc_lib PUBLIC spirv-reflect-static)

	# This is used to convert HLSL to SPIR-V. DXC is an alternative, but is far
	# more difficult to compile, and generates a much larger binary.
	message(STATUS "Fetching glslang...")
	FetchContent_Declare( glslang
		GIT_REPOSITORY https://github.com/maluoi/glslang.git # Custom fork of glslang for vk::image_format support
		GIT_TAG        1d31fc505b53016a62167496da49bf0386fa08f7
	)
	set(GLSLANG_ENABLE_INSTALL  OFF  CACHE BOOL "")
	set(GLSLANG_TESTS           OFF  CACHE BOOL "")
	set(ENABLE_GLSLANG_BINARIES OFF  CACHE BOOL "")
	set(ENABLE_SPVREMAPPER      OFF  CACHE BOOL "")
	set(ENABLE_GLSLANG_JS       OFF  CACHE BOOL "")
	set(ENABLE_HLSL             ON   CACHE BOOL "")
	set(ENABLE_RTTI             OFF  CACHE BOOL "")
	FetchContent_MakeAvailable(glslang)
	target_link_libraries     (skshaderc_lib PUBLIC glslang SPIRV)
	target_compile_definitions(skshaderc_lib PUBLIC SKSC_HAS_GLSLANG)
	target_sources            (skshaderc_lib PRIVATE sksc_hlsl.cpp sksc_meta.cpp sksc_wgsl.cpp)
endif()

if(SKSHADERC_ENABLE_WGSL)
	# Tint comes from the Dawn source tree at the pinned release. With every
	# Dawn backend disabled, Dawn's root CMake only adds third_party +
	# src/utils + src/tint — no code generators, no gigantic dependency fetch.
	# SPIRV-Headers/Tools targets already exist (fetched above at Dawn's pins),
	# so Dawn's third_party reuses them; abseil is populated here for it.
	message(STATUS "Fetching abseil (for Tint)...")
	FetchContent_Declare( dawn_abseil
		URL https://chromium.googlesource.com/chromium/src/third_party/abseil-cpp/+archive/88f6126f3dc8848ddf7901b6df3f741dc0a1f2c2.tar.gz
		SOURCE_SUBDIR  use-dawns-add-subdirectory # populate only, dawn adds it
	)
	FetchContent_MakeAvailable(dawn_abseil)

	message(STATUS "Fetching Dawn source (for Tint)...")
	FetchContent_Declare( dawn_tint
		URL https://github.com/google/dawn/archive/refs/tags/v20260713.082812.tar.gz
	)

	set(DAWN_ENABLE_D3D11      OFF CACHE BOOL "" FORCE)
	set(DAWN_ENABLE_D3D12      OFF CACHE BOOL "" FORCE)
	set(DAWN_ENABLE_METAL      OFF CACHE BOOL "" FORCE)
	set(DAWN_ENABLE_NULL       OFF CACHE BOOL "" FORCE)
	set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "" FORCE)
	set(DAWN_ENABLE_OPENGLES   OFF CACHE BOOL "" FORCE)
	set(DAWN_ENABLE_VULKAN     OFF CACHE BOOL "" FORCE)
	set(DAWN_USE_GLFW          OFF CACHE BOOL "" FORCE)
	set(DAWN_USE_X11           OFF CACHE BOOL "" FORCE)
	set(DAWN_USE_WAYLAND       OFF CACHE BOOL "" FORCE)
	set(DAWN_BUILD_SAMPLES     OFF CACHE BOOL "" FORCE)
	set(DAWN_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
	set(DAWN_BUILD_PROTOBUF    OFF CACHE BOOL "" FORCE)
	set(DAWN_BUILD_MONOLITHIC_LIBRARY OFF CACHE STRING "" FORCE)
	set(TINT_BUILD_SPV_READER     ON  CACHE BOOL "" FORCE)
	set(TINT_BUILD_WGSL_READER    ON  CACHE BOOL "" FORCE)
	set(TINT_BUILD_WGSL_WRITER    ON  CACHE BOOL "" FORCE)
	set(TINT_BUILD_SPV_WRITER     OFF CACHE BOOL "" FORCE)
	set(TINT_BUILD_GLSL_WRITER    OFF CACHE BOOL "" FORCE)
	set(TINT_BUILD_GLSL_VALIDATOR OFF CACHE BOOL "" FORCE)
	set(TINT_BUILD_HLSL_WRITER    OFF CACHE BOOL "" FORCE)
	set(TINT_BUILD_MSL_WRITER     OFF CACHE BOOL "" FORCE)
	set(TINT_BUILD_NULL_WRITER    OFF CACHE BOOL "" FORCE)
	set(TINT_BUILD_IR_BINARY      OFF CACHE BOOL "" FORCE)
	set(TINT_BUILD_TESTS          OFF CACHE BOOL "" FORCE)
	set(TINT_BUILD_CMD_TOOLS      OFF CACHE BOOL "" FORCE)
	set(TINT_LIB_FUZZING_ENGINE_LINK_OPTIONS "" CACHE STRING "" FORCE)
	set(DAWN_ABSEIL_DIR        "${dawn_abseil_SOURCE_DIR}"   CACHE PATH "" FORCE)
	set(DAWN_SPIRV_HEADERS_DIR "${spirv-headers_SOURCE_DIR}" CACHE PATH "" FORCE)
	set(DAWN_SPIRV_TOOLS_DIR   "${spirv-tools_SOURCE_DIR}"   CACHE PATH "" FORCE)

	FetchContent_MakeAvailable(dawn_tint)

	target_link_libraries     (skshaderc_lib PUBLIC libtint)
	target_include_directories(skshaderc_lib PRIVATE
		"${dawn_tint_SOURCE_DIR}"
		"${dawn_tint_SOURCE_DIR}/include")
	target_compile_definitions(skshaderc_lib PUBLIC SKSC_HAS_TINT)
	# Tint's public headers need C++20
	set_target_properties(skshaderc_lib PROPERTIES CXX_STANDARD 20)
endif()

# Optional SVSL backend. To build against an in-progress SVSL
# checkout instead of the fetched tag, override the source dir:
#   cmake -B build -DFETCHCONTENT_SOURCE_DIR_SVSL=/path/to/SVSL
option(SKSHADERC_ENABLE_SVSL "Build skshaderc with the optional SVSL shader compiler backend" ON)
if(SKSHADERC_ENABLE_SVSL)
	message(STATUS "Fetching SVSL...")
	FetchContent_Declare( SVSL
		GIT_REPOSITORY https://github.com/maluoi/SVSL.git
		GIT_TAG        v2026.7.17
		GIT_SHALLOW    TRUE
		GIT_PROGRESS   TRUE
	)
	FetchContent_MakeAvailable(SVSL)
	target_link_libraries     (skshaderc_lib PUBLIC svsl::svsl)
	target_compile_definitions(skshaderc_lib PUBLIC SKSC_HAS_SVSL)
	target_sources            (skshaderc_lib PRIVATE sksc_svsl.cpp)
endif()

if(NOT SKSHADERC_ENABLE_GLSLANG AND NOT SKSHADERC_ENABLE_SVSL)
	message(FATAL_ERROR "skshaderc needs at least one compiler backend: enable SKSHADERC_ENABLE_GLSLANG and/or SKSHADERC_ENABLE_SVSL.")
endif()

target_sources(skshaderc_lib PRIVATE
	sksc.cpp
	sksc_log.cpp
	sksc.h
	_sksc.h
	miniz.cpp
	miniz.h
	../sk_renderer/sksc_file.c
)
target_include_directories(skshaderc_lib PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	../sk_renderer/include
)

###############################################################################
# Executable: skshaderc
###############################################################################

add_executable(skshaderc main.cpp)
target_link_libraries(skshaderc PRIVATE skshaderc_lib)

# Static link MinGW runtime when cross-compiling to Windows
if(MINGW)
	target_link_options(skshaderc PRIVATE -static-libgcc -static-libstdc++ -static)
endif()

# Enable LTO for native builds, but not MinGW (GCC LTO + static libs = symbol issues)
if(NOT MINGW)
	set_target_properties(skshaderc PROPERTIES
		INTERPROCEDURAL_OPTIMIZATION TRUE
	)
endif()

if (NOT MSVC AND NOT APPLE)
	add_custom_command(TARGET skshaderc POST_BUILD
		COMMAND ${CMAKE_OBJCOPY} --only-keep-debug $<TARGET_FILE:skshaderc> $<TARGET_FILE_DIR:skshaderc>/$<TARGET_FILE_BASE_NAME:skshaderc>.dbg
		COMMAND ${CMAKE_STRIP} --strip-debug $<TARGET_FILE:skshaderc>
		COMMAND ${CMAKE_OBJCOPY} --add-gnu-debuglink=$<TARGET_FILE_DIR:skshaderc>/$<TARGET_FILE_BASE_NAME:skshaderc>.dbg $<TARGET_FILE:skshaderc>
	)
endif()

endif() # NOT SKSHADERC_SKIP_BUILD

# Set SKSHADERC_EXE_PATH based on build mode:
# - Native builds (including Wine): use the skshaderc we just built
# - Cross-compiling: use the pre-built host binary for shader compilation
if(SKSHADERC_CROSS_COMPILING)
	set(SKSHADERC_EXE_PATH "${SKSHADERC_HOST_BIN}" CACHE INTERNAL "Path to skshaderc executable (host binary for cross-compilation)")
	set(SKSHADERC_DEPENDS  "${SKSHADERC_HOST_BIN}" CACHE INTERNAL "Extra DEPENDS so shaders recompile when skshaderc changes")
else()
	set(SKSHADERC_EXE_PATH $<TARGET_FILE:skshaderc> CACHE INTERNAL "Path to skshaderc executable")
	set(SKSHADERC_DEPENDS  skshaderc                CACHE INTERNAL "Extra DEPENDS so shaders recompile when skshaderc changes")
endif()
# Note: WINE variable is set earlier when detecting Windows-on-Unix cross-compilation

###############################################################################
# Install skshaderc binary to bin/tools/{platform}/ for distribution
# (only when we actually built skshaderc)
###############################################################################

if(NOT SKSHADERC_SKIP_BUILD)

# Copy skshaderc binary to bin/ in the repository root.

# Use a custom target (not POST_BUILD) so it runs every build, even if
# skshaderc itself is up-to-date. This ensures bin/ is repopulated if deleted.
add_custom_target(skshaderc_install ALL
	COMMAND ${CMAKE_COMMAND} -E make_directory "${SKSHADERC_TARGET_FOLDER}"
	# The install script refuses to replace a WGSL-capable tool with one that
	# lacks the target — web cross-builds depend on the fuller tool staying put
	COMMAND ${CMAKE_COMMAND} -DSKSHADERC_SRC=$<TARGET_FILE:skshaderc> -DSKSHADERC_DST_DIR=${SKSHADERC_TARGET_FOLDER} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_tool.cmake
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.distribute.txt" "${CMAKE_SOURCE_DIR}/bin/CMakeLists.txt"
	DEPENDS skshaderc
	COMMENT "Installing skshaderc to ${SKSHADERC_TARGET_FOLDER}"
)
message(STATUS "skshaderc will be copied to: ${SKSHADERC_TARGET_FOLDER}")

endif() # NOT SKSHADERC_SKIP_BUILD

# functions for compiling shaders easily
function(SKSHADERC_COMPILE_ASSETS TARGET OUT_PATH COMMAND_STRING OUT_LIST)
	set(SKSHADERC_COMPILE_COMMANDS "-e $<$<CONFIG:Debug>:-d> -o ${OUT_PATH} ${COMMAND_STRING}")
	message(STATUS "sk_renderer compiling shader assets with args 'skshaderc ${SKSHADERC_COMPILE_COMMANDS} [shaders]'")
	separate_arguments(SKSHADERC_COMPILE_COMMANDS)

	set(SHADER_LIST)
	set(COMPILED_SHADERS)

	foreach(SHADER IN LISTS ARGN)
		get_filename_component(SHADER_NAME ${SHADER} NAME)
		set(COMPILED_SHADER "${OUT_PATH}/${SHADER_NAME}.sks")

		add_custom_command(
			OUTPUT ${COMPILED_SHADER}
			COMMAND ${WINE} ${SKSHADERC_EXE_PATH} ${SKSHADERC_COMPILE_COMMANDS} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
			DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} ${SKSHADERC_DEPENDS}
			VERBATIM
		)
		list(APPEND SHADER_LIST ${SHADER})
		list(APPEND COMPILED_SHADERS ${COMPILED_SHADER})
	endforeach(SHADER)

	add_custom_target(${TARGET}_shaders DEPENDS ${COMPILED_SHADERS})
	add_dependencies(${TARGET} ${TARGET}_shaders)

	set(${OUT_LIST} ${SHADER_LIST} PARENT_SCOPE)
endfunction()

function(SKSHADERC_COMPILE_HEADERS ADD_TARGET OUTPUT_FOLDER COMMAND_STRING)
	set(SKSHADERC_COMPILE_COMMANDS "-h $<$<CONFIG:Debug>:-d> -o ${OUTPUT_FOLDER} ${COMMAND_STRING}")
	message(STATUS "sk_renderer compiling shader headers with args 'skshaderc ${SKSHADERC_COMPILE_COMMANDS} [shaders]'")
	separate_arguments(SKSHADERC_COMPILE_COMMANDS)

	set(SHADER_LIST)
	foreach(SHADER IN LISTS ARGN)
		get_filename_component(SHADER_NAME ${SHADER} NAME)
		add_custom_command(
			OUTPUT ${OUTPUT_FOLDER}/${SHADER_NAME}.h
			COMMAND ${WINE} ${SKSHADERC_EXE_PATH} ${SKSHADERC_COMPILE_COMMANDS} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
			DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} ${SKSHADERC_DEPENDS}
			VERBATIM
		)
		list(APPEND SHADER_LIST ${OUTPUT_FOLDER}/${SHADER_NAME}.h)
	endforeach(SHADER)

	target_include_directories(${ADD_TARGET} PRIVATE ${OUTPUT_FOLDER})
	target_sources            (${ADD_TARGET} PRIVATE ${SHADER_LIST})
endfunction()