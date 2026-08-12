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

cmake_minimum_required(VERSION 3.19)

# Set again here rather than inherited: skshaderc is also a standalone project,
# and the cmake_minimum_required above resets policies for this scope either way.
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

project(skshaderc VERSION 1.0 DESCRIPTION "Shader compiler for sk_renderer")

# skr_target_warnings comes from sk_renderer's root when this is a subdirectory,
# but the cross-compile host tool below configures skshaderc standalone, with no
# sk_renderer scope to inherit it from.
if(NOT COMMAND skr_target_warnings)
	include("${CMAKE_CURRENT_LIST_DIR}/../cmake/skr_warnings.cmake")
endif()

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

###############################################################################
# Tool locations
#
# Tools install to <bin>/tools/<platform>/, and cross-builds look for their
# host tool in that same layout. Both default into the build folder, so a build
# tree builds every tool it needs, and never reads or writes another's.
#
#   SKSHADERC_BIN_DIR         Where tools install. Point it somewhere shared to
#                             collect tools from several configs for
#                             distribution. That folder holds one tool per
#                             platform, so configs built with different
#                             backends will overwrite each other there.
#   SKSHADERC_BUILD_HOST_TOOL Build a host tool for cross-builds. ON by default.
#                             OFF takes the host tool from SKSHADERC_BIN_DIR.
#   SKSHADERC_HOST_EXE        An exact prebuilt host tool, wherever it lives.
#
# A parent project can set these as normal variables before
# add_subdirectory(skshaderc), which shadows the cache entries.
###############################################################################

set(SKSHADERC_BIN_DIR  "${CMAKE_BINARY_DIR}/bin" CACHE PATH     "Folder skshaderc binaries are installed to, as <dir>/tools/<platform>/skshaderc")
set(SKSHADERC_HOST_EXE ""                        CACHE FILEPATH "Prebuilt host skshaderc to use when cross-compiling (empty = build one, or look in SKSHADERC_BIN_DIR)")
option(SKSHADERC_BUILD_HOST_TOOL "When cross-compiling, build a host skshaderc from source rather than expecting a prebuilt one" ON)

set(SKSHADERC_HOST_BIN      "${SKSHADERC_BIN_DIR}/tools/${_HOST_SYS}_${_HOST_ARCH}/skshaderc${_HOST_EXTENSION}")
set(SKSHADERC_TARGET_FOLDER "${SKSHADERC_BIN_DIR}/tools/${_TARGET_SYS}_${_TARGET_ARCH}")
if(SKSHADERC_HOST_EXE)
	set(SKSHADERC_HOST_BIN "${SKSHADERC_HOST_EXE}")
endif()

# Sort out the host tool for cross-builds (the Wine case isn't cross-compiling)
if(SKSHADERC_CROSS_COMPILING AND SKSHADERC_BUILD_HOST_TOOL AND NOT SKSHADERC_HOST_EXE)
	# skshaderc is a standalone CMake project, so ExternalProject can configure
	# it again with the host's toolchain, inheriting nothing from the cross one.
	include(ExternalProject)

	# Always Release, so a Debug cross build doesn't wait on a debug-speed
	# shader compiler.
	set(_HOST_TOOL_ARGS
		"-DSKSHADERC_BIN_DIR=${SKSHADERC_BIN_DIR}"
		"-DCMAKE_BUILD_TYPE=Release"
		# Picks the backends the target build expects: WebGPU gets SVSL with the
		# WGSL emitter, everything else keeps glslang.
		"-DSKR_USE_WEBGPU=${SKR_USE_WEBGPU}")
	# A parent project may have picked backends itself (e.g. SVSL-only, which
	# skips the expensive glslang build). Mirror any explicit choice; undefined
	# means the sub-build keeps its own defaults.
	if(DEFINED SKSHADERC_ENABLE_GLSLANG)
		list(APPEND _HOST_TOOL_ARGS "-DSKSHADERC_ENABLE_GLSLANG=${SKSHADERC_ENABLE_GLSLANG}")
	endif()
	if(DEFINED SKSHADERC_ENABLE_SVSL)
		list(APPEND _HOST_TOOL_ARGS "-DSKSHADERC_ENABLE_SVSL=${SKSHADERC_ENABLE_SVSL}")
	endif()
	if(FETCHCONTENT_SOURCE_DIR_SVSL) # honor an in-progress local SVSL checkout
		list(APPEND _HOST_TOOL_ARGS "-DFETCHCONTENT_SOURCE_DIR_SVSL=${FETCHCONTENT_SOURCE_DIR_SVSL}")
	endif()
	if(CMAKE_MAKE_PROGRAM) # e.g. the Android SDK's ninja, which is a host binary
		list(APPEND _HOST_TOOL_ARGS "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")
	endif()
	if(MSVC) # don't inherit the cross target's -A, or we'd rebuild the same arch
		if(_HOST_ARCH STREQUAL "arm64")
			list(APPEND _HOST_TOOL_ARGS "-AARM64")
		else()
			list(APPEND _HOST_TOOL_ARGS "-Ax64")
		endif()
	endif()

	# emcmake exports CC/CXX pointing at emcc; scrub them so the host build
	# finds a real native compiler.
	set(_HOST_TOOL_ENV)
	if(EMSCRIPTEN)
		set(_HOST_TOOL_ENV ${CMAKE_COMMAND} -E env --unset=CC --unset=CXX)
	endif()

	# A custom BUILD_COMMAND doesn't inherit make's jobserver, so without this
	# the host tool builds one file at a time.
	include(ProcessorCount)
	ProcessorCount(_HOST_TOOL_JOBS)
	if(_HOST_TOOL_JOBS EQUAL 0)
		set(_HOST_TOOL_JOBS 1)
	endif()

	ExternalProject_Add(skshaderc_host
		SOURCE_DIR        "${CMAKE_CURRENT_SOURCE_DIR}"
		BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/host-tool"
		CONFIGURE_COMMAND ${_HOST_TOOL_ENV} ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ${_HOST_TOOL_ARGS} <SOURCE_DIR>
		BUILD_COMMAND     ${_HOST_TOOL_ENV} ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --parallel ${_HOST_TOOL_JOBS}
		INSTALL_COMMAND   "" # the sub-build's skshaderc_install target does the copy
		BUILD_ALWAYS      TRUE # so edits to skshaderc's own sources take effect
		BUILD_BYPRODUCTS  "${SKSHADERC_HOST_BIN}"
		USES_TERMINAL_CONFIGURE TRUE
		USES_TERMINAL_BUILD     TRUE)

	message(STATUS "Cross-compiling (${_HOST_SYS}_${_HOST_ARCH} -> ${_TARGET_SYS}_${_TARGET_ARCH}) building host skshaderc into ${SKSHADERC_HOST_BIN}")
elseif(SKSHADERC_CROSS_COMPILING)
	if(NOT EXISTS "${SKSHADERC_HOST_BIN}")
		message(FATAL_ERROR
			"Cross-compiling (${_HOST_SYS}_${_HOST_ARCH} -> ${_TARGET_SYS}_${_TARGET_ARCH}) requires "
			"a host compatible skshaderc binary, and there is none at ${SKSHADERC_HOST_BIN}!\n"
			""
			"SKSHADERC_BUILD_HOST_TOOL is OFF, so no host tool is being built. Either "
			"turn it back ON to build one as part of this build, point "
			"-DSKSHADERC_HOST_EXE=<path> at an existing binary, or build for the host "
			"platform (${_HOST_SYS}_${_HOST_ARCH}) into the folder this build reads "
			"from:\n"
			""
			"    cmake -B build-${_HOST_SYS} -DSKSHADERC_BIN_DIR=${SKSHADERC_BIN_DIR} . && cmake --build build-${_HOST_SYS}")
	else()
		message(STATUS "Cross-compiling (${_HOST_SYS}_${_HOST_ARCH} -> ${_TARGET_SYS}_${_TARGET_ARCH}) using host skshaderc from ${SKSHADERC_HOST_BIN}")
		# Ensure executable permission on Unix hosts
		if(CMAKE_HOST_UNIX)
			file(CHMOD "${SKSHADERC_HOST_BIN}" PERMISSIONS
				OWNER_EXECUTE OWNER_WRITE OWNER_READ
				GROUP_EXECUTE GROUP_READ
				WORLD_EXECUTE WORLD_READ)
		endif()
		# Web builds compile shaders to WGSL, which comes solely from SVSL's
		# native WGSL emitter; a host tool built without it would emit
		# unloadable .sks files. The emitter embeds the reserved override
		# name, so probe the binary for it.
		if(EMSCRIPTEN)
			file(STRINGS "${SKSHADERC_HOST_BIN}" _skshaderc_wgsl_probe REGEX "sk_view_index" LIMIT_COUNT 1)
			if(NOT _skshaderc_wgsl_probe)
				message(FATAL_ERROR
					"The host skshaderc at ${SKSHADERC_HOST_BIN} was built without the SVSL "
					"WGSL emitter (SVSL_ENABLE_WGSL), which web builds require.\n"
					"Turn SKSHADERC_BUILD_HOST_TOOL back ON to build a WGSL-capable host "
					"tool as part of this build, or point -DSKSHADERC_HOST_EXE=<path> at one.")
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
	C_STANDARD 11
	C_STANDARD_REQUIRED ON
	CXX_STANDARD 11
	CXX_STANDARD_REQUIRED ON)

skr_target_warnings(skshaderc_lib)

# Optional glslang backend: HLSL -> SPIR-V via glslang, optimized with the
# SPIRV-Tools optimizer and reflected with SPIRV-Reflect. glslang also pulls in
# the SPIRV-Tools disassembler, which powers the -raw/-h SPIR-V text output; a
# build without glslang skips all of SPIRV-Headers/Tools/Reflect (well over 3MB
# of the binary) and instead writes raw .spv for -raw and omits SPIR-V comments
# in -h. SKSC_HAS_GLSLANG and SKSC_HAS_SPIRV_TOOLS gate this in code.
option(SKSHADERC_ENABLE_GLSLANG "Build skshaderc with the glslang (HLSL) shader compiler backend" ON)

# WebGPU targets compile every shader through SVSL: WGSL comes from SVSL's
# native emitter, and keeping glslang out of these builds guarantees a .sks
# never mixes compiler backends across its stages.
if(SKR_USE_WEBGPU AND SKSHADERC_ENABLE_GLSLANG)
	message(STATUS "WebGPU build: shaders compile through SVSL only, disabling the glslang backend")
	set(SKSHADERC_ENABLE_GLSLANG OFF)
endif()

if(SKSHADERC_ENABLE_GLSLANG)
	# SPIRV-Headers underpins both SPIRV-Tools and glslang.
	message(STATUS "Fetching SPIRV-Headers...")
	FetchContent_Declare( SPIRV-Headers
		URL https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/${VULKAN_VERSION}.zip
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	)
	FetchContent_MakeAvailable(SPIRV-Headers)

	# SPIRV-Tools: core provides the disassembler (-raw/-h text output), and
	# SPIRV-Tools-opt the optimizer. A baseline amount of optimization is crucial
	# for meta compatability with HLSL compilers.
	message(STATUS "Fetching SPIRV-Tools...")
	FetchContent_Declare( SPIRV-Tools
		URL https://github.com/KhronosGroup/SPIRV-Tools/archive/refs/tags/${VULKAN_VERSION}.zip
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
	target_sources            (skshaderc_lib PRIVATE sksc_hlsl.cpp sksc_meta.cpp)
endif()

# Optional SVSL backend. To build against an in-progress SVSL
# checkout instead of the fetched tag, override the source dir:
#   cmake -B build -DFETCHCONTENT_SOURCE_DIR_SVSL=/path/to/SVSL
option(SKSHADERC_ENABLE_SVSL "Build skshaderc with the optional SVSL shader compiler backend" ON)
if(SKSHADERC_ENABLE_SVSL)
	message(STATUS "Fetching SVSL...")
	FetchContent_Declare( SVSL
		GIT_REPOSITORY https://github.com/maluoi/SVSL.git
		GIT_TAG        v2026.8.11
		GIT_SHALLOW    TRUE
		GIT_PROGRESS   TRUE
	)
	# SVSL's WGSL backend is pure C text emission (still zero-dependency), and
	# supplies skshaderc's whole '-t w' target — WebGPU builds have no other
	# WGSL source.
	set(SVSL_ENABLE_WGSL ON CACHE BOOL "" FORCE)
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
	../sk_renderer/smolv.c
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

skr_target_warnings(skshaderc)

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
	if(TARGET skshaderc_host)
		# The target orders shaders after the host build, the file makes them
		# recompile when it changes.
		set(SKSHADERC_DEPENDS "${SKSHADERC_HOST_BIN}" skshaderc_host CACHE INTERNAL "Extra DEPENDS so shaders recompile when skshaderc changes")
	else()
		set(SKSHADERC_DEPENDS "${SKSHADERC_HOST_BIN}" CACHE INTERNAL "Extra DEPENDS so shaders recompile when skshaderc changes")
	endif()
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

# Use a custom target (not POST_BUILD) so it runs every build, even if
# skshaderc itself is up-to-date. This ensures bin/ is repopulated if deleted.
add_custom_target(skshaderc_install ALL
	COMMAND ${CMAKE_COMMAND} -E make_directory "${SKSHADERC_TARGET_FOLDER}"
	# The install script refuses to replace a WGSL-capable tool with one that
	# lacks the target — web cross-builds depend on the fuller tool staying put
	COMMAND ${CMAKE_COMMAND} -DSKSHADERC_SRC=$<TARGET_FILE:skshaderc> -DSKSHADERC_DST_DIR=${SKSHADERC_TARGET_FOLDER} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_tool.cmake
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.distribute.txt" "${SKSHADERC_BIN_DIR}/CMakeLists.txt"
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