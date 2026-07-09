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
	endif()
elseif(NOT SKSHADERC_USE_WINE)
	message(STATUS "Native build: building skshaderc from source")
endif()

###############################################################################
# Skip building skshaderc binaries for certain platforms
# Android doesn't need the skshaderc binary - shaders are compiled using the
# host binary during the build, and the resulting .sks/.h files are used at runtime.
###############################################################################

if(ANDROID)
	set(SKSHADERC_SKIP_BUILD TRUE)
	message(STATUS "Android target: skipping skshaderc build (using host binary for shader compilation)")
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
if(SKSHADERC_ENABLE_GLSLANG)
	# SPIRV-Headers underpins both SPIRV-Tools and glslang.
	message(STATUS "Fetching SPIRV-Headers...")
	FetchContent_Declare( SPIRV-Headers
		GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
		GIT_TAG        ${VULKAN_VERSION}
		GIT_SHALLOW    TRUE
		GIT_PROGRESS   TRUE
	)
	FetchContent_MakeAvailable(SPIRV-Headers)

	# SPIRV-Tools: core provides the disassembler (-raw/-h text output), and
	# SPIRV-Tools-opt the optimizer. A baseline amount of optimization is crucial
	# for meta compatability with HLSL compilers.
	message(STATUS "Fetching SPIRV-Tools...")
	FetchContent_Declare( SPIRV-Tools
		GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
		GIT_TAG        ${VULKAN_VERSION}
		GIT_SHALLOW    TRUE
		GIT_PROGRESS   TRUE
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

# Optional SVSL backend. When enabled, we fetch libsvsl (its CLI, tests, and
# install rules stay off automatically for a FetchContent consumer), link it,
# and compile the sksc_svsl.cpp bridge. SKSC_HAS_SVSL gates the backend in code.
# The pinned libsvsl must emit the same SKS container version this repo's loader
# expects (currently v10) — the tag below and sksc_file.c's version check move
# in lockstep. See docs/PLAN_attribute_remap.md.
option(SKSHADERC_ENABLE_SVSL "Build skshaderc with the optional SVSL shader compiler backend" ON)
if(SKSHADERC_ENABLE_SVSL)
	message(STATUS "Fetching SVSL...")
	FetchContent_Declare( SVSL
		GIT_REPOSITORY https://github.com/maluoi/SVSL.git
		GIT_TAG        v2026.7.9
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
else()
	set(SKSHADERC_EXE_PATH $<TARGET_FILE:skshaderc> CACHE INTERNAL "Path to skshaderc executable")
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
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:skshaderc> "${SKSHADERC_TARGET_FOLDER}/"
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
			DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
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
			VERBATIM
		)
		list(APPEND SHADER_LIST ${OUTPUT_FOLDER}/${SHADER_NAME}.h)
	endforeach(SHADER)

	target_include_directories(${ADD_TARGET} PRIVATE ${OUTPUT_FOLDER})
	target_sources            (${ADD_TARGET} PRIVATE ${SHADER_LIST})
endfunction()