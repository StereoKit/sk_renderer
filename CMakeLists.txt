# Android build. Needs ANDROID_HOME set, a JDK for APK signing, and the
# platform matching the example's MIN_SDK for the compile jar
# (`sdkmanager "platforms;android-29"`). Android can't run its own shader
# compiler, so the build makes a host skshaderc inside the build folder and
# compiles shaders with that. The first build of a fresh folder pays for it:
#
#   export ANDROID_NDK=$ANDROID_HOME/ndk/25.2.9519653   # or whichever is installed
#   cmake -B build-android \
#       -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#       -DANDROID_ABI=arm64-v8a \
#       -DANDROID_PLATFORM=android-32 \
#       -DANDROID_STL=c++_static \
#       -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-android -j$(nproc)
#
#   # Install + launch on a connected device (or manually:
#   #   adb install -r build-android/example/sk_renderer_test.apk
#   #   adb shell am start -n net.stereokit.renderer_test/net.stereokit.sk_app.SkAppActivity)
#   cmake --build build-android --target sk_renderer_test-run
#
#   # Filtered logcat for the app
#   adb logcat -v color --uid `adb shell pm list package -U net.stereokit.renderer_test | cut -d ":" -f3`
cmake_minimum_required(VERSION 3.10)
project(sk_renderer VERSION 0.1.0)

# Add custom cmake modules (includes headers-only FindVulkan for cross-compilation)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build options
option(SKR_BUILD_SKSHADERC "Build skshaderc shader compiler" ON)
option(SKR_BUILD_EXAMPLES "Build sk_renderer example application" ON)
option(SKR_BUILD_XR_EXAMPLE "Build OpenXR example application (Linux/Windows only)" OFF)
set(SKR_SK_APP_DIR "" CACHE PATH "Local sk_app checkout for the example (empty = fetch the pinned release)")

# Rendering backend, chosen at build time: Vulkan on native platforms, WebGPU
# for web builds. Native WebGPU (via Dawn) exists for developing/debugging the
# web backend and follows the same code paths as the WASM build.
option(SKR_USE_WEBGPU "Use the WebGPU backend instead of Vulkan (forced ON for Emscripten)" OFF)
if(EMSCRIPTEN)
	set(SKR_USE_WEBGPU ON)
endif()
if(SKR_USE_WEBGPU)
	message(STATUS "sk_renderer backend: webgpu")
else()
	message(STATUS "sk_renderer backend: vulkan")
endif()

include(FetchContent)

###############################################################################
# MoltenVK for macOS (Vulkan implementation via Metal)
###############################################################################

if(APPLE AND NOT SKR_USE_WEBGPU)
	message(STATUS "Fetching MoltenVK...")
	FetchContent_Declare(moltenvk
		URL https://github.com/KhronosGroup/MoltenVK/releases/download/v1.4.1/MoltenVK-macos.tar
	)
	FetchContent_MakeAvailable(moltenvk)

	# Create IMPORTED SHARED library for MoltenVK
	add_library(MoltenVK SHARED IMPORTED GLOBAL)
	set_target_properties(MoltenVK PROPERTIES
		IMPORTED_LOCATION "${moltenvk_SOURCE_DIR}/MoltenVK/dylib/macOS/libMoltenVK.dylib"
	)
	message(STATUS "MoltenVK: ${moltenvk_SOURCE_DIR}/MoltenVK/dylib/macOS/libMoltenVK.dylib")
endif()

# Helper function: copy MoltenVK dylib to an executable's output directory
# No-op on non-Apple platforms, so downstream can call unconditionally
# Usage: skr_copy_moltenvk(my_executable_target)
function(skr_copy_moltenvk target)
	if(APPLE AND TARGET MoltenVK)
		add_custom_command(TARGET ${target} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"$<TARGET_FILE:MoltenVK>"
				"$<TARGET_FILE_DIR:${target}>"
			COMMENT "Copying MoltenVK to ${target} output directory"
		)
	endif()
endfunction()

###############################################################################
# skshaderc shader compiler
###############################################################################

if(SKR_BUILD_SKSHADERC)
	# skshaderc handles cross-compilation internally:
	# - Native builds: builds skshaderc from source and uses it for shaders
	# - Cross-compiling: builds a second, host-targeted skshaderc to compile
	#   shaders with, since the target one can't run here
	add_subdirectory(skshaderc)
endif()

###############################################################################
# sk_renderer library
###############################################################################

if(ANDROID)
	add_library(sk_renderer STATIC)
	# Enable position-independent code for static library on Android
	set_target_properties(sk_renderer PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
	add_library(sk_renderer)
endif()

set_target_properties(sk_renderer PROPERTIES
	C_STANDARD 11
	C_STANDARD_REQUIRED ON)

# If we are building skshaderc, then sk_renderer should make skshaderc build
# first. This allows consuming projects to use skshaderc's compile functions.
# (Note: on some platforms like Android, the skshaderc target is skipped since
# shaders are compiled using the host binary and the target binary is unused)
if(SKR_BUILD_SKSHADERC AND TARGET skshaderc)
	add_dependencies(sk_renderer skshaderc)
endif()

if(NOT SKR_USE_WEBGPU)
	# Grab Vulkan headers first before Volk goes looking for its own. Good for a
	# few reasons, but notably avoids a problem when cross-compiling with mingw.
	message(STATUS "Fetching Vulkan-Headers...")
	FetchContent_Declare(
		vulkan_headers
		URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.328.1.zip
	)
	FetchContent_MakeAvailable(vulkan_headers)
	# Set paths for volk and our custom FindVulkan.cmake
	set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}" CACHE PATH "Vulkan headers for volk" FORCE)
	set(Vulkan_INCLUDE_DIR "${vulkan_headers_SOURCE_DIR}/include" CACHE PATH "Vulkan headers for FindVulkan" FORCE)

	message(STATUS "Fetching volk...")
	FetchContent_Declare( volk
		GIT_REPOSITORY https://github.com/zeux/volk.git
		GIT_TAG        1.4.304
		GIT_SHALLOW    TRUE
		GIT_PROGRESS   TRUE
	)
	FetchContent_MakeAvailable(volk)
	target_compile_definitions(sk_renderer PUBLIC SKR_VK)
	target_link_libraries(sk_renderer PUBLIC volk)
	if(ANDROID)
		set_target_properties(volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
		target_compile_definitions(volk         PRIVATE VK_USE_PLATFORM_ANDROID_KHR)
		target_compile_definitions(sk_renderer  PUBLIC  VK_USE_PLATFORM_ANDROID_KHR)
		target_link_libraries(sk_renderer PRIVATE log)  # Android logging library
	elseif(WIN32)
		target_compile_definitions(volk         PRIVATE VK_USE_PLATFORM_WIN32_KHR)
		target_compile_definitions(sk_renderer  PUBLIC  VK_USE_PLATFORM_WIN32_KHR)
	endif()
else() # webgpu
	target_compile_definitions(sk_renderer PUBLIC SKR_WEBGPU)
	if(EMSCRIPTEN)
		# emdawnwebgpu supplies webgpu.h plus the JS<->WASM bindings, and is
		# versioned with the same Dawn release the native build pins below.
		target_compile_options(sk_renderer PUBLIC "--use-port=emdawnwebgpu")
		target_link_options   (sk_renderer PUBLIC "--use-port=emdawnwebgpu")
	else()
		# Native WebGPU via Dawn's official prebuilt release binaries — building
		# Dawn from source pulls a multi-gigabyte Chromium-style dependency
		# tree, so don't. The same pinned release supplies emdawnwebgpu (web
		# builds above); keep both on this one version anchor.
		set(SKR_DAWN_VERSION "v20260713.082812" CACHE STRING "Pinned Dawn release tag")
		set(SKR_DAWN_COMMIT  "d8efac1b7b42d86e4d235686d4d824989d21369f" CACHE STRING "Commit hash in the pinned Dawn release's asset names")
		if(WIN32)
			# MSVC can't mix runtime configs with a static lib, so match ours
			set(_dawn_platform "windows-latest")
			if(CMAKE_BUILD_TYPE STREQUAL "Debug")
				set(_dawn_config "Debug")
			else()
				set(_dawn_config "Release")
			endif()
		elseif(APPLE)
			if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
				set(_dawn_platform "macos-15-intel")
			else()
				set(_dawn_platform "macos-latest")
			endif()
			set(_dawn_config "Release")
		else()
			set(_dawn_platform "ubuntu-latest")
			set(_dawn_config "Release")
		endif()
		message(STATUS "Fetching prebuilt Dawn ${SKR_DAWN_VERSION} (${_dawn_platform}-${_dawn_config})...")
		FetchContent_Declare(dawn_prebuilt
			URL https://github.com/google/dawn/releases/download/${SKR_DAWN_VERSION}/Dawn-${SKR_DAWN_COMMIT}-${_dawn_platform}-${_dawn_config}.tar.gz
		)
		FetchContent_MakeAvailable(dawn_prebuilt)
		find_package(Threads REQUIRED)  # DawnTargets links Threads::Threads but doesn't find_dependency it
		find_package(Dawn REQUIRED CONFIG NO_DEFAULT_PATH
			PATHS         "${dawn_prebuilt_SOURCE_DIR}"
			PATH_SUFFIXES lib64/cmake/Dawn lib/cmake/Dawn cmake/Dawn)
		target_link_libraries(sk_renderer PUBLIC dawn::webgpu_dawn)
	endif()
endif()

# Set symbol visibility to hidden by default, only export what's marked with SKR_API/SKSC_API
if(NOT MSVC)
	target_compile_options    (sk_renderer PRIVATE -fvisibility=hidden)
endif()
if (MINGW OR APPLE OR ANDROID)
	# MinGW, macOS, and Android may not support C11 threads, use pthread fallback
	if (NOT ANDROID)
		target_link_libraries(sk_renderer PRIVATE pthread)
	endif()
	target_include_directories(sk_renderer PRIVATE sk_renderer/threads)
endif()

target_include_directories(sk_renderer PUBLIC sk_renderer/include)
target_sources(sk_renderer PRIVATE
	sk_renderer/include/sk_renderer.h
	sk_renderer/skr_log.c
	sk_renderer/sksc_file.c
	sk_renderer/smolv.c
)
if(NOT SKR_USE_WEBGPU)
	target_sources(sk_renderer PRIVATE
		sk_renderer/vk/_sk_renderer.h
		sk_renderer/vk/skr_initialize.c
		sk_renderer/vk/skr_renderer.c
		sk_renderer/vk/skr_surface.c
		sk_renderer/vk/skr_buffer.c
		sk_renderer/vk/skr_shader.c
		sk_renderer/vk/skr_material.c
		sk_renderer/vk/skr_mesh.c
		sk_renderer/vk/skr_render_list.c
		sk_renderer/vk/skr_texture.c
		sk_renderer/vk/skr_scratch.h
		sk_renderer/vk/skr_scratch.c
		sk_renderer/vk/skr_transient.h
		sk_renderer/vk/skr_transient.c
		sk_renderer/vk/skr_pipeline.h
		sk_renderer/vk/skr_pipeline.c
		sk_renderer/vk/skr_compute.c
		sk_renderer/vk/skr_command.c
		sk_renderer/vk/skr_conversions.c
		sk_renderer/vk/skr_debug.c
		sk_renderer/vk/skr_destroy_list.c
	)
else() # webgpu
	target_sources(sk_renderer PRIVATE
		sk_renderer/wgpu/_sk_renderer.h
		sk_renderer/wgpu/skr_initialize.c
		sk_renderer/wgpu/skr_renderer.c
		sk_renderer/wgpu/skr_surface.c
		sk_renderer/wgpu/skr_buffer.c
		sk_renderer/wgpu/skr_shader.c
		sk_renderer/wgpu/skr_material.c
		sk_renderer/wgpu/skr_mesh.c
		sk_renderer/wgpu/skr_render_list.c
		sk_renderer/wgpu/skr_texture.c
		sk_renderer/wgpu/skr_pipeline.h
		sk_renderer/wgpu/skr_pipeline.c
		sk_renderer/wgpu/skr_compute.c
		sk_renderer/wgpu/skr_command.c
		sk_renderer/wgpu/skr_conversions.c
	)
endif()

# Built-in shaders: compiled to embedded byte-array headers so they ship inside
# the library binary. Used as fallbacks when blit-based mipgen isn't supported
# for a given texture format (see skr_texture.c skr_tex_generate_mips).
# A container carries one language, so the target follows the backend. WGSL
# comes only from the SVSL backend, so those builds pass -svsl; the host tool
# for web cross-builds may still carry glslang, and mixing backends isn't
# allowed.
if(SKR_USE_WEBGPU) # includes EMSCRIPTEN, which forces it ON above
	set(_skr_builtin_shader_args "-o3 -e -svsl -t w")
else()
	set(_skr_builtin_shader_args "-o3 -e -t s")
endif()
set(_skr_builtin_shaders
	sk_renderer/vk/builtin_shaders/skr_mipgen_2d.hlsl
	sk_renderer/vk/builtin_shaders/skr_mipgen_cube.hlsl
)
if(SKR_USE_WEBGPU) # multisampled depth resolve for pass lowering; Vulkan uses subpasses instead
	list(APPEND _skr_builtin_shaders sk_renderer/wgpu/builtin_shaders/skr_depth_resolve.hlsl)
endif()
skshaderc_compile_headers(sk_renderer ${CMAKE_BINARY_DIR}/sk_renderer_generated/builtin_shaders "${_skr_builtin_shader_args}"
	${_skr_builtin_shaders}
)

###############################################################################
# Example application
###############################################################################

if(SKR_BUILD_EXAMPLES)
	if(NOT SKR_BUILD_SKSHADERC)
		message(WARNING "SKR_BUILD_EXAMPLES requires SKR_BUILD_SKSHADERC to be enabled. Disabling examples.")
		set(SKR_BUILD_EXAMPLES OFF)
	else()
		# Set Android NDK path for example project
		if(ANDROID)
			set(ANDROID_NDK ${CMAKE_ANDROID_NDK})
		endif()

		add_subdirectory(example)
		if(NOT SKR_USE_WEBGPU)
			add_subdirectory(example_xr) # OpenXR is Vulkan-only
		endif()
	endif()
endif() # SKR_BUILD_EXAMPLES