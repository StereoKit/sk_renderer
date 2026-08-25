cmake_minimum_required(VERSION 3.19)

# OpenXR Example for sk_renderer
# Demonstrates VR rendering using OpenXR, on both of sk_renderer's backends:
#
#   native  Vulkan (XR_KHR_vulkan_enable2) against the system OpenXR loader
#   web     WebGPU (XR_SKW_webgpu_enable)  against WebOpenXR, an OpenXR runtime
#           implemented on top of WebXR
#
# Reuses scene implementations from the example/ directory - those, and app_xr.c,
# are backend-agnostic sk_renderer code and compile unchanged for both.

# Ensure our cmake/ folder is in path to override FindMetalTools (skip Metal on macOS)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(FetchContent)

###################################
# The OpenXR implementation
#
#   native  the Khronos loader, which dispatches to whatever runtime is active
#   web     WebOpenXR. The loader is not an option here: it cannot build for
#           wasm, and there is no runtime for it to find - WebOpenXR *is* the
#           runtime, linked directly in place of the loader.
###################################

if(EMSCRIPTEN)

	set(SKR_WEBOPENXR_DIR "$ENV{HOME}/projects/WebOpenXR" CACHE PATH "WebOpenXR checkout")
	if(NOT EXISTS "${SKR_WEBOPENXR_DIR}/CMakeLists.txt")
		message(FATAL_ERROR
			"SKR_WEBOPENXR_DIR (${SKR_WEBOPENXR_DIR}) doesn't look like a WebOpenXR "
			"checkout - set -DSKR_WEBOPENXR_DIR=<path>.")
	endif()

	# Its test apps switch themselves off when it isn't the top-level project, so
	# this builds the runtime library and nothing else. It brings its own pinned
	# OpenXR 1.1.61 headers and the emdawnwebgpu port along on the target.
	FetchContent_Declare(webopenxr SOURCE_DIR "${SKR_WEBOPENXR_DIR}")
	FetchContent_MakeAvailable(webopenxr)
	set(SKR_XR_IMPL WebOpenXR::webopenxr)

else()

	message(STATUS "VULKAN_HEADERS_INSTALL_DIR = ${VULKAN_HEADERS_INSTALL_DIR}")
	FetchContent_Declare(
		openxr_loader
		GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-SDK.git
		GIT_TAG        release-1.1.43
		GIT_SHALLOW    TRUE
	)

	# Only build the loader, not samples/tests
	set(BUILD_ALL_EXTENSIONS OFF CACHE BOOL "" FORCE)  # Only need Vulkan, not Metal/OpenGL
	set(BUILD_LOADER ON CACHE BOOL "" FORCE)
	set(BUILD_API_LAYERS OFF CACHE BOOL "" FORCE)
	set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(BUILD_CONFORMANCE_TESTS OFF CACHE BOOL "" FORCE)
	set(DYNAMIC_LOADER OFF CACHE BOOL "" FORCE)

	FetchContent_MakeAvailable(openxr_loader)
	set(SKR_XR_IMPL openxr_loader)

endif()

###################################
# Dependencies from example/
# (FetchContent reuses existing downloads)
###################################

FetchContent_Declare(stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(cgltf
	GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
	GIT_TAG        v1.14
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cgltf)

FetchContent_Declare(cimgui
	GIT_REPOSITORY https://github.com/cimgui/cimgui.git
	GIT_TAG        1.91.6
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cimgui)

###################################
# Executable
###################################

add_skapp(sk_renderer_xr_test
	PACKAGE_NAME net.stereokit.renderer_test_xr
	APP_NAME "sk_renderer_test_xr"
	MIN_SDK 29
	MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/AndroidManifest.xml)

set(EXAMPLE_DIR ${CMAKE_SOURCE_DIR}/example)

target_sources(sk_renderer_xr_test PRIVATE
	main.c
	app_xr.c

	# Scene utilities and tools from example/
	${EXAMPLE_DIR}/tools/scene_util.c
	${EXAMPLE_DIR}/tools/compress/tex_compress.c
	${EXAMPLE_DIR}/tools/tex_psnr.c
	${EXAMPLE_DIR}/tools/compress/astc_io.c
	${EXAMPLE_DIR}/text/text.c

	# Scene implementations from example/ (linked, not copied)
	${EXAMPLE_DIR}/scene_meshes.c
	${EXAMPLE_DIR}/scene_reaction_diffusion.c
	${EXAMPLE_DIR}/scene_orbital_particles.c
	${EXAMPLE_DIR}/scene_impostor.c
	${EXAMPLE_DIR}/scene_array_texture.c
	${EXAMPLE_DIR}/scene_3d_texture.c
	${EXAMPLE_DIR}/scene_cubemap.c
	${EXAMPLE_DIR}/scene_gltf.c
	${EXAMPLE_DIR}/scene_shadows.c
	${EXAMPLE_DIR}/scene_cloth.c
	${EXAMPLE_DIR}/scene_text.c
	${EXAMPLE_DIR}/scene_tex_copy.c
	${EXAMPLE_DIR}/scene_lifetime_stress.c
	${EXAMPLE_DIR}/scene_gaussian_splat.c
	${EXAMPLE_DIR}/scene_tex_compress.c
	${EXAMPLE_DIR}/scene_stars.c
	${EXAMPLE_DIR}/scene_yuv_test.c
	${EXAMPLE_DIR}/scene_gi.c

	# cimgui (needed by scenes that have render_ui methods)
	${cimgui_SOURCE_DIR}/imgui/imgui.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui_draw.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui_tables.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui_widgets.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui_demo.cpp
	${cimgui_SOURCE_DIR}/cimgui.cpp
)

target_link_libraries(sk_renderer_xr_test PRIVATE
	sk_renderer
	${SKR_XR_IMPL}
)

skr_target_warnings(sk_renderer_xr_test)

target_include_directories(sk_renderer_xr_test PRIVATE
	${EXAMPLE_DIR}                          # For scene.h, app.h
	${EXAMPLE_DIR}/tools                    # For float_math.h, scene_util.h
	${stb_SOURCE_DIR}                       # For stb_image.h
	${cgltf_SOURCE_DIR}                     # For cgltf.h
	${cimgui_SOURCE_DIR}/imgui              # For imgui headers
	${cimgui_SOURCE_DIR}                    # For cimgui.h
)

if(NOT EMSCRIPTEN)
	# WebOpenXR carries its own headers on its target; the loader doesn't.
	target_include_directories(sk_renderer_xr_test PRIVATE
		${openxr_loader_SOURCE_DIR}/include
		${CMAKE_CURRENT_SOURCE_DIR}/../sk_renderer  # For vk/skr_vulkan.h
	)
	# Vulkan headers are already available via FetchContent in the root CMakeLists.txt
	# (vulkan_headers target and Vulkan_INCLUDE_DIR are set there)
	target_include_directories(sk_renderer_xr_test PRIVATE ${Vulkan_INCLUDE_DIR})
endif()

if(EMSCRIPTEN)
	# The page, and the WebGPU device. pre.js requests the adapter with
	# {xrCompatible:true}, which is the difference between this example entering
	# VR and failing at `new XRGPUBinding` - see the file for why.
	#
	# The shell supplies its own #wxr-enter-vr button so it matches the page;
	# WebOpenXR finds it by id and wires up the click. (It injects a plain one
	# when the page has none, which works but looks like a stray control.)
	target_link_options(sk_renderer_xr_test PRIVATE
		"SHELL:--pre-js ${CMAKE_CURRENT_SOURCE_DIR}/web/pre.js"
		"SHELL:--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html"
		# Scene matrices and instance staging live on the stack; emscripten's
		# 64KB default is far below what the native builds enjoy.
		-sSTACK_SIZE=2MB)
	set_property(TARGET sk_renderer_xr_test APPEND PROPERTY LINK_DEPENDS
		"${CMAKE_CURRENT_SOURCE_DIR}/web/pre.js"
		"${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html")
endif()

# Math library on Unix
if(UNIX AND NOT APPLE)
	target_link_libraries(sk_renderer_xr_test PRIVATE m)
endif()

# Static link MinGW runtime when cross-compiling to Windows
if(MINGW)
	target_link_options       (sk_renderer_xr_test PRIVATE -static-libgcc -static-libstdc++ -static)
	target_link_libraries     (sk_renderer_xr_test PRIVATE pthread)
	target_include_directories(sk_renderer_xr_test PRIVATE ${CMAKE_SOURCE_DIR}/sk_renderer/threads)
	# MSVC enables SSE4.1 intrinsics on x64 unconditionally; mingw-w64 GCC needs
	# -msse4.1 explicitly so BC1/ETC SIMD paths can compile.
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
		target_compile_options(sk_renderer_xr_test PRIVATE -msse4.1)
	endif()
endif()

# macOS needs pthread fallback for C11 threads (no static linking)
if(APPLE)
	target_link_libraries     (sk_renderer_xr_test PRIVATE pthread)
	target_include_directories(sk_renderer_xr_test PRIVATE ${CMAKE_SOURCE_DIR}/sk_renderer/threads)
endif()

###################################
# Shader compilation
###################################

# Compile all shaders needed by the example scenes.
#
# The target must be named explicitly for the web: skshaderc adds no -t of its
# own, and passing "" compiles "successfully" while emitting no WGSL at all.
# That surfaces much later, and far away, as "Shader has no WGSL stages". WGSL
# comes only from the SVSL backend and backends can't be mixed, hence -svsl
# alongside -t w.
if(EMSCRIPTEN)
	set(SKR_XR_SHADER_ARGS "-o3 -svsl -t w")
elseif(SKR_USE_WEBGPU)
	set(SKR_XR_SHADER_ARGS "-o3 -svsl -t sw")   # native Dawn: SPIR-V and WGSL
else()
	set(SKR_XR_SHADER_ARGS "")                  # Vulkan: skshaderc's own default
endif()

skshaderc_compile_assets(sk_renderer_xr_test ${CMAKE_CURRENT_BINARY_DIR}/generated/shaders "${SKR_XR_SHADER_ARGS}" _
	../example/assets/shaders/test.hlsl
	../example/assets/shaders/compute_test.hlsl
	../example/assets/shaders/cubemap_mipgen.hlsl
	../example/assets/shaders/cubemap_reflection.hlsl
	../example/assets/shaders/cubemap_skybox.hlsl
	../example/assets/shaders/equirect_to_cubemap.hlsl
	../example/assets/shaders/mipgen_alpha_weighted_render.hlsl
	../example/assets/shaders/orbital_particles.hlsl
	../example/assets/shaders/pbr.hlsl
	../example/assets/shaders/shadow_caster.hlsl
	../example/assets/shaders/shadow_receiver.hlsl
	../example/assets/shaders/stereo_display.hlsl
	../example/assets/shaders/text.hlsl
	../example/assets/shaders/texture3d.hlsl
	../example/assets/shaders/unlit.hlsl
	../example/assets/shaders/yuv_unlit.hlsl
	../example/assets/shaders/gaussian_splat.hlsl
	../example/assets/shaders/gpu_sort_init.hlsl
	../example/assets/shaders/gpu_sort_upsweep.hlsl
	../example/assets/shaders/gpu_sort_scan.hlsl
	../example/assets/shaders/gpu_sort_downsweep.hlsl
	../example/assets/shaders/stars.hlsl
	../example/assets/shaders/pbr_shadow.hlsl
	../example/assets/shaders/pbr_gi.hlsl
	../example/assets/shaders/pbr_gi_vertex.hlsl
	../example/assets/shaders/pbr_cubemap.hlsl
	../example/assets/shaders/gi_accumulate.hlsl
	../example/assets/shaders/gi_capture.hlsl
	../example/assets/shaders/gi_clear.hlsl
	../example/assets/shaders/gi_debug.hlsl
	../example/assets/shaders/gi_debug_voxel.hlsl
	../example/assets/shaders/gi_propagate.hlsl
	../example/assets/shaders/gi_voxelize.hlsl
	../example/assets/shaders/gi_fast_voxelize.hlsl
	../example/assets/shaders/gi_voxel_to_sh.hlsl

	INCLUDES
	../example/assets/shaders/common.hlsli
	../example/assets/shaders/gpu_sort_common.hlsli
	../example/assets/shaders/pbr.hlsli
)
target_skapp_assets(sk_renderer_xr_test ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Include static assets from example/ (textures, models, fonts, etc.)
target_skapp_assets(sk_renderer_xr_test ${EXAMPLE_DIR}/assets)

###################################
# Run target
###################################

add_custom_target(run_xr DEPENDS sk_renderer_xr_test
	COMMAND $<TARGET_FILE:sk_renderer_xr_test>
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
