cmake_minimum_required(VERSION 3.19)
project(sk_renderer_example VERSION 0.1.0)

include(FetchContent)

###############################################################################
# Dependencies
###############################################################################

# sk_app for windowing and platform abstraction
set(SKA_BUILD_EXAMPLES OFF CACHE BOOL "Disable sk_app examples" FORCE)
# Ensure sk_app is built with PIC for Android shared library
if(ANDROID)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
# SKR_SK_APP_DIR points at a local sk_app checkout for development (e.g.
# -DSKR_SK_APP_DIR=../sk_app, resolved against the top-level source dir);
# empty — the default — fetches the pinned release. v2026.7.20+ carries the
# WebGPU surface + web platform support the WebGPU backend needs.
if(SKR_SK_APP_DIR)
	get_filename_component(_skr_sk_app_dir "${SKR_SK_APP_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
	if(NOT EXISTS "${_skr_sk_app_dir}/CMakeLists.txt")
		message(FATAL_ERROR "SKR_SK_APP_DIR (${_skr_sk_app_dir}) doesn't look like an sk_app checkout")
	endif()
	message(STATUS "Using sk_app from ${_skr_sk_app_dir}")
	FetchContent_Declare(sk_app
		SOURCE_DIR "${_skr_sk_app_dir}"
	)
else()
	message(STATUS "Fetching sk_app v2026.7.20...")
	FetchContent_Declare(sk_app
		GIT_REPOSITORY https://github.com/StereoKit/sk_app.git
		GIT_TAG        v2026.7.20
		GIT_SHALLOW    TRUE
		GIT_PROGRESS   TRUE
	)
endif()
FetchContent_MakeAvailable(sk_app)
if(ANDROID)
	set_target_properties(sk_app PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# stb for image loading
message(STATUS "Fetching stb...")
FetchContent_Declare(stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
)
FetchContent_MakeAvailable(stb)

# cgltf for GLTF model loading
message(STATUS "Fetching cgltf...")
FetchContent_Declare(cgltf
	GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
	GIT_TAG        v1.14
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cgltf)

# cimgui C bindings for ImGui
message(STATUS "Fetching cimgui...")
FetchContent_Declare(cimgui
	GIT_REPOSITORY https://github.com/cimgui/cimgui.git
	GIT_TAG        1.91.6
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cimgui)

# FFmpeg for video playback (Linux only for now; the zero-copy path imports
# VkImages, so it's Vulkan-backend only)
if(UNIX AND NOT ANDROID AND NOT APPLE AND NOT SKR_USE_WEBGPU)
	find_package(PkgConfig QUIET)
	if(PkgConfig_FOUND)
		pkg_check_modules(FFMPEG
			libavcodec>=58.0
			libavformat>=58.0
			libavutil>=56.0
			libswscale>=5.0
		)
		if(FFMPEG_FOUND)
			message(STATUS "FFmpeg found - video playback enabled")
			set(SKR_HAS_VIDEO TRUE)
		else()
			message(STATUS "FFmpeg not found - video playback disabled")
			set(SKR_HAS_VIDEO FALSE)
		endif()
	else()
		message(STATUS "PkgConfig not found - video playback disabled")
		set(SKR_HAS_VIDEO FALSE)
	endif()
else()
	set(SKR_HAS_VIDEO FALSE)
endif()

# Android: use NDK MediaCodec + AHB for zero-copy video
if(ANDROID AND NOT SKR_HAS_VIDEO)
	set(SKR_HAS_VIDEO TRUE)
	set(SKR_VIDEO_ANDROID TRUE)
endif()

###############################################################################
# sk_renderer_test target
###############################################################################

# Use sk_app's add_skapp() helper for cross-platform builds
# Always use our custom manifest on Android for permissions (INTERNET, etc.)
if(ANDROID)
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		set(SKR_MANIFEST MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/android/AndroidManifest_Debug.xml")
	else()
		set(SKR_MANIFEST MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/android/AndroidManifest.xml")
	endif()
endif()

add_skapp(sk_renderer_test
	PACKAGE_NAME net.stereokit.renderer_test
	APP_NAME "sk_renderer_test"
	MIN_SDK 29
	${SKR_MANIFEST}
)

set_target_properties(sk_renderer_test PROPERTIES
	C_STANDARD 11
	C_STANDARD_REQUIRED ON
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)

if(UNIX AND NOT ANDROID)
	target_link_libraries(sk_renderer_test PRIVATE m)
endif()

if(EMSCRIPTEN)
	# The pre-init JS requests the WebGPU adapter/device before main() runs;
	# EXIT_RUNTIME lets test runs report a real exit code through emrun
	target_link_options(sk_renderer_test PRIVATE
		"SHELL:--pre-js ${CMAKE_CURRENT_SOURCE_DIR}/web/pre.js"
		"SHELL:--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html"
		-sEXIT_RUNTIME=1
		# The scenes keep matrices/instance staging on the stack; emscripten's
		# 64KB default is far below the 8MB native builds enjoy
		-sSTACK_SIZE=2MB)
	# Size pass: closure-compile the JS glue in MinSizeRel builds (pre.js uses
	# quoted Module properties so the minifier can't break the boundary)
	if(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
		target_link_options(sk_renderer_test PRIVATE --closure=1)
	endif()
	set_property(TARGET sk_renderer_test APPEND PROPERTY LINK_DEPENDS
		"${CMAKE_CURRENT_SOURCE_DIR}/web/pre.js"
		"${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html")
endif()

# Static link MinGW runtime when cross-compiling to Windows
if(MINGW)
	target_link_options       (sk_renderer_test PRIVATE -static-libgcc -static-libstdc++ -static)
	target_link_libraries     (sk_renderer_test PRIVATE pthread)
	target_include_directories(sk_renderer_test PRIVATE ${CMAKE_SOURCE_DIR}/sk_renderer/threads)
	# MSVC enables SSE4.1 intrinsics on x64 unconditionally; mingw-w64 GCC needs
	# -msse4.1 explicitly so BC1/ETC SIMD paths can compile.
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
		target_compile_options(sk_renderer_test PRIVATE -msse4.1)
	endif()
endif()

# macOS needs pthread fallback for C11 threads (no static linking)
if(APPLE)
	target_link_libraries     (sk_renderer_test PRIVATE pthread)
	target_include_directories(sk_renderer_test PRIVATE ${CMAKE_SOURCE_DIR}/sk_renderer/threads)
endif()

target_link_libraries(sk_renderer_test PRIVATE sk_renderer)
skr_copy_moltenvk(sk_renderer_test)
target_include_directories(sk_renderer_test PRIVATE
	${stb_SOURCE_DIR}
	${cgltf_SOURCE_DIR}
	${cimgui_SOURCE_DIR}/imgui
	${cimgui_SOURCE_DIR}
	${sk_app_SOURCE_DIR}/examples/imgui_example
)

target_sources(sk_renderer_test PRIVATE
	main.c
	app.c
	scene_meshes.c
	scene_reaction_diffusion.c
	scene_orbital_particles.c
	scene_impostor.c
	scene_array_texture.c
	scene_3d_texture.c
	scene_cubemap.c
	scene_gltf.c
	scene_shadows.c
	scene_cloth.c
	scene_text.c
	scene_tex_copy.c
	scene_lifetime_stress.c
	scene_gaussian_splat.c
	scene_tex_compress.c
	scene_stars.c
	scene_yuv_test.c
	scene_gi.c
	scene_pbr.c
	bloom.c
	text/text.c
	tools/scene_util.c
	tools/compress/tex_compress.c
	tools/tex_psnr.c
	tools/compress/astc_io.c
	imgui_backend/imgui_impl_sk_renderer.cpp
	${sk_app_SOURCE_DIR}/examples/imgui_example/imgui_impl_sk_app.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui_draw.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui_tables.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui_widgets.cpp
	${cimgui_SOURCE_DIR}/imgui/imgui_demo.cpp
	${cimgui_SOURCE_DIR}/cimgui.cpp
)
target_skapp_assets(sk_renderer_test
	${CMAKE_CURRENT_SOURCE_DIR}/assets)

# Video playback
if(SKR_HAS_VIDEO)
	target_compile_definitions(sk_renderer_test PRIVATE SKR_HAS_VIDEO)
	target_sources            (sk_renderer_test PRIVATE scene_video.c)
	if(SKR_VIDEO_ANDROID)
		# Android: NDK MediaCodec + AHB zero-copy path
		target_sources        (sk_renderer_test PRIVATE tools/video_android.c)
		target_link_libraries (sk_renderer_test PRIVATE mediandk)
	else()
		# Desktop: FFmpeg with Vulkan hardware acceleration
		target_include_directories(sk_renderer_test PRIVATE ${FFMPEG_INCLUDE_DIRS})
		target_link_libraries     (sk_renderer_test PRIVATE ${FFMPEG_LIBRARIES})
		target_sources            (sk_renderer_test PRIVATE tools/video.c)
	endif()
endif()

###############################################################################
# Shader compilation
###############################################################################

# Compile ImGui shader to header file (no debug info, embedded in binary).
# A container carries one language, so the target follows the backend. WGSL
# comes only from the SVSL backend, so those builds pass -svsl; the web host
# tool may still carry glslang, and mixing backends isn't allowed.
if(SKR_USE_WEBGPU) # includes EMSCRIPTEN, which forces it ON above
	set(_skr_example_shader_targets "-svsl -t w")
else()
	set(_skr_example_shader_targets "-t s")
endif()
# Output goes to the build dir: the flags differ per backend (-t s vs -t w),
# so a source-tree copy would ping-pong between build configurations
skshaderc_compile_headers(sk_renderer_test ${CMAKE_CURRENT_BINARY_DIR}/imgui_shader "-o3 -e ${_skr_example_shader_targets}"
	imgui_backend/imgui.hlsl
)

# Compile other shaders to .sks files
skshaderc_compile_assets(sk_renderer_test ${CMAKE_CURRENT_BINARY_DIR}/generated/shaders "${_skr_example_shader_targets}" _
	assets/shaders/bloom_composite.hlsl
	assets/shaders/bloom_downsample.hlsl
	assets/shaders/bloom_upsample.hlsl
	assets/shaders/compute_test.hlsl
	assets/shaders/cubemap_mipgen.hlsl
	assets/shaders/cubemap_reflection.hlsl
	assets/shaders/cubemap_skybox.hlsl
	assets/shaders/equirect_to_cubemap.hlsl
	assets/shaders/mipgen_alpha_weighted_render.hlsl
	assets/shaders/orbital_particles.hlsl
	assets/shaders/pbr.hlsl
	assets/shaders/shadow_caster.hlsl
	assets/shaders/shadow_receiver.hlsl
	assets/shaders/stereo_display.hlsl
	assets/shaders/test.hlsl
	assets/shaders/text.hlsl
	assets/shaders/texture3d.hlsl
	assets/shaders/unlit.hlsl
	assets/shaders/yuv_unlit.hlsl
	assets/shaders/video.hlsl
	assets/shaders/video_ahb.hlsl
	assets/shaders/gaussian_splat.hlsl
	assets/shaders/gpu_sort_init.hlsl
	assets/shaders/gpu_sort_upsweep.hlsl
	assets/shaders/gpu_sort_scan.hlsl
	assets/shaders/gpu_sort_downsweep.hlsl
	assets/shaders/stars.hlsl
	assets/shaders/pbr_shadow.hlsl
	assets/shaders/pbr_gi.hlsl
	assets/shaders/pbr_gi_vertex.hlsl
	assets/shaders/pbr_cubemap.hlsl
	assets/shaders/gi_accumulate.hlsl
	assets/shaders/gi_capture.hlsl
	assets/shaders/gi_clear.hlsl
	assets/shaders/gi_debug.hlsl
	assets/shaders/gi_debug_voxel.hlsl
	assets/shaders/gi_propagate.hlsl
	assets/shaders/gi_voxelize.hlsl
	assets/shaders/gi_fast_voxelize.hlsl
	assets/shaders/gi_voxel_to_sh.hlsl
	assets/shaders/postfx_invert.hlsl
	assets/shaders/postfx_depth_fog.hlsl
	assets/shaders/postfx_depth_fog_ms.hlsl
	assets/shaders/postfx_fog_density.svsl
	assets/shaders/postfx_fog_scatter.svsl
	assets/shaders/postfx_fog_scatter_taps.svsl
	assets/shaders/postfx_fog_tile.svsl
	assets/shaders/postfx_halation.svsl
	assets/shaders/msaa_resolve.hlsl
	assets/shaders/msaa_resolve_wide.hlsl
	assets/shaders/msaa_resolve_oled.hlsl
	assets/shaders/upscale.hlsl
	assets/shaders/tex_compare.hlsl
	# GPU texture compression toolkit — sources live with the tool code in
	# tools/compress/ (see tools/compress/README.md); compiled .sks output
	# lands in generated/shaders/ alongside everything else.
	tools/compress/shaders/bc1_compress.hlsl
	tools/compress/shaders/bc6h_compress.hlsl
	tools/compress/shaders/bc7_compress.hlsl
	tools/compress/shaders/astc4x4_compress.hlsl
	tools/compress/shaders/astc6x6_compress.hlsl
	tools/compress/shaders/astc8x8hdr_compress.hlsl
	tools/compress/shaders/tex_psnr.hlsl
)
target_skapp_assets(sk_renderer_test
	${CMAKE_CURRENT_BINARY_DIR}/generated)

###############################################################################
# ASTC encoder validation tool (desktop only)
###############################################################################

add_subdirectory(tools/compress/validate)

###############################################################################
# "run" command
###############################################################################

if(NOT ANDROID)
	add_custom_target(run DEPENDS sk_renderer_test-package COMMAND $<TARGET_FILE:sk_renderer_test>)
endif()