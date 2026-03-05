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
FetchContent_Declare(sk_app
	GIT_REPOSITORY https://github.com/StereoKit/sk_app.git
	GIT_TAG        42d3193afcaf16a0b2f1c1aacb7c964d1f0910cc
	GIT_SHALLOW    TRUE
	GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(sk_app)
if(ANDROID)
	set_target_properties(sk_app PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# stb for image loading
FetchContent_Declare(stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
	GIT_SHALLOW    TRUE
	GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(stb)

# cgltf for GLTF model loading
FetchContent_Declare(cgltf
	GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
	GIT_TAG        v1.14
	GIT_SHALLOW    TRUE
	GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(cgltf)

# cimgui C bindings for ImGui
FetchContent_Declare(cimgui
	GIT_REPOSITORY https://github.com/cimgui/cimgui.git
	GIT_TAG        1.91.6
	GIT_SHALLOW    TRUE
	GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(cimgui)

# FFmpeg for video playback (Linux only for now)
if(UNIX AND NOT ANDROID AND NOT APPLE)
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

# Static link MinGW runtime when cross-compiling to Windows
if(MINGW)
	target_link_options       (sk_renderer_test PRIVATE -static-libgcc -static-libstdc++ -static)
	target_link_libraries     (sk_renderer_test PRIVATE pthread)
	target_include_directories(sk_renderer_test PRIVATE ${CMAKE_SOURCE_DIR}/sk_renderer/threads)
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
	bloom.c
	text/text.c
	tools/scene_util.c
	tools/tex_compress.c
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

# Compile ImGui shader to header file (no debug info, embedded in binary)
# Output goes to imgui_backend/ to keep all ImGui backend files together
skshaderc_compile_headers(sk_renderer_test ${CMAKE_CURRENT_SOURCE_DIR}/imgui_backend "-o3 -e -t s"
	imgui_backend/imgui.hlsl
)

# Compile other shaders to .sks files
skshaderc_compile_assets(sk_renderer_test ${CMAKE_CURRENT_BINARY_DIR}/generated/shaders "" _
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
)
target_skapp_assets(sk_renderer_test
	${CMAKE_CURRENT_BINARY_DIR}/generated)

###############################################################################
# "run" command
###############################################################################

if(NOT ANDROID)
	add_custom_target(run DEPENDS sk_renderer_test-package COMMAND $<TARGET_FILE:sk_renderer_test>)
endif()