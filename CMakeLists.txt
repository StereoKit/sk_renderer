cmake_minimum_required(VERSION 3.10)
project(sk_renderer VERSION 0.1.0 LANGUAGES CXX C)

# For thread_local
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(FetchContent)

###############################################################################
# skshaderc shader compiler
###############################################################################

if(ANDROID)
	# For Android builds, use pre-built skshaderc from bin/
	# (Build the host binary first with a native build)
	add_subdirectory(${CMAKE_SOURCE_DIR}/bin bin_tools)
else()
	# Native builds: build skshaderc from source
	add_subdirectory(skshaderc)
endif()

###############################################################################
# sk_renderer library
###############################################################################

FetchContent_Declare( volk
	GIT_REPOSITORY https://github.com/zeux/volk.git
	GIT_TAG        1.3.270
	GIT_SHALLOW    TRUE
	GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(volk)

if(ANDROID)
	# Enable PIC for volk static library on Android
	set_target_properties(volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(ANDROID)
	add_library(sk_renderer STATIC)
	# Enable position-independent code for static library on Android
	set_target_properties(sk_renderer PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
	add_library(sk_renderer)
endif()
target_link_libraries     (sk_renderer PUBLIC volk)
target_include_directories(sk_renderer PUBLIC sk_renderer/include)
target_sources            (sk_renderer PRIVATE
	sk_renderer/include/sk_renderer.h
	sk_renderer/skr_log.c
	sk_renderer/sksc_file.c
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
	sk_renderer/vk/skr_pipeline.h
	sk_renderer/vk/skr_pipeline.c
	sk_renderer/vk/skr_compute.c
	sk_renderer/vk/skr_command.c
	sk_renderer/vk/skr_conversions.c
	sk_renderer/vk/skr_debug.c
	sk_renderer/vk/skr_destroy_list.c
)
if(NOT ANDROID)
	target_link_libraries(sk_renderer PRIVATE pthread)
endif()

###############################################################################
# sk_renderer test app
###############################################################################

if(ANDROID)
	set(ANDROID_NDK ${CMAKE_ANDROID_NDK})
endif()
FetchContent_Declare( sdl2
	GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
	GIT_TAG        release-2.30.10
	GIT_SHALLOW    TRUE
	GIT_PROGRESS   TRUE
	OPTIONS
		"SDL_TEST_LIBRARY OFF"
		"SDL_SHARED ON"
		"SDL_STATIC OFF"
)
FetchContent_MakeAvailable(sdl2)

FetchContent_Declare( stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG        master
	GIT_SHALLOW    TRUE
	GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(stb)

FetchContent_Declare( cgltf
	GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
	GIT_TAG        v1.14
	GIT_SHALLOW    TRUE
	GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(cgltf)

if(ANDROID)
	add_library(sk_renderer_test SHARED)
	set_target_properties(sk_renderer_test PROPERTIES OUTPUT_NAME "main")
else()
	add_executable(sk_renderer_test)
endif()
target_link_libraries     (sk_renderer_test PRIVATE SDL2::SDL2 sk_renderer m)
target_include_directories(sk_renderer_test PRIVATE ${stb_SOURCE_DIR} ${cgltf_SOURCE_DIR})
target_sources            (sk_renderer_test PRIVATE
	example/main.c
	example/app.c
	example/scene_util.c
	example/scene_meshes.c
	example/scene_reaction_diffusion.c
	example/scene_orbital_particles.c
	example/scene_impostor.c
	example/scene_array_texture.c
	example/scene_3d_texture.c
	example/scene_cubemap.c
	example/scene_gltf.c
	example/scene_shadows.c
	example/scene_erosion.c
	example/bloom.c
)

if(ANDROID)
	set(ASSET_PATH ${CMAKE_BINARY_DIR}/assets)
else()
	set(ASSET_PATH ${CMAKE_BINARY_DIR})
endif()
skshaderc_compile_assets(sk_renderer_test ${ASSET_PATH}/shaders ""
    example/assets/shaders/bloom_composite.hlsl
    example/assets/shaders/bloom_downsample.hlsl
    example/assets/shaders/bloom_upsample.hlsl
    example/assets/shaders/compute_flow.hlsl
    example/assets/shaders/compute_test.hlsl
    example/assets/shaders/compute_water.hlsl
    example/assets/shaders/cubemap_mipgen.hlsl
    example/assets/shaders/cubemap_reflection.hlsl
    example/assets/shaders/cubemap_skybox.hlsl
    example/assets/shaders/equirect_to_cubemap.hlsl
    example/assets/shaders/mipgen_alpha_weighted_render.hlsl
    example/assets/shaders/orbital_particles_compute.hlsl
    example/assets/shaders/orbital_particles.hlsl
    example/assets/shaders/pbr.hlsl
    example/assets/shaders/shadow_caster.hlsl
    example/assets/shaders/shadow_receiver.hlsl
    example/assets/shaders/stereo_display.hlsl
    example/assets/shaders/terrain_erosion.hlsl
    example/assets/shaders/test.hlsl
    example/assets/shaders/texture3d.hlsl
)

if(NOT ANDROID)
	target_link_libraries(sk_renderer_test PRIVATE pthread)
endif()

###############################################################################
# Android build configuration
###############################################################################

if(ANDROID)
	set(PROJECT_NAME               "sk_renderer_test")  # Tell AndroidBuild which target to package
	set(ANDROID_PACKAGE_NAME       "net.stereokit.renderer_test")
	set(ANDROID_MIN_SDK_VERSION    29)
	set(ANDROID_TARGET_SDK_VERSION 36)
	set(ANDROID_BUILD_TOOLS_VERSION "36.1.0")
	include(android/AndroidBuild.cmake)
endif()

###############################################################################
# "run" command
###############################################################################

if(ANDROID)
	# Android run target is defined in AndroidBuild.cmake
else()
	add_custom_target(run DEPENDS sk_renderer_test COMMAND $<TARGET_FILE:sk_renderer_test>)
endif()

#######################################
## App Code and Assets               ##
#######################################

# Copy Assets to the folder with the executable, so the primary project targets
# can just run without helpers too.
add_custom_command(TARGET sk_renderer_test POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/example/assets ${ASSET_PATH}
	COMMENT "Copying App Assets")