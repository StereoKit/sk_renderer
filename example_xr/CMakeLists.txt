cmake_minimum_required(VERSION 3.14)

# OpenXR Example for sk_renderer
# Demonstrates VR rendering using OpenXR with Vulkan backend
# Reuses scene implementations from the example/ directory

# Ensure our cmake/ folder is in path to override FindMetalTools (skip Metal on macOS)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(FetchContent)

###################################
# OpenXR Loader via FetchContent
###################################

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
	${EXAMPLE_DIR}/tools/tex_compress.c
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
	openxr_loader
)

target_include_directories(sk_renderer_xr_test PRIVATE
	${openxr_loader_SOURCE_DIR}/include
	${EXAMPLE_DIR}                          # For scene.h, app.h
	${EXAMPLE_DIR}/tools                    # For float_math.h, scene_util.h
	${CMAKE_CURRENT_SOURCE_DIR}/../sk_renderer  # For vk/skr_vulkan.h
	${stb_SOURCE_DIR}                       # For stb_image.h
	${cgltf_SOURCE_DIR}                     # For cgltf.h
	${cimgui_SOURCE_DIR}/imgui              # For imgui headers
	${cimgui_SOURCE_DIR}                    # For cimgui.h
)

# Vulkan headers are already available via FetchContent in the root CMakeLists.txt
# (vulkan_headers target and Vulkan_INCLUDE_DIR are set there)
target_include_directories(sk_renderer_xr_test PRIVATE ${Vulkan_INCLUDE_DIR})

# Math library on Unix
if(UNIX AND NOT APPLE)
	target_link_libraries(sk_renderer_xr_test PRIVATE m)
endif()

###################################
# Shader compilation
###################################

# Compile all shaders needed by the example scenes
skshaderc_compile_assets(sk_renderer_xr_test ${CMAKE_CURRENT_BINARY_DIR}/generated/shaders "" _
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
