cmake_minimum_required(VERSION 3.10)
project(sk_ktx2 VERSION 0.1.0 DESCRIPTION "KTX2 / Basis Universal transcoder in C11")

# Standing alone, an unset CMAKE_BUILD_TYPE means no optimization at all, which
# silently makes this library about 3x slower and is not what anyone configuring
# it directly wants. As a subdirectory the parent's choice is left alone.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
	message(STATUS "sk_ktx2: no CMAKE_BUILD_TYPE set, defaulting to Release")
endif()

# UASTC adds the ASTC and BC7 repacks, and Zstd supercompressed UASTC needs an
# inflate callback from the host - see ktx2_context_t. Nothing is vendored
# either way. ETC1S needs neither, which is why this is a separate switch.
option(SK_KTX2_UASTC "UASTC LDR 4x4 support" ON)
# NOT a debug feature: this is the last-resort path for hardware that samples no
# compressed format at all, which is what StereoKit's
# texture_preferred_compressed_format falls back to. Off is a size-conscious
# opt-out, and the test target always builds it on.
option(SK_KTX2_DECODE_UNCOMPRESSED "R8/RG8/RGBA32 decode fallback" ON)
option(SK_KTX2_BUILD_TESTS "Build the sk_ktx2 conformance runner" OFF)

add_library(sk_ktx2)
set_target_properties(sk_ktx2 PROPERTIES
	C_STANDARD          11
	C_STANDARD_REQUIRED ON
	POSITION_INDEPENDENT_CODE ON)

target_include_directories(sk_ktx2 PUBLIC include)
target_sources(sk_ktx2 PRIVATE
	include/sk_ktx2.h
	ktx2_internal.h
	ktx2_container.c
	ktx2_context.c
	ktx2_format.c
	ktx2_huffman.c
	ktx2_etc1s.c
	ktx2_etc1s_repack.c
	ktx2_etc1s_gray.c
	ktx2_bc1.c
	ktx2_uastc.c
	ktx2_uastc_tables.c
	ktx2_uastc_astc.c
	ktx2_uastc_bc7_tables.c
	ktx2_uastc_bc7.c
	ktx2_transcode.c
)

if(SK_KTX2_UASTC)
	target_compile_definitions(sk_ktx2 PUBLIC SK_KTX2_UASTC)
endif()
if(SK_KTX2_DECODE_UNCOMPRESSED)
	target_compile_definitions(sk_ktx2 PUBLIC SK_KTX2_DECODE_UNCOMPRESSED)
endif()

if(NOT MSVC)
	target_compile_options(sk_ktx2 PRIVATE -fvisibility=hidden -Wall -Wextra)
endif()

if(SK_KTX2_BUILD_TESTS)
	# enable_testing() has to happen in the project's top-level list file or
	# ctest never descends into tests/. Harmless when we're a subdirectory of
	# sk_renderer, which enables testing itself.
	enable_testing()
	add_subdirectory(tests)
endif()
