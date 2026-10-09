# SPDX-License-Identifier: MIT
# Encoder timing CLI, desktop and Android. Needs sk_texenc's debug API.

if(NOT SK_TEXENC_DEBUG)
	message(STATUS "texenc_bench: skipped, needs -DSK_TEXENC_DEBUG=ON")
	return()
endif()

add_executable(texenc_bench
	main.c
	${CMAKE_CURRENT_SOURCE_DIR}/../astc_io.c
)

target_include_directories(texenc_bench PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/..
	${stb_SOURCE_DIR}
)

target_link_libraries(texenc_bench PRIVATE sk_renderer sk_texenc)
if(UNIX)
	target_link_libraries(texenc_bench PRIVATE m)
endif()

set_target_properties(texenc_bench PROPERTIES
	C_STANDARD 11
	C_STANDARD_REQUIRED ON
)
