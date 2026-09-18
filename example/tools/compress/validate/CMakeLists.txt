# SPDX-License-Identifier: MIT
# ASTC encoder validation CLI — desktop only.

if(ANDROID)
	return()
endif()

add_executable(astc_validate
	main.c
	metrics.c
	${CMAKE_CURRENT_SOURCE_DIR}/../astc_io.c
)

target_include_directories(astc_validate PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/..
	${stb_SOURCE_DIR}
)

if(UNIX)
	target_link_libraries(astc_validate PRIVATE m)
endif()

# Match the project's C standard so we don't surprise with compiler quirks.
set_target_properties(astc_validate PROPERTIES
	C_STANDARD 11
	C_STANDARD_REQUIRED ON
)
