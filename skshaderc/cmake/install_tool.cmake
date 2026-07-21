# Installs a freshly built skshaderc into the shared bin/tools folder —
# unless doing so would REPLACE a more capable tool with a lesser one.
#
# Different build configs produce different tools: an SVSL-only build
# (SKSHADERC_ENABLE_GLSLANG=OFF) has no WGSL target, while web (Emscripten)
# cross-builds depend on the installed host tool having it. Whichever native
# build ran last used to win, silently breaking the next web build.
#
# Usage: cmake -DSKSHADERC_SRC=<binary> -DSKSHADERC_DST_DIR=<dir> -P install_tool.cmake

if(NOT SKSHADERC_SRC OR NOT SKSHADERC_DST_DIR)
	message(FATAL_ERROR "install_tool.cmake needs SKSHADERC_SRC and SKSHADERC_DST_DIR")
endif()

get_filename_component(_name "${SKSHADERC_SRC}" NAME)
set(_dst "${SKSHADERC_DST_DIR}/${_name}")

# The WGSL (Tint) backend leaves this diagnostic string in the binary; its
# absence identifies a tool without the WGSL target
function(_has_wgsl out_var binary)
	set(${out_var} FALSE PARENT_SCOPE)
	if(EXISTS "${binary}")
		file(STRINGS "${binary}" _probe REGEX "Tint couldn't read" LIMIT_COUNT 1)
		if(_probe)
			set(${out_var} TRUE PARENT_SCOPE)
		endif()
	endif()
endfunction()

_has_wgsl(_src_wgsl "${SKSHADERC_SRC}")
_has_wgsl(_dst_wgsl "${_dst}")

if(_dst_wgsl AND NOT _src_wgsl)
	message(STATUS "Keeping ${_dst} (WGSL-capable); this build's skshaderc has no WGSL target and web builds need one")
else()
	file(MAKE_DIRECTORY "${SKSHADERC_DST_DIR}")
	execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${SKSHADERC_SRC}" "${SKSHADERC_DST_DIR}/")
endif()
