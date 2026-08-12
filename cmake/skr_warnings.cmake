# Warning level for first-party targets. PRIVATE so consumers of the headers
# don't inherit it, and not -Werror: vendored dependencies and platform SDK
# headers compile in the same tree. The suppressions below are categories this
# codebase hits by design rather than by accident.
#
# This lives in a module rather than the root CMakeLists because skshaderc also
# configures as a standalone project, where that root never runs.
function(skr_target_warnings target)
	if(MSVC)
		target_compile_options(${target} PRIVATE
			/W4
			/wd4100 # unreferenced formal parameter: callback signatures are fixed
			/wd4127 # conditional expression is constant
			/wd4201 # nameless struct/union, used throughout Vulkan and OpenXR headers
			/wd4204 # non-constant aggregate initializer
			/wd4221)# aggregate initialized from the address of an automatic
	else()
		target_compile_options(${target} PRIVATE
			-Wall -Wextra
			-Wvla                          # MSVC has no VLAs at all, so catch them here
			-Wno-unused-parameter          # callback signatures are fixed
			-Wno-missing-field-initializers # designated init of Vulkan/OpenXR structs
			# A non-static function with no prototype is invisible to
			# -fvisibility=hidden and to __declspec(dllexport), so an export
			# that drifts out of a header goes quiet until link time.
			$<$<COMPILE_LANGUAGE:C>:-Wmissing-prototypes>
			$<$<COMPILE_LANGUAGE:C>:-Wstrict-prototypes>
			$<$<COMPILE_LANGUAGE:C>:-Wshadow>)
	endif()
endfunction()
