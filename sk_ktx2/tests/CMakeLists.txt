# The library ships no decompressor - Zstd supercompressed UASTC needs one from
# the host. The tests supply the system libzstd; without it those paths skip.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
	pkg_check_modules(ZSTD QUIET libzstd)
endif()
if(ZSTD_FOUND)
	message(STATUS "sk_ktx2: libzstd ${ZSTD_VERSION}, Zstd supercompression tests enabled")
else()
	message(STATUS "sk_ktx2: no libzstd, Zstd supercompression tests will skip")
endif()

function(sk_ktx2_host_zstd target)
	if(ZSTD_FOUND)
		target_compile_definitions(${target} PRIVATE SK_KTX2_TEST_ZSTD)
		target_include_directories(${target} PRIVATE ${ZSTD_INCLUDE_DIRS})
		target_link_libraries(${target} PRIVATE ${ZSTD_LIBRARIES})
		target_link_directories(${target} PRIVATE ${ZSTD_LIBRARY_DIRS})
	endif()
endfunction()

# KhronosGroup/KTX-Software-CTS (Apache-2.0). Neither vendored nor a submodule:
# it's ~400 MB and only matters while working on this library, so it has no
# business in a general sk_renderer checkout. Clone it as a sister folder of the
# repo and the conformance tests appear; without it they skip. The fuzzer takes
# its seeds from here too, so it is resolved before any target is declared.
#
#   git clone --depth 1 https://github.com/KhronosGroup/KTX-Software-CTS.git
#
# SK_KTX2_CTS_DIR overrides the location for a checkout kept elsewhere.
get_filename_component(_cts_sister "${CMAKE_CURRENT_SOURCE_DIR}/../../../KTX-Software-CTS" ABSOLUTE)
set(SK_KTX2_CTS_DIR "${_cts_sister}" CACHE PATH
	"KTX-Software-CTS checkout (default: sister folder of the sk_renderer repo)")

# Headless conformance runner. No renderer, no GPU, no window - it must stay
# runnable in CI, which is why sk_ktx2 has no sk_renderer dependency.
add_executable(ktx2_cts cts_runner.c)
set_target_properties(ktx2_cts PROPERTIES
	C_STANDARD          11
	C_STANDARD_REQUIRED ON)
target_link_libraries(ktx2_cts PRIVATE sk_ktx2)
sk_ktx2_host_zstd(ktx2_cts)

if(NOT MSVC)
	target_compile_options(ktx2_cts PRIVATE -Wall -Wextra)
endif()

# Synthesizes its own KTX2 files, so it needs no corpus and always registers -
# which also keeps ctest from reporting an empty suite when the CTS is absent.
add_executable(ktx2_unit unit_tests.c)
set_target_properties(ktx2_unit PROPERTIES
	C_STANDARD          11
	C_STANDARD_REQUIRED ON)
target_link_libraries(ktx2_unit PRIVATE sk_ktx2)
sk_ktx2_host_zstd(ktx2_unit)
if(NOT MSVC)
	target_compile_options(ktx2_unit PRIVATE -Wall -Wextra)
endif()
add_test(NAME ktx2_unit COMMAND ktx2_unit)

# Mutation fuzzing of everything that parses untrusted bytes, seeded from the CTS
# when it is present. The registered run is deliberately short - a few seconds,
# so it stays in the routine suite; pass a larger count for a real session:
#
#   tests/ktx2_fuzz 5000000 <cts dir>
#
# SK_KTX2_FUZZ_LIBFUZZER builds it as a coverage-guided libFuzzer target instead.
# Only meaningful under ASan/UBSan - see the note in fuzz_test.c.
option(SK_KTX2_FUZZ_LIBFUZZER "Build the fuzz target for libFuzzer instead of standalone" OFF)

add_executable(ktx2_fuzz fuzz_test.c)
set_target_properties(ktx2_fuzz PROPERTIES
	C_STANDARD          11
	C_STANDARD_REQUIRED ON)
target_link_libraries(ktx2_fuzz PRIVATE sk_ktx2)
sk_ktx2_host_zstd(ktx2_fuzz)
if(NOT MSVC)
	target_compile_options(ktx2_fuzz PRIVATE -Wall -Wextra)
endif()
if(SK_KTX2_FUZZ_LIBFUZZER)
	target_compile_definitions(ktx2_fuzz PRIVATE SK_KTX2_FUZZ_LIBFUZZER)
	target_compile_options    (ktx2_fuzz PRIVATE -fsanitize=fuzzer)
	target_link_options       (ktx2_fuzz PRIVATE -fsanitize=fuzzer)
else()
	add_test(NAME ktx2_fuzz COMMAND ktx2_fuzz 5000 "${SK_KTX2_CTS_DIR}")
endif()

# Files built to be valid down to the entropy coder and hostile inside it, which
# is the part mutation cannot reach. Corpus-free: it writes its own bitstreams.
add_executable(ktx2_craft craft_test.c)
set_target_properties(ktx2_craft PROPERTIES
	C_STANDARD          11
	C_STANDARD_REQUIRED ON)
target_link_libraries(ktx2_craft PRIVATE sk_ktx2)
sk_ktx2_host_zstd(ktx2_craft)
if(NOT MSVC)
	target_compile_options(ktx2_craft PRIVATE -Wall -Wextra)
endif()
add_test(NAME ktx2_craft COMMAND ktx2_craft)

# UASTC block decode against the specification's 64 published test vectors.
# Corpus-free like ktx2_unit, and it reaches into ktx2_internal.h because the
# block decoder is deliberately below the public surface.
add_executable(ktx2_uastc uastc_test.c)
set_target_properties(ktx2_uastc PROPERTIES
	C_STANDARD          11
	C_STANDARD_REQUIRED ON)
target_link_libraries(ktx2_uastc PRIVATE sk_ktx2)
target_include_directories(ktx2_uastc PRIVATE ..)
if(NOT MSVC)
	target_compile_options(ktx2_uastc PRIVATE -Wall -Wextra)
endif()
add_test(NAME ktx2_uastc COMMAND ktx2_uastc)

# Bit-exact comparison against the KTX-Software transcoder. Needs the corpus.
add_executable(ktx2_golden golden_test.c)
set_target_properties(ktx2_golden PROPERTIES
	C_STANDARD          11
	C_STANDARD_REQUIRED ON)
target_link_libraries(ktx2_golden PRIVATE sk_ktx2)
sk_ktx2_host_zstd(ktx2_golden)
# astc_reference.h checks the repack below the public surface.
target_include_directories(ktx2_golden PRIVATE ..)
if(NOT MSVC)
	target_compile_options(ktx2_golden PRIVATE -Wall -Wextra)
endif()

if(NOT EXISTS "${SK_KTX2_CTS_DIR}/clitests/input/validate")
	message(STATUS "sk_ktx2: no CTS corpus at ${SK_KTX2_CTS_DIR}, conformance tests skipped")
	return()
endif()
message(STATUS "sk_ktx2: CTS corpus at ${SK_KTX2_CTS_DIR}")

set(_cts_in "${SK_KTX2_CTS_DIR}/clitests/input")

# Every error_/fatal_ input must be rejected by open, the glTF check, or plan.
# KTXmapRange is the one deliberate exception: the key may only appear on float
# or linear UNORM/SNORM formats, so it is invalid here, but it is metadata we
# never read and libktx accepts the file too - its own golden output says
# "Non-conformant texture file accepted by libktx". Refusing a texture that
# loads in every other engine is the worse failure.
add_test(NAME ktx2_validate COMMAND ktx2_cts --quiet --expect cts
	--allow error_KTXmapRange "${_cts_in}/validate")

# The KHR_texture_basisu family specifically - our actual scope. Every one of
# these must be rejected, and the interesting ones land on the glTF check rather
# than on open, which is what keeps the diagnostic useful.
add_test(NAME ktx2_gltf_basisu COMMAND ktx2_cts --quiet --expect reject
	"${_cts_in}/validate/3101" "${_cts_in}/validate/3102" "${_cts_in}/validate/3103"
	"${_cts_in}/validate/3104" "${_cts_in}/validate/6301" "${_cts_in}/validate/6302"
	"${_cts_in}/validate/6303" "${_cts_in}/validate/6304" "${_cts_in}/validate/7201"
	"${_cts_in}/validate/7202")

# Real content must survive all three stages.
add_test(NAME ktx2_accept COMMAND ktx2_cts --quiet --expect accept
	"${_cts_in}/ktx2_sample/color_grid_basis.ktx2"
	"${_cts_in}/ktx2_sample/alpha_simple_basis.ktx2"
	"${_cts_in}/ktx2_sample/ktx_document_basis.ktx2"
	"${_cts_in}/ktx2_sample/camera_camera_BaseColor_basis.ktx2"
	"${_cts_in}/ktx2_sample/color_grid_uastc.ktx2"
	"${_cts_in}/ktx2_sample/color_grid_uastc_zstd.ktx2"
	"${_cts_in}/ktx2_sample/cimg5293_uastc.ktx2")

# The two-slice distinction, on the only real files that carry it. Both are
# two-slice ETC1S; only the DFD channel IDs say one is RG and the other RGBA.
# They need --no-gltf because toktx wrote KTXswizzle "ra01" on the RG one.
add_test(NAME ktx2_two_slice COMMAND ktx2_cts --no-gltf --caps etc2 --expect accept
	"${_cts_in}/ktx2_sample/rg_reference_basis.ktx2"
	"${_cts_in}/ktx2_sample/luminance_alpha_reference_basis.ktx2"
	"${_cts_in}/ktx2_sample/r_reference_basis.ktx2")

add_test(NAME ktx2_golden COMMAND ktx2_golden "${SK_KTX2_CTS_DIR}")

# Robustness sweep: the whole corpus, every capability combination reachable,
# with no expectation beyond "does not crash, hang, or read out of bounds".
# Worth running under ASan - this parses untrusted bytes out of downloaded glTF.
add_test(NAME ktx2_corpus COMMAND ktx2_cts --quiet --no-gltf --caps none "${SK_KTX2_CTS_DIR}")

# The same sweep with the entropy decoder actually running, so the malformed
# BasisLZ inputs drive the Huffman and slice decoders rather than stopping at
# the container. This is the closest thing to a fuzz run the corpus provides.
add_test(NAME ktx2_corpus_transcode COMMAND ktx2_cts --quiet --no-gltf --transcode --caps etc2 "${SK_KTX2_CTS_DIR}")

# The same again with ASTC available, so the UASTC sources take the block repack
# rather than the uncompressed decode. Malformed UASTC in the corpus drives the
# ASTC writer this way, which the goldens never do.
add_test(NAME ktx2_corpus_astc COMMAND ktx2_cts --quiet --no-gltf --transcode --caps astc "${SK_KTX2_CTS_DIR}")

# And once more with only BC, which sends ETC1S down the BC1/BC3 fits and UASTC
# down the BC7 repack - the one path with p-bit search in it.
add_test(NAME ktx2_corpus_bc COMMAND ktx2_cts --quiet --no-gltf --transcode --caps bc "${SK_KTX2_CTS_DIR}")
