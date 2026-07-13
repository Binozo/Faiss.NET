# Injected into MKL-variant Faiss builds via:
#   -DCMAKE_PROJECT_faiss_INCLUDE=<repo>/scripts/faiss-mkl-output-names.cmake
#
# Renames library output names so the MKL-linked variant can coexist with the
# default (OpenBLAS) variant in the same runtimes/<rid>/native directory:
#   faiss        -> (lib)faiss.mkl        faiss_avx2   -> (lib)faiss_avx2.mkl
#   faiss_c      -> (lib)faiss_c.mkl      faiss_c_avx2 -> (lib)faiss_c.mkl
#
# The C API targets are normalized to "faiss_c.mkl" regardless of SIMD suffix,
# because the .NET resolver probes exactly NativeLibrary.TryLoad("faiss_c.mkl").
# Core-library names keep their suffix; dependent import tables / DT_NEEDED and
# SONAME entries are correct automatically since the rename happens at link time.
#
# Runs deferred so all targets (including c_api/) exist when it executes.

cmake_minimum_required(VERSION 3.19) # cmake_language(DEFER)

function(_faissnet_mkl_output_names)
    foreach(_tgt IN ITEMS faiss faiss_avx2 faiss_avx512 faiss_avx512_spr faiss_sve)
        if(TARGET ${_tgt})
            set_target_properties(${_tgt} PROPERTIES OUTPUT_NAME "${_tgt}.mkl")
        endif()
    endforeach()
    foreach(_tgt IN ITEMS faiss_c faiss_c_avx2 faiss_c_avx512 faiss_c_avx512_spr faiss_c_sve)
        if(TARGET ${_tgt})
            set_target_properties(${_tgt} PROPERTIES OUTPUT_NAME "faiss_c.mkl")
        endif()
    endforeach()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL _faissnet_mkl_output_names)
