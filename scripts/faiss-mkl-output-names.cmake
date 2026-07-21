# Injected into MKL-variant Faiss builds via:
#   -DCMAKE_PROJECT_faiss_INCLUDE=<repo>/scripts/faiss-mkl-output-names.cmake
#
# Renames library output names so the MKL-linked variant can coexist with the
# default (OpenBLAS) variant in the same runtimes/<rid>/native directory:
#   faiss        -> (lib)faiss.mkl        faiss_avx2   -> (lib)faiss_avx2.mkl
#   faiss_c      -> (lib)faiss_c.mkl      faiss_c_avx2 -> (lib)faiss_c.mkl
#
# FAISS_OPT_LEVEL=dd (fleet default) builds only faiss + faiss_c, so the
# effective rename is faiss -> faiss.mkl and faiss_c -> faiss_c.mkl. The
# SIMD-suffixed targets exist but are EXCLUDE_FROM_ALL and never built.
#
# The C API targets are normalized to "faiss_c.mkl" regardless of SIMD suffix,
# because the .NET resolver probes exactly NativeLibrary.TryLoad("faiss_c.mkl").
# Core-library names keep their suffix; dependent import tables / DT_NEEDED and
# SONAME entries are correct automatically since the rename happens at link time.
#
# Runs deferred so all targets (including c_api/) exist when it executes.

cmake_minimum_required(VERSION 3.19) # cmake_language(DEFER)

function(_faissnet_mkl_output_names)
    # Core libraries have distinct default names; suffixing them all is safe.
    foreach(_tgt IN ITEMS faiss faiss_avx2 faiss_avx512 faiss_avx512_spr faiss_sve)
        if(TARGET ${_tgt})
            set_target_properties(${_tgt} PROPERTIES OUTPUT_NAME "${_tgt}.mkl")
        endif()
    endforeach()

    # Only ONE C API target may claim the normalized name: every faiss_c*
    # target is defined (inactive ones are merely EXCLUDE_FROM_ALL), and Ninja
    # emits build edges for all of them — two targets with the same OUTPUT_NAME
    # fail with "multiple rules generate faiss_c.mkl.dll".
    if(FAISS_OPT_LEVEL STREQUAL "avx2")
        set(_c_target faiss_c_avx2)
    elseif(FAISS_OPT_LEVEL STREQUAL "avx512")
        set(_c_target faiss_c_avx512)
    elseif(FAISS_OPT_LEVEL STREQUAL "avx512_spr")
        set(_c_target faiss_c_avx512_spr)
    elseif(FAISS_OPT_LEVEL STREQUAL "sve")
        set(_c_target faiss_c_sve)
    else() # generic, dd
        set(_c_target faiss_c)
    endif()

    if(TARGET ${_c_target})
        set_target_properties(${_c_target} PROPERTIES OUTPUT_NAME "faiss_c.mkl")
    endif()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL _faissnet_mkl_output_names)
