# Injected into generic (non-SIMD) Faiss builds via:
#   -DCMAKE_PROJECT_faiss_INCLUDE=<repo>/scripts/faiss-generic-output-names.cmake
#
# Renames library output names so the generic variant can coexist with the
# avx2/MKL variants in the same runtimes/<rid>/native directory:
#   faiss   -> faiss.generic      faiss_c -> faiss_c.generic
#
# Only used for FAISS_OPT_LEVEL=generic builds, where exactly these two
# targets exist. Dependent import tables are correct automatically since the
# rename happens at link time.
#
# Runs deferred so all targets (including c_api/) exist when it executes.

cmake_minimum_required(VERSION 3.19) # cmake_language(DEFER)

function(_faissnet_generic_output_names)
    if(TARGET faiss)
        set_target_properties(faiss PROPERTIES OUTPUT_NAME "faiss.generic")
    endif()
    if(TARGET faiss_c)
        set_target_properties(faiss_c PROPERTIES OUTPUT_NAME "faiss_c.generic")
    endif()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL _faissnet_generic_output_names)
