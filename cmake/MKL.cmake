##############################################################################
# Locate Intel(R) MKL installation using MKLROOT or look in
# ${CMAKE_CURRENT_SOURCE_DIR}/external
#
function(detect_mkl LIBNAME)
    if(HAVE_MKL)
        return()
    endif()

    find_path(MKLINC mkl_cblas.h
        PATHS ${MKLROOT}/include $ENV{MKLROOT}/include)
    if(NOT MKLINC)
        file(GLOB_RECURSE MKLINC
                ${CMAKE_CURRENT_SOURCE_DIR}/external/*/mkl_cblas.h)
        if(MKLINC)
            get_filename_component(MKLINC ${MKLINC} PATH)
            set(MKLINC ${MKLINC} PARENT_SCOPE)
        endif()
    endif()

    get_filename_component(__mklinc_root "${MKLINC}" PATH)
    find_library(MKLLIB NAMES ${LIBNAME}
        PATHS   ${MKLROOT}/lib ${MKLROOT}/lib/intel64
                $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64
                ${__mklinc_root}/lib ${__mklinc_root}/lib/intel64)
    if(MKLINC AND MKLLIB)
        set(HAVE_MKL TRUE PARENT_SCOPE)
        get_filename_component(MKLLIBPATH "${MKLLIB}" PATH)
        string(FIND "${MKLLIBPATH}" ${CMAKE_CURRENT_SOURCE_DIR}/external __idx)
        if(${__idx} EQUAL 0)
            install(PROGRAMS ${MKLLIB} ${MKLLIBPATH}/libiomp5.so
                    DESTINATION lib)
        endif()
    endif()
endfunction()

detect_mkl("libmklml_intel.so")
detect_mkl("libmkl_rt.so")

set(FAIL_WITHOUT_MKL)

if(HAVE_MKL)
    add_definitions(-DUSE_MKL -DUSE_CBLAS)
    include_directories(AFTER ${MKLINC})
    list(APPEND mkldnn_LINKER_LIBS ${MKLLIB})
    message(STATUS "Intel(R) MKL found: include ${MKLINC}, lib ${MKLLIB}")
else()
    if(DEFINED ENV{FAIL_WITHOUT_MKL} OR DEFINED FAIL_WITHOUT_MKL)
        set(SEVERITY "FATAL_ERROR")
    else()
        set(SEVERITY "WARNING")
    endif()
    message(${SEVERITY} "Intel(R) MKL not found. Some performance features may not be "
        "available. Please run scripts/prepare_mkl.sh to download a minimal "
        "set of libraries or get a full version from "
        "https://software.intel.com/en-us/intel-mkl")
endif()
