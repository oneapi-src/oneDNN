##############################################################################
# Locate MKL installation using MKLROOT
function(detect_mkl LIBNAME)
    if(HAVE_MKL)
        return()
    endif()
    find_path(MKLINC mkl_blas.h
        PATHS ${MKLROOT}/include $ENV{MKLROOT}/include)
    find_library(MKLLIB NAMES ${LIBNAME}
        PATHS ${MKLROOT}/lib ${MKLROOT}/lib/intel64
            $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64)
    if(MKLINC AND MKLLIB)
        set(HAVE_MKL TRUE PARENT_SCOPE)
    endif()
endfunction()

detect_mkl("libmklml_intel.so")
detect_mkl("libmkl_rt.so")

if(HAVE_MKL)
    add_definitions(-DUSE_MKL -DUSE_BLAS)
    include_directories(AFTER ${MKLINC})
    list(APPEND mkldnn_LINKER_LIBS ${MKLLIB})
    message(STATUS "MKL found: include ${MKLINC}, lib ${MKLLIB}")
elseif(DEFINED ENV{FAIL_WITHOUT_MKL} OR DEFINED FAIL_WITHOUT_MKL)
    message(FATAL_ERROR "MKL not found. Please run scripts/prepare_mkl.sh to "
            "download a minimal set of libraries or get a full version from "
            "https://software.intel.com/en-us/intel-mkl")
else()
    message(WARNING "MKL not found. "
            "Some performance features may be unavailable.")
endif()
