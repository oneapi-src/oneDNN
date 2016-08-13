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
else()
    message(WARNING "MKL not found; some performance features may be unavailable.")
endif()
