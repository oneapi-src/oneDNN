if(cblas_cmake_included)
    return()
endif()

set(cblas_cmake_included true)
if (NOT DNNL_CPU_EXTERNAL_GEMM STREQUAL "CBLAS")
    # Other options are:
    #   NONE - never want cblas (src/cpu/gemm has a reasonable ref gemm)
    #   MKL  - if found, MKL will supply CBLAS, if not use mkl-dnn ref gemm
    return()
endif()

include("cmake/options.cmake")
include("cmake/SDL.cmake")

set(CBLAS_FIND_REQUIRED TRUE)
find_package(CBLAS)

if(CBLAS_FOUND)
    message(STATUS "BLAS_FOUND          	${BLAS_FOUND}")
    message(STATUS "BLAS_LINKER_FLAGS   	${BLAS_LINKER_FLAGS}")
    message(STATUS "BLAS_LIBRARIES      	${BLAS_LIBRARIES}")
    message(STATUS "BLAS95_FOUND        	${BLAS95_FOUND}")
    message(STATUS "BLAS95_LIBRARIES    	${BLAS95_LIBRARIES}")
    message(STATUS "CBLAS_FOUND         	${CBLAS_FOUND}")
    message(STATUS "CBLAS_INCLUDE_DIRS  	${CBLAS_INCLUDE_DIRS}")
    message(STATUS "CBLAS_LIBRARY_DIRS  	${CBLAS_LIBRARY_DIRS}")
    message(STATUS "CBLAS_LIBRARIES  		${CBLAS_LIBRARIES}")
    message(STATUS "CBLAS_INCLUDE_DIRS_DEP  ${CBLAS_INCLUDE_DIRS_DEP}")
    message(STATUS "CBLAS_LIBRARY_DIRS_DEP  ${CBLAS_LIBRARY_DIRS_DEP}")
    message(STATUS "CBLAS_LIBRARIES_DEP  	${CBLAS_LIBRARIES_DEP}")
    include_directories(${CBLAS_INCLUDE_DIRS_DEP})
    # adjust as things get more complicated...
    if(${CBLAS_LIBRARY_DIRS_DEP})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_LIBRARY_PATH_FLAG}${CBLAS_LIBRARY_DIRS_DEP}")
    endif()
    list(APPEND EXTRA_SHARED_LIBS ${CBLAS_LINKER_FLAGS} ${CBLAS_LIBRARIES_DEP})
    message(STATUS "cblas.cmake CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")
    message(STATUS "cblas.cmake EXTRA_SHARED_LIBS      = ${EXTRA_SHARED_LIBS}")
    #add_definitions(-DUSE_CBLAS) 
    # New: DNNL_USE_CBLAS is provided by dnnl_config.h.in,
    #      and mapped to USE_CBLAS in cpu_target.h
endif()
# vim: sw=4 ts=4 et	    
