#*******************************************************************************
# Copyright (c), 2022, KNS Group LLC (YADRO).
# All Rights Reserved.
#
# This software contains the intellectual property of YADRO
# or is licensed to YADRO from third parties.  Use of this
# software and the intellectual property contained therein is expressly
# limited to the terms and conditions of the License Agreement under which
# it is provided by YADRO.
#*******************************************************************************

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
)

message(STATUS DNNL_RISCV_USE_RVV: ${DNNL_RISCV_USE_RVV})
if(NOT DNNL_RISCV_USE_RVV)
    # Do not compile RISC-V implementations optimized with RVV intrinsics
    list(FILTER SOURCES EXCLUDE REGEX "rvv_*")

    # Registering a library w/o any sources leads to a CMake error.
    # If required, create an empty source file to prevent it.
    list(LENGTH SOURCES NUM_OF_SOURCES)
    if(NUM_OF_SOURCES EQUAL 0)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp "")
        set(SOURCES ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp)
    endif()
endif()

if(NOT DNNL_CPU_RUNTIME STREQUAL "SEQ")
    message(FATAL_ERROR "Only sequential runtime is now supported for a RISC-V CPU")
endif()

# Build sources into an object library
set(OBJ_LIB ${DNNL_LIBRARY_NAME}_cpu_riscv)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

# Add compiled object files to oneDNN dependencies
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
             $<TARGET_OBJECTS:${OBJ_LIB}>)
