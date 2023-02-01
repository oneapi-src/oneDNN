#===============================================================================
# Copyright 2023 KNS Group LLC (YADRO)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
)

if(NOT DNNL_RISCV_USE_RVV_INTRINSICS)
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
