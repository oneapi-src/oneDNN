#===============================================================================
# Copyright 2024 Intel Corporation
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

# TODO: Extend conversion warning support to GEMM code
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    append(CMAKE_CXX_FLAGS " -Wno-implicit-int-conversion")
endif()

set(DIRS "generator;generator/pieces;selector")

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )
foreach(d ${DIRS})
    file(GLOB_RECURSE d_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/${d}/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/${d}/*.cpp
        )
    list(APPEND SOURCES "${d_sources}")
endforeach()

# There is no easy way to emulate target_compile_definitions() with the dpcpp
# host compiler arguments. As such, we cannot split compilation of the generator
# templates by using defines.
if(DPCPP_HOST_COMPILER_KIND STREQUAL "DEFAULT")
   list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/generator/generator.cpp)

    file (GLOB GENERATOR_SOURCES
        generator/generator.cpp
     )

    if (DNNL_ENABLE_PRIMITIVE_GPU_ISA STREQUAL "ALL")
        set(DNNL_GPU_ISA_LIST "GEN9;GEN11;XELP;XEHP;XEHPG;XEHPC;XE2;XE3")
    else()
        foreach(isa ${DNNL_ENABLE_PRIMITIVE_GPU_ISA})
            string(TOUPPER ${isa} ISA)
            set(DNNL_GPU_ISA_LIST "${DNNL_GPU_ISA_LIST};${ISA}")
        endforeach()
    endif()

    foreach(isa ${DNNL_GPU_ISA_LIST})
        set(GENERATOR_LIB generator${isa})
        add_library(${GENERATOR_LIB} OBJECT ${GENERATOR_SOURCES})
        target_compile_definitions(${GENERATOR_LIB} PRIVATE DNNL_GPU_ISA_${isa})
        set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
                        $<TARGET_OBJECTS:${GENERATOR_LIB}>)
    endforeach()
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_intel_jit_gemm)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

include_directories_with_host_compiler_before(${OBJ_LIB} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../ngen)

# Workaround for LTO bug in GCC 10, 11, 12 (possibly other versions)
if(CMAKE_COMPILER_IS_GNUCC)
    set_source_files_properties(generator/pieces/loop_sequencer.cpp PROPERTIES COMPILE_FLAGS -fno-lto)
    set_source_files_properties(generator/generator.cpp PROPERTIES COMPILE_FLAGS -fno-lto)
endif()
