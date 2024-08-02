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

set(DIRS "generator;selector")

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

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_intel_jit_gemm)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

target_include_directories(${OBJ_LIB} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../ngen)

# Workaround for LTO bug in GCC 10, 11, 12 (possibly other versions)
if(CMAKE_COMPILER_IS_GNUCC)
    set_source_files_properties(generator/pieces/loop_sequencer.cpp PROPERTIES COMPILE_FLAGS -fno-lto)
    set_source_files_properties(generator/generator.cpp PROPERTIES COMPILE_FLAGS -fno-lto)
endif()
