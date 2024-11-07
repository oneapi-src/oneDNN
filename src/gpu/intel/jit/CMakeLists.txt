#===============================================================================
# Copyright 2020-2024 Intel Corporation
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

set(CMAKE_CXX_CLANG_TIDY "")

set(DIRS "codegen;conv;ir;ngen;pass;pooling;reorder;utils;v2")

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


if(DNNL_EXPERIMENTAL)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/v2/conv/planner/planner_main.cpp")
else()
    file(GLOB planner_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/conv/planner/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/conv/planner/*.cpp
        )
    foreach(s ${planner_sources})
        list(REMOVE_ITEM SOURCES "${s}")
    endforeach()
endif()

add_subdirectory(gemm)

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_intel_jit)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

if(DNNL_EXPERIMENTAL)
    add_subdirectory(v2/conv/planner)
endif()

# Workaround for unordered_map related issues with DPC++ on Windows
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)
if(WIN32 AND UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "RELWITHMDD" AND CMAKE_BASE_NAME MATCHES "(icx|icpx)")
    file(GLOB_RECURSE FILES_LIMIT_INLINE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    set_source_files_properties(${FILES_LIMIT_INLINE} PROPERTIES COMPILE_FLAGS "/Ob1")
endif()
