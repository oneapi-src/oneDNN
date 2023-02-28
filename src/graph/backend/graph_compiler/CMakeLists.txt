#===============================================================================
# Copyright 2021-2023 Intel Corporation
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

if(NOT ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND)
    message(STATUS "Graph compiler backend is disabled.")
    return()
endif()

SET(SC_LLVM_VERSION "OFF" CACHE STRING "version of LLVM")
SET(SC_LLVM_LIB_NAME "OFF" CACHE STRING "the lib name of LLVM for linker")
SET(SC_LLVM_INCLUDE_PATH "OFF" CACHE STRING "the header include path of LLVM")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/core/cmake")
include("core/cmake/find_llvm.cmake")

set(SC_LLVM_CONFIG ${ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_LLVM_CONFIG})

if(DNNL_LIBRARY_TYPE STREQUAL "SHARED")
    add_definitions(-DSC_DLL)
    set(SC_LIBRARY_TYPE "SHARED")
else()
    set(SC_LIBRARY_TYPE "STATIC")
endif()

list (FIND ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT "llvm" _index)
if (${_index} GREATER -1)
    set(SC_LLVM_ENABLED ON)
else()
    set(SC_LLVM_ENABLED OFF)
endif()

list (FIND ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT "c" _index)
if (NOT WIN32 AND NOT APPLE AND ${_index} GREATER -1)
    set(SC_CFAKE_ENABLED ON)
else()
    set(SC_CFAKE_ENABLED OFF)
endif()

list (FIND ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT "builtin" _index)
if (${_index} GREATER -1)
    set(SC_BUILTIN_ENABLED ON)
else()
    set(SC_BUILTIN_ENABLED OFF)
endif()

if(NOT SC_LLVM_ENABLED AND NOT SC_BUILTIN_ENABLED AND NOT SC_CFAKE_ENABLED)
    message(FATAL_ERROR "ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT should include at least one of llvm;c;builtin")
endif()

if(SC_LLVM_ENABLED)
    find_llvm()

    if(${SC_LLVM_VERSION} STREQUAL OFF OR ${SC_LLVM_VERSION} LESS 8)
        message(FATAL_ERROR "Expecting LLVM version >= 8, got ${SC_LLVM_VERSION}. "
        "Consider turning off graph compiler backend, or use a higher LLVM version.")
    else()
        if("${SC_LLVM_INCLUDE_PATH}" STREQUAL OFF)
            message(FATAL_ERROR "LLVM_INCLUDE_PATH is not successfully set. "
            "Consider turning off graph compiler backend, or recheck LLVM Config.")
        else()
            message(STATUS "Found LLVM_VERSION=" ${SC_LLVM_VERSION})
        endif()
    endif()
else()
    message(STATUS "LLVM JIT is OFF")
endif()

if(DNNL_CPU_RUNTIME MATCHES "(DPCPP|THREADPOOL)$")
    message(FATAL_ERROR "Graph compiler backend does not support the chosen CPU runtime. "
    "Consider changing DNNL_CPU_RUNTIME to OMP or turning off graph compiler backend.")
endif()

if(NOT DNNL_GPU_RUNTIME MATCHES "NONE")
    message(FATAL_ERROR "Graph compiler backend does not support GPU runtime. "
    "Consider changing DNNL_CPU_RUNTIME to NONE or turning off graph compiler backend.")
endif()

message(STATUS "Graph compiler backend is enabled.")
set(SC_CPU_RUNTIME ${DNNL_CPU_RUNTIME})

if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4200")
    # allow usage of "deprecated" functions
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4996")
    # inherits via dominance
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4250")
    # conversion from 'size_t' to 'uint16_t'
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4267")
    # function assumed not to throw an exception but does
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4297")
    #  format string '%lu' requires an argument of type 'unsigned long'
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4477")
    # not enough arguments for function-like macro
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4003")
    # 
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4624")
    # 'elem_type': unreferenced local variable
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4101")
    # unary minus operator applied to unsigned type
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4146")
    # destructor never returns, potential memory leak
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4722")
    # needs to have dll-interface to be used by clients of struct
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4251")
endif()

append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/src/)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

set(OBJ_LIB dnnl_graph_backend_compiler)
add_library(${OBJ_LIB} OBJECT ${SOURCES})


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)