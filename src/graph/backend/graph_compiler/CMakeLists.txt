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
    add_definitions_with_host_compiler(-DSC_DLL)
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
set_property(GLOBAL APPEND PROPERTY GRAPH_COMPILER_BUILTIN_ENABLED ${SC_BUILTIN_ENABLED})

if(NOT SC_LLVM_ENABLED AND NOT SC_BUILTIN_ENABLED AND NOT SC_CFAKE_ENABLED)
    message(FATAL_ERROR "ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT should include at least one of llvm;c;builtin")
endif()

if(SC_LLVM_ENABLED)
    find_llvm()

    if(${SC_LLVM_VERSION} STREQUAL OFF OR ${SC_LLVM_VERSION} LESS 10)
        message(FATAL_ERROR "Expecting LLVM version >= 10, got ${SC_LLVM_VERSION}. "
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

if(DNNL_CPU_RUNTIME MATCHES "(DPCPP)$")
    message(FATAL_ERROR "Graph compiler backend does not support the chosen CPU runtime. "
    "Consider changing DNNL_CPU_RUNTIME to OMP or turning off graph compiler backend.")
endif()

message(STATUS "Graph compiler backend is enabled.")
set(SC_CPU_RUNTIME ${DNNL_CPU_RUNTIME})
if("${SC_CPU_RUNTIME}" STREQUAL "SYCL")
    set(SC_CPU_RUNTIME "TBB")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "OMP")
    add_definitions_with_host_compiler("-DSC_CPU_THREADPOOL=1")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "TBB")
    add_definitions_with_host_compiler("-DSC_CPU_THREADPOOL=2")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "SEQ")
    add_definitions_with_host_compiler("-DSC_CPU_THREADPOOL=0")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "THREADPOOL")
    add_definitions_with_host_compiler("-DSC_CPU_THREADPOOL=3")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CCXX_NOWARN_FLAGS "")
    append(CCXX_NOWARN_FLAGS "/wd4200")
    # allow usage of "deprecated" functions
    append(CCXX_NOWARN_FLAGS "/wd4996")
    # inherits via dominance
    append(CCXX_NOWARN_FLAGS "/wd4250")
    # conversion from 'size_t' to 'uint16_t'
    append(CCXX_NOWARN_FLAGS "/wd4267")
    # function assumed not to throw an exception but does
    append(CCXX_NOWARN_FLAGS "/wd4297")
    #  format string '%lu' requires an argument of type 'unsigned long'
    append(CCXX_NOWARN_FLAGS "/wd4477")
    # not enough arguments for function-like macro
    append(CCXX_NOWARN_FLAGS "/wd4003")
    # 
    append(CCXX_NOWARN_FLAGS "/wd4624")
    # 'elem_type': unreferenced local variable
    append(CCXX_NOWARN_FLAGS "/wd4101")
    # unary minus operator applied to unsigned type
    append(CCXX_NOWARN_FLAGS "/wd4146")
    # destructor never returns, potential memory leak
    append(CCXX_NOWARN_FLAGS "/wd4722")
    # needs to have dll-interface to be used by clients of struct
    append(CCXX_NOWARN_FLAGS "/wd4251")
    
    append(CMAKE_CCXX_NOWARN_FLAGS ${CCXX_NOWARN_FLAGS})
    set_property(GLOBAL PROPERTY GRAPH_COMPILER_CCXX_NOWARN_FLAGS "${CCXX_NOWARN_FLAGS}")
endif()

append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append_host_compiler_options(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

include_directories_with_host_compiler(${CMAKE_CURRENT_SOURCE_DIR}/core/src/)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

set(OBJ_LIB dnnl_graph_backend_compiler)
add_library(${OBJ_LIB} OBJECT ${SOURCES})


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)