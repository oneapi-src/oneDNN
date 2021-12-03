#===============================================================================
# Copyright 2021 Intel Corporation
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

if(NOT DNNL_GRAPH_BUILD_COMPILER_BACKEND)
    message(STATUS "graph compiler backend is disabled.")
    return()
endif()

SET(SC_LLVM_VERSION "OFF" CACHE STRING "version of LLVM")
SET(SC_LLVM_LIB_NAME "OFF" CACHE STRING "the lib name of LLVM for linker")
SET(SC_LLVM_INCLUDE_PATH "OFF" CACHE STRING "the header include path of LLVM")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/core/cmake")
include("core/cmake/find_llvm.cmake")

set(SC_LLVM_CONFIG ${DNNL_GRAPH_LLVM_CONFIG})

find_llvm()

if(${SC_LLVM_VERSION} STREQUAL OFF OR ${SC_LLVM_VERSION} LESS 8)
    message(FATAL_ERROR "Expecting LLVM version >=8, got ${SC_LLVM_VERSION}, 
            graph compiler backend is disabled.")
else()
    if("${SC_LLVM_INCLUDE_PATH}" STREQUAL OFF)
        message(FATAL_ERROR "LLVM_INCLUDE_PATH is not successfully set, 
            graph compiler backend is disabled.")
    else()
        message(STATUS "Found LLVM_VERSION=" ${SC_LLVM_VERSION})
    endif()
endif()

if(DNNL_GRAPH_CPU_RUNTIME MATCHES "(TBB|SEQ|DPCPP|THREADPOOL)$")
    message(FATAL_ERROR "graph compiler backend does not support the chosen CPU runtime. "
    "Consider changing DNNL_GRAPH_CPU_RUNTIME to OMP or turning of graph compiler backend.")
endif()

message(STATUS "graph compiler backend is enabled.")
if(DNNL_GRAPH_CPU_RUNTIME STREQUAL "OMP")
    set(SC_OMP_ENABLED "ON")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/src/)
