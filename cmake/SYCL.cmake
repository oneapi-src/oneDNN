#===============================================================================
# Copyright 2019 Intel Corporation
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

# Manage SYCL-related compiler flags
#===============================================================================

cmake_minimum_required(VERSION 3.4.3)

if(SYCL_cmake_included)
    return()
endif()
set(SYCL_cmake_included true)

set(sycl_root_hint)
if(DEFINED SYCLROOT)
    set(sycl_root_hint ${SYCLROOT})
elseif(DEFINED ENV{SYCLROOT})
    set(sycl_root_hint $ENV{SYCLROOT})
endif()

set(sycl_root_hints)
if(sycl_root_hint)
    list(APPEND sycl_root_hints ${sycl_root_hint})
else()
    list(APPEND sycl_root_hints ${SYCL_BUNDLE_ROOT})
    list(APPEND sycl_root_hints $ENV{SYCL_BUNDLE_ROOT})
endif()

# Try to find Intel SYCL version.hpp header
find_file(INTEL_SYCL_VERSION
    NAMES version.hpp
    PATHS
        ${sycl_root_hints}
    PATH_SUFFIXES
        include/CL/sycl
        lib/clang/9.0.0/include/CL/sycl
        lib/clang/8.0.0/include/CL/sycl
    NO_DEFAULT_PATH)

if(INTEL_SYCL_VERSION)
    get_filename_component(SYCL_INCLUDE_DIR
            "${INTEL_SYCL_VERSION}/../../.." ABSOLUTE)

    # Suppress the compiler warning about undefined CL_TARGET_OPENCL_VERSION
    add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

    find_library(SYCL_LIBRARY
        NAMES "sycl"
        HINTS
            ${sycl_root_hints}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
    if(NOT SYCL_LIBRARY)
        message(FATAL_ERROR "SYCL library not found")
    endif()

    # Find the OpenCL library from the SYCL distribution
    find_library(OpenCL_LIBRARY
        NAMES "OpenCL"
        HINTS
            ${sycl_root_hints}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
    if(NOT OpenCL_LIBRARY)
        message(FATAL_ERROR "OpenCL library not found")
    endif()
    set(OpenCL_INCLUDE_DIR ${SYCL_INCLUDE_DIR} CACHE STRING "")

    set(MKLDNN_SYCL_INTEL true)

    message(STATUS "Intel SYCL include: ${SYCL_INCLUDE_DIR}")
    message(STATUS "Intel SYCL library: ${SYCL_LIBRARY}")
    message(STATUS "OpenCL include: ${OpenCL_INCLUDE_DIR}")
    message(STATUS "OpenCL library: ${OpenCL_LIBRARY}")

    if(NOT ${SYCL_INCLUDE_DIR} STREQUAL ${OpenCL_INCLUDE_DIR})
        include_directories(${OpenCL_INCLUDE_DIR})
    endif()

    include_directories(${SYCL_INCLUDE_DIR})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl")

    list(APPEND EXTRA_SHARED_LIBS ${SYCL_LIBRARY})
    list(APPEND EXTRA_SHARED_LIBS ${OpenCL_LIBRARY})
else()
    # ComputeCpp-specific flags
    # 1. Ignore the warning about undefined symbols in SYCL kernels - comes from
    #    SYCL CPU thunks
    # 2. Fix remark [Computecpp:CC0027] about memcpy/memset intrinsics
    set(COMPUTECPP_USER_FLAGS
        -Wno-sycl-undef-func
        -no-serial-memop
        CACHE STRING "")
    set(ComputeCpp_DIR ${sycl_root_hint})
    include(cmake/FindComputeCpp.cmake)
    if(NOT ComputeCpp_FOUND)
        message(FATAL_ERROR "SYCL not found")
    endif()

    set(MKLDNN_SYCL_COMPUTECPP true)
    include_directories(SYSTEM ${ComputeCpp_INCLUDE_DIRS})
    list(APPEND EXTRA_SHARED_LIBS ${COMPUTECPP_RUNTIME_LIBRARY})

    include_directories(${OpenCL_INCLUDE_DIRS})
    list(APPEND EXTRA_SHARED_LIBS ${OpenCL_LIBRARIES})
endif()
