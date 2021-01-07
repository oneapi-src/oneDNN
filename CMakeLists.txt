#===============================================================================
# Copyright 2020 Intel Corporation
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

cmake_minimum_required(VERSION 2.8.11)
message(STATUS "cmake version: ${CMAKE_VERSION}")

# https://github.com/oneapi-src/oneDNN/blob/master/CMakeLists.txt

if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW)
endif()

# Foo::Bar always refers to an IMPORTED target
if(POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW)
endif()

if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Enable RPATH on MacOS/OSX
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()

# Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

# Do not export symbols from executables
if(POLICY CMP0065)
    cmake_policy(SET CMP0065 NEW)
endif()

# Pass linker flags to try_compile
if(POLICY CMP0056)
    cmake_policy(SET CMP0056 NEW)
endif()

# Always link with full path
if(POLICY CMP0060)
    cmake_policy(SET CMP0060 NEW)
endif()

# Pass compiler flags to try_compile
if(POLICY CMP0066)
    cmake_policy(SET CMP0066 NEW)
endif()

# Use <PackageName>_ROOT env. variable as a prefix
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Install rules order
if(POLICY CMP0082)
    cmake_policy(SET CMP0082 NEW)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert ...")
endif()

# oneDNN Graph

set(LIB_NAME dnnl_graph)
set(PROJECT_NAME "oneDNN Graph")
set(PROJECT_FULL_NAME "oneAPI Deep Neural Network Library Graph API (oneDNN Graph)")
set(PROJECT_VERSION 0.1.0)

if (CMAKE_VERSION VERSION_LESS 3.0)
    project(${PROJECT_NAME} C CXX)
else()
    cmake_policy(SET CMP0048 NEW)
    project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)
endif()

string(REPLACE "." ";" VERSION_LIST ${PROJECT_VERSION})
list(GET VERSION_LIST 0 DNNL_GRAPH_VERSION_MAJOR)
list(GET VERSION_LIST 1 DNNL_GRAPH_VERSION_MINOR)
list(GET VERSION_LIST 2 DNNL_GRAPH_VERSION_PATCH)

if(MSVC)
    message(FATAL_ERROR "oneDNN Graph doesn't support MSVC at this moment.")
endif()

if(UNIX OR MINGW)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

# if gcc version is less than gcc 5, DNNL_GRAPH_SUPPORT_CXX17 should set OFF
option(DNNL_GRAPH_SUPPORT_CXX17 "uses features from c++ standard 17" OFF)
if(DNNL_GRAPH_SUPPORT_CXX17)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
        message(FATAL_ERROR "Current compiler version does not support C++17 features. Please undefine DNNL_GRAPH_SUPPORT_CXX17.")
    endif()
    if(NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
        message(WARNING "Current compiler version may not support full C++17 features. Please undefine DNNL_GRAPH_SUPPORT_CXX17 if fail to build.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    add_definitions(-DDNNL_GRAPH_SUPPORT_CXX17=1)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DNNL_GRAPH_VERBOSE
    "allows oneDNN Graph library be verbose whenever DNNL_GRAPH_VERBOSE
    environment variable set to 1, 2 or 3" ON) # enabled by default

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "oneDNN supports 64 bit platforms only")
endif()

set(DNNL_GRAPH_CPU_RUNTIME "OMP" CACHE STRING
    "specifies the threading runtime for CPU engines;
    supports OMP (default).")

if(NOT "${DNNL_GRAPH_CPU_RUNTIME}" MATCHES "^(OMP|SEQ|DPCPP|THREADPOOL)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${DNNL_GRAPH_CPU_RUNTIME}")
endif()

set(DNNL_GRAPH_GPU_RUNTIME "NONE" CACHE STRING
    "specifies the runtime to use for GPU engines.
    Can be NONE (default; no GPU engines)
    or DPCPP (DPC++ GPU engines).")

if(NOT "${DNNL_GRAPH_GPU_RUNTIME}" MATCHES "^(NONE|DPCPP)$")
    message(FATAL_ERROR "Unsupported GPU runtime: ${DNNL_GRAPH_GPU_RUNTIME}")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include("cmake/build_oneDNN.cmake")

set(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-Wall -Wextra -pedantic -Wconversion -Wno-sign-compare -Wunreachable-code")

if(DNNL_GRAPH_WITH_SYCL)
    # Inhibit all warning messages from sycl
    append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-w -Wno-deprecated-declarations")
else()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        # suppress some warnings for gtest header files when built with Clang
        append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-w -Werror")
    else()
        append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-Werror -Wfatal-errors")
    endif()
endif()

if(${COVERAGE_REPORT})
    message("Compiling oneDNN Graph with coverage support")
    append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-fprofile-arcs -ftest-coverage")
    set(DNNL_GRAPH_COMMON_LINK_FLAGS ${DNNL_GRAPH_COMMON_LINK_FLAGS}
        -lgcov
        -fprofile-arcs
        -ftest-coverage
        )
endif()

if(${DNNL_GRAPH_ENABLE_ASAN})
    message(STATUS "Build oneDNN Graph with AddressSanitizer")
    append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-fsanitize=address -fno-omit-frame-pointer -fno-var-tracking")
    if (NOT CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
        append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-g")
    endif()
    set(DNNL_GRAPH_COMMON_LINK_FLAGS ${DNNL_GRAPH_COMMON_LINK_FLAGS}
        -fsanitize=address
        -fno-omit-frame-pointer
        )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-Wno-gnu-statement-expression")
endif()

include("cmake/check_headers.cmake")
include("cmake/Doxygen.cmake")

enable_testing()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(src)
add_subdirectory(tests/cpp)
add_subdirectory(examples/c)
add_subdirectory(examples/cpp)

