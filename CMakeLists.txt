#===============================================================================
# Copyright 2020-2022 Intel Corporation
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

cmake_minimum_required(VERSION 2.8.12)
message(STATUS "cmake version: ${CMAKE_VERSION}")

# https://github.com/oneapi-src/oneDNN/blob/master/CMakeLists.txt

if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW)
endif()

# Foo::Bar always refers to an IMPORTED target
if(POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW)
endif()

if(NOT CMAKE_VERSION VERSION_LESS 3.0)
    # https://cmake.org/cmake/help/latest/policy/CMP0025.html#policy:CMP0025
    # Enforce CMake 3.0 and above recognize like that Apple Clang is Clang to
    # ensure the compatibility with version prior to 3.0.
    if(POLICY CMP0025)
        cmake_policy(SET CMP0025 OLD)
    endif()
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
set(PROJECT_VERSION 0.5.0)

if (CMAKE_VERSION VERSION_LESS 3.0)
    project(${PROJECT_NAME} C CXX)
else()
    cmake_policy(SET CMP0048 NEW)
    project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)
endif()

if(UNIX OR MINGW)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include("cmake/options.cmake")
include("cmake/SYCL.cmake")

# Let SYCL to choose the C++ standard it needs.
if(NOT (DNNL_GRAPH_WITH_SYCL OR CMAKE_BASE_NAME STREQUAL "icx" OR CMAKE_BASE_NAME STREQUAL "icpx"))
    if(DNNL_GRAPH_SUPPORT_CXX17)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
            message(FATAL_ERROR "Current compiler version does not support C++17 features. Please undefine DNNL_GRAPH_SUPPORT_CXX17.")
        endif()
        if(NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
            message(WARNING "Current compiler version may not support full C++17 features. Please undefine DNNL_GRAPH_SUPPORT_CXX17 if fail to build.")
        endif()
        add_definitions(-DDNNL_GRAPH_SUPPORT_CXX17=1)
        if (UNIX OR MINGW)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
        elseif (MSVC)
            set(CMAKE_CXX_STANDARD 17)
            set(CMAKE_CXX_STANDARD_REQUIRED ON)
        endif()
    else()
        if (UNIX OR MINGW)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        elseif (MSVC)
            set(CMAKE_CXX_STANDARD 11)
            set(CMAKE_CXX_STANDARD_REQUIRED ON)
        endif()
    endif()
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "oneDNN supports 64 bit platforms only")
endif()

include("cmake/build_oneDNN.cmake")
include("cmake/utils.cmake")

include_directories(include)

set(DNNL_GRAPH_COMMON_COMPILATION_FLAGS)
set(DNNL_GRAPH_COMMON_LINK_FLAGS)

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

# Set up debug mode for memory layout (option is defined in cmake/options.cmake)
if(DNNL_GRAPH_LAYOUT_DEBUG)
    message(STATUS "Layout debugging mode enabled")
    append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-DDNNL_GRAPH_LAYOUT_DEBUG")
endif()

if(DNNL_GRAPH_ENABLE_DUMP)
    message(STATUS "Graph/passes dump enabled")
    append(DNNL_GRAPH_COMMON_COMPILATION_FLAGS "-DDNNL_GRAPH_ENABLE_DUMP")
endif()

if(NOT WIN32)
    include("cmake/check_headers.cmake")
endif()

include("cmake/Doxygen.cmake")
include("cmake/version.cmake")
include("cmake/OpenMP.cmake")
include("cmake/Threadpool.cmake")
include("cmake/platform.cmake")
include("cmake/SDL.cmake")
include("cmake/coverage.cmake")
include("cmake/testing.cmake")

enable_testing()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Add pre-processor definition for CPU runtime.
message(STATUS "Compiling oneDNN Graph with CPU runtime ${DNNL_GRAPH_CPU_RUNTIME} support")

if (DNNL_GRAPH_CPU_RUNTIME STREQUAL "SEQ")
    add_definitions(-DDNNL_GRAPH_CPU_RUNTIME=1)
elseif (DNNL_GRAPH_CPU_RUNTIME STREQUAL "OMP")
    if (OpenMP_CXX_FOUND)
        add_definitions(-DDNNL_GRAPH_CPU_RUNTIME=2)
    else()
        add_definitions(-DDNNL_GRAPH_CPU_RUNTIME=1)
    endif()
elseif (DNNL_GRAPH_CPU_RUNTIME STREQUAL "THREADPOOl")
    add_definitions(-DDNNL_GRAPH_CPU_RUNTIME=8)
elseif (DNNL_GRAPH_CPU_RUNTIME STREQUAL "DPCPP")
    add_definitions(-DDNNL_GRAPH_CPU_RUNTIME=512)
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples/c)
add_subdirectory(examples/cpp)

install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_NAME})
install(FILES THIRD-PARTY-PROGRAMS DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_NAME})
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_NAME})
