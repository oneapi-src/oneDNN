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

include(FindPackageHandleStandardArgs)

set(sycl_root_hint)
if(DEFINED DPCPP_ROOT)
    set(sycl_root_hint ${DPCPP_ROOT})
elseif(DEFINED ENV{DPCPP_ROOT})
    set(sycl_root_hint $ENV{DPCPP_ROOT})
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
        include/sycl/CL/sycl
        lib/clang/11.0.0/include/CL/sycl
        lib/clang/10.0.0/include/CL/sycl
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

    set(USE_DPCPP true)
    add_definitions(-DUSE_DPCPP)

    message(STATUS "Intel SYCL include: ${SYCL_INCLUDE_DIR}")
    message(STATUS "Intel SYCL library: ${SYCL_LIBRARY}")
    message(STATUS "OpenCL include: ${OpenCL_INCLUDE_DIR}")
    message(STATUS "OpenCL library: ${OpenCL_LIBRARY}")

    if(NOT ${SYCL_INCLUDE_DIR} STREQUAL ${OpenCL_INCLUDE_DIR})
        include_directories(${OpenCL_INCLUDE_DIR})
    endif()

    include_directories(${SYCL_INCLUDE_DIR})

    list(APPEND EXTRA_SHARED_LIBS ${SYCL_LIBRARY})
    list(APPEND EXTRA_SHARED_LIBS ${OpenCL_LIBRARY})
else()
    message(FATAL_ERROR "DPCPP library not found, Please specify the DPCPP_ROOT env")
endif()
