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

# Manage OpenCL-related compiler flags
#===============================================================================

if(OpenCL_cmake_included)
    return()
endif()
set(OpenCL_cmake_included true)

if(NOT OPENCLROOT STREQUAL "")
    message(STATUS "Path to OpenCL is specified: trying ${OPENCLROOT}")
endif()

find_path(OpenCL_INCLUDE_DIR
    NAMES CL/cl.h
    PATHS
        ${OPENCLROOT}
        $ENV{OPENCLROOT}
        ${SYCL_BUNDLE_ROOT}
        $ENV{SYCL_BUNDLE_ROOT}
        $ENV{INTELOPENCLSDK}
        $ENV{INTELOCLSDKROOT}
    PATH_SUFFIXES
        include)

find_library(OpenCL_LIBRARY
    NAMES OpenCL
    PATHS
        ${OPENCLROOT}
        $ENV{OPENCLROOT}
        ${SYCL_BUNDLE_ROOT}
        $ENV{SYCL_BUNDLE_ROOT}
        $ENV{INTELOPENCLSDK}
        $ENV{INTELOCLSDKROOT}
    PATH_SUFFIXES
        lib64
        lib
        lib/x64)

mark_as_advanced(
    OpenCL_INCLUDE_DIR
    OpenCL_LIBRARY)

if(NOT OpenCL_INCLUDE_DIR OR NOT OpenCL_LIBRARY)
    message(FATAL_ERROR
        "Could NOT find OpenCL (missing: OpenCL_LIBRARY OpenCL_INCLUDE_DIR)")
endif()

message(STATUS "OpenCL include: ${OpenCL_INCLUDE_DIR}")
message(STATUS "OpenCL library: ${OpenCL_LIBRARY}")

include_directories(${OpenCL_INCLUDE_DIR})
list(APPEND EXTRA_SHARED_LIBS ${OpenCL_LIBRARY})
