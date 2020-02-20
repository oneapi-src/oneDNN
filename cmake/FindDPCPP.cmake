#===============================================================================
# Copyright 2019-2020 Intel Corporation
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

if(UNIX AND NOT APPLE)
    set(dpcpp_os "linux")
elseif(WIN32)
    set(dpcpp_os "windows")
else()
    message(FATAL_ERROR "OS is not supported")
endif()

if(DPCPPROOT)
    list(APPEND dpcpp_root_hints ${DPCPPROOT}/compiler/latest/${dpcpp_os})
endif()

if(DEFINED ENV{DPCPP_ROOT})
    list(APPEND dpcpp_root_hints $ENV{DPCPP_ROOT}/compiler/latest/${dpcpp_os})
endif()

list(APPEND sycl_root_hints
            ${dpcpp_root_hints}
            ${SYCLROOT}
            $ENV{SYCLROOT})

# This is used to prioritize OpenCL from SYCL package against the system OpenCL
set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(sycl_root_hints)
    list(INSERT CMAKE_PREFIX_PATH 0 ${sycl_root_hints})
endif()

# XXX: workaround to use OpenCL from DPC++ package
set(DPCPP_COMPILER_VERSION 10.0.0)

if(DEFINED ENV{DPCPP_ROOT})
    list(INSERT CMAKE_PREFIX_PATH 0
        $ENV{DPCPP_ROOT}/compiler/latest/${dpcpp_os}/lib/clang/${DPCPP_COMPILER_VERSION})
endif()

find_package(OpenCL REQUIRED)

include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

unset(DPCPP_SUPPORTED CACHE)
check_cxx_compiler_flag("-fsycl" DPCPP_SUPPORTED)

get_filename_component(DPCPP_BINARY_DIR ${CMAKE_CXX_COMPILER} PATH)

# Try to find version.hpp header from DPC++
find_path(DPCPP_INCLUDE_DIRS
    NAMES CL/sycl/version.hpp
    PATHS
      ${sycl_root_hints}
      "${DPCPP_BINARY_DIR}/.."
    PATH_SUFFIXES
        include
        lib/clang/${DPCPP_COMPILER_VERSION}/include
    NO_DEFAULT_PATH)

find_library(DPCPP_LIBRARIES
    NAMES "sycl"
    PATHS
        ${sycl_root_hints}
        "${DPCPP_BINARY_DIR}/.."
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)

find_package_handle_standard_args(DPCPP
    FOUND_VAR DPCPP_FOUND
    REQUIRED_VARS
        DPCPP_LIBRARIES
        DPCPP_INCLUDE_DIRS
        DPCPP_SUPPORTED)

if(DPCPP_FOUND AND NOT TARGET DPCPP::DPCPP)
    add_library(DPCPP::DPCPP UNKNOWN IMPORTED)
    set(imp_libs
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:-fsycl>
        OpenCL::OpenCL)
    set_target_properties(DPCPP::DPCPP PROPERTIES
        IMPORTED_LINK_INTERFACE_LIBRARIES "${imp_libs}"
        INTERFACE_INCLUDE_DIRECTORIES "${DPCPP_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${DPCPP_LIBRARIES}"
        IMPORTED_LOCATION_DEBUG "${DPCPP_LIBRARIES}")
    set(DPCPP_FLAGS "-fsycl")
    mark_as_advanced(
        DPCPP_FLAGS
        DPCPP_LIBRARIES
        DPCPP_INCLUDE_DIRS)
endif()

# Reverting the CMAKE_PREFIX_PATH to its original state
set(CMAKE_PREFIX_PATH ${original_cmake_prefix_path})
