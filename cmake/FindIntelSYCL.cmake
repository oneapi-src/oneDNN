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

find_package(OpenCL REQUIRED)

include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

unset(INTEL_SYCL_SUPPORTED CACHE)
check_cxx_compiler_flag("-fsycl" INTEL_SYCL_SUPPORTED)

get_filename_component(INTEL_SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} PATH)

# Try to find Intel SYCL version.hpp header
find_path(INTEL_SYCL_INCLUDE_DIRS
    NAMES CL/sycl/version.hpp
    PATHS
      ${sycl_root_hints}
      "${INTEL_SYCL_BINARY_DIR}/.."
    PATH_SUFFIXES
        include
        lib/clang/9.0.0/include
        lib/clang/8.0.0/include
    NO_DEFAULT_PATH)

find_library(INTEL_SYCL_LIBRARIES
    NAMES "sycl"
    PATHS
        ${sycl_root_hints}
        "${INTEL_SYCL_BINARY_DIR}/.."
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)

find_package_handle_standard_args(IntelSYCL
    FOUND_VAR IntelSYCL_FOUND
    REQUIRED_VARS
        INTEL_SYCL_LIBRARIES
        INTEL_SYCL_INCLUDE_DIRS
        INTEL_SYCL_SUPPORTED)

if(IntelSYCL_FOUND AND NOT TARGET Intel::SYCL)
    add_library(Intel::SYCL UNKNOWN IMPORTED)
    set(imp_libs
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:-fsycl>
        OpenCL::OpenCL)
    set_target_properties(Intel::SYCL PROPERTIES
        IMPORTED_LINK_INTERFACE_LIBRARIES "${imp_libs}"
        INTERFACE_INCLUDE_DIRECTORIES "${INTEL_SYCL_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${INTEL_SYCL_LIBRARIES}")
    set(INTEL_SYCL_FLAGS "-fsycl")
    mark_as_advanced(
        INTEL_SYCL_FLAGS
        INTEL_SYCL_LIBRARIES
        INTEL_SYCL_INCLUDE_DIRS)
endif()
