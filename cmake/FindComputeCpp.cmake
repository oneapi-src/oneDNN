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

check_cxx_compiler_flag("-sycl-driver" COMPUTECPP_SYCL_DRIVER_SUPPORTED)

get_filename_component(COMPUTECPP_BINARY_DIR ${CMAKE_CXX_COMPILER} PATH)

find_library(COMPUTECPP_LIBRARIES NAMES ComputeCpp
    PATHS "${COMPUTECPP_BINARY_DIR}/../lib")
find_path(COMPUTECPP_INCLUDE_DIRS CL/sycl.hpp
    PATHS "${COMPUTECPP_BINARY_DIR}/../include")

find_package_handle_standard_args(ComputeCpp
    FOUND_VAR ComputeCpp_FOUND
    REQUIRED_VARS
        COMPUTECPP_LIBRARIES
        COMPUTECPP_INCLUDE_DIRS
        COMPUTECPP_SYCL_DRIVER_SUPPORTED)

if(ComputeCpp_FOUND AND NOT TARGET Codeplay::ComputeCpp)
  # CMake assumes that this directory is an implicit link directory, so it
  # wouldn't need to attach an rpath. This is not the case, remove this
  # directory from the list so the rpaths work correctly.
  get_filename_component(COMPUTECPP_LIBRARY_DIR
      ${COMPUTECPP_LIBRARIES} PATH)
  list(REMOVE_ITEM
      CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES ${COMPUTECPP_LIBRARY_DIR})

  add_library(Codeplay::ComputeCpp UNKNOWN IMPORTED)
  set(COMPUTECPP_FLAGS
      "-sycl-driver -Wno-sycl-undef-func -no-serial-memop -intelspirmetadata")
  set_target_properties(Codeplay::ComputeCpp PROPERTIES
        IMPORTED_LINK_INTERFACE_LIBRARIES OpenCL::OpenCL
        INTERFACE_INCLUDE_DIRECTORIES "${COMPUTECPP_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${COMPUTECPP_LIBRARIES}"
        IMPORTED_NO_SONAME OFF)
        mark_as_advanced(
            COMPUTECPP_LIBRARIES
            COMPUTECPP_FLAGS
            COMPUTECPP_INCLUDE_DIRS
            COMPUTECPP_BINARY_DIR)
endif()
