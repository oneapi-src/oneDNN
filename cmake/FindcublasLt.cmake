# ===============================================================================
# Copyright 2020-2025 Intel Corporation 
# Copyright 2020-2024 Codeplay Software Limited
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ===============================================================================

find_package(CUDA 10.0 REQUIRED)
find_package(Threads REQUIRED)

find_path(
  CUBLASLT_INCLUDE_DIR "cublasLt.h"
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(CUDA_DRIVER_LIBRARY cuda)

find_library(
  CUBLASLT_LIBRARY cublasLt
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 bin)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  cublasLt REQUIRED_VARS CUBLASLT_INCLUDE_DIR CUDA_INCLUDE_DIRS CUBLASLT_LIBRARY
                         CUDA_LIBRARIES CUDA_DRIVER_LIBRARY)

if(NOT TARGET cublasLt::cublasLt)
  add_library(cublasLt::cublasLt SHARED IMPORTED)
  set_target_properties(
    cublasLt::cublasLt
    PROPERTIES IMPORTED_LOCATION ${CUBLASLT_LIBRARY}
               INTERFACE_INCLUDE_DIRECTORIES
               "${CUBLASLT_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}"
               INTERFACE_LINK_LIBRARIES
               "Threads::Threads;${CUDA_DRIVER_LIBRARY};${CUDA_LIBRARIES}"
               INTERFACE_COMPILE_DEFINITIONS CUDA_NO_HALF)
endif()
