# *******************************************************************************
# Copyright 2020 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
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
# *******************************************************************************

if(blas_cmake_included)
    return()
endif()
set(blas_cmake_included true)
include("cmake/options.cmake")

if(DNNL_TARGET_ARCH STREQUAL "AARCH64")
    if(DNNL_AARCH64_USE_ARMPL)
        set(_DNNL_USE_CBLAS ON)
        if(DNNL_CPU_RUNTIME STREQUAL "OMP")
            set(BLA_VENDOR "Arm_mp")
        else()
            set(BLA_VENDOR "Arm")
        endif()
    elseif(DNNL_AARCH64_USE_OPENBLAS)
        set(_DNNL_USE_CBLAS ON)
        set(BLA_VENDOR "OpenBLAS")
    endif()
endif()

if(NOT _DNNL_USE_CBLAS)
    return()
endif()

find_package(BLAS REQUIRED)

if(BLAS_FOUND)
     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
     list(APPEND EXTRA_SHARED_LIBS ${BLAS_LIBRARIES})

     # Check that the BLAS library supports the CBLAS interface.
     set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LINKER_FLAGS};${BLAS_LIBRARIES}")
     set(CMAKE_REQUIRED_FLAGS "${BLAS_COMPILER_FLAGS}")

     # Find and include  accompanying cblas.h
     get_filename_component(BLAS_LIB_DIR ${BLAS_LIBRARIES} PATH)
     find_path(BLAS_INCLUDE_DIR cblas.h $ENV{CPATH} ${BLAS_LIB_DIR}/../include)
     include_directories(${BLAS_INCLUDE_DIR})

     unset(CBLAS_WORKS CACHE)

     # Check we have a working CBLAS interface
     check_function_exists(cblas_sgemm CBLAS_WORKS)
     if(NOT CBLAS_WORKS)
         message(FATAL_ERROR "BLAS library does not support CBLAS interface.")
     endif()
     message(STATUS "Found CBLAS: ${BLAS_LIBRARIES}")
     message(STATUS "CBLAS include path: ${BLAS_INCLUDE_DIR}")
     add_definitions(-DUSE_CBLAS)
endif()
