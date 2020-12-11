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

if(SYCL_cmake_included)
    return()
endif()
set(SYCL_cmake_included true)

if(NOT DNNL_WITH_SYCL)
    return()
endif()

include(FindPackageHandleStandardArgs)
include(CheckCXXCompilerFlag)

# Check if CXX is Intel oneAPI DPC++ Compiler
CHECK_CXX_COMPILER_FLAG(-fsycl DPCPP_SUPPORTED)
find_package(LevelZero)

if(DPCPP_SUPPORTED)
    if(LevelZero_FOUND)
        message(STATUS "DPC++ support is enabled (OpenCL and Level Zero)")
    else()
        message(STATUS "DPC++ support is enabled (OpenCL)")
    endif()

    # Explicitly link against sycl as Intel oneAPI DPC++ Compiler does not
    # always do it implicitly.
    if(WIN32)
        list(APPEND EXTRA_SHARED_LIBS $<$<CONFIG:Debug>:sycld> $<$<NOT:$<CONFIG:Debug>>:sycl>)
    else()
        list(APPEND EXTRA_SHARED_LIBS sycl)
    endif()
    if(DNNL_SYCL_CUDA)
        # Explicitly linking against OpenCL without finding the right one can
        # end up linking the tests against Nvidia OpenCL. This can be
        # problematic as Intel OpenCL CPU backend will not work. When multiple
        # OpenCL backends are available we need to make sure that we are linking
        # against the correct one.
        find_package(OpenCL REQUIRED)
        find_package(cuBLAS REQUIRED)
        find_package(cuDNN REQUIRED)
        list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)
    else()
        list(APPEND EXTRA_SHARED_LIBS OpenCL)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

    if(LevelZero_FOUND)
        set(DNNL_WITH_LEVEL_ZERO TRUE)
        include_directories(${LevelZero_INCLUDE_DIRS})
    endif()
else()
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} is not Intel oneAPI DPC++ Compiler")
endif()
