#===============================================================================
# Copyright 2022-2023 Intel Corporation
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

if(DNNL_SYCL_CUDA OR DNNL_SYCL_HIP)
    file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )
    # Interop kernel is not used for non-Intel GPU vendor.
    list(REMOVE_ITEM SOURCES
        "sycl_interop_gpu_kernel.hpp"
        "sycl_interop_gpu_kernel.cpp"
        )
    if(DNNL_SYCL_CUDA)
        append(CMAKE_CXX_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
        # Suppress irrelevant warning about incompatible triplets.
        append(CMAKE_CXX_FLAGS "-Wno-linker-warnings")
    endif()
else()
    # GPU vendor Intel needs only the following files from this directory.
    set(SOURCES
       "sycl_interop_gpu_kernel.hpp"
       "sycl_interop_gpu_kernel.cpp"
       "sycl_gpu_engine.hpp"
       )
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_sycl)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
