#===============================================================================
# Copyright 2019-2024 Intel Corporation
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

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

if(DNNL_SYCL_CUDA OR (DNNL_SYCL_GENERIC AND NVIDIA_TARGET_SUPPORTED))
    append(CMAKE_CXX_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
    # Suppress irrelevant warning about incompatible triplets.
    append(CMAKE_CXX_FLAGS "-Wno-linker-warnings")
endif()

if(DNNL_AMD_ENABLE_SYCL_KERNELS)
    append(CMAKE_CXX_FLAGS "-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${DNNL_AMD_SYCL_KERNELS_TARGET_ARCH}")
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_generic_sycl)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
