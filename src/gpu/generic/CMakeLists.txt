#===============================================================================
# Copyright 2024 Intel Corporation
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

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

# - Always enable generic SYCL kernels for generic vendor.
# - Always enable generic SYCL kernels for NVIDIA vendor.
# - Only enable the generic SYCL kernels for AMD vendor if target architecture has been specified.
if(DNNL_SYCL_GENERIC OR DNNL_SYCL_CUDA OR DNNL_AMD_ENABLE_SYCL_KERNELS)
    add_subdirectory(sycl)
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_generic)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
