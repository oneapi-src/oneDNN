#===============================================================================
# Copyright 2024-2025 Intel Corporation
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

# TODO: Extend conversion warning support to GEMM code
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    append(CMAKE_CXX_FLAGS " -Wno-implicit-int-conversion")
endif()


add_subdirectory(${ONEDNN_GEMMSTONE_DIR} gemmstone)
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
   $<TARGET_OBJECTS:${GEMMSTONE_LIBRARY_NAME}>)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )


set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_intel_jit_gemm)
include_directories(${OBJ_LIB} ${ONEDNN_GEMMSTONE_DIR}/.. ${ONEDNN_GEMMSTONE_DIR} ${ONEDNN_GEMMSTONE_DIR}/include)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
