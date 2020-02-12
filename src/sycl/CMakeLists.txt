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

set(TARGET_NAME ${LIB_NAME}_sycl)

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

set(OBJ_LIB ${LIB_NAME}_sycl)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

if(DNNL_SYCL_CUDA)
    find_package(OpenCL REQUIRED)
    add_library(${OBJ_LIB}_interface INTERFACE)
    target_link_libraries(${OBJ_LIB}_interface INTERFACE cuBLAS::cuBLAS cuDNN::cuDNN OpenCL::OpenCL)
    set_target_properties(
        ${OBJ_LIB}
        PROPERTIES
            COMPILE_DEFINITIONS
            "$<TARGET_PROPERTY:cuBLAS::cuBLAS,INTERFACE_COMPILE_DEFINITIONS>;$<TARGET_PROPERTY:cuDNN::cuDNN,INTERFACE_COMPILE_DEFINITIONS>"
            COMPILE_OPTIONS
            "$<TARGET_PROPERTY:cuBLAS::cuBLAS,INTERFACE_COMPILE_OPTIONS>;$<TARGET_PROPERTY:cuDNN::cuDNN,INTERFACE_COMPILE_OPTIONS>;$<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_COMPILE_OPTIONS>"
    )
    target_include_directories(
        ${OBJ_LIB}
        PRIVATE $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:cuDNN::cuDNN,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:cuBLAS::cuBLAS,INTERFACE_INCLUDE_DIRECTORIES>
                ${CMAKE_CURRENT_SOURCE_DIR}/.. ${PROJECT_SOURCE_DIR}/include)
endif()

if(DNNL_SYCL_COMPUTECPP)
    add_sycl_to_target(
        TARGET ${TARGET_NAME}
        SOURCES ${SOURCES}
    )
endif()

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
