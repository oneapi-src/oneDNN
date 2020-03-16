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

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

set(OCL_KERNEL_LIST_EXTERN)
set(OCL_KERNEL_LIST_ENTRIES)

set(unique_kernel_names)

# Parses kernel names from .cl file and updates OCL_KERNEL_LIST_EXTERN and
# OCL_KERNEL_LIST_ENTRIES variables with the parsed kernel names
function(parse_ocl_kernels cl_name cl_path)
    set(entries "${OCL_KERNEL_LIST_ENTRIES}")

    file(READ ${cl_path} contents)
    string(REGEX MATCHALL "kernel[ \n]+void[ \n]+([a-z0-9_]+)" kernels
        ${contents})
    set(cur_kernel_names)
    foreach(k ${kernels})
        string(REGEX REPLACE "kernel[ \n]+void[ \n]+" "" k ${k})
        list(APPEND cur_kernel_names ${k})
        list(FIND unique_kernel_names ${k} index)
        if (${index} GREATER -1)
            message(WARNING "OpenCL kernel name is not unique: ${k}")
        endif()
        set(entries "${entries}\n        { \"${k}\", ${cl_name}_kernel },")
    endforeach()

    set(OCL_KERNEL_LIST_EXTERN
        "${OCL_KERNEL_LIST_EXTERN}\nextern const char *${cl_name}_kernel[];"
        PARENT_SCOPE)
    set(OCL_KERNEL_LIST_ENTRIES "${entries}" PARENT_SCOPE)

    set(unique_kernel_names "${unique_kernel_names};${cur_kernel_names}"
        PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE CL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cl)
foreach(cl_path ${CL_SOURCES})
    get_filename_component(cl_name ${cl_path} NAME_WE)
    set(cpp_file "${CMAKE_CURRENT_BINARY_DIR}/${cl_name}_cl.cpp")
    # XXX: incremental build does not work with headers from OpenCL code
    add_custom_command(
        OUTPUT ${cpp_file}
        COMMAND ${CMAKE_COMMAND}
            -DCL_FILE="${cl_path}"
            -DCPP_FILE="${cpp_file}"
            -DCL_INC_DIR="${PROJECT_SOURCE_DIR}/src"
            -P ${PROJECT_SOURCE_DIR}/cmake/gen_ocl_source.cmake
        DEPENDS ${cl_path}
    )
    list(APPEND SOURCES ${cpp_file})
    parse_ocl_kernels(${cl_name} ${cl_path})
endforeach()

set(kernel_list_src ${PROJECT_BINARY_DIR}/src/gpu/ocl/ocl_kernel_list.cpp)

configure_file(
    "${PROJECT_SOURCE_DIR}/src/gpu/ocl/ocl_kernel_list.cpp.in"
    ${kernel_list_src}
)

list(APPEND SOURCES ${kernel_list_src})

set(OBJ_LIB ${LIB_NAME}_gpu_ocl)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
