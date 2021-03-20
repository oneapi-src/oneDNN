#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

# Interface headers for the library must be self-sufficient, i.e., must include all
# necessary dependencies by themself.

if(check_headers_cmake_included)
    return()
endif()
set(check_headers_cmake_included true)

include(CheckIncludeFile)
include(CheckIncludeFileCXX)

function(check_headers)
    if(DNNL_GRAPH_SUPPORT_CXX17)
        set(CMAKE_CXX_STANDARD 14)
    else()
        set(CMAKE_CXX_STANDARD 11)
    endif()
    set(DNNL_GRAPH_HDRS
        ${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/dnnl_graph.h
        ${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/dnnl_graph_types.h
        ${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/dnnl_graph.hpp
        )

    set(CMAKE_REQUIRED_INCLUDES "${PROJECT_SOURCE_DIR}/include/")

    foreach(FILE ${DNNL_GRAPH_HDRS})
        get_filename_component(${FILE}_EXT ${FILE} EXT)
        if(${FILE}_EXT STREQUAL ".h")
            CHECK_INCLUDE_FILE("${FILE}" ${FILE}_VALID)
        else()
            CHECK_INCLUDE_FILE_CXX("${FILE}" ${FILE}_VALID "-Werror")
        endif()
        if(NOT ${FILE}_VALID)
            message(FATAL_ERROR "oneDNN Graph header: ${FILE} is not self-sufficient")
        endif()
    endforeach()
endfunction()

check_headers()
