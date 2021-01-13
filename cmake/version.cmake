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

# This file is leverage from:
# https://github.com/oneapi-src/oneDNN/blob/master/cmake/version.cmake

if(dnnl_graph_version_cmake_included)
    return()
endif()
set(dnnl_graph_version_cmake_included true)
include("cmake/utils.cmake")

string(REPLACE "." ";" VERSION_LIST ${PROJECT_VERSION})
list(GET VERSION_LIST 0 DNNL_GRAPH_VERSION_MAJOR)
list(GET VERSION_LIST 1 DNNL_GRAPH_VERSION_MINOR)
list(GET VERSION_LIST 2 DNNL_GRAPH_VERSION_PATCH)

find_package(Git)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE DNNL_GRAPH_VERSION_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
    set(DNNL_GRAPH_VERSION_HASH "N/A")
endif()

# Apply version definitions to particular files
function(APPLY_VERSION_DEFINITIONS FILES)
    set(VERSION_DEFINITIONS
        DNNL_GRAPH_VERSION_MAJOR=${DNNL_GRAPH_VERSION_MAJOR}
        DNNL_GRAPH_VERSION_MINOR=${DNNL_GRAPH_VERSION_MINOR}
        DNNL_GRAPH_VERSION_PATCH=${DNNL_GRAPH_VERSION_PATCH}
        DNNL_GRAPH_VERSION_HASH="${DNNL_GRAPH_VERSION_HASH}")
    if(CMAKE_VERSION VERSION_LESS 3.12)
        JOIN("${VERSION_DEFINITIONS}" ";" VERSION_MACROS)
    else()
        list(JOIN VERSION_DEFINITIONS "$<SEMICOLON>" VERSION_MACROS)
    endif()

    foreach(FILE ${FILES})
        message(STATUS "Set version definitions to ${FILE}")
    endforeach()
    set_source_files_properties(${FILES} PROPERTIES
        COMPILE_DEFINITIONS "${VERSION_MACROS}")
endfunction()
