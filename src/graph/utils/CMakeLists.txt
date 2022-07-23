#===============================================================================
# Copyright 2021-2022 Intel Corporation
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
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/pm/*.[ch]pp
    )

function(JOIN VALUES GLUE OUTPUT)
    string (REGEX REPLACE ";" "${GLUE}" _TMP_STR "${VALUES}")
    set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

function(APPLY_VERSION_DEFINITIONS FILES)
    set(VERSION_DEFINITIONS
        DNNL_GRAPH_VERSION_MAJOR=${DNNL_VERSION_MAJOR}
        DNNL_GRAPH_VERSION_MINOR=${DNNL_VERSION_MINOR}
        DNNL_GRAPH_VERSION_PATCH=${DNNL_VERSION_PATCH}
        DNNL_GRAPH_VERSION_HASH="${DNNL_VERSION_HASH}")
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

APPLY_VERSION_DEFINITIONS(${CMAKE_CURRENT_SOURCE_DIR}/verbose.cpp)

set(OBJ_LIB ${DNNL_LIBRARY_NAME}_graph_utils)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
