#===============================================================================
# Copyright 2021-2023 Intel Corporation
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

file(GLOB HEADERS
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    )

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

file(GLOB_RECURSE SOURCES_EXTRA
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/patterns/*.[ch]pp
    )

foreach(SOURCE_FILE ${SOURCES_EXTRA})
    list(APPEND SOURCES "${SOURCE_FILE}")
endforeach()

# The build option is for debug and testing purpose. Please do not use it in
# production.
if(_ONEDNN_GRAPH_LAYOUT_DEBUG)
    message(STATUS "Graph: enable layout debug mode in DNNL backend")

    file(GLOB FILES_REQUIRED_LAYOUT_DEBUG
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layout_id_mgr.cpp
    )

    set_source_files_properties(${FILES_REQUIRED_LAYOUT_DEBUG}
        PROPERTIES COMPILE_FLAGS "-DDNNL_GRAPH_LAYOUT_DEBUG")
endif()

if(MSVC AND (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel"))
    file(GLOB FILES_REQUIRED_BIGOBJ
        ${CMAKE_CURRENT_SOURCE_DIR}/dnnl_backend.cpp
        )
    set_source_files_properties(${FILES_REQUIRED_BIGOBJ}
        PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_graph_backend_dnnl)
add_library(${OBJ_LIB} OBJECT ${HEADERS} ${SOURCES})

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
