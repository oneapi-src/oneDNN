#===============================================================================
# Copyright 2021 Intel Corporation
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
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include/*.h
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include/*.hpp
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

if(MSVC AND (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel"))
    file(GLOB FILES_REQUIRED_BIGOBJ
        ${CMAKE_CURRENT_SOURCE_DIR}/dnnl_backend.cpp
        )
    set_source_files_properties(${FILES_REQUIRED_BIGOBJ}
        PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

include_directories(${PROJECT_SOURCE_DIR}/third_party/oneDNN/include)
include_directories(${PROJECT_BINARY_DIR}/third_party/oneDNN/include)

set(OBJ_LIB ${LIB_NAME}_backend_dnnl)
add_library(${OBJ_LIB} OBJECT ${HEADERS} ${SOURCES})

set_target_properties(${OBJ_LIB} 
    PROPERTIES 
    POSITION_INDEPENDENT_CODE ON 
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN 1)

set_property(GLOBAL APPEND PROPERTY ${LIB_NAME}_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
