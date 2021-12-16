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

if(NOT DNNL_GRAPH_BUILD_COMPILER_BACKEND)
    return()
endif()

set(OBJ_LIB ${LIB_NAME}_backend_compiler_unit_test)

add_library(${OBJ_LIB} OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/test_backend_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compile_execute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pattern.cpp
    )

target_include_directories(${OBJ_LIB}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/tests # gtest related headers
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/backend/graph_compiler/core/src # for context
    )

set_property(GLOBAL APPEND PROPERTY ${LIB_NAME}_UNIT_TEST_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
