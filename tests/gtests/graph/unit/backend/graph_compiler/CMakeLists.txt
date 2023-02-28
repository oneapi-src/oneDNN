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

if(NOT ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND)
    return()
endif()

get_property(DEFINITIONS GLOBAL PROPERTY GRAPH_COMPILER_DEFINITIONS)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    # inherits via dominance
    append(CMAKE_CCXX_NOWARN_FLAGS "/wd4250")
endif()

append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

add_definitions(${DEFINITIONS})
set(OBJ_LIB graph_unit_test_compiler_backend)

file(GLOB_RECURSE TEST_SRC
${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_library(${OBJ_LIB} OBJECT ${TEST_SRC})

target_include_directories(${OBJ_LIB}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/tests/gtests/graph # gtest related headers
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/graph/backend/graph_compiler/core/src # for context
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/reference
    ${CMAKE_SOURCE_DIR}/src/cpu/x64
    )

set_property(GLOBAL APPEND PROPERTY GRAPH_UNIT_TEST_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
