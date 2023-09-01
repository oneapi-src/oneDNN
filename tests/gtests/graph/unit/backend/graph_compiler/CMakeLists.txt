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
    get_property(CCXX_NOWARN_FLAGS GLOBAL PROPERTY GRAPH_COMPILER_CCXX_NOWARN_FLAGS)
    append(CMAKE_CCXX_NOWARN_FLAGS ${CCXX_NOWARN_FLAGS})
endif()

if(CMAKE_BASE_NAME MATCHES "(icx|icpx)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-function")
    append_host_compiler_options(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-function")
endif()

append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append_host_compiler_options(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

add_definitions_with_host_compiler(${DEFINITIONS})

file(GLOB_RECURSE GRAPH_COMPILER_CORE_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp
)

file(GLOB GRAPH_COMPILER_BRIDGE_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_backend_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compile_execute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pattern.cpp
)

get_property(SC_BUILTIN_ENABLED GLOBAL PROPERTY GRAPH_COMPILER_BUILTIN_ENABLED)
if(NOT SC_BUILTIN_ENABLED)
    list(REMOVE_ITEM GRAPH_COMPILER_CORE_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/core/test_intrisics_combine.cpp)
endif()

include_directories_with_host_compiler(
    ${PROJECT_SOURCE_DIR}/tests/gtests/graph # gtest related headers
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/graph/backend/graph_compiler/core/src # for context
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/reference
    ${PROJECT_SOURCE_DIR}/src/cpu/x64
)

set(TEST_EXE test_unit_gc_core)
register_graph_unit_exe(${TEST_EXE} "${GRAPH_COMPILER_CORE_TEST_SOURCES}" "")

foreach(TEST_FILE ${GRAPH_COMPILER_BRIDGE_TEST_SOURCES})
    get_filename_component(exe ${TEST_FILE} NAME_WE)
    register_graph_unit_exe(${exe} ${TEST_FILE} "unit_gc_")
endforeach()
