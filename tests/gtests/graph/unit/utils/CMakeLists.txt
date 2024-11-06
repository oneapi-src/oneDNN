#===============================================================================
# Copyright 2022-2024 Intel Corporation
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

set(OBJ_LIB graph_unit_test_utils)

file(GLOB TEST_UTILS_ENGINE_DEPENDENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_allocator.cpp
)

file(GLOB TEST_UTILS_ENGINE_INDEPENDENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_attribute_value_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_debug_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_json_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pattern_matcher_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_cpu.cpp
)

if(DNNL_USE_CLANG_SANITIZER)
    # Due to the following tests are testing out-of-range enum values.
    list(REMOVE_ITEM TEST_UTILS_ENGINE_INDEPENDENT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_debug_cpu.cpp)
endif()

add_library(${OBJ_LIB} OBJECT ${TEST_UTILS_ENGINE_DEPENDENT_SOURCES} ${TEST_UTILS_ENGINE_INDEPENDENT_SOURCES})

set_property(GLOBAL APPEND PROPERTY GRAPH_UNIT_TEST_DEPS $<TARGET_OBJECTS:${OBJ_LIB}>)

foreach(TEST_FILE ${TEST_UTILS_ENGINE_DEPENDENT_SOURCES})
    get_filename_component(file_name ${TEST_FILE} NAME_WE)
    string(REPLACE "test_" "test_graph_unit_utils_" test_suite_name ${file_name})
    string(REPLACE "test_" "test_utils_" filter ${file_name})
    register_graph_test_suite(${test_suite_name} "${filter}*")
endforeach()

foreach(TEST_FILE ${TEST_UTILS_ENGINE_INDEPENDENT_SOURCES})
    get_filename_component(file_name ${TEST_FILE} NAME_WE)
    string(REPLACE "_cpu" "" file_name ${file_name})
    string(REPLACE "test_" "test_graph_unit_utils_" test_suite_name ${file_name})
    string(REPLACE "test_" "test_utils_" filter ${file_name})
    register_graph_test_suite(${test_suite_name}_cpu "${filter}*")
endforeach()
