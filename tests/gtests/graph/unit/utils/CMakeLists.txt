#===============================================================================
# Copyright 2022-2023 Intel Corporation
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

set(TEST_EXE test_unit_utils)

file(GLOB TEST_UTILS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_attribute_value.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pattern_matcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_verbose.cpp
)

if(DNNL_USE_CLANG_SANITIZER)
    # Due to the following tests are testing out-of-range enum values.
    list(REMOVE_ITEM TEST_UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_debug.cpp)
endif()

register_graph_unit_exe(${TEST_EXE} "${TEST_UTILS_SOURCES}" "")
