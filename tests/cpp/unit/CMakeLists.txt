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

# These files access internal api and needs visibility options set to avoid warnings on macos
add_library(${LIB_NAME}_unit_test_objlib OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_attribute_value.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dnnl_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_v2_patterns.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_value.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_subgraph_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scratchpad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_thread_local_cache.cpp
    )

# Set path to exported and non exported headers
target_include_directories(${LIB_NAME}_unit_test_objlib
    PRIVATE
    ${PROJECT_BINARY_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/tests # gtest related headers
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/include # Public but need to set manually for OBJECT target
    )

add_executable(${LIB_NAME}_unit_test $<TARGET_OBJECTS:${LIB_NAME}_unit_test_objlib>)
# Set path to non exported headers
target_include_directories(${LIB_NAME}_unit_test
    PRIVATE
    ${PROJECT_BINARY_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src/
    )
target_link_libraries(${LIB_NAME}_unit_test
    LINK_PRIVATE
    ${DNNL_GRAPH_TEST_LINK_FLAGS}
    ${LIB_NAME}_gtest
    ${LIB_NAME}_static # Use static to access private symbols not available in public API
    ${EXTRA_SHARED_LIBS}
    ${DNNL_GRAPH_COMMON_LINK_FLAGS} # FIXME: consider target_link_options from cmake 3.13
    )

add_test("${LIB_NAME}_unit_test" "${LIB_NAME}_unit_test")
maybe_configure_windows_test("${LIB_NAME}_unit_test" TEST)
