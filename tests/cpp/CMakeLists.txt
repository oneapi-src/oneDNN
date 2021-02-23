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

if(NOT DNNL_GRAPH_BUILD_TESTS)
    return()
endif()

set(gtest_hide_internal_symbols ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/tests/gtest
    ${CMAKE_BINARY_DIR}/tests/gtest)

enable_testing()

append(CMAKE_CXX_FLAGS ${DNNL_GRAPH_COMMON_COMPILATION_FLAGS})
append(CMAKE_C_FLAGS ${DNNL_GRAPH_COMMON_COMPILATION_FLAGS})

if(UNIX)
    if(APPLE)
        # TODO(wuxun): now it will fail to build with macOS in Debug mode, need
        # to enable again once the root cause is found
        #set(DNNL_GRAPH_TEST_LINK_FLAGS "-Wl,-fatal_warnings")
    else()
        set(DNNL_GRAPH_TEST_LINK_FLAGS "-Wl,--fatal-warnings")
    endif()
endif()

set(DNNL_GRAPH_API_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_add_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_compile_parametrized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_api_common.cpp
    )
add_executable(${LIB_NAME}_api_test ${DNNL_GRAPH_API_TEST_SRCS})
set_target_properties(${LIB_NAME}_api_test PROPERTIES CXX_VISIBILITY_PRESET "hidden")
set_target_properties(${LIB_NAME}_api_test PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
target_link_libraries(${LIB_NAME}_api_test
    LINK_PRIVATE
    ${DNNL_GRAPH_TEST_LINK_FLAGS}
    ${LIB_NAME}_gtest
    ${LIB_NAME} # Use shared to prevent accidental access to private symbols
    ${DNNL_GRAPH_SYCL_LINK_FLAGS}
    ${DNNL_GRAPH_COMMON_LINK_FLAGS} # FIXME: consider target_link_options from cmake 3.13
    )
target_include_directories(${LIB_NAME}_api_test
    PRIVATE
    ${PROJECT_BINARY_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/tests # gtest related headers
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    )
set(TEST_FILES_NEED_VERSION ${CMAKE_CURRENT_SOURCE_DIR}/api/test_c_api_version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/api/test_cpp_api_version.cpp)
APPLY_VERSION_DEFINITIONS("${TEST_FILES_NEED_VERSION}")

# These files access internal api and needs visibility options set to avoid warnings on macos
add_library(${LIB_NAME}_unit_test_objlib OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/unit_test_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_attribute_value.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_dnnl_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_layout_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_op_schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_value.cpp
    )
set_target_properties(${LIB_NAME}_unit_test_objlib PROPERTIES CXX_VISIBILITY_PRESET "hidden")
set_target_properties(${LIB_NAME}_unit_test_objlib PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
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
    dl
    ${DNNL_GRAPH_COMMON_LINK_FLAGS} # FIXME: consider target_link_options from cmake 3.13
    )

add_test("${LIB_NAME}_api_test" "${LIB_NAME}_api_test")
add_test("${LIB_NAME}_unit_test" "${LIB_NAME}_unit_test")
