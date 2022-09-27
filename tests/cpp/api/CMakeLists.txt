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

set(API_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/api_test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_api_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_add_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_compile_parametrized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_constant_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_version.cpp
    ${TEST_ALLOCATOR}
)

set(sycl_rt_pattern "(DPCPP)")

if(DNNL_GRAPH_CPU_RUNTIME MATCHES ${sycl_rt_pattern} OR DNNL_GRAPH_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_subdirectory(sycl)
endif()

get_property(API_TEST_DEPS GLOBAL PROPERTY ${LIB_NAME}_API_TEST_DEPS)

add_executable(${LIB_NAME}_api_test ${API_TEST_SRCS} ${API_TEST_DEPS})

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
    ${PROJECT_SOURCE_DIR}/tests # gtest related headers
    ${PROJECT_SOURCE_DIR}/include
)

set(TEST_FILES_NEED_VERSION ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_version.cpp)
APPLY_VERSION_DEFINITIONS("${TEST_FILES_NEED_VERSION}")

# Add CPU test if CPU runtime is enabled
if(NOT DNNL_GRAPH_CPU_RUNTIME STREQUAL "NONE")
    add_test("${LIB_NAME}_api_test" "${LIB_NAME}_api_test" "--engine=cpu")
    maybe_configure_windows_test("${LIB_NAME}_api_test" TEST)
endif()

# Add GPU test if GPU runtime is enabled
if(DNNL_GRAPH_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_test("${LIB_NAME}_api_test_gpu" "${LIB_NAME}_api_test" "--engine=gpu"
        "--gtest_filter=CAPI.Compile*:*Test_CompileConv*:*Test_Max_Pool_Compile*:*Test_Average_Pool_Compile*")
    maybe_configure_windows_test("${LIB_NAME}_api_test_gpu" TEST)
endif()
