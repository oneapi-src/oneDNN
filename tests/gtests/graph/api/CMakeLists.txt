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
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_engine.cpp
    ${TEST_ALLOCATOR}
    ${TEST_THREAD}
)

set(sycl_rt_pattern "(SYCL|DPCPP)")
if(DNNL_CPU_RUNTIME MATCHES ${sycl_rt_pattern} OR DNNL_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_subdirectory(sycl)
endif()

get_property(API_TEST_DEPS GLOBAL PROPERTY GRAPH_API_TEST_DEPS)

add_executable(test_graph_api ${API_TEST_SRCS} ${API_TEST_DEPS})

target_link_libraries(test_graph_api
    dnnl_gtest
    ${DNNL_LIBRARY_NAME}
    ${EXTRA_SHARED_LIBS}
    )

include_directories_with_host_compiler(
    ${PROJECT_SOURCE_DIR}/tests/gtests/graph
    ${PROJECT_SOURCE_DIR}/tests/gtests/gtest
    ${PROJECT_SOURCE_DIR}/include
    )

# Add CPU test if CPU runtime is enabled
if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    add_test("test_graph_api_cpu" "test_graph_api" "--engine=cpu")
    maybe_configure_windows_test("test_graph_api_cpu" TEST)
endif()

# Add GPU test if GPU runtime is enabled
if(DNNL_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_test("test_graph_api_gpu" "test_graph_api" "--engine=gpu")
    maybe_configure_windows_test("test_graph_api_gpu" TEST)
endif()
