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

include_directories_with_host_compiler(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tests/gtests/graph
    ${PROJECT_SOURCE_DIR}/tests/gtests/gtest
    ${PROJECT_SOURCE_DIR}/include
)

FILE(GLOB COMMON_API_TEST_DEPS
    ${CMAKE_CURRENT_SOURCE_DIR}/api_test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_api_common.cpp
    ${TEST_ALLOCATOR}
)

function(register_graph_api_exe exe_name srcs)
    string(REPLACE "test_" "test_graph_" exe_new_name ${exe_name})
    list(APPEND srcs ${COMMON_API_TEST_DEPS})
    set(DEPS 
        ${LIB_PACKAGE_NAME}
        ${EXTRA_SHARED_LIBS}
        dnnl_gtest
    )
    register_graph_exe(${exe_new_name} "${srcs}" "${DEPS}")
endfunction()

# engine indepnendent test
FILE(GLOB API_TEST_ENGINE_INDEPENDENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_add_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_constant_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_tensor.cpp
)

FILE(GLOB API_TEST_ENGINE_DEPENDENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_compile_parametrized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_partition.cpp
)


foreach(TEST_FILE ${API_TEST_ENGINE_INDEPENDENT_SOURCES})
    get_filename_component(exe ${TEST_FILE} NAME_WE)
    register_graph_api_exe(${exe} ${TEST_FILE})
endforeach()

foreach(TEST_FILE ${API_TEST_ENGINE_DEPENDENT_SOURCES})
    get_filename_component(exe ${TEST_FILE} NAME_WE)
    # add suffix _usm to test both CPU and GPU
    register_graph_api_exe(${exe}_usm ${TEST_FILE})
endforeach()

set(sycl_rt_pattern "(SYCL|DPCPP)")
if(DNNL_CPU_RUNTIME MATCHES ${sycl_rt_pattern} OR DNNL_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_subdirectory(sycl)
endif()
