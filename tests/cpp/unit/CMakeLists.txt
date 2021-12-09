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

include_directories(
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include)

file(GLOB COMMON_UNIT_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

add_subdirectory(interface)
add_subdirectory(backend)

get_property(UNIT_TEST_DEPS GLOBAL PROPERTY ${LIB_NAME}_UNIT_TEST_DEPS)
get_property(LIB_DEPS GLOBAL PROPERTY ${LIB_NAME}_LIB_DEPS)

add_executable(${LIB_NAME}_unit_test ${COMMON_UNIT_TESTS} ${UNIT_TEST_DEPS} ${LIB_DEPS})

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
    dnnl
    ${${LIB_NAME}_EXTRA_INTERFACE}
    ${DNNL_GRAPH_EXTRA_SHARED_LIBS}
    ${DNNL_GRAPH_COMMON_LINK_FLAGS} # FIXME: consider target_link_options from cmake 3.13
    )

set(sycl_rt_pattern "(DPCPP)")
add_test("${LIB_NAME}_unit_test" "${LIB_NAME}_unit_test" "--engine=cpu")
maybe_configure_windows_test("${LIB_NAME}_unit_test" TEST)

# Add GPU test if GPU runtime is enabled
if(DNNL_GRAPH_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_test("${LIB_NAME}_unit_test_gpu" "${LIB_NAME}_unit_test" "--engine=gpu"
        "--gtest_filter=Execute*:Compile.*:CompiledPartition.*:Scratchpad.*:\
SyclAllocatorGpu*:SyclEngineGpu*")
    maybe_configure_windows_test("${LIB_NAME}_unit_test_gpu" TEST)
endif()
