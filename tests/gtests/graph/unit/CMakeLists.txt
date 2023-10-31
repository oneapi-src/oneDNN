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
set(BINARY_NAME test_graph_unit)

function(register_graph_test_suite test_suite_name filter)
    set_property(GLOBAL APPEND PROPERTY GRAPH_UNIT_TEST_SUITES ${test_suite_name})
    set_property(GLOBAL APPEND PROPERTY GRAPH_UNIT_TEST_FILTERS ${filter})
endfunction()

include_directories_with_host_compiler(
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/src/graph
    ${PROJECT_SOURCE_DIR}/include)

file(GLOB COMMON_UNIT_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

list(APPEND COMMON_UNIT_TESTS ${TEST_THREAD})
list(APPEND COMMON_UNIT_TESTS ${TEST_ALLOCATOR})

add_subdirectory(interface)
add_subdirectory(backend)
add_subdirectory(utils)

get_property(UNIT_TEST_DEPS GLOBAL PROPERTY GRAPH_UNIT_TEST_DEPS)
get_property(LIB_DEPS GLOBAL PROPERTY DNNL_LIB_DEPS)

add_executable(${BINARY_NAME} ${COMMON_UNIT_TESTS} ${UNIT_TEST_DEPS} ${LIB_DEPS})

# Set path to non exported headers
include_directories_with_host_compiler(
    ${PROJECT_SOURCE_DIR}/tests/gtests/graph
    ${PROJECT_SOURCE_DIR}/tests/gtests/gtest
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/graph
    )

if(DNNL_ENABLE_JIT_PROFILING OR DNNL_ENABLE_ITT_TASKS)
    if (UNIX AND NOT APPLE)
        # Not every compiler adds -ldl and -lrt automatically
        list(APPEND EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
        list(APPEND EXTRA_SHARED_LIBS rt)
    endif()
endif()

get_property(STATIC_LIB_DEPS GLOBAL PROPERTY DNNL_SUBDIR_EXTRA_STATIC_LIBS)
get_property(SHARED_LIB_DEPS GLOBAL PROPERTY DNNL_SUBDIR_EXTRA_SHARED_LIBS)

target_link_libraries(${BINARY_NAME}
    dnnl_gtest
    ${EXTRA_SHARED_LIBS}
    ${EXTRA_STATIC_LIBS}
    ${STATIC_LIB_DEPS}
    ${SHARED_LIB_DEPS}
)

get_property(test_suite_names GLOBAL PROPERTY GRAPH_UNIT_TEST_SUITES)
get_property(test_filters GLOBAL PROPERTY GRAPH_UNIT_TEST_FILTERS)
list(LENGTH test_suite_names count)
math(EXPR count "${count}-1")
set(sycl_rt_pattern "(SYCL|DPCPP)")
foreach(i RANGE ${count})
    list(GET test_suite_names ${i} s)
    list(GET test_filters ${i} f)
    if(${s} MATCHES "_usm")
        if(DNNL_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
            add_test(${s}_gpu ${BINARY_NAME} --gtest_filter=${f} --engine=gpu)
            maybe_configure_windows_test(${s}_gpu TEST)
        endif()

        if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
            add_test(${s}_cpu ${BINARY_NAME} --gtest_filter=${f} --engine=cpu)
            maybe_configure_windows_test(${s}_cpu TEST)
        endif()
    else()
        if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
            add_test(${s} ${BINARY_NAME} --gtest_filter=${f})
            maybe_configure_windows_test(${s} TEST)
        endif()
    endif()
endforeach()
