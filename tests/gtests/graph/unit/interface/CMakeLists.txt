#===============================================================================
# Copyright 2021-2024 Intel Corporation
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
set(OBJ_LIB graph_unit_test_interface)

file(GLOB TEST_INTERFACE_ENGINE_DEPENDENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_partition_hashing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp
)

file(GLOB TEST_INTERFACE_ENGINE_INDEPENDENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_backend_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logical_tensor_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_schema_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_shape_infer_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_def_constraint_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_value_cpu.cpp
)

add_library(${OBJ_LIB} OBJECT ${TEST_INTERFACE_ENGINE_DEPENDENT_SOURCES} ${TEST_INTERFACE_ENGINE_INDEPENDENT_SOURCES})

set(sycl_rt_pattern "(SYCL|DPCPP)")
if(DNNL_CPU_RUNTIME MATCHES ${sycl_rt_pattern} OR DNNL_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_subdirectory(sycl)
endif()

set_property(GLOBAL APPEND PROPERTY GRAPH_UNIT_TEST_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

foreach(TEST_FILE ${TEST_INTERFACE_ENGINE_DEPENDENT_SOURCES})
    get_filename_component(file_name ${TEST_FILE} NAME_WE)
    string(REPLACE "test_" "test_graph_unit_interface_" test_suite_name ${file_name})
    string(REPLACE "test_" "test_interface_" filter ${file_name})
    register_graph_test_suite(${test_suite_name} "${filter}.*:${filter}/*")
endforeach()

foreach(TEST_FILE ${TEST_INTERFACE_ENGINE_INDEPENDENT_SOURCES})
    get_filename_component(file_name ${TEST_FILE} NAME_WE)
    string(REPLACE "_cpu" "" file_name ${file_name})
    string(REPLACE "test_" "test_graph_unit_interface_" test_suite_name ${file_name})
    string(REPLACE "test_" "test_interface_" filter ${file_name})
    register_graph_test_suite(${test_suite_name}_cpu "${filter}.*:${filter}/*")
endforeach()
