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
set(OBJ_LIB graph_unit_test_interface)

add_library(${OBJ_LIB} OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/test_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_partition_hashing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_shape_infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_def_constraint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_value.cpp
)

set(sycl_rt_pattern "(SYCL|DPCPP)")
if(DNNL_CPU_RUNTIME MATCHES ${sycl_rt_pattern} OR DNNL_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_subdirectory(sycl)
endif()

set_property(GLOBAL APPEND PROPERTY GRAPH_UNIT_TEST_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)