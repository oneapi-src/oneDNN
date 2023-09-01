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
set(TEST_EXE test_unit_interface)

file(GLOB SOURCES
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

register_graph_unit_exe(${TEST_EXE} "${SOURCES}" "")

if(DNNL_WITH_SYCL)
    add_subdirectory(sycl)
endif()
