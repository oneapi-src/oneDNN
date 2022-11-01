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

set(OBJ_LIB ${LIB_NAME}_backend_dnnl_unit_test)

add_library(${OBJ_LIB} OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/test_batch_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_binary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_constant_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_convtranspose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dequantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dnnl_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dnnl_infer_shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dnnl_partition_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_eltwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fusion_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_insert_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_internal_attrs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_large_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layer_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_propagator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_memory_planning.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_executable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_prelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scratchpad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_subgraph_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_thread_local_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_typecast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp
    ${GRAPH_TEST_THREAD}
    ${TEST_ALLOCATOR}
)

target_include_directories(${OBJ_LIB}
    PRIVATE
    ${PROJECT_BINARY_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/src/
    )
    
set_property(GLOBAL APPEND PROPERTY ${LIB_NAME}_UNIT_TEST_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
