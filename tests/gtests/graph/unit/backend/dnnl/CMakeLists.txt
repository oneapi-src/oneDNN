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

# make gtest and library same definition for subgraph_visualizer_t
if(ONEDNN_ENABLE_GRAPH_DUMP)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_GRAPH_DUMP)
endif()

FILE(GLOB DNNL_COMMON_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_constant_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dnnl_infer_shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dnnl_partition_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fusion_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_insert_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_internal_attrs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_propagator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_memory_planning.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_executable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scratchpad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_subgraph_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_thread_local_cache.cpp
)

if(_ONEDNN_GRAPH_LAYOUT_DEBUG)
    file(GLOB FILES_REQUIRED_LAYOUT_DEBUG
        ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_id.cpp
    )

    set_source_files_properties(${FILES_REQUIRED_LAYOUT_DEBUG}
        PROPERTIES COMPILE_FLAGS "-DDNNL_GRAPH_LAYOUT_DEBUG")
endif()

foreach(TEST_FILE ${DNNL_COMMON_TEST_SOURCES})
    get_filename_component(exe ${TEST_FILE} NAME_WE)
    register_graph_unit_exe(${exe} ${TEST_FILE} "unit_dnnl_")
endforeach()


# op compile/execution need test on both CPU and GPU
FILE(GLOB DNNL_OP_EXECUTION_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_batch_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_binary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_convtranspose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dequantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_eltwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_large_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layer_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_prelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_typecast.cpp
)

foreach(TEST_FILE ${DNNL_OP_EXECUTION_TEST_SOURCES})
    get_filename_component(exe ${TEST_FILE} NAME_WE)
    # add suffix _usm to test both CPU and GPU
    register_graph_unit_exe(${exe}_usm ${TEST_FILE} "unit_dnnl_")
endforeach()
