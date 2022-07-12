/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example cpu_graph_serialization.cpp
/// @copybrief cpu_graph_serialization
/// > Annotated version: @ref cpu_graph_serialization

/// @page cpu_graph_serialization CPU example for graph serialization
///
/// > Example code: @ref cpu_graph_serialization.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/deserialize.hpp"
#include "common/execution_context.hpp"
#include "common/helpers_any_layout.hpp"
#include "common/utils.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    std::cout << "========Example: Graph Serialization===========\n";

    const engine::kind ekind = engine::kind::cpu;
    const graph::fpmath_mode fpmath_mode = graph::fpmath_mode::any;

    /// create a graph
    graph g(ekind, fpmath_mode);

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input0_dims {1, 64};
    std::vector<int64_t> input1_dims {64, 1};
    std::vector<int64_t> dst_dims {1, 1};

    auto &id_mgr = logical_id_manager::get();

    logical_tensor matmul_input0_desc {id_mgr["matmul_input0"], data_type::f32,
            input0_dims, layout_type::strided};
    logical_tensor matmul_input1_desc {id_mgr["matmul_input1"], data_type::f32,
            input1_dims, layout_type::strided};
    logical_tensor matmul_dst_desc {id_mgr["matmul_dst"], data_type::f32,
            dst_dims, layout_type::strided};

    op wildcard {id_mgr["wildcard"], op::kind::Wildcard, {},
            {matmul_input0_desc, matmul_input1_desc}, "wildcard"};

    op matmul {id_mgr["matmul"], op::kind::MatMul,
            {matmul_input0_desc, matmul_input1_desc}, {matmul_dst_desc},
            "matmul"};

    logical_tensor relu_dst_desc {
            id_mgr["relu_dst"], data_type::f32, dst_dims, layout_type::strided};

    op relu {id_mgr["relu"], op::kind::ReLU, {matmul_dst_desc}, {relu_dst_desc},
            "relu"};
    std::cout << "Success!\n";

    std::unordered_map<size_t, op::kind> op_id_kind_map {
            {id_mgr["wildcard"], op::kind::Wildcard},
            {id_mgr["matmul"], op::kind::MatMul},
            {id_mgr["relu"], op::kind::ReLU}};

    /// Add OP
    std::cout << "Add op to graph--------------------------------";
    g.add_op(wildcard);
    g.add_op(matmul);
    g.add_op(relu);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    const int len = 64;
    char original_str[len]; // NOLINT
    getenv("ONEDNN_GRAPH_DUMP", original_str, len);
    custom_setenv("ONEDNN_GRAPH_DUMP", "graph", 1);
    /// The graph will be partitioned into 1 partitions: `matmul + relu`
    auto partitions = g.get_partitions();
    custom_setenv("ONEDNN_GRAPH_DUMP", original_str, 1);

    /// mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    set_any_layout(partitions, id_to_set_any_layout);

    /// construct a new engine
    engine e {ekind, 0};

    /// construct a new stream
    stream s {e};

    std::vector<compiled_partition> c_partitions(partitions.size());

    // mapping from id to tensors
    tensor_map tm;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    // Serialize Graph
    std::cout << "Serialize Graph--------------------------------";
    custom_setenv("ONEDNN_GRAPH_DUMP", "subgraph", 1);
    std::cout << "Success!\n";

    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::cout << "\nPartition[" << partitions[i].get_id()
                      << "] is being processed.\n";
            std::vector<logical_tensor> inputs = partitions[i].get_in_ports();
            std::vector<logical_tensor> outputs = partitions[i].get_out_ports();

            /// replace input logical tensor with the queried one
            replace_with_queried_logical_tensors(
                    inputs, id_to_queried_logical_tensors);

            /// update output logical tensors with ANY layout
            update_tensors_with_any_layout(outputs, id_to_set_any_layout);

            std::cout << "Compiling--------------------------------------";
            /// compile to generate compiled partition
            c_partitions[i] = partitions[i].compile(inputs, outputs, e);
            std::cout << "Success!\n";

            record_queried_logical_tensors(partitions[i].get_out_ports(),
                    c_partitions[i], id_to_queried_logical_tensors);

            std::cout << "Creating tensors and allocating memory buffer--";
            std::vector<tensor> input_ts = tm.construct_and_initialize_tensors(
                    inputs, c_partitions[i], e, 1);
            std::vector<tensor> output_ts = tm.construct_and_initialize_tensors(
                    outputs, c_partitions[i], e, 0);
            std::cout << "Success!\n";

            std::cout << "Executing compiled partition-------------------";
            /// execute the compiled partition
            c_partitions[i].execute(s, input_ts, output_ts);
            std::cout << "Success!\n";
        }
    }
    custom_setenv("ONEDNN_GRAPH_DUMP", original_str, 1);

    // Step 6: Check correctness of the output results
    std::cout << "Check correctness------------------------------";
    float expected_result = 64.0;
    float *actual_output_ptr
            = tm.get(relu_dst_desc.get_id()).get_data_handle<float>();
    auto output_dims = relu_dst_desc.get_dims();
    auto num_elem = product(output_dims);
    std::vector<float> expected_output(num_elem, expected_result);
    compare_data(expected_output.data(), actual_output_ptr, num_elem);
    std::cout << "Success!\n";

    // Step 7: Deserialize Graph
    std::cout << "Deserialize Graph------------------------------";
    deserialized_graph dg;
    auto g2 = dg.load("./graph-100003-17603300391015650701.json");

    // Step 8: Filter partitions
    std::cout << "Filter partitions of deserialized graph--------";
    auto partitions2 = g2.get_partitions(partition::policy::fusion);
    std::cout << "Success!\n";

    std::cout << "Number of returned partitions: " << partitions2.size()
              << "\n";
    if (partitions2.size() != 1) {
        throw std::runtime_error(
                "wrong partition size for deserialized graph.\n");
    }

    /// mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout2;
    set_any_layout(partitions, id_to_set_any_layout2);
    std::vector<compiled_partition> c_partitions2(partitions2.size());

    // mapping from id to tensors
    tensor_map tm2;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors2;

    for (size_t i = 0; i < partitions2.size(); ++i) {
        if (partitions2[i].is_supported()) {
            std::cout << "\nPartition[" << partitions2[i].get_id()
                      << "] is being processed.\n";
            std::vector<logical_tensor> inputs = partitions2[i].get_in_ports();
            std::vector<logical_tensor> outputs
                    = partitions2[i].get_out_ports();

            /// replace input logical tensor with the queried one
            replace_with_queried_logical_tensors(
                    inputs, id_to_queried_logical_tensors2);

            /// update output logical tensors with ANY layout
            update_tensors_with_any_layout(outputs, id_to_set_any_layout2);

            std::cout << "Compiling--------------------------------------";
            /// compile to generate compiled partition
            c_partitions2[i] = partitions2[i].compile(inputs, outputs, e);
            std::cout << "Success!\n";

            record_queried_logical_tensors(partitions2[i].get_out_ports(),
                    c_partitions2[i], id_to_queried_logical_tensors2);

            std::cout << "Creating tensors and allocating memory buffer--";
            std::vector<tensor> input_ts = tm2.construct_and_initialize_tensors(
                    inputs, c_partitions2[i], e, 1);
            std::vector<tensor> output_ts
                    = tm2.construct_and_initialize_tensors(
                            outputs, c_partitions2[i], e, 0);
            std::cout << "Success!\n";

            std::cout << "Executing compiled partition-------------------";
            /// execute the compiled partition
            c_partitions2[i].execute(s, input_ts, output_ts);
            std::cout << "Success!\n";
        }
    }

    // Step 9: Check correctness of the output results
    std::cout << "Check correctness of deserialized graph--------";
    float *actual_output_ptr2
            = tm2.get(relu_dst_desc.get_id()).get_data_handle<float>();
    compare_data(expected_output.data(), actual_output_ptr2, num_elem);
    std::cout << "Success!\n";
    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
