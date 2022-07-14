/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

/// @example cpu_programming_aot.cpp
/// @copybrief cpu_programming_aot_cpp
/// > Annotated version: @ref cpu_programming_aot_cpp

/// @page cpu_programming_aot_cpp Example for demonstrating programming model with ahead-of-time compilation
///
/// > Example code: @ref cpu_programming_aot.cpp
///
/// This example will construct the below graph. The graph has two outputs which
/// are connected to End op. Now, Conv and Add ops should not be fused due to
/// tensor1 is also used as an output of the graph.
///
///         Conv     Wildcard
///           |         |
///        tensor1   tensor2
///       /      \     /
///     End        Add
///                 |
///              tensor3
///                 |
///                ReLU
///                 |
///              tensor4
///                 |
///                End
///

#include <cassert>
#include <iostream>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/execution_context.hpp"
#include "common/utils.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

// clang-format off
int cpu_programming_aot_tutorial(engine::kind engine_kind) {
    /// construct a graph based on the given engine kind
    graph g(engine_kind);

    std::vector<int64_t> input_dims {8, 3, 227, 227};
    std::vector<int64_t> weight_dims {96, 3, 11, 11};
    std::vector<int64_t> bias_dims {96};
    std::vector<int64_t> dst_dims {8, 96, 217, 217};
    std::vector<int64_t> add_input_2_dims {8, 96, 217, 217};

    logical_id_manager &id_mgr = logical_id_manager::get();

    /// each logical tensor should be given with unique name
    /// the name is 1:1 mapping with the given id
    logical_tensor conv_data_lt {id_mgr["conv_data"], data_type::f32, input_dims, layout_type::strided};
    logical_tensor conv_weight_lt {id_mgr["conv_weight"], data_type::f32, weight_dims, layout_type::strided};
    logical_tensor conv_bias_lt {id_mgr["conv_bias"], data_type::f32, bias_dims, layout_type::strided};
    logical_tensor conv_dst_lt {id_mgr["dst_dims"], data_type::f32, dst_dims, layout_type::strided};

    op conv {0, op::kind::Convolution, {conv_data_lt, conv_weight_lt, conv_bias_lt}, {conv_dst_lt}, "conv_0"};
    conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv.set_attr<int64_t>("groups", 1);
    conv.set_attr<std::string>("data_format", "NCX");
    conv.set_attr<std::string>("filter_format", "OIX");

    logical_tensor add_input_2_lt {id_mgr["add_input_2"], data_type::f32, add_input_2_dims, layout_type::strided};
    logical_tensor add_dst_lt {id_mgr["add_dst"], data_type::f32, dst_dims, layout_type::strided};

    op add {1, op::kind::Add, {conv_dst_lt, add_input_2_lt}, {add_dst_lt}, "add_0"};

    logical_tensor relu_dst_lt {id_mgr["relu_dst"], data_type::f32, dst_dims, layout_type::strided};

    op relu {2, op::kind::ReLU, {add_dst_lt}, {relu_dst_lt}, "relu_0"};

    op wildcard {3, op::kind::Wildcard, {}, {add_input_2_lt}, "wildcard_0"};

    op end_0 {4, op::kind::End, {conv_dst_lt}, {}, "end_0"};
    op end_1 {5, op::kind::End, {relu_dst_lt}, {}, "end_1"};

    /// mapping from op id to op kind
    std::unordered_map<size_t, op::kind> op_id_kind_map {{0, op::kind::Convolution},
        {1, op::kind::Add}, {2, op::kind::ReLU}, {3, op::kind::Wildcard}, {4, op::kind::End}, {5, op::kind::End}};

    /// add op to graph
    g.add_op(conv);
    g.add_op(add);
    g.add_op(relu);
    g.add_op(wildcard);
    g.add_op(end_0);
    g.add_op(end_1);

    /// get partitions from the graph
    std::vector<partition> partitions = g.get_partitions();

    std::cout << "Number of returned partitions: " << partitions.size() << "\n";
    for (size_t i = 0; i < partitions.size(); ++i) {
        std::cout << "Partition[" << partitions[i].get_id()
                  << "]'s supporting status: "
                  << (partitions[i].is_supported() ? "true" : "false") << "\n";
    }

    /// construct a new engine
    engine e {engine_kind, 0};

    /// construct a new stream
    stream s {e};

    std::vector<compiled_partition> c_partitions(partitions.size());
    /// compilation loop
    for (size_t i = 0; i < partitions.size(); ++i) {
        /// just skip compilation if this partition is not supported by oneDNN Graph backend
        if (partitions[i].is_supported()) {
            std::vector<logical_tensor> inputs = partitions[i].get_in_ports();
            std::vector<logical_tensor> outputs = partitions[i].get_out_ports();
            std::cout << "Compiling partition[" << partitions[i].get_id() << "]--------------------";
            /// compile to generate compiled partition
            c_partitions[i] = partitions[i].compile(inputs, outputs, e);
            std::cout << "Success!\n";
        }
    }

    // mapping from id to tensors
    tensor_map tm;

    /// execution loop
    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::cout << "\nPartition[" << partitions[i].get_id() << "] is being executed.\n";
            std::cout << "Creating tensors and allocating memory buffer--";
            std::vector<tensor> input_ts = tm.construct_and_initialize_tensors(partitions[i].get_in_ports(), c_partitions[i], e, 1);
            std::vector<tensor> output_ts = tm.construct_and_initialize_tensors(partitions[i].get_out_ports(), c_partitions[i], e, 0);
            std::cout << "Success!\n";

            std::cout << "Executing compiled partition-------------------";
            /// execute the compiled partition
            c_partitions[i].execute(s, input_ts, output_ts);
            std::cout << "Success!\n";
        } else {
            std::vector<size_t> unsupported_op_ids = partitions[i].get_ops();
            assertm(unsupported_op_ids.size() == 1, "Unsupported partition only "
                "contains single op.");
            if (op_id_kind_map[unsupported_op_ids[0]] == op::kind::Wildcard) {
                std::cout << "\nWarning (actually an error): partition " << partitions[i].get_id() <<
                        " contains only a Wildcard op which cannot be computed.\n";
            } else {
                /// Users need to write implementation code by themselves.
                continue;
            }
        }
    }

    /// check correctness
    float expected_result_1 = /*weight = */11 * 11 * /*channel = */3 + /*bias = */ + /*bias = */1.0f;
    float expected_result_2 = (/*weight = */11 * 11 * /*channel = */3 + /*bias = */ + /*bias = */1.0f) + /*add*/1.0f;

    if (partitions.size() == 6) {
        void *actual_output_ptr1 = tm.get(conv_dst_lt.get_id()).get_data_handle();
        auto output_dims = conv_dst_lt.get_dims();
        auto num_elem = product(output_dims);
        std::vector<float> expected_output1(num_elem, expected_result_1);
        compare_data(expected_output1.data(), reinterpret_cast<float *>(actual_output_ptr1),
                num_elem, (float)1e-5, (float)1e-6);
    }

    void *actual_output_ptr2 = tm.get(relu_dst_lt.get_id()).get_data_handle();
    auto output_dims = relu_dst_lt.get_dims();
    auto num_elem = product(output_dims);
    std::vector<float> expected_output2(num_elem, expected_result_2);
    compare_data(expected_output2.data(), reinterpret_cast<float *>(actual_output_ptr2), num_elem);
    std::cout << "Example passed successfully!\n";
    return 0;
}

// clang-format on
int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    return cpu_programming_aot_tutorial(engine_kind);
}
