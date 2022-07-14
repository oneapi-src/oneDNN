/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/// @example cpu_conv_bn_add_relu_pattern.cpp
/// @copybrief cpu_conv_bn_add_relu_pattern_cpp
/// > Annotated version: @ref cpu_conv_bn_add_relu_pattern_cpp

/// @page cpu_conv_bn_add_relu_pattern_cpp CPU example for conv+bn+add+relu pattern
///
/// > Example code: @ref cpu_conv_bn_add_relu_pattern.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/execution_context.hpp"
#include "common/helpers_any_layout.hpp"
#include "common/utils.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

// digraph G {
// Wildcard_100002 -> Convolution_100003;
// Convolution_100003 -> BatchNormInference_100004;
// BatchNormInference_100004 -> ReLU_100005;
// ReLU_100005 -> Convolution_100006;
// Convolution_100006 -> BatchNormInference_100007;
// BatchNormInference_100007 -> Add_100008;
// Wildcard_100002 -> Add_100008;
// Add_100008 -> ReLU_100009;
// }
// clang-format off
int main(int argc, char **argv) {
    std::cout << "========Example: Conv+BN+ReLU+Conv+BN+Add+ReLU========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        std::cout << "Don't support gpu now\n";
        return -1;
    }

    // Step 2: Construct a graph
    graph g(engine_kind);

    auto &id_mgr = logical_id_manager::get();

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input_dims {8, 56, 56, 256};
    std::vector<int64_t> conv0_weight_dims {64, 256, 1, 1};
    std::vector<int64_t> conv0_bias_dims {64};
    std::vector<int64_t> conv0_dst_dims {8, 56, 56, 64};

    std::vector<int64_t> conv1_weight_dims {256, 64, 1, 1};
    std::vector<int64_t> conv1_bias_dims {256};
    std::vector<int64_t> conv1_dst_dims {8, 56, 56, 256};

    logical_tensor conv0_src_desc {id_mgr["conv0_src"], data_type::f32, input_dims, layout_type::strided};
    logical_tensor conv0_weight_desc {id_mgr["conv0_weight"], data_type::f32, conv0_weight_dims, layout_type::strided};
    logical_tensor conv0_dst_desc {id_mgr["conv0_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};
    
    /// inuput op
    op input0 {id_mgr["input0"], op::kind::Wildcard, {}, {conv0_src_desc}, "input0"};

    op conv0 {id_mgr["conv0"], op::kind::Convolution, {conv0_src_desc, conv0_weight_desc}, {conv0_dst_desc}, "conv0"};
    conv0.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NXC");
    conv0.set_attr<std::string>("filter_format", "OIX");
    conv0.set_attr<int64_t>("groups", 1);
    
    logical_tensor bn0_scale_desc {id_mgr["bn0_scale"], data_type::f32, conv0_bias_dims, layout_type::strided};
    logical_tensor bn0_shift_desc {id_mgr["bn0_shift"], data_type::f32, conv0_bias_dims, layout_type::strided};
    logical_tensor bn0_mean_desc {id_mgr["bn0_mean"], data_type::f32, conv0_bias_dims, layout_type::strided};
    logical_tensor bn0_var_desc {id_mgr["bn0_var"], data_type::f32, conv0_bias_dims, layout_type::strided};
    logical_tensor bn0_dst_desc {id_mgr["bn0_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};
    
    op bn0 {id_mgr["bn0"], op::kind::BatchNormInference, {conv0_dst_desc, bn0_scale_desc, bn0_shift_desc, bn0_mean_desc, bn0_var_desc}, {bn0_dst_desc}, "bn0"};
    bn0.set_attr<float>("epsilon", 0.f);
    bn0.set_attr<std::string>("data_format", "NXC");

    logical_tensor relu0_dst_desc {id_mgr["relu0_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};

    op relu0 {id_mgr["relu0"], op::kind::ReLU, {bn0_dst_desc}, {relu0_dst_desc}, "relu0"};

    logical_tensor conv1_weight_desc {id_mgr["conv1_weight"], data_type::f32, conv1_weight_dims, layout_type::strided};
    logical_tensor conv1_dst_desc {id_mgr["conv1_dst"], data_type::f32, conv1_dst_dims, layout_type::strided};
    
    op conv1 {id_mgr["conv1"], op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc}, {conv1_dst_desc}, "conv1"};
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<std::string>("data_format", "NXC");
    conv1.set_attr<std::string>("filter_format", "OIX");
    conv1.set_attr<int64_t>("groups", 1);

    logical_tensor bn1_scale_desc {id_mgr["bn1_scale"], data_type::f32, conv1_bias_dims, layout_type::strided};
    logical_tensor bn1_shift_desc {id_mgr["bn1_shift"], data_type::f32, conv1_bias_dims, layout_type::strided};
    logical_tensor bn1_mean_desc {id_mgr["bn1_mean"], data_type::f32, conv1_bias_dims, layout_type::strided};
    logical_tensor bn1_var_desc {id_mgr["bn1_var"], data_type::f32, conv1_bias_dims, layout_type::strided};
    logical_tensor bn1_dst_desc {id_mgr["bn1_dst"], data_type::f32, conv1_dst_dims, layout_type::strided};
    logical_tensor add0_src_desc {id_mgr["add0_src"], data_type::f32, input_dims, layout_type::strided};
    
    op bn1 {id_mgr["bn1"], op::kind::BatchNormInference, {conv1_dst_desc, bn1_scale_desc, bn1_shift_desc, bn1_mean_desc, bn1_var_desc}, {bn1_dst_desc}, "bn1"};
    bn1.set_attr<float>("epsilon", 0.f);
    bn1.set_attr<std::string>("data_format", "NXC");

    logical_tensor add0_dst_desc {id_mgr["add0_dst"], data_type::f32, conv1_dst_dims, layout_type::strided};

    op add0 {id_mgr["add0"], op::kind::Add, {bn1_dst_desc, add0_src_desc}, {add0_dst_desc}, "add0"};

    logical_tensor relu1_dst_desc {id_mgr["relu1_dst"], data_type::f32, conv1_dst_dims, layout_type::strided};
    
    op relu1 {id_mgr["relu1"], op::kind::ReLU, {add0_dst_desc}, {relu1_dst_desc}, "relu1"};

    op end {id_mgr["end"], op::kind::End, {relu1_dst_desc}, {}, "end"};
    std::cout << "Success!\n";

    std::unordered_map<size_t, op::kind> op_id_kind_map {{id_mgr["input0"], op::kind::Wildcard},
        {id_mgr["conv0"], op::kind::Convolution}, {id_mgr["bn0"], op::kind::BatchNormInference},
        {id_mgr["relu0"], op::kind::ReLU}, {id_mgr["conv1"], op::kind::Convolution}, {id_mgr["bn1"], op::kind::BatchNormInference},
        {id_mgr["add0"], op::kind::Add}, {id_mgr["relu1"], op::kind::ReLU}, {id_mgr["end"], op::kind::End}};

    /// Add OP
    std::cout << "Add OP to graph--------------------------------";
    g.add_op(input0);
    g.add_op(conv0);
    g.add_op(bn0);
    g.add_op(relu0);
    g.add_op(conv1);
    g.add_op(bn1);
    g.add_op(add0);
    g.add_op(relu1);
    g.add_op(end);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter and get partitions
    /// Graph will be filtered into 2 partitions: `conv0+bn0+relu0` and `conv1+bn1+add0+relu1`
    /// `export DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter and get partition-----------------------";
    auto partitions = g.get_partitions(partition::policy::fusion);
    std::cout << "Success!\n";

    std::cout << "Number of returned partitions: " << partitions.size() << "\n";
    for (size_t i = 0; i < partitions.size(); ++i) {
        std::cout << "Partition[" << partitions[i].get_id()
                  << "]'s supporting status: "
                  << (partitions[i].is_supported() ? "true" : "false") << "\n";
    }

    /// mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    set_any_layout(partitions, id_to_set_any_layout);

    /// construct a new engine
    engine e {engine_kind, 0};

    /// construct a new stream
    stream s {e};

    std::vector<compiled_partition> c_partitions(partitions.size());

    // mapping from id to tensors
    tensor_map tm;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::cout << "\nPartition[" << partitions[i].get_id() << "] is being processed.\n";
            std::vector<logical_tensor> inputs = partitions[i].get_in_ports();
            std::vector<logical_tensor> outputs = partitions[i].get_out_ports();

            /// replace input logical tensor with the queried one
            replace_with_queried_logical_tensors(inputs, id_to_queried_logical_tensors);

            /// update output logical tensors with ANY layout
            update_tensors_with_any_layout(outputs, id_to_set_any_layout);

            std::cout << "Compiling--------------------------------------";
            /// compile to generate compiled partition
            c_partitions[i] = partitions[i].compile(inputs, outputs, e);
            std::cout << "Success!\n";

            record_queried_logical_tensors(partitions[i].get_out_ports(), c_partitions[i],
                id_to_queried_logical_tensors);

            std::cout << "Creating tensors and allocating memory buffer--";
            std::vector<tensor> input_ts = tm.construct_and_initialize_tensors(inputs, c_partitions[i], e, 1);
            std::vector<tensor> output_ts = tm.construct_and_initialize_tensors(outputs, c_partitions[i], e, 0);
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
    
    float bn_scale = 1.0, bn_mean = 1.0, bn_shift = 1.0, bn_var = 1.0;
    // Step 6: Check correctness of the output results
    std::cout << "Check correctness------------------------------";
    float expected_result = bn_scale
                    * ((bn_scale * ((1 * 1 * 1 * 256) - bn_mean)
                                       / std::sqrt(bn_var)
                               + bn_shift)
                                    * (1 * 1 * 64)
                            - bn_mean)
                    / std::sqrt(bn_var)
            + bn_shift + /* residual connection */ 1;

    void *actual_output_ptr = tm.get(relu1_dst_desc.get_id()).get_data_handle();
    auto output_dims = relu1_dst_desc.get_dims();
    auto num_elem = product(output_dims);
    std::vector<float> expected_output(num_elem, expected_result);
    compare_data(expected_output.data(), reinterpret_cast<float *>(actual_output_ptr), num_elem);
    std::cout << "Success!\n";
    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
// clang-format on
