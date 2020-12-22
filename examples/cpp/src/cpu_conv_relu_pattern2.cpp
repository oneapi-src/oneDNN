/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/// @example cpu_conv_relu_pattern2.cpp
/// @copybrief cpu_conv_relu_pattern2_cpp
/// > Annotated version: @ref cpu_conv_relu_pattern2_cpp

/// @page cpu_conv_relu_pattern2_cpp CPU example for conv+relu with XIO format
///
/// > Example code: @ref cpu_conv_relu_pattern2.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/utils.hpp"

using namespace dnnl::graph;

// digraph G {
// Wildcard_100002 -> Convolution_100003;
// Convolution_100003 -> ReLU_100005;
// }

// Test conv relu different shape compile and execute
int main(int argc, char **argv) {
    // Conv with auto_pad = VALID, data_format = NXC, filter_format = XIO
    std::cout << "========Example: Conv+ReLU========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        std::cout << "Don't support gpu now\n";
        return -1;
    }

    // Step 1: Initialize engine and stream
    /// (todo)xinyu: improve this part when gpu pass is ready
    std::cout << "Initialize CPU engine and stream---------------";
    const int32_t device_id = 0;
    engine eng {engine_kind, device_id};
    stream strm {eng};
    std::cout << "Success!\n";

    // Step 2: Construct a graph
    graph g(engine_kind);

    auto &id_mgr = logical_id_manager::get();

    /// Create OP and set attributes
    std::cout << "Create op--------------------------------------";
    /// inuput node
    op input0(id_mgr["input0"], op::kind::Wildcard, "input0");

    /// conv+relu
    op conv0(id_mgr["conv0"], op::kind::Convolution, "conv0");
    conv0.set_attr("strides", std::vector<int64_t> {1, 1});
    conv0.set_attr("pads_begin", std::vector<int64_t> {1, 1});
    conv0.set_attr("pads_end", std::vector<int64_t> {1, 1});
    conv0.set_attr<std::string>("auto_pad", "VALID");
    conv0.set_attr("dilations", std::vector<int64_t> {1, 1});
    conv0.set_attr("data_format", std::string("NXC"));
    conv0.set_attr("filter_format", std::string("XIO"));
    conv0.set_attr("groups", int64_t {1});
    op relu0(id_mgr["relu0"], op::kind::ReLU, "relu0");
    std::cout << "Success!\n";

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input_dims {1, 3, 4, 1}; // NXC
    std::vector<int64_t> conv0_weight_dims {3, 3, 1, 1}; // XIO
    std::vector<int64_t> conv0_dst_dims {-1, -1, -1, -1};

    logical_tensor conv0_src_desc {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_weight_desc {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_dst_desc {id_mgr["conv0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu0_dst_desc {id_mgr["relu0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};

    std::cout << "Success!\n";

    /// Add Input/Output
    std::cout << "Add logical tensor to op-----------------------";
    input0.add_output(conv0_src_desc);

    conv0.add_inputs({conv0_src_desc, conv0_weight_desc});
    conv0.add_output(conv0_dst_desc);
    relu0.add_input(conv0_dst_desc);
    relu0.add_output(relu0_dst_desc);
    std::cout << "Success!\n";

    /// Select OP
    std::cout << "Select op to graph-----------------------------";
    g.add_op(input0);
    g.add_op(conv0);
    g.add_op(relu0);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter partitions
    /// Graph will be filtered into 1 partitions: `conv0+relu0`
    /// `export DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter partitions------------------------------";
    auto partitions = g.get_partitions(partition::policy::fusion);

    if (partitions.size() != 1) {
        throw std::runtime_error("wrong partition number");
    }
    std::cout << "Success!\n";

    // Step 4: Prepare logical tensors with proper format and compile partitions
    std::cout << "Prepare logical tensors with proper format-----";
    // layout_id::abcd, but format is NXC
    logical_tensor conv0_src_desc_plain {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc_plain {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::strided};
    logical_tensor relu0_dst_desc_plain {id_mgr["relu0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::strided};
    std::cout << "Success!\n";

    std::cout << "Infer shape------------------------------------\n";
    std::vector<logical_tensor> in0 {
            conv0_src_desc_plain, conv0_weight_desc_plain};
    std::vector<logical_tensor> out0({relu0_dst_desc_plain});
    partitions[0].infer_shape(in0, out0);
    std::cout << "Success!\n";
    std::vector<int64_t> infered_relu0_dst_dims = out0[0].get_dims();
    std::cout << "infered relu0 out shape: " << infered_relu0_dst_dims[0] << ","
              << infered_relu0_dst_dims[1] << "," << infered_relu0_dst_dims[2]
              << "," << infered_relu0_dst_dims[3] << "\n";

    std::cout << "Compile partition 0----------------------------";
    auto cp1 = partitions[0].compile(in0, out0, eng);
    std::cout << "Success!\n";

    std::cout << "Query layout id from compiled partition 0------";
    relu0_dst_desc_plain = cp1.query_logical_tensor(id_mgr["relu0_dst"]);
    std::cout << "Success!\n";

    // Step 5: Prepare tensor and submit compiled partitions
    std::cout << "Prepare tensor and submit compiled partitions--";
    std::vector<float> conv0_src_data(
            static_cast<size_t>(product(input_dims)), 1.0f);
    std::vector<float> conv0_weight_data(
            static_cast<size_t>(product(conv0_weight_dims)), 1.0f);
    std::vector<float> relu0_dst_data(
            relu0_dst_desc_plain.get_mem_size() / sizeof(float), 0.0);

    tensor conv0_src(conv0_src_desc_plain, conv0_src_data.data());
    tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data.data());
    tensor relu0_dst(relu0_dst_desc_plain, relu0_dst_data.data());

    std::vector<tensor> in_list_0 {conv0_src, conv0_weight};
    std::vector<tensor> out_list_0 {relu0_dst};
    cp1.execute(strm, in_list_0, out_list_0);

    std::cout << "Success!\n";
    std::cout << "Check correctness------------------------------\n";
    std::cout << "relu0 output result:";
    for (auto v : relu0_dst_data) {
        std::cout << v << " ";
        if (std::abs(9 - v) > 1e-6f) {
            throw std::runtime_error(
                    "output result is not equal to excepted "
                    "results");
        }
    }
    std::cout << "\nExecute Success!\n";

    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
