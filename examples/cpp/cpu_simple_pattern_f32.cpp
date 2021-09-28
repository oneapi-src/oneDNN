/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

/// @example cpu_simple_pattern_f32.cpp
/// @copybrief cpu_simple_pattern_f32_cpp
/// Annotated version: @ref cpu_simple_pattern_f32_cpp

/// @page cpu_simple_pattern_f32_cpp CPU example for a simple f32 pattern
///
/// > Example code: @ref cpu_simple_pattern_f32.cpp

#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    const engine::kind ekind = engine::kind::cpu;

    /// create a graph
    graph g(ekind);

    /// create logical tensors
    std::vector<int64_t> conv_input_dims {8, 56, 56, 256}; // NXC
    std::vector<int64_t> conv_weight_dims {64, 256, 1, 1}; // OIX

    logical_tensor conv_src_desc {
            0, data_type::f32, conv_input_dims, layout_type::strided};
    logical_tensor conv_weight_desc {
            1, data_type::f32, conv_weight_dims, layout_type::strided};

    /// ndims = 4, let the library to calculate the output shape.
    logical_tensor conv_dst_desc {2, data_type::f32, 4, layout_type::strided};

    /// create op conv
    op conv {2, op::kind::Convolution, {conv_src_desc, conv_weight_desc},
            {conv_dst_desc}, "conv"};
    conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv.set_attr<std::string>("data_format", "NXC");
    conv.set_attr<std::string>("filter_format", "OIX");
    conv.set_attr<int64_t>("groups", 1);

    /// ndims = 4, let the library to calculate the output shape.
    logical_tensor relu_dst_desc {3, data_type::f32, 4, layout_type::strided};

    /// create op relu
    op relu {4, op::kind::ReLU, {conv_dst_desc}, {relu_dst_desc}, "relu"};

    /// add the ops to the graph
    g.add_op(conv);
    g.add_op(relu);

    /// The graph will be partitioned into 1 partitions: `conv + relu`
    auto partitions = g.get_partitions();
    if (partitions.size() != 1)
        throw std::runtime_error(
                "cpu_simple_pattern_f32: incorrect partition number");

    /// create a new engine and stream
    engine eng {ekind, 0};
    stream strm {eng};

    /// compile the partition. the shape and layout of output logical tensor
    /// will be inferred during the compilation.
    auto cp = partitions[0].compile(
            {conv_src_desc, conv_weight_desc}, {relu_dst_desc}, eng);

    /// get output logical tensor from the compiled partition
    logical_tensor relu_dst_desc_q
            = cp.query_logical_tensor(relu_dst_desc.get_id());
    std::vector<int64_t> relu_dst_dims = relu_dst_desc_q.get_dims();

    if (relu_dst_dims.size() != 4)
        throw std::runtime_error(
                "cpu_simple_pattern_f32: incorrect inferred output shape");

    /// prepare data
    std::vector<float> conv_src_data(8 * 56 * 56 * 256);
    std::vector<float> conv_weight_data(64 * 256 * 1 * 1);
    std::vector<float> relu_dst_data(
            relu_dst_desc_q.get_mem_size() / sizeof(float));

    /// create tensors for the execution
    tensor conv_src_ts {conv_src_desc, eng, conv_src_data.data()};
    tensor conv_weight_ts {conv_weight_desc, eng, conv_weight_data.data()};
    tensor relu_dst_ts {relu_dst_desc_q, eng, relu_dst_data.data()};

    /// execute the compile partition
    cp.execute(strm, {conv_src_ts, conv_weight_ts}, {relu_dst_ts});

    return 0;
}
