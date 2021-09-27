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

/// @example cpu_inplace_ports.cpp
/// @copybrief cpu_inplace_ports_cpp
/// Annotated version: @ref cpu_inplace_ports_cpp

/// @page cpu_inplace_ports_cpp CPU example supports inplace computation
///
/// > Example code: @ref cpu_inplace_ports.cpp

#include <algorithm>

#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    const engine::kind ekind = engine::kind::cpu;

    /// create a graph
    graph g(ekind);

    std::vector<int64_t> conv_input_dims = {8, 256, 56, 56};
    std::vector<int64_t> conv_weight_dims = {64, 256, 1, 1};
    std::vector<int64_t> conv_bias_dims = {64};
    std::vector<int64_t> conv_dst_dims = {8, 64, 56, 56};

    /// create logical tensors for op conv
    logical_tensor conv_src_desc {
            0, data_type::f32, conv_input_dims, layout_type::strided};
    logical_tensor conv_weight_desc {
            1, data_type::f32, conv_weight_dims, layout_type::strided};
    logical_tensor conv_bias_desc {
            2, data_type::f32, conv_bias_dims, layout_type::strided};
    logical_tensor conv_dst_desc {
            3, data_type::f32, conv_dst_dims, layout_type::strided};

    /// create op conv
    op conv(4, op::kind::Convolution, {conv_src_desc, conv_weight_desc},
            {conv_dst_desc}, "conv");
    conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv.set_attr<std::string>("data_format", "NCX");
    conv.set_attr<std::string>("filter_format", "OIX");
    conv.set_attr<int64_t>("groups", 1);

    /// create output logical tensor for op bias_add
    logical_tensor conv_bias_dst_desc {
            5, data_type::f32, conv_dst_dims, layout_type::strided};

    /// create op bias_add
    op bias(6, op::kind::BiasAdd, {conv_dst_desc, conv_bias_desc},
            {conv_bias_dst_desc}, "bias");

    /// create another input logical tensor for op add
    logical_tensor add_src1_desc {
            7, data_type::f32, conv_dst_dims, layout_type::strided};

    /// create output logical tensor for op add
    logical_tensor add_dst_desc {
            8, data_type::f32, conv_dst_dims, layout_type::strided};

    /// create op add
    op add(9, op::kind::Add, {conv_bias_dst_desc, add_src1_desc},
            {add_dst_desc}, "add");

    /// create output logical tensor for op abs
    logical_tensor abs_dst_desc {
            10, data_type::f32, conv_dst_dims, layout_type::strided};

    /// create op abs
    op abs(11, op::kind::Abs, {add_dst_desc}, {abs_dst_desc}, "abs0");

    /// add above operators to the graph
    g.add_op(conv);
    g.add_op(bias);
    g.add_op(add);
    g.add_op(abs);

    /// conv
    ///   |
    /// bias       src1
    ///     \       /
    ///        add
    ///         |
    ///        abs
    ///         |

    /// the graph will be partitioned into 2 partitions:
    ///   - conv + bias + add
    ///   - abs
    auto partitions = g.get_partitions(partition::policy::fusion);

    if (partitions.size() != 2)
        throw std::runtime_error("cpu_inplace_ports: incorrect partition size");

    /// create engine, allocator, and stream
    engine eng {ekind, 0};
    stream strm {eng};

    /// compile the first partition: conv + bias + add
    auto cp = partitions[0].compile(
            {conv_src_desc, conv_weight_desc, conv_bias_desc, add_src1_desc},
            {add_dst_desc}, eng);

    /// get the inplace ports from compiled partition
    auto inplace_ports = cp.get_inplace_ports();

    /// in this case, the src1 and dst tensor of add can be inplaced.
    if (inplace_ports.empty() || inplace_ports[0].first != 7
            || inplace_ports[0].second != 8) {
        throw std::runtime_error("cpu_inplace_ports: incorrect inplace ports");
    }

    /// prepare data, the ground truth of output = 256 * (1 * 1) + 1 + 1 = 288.
    std::vector<float> conv_src_data(8 * 256 * 56 * 56, 1.0f);
    std::vector<float> conv_weight_data(64 * 256 * 1 * 1, 1.0f);
    std::vector<float> conv_bias_data(64, 1.0f);
    std::vector<float> add_src1_data(8 * 64 * 56 * 56, 1.0f);

    /// create tensors for execution
    tensor conv_src_ts {conv_src_desc, eng, conv_src_data.data()};
    tensor conv_weight_ts {conv_weight_desc, eng, conv_weight_data.data()};
    tensor conv_bias_ts {conv_bias_desc, eng, conv_bias_data.data()};

    /// src1 and dst share the same buffer
    tensor add_src1_ts {add_src1_desc, eng, add_src1_data.data()};
    tensor add_dst_ts {add_dst_desc, eng, add_src1_data.data()};

    /// execute the compiled partition with input and output inplaced
    cp.execute(strm, {conv_src_ts, conv_weight_ts, conv_bias_ts, add_src1_ts},
            {add_dst_ts});

    /// check the output results
    bool corr = std::all_of(add_src1_data.begin(), add_src1_data.end(),
            [](float v) { return v == 258.f; });
    if (!corr)
        throw std::runtime_error("cpu_inplace_ports: incorrect output results");

    return 0;
}
