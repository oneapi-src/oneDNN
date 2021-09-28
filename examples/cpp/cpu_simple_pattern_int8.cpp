/*******************************************************************************
* Copyright 2021 Intel Corporation
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

/// @example cpu_simple_pattern_int8.cpp
/// @copybrief cpu_simple_pattern_int8_cpp
/// Annotated version: @ref cpu_simple_pattern_int8_cpp

/// @page cpu_simple_pattern_int8_cpp CPU example for int8 conv + relu pattern
///
/// > Example code: @ref cpu_simple_pattern_int8.cpp

#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    const engine::kind ekind = engine::kind::cpu;

    /// create a graph
    graph g(ekind);

    /// create logical tensor
    std::vector<int64_t> conv_input_dims {8, 56, 56, 256}; // NXC
    std::vector<int64_t> conv_weight_dims {1, 1, 256, 64}; // XIO
    std::vector<int64_t> conv_bias_dims {64};

    /// per-tensor asymmetric quantized src activation with op dequant0
    logical_tensor dequant0_src_desc {
            0, data_type::u8, conv_input_dims, layout_type::strided};
    logical_tensor conv_src_desc {
            1, data_type::f32, conv_input_dims, layout_type::strided};
    op dequant0(2, op::kind::Dequantize, {dequant0_src_desc}, {conv_src_desc},
            "dequant0");
    dequant0.set_attr<std::string>("qtype", "per_tensor");
    dequant0.set_attr<std::vector<float>>("scales", {0.1f});
    dequant0.set_attr<std::vector<int64_t>>("zps", {10});

    /// per-channel symmetric quantized weight with op dequant1
    logical_tensor dequant1_src_desc {
            3, data_type::s8, conv_weight_dims, layout_type::strided};
    logical_tensor conv_weight_desc {
            4, data_type::f32, conv_weight_dims, layout_type::strided};
    op dequant1(5, op::kind::Dequantize, {dequant1_src_desc},
            {conv_weight_desc}, "dequant1");
    dequant1.set_attr<std::string>("qtype", "per_channel");
    std::vector<float> wei_scales(64, 0.1f);
    std::vector<int64_t> wei_zps(64, 0);
    dequant1.set_attr<std::vector<float>>("scales", wei_scales);
    dequant1.set_attr<std::vector<int64_t>>("zps", wei_zps);
    dequant1.set_attr<int64_t>("axis", 1);

    logical_tensor conv_bias_desc {
            6, data_type::f32, conv_bias_dims, layout_type::strided};
    /// output tensor, we even don't know its ndim.
    logical_tensor conv_dst_desc {7, data_type::f32, -1, layout_type::strided};

    /// create the convolution op
    op conv(8, op::kind::Convolution,
            {conv_src_desc, conv_weight_desc, conv_bias_desc}, {conv_dst_desc},
            "conv");
    conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv.set_attr<std::string>("data_format", "NXC");
    conv.set_attr<std::string>("filter_format", "XIO");
    conv.set_attr<int64_t>("groups", 1);

    /// create op relu
    logical_tensor relu_dst_desc {9, data_type::f32, -1, layout_type::strided};
    op relu(10, op::kind::ReLU, {conv_dst_desc}, {relu_dst_desc}, "relu");

    /// create op quant
    logical_tensor quant_dst_desc {11, data_type::u8, -1, layout_type::strided};
    op quant(
            12, op::kind::Quantize, {relu_dst_desc}, {quant_dst_desc}, "quant");
    quant.set_attr<std::string>("qtype", "per_tensor");
    quant.set_attr<std::vector<float>>("scales", {0.1f});
    quant.set_attr<std::vector<int64_t>>("zps", {10});

    /// add the operators to the graph
    g.add_op(dequant0);
    g.add_op(dequant1);
    g.add_op(conv);
    g.add_op(relu);
    g.add_op(quant);

    /// dequant0    dequant1
    ///       \      /
    ///         conv
    ///          |
    ///         relu
    ///          |
    ///         quant

    /// The graph will be partitioned into 1 partitions.
    auto partitions = g.get_partitions();
    if (partitions.size() != 1)
        throw std::runtime_error(
                "cpu_simple_pattern_int8: incorrect partition number");

    /// construct a new engine and stream
    engine eng {ekind, 0};
    stream strm {eng};

    auto cp = partitions[0].compile(
            {dequant0_src_desc, dequant1_src_desc, conv_bias_desc},
            {quant_dst_desc}, eng);

    /// query the output logical tensor from the compiled partition
    logical_tensor quant_dst_desc_q
            = cp.query_logical_tensor(quant_dst_desc.get_id());

    /// prepare data for the execution
    std::vector<uint8_t> dequant0_src_data(8 * 56 * 56 * 256);
    std::vector<int8_t> dequant1_src_data(1 * 1 * 256 * 64);
    std::vector<float> conv_bias_data(64);
    std::vector<uint8_t> quant_dst_data(
            quant_dst_desc_q.get_mem_size() / sizeof(uint8_t));

    /// create tensors
    tensor dequant0_src_ts {dequant0_src_desc, eng, dequant0_src_data.data()};
    tensor dequant1_src_ts {dequant1_src_desc, eng, dequant1_src_data.data()};
    tensor conv_bias_ts {conv_bias_desc, eng, conv_bias_data.data()};
    tensor quant_dst_ts {quant_dst_desc, eng, quant_dst_data.data()};

    /// execute the compile partition
    cp.execute(strm, {dequant0_src_ts, dequant1_src_ts, conv_bias_ts},
            {quant_dst_ts});

    return 0;
}
