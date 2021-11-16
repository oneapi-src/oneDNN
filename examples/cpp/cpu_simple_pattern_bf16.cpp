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

/// @example cpu_simple_pattern_bf16.cpp
/// @copybrief cpu_simple_pattern_bf16_cpp
/// Annotated version: @ref cpu_simple_pattern_bf16_cpp

/// @page cpu_simple_pattern_bf16_cpp CPU example for a simple bf16 pattern
///
/// > Example code: @ref cpu_simple_pattern_bf16.cpp

#include <iostream>

#include "dnnl.hpp"

#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    const auto isa = dnnl_get_effective_cpu_isa();
    if (isa < dnnl_cpu_isa_avx512_core) {
        std::cout << "cpu_simple_pattern_bf16: skip bf16 examples for systems"
                     "that do not support avx512_core"
                  << std::endl;
        return 0;
    }

    /// create a graph
    const engine::kind ekind = engine::kind::cpu;
    graph g(ekind);

    /// create logical tensor with bf16 data type
    std::vector<int64_t> conv0_src_dims {8, 3, 227, 227};
    std::vector<int64_t> conv0_weight_dims {96, 3, 11, 11};
    std::vector<int64_t> conv0_bias_dims {96};

    logical_tensor conv0_src_desc {
            0, data_type::bf16, conv0_src_dims, layout_type::strided};
    logical_tensor conv0_weight_desc {
            1, data_type::bf16, conv0_weight_dims, layout_type::strided};
    logical_tensor conv0_bias_desc {
            2, data_type::bf16, conv0_bias_dims, layout_type::strided};

    /// don't know the output shape of conv1, let the library to infer
    logical_tensor conv0_dst_desc {3, data_type::bf16, 4, layout_type::strided};

    /// create op conv0
    op conv0(4, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
            {conv0_dst_desc}, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {111, 111});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {111, 111});
    conv0.set_attr<std::string>("auto_pad", "VALID");
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");
    conv0.set_attr<int64_t>("groups", 1);

    logical_tensor conv0_bias_add_dst_desc {
            5, data_type::bf16, 4, layout_type::strided};

    op conv0_bias_add(6, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
            {conv0_bias_add_dst_desc}, "conv0_bias_add");

    logical_tensor relu0_dst_desc {7, data_type::bf16, 4, layout_type::strided};

    op relu0(8, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
            "relu0");

    /// create logical tensors for conv1 with bf16 data type
    std::vector<int64_t> conv1_weight_dims {96, 96, 1, 1};
    std::vector<int64_t> conv1_bias_dims {96};

    logical_tensor conv1_weight_desc {
            9, data_type::bf16, conv1_weight_dims, layout_type::strided};
    logical_tensor conv1_bias_desc {
            10, data_type::bf16, conv1_bias_dims, layout_type::strided};
    logical_tensor conv1_dst_desc {
            11, data_type::bf16, 4, layout_type::strided};

    /// create op conv1
    op conv1(12, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
            {conv1_dst_desc}, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");
    conv1.set_attr<int64_t>("groups", 1);

    logical_tensor conv1_bias_add_dst_desc {
            13, data_type::bf16, 4, layout_type::strided};

    op conv1_bias_add(14, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
            {conv1_bias_add_dst_desc}, "conv1_bias_add");

    logical_tensor relu1_dst_desc {
            15, data_type::bf16, 4, layout_type::strided};

    op relu1(16, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
            "relu1");

    /// add the ops into the graph
    g.add_op(conv0);
    g.add_op(conv0_bias_add);
    g.add_op(relu0);

    g.add_op(conv1);
    g.add_op(conv1_bias_add);
    g.add_op(relu1);

    /// The graph will be partitioned into two partitions:
    /// - conv0 + conv0_bias_add + relu0
    /// - conv1 + conv1_bias_add + relu1
    auto partitions = g.get_partitions(partition::policy::fusion);

    if (partitions.size() != 2) {
        throw std::runtime_error(
                "cpu_simple_pattern_bf16: incorrect partition number");
    }

    /// create a new engine and stream
    engine eng {ekind, 0};
    stream strm {eng};

    /// compile the first partition
    auto cp0 = partitions[0].compile(
            {conv0_src_desc, conv0_weight_desc, conv0_bias_desc},
            {relu0_dst_desc}, eng);

    /// get the output logical tensor for the first compiled partition
    logical_tensor relu0_dst_desc_q
            = cp0.query_logical_tensor(relu0_dst_desc.get_id());

    /// prepare data for the first execution, we use uint16_t to mimic bf16 for
    /// memory allocation.
    std::vector<uint16_t> conv0_src_data(8 * 3 * 227 * 227);
    std::vector<uint16_t> conv0_weight_data(96 * 3 * 11 * 11);
    std::vector<uint16_t> conv0_bias_data(96);

    std::vector<uint16_t> relu0_dst_data(
            relu0_dst_desc_q.get_mem_size() / sizeof(uint16_t));

    /// create tensors for the execution
    tensor conv0_src_ts {conv0_src_desc, eng, conv0_src_data.data()};
    tensor conv0_weight_ts {conv0_weight_desc, eng, conv0_weight_data.data()};
    tensor conv0_bias_ts {conv0_bias_desc, eng, conv0_bias_data.data()};
    tensor relu0_dst_ts {relu0_dst_desc_q, eng, relu0_dst_data.data()};

    /// execute the first compiled partition
    cp0.execute(strm, {conv0_src_ts, conv0_weight_ts, conv0_bias_ts},
            {relu0_dst_ts});

    /// compile the second partition
    auto cp1 = partitions[1].compile(
            {relu0_dst_desc_q, conv1_weight_desc, conv1_bias_desc},
            {relu1_dst_desc}, eng);

    /// get the output logical tensor for the second compiled partition
    logical_tensor relu1_dst_desc_q
            = cp1.query_logical_tensor(relu1_dst_desc.get_id());

    /// check the final output shape, should be {8, 96, 55, 55}
    auto relu1_dst_dims = relu1_dst_desc_q.get_dims();
    if (relu1_dst_dims.size() != 4 || relu1_dst_dims[0] != 8
            || relu1_dst_dims[1] != 96 || relu1_dst_dims[2] != 55
            || relu1_dst_dims[3] != 55) {
        throw std::runtime_error(
                "cpu_simple_pattern_bf16: incorrect inferred output shape");
    }

    /// prepare data for cp1 execution, we use uint16_t to mimic bf16 for memory
    /// allocation.
    std::vector<uint16_t> conv1_weight_data(96 * 96 * 1 * 1);
    std::vector<uint16_t> conv1_bias_data(96);
    std::vector<uint16_t> relu1_dst_data(
            relu1_dst_desc_q.get_mem_size() / sizeof(uint16_t));

    /// create tensors for the execution
    tensor conv1_weight_ts {conv1_weight_desc, eng, conv1_weight_data.data()};
    tensor conv1_bias_ts {conv1_bias_desc, eng, conv1_bias_data.data()};
    tensor relu1_dst_ts {relu1_dst_desc_q, eng, relu1_dst_data.data()};

    /// execute cp1, directly use the output tensor of cp0
    cp1.execute(strm, {relu0_dst_ts, conv1_weight_ts, conv1_bias_ts},
            {relu1_dst_ts});

    return 0;
}
