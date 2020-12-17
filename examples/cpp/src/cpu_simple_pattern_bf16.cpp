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

/// @example cpu_simple_pattern_bf16.cpp
/// @copybrief cpu_simple_pattern_bf16_cpp
/// Annotated version: @ref cpu_simple_pattern_bf16_cpp

/// @page cpu_simple_pattern_bf16_cpp CPU example for conv+bias+relu pattern
///
/// > Example code: @ref cpu_simple_pattern_bf16.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnnl.h"

#include "common/utils.hpp"

using namespace dnnl::graph;

int main(int argc, char **argv) {
    std::cout << "========Example: Conv->ReLU->Conv->ReLU========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        printf("Don't support gpu now\n");
        return -1;
    }

    static auto isa = dnnl_get_effective_cpu_isa();
    if (isa < dnnl_cpu_isa_avx512_core) {
        printf("Skip bf16 examples for systems that do not support "
               "avx512_core.");
        return 0;
    }

    // Step 1: Initialize engine and stream
    /// (todo)xinyu: improve this part when gpu pass is ready
    std::cout << "Initialize CPU engine and stream---------------";
    const int32_t device_id = 0;
    engine eng {engine_kind, device_id};
    allocator alloc {&allocate, &deallocate};
    eng.set_allocator(alloc);
    stream strm {eng};
    std::cout << "Success!\n";

    // Step 2: Construct a example graph: `conv->relu->conv->relu`
    graph g(engine_kind);

    /// Create OP and set attributes
    std::cout << "Create op--------------------------------------";
    op conv0(0, op::kind::Convolution, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {111, 111});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {111, 111});
    conv0.set_attr<std::string>("auto_pad", "VALID");
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");
    conv0.set_attr<int64_t>("groups", 1);
    op relu0(1, op::kind::ReLU, "relu0");
    op conv1(2, op::kind::Convolution, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");
    conv1.set_attr<int64_t>("groups", 1);
    op relu1(3, op::kind::ReLU, "relu1");
    op conv0_bias_add(5, op::kind::BiasAdd, "conv0_bias_add");
    op conv1_bias_add(6, op::kind::BiasAdd, "conv1_bias_add");
    std::cout << "Success!\n";

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";
    const std::vector<size_t> logical_id {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::vector<int64_t> input_dims {8, 3, 227, 227};
    std::vector<int64_t> weight_dims {96, 3, 11, 11};
    std::vector<int64_t> bias_dims {96};
    std::vector<int64_t> weight1_dims {96, 96, 1, 1};
    std::vector<int64_t> bias1_dims {96};
    /// output shape needs to be infered
    std::vector<int64_t> dst_dims {-1, -1, -1, -1};

    logical_tensor conv0_src_desc {logical_id[0],
            logical_tensor::data_type::bf16, input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_weight_desc {logical_id[1],
            logical_tensor::data_type::bf16, weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_bias_desc {logical_id[2],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_dst_desc {logical_id[3],
            logical_tensor::data_type::bf16, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu0_dst_desc {logical_id[4],
            logical_tensor::data_type::bf16, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_weight_desc {logical_id[5],
            logical_tensor::data_type::bf16, weight1_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_bias_desc {logical_id[6],
            logical_tensor::data_type::f32, bias1_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_dst_desc {logical_id[7],
            logical_tensor::data_type::bf16, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu1_dst_desc {logical_id[8],
            logical_tensor::data_type::bf16, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_bias_add_dst_desc {logical_id[9],
            logical_tensor::data_type::bf16, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_bias_add_dst_desc {logical_id[10],
            logical_tensor::data_type::bf16, dst_dims,
            logical_tensor::layout_type::undef};
    std::cout << "Success!\n";

    /// Add Input/Output
    std::cout << "Add logical tensor to op-----------------------";
    conv0.add_inputs({conv0_src_desc, conv0_weight_desc});
    conv0.add_output(conv0_dst_desc);
    conv0_bias_add.add_inputs({conv0_dst_desc, conv0_bias_desc});
    conv0_bias_add.add_output(conv0_bias_add_dst_desc);
    relu0.add_input(conv0_bias_add_dst_desc);
    relu0.add_output(relu0_dst_desc);
    conv1.add_inputs({relu0_dst_desc, conv1_weight_desc});
    conv1.add_output(conv1_dst_desc);
    conv1_bias_add.add_inputs({conv1_dst_desc, conv1_bias_desc});
    conv1_bias_add.add_output(conv1_bias_add_dst_desc);
    relu1.add_input(conv1_bias_add_dst_desc);
    relu1.add_output(relu1_dst_desc);
    std::cout << "Success!\n";

    /// Add OP
    std::cout << "Add OP to graph--------------------------------";
    g.add_op(conv0);
    g.add_op(relu0);
    g.add_op(conv1);
    g.add_op(relu1);
    g.add_op(conv0_bias_add);
    g.add_op(conv1_bias_add);
    std::cout << "Success!\n";

    // Step 3: Filter and get partitions
    /// Graph will be filtered into two partitions: `conv0+relu0` and `conv1+relu1`
    /// Setting `DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter and get partition-----------------------";
    auto partitions = g.get_partitions(partition::policy::fusion);

    if (partitions.size() != 2) {
        throw std::runtime_error("wrong partition number");
    }
    std::cout << "Success!\n";

    // Step 4: Prepare logical tensors with proper format and compile partitions
    /// In this example, graph inputs(conv0), outputs(relu1) and weights logical tensors are created with plain layout
    /// layout of logical tensors between partitions can be queried from compiled partition
    std::cout << "Prepare logical tensors with proper format-----";
    logical_tensor conv0_src_desc_plain {logical_id[0],
            logical_tensor::data_type::bf16, input_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc_plain {logical_id[1],
            logical_tensor::data_type::bf16, weight_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_bias_desc_plain {logical_id[2],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor relu0_dst_desc_any {logical_id[4],
            logical_tensor::data_type::bf16, dst_dims,
            logical_tensor::layout_type::any};
    std::cout << "Success!\n";

    std::cout << "Infer shape for partition 0--------------------";
    std::vector<logical_tensor> in0 {conv0_src_desc_plain,
            conv0_weight_desc_plain, conv0_bias_desc_plain};
    std::vector<logical_tensor> out0 {relu0_dst_desc_any};
    partitions[0].infer_shape(in0, out0);
    std::cout << "Success!\n";
    std::vector<int64_t> infered_relu0_dst_dims = out0[0].get_dims();
    std::cout << "Infered relu0 output shape:" << infered_relu0_dst_dims[0]
              << "," << infered_relu0_dst_dims[1] << ","
              << infered_relu0_dst_dims[2] << "," << infered_relu0_dst_dims[3]
              << "\n";

    std::cout << "Compile partition 0----------------------------";
    auto cp0 = partitions[0].compile(in0, out0, eng);
    std::cout << "Success!\n";

    std::cout << "Query logical tensor from compiled partition 0-";
    auto conv1_src_desc_opaque = cp0.query_logical_tensor(logical_id[4]);

    logical_tensor conv1_weight_desc_plain {logical_id[5],
            logical_tensor::data_type::bf16, weight1_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv1_bias_desc_plain {logical_id[6],
            logical_tensor::data_type::f32, bias1_dims,
            logical_tensor::layout_type::strided};
    logical_tensor relu1_dst_desc_plain {logical_id[8],
            logical_tensor::data_type::bf16, dst_dims,
            logical_tensor::layout_type::strided};
    std::cout << "Success!\n";

    std::cout << "Infer shape for partition 1--------------------";
    std::vector<logical_tensor> in1 {conv1_src_desc_opaque,
            conv1_weight_desc_plain, conv1_bias_desc_plain};
    std::vector<logical_tensor> out1 {relu1_dst_desc_plain};
    partitions[1].infer_shape(in1, out1);
    std::cout << "Success!\n";
    std::vector<int64_t> infered_relu1_dst_dims = out1[0].get_dims();
    std::cout << "Infered relu1 output shape:" << infered_relu1_dst_dims[0]
              << "," << infered_relu1_dst_dims[1] << ","
              << infered_relu1_dst_dims[2] << "," << infered_relu1_dst_dims[3]
              << "\n";

    std::cout << "Compile partition 1----------------------------";
    auto cp1 = partitions[1].compile(in1, out1, eng);
    std::cout << "Success!\n";

    // Step 5: Prepare tensor and execute compiled partitions
    std::cout << "Prepare tensor and execute compiled partitions--";
    auto relu0_dst_desc_opaque = cp0.query_logical_tensor(logical_id[4]);
    relu1_dst_desc_plain = cp1.query_logical_tensor(logical_id[8]);
    std::vector<float> conv0_src_data(
            static_cast<size_t>(product(input_dims)), 1.0f);
    std::vector<float> conv0_weight_data(
            static_cast<size_t>(product(weight_dims)), 1.0f);
    std::vector<float> conv0_bias_data(
            static_cast<size_t>(product(bias_dims)), 1.0f);
    std::vector<float> relu0_dst_data(
            relu0_dst_desc_opaque.get_mem_size() / sizeof(float), 0.0);
    std::vector<float> conv1_weight_data(
            static_cast<size_t>(product(weight1_dims)), 1.0f);
    std::vector<float> conv1_bias_data(
            static_cast<size_t>(product(bias1_dims)), 1.0f);
    std::vector<float> relu1_dst_data(
            relu1_dst_desc_plain.get_mem_size() / sizeof(float), 0.0);

    tensor conv0_src(conv0_src_desc_plain, conv0_src_data.data());
    tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data.data());
    tensor conv0_bias(conv0_bias_desc_plain, conv0_bias_data.data());
    tensor relu0_dst(relu0_dst_desc_opaque, relu0_dst_data.data());

    std::vector<tensor> in_list_0 {conv0_src, conv0_weight, conv0_bias};
    std::vector<tensor> out_list_0 {relu0_dst};
    cp0.execute(strm, in_list_0, out_list_0);

    tensor conv1_weight(conv1_weight_desc_plain, conv1_weight_data.data());
    tensor conv1_bias(conv1_bias_desc_plain, conv1_bias_data.data());
    tensor relu1_dst(relu1_dst_desc_plain, relu1_dst_data.data());

    std::vector<tensor> in_list_1 {relu0_dst, conv1_weight, conv1_bias};
    std::vector<tensor> out_list_1 {relu1_dst};
    cp1.execute(strm, in_list_1, out_list_1);
    std::cout << "Success!\n";

    // Step 6: Check correctness of the output results
    std::cout << "Check correctness------------------------------";
    std::cout << "Skip!\n";

    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
