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

/// @example cpu_conv_bias_bn_relu_pattern.cpp
/// @copybrief cpu_conv_bias_bn_relu_pattern_cpp
/// > Annotated version: @ref cpu_conv_bias_bn_relu_pattern_cpp

/// @page cpu_conv_bias_bn_relu_pattern_cpp CPU example for conv+bias+bn+relu pattern
///
/// > Example code: @ref cpu_conv_bias_bn_relu_pattern.cpp

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

int main(int argc, char **argv) {
    std::cout
            << "========Example: "
               "Conv->BiasAdd->BatchNorm->ReLU->Conv->BiasAdd->ReLU========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        printf("Don't support gpu now\n");
        return -1;
    }

    // Step 1: Initialize engine and stream
    /// (todo)xinyu: improve this part when gpu pass is ready
    std::cout << "Initialize CPU engine and stream---------------";
    const int32_t device_id = 0;
    engine eng {engine_kind, device_id};
    allocator alloc {&allocate, &deallocate};
    eng.set_allocator(alloc);
    stream strm {eng, nullptr};
    std::cout << "Success!\n";

    // Step 2: Construct a example graph: `Conv->BiasAdd->BatchNorm->ReLU->Conv->BiasAdd->ReLU`
    graph g(engine_kind);

    /// Create OP and set attributes
    std::cout << "Create op--------------------------------------";
    op conv0(0, op::kind::Convolution, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");
    conv0.set_attr<int64_t>("groups", 1);
    op bn0(1, op::kind::BatchNormInference, "bn0");
    const float epsilon = 0.f;
    bn0.set_attr<float>("epsilon", epsilon);
    op relu0(2, op::kind::ReLU, "relu0");
    op conv1(3, op::kind::Convolution, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");
    conv1.set_attr<int64_t>("groups", 1);
    op relu1(4, op::kind::ReLU, "relu1");
    op conv0_bias_add(5, op::kind::BiasAdd, "conv0_bias_add");
    op conv1_bias_add(6, op::kind::BiasAdd, "conv1_bias_add");
    std::cout << "Success!\n";

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input_dims {8, 3, 227, 227};
    std::vector<int64_t> weight_dims {96, 3, 11, 11};
    std::vector<int64_t> bias_dims {96};
    std::vector<int64_t> weight1_dims {96, 96, 1, 1};
    std::vector<int64_t> bias1_dims {96};
    std::vector<int64_t> dst_dims {8, 96, 55, 55};

    auto &id_mgr = logical_id_manager::get();
    logical_tensor conv0_src_desc {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_weight_desc {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_bias_desc {id_mgr["conv0_bias"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_dst_desc {id_mgr["conv0_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_scale_desc {id_mgr["bn0_scale"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_shift_desc {id_mgr["bn0_shift"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_mean_desc {id_mgr["bn0_mean"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_var_desc {id_mgr["bn0_var"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_dst_desc {id_mgr["bn0_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu0_dst_desc {id_mgr["relu0_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_weight_desc {id_mgr["conv1_weight"],
            logical_tensor::data_type::f32, weight1_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_bias_desc {id_mgr["conv1_bias"],
            logical_tensor::data_type::f32, bias1_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_dst_desc {id_mgr["conv1_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu1_dst_desc {id_mgr["relu1_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_bias_add_dst_desc {id_mgr["conv0_bias_add_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_bias_add_dst_desc {id_mgr["conv1_bias_add_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};
    std::cout << "Success!\n";

    /// Add Input/Output
    std::cout << "Add logical tensor to op-----------------------";
    conv0.add_inputs({conv0_src_desc, conv0_weight_desc});
    conv0.add_output(conv0_dst_desc);
    conv0_bias_add.add_inputs({conv0_dst_desc, conv0_bias_desc});

    conv0_bias_add.add_output(conv0_bias_add_dst_desc);
    bn0.add_inputs({conv0_bias_add_dst_desc, bn0_scale_desc, bn0_shift_desc,
            bn0_mean_desc, bn0_var_desc});
    bn0.add_output(bn0_dst_desc);
    relu0.add_input(bn0_dst_desc);
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
    g.add_op(conv0_bias_add);
    g.add_op(bn0);
    g.add_op(relu0);
    g.add_op(conv1);
    g.add_op(conv1_bias_add);
    g.add_op(relu1);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter and get partitions
    /// Graph will be filtered into two partitions: `conv0+biasadd+bn0+relu0` and `conv1+biasadd+relu1`
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
    logical_tensor conv0_src_desc_plain {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc_plain {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, weight_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_bias_desc_plain {id_mgr["conv0_bias"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn0_scale_desc_plain {id_mgr["bn0_scale"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn0_shift_desc_plain {id_mgr["bn0_shift"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn0_mean_desc_plain {id_mgr["bn0_mean"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn0_var_desc_plain {id_mgr["bn0_var"],
            logical_tensor::data_type::f32, bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor relu0_dst_desc_any {id_mgr["relu0_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::any};
    std::cout << "Success!\n";

    std::cout << "Compile partition 0----------------------------";
    std::vector<logical_tensor> in0 {conv0_src_desc_plain,
            conv0_weight_desc_plain, conv0_bias_desc_plain,
            bn0_scale_desc_plain, bn0_shift_desc_plain, bn0_mean_desc_plain,
            bn0_var_desc_plain};
    std::vector<logical_tensor> out0 {relu0_dst_desc_any};
    auto cp0 = partitions[0].compile(in0, out0, eng);
    std::cout << "Success!\n";

    std::cout << "Query logical tensor from compiled partition 0-";
    auto conv1_src_desc_any = cp0.query_logical_tensor(id_mgr["relu0_dst"]);

    logical_tensor conv1_weight_desc_plain {id_mgr["conv1_weight"],
            logical_tensor::data_type::f32, weight1_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv1_bias_desc_plain {id_mgr["conv1_bias"],
            logical_tensor::data_type::f32, bias1_dims,
            logical_tensor::layout_type::strided};
    logical_tensor relu1_dst_desc_plain {id_mgr["relu1_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::strided};
    std::cout << "Success!\n";

    std::cout << "Compile partition 1----------------------------";
    std::vector<logical_tensor> in1 {
            conv1_src_desc_any, conv1_weight_desc_plain, conv1_bias_desc_plain};
    std::vector<logical_tensor> out1 {relu1_dst_desc_plain};
    auto cp1 = partitions[1].compile(in1, out1, eng);
    std::cout << "Success!\n";

    // Step 5: Prepare tensor and execute compiled partitions
    std::cout << "Prepare tensor and execute compiled partitions-";
    std::vector<float> conv0_src_data(
            static_cast<size_t>(product(input_dims)), 1.0f);
    std::vector<float> conv0_weight_data(
            static_cast<size_t>(product(weight_dims)), 1.0f);
    std::vector<float> conv0_bias_data(
            static_cast<size_t>(product(bias_dims)), 1.0f);
    const float bn_scale = 1.0f;
    const float bn_shift = 1.0f;
    const float bn_mean = 1.0f;
    const float bn_var = 1.0f;
    std::vector<float> bn0_scale_data(
            static_cast<size_t>(product(bias_dims)), bn_scale);
    std::vector<float> bn0_shift_data(
            static_cast<size_t>(product(bias_dims)), bn_shift);
    std::vector<float> bn0_mean_data(
            static_cast<size_t>(product(bias_dims)), bn_mean);
    std::vector<float> bn0_var_data(
            static_cast<size_t>(product(bias_dims)), bn_var);
    std::vector<float> relu0_dst_data(
            cp0.query_logical_tensor(id_mgr["relu0_dst"]).get_mem_size()
                    / sizeof(float),
            0.0);
    std::vector<float> conv1_weight_data(
            static_cast<size_t>(product(weight1_dims)), 1.0f);
    std::vector<float> conv1_bias_data(
            static_cast<size_t>(product(bias1_dims)), 1.0f);
    std::vector<float> relu1_dst_data(
            cp1.query_logical_tensor(id_mgr["relu1_dst"]).get_mem_size()
                    / sizeof(float),
            0.0);

    tensor conv0_src(conv0_src_desc_plain, conv0_src_data.data());
    tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data.data());
    tensor conv0_bias(conv0_bias_desc_plain, conv0_bias_data.data());
    tensor bn0_scale(bn0_scale_desc_plain, bn0_scale_data.data());
    tensor bn0_shift(bn0_shift_desc_plain, bn0_shift_data.data());
    tensor bn0_mean(bn0_mean_desc_plain, bn0_mean_data.data());
    tensor bn0_var(bn0_var_desc_plain, bn0_var_data.data());

    logical_tensor relu0_dst_desc_opaque
            = cp0.query_logical_tensor(id_mgr["relu0_dst"]);
    tensor relu0_dst(relu0_dst_desc_opaque, relu0_dst_data.data());

    std::vector<tensor> in_list_0 {conv0_src, conv0_weight, conv0_bias,
            bn0_scale, bn0_shift, bn0_mean, bn0_var};
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
    float expected_result
            = (bn_scale * ((1 * 11 * 11 * 3 + /* conv0 bias */ 1.0f) - bn_mean)
                              / std::sqrt(bn_var + epsilon)
                      + bn_shift)
                    * (1 * 1 * 96)
            + /* conv1 bias */ 1.0f;

    for (auto v : relu1_dst_data) {
        if (std::abs(expected_result - v) > 1e-6f) {
            throw std::runtime_error(
                    "output result is not equal to excepted "
                    "results");
        }
    }
    std::cout << "Success!\n";

    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
