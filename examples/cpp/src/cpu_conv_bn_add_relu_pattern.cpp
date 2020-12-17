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

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/utils.hpp"

using namespace dnnl::graph;

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
int main(int argc, char **argv) {
    std::cout << "========Example: Conv+BN+ReLU+Conv+BN+Add+ReLU========\n";

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
    allocator alloc {&allocate, &deallocate};
    eng.set_allocator(alloc);
    stream strm {eng};
    std::cout << "Success!\n";

    // Step 2: Construct a graph
    graph g(engine_kind);

    auto &id_mgr = logical_id_manager::get();

    /// Create OP and set attributes
    std::cout << "Create op--------------------------------------";
    /// inuput node
    op input0(id_mgr["input0"], op::kind::Wildcard, "input0");

    /// conv+bn+relu
    op conv0(id_mgr["conv0"], op::kind::Convolution, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NXC");
    conv0.set_attr<std::string>("filter_format", "XIO");
    conv0.set_attr<int64_t>("groups", 1);
    op bn0(id_mgr["bn0"], op::kind::BatchNormInference, "bn0");
    bn0.set_attr<float>("epsilon", 0.f);
    bn0.set_attr<std::string>("data_format", "NXC");
    op relu0(id_mgr["relu0"], op::kind::ReLU, "relu0");
    /// conv+bn+add+relu
    op conv1(id_mgr["conv1"], op::kind::Convolution, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<std::string>("data_format", "NXC");
    conv1.set_attr<std::string>("filter_format", "OIX");
    conv1.set_attr<int64_t>("groups", 1);
    op bn1(id_mgr["bn1"], op::kind::BatchNormInference, "bn1");
    bn1.set_attr<float>("epsilon", 0.f);
    bn1.set_attr<std::string>("data_format", "NXC");
    op add0(id_mgr["add0"], op::kind::Add, "add0");
    op relu1(id_mgr["relu1"], op::kind::ReLU, "relu1");
    std::cout << "Success!\n";

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input_dims {8, 56, 56, 256};
    std::vector<int64_t> conv0_weight_dims {1, 1, 256, 64};
    std::vector<int64_t> conv0_bias_dims {64};
    std::vector<int64_t> conv0_dst_dims {-1, -1, -1, -1};

    std::vector<int64_t> conv1_weight_dims {256, 64, 1, 1};
    std::vector<int64_t> conv1_bias_dims {256};
    std::vector<int64_t> conv1_dst_dims {-1, -1, -1, -1};

    logical_tensor conv0_src_desc {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_weight_desc {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_dst_desc {id_mgr["conv0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_scale_desc {id_mgr["bn0_scale"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_shift_desc {id_mgr["bn0_shift"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_mean_desc {id_mgr["bn0_mean"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_var_desc {id_mgr["bn0_var"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn0_dst_desc {id_mgr["bn0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu0_dst_desc {id_mgr["relu0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};

    logical_tensor conv1_weight_desc {id_mgr["conv1_weight"],
            logical_tensor::data_type::f32, conv1_weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_dst_desc {id_mgr["conv1_dst"],
            logical_tensor::data_type::f32, conv1_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn1_scale_desc {id_mgr["bn1_scale"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn1_shift_desc {id_mgr["bn1_shift"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn1_mean_desc {id_mgr["bn1_mean"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn1_var_desc {id_mgr["bn1_var"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor bn1_dst_desc {id_mgr["bn1_dst"],
            logical_tensor::data_type::f32, conv1_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor add0_dst_desc {id_mgr["add0_dst"],
            logical_tensor::data_type::f32, conv1_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu1_dst_desc {id_mgr["relu1_dst"],
            logical_tensor::data_type::f32, conv1_dst_dims,
            logical_tensor::layout_type::undef};
    std::cout << "Success!\n";

    /// Add Input/Output
    std::cout << "Add logical tensor to op-----------------------";
    input0.add_output(conv0_src_desc);

    conv0.add_inputs({conv0_src_desc, conv0_weight_desc});
    conv0.add_output(conv0_dst_desc);
    bn0.add_inputs({conv0_dst_desc, bn0_scale_desc, bn0_shift_desc,
            bn0_mean_desc, bn0_var_desc});
    bn0.add_output(bn0_dst_desc);
    relu0.add_input(bn0_dst_desc);
    relu0.add_output(relu0_dst_desc);

    conv1.add_inputs({relu0_dst_desc, conv1_weight_desc});
    conv1.add_output(conv1_dst_desc);
    bn1.add_inputs({conv1_dst_desc, bn1_scale_desc, bn1_shift_desc,
            bn1_mean_desc, bn1_var_desc});
    bn1.add_output(bn1_dst_desc);
    add0.add_inputs({bn1_dst_desc, conv0_src_desc});
    add0.add_output(add0_dst_desc);
    relu1.add_input(add0_dst_desc);
    relu1.add_output(relu1_dst_desc);
    std::cout << "Success!\n";

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
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter and get partitions
    /// Graph will be filtered into 2 partitions: `conv0+bn0+relu0` and `conv1+bn1+add0+relu1`
    /// `export DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter and get partition-----------------------";
    auto partitions = g.get_partitions(partition::policy::fusion);

    if (partitions.size() != 2) {
        throw std::runtime_error("wrong partition number");
    }
    std::cout << "Success!\n";

    // Step 4: Prepare logical tensors with proper format and compile partitions
    std::cout << "Prepare logical tensors with proper format-----";
    logical_tensor conv0_src_desc_plain {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc_plain {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn0_scale_desc_plain {id_mgr["bn0_scale"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn0_shift_desc_plain {id_mgr["bn0_shift"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn0_mean_desc_plain {id_mgr["bn0_mean"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn0_var_desc_plain {id_mgr["bn0_var"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor relu0_dst_desc_any {id_mgr["relu0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::any};
    std::cout << "Success!\n";

    std::cout << "Infer shape for partition 0--------------------";
    std::vector<logical_tensor> in0 {conv0_src_desc_plain,
            conv0_weight_desc_plain, bn0_scale_desc_plain, bn0_shift_desc_plain,
            bn0_mean_desc_plain, bn0_var_desc_plain};
    std::vector<logical_tensor> out0 {relu0_dst_desc_any};
    partitions[0].infer_shape(in0, out0);
    std::cout << "Success!\n";
    std::vector<int64_t> infered_relu0_dst_dims = out0[0].get_dims();
    std::cout << "Infered partition 1 output shape:"
              << infered_relu0_dst_dims[0] << "," << infered_relu0_dst_dims[1]
              << "," << infered_relu0_dst_dims[2] << ","
              << infered_relu0_dst_dims[3] << "\n";

    std::cout << "Compile partition 0----------------------------";
    auto cp0 = partitions[0].compile(in0, out0, eng);
    std::cout << "Success!\n";

    std::cout << "Query logical tensor from compiled partition 0-";
    logical_tensor conv0_src_desc_opaque
            = cp0.query_logical_tensor(id_mgr["relu0_dst"]);

    logical_tensor conv1_weight_desc_plain {id_mgr["conv1_weight"],
            logical_tensor::data_type::f32, conv1_weight_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn1_scale_desc_plain {id_mgr["bn1_scale"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn1_shift_desc_plain {id_mgr["bn1_shift"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn1_mean_desc_plain {id_mgr["bn1_mean"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor bn1_var_desc_plain {id_mgr["bn1_var"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor relu1_dst_desc_plain {id_mgr["relu1_dst"],
            logical_tensor::data_type::f32, conv1_dst_dims,
            logical_tensor::layout_type::strided};
    std::cout << "Success!\n";

    std::cout << "Infer shape for partition 1--------------------";
    std::vector<logical_tensor> in1 {conv0_src_desc_opaque,
            conv1_weight_desc_plain, bn1_scale_desc_plain, bn1_shift_desc_plain,
            bn1_mean_desc_plain, bn1_var_desc_plain, conv0_src_desc_plain};
    std::vector<logical_tensor> out1 {relu1_dst_desc_plain};
    partitions[1].infer_shape(in1, out1);
    std::cout << "Success!\n";
    std::vector<int64_t> infered_relu1_dst_dims = out1[0].get_dims();
    std::cout << "Infered partition 0 output shape:"
              << infered_relu1_dst_dims[0] << "," << infered_relu1_dst_dims[1]
              << "," << infered_relu1_dst_dims[2] << ","
              << infered_relu1_dst_dims[3] << "\n";

    std::cout << "Compile partition 1----------------------------";
    auto cp1 = partitions[1].compile(in1, out1, eng);
    relu1_dst_desc_plain = cp1.query_logical_tensor(id_mgr["relu1_dst"]);
    std::cout << "Success!\n";

    // Step 5: Prepare tensor and execute compiled partitions
    const float bn_scale = 1.0, bn_shift = 1.0, bn_mean = 1.0, bn_var = 1.0;
    std::cout << "Prepare tensor and execute compiled partitions-";
    std::vector<float> conv0_src_data(
            static_cast<size_t>(product(input_dims)), 1.0f);
    std::vector<float> conv0_weight_data(
            static_cast<size_t>(product(conv0_weight_dims)), 1.0f);
    std::vector<float> bn0_scale_data(
            static_cast<size_t>(product(conv0_bias_dims)), bn_scale);
    std::vector<float> bn0_shift_data(
            static_cast<size_t>(product(conv0_bias_dims)), bn_shift);
    std::vector<float> bn0_mean_data(
            static_cast<size_t>(product(conv0_bias_dims)), bn_mean);
    std::vector<float> bn0_var_data(
            static_cast<size_t>(product(conv0_bias_dims)), bn_var);
    std::vector<float> relu0_dst_data(
            cp0.query_logical_tensor(id_mgr["relu0_dst"]).get_mem_size()
                    / sizeof(float),
            0.0);

    std::vector<float> conv1_weight_data(
            static_cast<size_t>(product(conv1_weight_dims)), 1.0f);
    std::vector<float> bn1_scale_data(
            static_cast<size_t>(product(conv1_bias_dims)), bn_scale);
    std::vector<float> bn1_shift_data(
            static_cast<size_t>(product(conv1_bias_dims)), bn_shift);
    std::vector<float> bn1_mean_data(
            static_cast<size_t>(product(conv1_bias_dims)), bn_mean);
    std::vector<float> bn1_var_data(
            static_cast<size_t>(product(conv1_bias_dims)), bn_var);
    std::vector<float> relu1_dst_data(
            cp1.query_logical_tensor(id_mgr["relu1_dst"]).get_mem_size()
                    / sizeof(float),
            0.0);

    tensor conv0_src(conv0_src_desc_plain, conv0_src_data.data());
    tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data.data());
    tensor bn0_scale(bn0_scale_desc_plain, bn0_scale_data.data());
    tensor bn0_shift(bn0_shift_desc_plain, bn0_shift_data.data());
    tensor bn0_mean(bn0_mean_desc_plain, bn0_mean_data.data());
    tensor bn0_var(bn0_var_desc_plain, bn0_var_data.data());
    tensor relu0_dst(cp0.query_logical_tensor(id_mgr["relu0_dst"]),
            relu0_dst_data.data());

    std::vector<tensor> in_list_0 {
            conv0_src, conv0_weight, bn0_scale, bn0_shift, bn0_mean, bn0_var};
    std::vector<tensor> out_list_0 {relu0_dst};
    cp0.execute(strm, in_list_0, out_list_0);

    tensor conv1_weight(conv1_weight_desc_plain, conv1_weight_data.data());
    tensor bn1_scale(bn1_scale_desc_plain, bn1_scale_data.data());
    tensor bn1_shift(bn1_shift_desc_plain, bn1_shift_data.data());
    tensor bn1_mean(bn1_mean_desc_plain, bn1_mean_data.data());
    tensor bn1_var(bn1_var_desc_plain, bn1_var_data.data());
    tensor relu1_dst(relu1_dst_desc_plain, relu1_dst_data.data());

    std::vector<tensor> in_list_1 {relu0_dst, conv1_weight, bn1_scale,
            bn1_shift, bn1_mean, bn1_var, conv0_src};
    std::vector<tensor> out_list_1 {relu1_dst};
    cp1.execute(strm, in_list_1, out_list_1);

    std::cout << "Success!\n";

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
