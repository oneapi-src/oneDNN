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

/// @example cpu_inplace_options.cpp
/// @copybrief cpu_inplace_options_cpp
/// Annotated version: @ref cpu_inplace_options_cpp

/// @page cpu_inplace_options_cpp CPU example support inplace options
///
/// > Example code: @ref cpu_inplace_options.cpp

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

// ======== Unoptimized graph ========
// digraph G {
//   Wildcard_100002 -> Convolution_100003;
//   Convolution_100003 -> BiasAdd_100004;
//   Wildcard_100005 -> Convolution_100006;
//   Convolution_100006 -> BiasAdd_100007;
//   BiasAdd_100004 -> Add_100008;
//   BiasAdd_100007 -> Add_100008;
// }
// ======== Optimized graph ==========
// digraph G {
//   Wildcard_100005 -> Conv_bias_100215;
//   Wildcard_100002 -> Conv_bias_add_100125;
//   Conv_bias_100215 -> Conv_bias_add_100125;
// }
int main(int argc, char **argv) {
    std::cout << "========Example: Add(Conv+BiasAdd, Conv+BiasAdd)========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        std::cout << "Don't support gpu now\n";
        return -1;
    }

    // Step 1: Initialize engine and stream
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
    op conv_input0(id_mgr["conv_input0"], op::kind::Wildcard, "conv_input0");
    op conv_input1(id_mgr["conv_input1"], op::kind::Wildcard, "conv_input1");

    /// conv+bias
    op conv0(id_mgr["conv0"], op::kind::Convolution, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");
    conv0.set_attr<int64_t>("groups", 1);
    op bias0(id_mgr["bias0"], op::kind::BiasAdd, "bias0");
    /// conv+bias
    op conv1(id_mgr["conv1"], op::kind::Convolution, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");
    conv1.set_attr<int64_t>("groups", 1);
    op bias1(id_mgr["bias1"], op::kind::BiasAdd, "bias1");
    // Add
    op add(id_mgr["add"], op::kind::Add, "add");
    std::cout << "Success!\n";

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> conv0_input_dims {8, 256, 56, 56};
    std::vector<int64_t> conv0_weight_dims {64, 256, 1, 1};
    std::vector<int64_t> conv0_bias_dims {64};
    std::vector<int64_t> conv0_dst_dims {-1, -1, -1, -1};

    std::vector<int64_t> conv1_input_dims {8, 256, 56, 56};
    std::vector<int64_t> conv1_weight_dims {64, 256, 1, 1};
    std::vector<int64_t> conv1_bias_dims {64};
    std::vector<int64_t> conv1_dst_dims {-1, -1, -1, -1};

    logical_tensor conv0_src_desc {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, conv0_input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_weight_desc {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_bias_desc {id_mgr["conv0_bias"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_dst_desc {id_mgr["conv0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_bias_dst_desc {id_mgr["conv0_bias_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};

    logical_tensor conv1_src_desc {id_mgr["conv1_src"],
            logical_tensor::data_type::f32, conv1_input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_weight_desc {id_mgr["conv1_weight"],
            logical_tensor::data_type::f32, conv1_weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_bias_desc {id_mgr["conv1_bias"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_dst_desc {id_mgr["conv1_dst"],
            logical_tensor::data_type::f32, conv1_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv1_bias_dst_desc {id_mgr["conv1_bias_dst"],
            logical_tensor::data_type::f32, conv1_dst_dims,
            logical_tensor::layout_type::undef};

    logical_tensor add_dst_desc {id_mgr["add_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};
    std::cout << "Success!\n";

    /// Add Input/Output
    std::cout << "Add logical tensor to op-----------------------";
    conv_input0.add_output(conv0_src_desc);
    conv0.add_inputs({conv0_src_desc, conv0_weight_desc});
    conv0.add_output(conv0_dst_desc);
    bias0.add_inputs({conv0_dst_desc, conv0_bias_desc});
    bias0.add_output(conv0_bias_dst_desc);

    conv_input1.add_output(conv1_src_desc);
    conv1.add_inputs({conv1_src_desc, conv1_weight_desc});
    conv1.add_output(conv1_dst_desc);
    bias1.add_inputs({conv1_dst_desc, conv1_bias_desc});
    bias1.add_output(conv1_bias_dst_desc);

    add.add_inputs({conv0_bias_dst_desc, conv1_bias_dst_desc});
    add.add_output(add_dst_desc);

    std::cout << "Success!\n";

    /// Add OP
    std::cout << "Add OP to graph--------------------------------";
    g.add_op(conv_input0);
    g.add_op(conv0);
    g.add_op(bias0);
    g.add_op(conv_input1);
    g.add_op(conv1);
    g.add_op(bias1);
    g.add_op(add);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter and get partitions
    /// Graph will be filtered into 2 partitions: `conv0+bias0+sum and `conv1+bias1`
    /// `export DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter and get partition-----------------------";
    auto partitions = g.get_partitions(partition::policy::fusion);

    if (partitions.size() != 2) {
        throw std::runtime_error("wrong partition number");
    }
    std::cout << "Success!\n";

    // Step 4: Prepare logical tensors with proper format and compile partitions
    std::cout << "Prepare logical tensors with proper format-----";
    logical_tensor conv1_src_desc_plain {id_mgr["conv1_src"],
            logical_tensor::data_type::f32, conv1_input_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv1_weight_desc_plain {id_mgr["conv1_weight"],
            logical_tensor::data_type::f32, conv1_weight_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv1_bias_desc_plain {id_mgr["conv1_bias"],
            logical_tensor::data_type::f32, conv1_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv1_bias_dst_desc_any {id_mgr["conv1_bias_dst"],
            logical_tensor::data_type::f32, conv1_dst_dims,
            logical_tensor::layout_type::any};

    std::cout << "Success!\n";

    std::cout << "Infer shape for partition 0--------------------";
    std::vector<logical_tensor> in0 {conv1_src_desc_plain,
            conv1_weight_desc_plain, conv1_bias_desc_plain};
    std::vector<logical_tensor> out0 {conv1_bias_dst_desc_any};
    partitions[0].infer_shape(in0, out0);
    std::cout << "Success!\n";

    std::cout << "Compile partition 0----------------------------";
    auto cp0 = partitions[0].compile(in0, out0, eng);
    std::cout << "Success!\n";

    std::cout << "Query logical tensor from compiled partition 0: \n";
    auto infered_conv1_bias_dst_desc
            = cp0.query_logical_tensor(id_mgr["conv1_bias_dst"]);
    auto infered_conv1_bias_dst_dims = infered_conv1_bias_dst_desc.get_dims();
    std::cout << "    Infered output shape: " << infered_conv1_bias_dst_dims[0]
              << ", " << infered_conv1_bias_dst_dims[1] << ", "
              << infered_conv1_bias_dst_dims[2] << ", "
              << infered_conv1_bias_dst_dims[3] << "\n";
    std::cout << "    Infered output layout type: "
              << static_cast<int>(infered_conv1_bias_dst_desc.get_layout_type())
              << '\n';

    std::cout << "Prepare logical tensors with proper format-----";
    logical_tensor conv0_src_desc_plain {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, conv0_input_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc_plain {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_bias_desc_plain {id_mgr["conv0_bias"],
            logical_tensor::data_type::f32, conv0_bias_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_bias_add_dst_desc_any {id_mgr["add_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::any};
    std::cout << "Success!\n";

    std::cout << "Infer shape for partition 1--------------------";
    std::vector<logical_tensor> in1 {conv0_src_desc_plain,
            conv0_weight_desc_plain, conv0_bias_desc_plain,
            infered_conv1_bias_dst_desc};
    std::vector<logical_tensor> out1 {conv0_bias_add_dst_desc_any};
    partitions[1].infer_shape(in1, out1);
    std::cout << "Success!\n";

    std::cout << "Compile partition 1----------------------------";
    auto cp1 = partitions[1].compile(in1, out1, eng);
    std::cout << "Success!\n";

    std::cout << "Query logical tensor from compiled partition 0: \n";
    auto infered_conv0_bias_add_dst_desc
            = cp1.query_logical_tensor(id_mgr["add_dst"]);
    auto infered_conv0_bias_add_dst_dims
            = infered_conv0_bias_add_dst_desc.get_dims();
    std::cout << "    Infered output shape: "
              << infered_conv0_bias_add_dst_dims[0] << ", "
              << infered_conv0_bias_add_dst_dims[1] << ", "
              << infered_conv0_bias_add_dst_dims[2] << ", "
              << infered_conv0_bias_add_dst_dims[3] << "\n";
    std::cout << "    Infered output layout type: "
              << static_cast<int>(
                         infered_conv0_bias_add_dst_desc.get_layout_type())
              << '\n';

    // Step 5: Get inplace options.
    std::cout << "Get inplace options from partitions------------";
    auto inplace_options0 = cp0.get_inplace_options();
    auto inplace_options1 = cp1.get_inplace_options();
    std::cout << "Success!\n";
    std::cout << "    Partition 0 has " << inplace_options0.size()
              << " in-place option(s)\n";
    std::cout << "    Partition 1 has " << inplace_options1.size()
              << " in-place option(s)\n";

    // Step 6: Prepare tensor and execute compiled partitions
    std::cout << "Prepare tensor and execute compiled partitions-";
    std::vector<float> conv0_src_data(
            static_cast<size_t>(product(conv0_input_dims)), 1.0f);
    std::vector<float> conv0_weight_data(
            static_cast<size_t>(product(conv0_weight_dims)), 1.0f);
    std::vector<float> conv0_bias_data(
            static_cast<size_t>(product(conv0_bias_dims)), 1.0f);
    std::vector<float> add_dst_data(
            infered_conv0_bias_add_dst_desc.get_mem_size() / sizeof(float),
            0.f);

    std::vector<float> conv1_src_data(
            static_cast<size_t>(product(conv1_input_dims)), 1.0f);
    std::vector<float> conv1_weight_data(
            static_cast<size_t>(product(conv1_weight_dims)), 1.0f);
    std::vector<float> conv1_bias_data(
            static_cast<size_t>(product(conv1_bias_dims)), 1.0f);
    std::vector<float> conv1_dst_data(
            infered_conv1_bias_dst_desc.get_mem_size() / sizeof(float), 0.f);

    tensor conv1_src(conv1_src_desc_plain, conv1_src_data.data());
    tensor conv1_weight(conv1_weight_desc_plain, conv1_weight_data.data());
    tensor conv1_bias(conv1_bias_desc_plain, conv1_bias_data.data());
    tensor conv1_dst(infered_conv1_bias_dst_desc, conv1_dst_data.data());
    std::vector<tensor> in_list_0 {conv1_src, conv1_weight, conv1_bias};
    std::vector<tensor> out_list_0 {conv1_dst};

    tensor conv0_src(conv0_src_desc_plain, conv0_src_data.data());
    tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data.data());
    tensor conv0_bias(conv0_bias_desc_plain, conv0_bias_data.data());
    tensor add_dst(infered_conv0_bias_add_dst_desc, add_dst_data.data());
    std::vector<tensor> in_list_1 {
            conv0_src, conv0_weight, conv0_bias, conv1_dst};
    std::vector<tensor> out_list_1 {add_dst};

    for (auto &&p : inplace_options1) {
        size_t input_id = p.first;
        size_t output_id = p.second;
        auto input_lt_iter = std::find_if(
                in1.begin(), in1.end(), [input_id](logical_tensor &lt) {
                    return input_id == lt.get_id();
                });
        auto input_lt_idx = static_cast<size_t>(
                std::distance(in1.begin(), input_lt_iter));
        auto output_lt_iter = std::find_if(
                out1.begin(), out1.end(), [output_id](logical_tensor &lt) {
                    return output_id == lt.get_id();
                });
        auto output_lt_idx = static_cast<size_t>(
                std::distance(out1.begin(), output_lt_iter));
        out_list_1[output_lt_idx].set_data_handle(
                in_list_1[input_lt_idx].get_data_handle<float>());
    }

    cp0.execute(strm, in_list_0, out_list_0);
    cp1.execute(strm, in_list_1, out_list_1);

    std::cout << "Success!\n";

    // Step 6: Check correctness of the output results
    std::cout << "Check correctness------------------------------";
    float expected_result = (1 * 1 * 1 * 256 + /* conv0 bias */ 1.f) * 2.f;

    float *dst_data = out_list_1[0].get_data_handle<float>();
    // Because the input/output channel is divisiable by 8(avx2) or 16(avx512),
    // we can check the result elementwise-ly.
    for (size_t i = 0; i < out_list_1[0].get_element_num(); ++i) {
        if (std::abs(dst_data[i] - expected_result) > 1e-6f) {
            throw std::runtime_error(
                    "output result is not equal to excepted "
                    "results");
        }
    }
    std::cout << "Success!\n";

    std::cout << "============Run Example Successfully===========\n";
    return 0;
}
