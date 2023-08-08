/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_TEST_GRAPH_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_TEST_GRAPH_HPP
#include <memory>
#include <string>
#include <vector>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <ops/templates/conv_fwd.hpp>
#include <ops/templates/managed_matmul_core.hpp>
#include <util/any_map.hpp>
using namespace dnnl::impl::graph::gc;

inline graph_tensor_ptr make_tensor(const sc_dims &shape,
        sc_data_type_t dtype = sc_data_type_t(sc_data_etype::F32, 1)) {
    return std::make_shared<graph_tensor>(
            nullptr, sc_data_format_t(), shape, dtype);
};

inline void get_logical_tensors(
        ltensors *ins, const std::vector<graph_tensor_ptr> &flts) {
    ins->reserve(flts.size());
    for (auto &in : flts) {
        ins->emplace_back(in->details_);
    }
}

inline sc_graph_t get_test_graph(sc_op_ptr &in_a, sc_op_ptr &in_b,
        sc_op_ptr &in_weight, sc_op_ptr &in_c) {
    sc_graph_t mgr;
    // make a graph of:
    // out = relu(conv(data=a+b, weight)) + c
    in_a = mgr.make_input({make_tensor({28, 64, 16, 16})}); // 0
    in_b = mgr.make_input({make_tensor({28, 64, 16, 16})}); // 1
    auto conv_data
            = mgr.make("add", {in_a->get_outputs()[0], in_b->get_outputs()[0]},
                    {make_tensor({28, 64, 16, 16})}, {}); // 2

    in_weight = mgr.make_input({make_tensor({64, 64, 1, 1})}); // 3
    auto conv_out = mgr.make("conv_fwd_core",
            {conv_data->get_outputs()[0], in_weight->get_outputs()[0]}, {},
            {{"strides", sc_dims {1, 1}}, {"paddings", sc_dims {0, 0}}});
    auto relu_out = mgr.make("relu", {conv_out->get_outputs()[0]}, {}, {}); // 5
    in_c = mgr.make_input({make_tensor({28, 64, 16, 16})}); // 6
    auto add_out = mgr.make("add",
            {relu_out->get_outputs()[0], in_c->get_outputs()[0]}, {}, {}); // 7
    auto out = mgr.make_output(add_out->get_outputs()); // 8
    return mgr;
}

inline sc_graph_t get_test_sorting_graph(
        sc_op_ptr &in_a, const sc_dims &src_dim) {
    // just test for last dim
    int rd_axis = (int)src_dim.size() - 1;
    sc_dims rd_dim(src_dim);
    rd_dim.at(rd_axis) = 1;

    sc_graph_t graph;
    // make a graph of layernorm
    in_a = graph.make_input({make_tensor(src_dim)}); // 0
    // x^2
    auto fsquare = graph.make("mul",
            {in_a->get_outputs()[0], in_a->get_outputs()[0]},
            {make_tensor(src_dim)}, {}); // 1
    // mean of x^2
    auto fsqd_mean = graph.make("reduce", {fsquare->get_outputs()[0]},
            {make_tensor(rd_dim)},
            {{"need_mean", true}, {"rd_axis", std::vector<int> {rd_axis}},
                    {"rd_op", 0}}); // 2
    // mean of X
    auto fmean = graph.make("reduce", {in_a->get_outputs()[0]},
            {make_tensor(rd_dim)},
            {{"need_mean", true}, {"rd_axis", std::vector<int> {rd_axis}},
                    {"rd_op", 0}}); // 3
    // square of mean
    auto fmean_sqd = graph.make("mul",
            {fmean->get_outputs()[0], fmean->get_outputs()[0]},
            {make_tensor(rd_dim)}, {}); // 4
    // x-x_mean
    auto fdiff = graph.make("sub",
            {in_a->get_outputs()[0], fmean->get_outputs()[0]},
            {make_tensor(src_dim)}, {}); // 5
    auto fvar = graph.make("sub",
            {fsqd_mean->get_outputs()[0], fmean_sqd->get_outputs()[0]},
            {make_tensor(rd_dim)}, {}); // 6
    // rsqrt
    auto frsqd_root = graph.make("squared_root", {fvar->get_outputs()[0]},
            {make_tensor(rd_dim)}, {{"reciprocal", true}}); // 7
    auto fmul = graph.make("mul",
            {fdiff->get_outputs()[0], frsqd_root->get_outputs()[0]},
            {make_tensor(src_dim)}, {}); // 8

    auto out = graph.make_output(fmul->get_outputs()); // 9
    return graph;
}

inline sc_graph_t get_test_speculative_graph() {
    sc_graph_t mgr;
    // make a graph of:
    auto in_a = mgr.make_input({make_tensor({32, 64})}); // 0
    auto relu0 = mgr.make("relu", {in_a->get_outputs()[0]}, {}, {}); // 1
    auto relu1 = mgr.make("relu", {relu0->get_outputs()[0]}, {}, {}); // 2
    auto in_b = mgr.make_input({make_tensor({128, 32, 64})}); // 3
    auto add = mgr.make("add",
            {relu0->get_outputs()[0], in_b->get_outputs()[0]}, {}, {}); // 4
    auto in_c = mgr.make_input({make_tensor({64, 32})}); // 5
    auto matmul = mgr.make("matmul_core",
            {relu0->get_outputs()[0], in_c->get_outputs()[0]}, {}, {}); // 6

    mgr.make_output(matmul->get_outputs()); // 7
    mgr.make_output(add->get_outputs()); // 8
    mgr.make_output(relu1->get_outputs()); // 9
    return mgr;
}

inline sc_graph_t get_conv_bn_relu_graph(const sc_dims &src_dims,
        const sc_dims &weight_dims, const sc_dims &dst_dims,
        const sc_dims &strides, const sc_dims &paddings,
        const sc_dims &mean_dims, const sc_dims &var_dims,
        const sc_dims &shift_dims, const sc_dims &scale_dims,
        const std::string &data_format, float epsilon) {
    sc_graph_t graph;
    // make a graph of:
    // out = relu(bn(conv))
    auto in = graph.make_input({make_tensor(src_dims), make_tensor(weight_dims),
            make_tensor(scale_dims), make_tensor(shift_dims),
            make_tensor(mean_dims), make_tensor(var_dims)});
    auto conv_out = graph.make("conv_fwd_core",
            {in->get_outputs()[0], in->get_outputs()[1]},
            {make_tensor(dst_dims)},
            {{"strides", strides}, {"paddings", paddings}});
    auto bn_out = graph.make("batchnorm_inference",
            {conv_out->get_outputs()[0], in->get_outputs()[2],
                    in->get_outputs()[3], in->get_outputs()[4],
                    in->get_outputs()[5]},
            {make_tensor(dst_dims)},
            {{"epsilon", epsilon}, {"data_format", data_format}});
    auto relu_out = graph.make(
            "relu", {bn_out->get_outputs()[0]}, {make_tensor(dst_dims)}, {});
    auto out = graph.make_output(relu_out->get_outputs());
    return graph;
}

inline sc_graph_t get_conv_relu_graph(const sc_dims &src_dims,
        const sc_dims &weight_dims, const sc_dims &dst_dims,
        const sc_dims &strides, const sc_dims &paddings) {
    sc_graph_t graph;
    // make a graph of:
    // out = relu(conv)
    auto in = graph.make_input(
            {make_tensor(src_dims), make_tensor(weight_dims)});
    auto conv_out = graph.make("conv_fwd_core",
            {in->get_outputs()[0], in->get_outputs()[1]},
            {make_tensor(dst_dims)},
            {{"strides", strides}, {"paddings", paddings}});
    auto relu_out = graph.make(
            "relu", {conv_out->get_outputs()[0]}, {make_tensor(dst_dims)}, {});
    auto out = graph.make_output(relu_out->get_outputs());
    return graph;
}

inline sc_graph_t get_conv_relu_add_graph(const sc_dims &src0_dims,
        const sc_dims &weight_dims, const sc_dims &src1_dims,
        const sc_dims &dst_dims, const sc_dims &strides,
        const sc_dims &paddings) {
    sc_graph_t graph;
    // make a graph of:
    // out = relu(conv) + add
    auto in = graph.make_input({make_tensor(src0_dims),
            make_tensor(weight_dims), make_tensor(src1_dims)});
    auto conv_out = graph.make("conv_fwd_core",
            {in->get_outputs()[0], in->get_outputs()[1]},
            {make_tensor(dst_dims)},
            {{"strides", strides}, {"paddings", paddings}});
    auto relu_out = graph.make(
            "relu", {conv_out->get_outputs()[0]}, {make_tensor(dst_dims)}, {});

    auto add_out = graph.make(
            "add", {relu_out->get_outputs()[0], in->get_outputs()[2]}, {}, {});
    auto out = graph.make_output(add_out->get_outputs());
    return graph;
}

inline sc_graph_t get_conv_bn_add_relu_graph(const sc_dims &src_dims,
        const sc_dims &weight_dims, const sc_dims &dst_dims,
        const sc_dims &strides, const sc_dims &paddings,
        const sc_dims &mean_dims, const sc_dims &var_dims,
        const sc_dims &shift_dims, const sc_dims &scale_dims,
        const std::string &data_format, float epsilon) {
    sc_graph_t graph;
    // make a graph of:
    // out = relu(add(bn(conv(relu(bn(conv))))))
    auto in = graph.make_input({make_tensor(src_dims), make_tensor(weight_dims),
            make_tensor(scale_dims), make_tensor(shift_dims),
            make_tensor(mean_dims), make_tensor(var_dims)});
    auto conv_out = graph.make("conv_fwd_core",
            {in->get_outputs()[0], in->get_outputs()[1]},
            {make_tensor(dst_dims)},
            {{"strides", strides}, {"paddings", paddings}});
    auto bn_out = graph.make("batchnorm_inference",
            {conv_out->get_outputs()[0], in->get_outputs()[2],
                    in->get_outputs()[3], in->get_outputs()[4],
                    in->get_outputs()[5]},
            {make_tensor(dst_dims)},
            {{"epsilon", epsilon}, {"data_format", data_format}});
    auto relu_out = graph.make(
            "relu", {bn_out->get_outputs()[0]}, {make_tensor(dst_dims)}, {});
    auto in1 = graph.make_input(
            {make_tensor(weight_dims), make_tensor(scale_dims),
                    make_tensor(shift_dims), make_tensor(mean_dims),
                    make_tensor(var_dims), make_tensor(dst_dims)});
    auto conv_out1 = graph.make("conv_fwd_core",
            {relu_out->get_outputs()[0], in1->get_outputs()[0]},
            {make_tensor(dst_dims)},
            {{"strides", strides}, {"paddings", paddings}});
    auto bn_out1 = graph.make("batchnorm_inference",
            {conv_out1->get_outputs()[0], in1->get_outputs()[1],
                    in1->get_outputs()[2], in1->get_outputs()[3],
                    in1->get_outputs()[4]},
            {make_tensor(dst_dims)},
            {{"epsilon", epsilon}, {"data_format", data_format}});
    auto add_out1 = graph.make(
            "add", {bn_out1->get_outputs()[0], in1->get_outputs()[5]}, {}, {});
    auto relu_out1 = graph.make(
            "relu", {add_out1->get_outputs()[0]}, {make_tensor(dst_dims)}, {});
    auto out = graph.make_output(relu_out1->get_outputs());
    return graph;
}

inline sc_graph_t get_parallel_merge_mlp_graph(int M = 10752, int N = 1024,
        int K = 1024,
        const ops::managed_matmul_core_config_t mmm_cfg = {14, 2, 1, 1, 1, 1}) {
    sc_graph_t graph;

    auto input0 = graph.make_input(
            {graph_tensor::make({M, K}, sc_data_format_t(format_kinds::MN))});
    auto weight0 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    auto weight1 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    auto weight2 = graph.make_input(
            {graph_tensor::make({K, N}, sc_data_format_t(format_kinds::KN))});
    // mmm0
    auto mmm0 = graph.make("managed_matmul_core",
            {input0->get_outputs()[0], weight0->get_outputs()[0]}, {}, {});
    mmm0->stc_cast<tunable_op_t>()->set_config(
            reflection::general_object_t::make(mmm_cfg));
    // mmm1
    auto mmm1 = graph.make("managed_matmul_core",
            {mmm0->get_outputs()[0], weight1->get_outputs()[0]}, {}, {});
    mmm1->stc_cast<tunable_op_t>()->set_config(
            reflection::general_object_t::make(mmm_cfg));
    mmm1 = graph.make("gelu", {mmm1->get_outputs()[0]}, {}, {});

    // mmm2
    auto mmm2 = graph.make("managed_matmul_core",
            {mmm1->get_outputs()[0], weight2->get_outputs()[0]}, {}, {});
    mmm2->stc_cast<tunable_op_t>()->set_config(
            reflection::general_object_t::make(mmm_cfg));
    mmm2 = graph.make(
            "add", {mmm1->get_outputs()[0], mmm2->get_outputs()[0]}, {}, {});
    graph.make_output(mmm2->get_outputs());
    return graph;
}

#endif
