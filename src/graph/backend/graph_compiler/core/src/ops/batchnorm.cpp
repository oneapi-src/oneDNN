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
#include "batchnorm.hpp"
#include <memory>
#include <string>
#include <utility>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

batchnorm_inference_op::batchnorm_inference_op(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "batchnorm_inference";
}

void batchnorm_inference_op::get_graph_impl(
        std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    float epsilon = attrs_.get<float>("epsilon");
    std::string format = attrs_.get<std::string>("data_format");
    auto bc_axis = format == "NCX"
            ? std::vector<int> {1}
            : std::vector<int> {static_cast<int>(
                    info_.inputs_[0]->details_.get_plain_dims().size() - 1)};
    // input
    graph->make_input(inputs);
    // eps constant;
    auto const_op = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {epsilon}),
            datatypes::f32, sc_dims {1});
    // variance+eps
    auto var_eps = graph->make(
            "add", {inputs[4], const_op->get_outputs()[0]}, {}, {});
    // rsqrt(variance+eps)
    auto rsqrt_op = graph->make("squared_root", {var_eps->get_outputs()[0]}, {},
            any_map_t({{"reciprocal", true}}));
    // gamma *rsqrt(variance+eps)
    auto bn_mul = graph->make(
            "mul", {inputs[1], rsqrt_op->get_outputs()[0]}, {}, {});
    // mean * gamma *rsqrt(variance+eps)
    auto mean_op
            = graph->make("mul", {inputs[3], bn_mul->get_outputs()[0]}, {}, {});
    // beta - mean*gamma*rsqrt(variance+eps)
    auto bn_add = graph->make(
            "sub", {inputs[2], mean_op->get_outputs()[0]}, {}, {});

    auto x1 = graph->make("mul", {inputs[0], bn_mul->get_outputs()[0]}, {},
            any_map_t({{"bc_axis", bc_axis}}));

    auto y1 = graph->make("add",
            {x1->get_outputs()[0], bn_add->get_outputs()[0]}, {outputs[0]},
            any_map_t({{"bc_axis", bc_axis}}));
    // output
    graph->make_output(y1->get_outputs());
}

void batchnorm_inference_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

batchnorm_forward_training_op::batchnorm_forward_training_op(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    COMPILE_ASSERT((ins.size() == 5),
            "batchnorm_forward_training inputs size should be 5.");
    for (int i = 2; i < 5; i++) {
        COMPILE_ASSERT(ins[1]->details_.dtype_ == ins[i]->details_.dtype_,
                "wrong data type");
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
        for (int i = 0; i < 4; i++) {
            info_.outputs_.emplace_back(
                    std::make_shared<graph_tensor>(this, ins[4]->details_));
        }
    } else {
        COMPILE_ASSERT((outs.size() == 5),
                "batchnorm_forward_training outputs size should be 5.");
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "batchnorm_forward_training";
}

void batchnorm_forward_training_op::get_graph_impl(
        std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    std::string format = attrs_.get_or_else("data_format", std::string("NXC"));

    // input
    graph->make_input(inputs);
    bool is_src_bf16
            = info_.inputs_[0]->details_.dtype_.is_etype(sc_data_etype::BF16);
    bool is_ssmv_bf16
            = info_.inputs_[1]->details_.dtype_.is_etype(sc_data_etype::BF16);
    bool is_3D = (inputs[0]->details_.get_plain_dims().size() == 5);
    auto src = inputs[0], mean = inputs[1], variance = inputs[2],
         scale = inputs[3], shift = inputs[4];
    auto src_pass2 = inputs[0];

    auto epsilon = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(
                    std::vector<float> {attrs_.get_or_else("epsilon", 1e-5f)}),
            datatypes::f32, sc_dims {1});
    if (is_src_bf16) {
        auto cast0 = graph->make(
                "cast", {inputs[0]}, {}, {{"dtype", datatypes::f32}});
        src = cast0->get_outputs()[0];
        auto cast0_pass2 = graph->make("cast", {inputs[0]}, {},
                {{"dtype", datatypes::f32},
                        {op_attr_key::break_pre_fuse, true}});
        src_pass2 = cast0_pass2->get_outputs()[0];
    }
    if (is_ssmv_bf16) {
        auto cast1 = graph->make(
                "cast", {inputs[1]}, {}, {{"dtype", datatypes::f32}});
        scale = cast1->get_outputs()[0];
        auto cast2 = graph->make(
                "cast", {inputs[2]}, {}, {{"dtype", datatypes::f32}});
        shift = cast2->get_outputs()[0];
        auto cast3 = graph->make(
                "cast", {inputs[3]}, {}, {{"dtype", datatypes::f32}});
        mean = cast3->get_outputs()[0];
        auto cast4 = graph->make(
                "cast", {inputs[4]}, {}, {{"dtype", datatypes::f32}});
        variance = cast4->get_outputs()[0];
    }

    std::vector<int> rd_axis, bc_axis;
    if (format == "NCX") {
        rd_axis = is_3D ? std::vector<int> {0, 2, 3, 4}
                        : std::vector<int> {0, 2, 3};
        bc_axis = std::vector<int> {1};
    } else {
        rd_axis = is_3D ? std::vector<int> {0, 1, 2, 3}
                        : std::vector<int> {0, 1, 2};
        bc_axis = is_3D ? std::vector<int> {4} : std::vector<int> {3};
    }
    // mean and var of src 1 pass
    float channel_size = 1.0f;
    for (auto ax : rd_axis) {
        channel_size *= inputs[0]->details_.get_plain_dims()[ax];
    }
    auto rchan_size_op = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(
                    std::vector<float> {1 / channel_size}),
            datatypes::f32, sc_dims {1});
    auto reduce0 = graph->make("reduce", {src}, {},
            {{"rd_axis", rd_axis}, {"rd_op", 0}, {"keep_dims", false}});
    auto new_mean = graph->make("mul",
            {reduce0->get_outputs()[0], rchan_size_op->get_outputs()[0]}, {},
            {{op_attr_key::break_post_fuse, true}});
    auto src_squared = graph->make("mul", {src, src}, {}, {});
    auto reduce0_squared = graph->make("mul",
            {reduce0->get_outputs()[0], reduce0->get_outputs()[0]}, {}, {});
    auto reduce0_squared_mul = graph->make("mul",
            {reduce0_squared->get_outputs()[0],
                    rchan_size_op->get_outputs()[0]},
            {}, {});
    auto reduce1 = graph->make("reduce", {src_squared->get_outputs()[0]}, {},
            {{"rd_axis", rd_axis}, {"rd_op", 0}, {"keep_dims", false}});
    auto sub0 = graph->make("sub",
            {reduce1->get_outputs()[0], reduce0_squared_mul->get_outputs()[0]},
            {}, {});
    auto new_var = graph->make("mul",
            {sub0->get_outputs()[0], rchan_size_op->get_outputs()[0]}, {},
            {{op_attr_key::break_post_fuse, true}});

    // normalization of src (x_normalized)
    auto sub1 = graph->make("sub", {src_pass2, new_mean->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    auto add0 = graph->make("add",
            {new_var->get_outputs()[0], epsilon->get_outputs()[0]}, {}, {});
    auto rsqrt = graph->make("squared_root", {add0->get_outputs()[0]}, {},
            {{"reciprocal", true}});
    auto mul1 = graph->make("mul",
            {sub1->get_outputs()[0], rsqrt->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    // gamma*x_normalized + beta
    auto mul2 = graph->make(
            "mul", {mul1->get_outputs()[0], scale}, {}, {{"bc_axis", bc_axis}});
    auto add1 = graph->make(
            "add", {mul2->get_outputs()[0], shift}, {}, {{"bc_axis", bc_axis}});

    // running_mean and running_variance
    sc_op_ptr add2, add3;
    if (attrs_.get_or_else("momentum", float(1.)) == 1.f) {
        add2 = graph->make("duplicate", {mean}, {}, {});
        add3 = graph->make("duplicate", {variance}, {}, {});
    } else if (attrs_.get_or_else("momentum", float(1.)) == 0.f) {
        add2 = graph->make("duplicate", {new_mean->get_outputs()[0]}, {}, {});
        add3 = graph->make("duplicate", {new_var->get_outputs()[0]}, {}, {});
    } else {
        auto momentum = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {
                        attrs_.get_or_else("momentum", float(1.))}),
                datatypes::f32, sc_dims {1});
        auto one_sub_momentum = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {
                        1 - attrs_.get_or_else("momentum", float(1.))}),
                datatypes::f32, sc_dims {1});

        // running_mean = momentum * mean + (1- momentum) * mean_x
        auto mul3 = graph->make(
                "mul", {mean, momentum->get_outputs()[0]}, {}, {});
        auto mul4 = graph->make("mul",
                {new_mean->get_outputs()[0],
                        one_sub_momentum->get_outputs()[0]},
                {}, {});
        add2 = graph->make("add",
                {mul3->get_outputs()[0], mul4->get_outputs()[0]}, {}, {});

        // running_var = momentum * variance + (1- momentum) * var_x
        auto mul5 = graph->make(
                "mul", {variance, momentum->get_outputs()[0]}, {}, {});
        auto mul6 = graph->make("mul",
                {new_var->get_outputs()[0], one_sub_momentum->get_outputs()[0]},
                {}, {});
        add3 = graph->make("add",
                {mul5->get_outputs()[0], mul6->get_outputs()[0]}, {}, {});
    }

    if (is_src_bf16) {
        add1 = graph->make(
                "cast", add1->get_outputs(), {}, {{"dtype", datatypes::bf16}});
    }
    if (is_ssmv_bf16) {
        if (attrs_.get_or_else("momentum", float(1.)) != 1.f) {
            add2 = graph->make("cast", add2->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
            add3 = graph->make("cast", add3->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
        }
        new_mean = graph->make("cast", new_mean->get_outputs(), {},
                {{"dtype", datatypes::bf16}});
        new_var = graph->make("cast", new_var->get_outputs(), {},
                {{"dtype", datatypes::bf16}});
    }
    // output
    graph->make_output({add1->get_outputs()[0], add2->get_outputs()[0],
            add3->get_outputs()[0], new_mean->get_outputs()[0],
            new_var->get_outputs()[0]});
}

void batchnorm_forward_training_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

batchnorm_training_backprop_op_t::batchnorm_training_backprop_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[2]->details_));
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[2]->details_));
    } else {
        info_.outputs_ = outs;
    }
    COMPILE_ASSERT(info_.inputs_.size() == 5 && info_.outputs_.size() == 3,
            "Batchnorm backprop op currently only allows 5-input + 3-output "
            "schema.");
    COMPILE_ASSERT(ins[0]->details_.get_plain_dims().size() == 4
                    || ins[0]->details_.get_plain_dims().size() == 5,
            "Batchnorm backprop op currently only supports 4D or 5D cases.");
    COMPILE_ASSERT(ins[0]->details_.dtype_ == ins[1]->details_.dtype_,
            "src and delta_output must have the same dtype.");
    COMPILE_ASSERT(ins[2]->details_.dtype_ == ins[3]->details_.dtype_
                    && ins[2]->details_.dtype_ == ins[4]->details_.dtype_,
            "gamma, mean and variance must have the same dtype.")
    attrs_ = attrs;
    op_name_ = "batchnorm_training_backprop";
}

void batchnorm_training_backprop_op_t::get_graph_impl(
        std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    bool is_3D = (inputs[0]->details_.get_plain_dims().size() == 5);
    bool is_bf16_src
            = (inputs[0]->details_.dtype_.is_etype(sc_data_etype::BF16));
    bool is_bf16_gamma
            = (inputs[2]->details_.dtype_.is_etype(sc_data_etype::BF16));

    float epsilon = attrs_.get<float>("epsilon");
    auto data_format = attrs_.get_or_else<std::string>("data_format", "NXC");
    // input
    graph->make_input(inputs);
    // calculating reduce axis
    std::vector<int> bc_axis, rd_axis;
    if (data_format == "NXC") {
        bc_axis = is_3D ? std::vector<int> {4} : std::vector<int> {3};
        rd_axis = is_3D ? std::vector<int> {0, 1, 2, 3}
                        : std::vector<int> {0, 1, 2};
    } else {
        bc_axis = std::vector<int> {1};
        rd_axis = is_3D ? std::vector<int> {0, 2, 3, 4}
                        : std::vector<int> {0, 2, 3};
    }

    graph_tensor_ptr src = inputs[0], output_delta = inputs[1],
                     mean = inputs[2], variance = inputs[3], gamma = inputs[4];
    graph_tensor_ptr src_pass2 = inputs[0], output_delta_pass2 = inputs[1];
    if (is_bf16_src) {
        auto cast0 = graph->make("cast", {inputs[0]}, {},
                {{"dtype", datatypes::f32},
                        {op_attr_key::not_redundant, true}});
        src = cast0->get_outputs()[0];
        auto cast0_pass2 = graph->make("cast", {inputs[0]}, {},
                {{"dtype", datatypes::f32}, {op_attr_key::break_pre_fuse, true},
                        {op_attr_key::not_redundant, true}});
        src_pass2 = cast0_pass2->get_outputs()[0];
        auto cast1 = graph->make("cast", {inputs[1]}, {},
                {{"dtype", datatypes::f32},
                        {op_attr_key::not_redundant, true}});
        output_delta = cast1->get_outputs()[0];
        auto cast1_pass2 = graph->make("cast", {inputs[1]}, {},
                {{"dtype", datatypes::f32}, {op_attr_key::break_pre_fuse, true},
                        {op_attr_key::not_redundant, true}});
        output_delta_pass2 = cast1_pass2->get_outputs()[0];
    }
    if (is_bf16_gamma) {
        auto cast2 = graph->make(
                "cast", {inputs[2]}, {}, {{"dtype", datatypes::f32}});
        gamma = cast2->get_outputs()[0];
        auto cast3 = graph->make(
                "cast", {inputs[3]}, {}, {{"dtype", datatypes::f32}});
        mean = cast3->get_outputs()[0];
        auto cast4 = graph->make(
                "cast", {inputs[4]}, {}, {{"dtype", datatypes::f32}});
        variance = cast4->get_outputs()[0];
    }
    // ------ calculate x_hat start ------
    // eps constant
    auto const_op = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {epsilon}),
            datatypes::f32, sc_dims {1});
    // reduce size constant, cast to float (potential inaccuracy/overflow)
    float channel_size = 1.0f;
    for (auto ax : rd_axis) {
        channel_size *= inputs[0]->details_.get_plain_dims()[ax];
    }
    auto rchan_size_op = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(
                    std::vector<float> {1 / channel_size}),
            datatypes::f32, sc_dims {1});
    // var + eps
    auto var_eps = graph->make(
            "add", {variance, const_op->get_outputs()[0]}, {}, {});
    // rsqrt(var + eps)
    auto rsqrt_var_eps = graph->make(
            "squared_root", var_eps->get_outputs(), {}, {{"reciprocal", true}});
    // x - mu
    auto x_mu = graph->make("sub", {src, mean}, {}, {{"bc_axis", bc_axis}});
    // x_hat = (x - mu) *  rsqrt(var + eps)
    auto x_hat = graph->make("mul",
            {x_mu->get_outputs()[0], rsqrt_var_eps->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    // ------ calculate x_hat end ------

    // ------ duplicate x_mu && x_hat start ------
    auto x_mu_pass2
            = graph->make("sub", {src_pass2, mean}, {}, {{"bc_axis", bc_axis}});
    // x_hat = (x - mu) *  rsqrt(var + eps)
    auto x_hat_pass2 = graph->make("mul",
            {x_mu_pass2->get_outputs()[0], rsqrt_var_eps->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    // ------ duplicate x_mu && x_hat end ------

    // gamma_delta = reducesum(dy * x_hat)
    auto dy_x_hat = graph->make(
            "mul", {output_delta, x_hat->get_outputs()[0]}, {}, {});
    auto gamma_delta = graph->make("reduce", dy_x_hat->get_outputs(), {},
            {{"rd_axis", rd_axis}, {"rd_op", 0}, {"keep_dims", false}});
    // gamma * x_hat + beta = y --> beta_delta = reducesum(dy)
    auto beta_delta = graph->make("reduce", {output_delta}, {},
            {{"rd_axis", rd_axis}, {"rd_op", 0}, {"keep_dims", false}});
    // ------ calculate x_delta start ------
    // calculate: (x - mu) * rsqrt(var + eps) * gamma_delta <==> x_hat *
    // gamma_delta
    auto x_hat_gamma_delta = graph->make("mul",
            {x_hat_pass2->get_outputs()[0], gamma_delta->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    // add beta_delta and x_hat_gamma_delta
    auto add = graph->make("add",
            {x_hat_gamma_delta->get_outputs()[0], beta_delta->get_outputs()[0]},
            {}, {{"bc_axis", bc_axis}});
    //  * (1 / channel_size)
    auto rescale = graph->make("mul",
            {add->get_outputs()[0], rchan_size_op->get_outputs()[0]}, {}, {});
    // get subtract results
    auto sub = graph->make(
            "sub", {output_delta_pass2, rescale->get_outputs()[0]}, {}, {});
    // final mul: x_delta = gamma * rsqrt(var + eps) * sub_result
    auto mul = graph->make(
            "mul", {rsqrt_var_eps->get_outputs()[0], gamma}, {}, {});
    auto x_delta
            = graph->make("mul", {sub->get_outputs()[0], mul->get_outputs()[0]},
                    {}, {{"bc_axis", bc_axis}});
    // ------ calculate x_delta end ------
    // output
    graph_tensor_ptr x_delta_out = x_delta->get_outputs()[0],
                     gamma_delta_out = gamma_delta->get_outputs()[0],
                     beta_delta_out = beta_delta->get_outputs()[0];
    if (is_bf16_src) {
        auto cast_out0 = graph->make(
                "cast", {x_delta_out}, {}, {{"dtype", datatypes::bf16}});
        x_delta_out = cast_out0->get_outputs()[0];
    }
    if (is_bf16_gamma) {
        auto cast_out1 = graph->make(
                "cast", {gamma_delta_out}, {}, {{"dtype", datatypes::bf16}});
        gamma_delta_out = cast_out1->get_outputs()[0];
        auto cast_out2 = graph->make(
                "cast", {beta_delta_out}, {}, {{"dtype", datatypes::bf16}});
        beta_delta_out = cast_out2->get_outputs()[0];
    }
    graph->make_output({x_delta_out, gamma_delta_out, beta_delta_out});
}

void batchnorm_training_backprop_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

} // namespace ops

OP_REGISTER(ops::batchnorm_inference_op, batchnorm_inference)
OP_REGISTER(ops::batchnorm_forward_training_op, batchnorm_forward_training)
OP_REGISTER(ops::batchnorm_training_backprop_op_t, batchnorm_training_backprop)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
