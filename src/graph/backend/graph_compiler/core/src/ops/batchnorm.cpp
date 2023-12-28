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

static sc_dim get_channel_size(
        const any_map_t &attrs, const graph_tensor_ptr &input) {
    std::string data_format = attrs.get<std::string>("data_format");
    const auto &input_shape = input->details_.get_plain_dims();
    sc_dim channel_size = data_format == "NCX"
            ? input_shape[1]
            : input_shape[input_shape.size() - 1];
    return channel_size;
}

static void validate_channel_related_input(
        const graph_tensor_ptr &gt, sc_dim channel_size, bool allow_bf16) {
    COMPILE_ASSERT(gc::graph::check_shape_equal(gt->details_.get_plain_dims(),
                           sc_dims {channel_size}),
            "Batchnorm's mean/var/gamma/beta shall have the same shape as "
            "channel dimension.");
    if (allow_bf16) {
        COMPILE_ASSERT(utils::is_one_of(gt->details_.dtype_.type_code_,
                               sc_data_etype::F32, sc_data_etype::BF16),
                "Batchnorm's mean/var/gamma/beta shall be with f32 or bf16 "
                "dtype.");
    } else {
        COMPILE_ASSERT(gt->details_.dtype_.type_code_ == sc_data_etype::F32,
                "Batchnorm's mean/var/gamma/beta shall be with f32 dtype.");
    }
}

batchnorm_inference_op::batchnorm_inference_op(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    COMPILE_ASSERT(info_.inputs_.size() == 3 || info_.inputs_.size() == 5,
            "batchnorm inference op shall have 3 or 5 inputs.");
    COMPILE_ASSERT(
            utils::is_one_of(info_.inputs_[0]->details_.dtype_.type_code_,
                    sc_data_etype::F32, sc_data_etype::BF16,
                    sc_data_etype::F16),
            "batchnorm inference's input shall be one of fp32/bf16/fp16 "
            "dtype.");
    sc_dim channel_size = get_channel_size(attrs, info_.inputs_[0]);
    bool allow_bf16 = (info_.inputs_[0]->details_.dtype_.type_code_
            == sc_data_etype::BF16);
    for (size_t i = 1; i < info_.inputs_.size(); ++i) {
        validate_channel_related_input(
                info_.inputs_[i], channel_size, allow_bf16);
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        info_.outputs_ = outs;
        COMPILE_ASSERT(info_.outputs_.size() == 1,
                "batchnorm inference op shall have only 1 output.")
        gc::graph::check_logical_tensor_shape_dtype_identical(
                info_.inputs_[0]->details_, info_.outputs_[0]->details_);
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
    // insert cast if input is of dtype bf16
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        inputs[idx] = cast_input_dtype(inputs[idx], graph);
    }
    // eps constant;
    auto const_op = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {epsilon}),
            datatypes::f32, sc_dims {1});
    // variance+eps
    auto var_eps = graph->make(
            "add", {inputs[4], const_op->get_outputs()[0]}, {}, {});
    // sqrt(variance+eps)
    auto sqrt_op
            = graph->make("squared_root", {var_eps->get_outputs()[0]}, {}, {});
    // gamma / sqrt(variance+eps) (Due to benchdnn accuracy require, we need to
    // use sqrt.)
    auto bn_div = graph->make(
            "div", {inputs[1], sqrt_op->get_outputs()[0]}, {}, {});
    // mean * gamma *rsqrt(variance+eps)
    auto mean_op
            = graph->make("mul", {inputs[3], bn_div->get_outputs()[0]}, {}, {});
    // beta - mean*gamma/sqrt(variance+eps)
    auto bn_add = graph->make(
            "sub", {inputs[2], mean_op->get_outputs()[0]}, {}, {});

    auto x1 = graph->make("mul", {inputs[0], bn_div->get_outputs()[0]}, {},
            any_map_t({{"bc_axis", bc_axis}}));

    auto y1 = graph->make("add",
            {x1->get_outputs()[0], bn_add->get_outputs()[0]}, {},
            any_map_t({{"bc_axis", bc_axis}}));
    // insert cast if output is bf16 dtype
    y1 = cast_output_dtype(outputs[0], graph, y1);
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
    COMPILE_ASSERT(ins.size() == 5,
            "batchnorm_forward_training's inputs size should be 5.");
    sc_dim channel_size = get_channel_size(attrs, info_.inputs_[0]);
    bool allow_bf16 = (info_.inputs_[0]->details_.dtype_.type_code_
            == sc_data_etype::BF16);
    for (size_t i = 1; i < info_.inputs_.size(); ++i) {
        validate_channel_related_input(
                info_.inputs_[i], channel_size, allow_bf16);
        COMPILE_ASSERT(info_.inputs_[1]->details_.dtype_
                        == info_.inputs_[i]->details_.dtype_,
                "Batchnorm forward training input dtypes shall be consistent.");
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
        gc::graph::check_logical_tensor_shape_dtype_identical(
                info_.inputs_[0]->details_, info_.outputs_[0]->details_);
        for (size_t i = 1; i < outs.size(); i++) {
            gc::graph::check_logical_tensor_shape_dtype_identical(
                    info_.inputs_[1]->details_, info_.outputs_[i]->details_);
        }
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
    int num_dims
            = static_cast<int>(inputs[0]->details_.get_plain_dims().size());
    auto src = inputs[0], mean = inputs[1], variance = inputs[2],
         scale = inputs[3], shift = inputs[4];
    auto src_pass2 = inputs[0], src_pass3 = inputs[0];

    auto epsilon = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(
                    std::vector<float> {attrs_.get_or_else("epsilon", 1e-5f)}),
            datatypes::f32, sc_dims {1});
    bool use_bnorm_opt = attrs_.get_or_else(op_attr_key::use_norm_opt, false);
    // insert cast if input is bf16
    src = cast_input_dtype(inputs[0], graph,
            {{"dtype", datatypes::f32}, {op_attr_key::not_redundant, true}});
    src_pass2 = cast_input_dtype(inputs[0], graph,
            {{"dtype", datatypes::f32}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::not_redundant, true}});
    if (!use_bnorm_opt) {
        src_pass3 = cast_input_dtype(inputs[0], graph,
                {{"dtype", datatypes::f32}, {op_attr_key::break_pre_fuse, true},
                        {op_attr_key::not_redundant, true}});
    }
    mean = cast_input_dtype(inputs[1], graph);
    variance = cast_input_dtype(inputs[2], graph);
    scale = cast_input_dtype(inputs[3], graph);
    shift = cast_input_dtype(inputs[4], graph);

    std::vector<int> rd_axis, bc_axis;
    if (format == "NCX") {
        bc_axis = std::vector<int> {1};
        for (int i = 0; i < num_dims; ++i) {
            if (i != 1) rd_axis.push_back(i);
        }
    } else {
        bc_axis = std::vector<int> {num_dims - 1};
        for (int i = 0; i < num_dims - 1; ++i) {
            rd_axis.push_back(i);
        }
    }
    // mean and var of src 1 pass
    float channel_size = 1.0f;
    for (auto ax : rd_axis) {
        channel_size *= inputs[0]->details_.get_plain_dims()[ax];
    }
    auto chan_size_op = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {channel_size}),
            datatypes::f32, sc_dims {1});
    auto reduce0 = graph->make("reduce", {src}, {},
            {{"rd_axis", rd_axis}, {"rd_op", 0}, {"keep_dims", false}});
    std::shared_ptr<sc_op> new_var;
    auto new_mean = graph->make("div",
            {reduce0->get_outputs()[0], chan_size_op->get_outputs()[0]}, {},
            {{op_attr_key::break_post_fuse, true},
                    {op_attr_key::must_div, true}});
    if (use_bnorm_opt) {
        auto src_squared = graph->make("mul", {src, src}, {}, {});
        auto reduce0_squared = graph->make("mul",
                {reduce0->get_outputs()[0], reduce0->get_outputs()[0]}, {}, {});
        auto reduce0_squared_mul = graph->make("div",
                {reduce0_squared->get_outputs()[0],
                        chan_size_op->get_outputs()[0]},
                {}, {{op_attr_key::must_div, true}});
        auto reduce1 = graph->make("reduce", {src_squared->get_outputs()[0]},
                {}, {{"rd_axis", rd_axis}, {"rd_op", 0}, {"keep_dims", false}});
        auto sub0 = graph->make("sub",
                {reduce1->get_outputs()[0],
                        reduce0_squared_mul->get_outputs()[0]},
                {}, {});
        new_var = graph->make("div",
                {sub0->get_outputs()[0], chan_size_op->get_outputs()[0]}, {},
                {{op_attr_key::break_post_fuse, true},
                        {op_attr_key::must_div, true}});
    } else {
        auto diff = graph->make("sub", {src_pass3, new_mean->get_outputs()[0]},
                {}, {{"bc_axis", bc_axis}});
        auto diff_squared = graph->make("mul",
                {diff->get_outputs()[0], diff->get_outputs()[0]}, {}, {});
        auto reduce1 = graph->make("reduce", {diff_squared->get_outputs()[0]},
                {}, {{"rd_axis", rd_axis}, {"rd_op", 0}, {"keep_dims", false}});
        new_var = graph->make("div",
                {reduce1->get_outputs()[0], chan_size_op->get_outputs()[0]}, {},
                {{op_attr_key::break_post_fuse, true},
                        {op_attr_key::must_div, true}});
    }

    // normalization of src (x_normalized)
    auto sub1 = graph->make("sub", {src_pass2, new_mean->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    auto add0 = graph->make("add",
            {new_var->get_outputs()[0], epsilon->get_outputs()[0]}, {}, {});
    auto sqrt = graph->make("squared_root", {add0->get_outputs()[0]}, {}, {});
    auto div1 = graph->make("div",
            {sub1->get_outputs()[0], sqrt->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    // gamma*x_normalized + beta
    auto mul2 = graph->make(
            "mul", {div1->get_outputs()[0], scale}, {}, {{"bc_axis", bc_axis}});
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

    // insert cast if output is bf16
    add1 = cast_output_dtype(outputs[0], graph, add1);
    if (attrs_.get_or_else("momentum", float(1.)) != 1.f) {
        add2 = cast_output_dtype(outputs[1], graph, add2);
        add3 = cast_output_dtype(outputs[2], graph, add3);
    }
    new_mean = cast_output_dtype(outputs[3], graph, new_mean);
    new_var = cast_output_dtype(outputs[4], graph, new_var);
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
    COMPILE_ASSERT(info_.inputs_.size() == 5,
            "Batchnorm backprop op shall have 5 inputs.");
    COMPILE_ASSERT(ins[0]->details_.dtype_ == ins[1]->details_.dtype_,
            "Batchnorm backprop's src and delta_output must have the "
            "same dtype.");
    sc_dim channel_size = get_channel_size(attrs, info_.inputs_[0]);
    bool allow_bf16 = (info_.inputs_[0]->details_.dtype_.type_code_
            == sc_data_etype::BF16);
    for (size_t i = 2; i < info_.inputs_.size(); ++i) {
        validate_channel_related_input(
                info_.inputs_[i], channel_size, allow_bf16);
        COMPILE_ASSERT(info_.inputs_[2]->details_.dtype_
                        == info_.inputs_[i]->details_.dtype_,
                "Batchnorm backprop's gamma, mean and variance must "
                "have the same dtype.");
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[2]->details_));
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[2]->details_));
    } else {
        info_.outputs_ = outs;
        COMPILE_ASSERT(info_.outputs_.size() == 3,
                "Batchnorm backprop op shall have 3 outputs.");
        gc::graph::check_logical_tensor_shape_dtype_identical(
                info_.inputs_[0]->details_, info_.outputs_[0]->details_);
        gc::graph::check_logical_tensor_shape_dtype_identical(
                info_.inputs_[2]->details_, info_.outputs_[1]->details_);
        gc::graph::check_logical_tensor_shape_dtype_identical(
                info_.inputs_[2]->details_, info_.outputs_[2]->details_);
    }
    attrs_ = attrs;
    op_name_ = "batchnorm_training_backprop";
}

void batchnorm_training_backprop_op_t::get_graph_impl(
        std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    int num_dims
            = static_cast<int>(inputs[0]->details_.get_plain_dims().size());

    float epsilon = attrs_.get<float>("epsilon");
    auto data_format = attrs_.get_or_else<std::string>("data_format", "NXC");
    // input
    graph->make_input(inputs);
    // calculating reduce axis
    std::vector<int> bc_axis, rd_axis;
    if (data_format == "NCX") {
        bc_axis = std::vector<int> {1};
        for (int i = 0; i < num_dims; ++i) {
            if (i != 1) rd_axis.push_back(i);
        }
    } else {
        bc_axis = std::vector<int> {num_dims - 1};
        for (int i = 0; i < num_dims - 1; ++i) {
            rd_axis.push_back(i);
        }
    }

    graph_tensor_ptr src = inputs[0], output_delta = inputs[1],
                     mean = inputs[2], variance = inputs[3], gamma = inputs[4];
    graph_tensor_ptr src_pass2 = inputs[0], output_delta_pass2 = inputs[1];
    // cast input if its dtype is bf16
    src = cast_input_dtype(inputs[0], graph,
            {{"dtype", datatypes::f32}, {op_attr_key::not_redundant, true}});
    src_pass2 = cast_input_dtype(inputs[0], graph,
            {{"dtype", datatypes::f32}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::not_redundant, true}});
    output_delta = cast_input_dtype(inputs[1], graph,
            {{"dtype", datatypes::f32}, {op_attr_key::not_redundant, true}});
    output_delta_pass2 = cast_input_dtype(inputs[1], graph,
            {{"dtype", datatypes::f32}, {op_attr_key::break_pre_fuse, true},
                    {op_attr_key::not_redundant, true}});
    mean = cast_input_dtype(inputs[2], graph);
    variance = cast_input_dtype(inputs[3], graph);
    gamma = cast_input_dtype(inputs[4], graph);
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
    auto chan_size_op = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {channel_size}),
            datatypes::f32, sc_dims {1});
    // var + eps
    auto var_eps = graph->make(
            "add", {variance, const_op->get_outputs()[0]}, {}, {});
    // rsqrt(var + eps)
    auto sqrt_var_eps = graph->make("squared_root", var_eps->get_outputs(), {},
            {{"reciprocal", false}});
    // x - mu
    auto x_mu = graph->make("sub", {src, mean}, {}, {{"bc_axis", bc_axis}});
    // x_hat = (x - mu) *  rsqrt(var + eps)
    auto x_hat = graph->make("div",
            {x_mu->get_outputs()[0], sqrt_var_eps->get_outputs()[0]}, {},
            {{"bc_axis", bc_axis}});
    // ------ calculate x_hat end ------

    // ------ duplicate x_mu && x_hat start ------
    auto x_mu_pass2
            = graph->make("sub", {src_pass2, mean}, {}, {{"bc_axis", bc_axis}});
    // x_hat = (x - mu) *  rsqrt(var + eps)
    auto x_hat_pass2 = graph->make("div",
            {x_mu_pass2->get_outputs()[0], sqrt_var_eps->get_outputs()[0]}, {},
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
    auto rescale = graph->make("div",
            {add->get_outputs()[0], chan_size_op->get_outputs()[0]}, {}, {});
    // get subtract results
    auto sub = graph->make(
            "sub", {output_delta_pass2, rescale->get_outputs()[0]}, {}, {});
    // final mul: x_delta = gamma * rsqrt(var + eps) * sub_result
    auto div = graph->make(
            "div", {gamma, sqrt_var_eps->get_outputs()[0]}, {}, {});
    auto x_delta
            = graph->make("mul", {sub->get_outputs()[0], div->get_outputs()[0]},
                    {}, {{"bc_axis", bc_axis}});
    // ------ calculate x_delta end ------
    // output
    x_delta = cast_output_dtype(outputs[0], graph, x_delta);
    gamma_delta = cast_output_dtype(outputs[1], graph, gamma_delta);
    beta_delta = cast_output_dtype(outputs[2], graph, beta_delta);
    graph->make_output({x_delta->get_outputs()[0],
            gamma_delta->get_outputs()[0], beta_delta->get_outputs()[0]});
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
