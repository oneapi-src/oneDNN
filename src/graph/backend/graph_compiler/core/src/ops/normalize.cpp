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
#include <memory>
#include <utility>
#include "normalize.hpp"
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

// compute the output data format after reduction given the plain reduction axis
static sc_data_format_t get_reduced_format(
        const sc_data_format_t &in_fmt, const std::vector<int> &rd_axis) {
    auto base_fmt = in_fmt;
    // we should set the blocking of the reduce axies to 1
    for (int ax : rd_axis) {
        for (int blocking_idx :
                in_fmt.format_code_.collect_blocking_index(ax)) {
            base_fmt.blocks_[blocking_idx] = 1;
        }
    }
    return base_fmt;
}

// compute the output data shape after reduction given the plain reduction axis
static sc_dims get_reduced_shape(
        const sc_dims &in_shape, const std::vector<int> &rd_axis) {
    sc_dims reduced_shape = in_shape;
    // we should set the reduce axies to 1
    for (int ax : rd_axis) {
        reduced_shape[ax] = 1;
    }
    return reduced_shape;
}

normalize_common_t::normalize_common_t(const normalize_kind &kind,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    attrs_ = attrs;
    if (kind == normalize_kind::layernorm)
        op_name_ = "layernorm";
    else if (kind == normalize_kind::instancenorm)
        op_name_ = "instancenorm";
    float epsilon = attrs_.get<float>("epsilon");
    bool keep_stats = attrs_.get_or_else("keep_stats", true);
    // TODO(xxx): deprecated, replaced by begin_norm_axis
    const std::vector<int> &rd_axis = attrs_.get<std::vector<int>>("rd_axis");
    bool use_affine = attrs_.get<bool>("use_affine");
    if (use_affine) {
        sc_dims expected_affine_shape;
        auto &plain_dims = ins[0]->details_.get_plain_dims();
        expected_affine_shape.reserve(rd_axis.size());
        for (auto &ax : rd_axis) {
            expected_affine_shape.push_back(plain_dims.at(ax));
        }
        COMPILE_ASSERT(ins.size() == 3UL,
                op_name_ + ": Expecting 3 inputs for use_affine=True");
        auto gamma_shape = ins[1]->details_.get_plain_dims();
        auto beta_shape = ins[2]->details_.get_plain_dims();
        COMPILE_ASSERT((expected_affine_shape == gamma_shape)
                        && (expected_affine_shape == beta_shape),
                "Wrong shape for beta and gamma of op "
                        << op_name_.c_str() << ". Expecting "
                        << utils::print_vector(expected_affine_shape)
                        << ", but got gamma with shape: "
                        << utils::print_vector(gamma_shape)
                        << ", and beta with shape: "
                        << utils::print_vector(beta_shape));
    } else {
        COMPILE_ASSERT(ins.size() == 1UL,
                op_name_ + ": Expecting 1 input for use_affine=False");
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
        if (keep_stats) {
            auto reduced_shape = get_reduced_shape(
                    ins[0]->details_.get_plain_dims(), rd_axis);
            auto reduced_format = get_reduced_format(
                    ins[0]->details_.get_format(), rd_axis);
            info_.outputs_.emplace_back(
                    graph_tensor::make(reduced_shape, reduced_format));
            info_.outputs_.emplace_back(
                    graph_tensor::make(reduced_shape, reduced_format));
        }

    } else {
        info_.outputs_ = outs;
        if (keep_stats) {
            COMPILE_ASSERT(outs.size() == 3UL,
                    "Expecting 3 output tensor: result, mean and variance")
        } else {
            COMPILE_ASSERT(outs.size() == 1UL, "Expecting 1 result tensor")
        }
        COMPILE_ASSERT(outs[0]->details_.get_plain_dims()
                                == ins[0]->details_.get_plain_dims()
                        && outs[0]->details_.dtype_ == ins[0]->details_.dtype_,
                "The output tensor should have the same shape and dtype of the "
                "input")
    }
}

void normalize_common_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    float epsilon = attrs_.get<float>("epsilon");
    const std::vector<int> &rd_axis = attrs_.get<std::vector<int>>("rd_axis");
    bool use_affine = attrs_.get<bool>("use_affine");
    bool keep_stats = attrs_.get_or_else("keep_stats", true);

    std::vector<int> non_normalized_bc_axis;
    for (size_t i = 0; i < info_.inputs_[0]->details_.get_plain_dims().size();
            i++) {
        if (std::find(rd_axis.begin(), rd_axis.end(), static_cast<int>(i))
                != rd_axis.end()) {
            continue;
        } else {
            non_normalized_bc_axis.emplace_back(i);
        }
    }

    // input
    graph->make_input(inputs);
    // constant op
    // epsilon
    COMPILE_ASSERT(
            utils::is_one_of(info_.inputs_[0]->details_.dtype_.type_code_,
                    sc_data_etype::F32, sc_data_etype::BF16),
            "Only support bf16/f32");
    bool is_bf16
            = info_.inputs_[0]->details_.dtype_.is_etype(sc_data_etype::BF16);
    sc_op_ptr feps = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {epsilon}),
            datatypes::f32, sc_dims {1});
    graph_tensor_ptr inputs0 = inputs[0], inputs1, inputs2;
    if (use_affine) {
        inputs1 = inputs[1];
        inputs2 = inputs[2];
    }
    if (is_bf16) {
        auto cast0 = graph->make(
                "cast", {inputs[0]}, {}, {{"dtype", datatypes::f32}});
        inputs0 = cast0->get_outputs()[0];
        if (use_affine) {
            // in case inputs0 and inputs1,2 have different datatypes
            if (inputs[1]->details_.dtype_.is_etype(sc_data_etype::BF16)) {
                auto cast1 = graph->make(
                        "cast", {inputs[1]}, {}, {{"dtype", datatypes::f32}});
                inputs1 = cast1->get_outputs()[0];
            }
            if (inputs[2]->details_.dtype_.is_etype(sc_data_etype::BF16)) {
                auto cast2 = graph->make(
                        "cast", {inputs[2]}, {}, {{"dtype", datatypes::f32}});
                inputs2 = cast2->get_outputs()[0];
            }
        }
    }
    // x^2
    auto fsquare = graph->make("mul", {inputs0, inputs0}, {}, {}); // 1
    // mean of x^2
    auto fsqd_mean = graph->make("reduce_mean", {fsquare->get_outputs()[0]}, {},
            {{"rd_axis", rd_axis}}); // 2
    // mean of X
    auto fmean = graph->make(
            "reduce_mean", {inputs0}, {}, {{"rd_axis", rd_axis}}); // 3
    // square of mean
    auto fmean_sqd = graph->make("mul",
            {fmean->get_outputs()[0], fmean->get_outputs()[0]}, {}, {}); // 4
    // x-x_mean
    auto fdiff = graph->make("sub", {inputs0, fmean->get_outputs()[0]}, {},
            {{"bc_axis", non_normalized_bc_axis}}); // 5

    // var(x)
    auto fvar = graph->make("sub",
            {fsqd_mean->get_outputs()[0], fmean_sqd->get_outputs()[0]}, {},
            {}); // 6
    auto fadd_eps = graph->make("add",
            {fvar->get_outputs()[0], feps->get_outputs()[0]}, {}, {}); // 6
    // rsqrt
    auto frsqd_root = graph->make("squared_root", {fadd_eps->get_outputs()[0]},
            {}, {{"reciprocal", true}}); // 7
    auto foutput = graph->make("mul",
            {fdiff->get_outputs()[0], frsqd_root->get_outputs()[0]}, {},
            {{"bc_axis", non_normalized_bc_axis}}); // 8

    if (use_affine) {
        foutput = graph->make("mul", {foutput->get_outputs()[0], inputs1}, {},
                any_map_t({{"bc_axis", rd_axis}}));
        foutput = graph->make("add", {foutput->get_outputs()[0], inputs2}, {},
                any_map_t({{"bc_axis", rd_axis}}));
    }
    // output
    if (keep_stats) {
        if (is_bf16) {
            foutput = graph->make("cast", foutput->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
            fmean = graph->make("cast", fmean->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
            fvar = graph->make("cast", fvar->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
        }
        graph->make_output({foutput->get_outputs()[0], fmean->get_outputs()[0],
                fvar->get_outputs()[0]});
    } else {
        if (is_bf16) {
            foutput = graph->make("cast", foutput->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
        }
        graph->make_output(foutput->get_outputs());
    }
}

void normalize_common_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

} // namespace ops

OP_REGISTER(ops::layernorm_op_t, layernorm)

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
