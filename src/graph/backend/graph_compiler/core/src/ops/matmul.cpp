/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
 ******************************************************************************/

#include "matmul.hpp"
#include <algorithm>
#include <numeric>
#include <utility>
#include "compiler/ir/graph/fusible_op.hpp"
#include "matmul_core.hpp"
#include "templates/utils.hpp"
#include <compiler/ir/graph/quantization/quantize_info.hpp>
#include <util/math_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

matmul_op::matmul_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT((ins.size() == 2 || ins.size() == 3),
            "matmul inputs size should be 2(a, b) or 3(a, b, bias).");
    COMPILE_ASSERT((ins[0]->details_.get_plain_dims().size() >= 2
                           && ins[1]->details_.get_plain_dims().size() >= 2),
            "matrix a and matrix b shape should be bigger or equal than 2.");
    info_.inputs_ = ins;
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    bool trans_a = attrs.get_or_else("transpose_a", false);
    bool trans_b = attrs.get_or_else("transpose_b", false);
    bool is_int8 = utils::is_one_of(
            ins[0]->details_.dtype_, datatypes::u8, datatypes::s8);
    bool is_bf16 = ins[0]->details_.dtype_ == datatypes::bf16;
    bool is_low_precision_fp = utils::is_one_of(
            ins[0]->details_.dtype_, datatypes::bf16, datatypes::f16);
    sc_dims output_shape;
    if (!is_dynamic()) {
        output_shape = {merge_vec(
                matmul_core_op_t::get_batch_dims_with_bc_impl(A_dims, B_dims),
                {A_dims[A_dims.size() - (trans_a ? 1 : 2)],
                        B_dims[B_dims.size() - (trans_b ? 2 : 1)]})};
    } else {
        output_shape = {
                merge_vec(matmul_core_op_t::get_batch_dims_impl(A_dims, B_dims),
                        {A_dims[A_dims.size() - (trans_a ? 1 : 2)],
                                B_dims[B_dims.size() - (trans_b ? 2 : 1)]})};
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                sc_data_format_t(), output_shape,
                is_int8 ? datatypes::s32
                        : (is_low_precision_fp ? (
                                   is_bf16 ? datatypes::bf16 : datatypes::f16)
                                               : datatypes::f32)));
    } else {
        info_.outputs_ = outs;
        if (!is_dynamic()) {
            COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims()
                            == output_shape,
                    "Bad out dims");
        }
    }
    for (auto &op : info_.outputs_) {
        op->producer_owner_ = this;
    }
    attrs_ = attrs;
    op_name_ = "matmul";
}

static void transed_matmul(const std::shared_ptr<sc_graph_t> &graph,
        any_map_t &attrs, const graph_tensor_ptr &ins0,
        const graph_tensor_ptr &ins1, graph_tensor_ptr &trans0,
        graph_tensor_ptr &trans1) {
    if (attrs.get_or_else("transpose_a", false)) {
        auto &original_dims = ins0->details_.get_plain_dims();
        sc_dims transed_plain_dims(original_dims.begin(), original_dims.end());
        COMPILE_ASSERT(transed_plain_dims.size() >= 2, "Bad input shape");
        std::swap(transed_plain_dims[transed_plain_dims.size() - 1],
                transed_plain_dims[transed_plain_dims.size() - 2]);
        std::vector<int> order(transed_plain_dims.size());
        std::iota(order.begin(), order.end(), 0);
        std::swap(order[transed_plain_dims.size() - 1],
                order[transed_plain_dims.size() - 2]);
        auto out = graph_tensor::make(transed_plain_dims,
                ins0->details_.get_format(), ins0->details_.dtype_);
        trans0 = graph->make("transpose", {ins0}, {out}, {{"order", order}})
                         ->get_outputs()[0];
        attrs.set("transpose_a", false);
    } else {
        attrs.set("transpose_a", false);
    }

    // if transpose_b is true: need to permute
    if (attrs.get_or_else("transpose_b", false)) {
        auto original_dims = ins1->details_.get_plain_dims();
        sc_dims transed_plain_dims(original_dims.begin(), original_dims.end());
        std::swap(transed_plain_dims[transed_plain_dims.size() - 1],
                transed_plain_dims[transed_plain_dims.size() - 2]);
        std::vector<int> order(transed_plain_dims.size());
        std::iota(order.begin(), order.end(), 0);
        std::swap(order[transed_plain_dims.size() - 1],
                order[transed_plain_dims.size() - 2]);
        auto out = graph_tensor::make(transed_plain_dims,
                ins1->details_.get_format(), ins1->details_.dtype_);
        trans1 = graph->make("transpose", {ins1}, {out}, {{"order", order}})
                         ->get_outputs()[0];
        attrs.set("transpose_b", false);
    } else {
        attrs.set("transpose_b", false);
    }
}

void matmul_op::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto ins = graph->make_input(inputs);
    sc_op_ptr matmul, graph_out;

    // analysis matmul is matmul_core_op_t which is tunable op by
    // inputs[0](the left matrix) and inputs[1](the right matrix).
    graph_tensor_ptr trans0 = ins->get_outputs()[0],
                     trans1 = ins->get_outputs()[1];
    // don't change attrs_ directly
    auto attrs = attrs_;
    // used for mmm_core
    bool transposed_a = attrs.get_or_else("transpose_a", false);
    bool transposed_b = attrs.get_or_else("transpose_b", false);
    std::vector<int> post_rd_axis
            = attrs.get_or_else("post_rd_axis", std::vector<int> {});
    transed_matmul(graph, attrs, ins->get_outputs()[0], ins->get_outputs()[1],
            trans0, trans1);

    bool is_bf16 = false;
    if (inputs[0]->details_.dtype_ == datatypes::bf16
            || inputs[1]->details_.dtype_ == datatypes::bf16
            || outputs[0]->details_.dtype_ == datatypes::bf16) {
        COMPILE_ASSERT(inputs[0]->details_.dtype_ == datatypes::bf16
                        && inputs[1]->details_.dtype_ == datatypes::bf16
                        && outputs[0]->details_.dtype_ == datatypes::bf16,
                "All inputs should have same data type.")
        is_bf16 = true;
    }

    bool is_f16 = false;
    if (inputs[0]->details_.dtype_ == datatypes::f16
            || inputs[1]->details_.dtype_ == datatypes::f16
            || outputs[0]->details_.dtype_ == datatypes::f16) {
        COMPILE_ASSERT(inputs[0]->details_.dtype_ == datatypes::f16
                        && inputs[1]->details_.dtype_ == datatypes::f16
                        && outputs[0]->details_.dtype_ == datatypes::f16,
                "All inputs should have same data type.")
        is_f16 = true;
    }

    // For Nd*2d and 2d*Nd non-dynamic cases, ND input will be reshaped into 2D
    // to meet more possibilities of M_block or N_block
    sc_dims trans0_plain_dims = trans0->details_.get_plain_dims(),
            trans1_plain_dims = trans1->details_.get_plain_dims();
    if (!is_dynamic()) {
        // check Nd*2d cases
        if (trans0_plain_dims.size() > 2 && trans1_plain_dims.size() == 2) {
            sc_dims reshape_dest = {math_utils::get_dims_product(sc_dims {
                                            trans0_plain_dims.begin(),
                                            trans0_plain_dims.end() - 1}),
                    trans0_plain_dims.back()};
            auto reshape_node = graph->make("tensor_view", {trans0},
                    {graph_tensor::make(reshape_dest, sc_data_format_t(),
                            trans0->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", sc_data_format_t()},
                            {"forbid_penetrate", true}});
            trans0 = reshape_node->get_outputs()[0];
            if (post_rd_axis.size() == 1
                    && post_rd_axis.at(0)
                            == static_cast<int>(trans0_plain_dims.size()) - 1) {
                post_rd_axis.at(0) = 1;
            }
        }
        // check 2d*Nd cases
        if (trans0_plain_dims.size() == 2 && trans1_plain_dims.size() > 2) {
            sc_dims reshape_dest
                    = {trans1_plain_dims.at(trans1_plain_dims.size() - 2),
                            math_utils::get_dims_product(
                                    sc_dims {trans1_plain_dims.begin(),
                                            trans1_plain_dims.end() - 2})
                                    * trans1_plain_dims.back()};
            auto reshape_fmt = sc_data_format_t::NKkn(
                    trans1_plain_dims.at(trans1_plain_dims.size() - 2),
                    trans1_plain_dims.back());
            sc_op_ptr reshape_node = graph->make("tensor_view", {trans1},
                    {graph_tensor::make(reshape_dest, reshape_fmt,
                            trans1->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", reshape_fmt},
                            {"forbid_penetrate", true}});
            trans1 = reshape_node->get_outputs()[0];
        }
    }
    int M = trans0->details_
                    .get_plain_dims()[trans0->details_.get_plain_dims().size()
                            - 2];
    int K = trans0->details_.get_plain_dims().back();
    int N = trans1->details_.get_plain_dims().back();
    bool use_mmm = attrs_.get_or_else("use_mmm", true);
    // manual define output channel axis here for NDxND cases.
    int output_channel_axis = static_cast<int>(
            std::max(trans0_plain_dims.size(), trans1_plain_dims.size()) - 1);
    // We don't allow the dynamic mmm exposed to external in this pr as the
    // total dispatch key is huge.
    if (is_dynamic() || trans0->details_.get_plain_dims().size() > 2
            || trans1->details_.get_plain_dims().size() > 2) {
        matmul = graph->make("matmul_core", {trans0, trans1}, {},
                {{attr_keys::output_channel_axis, output_channel_axis},
                        {"transposed_a", transposed_a},
                        {"transposed_b", transposed_b}});
    } else {
        if (use_mmm) {
            matmul = graph->make("managed_matmul_core", {trans0, trans1}, {},
                    {{"transposed_a", transposed_a},
                            {"transposed_b", transposed_b},
                            {"post_rd_axis", post_rd_axis},
                            {attr_keys::output_channel_axis,
                                    output_channel_axis},
                            {"post_binary",
                                    attrs.get_or_else("post_binary", false)}});
        } else {
            matmul = graph->make("matmul_core", {trans0, trans1}, {},
                    {{attr_keys::output_channel_axis, output_channel_axis},
                            {"transposed_a", transposed_a},
                            {"transposed_b", transposed_b}});
        }
    }

    // view the shape back
    if (!is_dynamic()) {
        // Nd*2d cases
        if (trans0_plain_dims.size() > 2 && trans1_plain_dims.size() == 2) {
            sc_dims reshape_dest
                    = {trans0_plain_dims.begin(), trans0_plain_dims.end() - 1};
            reshape_dest.emplace_back(trans1_plain_dims.back());
            matmul = graph->make("tensor_view", {matmul->get_outputs()[0]},
                    {graph_tensor::make(reshape_dest, sc_data_format_t(),
                            matmul->get_outputs()[0]->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", sc_data_format_t()}});
        }
        // 2d*Nd cases
        if (trans0_plain_dims.size() == 2 && trans1_plain_dims.size() > 2) {
            sc_dims reshape_dest
                    = {trans1_plain_dims.begin(), trans1_plain_dims.end() - 2};
            reshape_dest.emplace_back(trans0_plain_dims[0]);
            reshape_dest.emplace_back(trans1_plain_dims.back());

            matmul = graph->make("tensor_view", {matmul->get_outputs()[0]},
                    {graph_tensor::make(reshape_dest, sc_data_format_t(),
                            matmul->get_outputs()[0]->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", sc_data_format_t()}});
        }
    }
    if (is_bf16 || is_f16) {
        matmul = graph->make("cast", matmul->get_outputs(), {},
                {{"dtype", inputs[0]->details_.dtype_}});
    }

    // check optional input lotgical tensor: bias
    if (info_.inputs_.size() == 3) {
        // create bias op by using broadcast op
        // considering: {bs0, bs1, .., M, N} and {M,N}, for bias, it shape
        // is equal with N.
        if (is_bf16 || is_f16) {
            COMPILE_ASSERT(
                    inputs[2]->details_.dtype_ == inputs[0]->details_.dtype_,
                    "All inputs should have same data type.")
        }
        int last_axis = outputs[0]->details_.get_plain_dims().size() - 1;
        auto bias = graph->make("add",
                {matmul->get_outputs()[0], ins->get_outputs()[2]}, {},
                {{"bc_axis", std::vector<int> {last_axis}}});
        graph->make_output(bias->get_outputs());
    } else {
        graph->make_output(matmul->get_outputs());
    }
} // namespace ops

void matmul_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

} // namespace ops

// matmul op is graph op, matmul_core_op_t is tunable op
OP_REGISTER(ops::matmul_op, matmul)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
