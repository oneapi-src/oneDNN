/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include "../fusible_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include <compiler/ir/graph/dynamic_utils.hpp>
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <runtime/config.hpp>

namespace sc {

bool conv1d_should_flatten(const sc_dims &weight_shape, const sc_dims &strides,
        const sc_dims &paddings, const sc_data_format_t &format,
        bool is_weight_constant) {
    bool res = true;
    if (weight_shape.size() != 4UL) { // should be 2d case
        return false;
    }
    sc_dim kh = weight_shape[2], kw = weight_shape[3];
    for (auto &p : paddings) {
        if (p != 0) { res = false; }
    }
    if (kh != 1 || kw != 1) { res = false; }
    if (format != sc_data_format_t::NCHW()
            && format != sc_data_format_t::NHWC()) {
        res = false;
    }
    if (!is_weight_constant) {
        // TODO(zhicong): improve f32/bf16 training fwd config
        res = false;
    }
    return res;
}

sc_dims get_conv1d_flatten_shape(const sc_data_format_t &format,
        const sc_dims &origin_shape, int merge_bs = 1) {
    COMPILE_ASSERT(
            origin_shape.size() == 4, "Conv1d flatten only support 2d case");
    sc_dims out_shape;
    out_shape.push_back(origin_shape[0]);
    if (format == sc_data_format_t::NCHW()) {
        out_shape.push_back(origin_shape[1]);
        out_shape.push_back(origin_shape[2] * origin_shape[3]);
    } else if (format == sc_data_format_t::NHWC()) {
        out_shape.push_back(origin_shape[3]);
        out_shape.push_back(origin_shape[1] * origin_shape[2]);
    }
    COMPILE_ASSERT(out_shape[0] % merge_bs == 0,
            "N % merge_bs should be equal to zero but get"
                    << out_shape[0] % merge_bs);
    out_shape[2] *= merge_bs;
    out_shape[0] /= merge_bs;

    return out_shape;
}

// Whether to flatten N axis, which will break bs fusion but may bring perf
// benefit to single layer
int should_merge_bs(const int &bs, const int &min_os) {
    auto num_threads = runtime_config_t::get().get_num_threads();
    int minibatch = std::max(sc_dim(1), sc_dim(28) / sc_dim(std::sqrt(min_os)));
    if ((bs / minibatch % num_threads != 0
                && bs / minibatch < 4 * num_threads)) {
        return 1;
    }
    return bs % minibatch == 0 ? minibatch : 1;
}

int minimum_spatial_shape(sc_graph_t &graph) {
    auto vis = op_visitor_t::bfs();
    int min_spatial = std::numeric_limits<int>::max();
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (auto op = node->dyn_cast<ops::conv_fwd_core_op_t>()) {
            int spatial_size = 1;
            auto plain_shape = op->get_inputs()[0]->details_.get_plain_dims();
            COMPILE_ASSERT(plain_shape.size() == 4,
                    "Conv1d flatten only support 2d case");
            for (auto i = 2UL; i < plain_shape.size(); i++) {
                spatial_size *= plain_shape[i];
            }
            if (spatial_size < min_spatial) { min_spatial = spatial_size; }
        }
    });
    return min_spatial;
}

void conv1d_flatten(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    auto minimum_os = minimum_spatial_shape(graph);
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (auto op = node->dyn_cast<ops::conv_fwd_core_op_t>()) {
            auto indims = op->get_inputs()[0]->details_.get_plain_dims();
            auto data_origin_shape
                    = op->get_inputs()[0]->details_.get_blocking_dims();
            auto weight_origin_shape
                    = op->get_inputs()[1]->details_.get_blocking_dims();
            auto ndims = indims.size();
            auto &stride = op->attrs_.get<sc_dims>("strides");
            auto &pads_begin = op->attrs_.has_key("pads_begin")
                    ? op->attrs_.get<sc_dims>("pads_begin")
                    : op->attrs_.get<sc_dims>("paddings");
            auto weight_plain_dims
                    = op->get_inputs()[1]->details_.get_plain_dims();
            auto kh = weight_plain_dims[ndims - 2];
            auto kw = weight_plain_dims[ndims - 1];
            auto data_format = op->get_inputs()[0]->details_.get_format();
            auto weight_format = op->get_inputs()[1]->details_.get_format();
            int merge_bs = should_merge_bs(indims[0], minimum_os);
            auto iw = op->get_inputs()[0]->details_.get_plain_dims()[3];
            auto ih = op->get_inputs()[0]->details_.get_plain_dims()[2];
            auto ow = op->get_outputs()[0]->details_.get_plain_dims()[3];
            auto oh = op->get_outputs()[0]->details_.get_plain_dims()[2];
            bool is_weight_constant
                    = op->get_inputs()[1]->producer_owner_->isa<constant_op_t>()
                    || op->get_inputs()[1]->producer_owner_->attrs_.get_or_else(
                            "constant", const_kind::not_const)
                    || op->get_inputs()[1]->attrs_.get_or_else(
                            "constant", const_kind::not_const);
            if (conv1d_should_flatten(
                        op->get_inputs()[1]->details_.get_plain_dims(), stride,
                        pads_begin, data_format, is_weight_constant)) {
                { // pre tensor_view(data)
                    auto shape
                            = get_conv1d_flatten_shape(sc_data_format_t::NCHW(),
                                    sc_data_format_t::get_padded_plain_shapes(
                                            data_origin_shape, data_format),
                                    merge_bs);
                    auto reorder_op
                            = graph.make("reorder", {op->get_inputs()[0]}, {},
                                    {{"out_format", sc_data_format_t::NHWC()},
                                            {"internal", true}});
                    auto view = graph.make("tensor_view",
                            reorder_op->get_outputs(), {},
                            {{"shape", shape},
                                    {"format", sc_data_format_t::NSC()},
                                    {"expand_dim", std::vector<int> {}}});
                    op->replace_input(0, view->get_outputs()[0]);
                    vis.update_state_for_visited(reorder_op);
                    vis.update_state_for_visited(view);
                }
                { // pre tensor_view(weight)
                    auto shape = get_conv1d_flatten_shape(
                            weight_format, weight_origin_shape);
                    auto reorder_op
                            = graph.make("reorder", {op->get_inputs()[1]}, {},
                                    {{"out_format", sc_data_format_t::KCRS()},
                                            {"internal", true}});
                    auto view = graph.make("tensor_view",
                            reorder_op->get_outputs(), {},
                            {{"shape", shape},
                                    {"format", sc_data_format_t::KCS()},
                                    {"expand_dim", std::vector<int> {}}});
                    op->replace_input(1, view->get_outputs()[0]);
                    vis.update_state_for_visited(reorder_op);
                    vis.update_state_for_visited(view);
                }
                { // post tensor view(output)
                    auto origin_out = op->get_outputs()[0]->copy();
                    op->get_outputs()[0]->replace_with(origin_out);
                    origin_out->producer_owner_ = nullptr;

                    auto dtype = origin_out->details_.dtype_;
                    auto new_conv_out = std::make_shared<graph_tensor>(&(*node),
                            sc_data_format_t::NCS(),
                            get_conv1d_flatten_shape(sc_data_format_t::NCHW(),
                                    op->get_outputs()[0]
                                            ->details_.get_blocking_dims(),
                                    merge_bs),
                            dtype);
                    op->info_.outputs_[0] = new_conv_out;

                    auto reorder_op = graph.make("reorder", op->get_outputs(),
                            {}, {{"out_format", sc_data_format_t::NSC()}});
                    auto view = graph.make("tensor_view",
                            reorder_op->get_outputs(), {},
                            {{"shape", origin_out->details_.get_plain_dims()},
                                    {"format", sc_data_format_t::NHWC()},
                                    {"expand_dim", std::vector<int> {}},
                                    {"push_back", true}});
                    origin_out->replace_with(view->get_outputs()[0]);
                    view->get_dispatch_key_set() = op->get_dispatch_key_set();
                    vis.update_state_for_visited(reorder_op);
                    vis.update_state_for_visited(view);
                }
                if (op->attrs_.has_key("pads_begin")) {
                    op->attrs_.get<sc_dims>("pads_begin") = {0};
                    op->attrs_.get<sc_dims>("pads_end") = {0};
                } else {
                    op->attrs_.get<sc_dims>("paddings") = {0};
                }
                op->attrs_["origin_ih"] = ih;
                op->attrs_["origin_iw"] = iw;
                op->attrs_["origin_ow"] = ow;
                op->attrs_["origin_oh"] = oh;
            }
        }
    });
    graph.reset_op_ids();
}

void useless_tensor_view_elimination(sc_graph_t &graph, const context_ptr &ctx);

enum binary_push_back_kind { NOT_SUPPORT, NORMAL, CONSTANT };
static int binary_can_be_push_back(sc_dims a, sc_dims b) {
    auto size = a.size();
    auto a_total_size = 1;
    auto b_total_size = 1;
    for (auto i = 0UL; i < size; i++) {
        a_total_size *= a[i];
        b_total_size *= b[i];
    }
    int kind = binary_push_back_kind::NOT_SUPPORT;
    if (a_total_size == b_total_size
            && kind == binary_push_back_kind::NOT_SUPPORT) {
        kind = binary_push_back_kind::NORMAL;
    }
    if (b.size() == 1 && kind == binary_push_back_kind::NOT_SUPPORT) {
        kind = binary_push_back_kind::CONSTANT;
    }
    return kind;
}

static sc_dims infer_binary_op_shape(const sc_dims &other_shape,
        const sc_dims &blocking_shape, sc_data_format_t format, int kind) {
    sc_dims real_shape = blocking_shape;
    if (kind == binary_push_back_kind::NORMAL) { real_shape = other_shape; }
    auto plain_shape
            = sc_data_format_t::get_padded_plain_shapes(real_shape, format);
    return plain_shape;
}

// When tensor view has multiple users, tensor_view will be split into multiple
// copy so that every tensor_view has only 1 user. It could increase the chance
// for push_back_tensor_view
void split_tensor_view(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (node->isa<tensor_view_op_t>()) {
            if (node->get_outputs().size() == 1
                    && !node->is_single_output_single_use()) {
                auto uses = node->get_outputs()[0]->uses_;
                int cnt = 0;
                for (auto &use : uses) {
                    if (cnt++ == 0) { continue; }
                    auto new_tv = graph.make("tensor_view", node->get_inputs(),
                            {}, node->attrs_);
                    use.second->replace_input(
                            use.first, new_tv->get_outputs()[0]);
                    vis.update_state_for_visited(new_tv);
                }
            }
        }
        vis.update_state_for_visited(node);
    });
    graph.reset_op_ids();
}

// Push back the tensor_view as far as possible so that the chance for
// fusion could be maximized.
// conv + tensor_view + mul + add + relu ->
// conv + mul + add +  relu + tensor_view
void push_back_tensor_view(sc_graph_t &graph, const context_ptr &ctx) {
    constexpr const int max_try_times = 10;
    bool changed = false;
    for (int i = 0; i < max_try_times; i++) {
        if (changed) {
            split_tensor_view(graph, ctx);
            useless_tensor_view_elimination(graph, ctx);
        }
        auto vis = op_visitor_t::bfs();
        vis.visit_graph(graph, [&](const sc_op_ptr &node) {
            if (node->isa<tensor_view_op_t>()
                    && node->attrs_.get_or_else("push_back", false)) {
                auto cur_node = node;
                auto details = node->get_inputs()[0]->details_;
                auto next_node = cur_node->get_outputs()[0]->uses_[0].second;
                while (cur_node->is_single_output_single_use()
                        && next_node->get_outputs().size() == 1) {
                    next_node = cur_node->get_outputs()[0]->uses_[0].second;
                    int use_idx = cur_node->get_outputs()[0]->uses_[0].first;
                    if (next_node->isa<unary_elementwise_op_t>()
                            || next_node->isa<sc::quantize::quantize_op_t>()
                            || next_node
                                       ->isa<sc::quantize::dequantize_op_t>()) {
                        if (cur_node == node) {
                            next_node->replace_input(
                                    use_idx, cur_node->get_inputs()[0]);
                        }
                        cur_node = next_node;
                        details.dtype_
                                = cur_node->get_inputs()[0]->details_.dtype_;
                        cur_node->get_inputs()[0]->details_ = details;
                        details.dtype_
                                = cur_node->get_outputs()[0]->details_.dtype_;
                        cur_node->get_outputs()[0]->details_ = details;
                    } else if (next_node->isa<binary_elementwise_op_t>()) {
                        auto in0_dim = next_node->get_inputs()[0]
                                               ->details_.get_plain_dims();
                        auto in1_dim = next_node->get_inputs()[1]
                                               ->details_.get_plain_dims();
                        if (cur_node == node) {
                            in0_dim = node->get_inputs()[0]
                                              ->details_.get_blocking_dims();
                        }
                        auto kind = binary_can_be_push_back(
                                node->get_outputs()[0]
                                        ->details_.get_plain_dims(),
                                in1_dim);
                        if (use_idx == 0) {
                            if (kind == binary_push_back_kind::NOT_SUPPORT) {
                                break;
                            }
                            // the input corresponding to tensor view output
                            if (cur_node == node) {
                                next_node->replace_input(
                                        use_idx, cur_node->get_inputs()[0]);
                            }
                            cur_node = next_node;
                            details.dtype_ = cur_node->get_inputs()[0]
                                                     ->details_.dtype_;
                            cur_node->get_inputs()[use_idx]->details_ = details;
                            details.dtype_ = cur_node->get_outputs()[0]
                                                     ->details_.dtype_;
                            cur_node->get_outputs()[0]->details_ = details;

                            // the input not corresponding to tensor view output
                            if (kind != binary_push_back_kind::CONSTANT) {
                                int other_use_idx = use_idx == 0 ? 1 : 0;
                                auto shape = infer_binary_op_shape(
                                        cur_node->get_inputs()[use_idx]
                                                ->details_.get_blocking_dims(),
                                        cur_node->get_inputs()[other_use_idx]
                                                ->details_.get_blocking_dims(),
                                        details.get_format(), kind);
                                auto new_view = graph.make("tensor_view",
                                        {cur_node->get_inputs()[other_use_idx]},
                                        {},
                                        {{"shape", shape},
                                                {"format",
                                                        details.get_format()},
                                                {"expand_dim",
                                                        std::vector<int> {}}});
                                cur_node->replace_input(other_use_idx,
                                        new_view->get_outputs()[0]);
                                vis.update_state_for_visited(new_view);
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if (cur_node != node) {
                    changed = true;
                    auto uses = cur_node->get_outputs()[0]->uses_;
                    node->replace_input(0, cur_node->get_outputs()[0]);
                    // correct datatype as we could meet a cast op.
                    auto out_tsr = node->get_outputs()[0];
                    out_tsr->details_.dtype_
                            = cur_node->get_outputs()[0]->details_.dtype_;
                    for (auto &use : uses) {
                        use.second->replace_input(use.first, out_tsr);
                    }
                }
            }
            vis.update_state_for_visited(node);
        });
        if (!changed) { break; }
    }
    graph.reset_op_ids();
}

void flatten_conv(sc_graph_t &graph, const context_ptr &ctx) {
    if (graph.attrs_.get_or_else("no_conv1d", false)) { return; }
    conv1d_flatten(graph, ctx);
    push_back_tensor_view(graph, ctx);
}
} // namespace sc
