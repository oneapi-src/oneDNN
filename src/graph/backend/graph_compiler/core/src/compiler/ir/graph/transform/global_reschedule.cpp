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

#include <functional>
#include <numeric>
#include <utility>
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

template <typename opT>
static void collect_linked_ops_nearby_input_until_opT(sc_graph_t &graph,
        const context_ptr &ctx,
        std::vector<std::vector<sc_op_ptr>> &target_ops_list) {
    auto vis = op_visitor_t::bfs();
    constexpr int max_step = 5;
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (!node->isa<input_op>()) return;
        for (auto &user : node->get_outputs()[0]->uses_) {
            sc_op_ptr next_node = user.second;
            int step = 1;
            std::vector<sc_op_ptr> target_ops;
            bool found = false;
            while (next_node->is_single_output_single_use()
                    && (!next_node->get_outputs()[0]
                                    ->uses_[0]
                                    .second->isa<output_op>())
                    && (next_node->attrs_.get_or_else(
                                "constant", const_kind::not_const)
                            == const_kind::not_const)) {
                target_ops.emplace_back(next_node);
                if (next_node->isa<opT>()) {
                    found = true;
                    break;
                }
                if ((step++) >= max_step) return;
                next_node = next_node->get_outputs()[0]->uses_[0].second;
            }
            if (found && target_ops.size() > 1) {
                target_ops_list.emplace_back(target_ops);
            }
        }
    });
}

/**
 * The reschedule bypass rule is:
 * 1. upwards visit.
 * 2. scan bypass ops list: find the first non-elementwise op and try to swap
 * reorder op position with it firstly in avoid of unnecessay fusion break.
 * */
static void reschedule_reorder_nearby_input(std::vector<sc_op_ptr> target_ops) {
    if (target_ops.empty()) return;
    // first op in bypass list
    auto begin_op = target_ops.front();
    auto end_op = target_ops.back();
    COMPILE_ASSERT(end_op->isa<reorder_op_t>(),
            "tensorview op is expected, but got " << end_op->op_name_ << "_"
                                                  << end_op->logical_op_id_)
    // support automatically position switch for unary and special
    // broadcast ops
    auto last_reo = end_op->dyn_cast<reorder_op_t>();
    sc_op *pre_op = end_op.get();
    sc_data_type_t cached_dtype = end_op->get_inputs()[0]->details_.dtype_;
    // search the swtichable ops
    int i = target_ops.size() - 1;
    int j = i;
    std::unordered_map<int, int> idx_mp;
    while (pre_op != begin_op.get()) {
        pre_op = target_ops[--i].get();
        if (pre_op->isa<unary_elementwise_op_t>()) {
            if (pre_op->isa<cast_op_t>()) {
                if (utils::get_sizeof_type(
                            pre_op->get_inputs()[0]->details_.dtype_)
                        > utils::get_sizeof_type(
                                pre_op->get_outputs()[0]->details_.dtype_)) {
                    break;
                }
                cached_dtype = pre_op->get_inputs()[0]->details_.dtype_;
            }
            idx_mp[i] = 0;
            j = i;
        } else if (pre_op->isa<binary_elementwise_op_t>()) {
            int bc_idx = pre_op->dyn_cast<binary_elementwise_op_t>()
                                 ->get_broadcast_input();
            if (bc_idx < 0) break;
            if (auto const_op
                    = pre_op->get_inputs()[bc_idx]
                              ->producer_owner_->dyn_cast<constant_op_t>()) {
                auto dims = const_op->get_constant_blocking_dims();
                if (dims.size() == 1 && dims[0] == 1) {
                    idx_mp[i] = 1 - bc_idx;
                    j = i;
                    continue;
                }
            }
            break;
        } else {
            break;
        }
    }
    pre_op = target_ops[j].get();

    // switch pre_op and last_reo position
    if (pre_op == begin_op.get()) {
        auto new_fmt = last_reo->get_outputs()[0]->details_.get_format();
        // disconnect reorder
        last_reo->get_outputs()[0]->replace_with(
                target_ops[target_ops.size() - 2]->get_outputs()[0]);
        // insert reorder
        last_reo->replace_input(0, pre_op->get_inputs()[idx_mp[j]]);
        pre_op->replace_input(idx_mp[j], last_reo->get_outputs()[0]);

        // reset attribute
        last_reo->info_.outputs_[0]->details_.dtype_ = cached_dtype;
        // update output format
        for (int k = j; k < static_cast<int>(target_ops.size()) - 1; k++) {
            target_ops[k]->get_outputs()[0]->details_.set_format(new_fmt);
        }
    }
}

static void nearby_input_reorder_rule(
        sc_graph_t &graph, const context_ptr &ctx) {
    std::vector<std::vector<sc_op_ptr>> target_ops_list;
    collect_linked_ops_nearby_input_until_opT<reorder_op_t>(
            graph, ctx, target_ops_list);
    // reschedule all bypass ops
    for (auto &target_ops : target_ops_list) {
        reschedule_reorder_nearby_input(target_ops);
    }
}

static void find_reorder_and_tv_from_matmul2D2ND(sc_graph_t &graph,
        const context_ptr &ctx,
        std::vector<std::vector<sc_op_ptr>> &target_ops_list,
        std::unordered_map<int, int> &op_bc_idx) {
    auto vis = op_visitor_t::dfs_topology_sort();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        // 1. Find the tensorview op which is generated by matmul. Its parent
        // should be a reorder op, whose input format should be blocking and
        // output format should be plain. If the reorder op is marked as
        // "break_post_fuse", it is our target.
        if (!node->isa<tensor_view_op_t>()) return;
        if (!node->attrs_.get_or_else("source_matmul_2D2ND", false)) return;
        if (!node->is_single_output_single_use()) return;
        auto tv_parent = node->info_.inputs_[0]->producer_owner_;
        if (!tv_parent->isa<reorder_op_t>()) return;
        // Note: break_post_fuse is marked by mixed_partition pass,
        // which is a later one. Can not get it here. if
        // (!tv_parent->attrs_.get_or_else(op_attr_key::break_post_fuse, false))
        //     return;
        if (!tv_parent->is_single_output_single_use()) return;
        auto &reorder_input_format
                = tv_parent->get_inputs()[0]->details_.get_format();
        auto &reorder_output_format
                = tv_parent->get_outputs()[0]->details_.get_format();
        // this reorder is from ABab to AB
        if (!reorder_input_format.format_code_.is_blocking()
                || reorder_input_format.format_code_ != format_kinds::ABab
                || !reorder_output_format.format_code_.is_plain()) {
            return;
        }

        std::vector<sc_op_ptr> target_ops {tv_parent->shared_from_this(), node};
        sc_op_ptr child_node = node->get_outputs()[0]->uses_[0].second;
        while (child_node->is_single_output_single_use()) {
            if (child_node->isa<unary_elementwise_op_t>()) {
                target_ops.emplace_back(child_node);
            } else if (child_node->isa<binary_elementwise_op_t>()) {
                /*
                for binary op, we need to check that it is with broadcast;
                because for topo like
                    matmul0 - reorder0 - tv0 \
                                               add,
                    matmul1 - reorder1 - tv1 /
                the add op should not be moved.
                */
                std::vector<int> non_bc_inputs
                        = child_node->dyn_cast<binary_elementwise_op_t>()
                                  ->get_non_broadcast_input_index(true);
                if (non_bc_inputs.size() != 2) {
                    int bc_idx = child_node->dyn_cast<binary_elementwise_op_t>()
                                         ->get_broadcast_input();
                    op_bc_idx[child_node->logical_op_id_] = bc_idx;
                    target_ops.emplace_back(child_node);
                } else {
                    break;
                }
            } else {
                break;
            }
            child_node = child_node->get_outputs()[0]->uses_[0].second;
        }
        if (target_ops.size() > 2) { target_ops_list.emplace_back(target_ops); }
    });
}

static void postpone_reorder_and_tv(sc_graph_t &graph,
        const std::vector<sc_op_ptr> &target_ops,
        const std::unordered_map<int, int> &op_bc_idx) {
    if (target_ops.size() < 2) return;
    auto first_op = target_ops.front();
    COMPILE_ASSERT(first_op->isa<reorder_op_t>(),
            "reorder op is expected, but got " << first_op->op_name_ << "_"
                                               << first_op->logical_op_id_);
    // target ops: {reorder, tensorview, unary/banary ops}
    auto &reorder_input = first_op->get_inputs()[0];
    auto &reorder_input_format // ABab blocking
            = reorder_input->details_.get_format();
    auto &reorder_output_format // AB plain
            = first_op->get_outputs()[0]->details_.get_format();
    sc_data_format_t::blocking_t blocks = reorder_input_format.blocks_;
    blocks[0] = 1; // only blocking at B-axis
    sc_data_format_t new_fmt
            = sc_data_format_t(reorder_input_format.format_code_, blocks);

    // Move the reorder op and tv op to last. Other ops in original order.
    auto &tv_op = target_ops[1];
    auto &third_op = target_ops[2]; // the op after reorder and tensorview
    auto &last_op = target_ops.back();
    for (auto &use : tv_op->get_outputs()[0]->uses_) {
        // tsr tv_output is single-used by third_op
        third_op->replace_input(use.first, reorder_input);
    }
    for (auto &use : last_op->get_outputs()[0]->uses_) {
        use.second->replace_input(use.first, tv_op->get_outputs()[0]);
    }
    first_op->replace_input(0, last_op->get_outputs()[0]);

    for (size_t i = 2; i < target_ops.size(); ++i) {
        auto &curr_op = target_ops[i];
        if (curr_op->isa<unary_elementwise_op_t>()) {
            auto new_output = std::make_shared<graph_tensor>(curr_op.get(),
                    curr_op->get_inputs()[0]->details_.get_format(),
                    curr_op->get_inputs()[0]->details_.get_plain_dims(),
                    // this may be a cast op, so use dtype of the output tensor
                    curr_op->get_outputs()[0]->details_.dtype_);
            curr_op->get_outputs()[0]->replace_with(new_output);
            curr_op->info_.outputs_[0] = new_output;
        } else if (curr_op->isa<binary_elementwise_op_t>()) {
            // For binary ops, add tensorview to change the format and shape
            int bc_idx = op_bc_idx.at(curr_op->logical_op_id_);
            auto &ori_tsr = curr_op->get_inputs()[bc_idx];
            auto &parent_op = ori_tsr->producer_owner_;
            if (ori_tsr->details_.get_format() != reorder_input_format
                    && ori_tsr->details_.get_plain_dims().size() > 1) {
                // reshape to 1D
                auto &ori_tsr_shape = ori_tsr->details_.get_blocking_dims();
                sc_dims plain_shape = {1,
                        std::accumulate(ori_tsr_shape.begin(),
                                ori_tsr_shape.end(), 1,
                                std::multiplies<sc_dim>())};
                auto new_tv = graph.make("tensor_view",
                        parent_op->get_outputs(), {},
                        {{"shape", plain_shape},
                                {"format",
                                        sc_data_format_t(format_kinds::AB)}});
                // reorder to blocking format
                auto new_reorder = graph.make("reorder", new_tv->get_outputs(),
                        {}, {{"internal", true}, {"out_format", new_fmt}});
                curr_op->replace_input(bc_idx, new_reorder->get_outputs()[0]);
            }
            auto new_output = std::make_shared<graph_tensor>(curr_op.get(),
                    curr_op->get_inputs()[1 - bc_idx]->details_.get_format(),
                    curr_op->get_inputs()[1 - bc_idx]
                            ->details_.get_plain_dims(),
                    curr_op->get_inputs()[1 - bc_idx]->details_.dtype_);
            curr_op->get_outputs()[0]->replace_with(new_output);
            curr_op->info_.outputs_[0] = new_output;
            if (!curr_op->attrs_.get_or_else("bc_axis", std::vector<int> {})
                            .empty()) {
                // Note: broadcast happens at axis #1 (axis N of matmul)
                curr_op->attrs_["bc_axis"] = std::vector<int> {1};
            }
            curr_op->dyn_cast<binary_elementwise_op_impl_t>()
                    ->set_plain_bc_axis();
        }
    }
    // the shape and format of output tensors of reorder and tv are not
    // affected; but if the dtype is changed, rebuild them.
    if (first_op->info_.inputs_[0]->details_.dtype_
            != first_op->info_.outputs_[0]->details_.dtype_) {
        for (size_t i = 0; i < 2; ++i) {
            auto &curr_op = target_ops[i];
            auto new_output = std::make_shared<graph_tensor>(curr_op.get(),
                    curr_op->get_outputs()[0]->details_.get_format(),
                    curr_op->get_outputs()[0]->details_.get_plain_dims(),
                    curr_op->get_inputs()[0]->details_.dtype_);
            curr_op->get_outputs()[0]->replace_with(new_output);
            curr_op->info_.outputs_[0] = new_output;
        }
    }
}

static void postpone_reorder_and_tv_from_matmul2D2ND(
        sc_graph_t &graph, const context_ptr &ctx) {
    std::vector<std::vector<sc_op_ptr>> target_ops_list;
    std::unordered_map<int, int> op_bc_idx;
    find_reorder_and_tv_from_matmul2D2ND(
            graph, ctx, target_ops_list, op_bc_idx);
    for (auto &target_ops : target_ops_list) {
        postpone_reorder_and_tv(graph, target_ops, op_bc_idx);
    }
}

using gr_rule = std::function<void(sc_graph_t &graph, const context_ptr &ctx)>;

void global_reschedule(sc_graph_t &graph, const context_ptr &ctx) {
    if (!graph.attrs_.get_or_else("temp.fuse", 1)) { return; }
    if (graph.is_dynamic()) { return; }
    std::vector<gr_rule> gr_rule_list;
    gr_rule_list.emplace_back(nearby_input_reorder_rule);
    gr_rule_list.emplace_back(postpone_reorder_and_tv_from_matmul2D2ND);
    for (auto &gr_rule : gr_rule_list) {
        gr_rule(graph, ctx);
    }
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
