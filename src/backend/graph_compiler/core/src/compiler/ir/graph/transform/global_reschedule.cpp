/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#include <utility>
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../tunable_op.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <ops/fusible/unary_elemwise.hpp>

namespace sc {
/**
 * What is bypass: it starts from the certain op which has more than one user
 * ops, and one of them is tunable op. it means fuse op pass will reparition
 * fused graph starting that tunable op.
 * E.g.
 *       in_a  in_b
 *         \    /
 *        matmul2d   in_c
 *           |      /
 *          bias
 *           |
 *    in_d  quan
 *      \    |       \
 *        matmul2d   dequan
 *           |       /
 *          add
 *           |
 *         output
 *
 * In graph above, `deq` or `cast+sub+mul`(after graph inline) is the bypass
 * what we want to search. For each op found in bypsas, it can be fused either
 * previous or post op, global reschedule is aimed to mark suitable fuse attr
 * (break pre/post fuse or even no fused) for each op them according several
 * different rules.
 * */
static bool search_bypass(const sc_op_ptr &master, sc_op_ptr start_node,
        const op_dep_matrix_t &dep,
        std::vector<std::vector<sc_op_ptr>> &bypass_ops_list,
        int max_step = 10) {
    COMPILE_ASSERT(master, "master op should be defined");
    sc_op_ptr next_node = std::move(start_node);
    int step = 1;
    std::vector<sc_op_ptr> bypass_ops;
    while (next_node->is_single_output_single_use()) {
        // This is input fusion, rather than pre-op fusion
        if (next_node == master) return false;
        // found bypass
        if (dep.lookup(master, next_node) == 1) break;
        bypass_ops.emplace_back(next_node);
        next_node = next_node->get_outputs()[0]->uses_[0].second;
        if ((step++) >= max_step) return false;
    }
    bypass_ops_list.emplace_back(bypass_ops);
    return true;
}

static sc_op_ptr search_master(sc_op_ptr start_node, int max_step = 5) {
    sc_op_ptr next_node = std::move(start_node);
    if (next_node->isa<tunable_op_t>()) return next_node;
    int step = 1;
    while (next_node->is_single_output_single_use()) {
        next_node = next_node->get_outputs()[0]->uses_[0].second;
        if (next_node->isa<tunable_op_t>()) return next_node;
        if (step >= max_step) return nullptr;
        ++step;
    }
    return nullptr;
}

// collect_bypass will check master and bypass branch
static void collect_bypass(sc_graph_t &graph, const context_ptr &ctx,
        std::vector<std::vector<sc_op_ptr>> &bypass_ops_list) {
    auto vis = op_visitor_t::bfs();
    op_dep_matrix_t dep(graph);
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (node->isa<constant_op_t>()) return;
        for (auto &out : node->get_outputs()) {
            if (out->uses_.size() > 1) {
                // this map record uses_[i] and tunable op in possible master
                // branch. If not searched, return nullptr.
                std::unordered_map<sc_op_ptr, sc_op_ptr> mp;
                // one of its use is tunable op
                for (auto &u : out->uses_) {
                    mp[u.second] = search_master(u.second);
                }
                for (auto &u : out->uses_) {
                    if (mp[u.second]) continue;
                    for (auto &p : mp) {
                        if (p.second
                                && search_bypass(p.second, u.second, dep,
                                        bypass_ops_list))
                            break;
                    }
                }
            }
        }
    });
}

/**
 * The reschedule bypass rule is:
 * 1. upwards visit
 * 2. scan bypass ops list
 *    a. if op is elementwise op: check dims to ensure no reduntant computation
 * is introduced.
 *    b. if op is reduce op: break post fuse, and return
 *    c. if op is reorder op: check padding firstly, if true, break post fuse,
 *       and return. Otherwise, check whether out loop mode is supported (dst
 *       is blocking format).
 *       @note: it will try to swap reorder op position firstly in avoid of
 *       unnecessay fusion break.
 * */
static void reschedule_bypass(std::vector<sc_op_ptr> bypass_ops) {
    if (bypass_ops.empty()) return;
    // the size of uses_ equals to one, ensured by search_bypass
    auto ref_op = bypass_ops.back()->get_outputs()[0]->uses_[0].second;
    if (!(ref_op->isa<binary_elementwise_op_t>())) return;
    // first op in bypass list
    auto begin_op = bypass_ops.front();
    for (int64_t i = static_cast<int64_t>(bypass_ops.size()) - 1; i >= 0; i--) {
        auto cur_op = bypass_ops[i];
        if (cur_op->isa<unary_elementwise_op_t>()
                || cur_op->isa<binary_elementwise_op_t>()) {
            // add rule here
        } else if (cur_op->isa<reduce_op_t>()) {
            // break post fuse
            if (cur_op != bypass_ops.back()) {
                // bypass_ops[i + 1] is elementwise or tensor_view op
                bypass_ops[i + 1]->attrs_.set(
                        op_attr_key::break_post_fuse, true);
            } else {
                cur_op->attrs_.set(op_attr_key::break_post_fuse, true);
            }
            break;
        } else if (cur_op->isa<reorder_op_t>()) {
            // support automatically position switch for unary and special
            // broadcast ops
            if (cur_op.get() == bypass_ops.back().get()) {
                auto last_reorder = cur_op->dyn_cast<reorder_op_t>();
                sc_op *pre_op = cur_op.get();
                sc_data_type_t cached_dtype
                        = cur_op->get_inputs()[0]->details_.dtype_;
                // search the swtichable ops
                int j = i;
                int new_i = j;
                std::unordered_map<int, int> idx_mp;
                while (pre_op != begin_op.get()) {
                    pre_op = bypass_ops[--j].get();
                    if (pre_op->isa<unary_elementwise_op_t>()) {
                        if (pre_op->isa<cast_op_t>()) {
                            cached_dtype
                                    = pre_op->get_inputs()[0]->details_.dtype_;
                        }
                        idx_mp[j] = 0;
                        new_i = j;
                    } else if (pre_op->isa<binary_elementwise_op_t>()) {
                        int bc_idx = pre_op->dyn_cast<binary_elementwise_op_t>()
                                             ->get_broadcast_input();
                        if (bc_idx < 0) break;
                        if (auto const_op
                                = pre_op->get_inputs()[bc_idx]
                                          ->producer_owner_
                                          ->dyn_cast<constant_op_t>()) {
                            auto dims = const_op->get_constant_blocking_dims();
                            if (dims.size() == 1 && dims[0] == 1) {
                                idx_mp[j] = 1 - bc_idx;
                                new_i = j;
                                continue;
                            }
                        }
                        break;
                    } else {
                        break;
                    }
                }
                if (new_i > 0 && new_i < i
                        && bypass_ops[new_i - 1]->isa<reduce_op_t>())
                    new_i++;
                pre_op = bypass_ops[new_i].get();

                // switch pre_op and last_reorder position
                if (pre_op != last_reorder) {
                    auto new_fmt = last_reorder->get_output_format();
                    // disconnect reorder
                    last_reorder->get_outputs()[0]->replace_with(
                            bypass_ops[i - 1]->get_outputs()[0]);
                    // insert reorder
                    last_reorder->replace_input(
                            0, pre_op->get_inputs()[idx_mp[new_i]]);
                    pre_op->replace_input(
                            idx_mp[new_i], last_reorder->get_outputs()[0]);

                    // reset attribute
                    last_reorder->info_.outputs_[0]->details_.dtype_
                            = cached_dtype;
                    // update output format
                    for (int k = new_i; k < i; k++) {
                        bypass_ops[k]->get_outputs()[0]->details_.set_format(
                                new_fmt);
                    }
                }

                if (last_reorder->get_output_format().is_blocking()
                        && !last_reorder->check_padding()) {
                    last_reorder->attrs_.set(op_attr_key::break_pre_fuse, true);
                    break;
                }
            }
            // break post fuse
            cur_op->attrs_.set(op_attr_key::break_post_fuse, true);
            break;
        } else if (cur_op->isa<tensor_view_op_t>()) {
            // add rule here
        } else {
            SC_WARN << "no reschedule rule found for this kind op: "
                    << cur_op->op_name_;
            // break post fuse
            cur_op->attrs_.set(op_attr_key::break_post_fuse, true);
            break;
        }
        if (i == 0) {
            cur_op->attrs_.set(op_attr_key::break_pre_fuse, true);
            return;
        }
    }

    // check begin op
    size_t cnt = 0;
    for (auto &user : begin_op->get_inputs()[0]->uses_) {
        if (user.second == begin_op) continue;
        if (user.second->isa<reorder_op_t>()) {
            user.second->attrs_.set(op_attr_key::break_pre_fuse, true);
        }
        if (user.second->attrs_.get_or_else(op_attr_key::break_pre_fuse, false)
                || user.second->isa<tunable_op_t>())
            ++cnt;
    }
    if ((cnt + 1) < begin_op->get_inputs()[0]->uses_.size())
        begin_op->attrs_.set(op_attr_key::break_pre_fuse, true);
}

static void bypass_preop_fusion_rule(
        sc_graph_t &graph, const context_ptr &ctx) {
    std::vector<std::vector<sc_op_ptr>> bypass_ops_list;
    collect_bypass(graph, ctx, bypass_ops_list);
    // reschedule all bypass ops
    for (auto &bypass_ops : bypass_ops_list) {
        reschedule_bypass(bypass_ops);
    }
}

using gr_rule = std::function<void(sc_graph_t &graph, context_ptr ctx)>;

static std::vector<gr_rule> gr_rule_list = {bypass_preop_fusion_rule};

void global_reschedule(sc_graph_t &graph, const context_ptr &ctx) {
    for (auto &gr_rule : gr_rule_list) {
        gr_rule(graph, ctx);
    }
}

} // namespace sc
