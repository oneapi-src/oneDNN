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
#include "visitor.hpp"
#include <algorithm>
#include <sstream>
#include <utility>
#include "fusible_op.hpp"
#include "fusible_op_utils.hpp"
#include "fusion_mgr.hpp"
#include "tunable_op.hpp"
#include <compiler/ir/graph/pass/pass.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/assert.hpp>
#include <util/def.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

void op_visitor_t::visit_graph(const sc_graph_t &mgr, const visitor_func &f) {
    for (auto &v : mgr.ops_) {
        if (v->isa<input_op>() || v->isa<constant_op_t>()) {
            to_visit_.emplace_back(v);
        }
    }
    int original_number_of_ops = mgr.ops_.size();
    visit(f);
    if (check_all_ops_visited_) {
        assert_all_ops_visited(mgr, original_number_of_ops);
    }
}

void op_visitor_t::visit(const visitor_func &f) {
    while (!to_visit_.empty()) {
        auto ptr = select_next_node(this);
        // if selector fails (e.g. found a node that has already been visited),
        // try again
        if (!ptr || ptr->is_removed_) { continue; }
        f(this, ptr);
        update_state_for_visited(std::move(ptr));
    }
}

void op_visitor_t::assert_all_ops_visited(
        const sc_graph_t &mgr, size_t concerned_size) {
#ifndef NDEBUG
    std::vector<sc_op_ptr> not_visited_ops;
    for (size_t i = 0; i < concerned_size; i++) {
        // check whether all non-removed ops within concerned_size are visited
        const auto &op = mgr.ops_[i];
        if (!op->is_removed_ && !has_visited(op->logical_op_id_)) {
            not_visited_ops.emplace_back(op);
        }
    }
    if (!not_visited_ops.empty()) {
        // some ops are not visited, assertion failed
        std::stringstream error_message;
        error_message << "Illegal state for op_visitor_t. The following "
                      << not_visited_ops.size()
                      << " ops were not visited, possibly due to changing the "
                         "graph during the visit without calling "
                         "update_state_for_visited(): ";
        for (const auto &op : not_visited_ops) {
            auto name = op->op_name_ + std::to_string(op->logical_op_id_);
            error_message << name << ' ';
        }
        COMPILE_ASSERT(false, error_message.str());
    }
#else
    for (size_t i = 0; i < concerned_size; i++) {
        const auto &op = mgr.ops_[i];
        COMPILE_ASSERT(op->is_removed_ || has_visited(op->logical_op_id_),
                "Illegal state for op_visitor_t. Some ops were not visited, "
                "possibly due to changing the graph during the visit without "
                "calling update_state_for_visited().");
    }
#endif
}

void op_dep_matrix_t::update(const sc_op_ptr &cur) {
    int cur_id = cur->logical_op_id_;
    // set all inputs=1
    for (auto &input : cur->get_inputs()) {
        int dep_id = input->producer_owner_->logical_op_id_;
        matrix_[dep_id][cur_id] = 1;
        matrix_[cur_id][dep_id] = -1;
        // set all recursive inputs=1
        for (size_t i = 0; i < matrix_.size(); i++) {
            if (matrix_[i][dep_id] == 1) {
                matrix_[i][cur_id] = 1;
                matrix_[cur_id][i] = -1;
            }
        }
    }
}

// Move Op in topology sequence(op_seq) from src position to dst position.
// Meanwhile, it will check op_dep_matrix_t, if dependency not met, it will do
// nothing.
static bool move_op_from_to(std::vector<sc_op_ptr> &op_seq,
        const op_dep_matrix_t &dep_matrix, int src, int dst,
        bool ignore_dep = false) {
    // automatically skip if src==dst
    if (src == dst) return false;
    // need to check op_seq from src+1 to dst dependence in dep_matrix
    int src_op_id = op_seq[src]->logical_op_id_;
    if (src < dst) {
        for (int i = src + 1; i < dst; i++) {
            int cur_op_id = op_seq[i]->logical_op_id_;
            if (!ignore_dep && dep_matrix.lookup(src_op_id, cur_op_id) == 1) {
                // can not move due to dependence
                return false;
            }
        }
    }
    // need to check op_seq from dst to src-1 dependence in dep_matrix
    else {
        for (int i = dst; i < src; i++) {
            int cur_op_id = op_seq[i]->logical_op_id_;
            if (!ignore_dep && dep_matrix.lookup(cur_op_id, src_op_id) == 1) {
                // can not move due to dependence
                return false;
            }
        }
    }
    // move op in seq from src to dst
    op_seq.insert(op_seq.begin() + dst, op_seq[src]);
    op_seq.erase(op_seq.begin() + src + (src <= dst ? 0 : 1));
    return true;
}

void op_sorting_visitor_t::visit_by_rules(sc_graph_t &graph,
        const std::function<void(sc_op_ptr)> &f, std::vector<sort_rule> rules) {
    std::sort(rules.begin(), rules.end());
    std::vector<rule_func> f_rule_list;
    for (auto &r : rules) {
        switch (r) {
            // This rule will make same kind op closer, which can expose more
            // opportunity to loop merge pass
            case sort_rule::same_kind:
                f_rule_list.emplace_back(create_same_kind_rule());
                break;
            case sort_rule::fusion_anchor:
                f_rule_list.emplace_back(create_fusion_anchor_rule());
                break;
            case sort_rule::preop_fusion:
                f_rule_list.emplace_back(create_preop_fusion_rule());
                break;
            default: break;
        }
    }
    auto f_rule = [f_rule_list](std::vector<sc_op_ptr> &op_seq,
                          const op_dep_matrix_t &dep_matrix) {
        for (auto &f_rule : f_rule_list) {
            f_rule(op_seq, dep_matrix);
        }
    };
    visit_by_rules(graph, f, f_rule);
}

op_dep_matrix_t::op_dep_matrix_t(const sc_graph_t &graph)
    : op_dep_matrix_t(graph.ops_.size()) {
    int op_size = graph.ops_.size();
    auto &dep_matrix = *this;
    op_visitor_t::dfs_topology_sort(op_size).visit_graph(
            graph, [&dep_matrix](op_visitor_t *vis, const sc_op_ptr &cur) {
                dep_matrix.update(cur);
            });
}

void op_sorting_visitor_t::visit_by_rules(sc_graph_t &graph,
        const std::function<void(sc_op_ptr)> &f, const rule_func &f_rule) {
    int op_size = graph.ops_.size();
    // Dependence Matrix for Graph Ops, value of postion<i,j> means whether j-th
    // op depends on i-th op.
    op_dep_matrix_t dep_matrix(op_size);
    std::vector<sc_op_ptr> op_seq;
    // Step 1: visit whole graph and update Adj-Matrix.
    op_visitor_t::dfs_topology_sort(op_size).visit_graph(graph,
            [&op_seq, &dep_matrix](op_visitor_t *vis, const sc_op_ptr &cur) {
                op_seq.emplace_back(cur);
                dep_matrix.update(cur);
            });
    // Step 2: change order by rules
    f_rule(op_seq, dep_matrix);
    // Step 3: excute ops by new order
    for (auto &op : op_seq) {
        f(std::move(op));
    }
}

std::vector<sc_op_ptr> op_sorting_visitor_t::sort_by_rules(
        sc_graph_t &graph, std::vector<sort_rule> rules) {
    std::vector<sc_op_ptr> sorted_op_list;
    auto f = [&sorted_op_list](const sc_op_ptr &cur) {
        sorted_op_list.emplace_back(cur);
    };
    visit_by_rules(graph, f, std::move(rules));
    return sorted_op_list;
}

std::vector<sc_op_ptr> op_sorting_visitor_t::sort_by_rules(
        sc_graph_t &graph, const rule_func &f_rule) {
    std::vector<sc_op_ptr> sorted_op_list;
    auto f = [&sorted_op_list](const sc_op_ptr &cur) {
        sorted_op_list.emplace_back(cur);
    };
    visit_by_rules(graph, f, f_rule);
    return sorted_op_list;
}

void op_visitor_t::update_state_for_visited(sc_op_ptr node) {
    update_visit_list(this, std::move(node));
}

op_visitor_t::op_visitor_t(selector_func select_next_node_func,
        updater_func update_visit_list_func, bool check_all_ops_visited)
    : select_next_node(std::move(select_next_node_func))
    , update_visit_list(std::move(update_visit_list_func))
    , check_all_ops_visited_(check_all_ops_visited) {
    visited_.reserve(256);
}

void op_visitor_t::set_visited(int id) {
    assert(id >= 0);
    if ((unsigned)id >= visited_.size()) { visited_.resize(id + 1); }
    visited_[id] = true;
}

bool op_visitor_t::has_visited(int id) {
    assert(id >= 0);
    if ((unsigned)id >= visited_.size()) { return false; }
    return visited_[id];
}

void op_visitor_t::push_back_updater(op_visitor_t *v, const sc_op_ptr &cur) {
    v->set_visited(cur->logical_op_id_);
    for (auto &lt : cur->get_outputs()) {
        for (auto &user : lt->uses_) {
            v->to_visit_.emplace_back(user.second);
        }
    }
};

namespace op_kind {
static constexpr int elementwise = 0;
static constexpr int broadcast = 1;
static constexpr int reduce = 2;
// TODO(xxx): extend it.
static constexpr int others = 4;
}; // namespace op_kind

op_sorting_visitor_t::rule_func op_sorting_visitor_t::create_same_kind_rule() {
    return [](std::vector<sc_op_ptr> &op_seq,
                   const op_dep_matrix_t &dep_matrix) {
        // op kind taxonomy
        auto get_op_kind = [](const sc_op_ptr &cur) {
            if (cur->isa<unary_elementwise_op_t>()) {
                return op_kind::elementwise;
            } else if (auto belemop
                    = cur->dyn_cast<binary_elementwise_op_t>()) {
                auto anchor_id = cur->dyn_cast<fusible_op_t>()->anchor_id_;
                if (belemop->get_broadcast_input() >= 0) {
                    return op_kind::broadcast;
                } else {
                    return op_kind::elementwise;
                }
            } else if (cur->isa<reduce_op_t>()) {
                return op_kind::reduce;
            } else {
                // Movement kind of op etc.
                return op_kind::others;
            }
        };
        // use a map to record same kind ops and their index in topology
        // sequence
        std::unordered_map<int, std::vector<int>> op_name_idx_map;
        for (int i = op_seq.size() - 1; i >= 0; i--) {
            auto kind = get_op_kind(op_seq[i]);
            if (kind != op_kind::others) {
                op_name_idx_map[kind].emplace_back(i);
            }
        }
        // Iterate map, and find opportunity for reorder.
        for (auto &m : op_name_idx_map) {
            int pre_idx = m.second.at(0);
            for (auto &cur_idx : m.second) {
                // if not neighboring, try to make them closer, note that pre is
                // larger than cur here
                if ((pre_idx - cur_idx) > 1) {
                    move_op_from_to(op_seq, dep_matrix, cur_idx, pre_idx);
                }
                pre_idx = cur_idx;
            }
        }
    };
}

op_sorting_visitor_t::rule_func
op_sorting_visitor_t::create_fusion_anchor_rule() {
    /**
     * The sorted rule will be three steps
     * 1. inquire fusion_anchor of each op,
     * 2. sort op by ascending fusion_anchor
     * 3. reset fusion_anchor for op which following up the reduce_op_t
     * */
    return [](std::vector<sc_op_ptr> &op_seq,
                   const op_dep_matrix_t &dep_matrix) {
        // automatically skip
        if (op_seq.empty()) return;

        // sorted op similar to insertion sorting by op anchor level
        for (int i = 1; i < static_cast<int>(op_seq.size()); i++) {
            auto cur_op = op_seq[i]->dyn_cast<fusible_op_t>();
            for (int j = i - 1; j >= 0; j--) {
                auto pre_op = op_seq[j]->dyn_cast<fusible_op_t>();
                if (cur_op->anchor_id_ < pre_op->anchor_id_) {
                    if (move_op_from_to(op_seq, dep_matrix, i, j)) break;
                }
            }
        }
    };
}

op_sorting_visitor_t::rule_func
op_sorting_visitor_t::create_preop_fusion_rule() {
    return [](std::vector<sc_op_ptr> &op_seq,
                   const op_dep_matrix_t &dep_matrix) {
        // automatically skip
        if (op_seq.empty()) return;
        std::vector<sc_op_ptr> anchor_input_list;
        for (auto &cur : op_seq) {
            if (auto input_cur = cur->dyn_cast<input_op>()) {
                if (!input_cur->is_arg_input()) {
                    anchor_input_list.emplace_back(cur);
                }
            }
        }

        auto can_move_forward = [&dep_matrix, &anchor_input_list](sc_op *cur) {
            auto is_depend
                    = [&dep_matrix](const std::vector<sc_op_ptr> &input_list,
                              sc_op *cur) {
                          for (auto &x : input_list) {
                              int input_id = x->logical_op_id_;
                              int target_id = cur->logical_op_id_;
                              if (dep_matrix.lookup(input_id, target_id) == 1) {
                                  return true;
                              }
                          }
                          return false;
                      };

            bool is_depend_anchor = is_depend(anchor_input_list, cur);

            if (is_depend_anchor) { return true; }
            return false;
        };

        // move all ops related to anchor input to the begining of op_seq
        int top_pos = 0;
        for (int i = 1; i < static_cast<int>(op_seq.size()); i++) {
            if (can_move_forward(op_seq[i].get())) {
                move_op_from_to(op_seq, dep_matrix, i, top_pos++, true);
            }
        }
    };
}

op_visitor_t::updater_func op_visitor_t::create_DAG_updater(
        size_t total_hint, const user_sort_func &sorter) {
    struct count_t {
        int count = -1;
    };
    // the count of pending depending logical tensors for each node
    std::vector<count_t> pending_count;
    return [pending_count, total_hint, sorter](
                   op_visitor_t *v, const sc_op_ptr &cur) mutable {
        v->set_visited(cur->logical_op_id_);
        for (auto &lt : cur->get_outputs()) {
            auto visit_index = sorter(lt);
            for (auto &idx : visit_index) {
                auto user = lt->uses_[idx];
                auto id = user.second->logical_op_id_;
                assert(id >= 0);
                if ((unsigned)id >= pending_count.size()) {
                    // need to extend pending_count
                    if ((unsigned)id < total_hint) {
                        pending_count.resize(total_hint);
                    } else {
                        pending_count.resize((id + 1) * 1.5f);
                    }
                }
                if (pending_count[id].count == -1) {
                    // we have not met it before, initialize the dependency
                    // count
                    pending_count[id].count = user.second->get_inputs().size();
                }
                // the pending count is decreased by 1 because current node is
                // done
                --pending_count[id].count;
                assert(pending_count[id].count >= 0);
                // all dependencies resolved, we can visit it now
                if (pending_count[id].count == 0) {
                    v->to_visit_.emplace_back(user.second);
                }
            }
        }
    };
}

op_visitor_t::updater_func op_visitor_t::create_DAG_updater_post(
        size_t total_hint) {
    struct count_t {
        int count = -1;
    };
    // the count of pending depending logical tensors for each node
    std::vector<count_t> pending_count;
    return [pending_count, total_hint](
                   op_visitor_t *v, const sc_op_ptr &cur) mutable {
        v->set_visited(cur->logical_op_id_);
        for (auto &lt : cur->get_inputs()) {
            auto id = lt->producer_owner_->logical_op_id_;
            if (v->has_visited(id)) { continue; }
            assert(id >= 0);
            if ((unsigned)id >= pending_count.size()) {
                // need to extend pending_count
                if ((unsigned)id < total_hint) {
                    pending_count.resize(total_hint);
                } else {
                    pending_count.resize((id + 1) * 1.5f);
                }
            }
            if (pending_count[id].count == -1) {
                // we have not met it before, initialize the dependency
                // count
                size_t num_count = 0;
                for (const graph_tensor_ptr &out :
                        lt->producer_owner_->get_outputs()) {
                    num_count += out->uses_.size();
                }
                pending_count[id].count = num_count;
            }
            // the pending count is decreased by 1 because current node is
            // done
            --pending_count[id].count;
            assert(pending_count[id].count >= 0);
            // all dependencies resolved, we can visit it now
            if (pending_count[id].count == 0) {
                v->to_visit_.emplace_back(
                        lt->producer_owner_->shared_from_this());
            }
        }
    };
}

static std::vector<int> usr_speculative_sorter(const graph_tensor_ptr &gt) {
    std::vector<int> visit_index;
    visit_index.reserve(gt->uses_.size());
    std::unordered_set<int> visited_set;
    // Step 1: sorted by tunable op cnt
    std::vector<std::pair<size_t, int>> priority_index_list;
    auto sort_by_prority = [&priority_index_list, &visit_index]() {
        std::sort(priority_index_list.begin(), priority_index_list.end(),
                [](const std::pair<size_t, int> &p1,
                        const std::pair<size_t, int> &p2) {
                    return p1.first > p2.first;
                });
        for (auto &p : priority_index_list) {
            visit_index.emplace_back(p.second);
        }
        priority_index_list.clear();
    };
    // counter tunable op users
    for (size_t i = 0; i < gt->uses_.size(); i++) {
        auto u = gt->uses_[i];
        auto tun_cnt = count_tuneop_linearly(u.second, 15);
        if (tun_cnt > 0) {
            priority_index_list.emplace_back(std::make_pair(tun_cnt, i));
            visited_set.insert(i);
        }
    }
    // sort tunable_cnt_index_list by descend
    sort_by_prority();
    // Step 2: sorted by user tensor size
    for (size_t i = 0; i < gt->uses_.size(); i++) {
        if (visited_set.find(i) != visited_set.end()) continue;
        auto u = gt->uses_[i];
        if (u.second->get_outputs().empty()) continue;
        auto dt = u.second->get_outputs()[0]->details_;
        size_t user_tsr_size = utils::get_sizeof_etype(dt.dtype_.type_code_)
                * get_dims_product(dt.get_blocking_dims());
        priority_index_list.emplace_back(std::make_pair(user_tsr_size, i));
        visited_set.insert(i);
    }
    // sort tensor_size_index_list by descend
    sort_by_prority();
    // Step 3: push remaining
    for (size_t i = 0; i < gt->uses_.size(); i++) {
        if (visited_set.find(i) != visited_set.end()) continue;
        visit_index.emplace_back(i);
    }
    return visit_index;
}

op_visitor_t::updater_func op_visitor_t::create_DAG_updater_speculative(
        size_t total_hint) {
    return create_DAG_updater(total_hint, usr_speculative_sorter);
}

sc_op_ptr op_visitor_t::pop_back_selector(op_visitor_t *v) {
    auto ret = v->to_visit_.back();
    v->to_visit_.pop_back();
    if (v->has_visited(ret->logical_op_id_)) { return nullptr; }
    return ret;
}

op_visitor_t op_visitor_t::dfs() {
    return op_visitor_t(pop_back_selector, push_back_updater, true);
}

sc_op_ptr op_visitor_t::dequeue_selector(op_visitor_t *v) {
    auto ret = v->to_visit_.front();
    v->to_visit_.pop_front();
    if (v->has_visited(ret->logical_op_id_)) { return nullptr; }
    return ret;
}

op_visitor_t op_visitor_t::bfs() {
    return op_visitor_t(dequeue_selector, push_back_updater, true);
}

op_visitor_t op_visitor_t::dfs_topology_sort(size_t total_nodes_hint) {
    return op_visitor_t(
            pop_back_selector, create_DAG_updater(total_nodes_hint), true);
}

op_visitor_t op_visitor_t::dfs_topology_speculative_sort(
        size_t total_nodes_hint) {
    return op_visitor_t(pop_back_selector,
            create_DAG_updater_speculative(total_nodes_hint), true);
}

op_visitor_t op_visitor_t::bfs_topology_sort(size_t total_nodes_hint) {
    return op_visitor_t(
            dequeue_selector, create_DAG_updater(total_nodes_hint), true);
}

op_visitor_t op_visitor_t::bfs_unchecked() {
    return op_visitor_t(dequeue_selector, push_back_updater, false);
}

op_visitor_t op_visitor_t::dfs_topology_sort_unchecked(
        size_t total_nodes_hint) {
    return op_visitor_t(
            pop_back_selector, create_DAG_updater(total_nodes_hint), false);
}

void op_visitor_t::post_visit_graph(
        const sc_graph_t &mgr, const visitor_func &f) {
    for (auto &v : mgr.ops_) {
        if (dynamic_cast<output_op *>(v.get())
                || dynamic_cast<constant_op_t *>(v.get())) {
            to_visit_.emplace_back(v);
        }
    }
    int original_number_of_ops = mgr.ops_.size();
    visit(f);
    if (check_all_ops_visited_) {
        assert_all_ops_visited(mgr, original_number_of_ops);
    }
}

sc_op_ptr search_tuneop_linearly(const sc_op_ptr &start_node, int max_step) {
    auto next_node = start_node;
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

int count_tuneop_linearly(const sc_op_ptr &start_node, int step) {
    int cnt = 0;
    auto next_node = start_node;
    if (next_node->isa<tunable_op_t>()) cnt++;
    while (next_node->is_single_output_single_use() && step > 0) {
        next_node = next_node->get_outputs()[0]->uses_[0].second;
        if (next_node->isa<tunable_op_t>()) cnt++;
        step--;
    }
    return cnt;
}

std::vector<sc_op_ptr> search_tuneop_bypass(const context_ptr &ctx,
        const sc_op_ptr &tuneop, const sc_op_ptr &start_node,
        const op_dep_matrix_t &dep, int max_step) {
    if (!tuneop) return {};
    auto next_node = start_node;
    int step = 1;
    std::vector<sc_op_ptr> bypass_ops;
    bool found = false;
    while (next_node->is_single_output_single_use()) {
        // This is input fusion, rather than pre-op fusion
        if (next_node == tuneop) break;
        // found bypass
        if (dep.lookup(tuneop, next_node) == 1) {
            found = true;
            break;
        }
        bypass_ops.emplace_back(next_node);
        next_node = next_node->get_outputs()[0]->uses_[0].second;
        if ((step++) >= max_step) break;
    }
    if (found) { return bypass_ops; }
    return {};
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
