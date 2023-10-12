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

#include <set>

#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <unordered_map>
#include <util/bf16.hpp>
#include <util/fp16.hpp>

SC_MODULE(graph.simplify)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct hash_sc_op_t {
    std::size_t operator()(const sc_op_ptr &v) const {
        size_t hash_ = 0;
        hash_combine(hash_, v->op_name_);
        hash_combine(hash_, v->info_.outputs_[0]->details_);
        hash_combine(hash_, v->hash_contents());
        return hash_;
    }
};

struct compare_sc_op_t {
    bool operator()(const sc_op_ptr &v0, const sc_op_ptr &v1) const {
        return v0->op_name_ == v1->op_name_
                && v0->info_.outputs_[0]->details_
                == v1->info_.outputs_[0]->details_
                && v0->compare_contents(v1.get());
    }
};

static void insert_tensor_view_op(sc_graph_t &graph, const graph_tensor_ptr &in,
        size_t in_index, const sc_op_ptr &cur_op) {
    auto ret = graph.make("tensor_view", {in}, {},
            {{"shape", in->details_.get_plain_dims()}});
    cur_op->replace_input(in_index, ret->get_outputs()[0]);
}

void drop_same_op_on_output(sc_graph_t &graph, const graph_tensor_ptr &output) {
    std::unordered_map<sc_op_ptr, std::vector<int>, hash_sc_op_t,
            compare_sc_op_t>
            same_op_map;
    for (size_t i = 0; i < output->uses_.size(); i++) {
        auto node = output->uses_[i];
        // do not eliminate redundant ops marked as "temp.not_redundant"
        if (node.second->attrs_.get_or_else(
                    op_attr_key::not_redundant, false)) {
            continue;
        }
        if (node.second->get_inputs().empty()
                || node.second->get_outputs().empty()) {
            continue;
        }
        if (node.second->get_inputs().size() > 1
                || node.second->get_outputs().size() > 1) {
            SC_MODULE_INFO
                    << "Currently we don't support multi-input/multi-output op "
                       "elimination.";
            continue;
        }
        // when uses is an output op, we should not remove the global buffer
        auto temp_node = node.second->get_outputs()[0];
        if (std::any_of(temp_node->uses_.begin(), temp_node->uses_.end(),
                    [](const std::pair<int, sc_op_weak_ptr_t> &j) {
                        return j.second->isa<output_op>();
                    })) {
            continue;
        }
        same_op_map[node.second].push_back(i);
    }
    std::vector<std::pair<int, sc_op_ptr>> next_nodes(
            output->uses_.begin(), output->uses_.end());
    for (auto &it : same_op_map) {
        if (it.second.size() > 1) {
            auto reserve_node = next_nodes[it.second[0]].second;
            std::vector<sc_op_ptr> del_node_list;
            for (size_t i = 1; i < it.second.size(); i++) {
                if (it.second[i] >= static_cast<int>(next_nodes.size())
                        || next_nodes[it.second[i]]
                                   .second->get_outputs()[0]
                                   ->uses_.empty()) {
                    break;
                }
                auto del_node = next_nodes[it.second[i]].second;
                std::vector<std::pair<int, sc_op_weak_ptr_t>> del_uses
                        = del_node->get_outputs()[0]->uses_;
                for (size_t u = 0; u < del_uses.size(); u++) {
                    auto node_after_del = del_uses[u];
                    node_after_del.second->replace_input(node_after_del.first,
                            reserve_node->get_outputs()[0]);
                }
                del_node_list.push_back(del_node);
            }
            for (auto &del_node : del_node_list) {
                del_node->remove();
            }
        }
    }
}

// eliminate the tensor_view in which the shape and format are the same for both
// input and output
void useless_tensor_view_elimination(
        sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<tensor_view_op_t>()
                && !node->get_inputs()[0]->producer_owner_->isa<input_op>()) {
            const auto &in_tensor = node->get_inputs()[0]->details_;
            const auto &in_shape = in_tensor.get_plain_dims();
            const auto &in_real_shape = in_tensor.get_blocking_dims();
            const auto &in_format = in_tensor.get_format();
            const auto &out_tensor = node->get_outputs()[0]->details_;
            const auto &out_shape = out_tensor.get_plain_dims();
            const auto &out_real_shape = out_tensor.get_blocking_dims();
            const auto &out_format = out_tensor.get_format();
            if (in_real_shape == out_real_shape && in_shape == out_shape
                    && in_format == out_format) {
                vis->update_state_for_visited(node);
                node->get_outputs()[0]->replace_with(node->get_inputs()[0]);
                node->remove();
            }
        }
    });
    graph.reset_op_ids();
}

// eliminate horizontal same ops, e.g. qkv input reorder
void horizontal_same_op_elimination(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (!node->isa<output_op>()) {
            for (size_t i = 0; i < node->get_outputs().size(); i++) {
                auto output = node->get_outputs()[i];
                drop_same_op_on_output(graph, output);
            }
        }
    });
    graph.reset_op_ids();
}

static bool is_single_use(const sc_op_ptr &node) {
    return node->get_outputs()[0]->uses_.size() == 1;
}

static void merge_dispatch_key_sets(const sc_op_ptr &op0, sc_op_ptr &op1) {
    auto &dispatch_key_set0 = op0->get_dispatch_key_set()->get_inner_set();
    auto &dispatch_key_set1 = op1->get_dispatch_key_set()->get_inner_set();
    std::unordered_map<sc_data_format_t, sc_data_format_t> cached_map;
    for (auto &key1 : dispatch_key_set1) {
        auto &in_fmt1 = key1.in_out_formats_[0];
        auto &out_fmt1 = key1.in_out_formats_[1];
        cached_map[in_fmt1] = out_fmt1;
    }
    dispatch_key_set_t::inner_set_t dispatch_key_set_new0;
    for (auto key0 : dispatch_key_set0) {
        auto &out_fmt0 = key0.in_out_formats_[1];
        auto it = cached_map.find(out_fmt0);
        assert(it != cached_map.end());
        key0.in_out_formats_[1] = it->second;
        dispatch_key_set_new0.insert(key0);
    }
    op0->get_dispatch_key_set()->get_inner_set()
            = std::move(dispatch_key_set_new0);
}

// eliminate excess tensor view, e.g. tensor_view->tensor_view->tensor_view
void excess_tensor_view_elimination(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<tensor_view_op_t>() && is_single_use(node)) {
            sc_op_ptr next_node
                    = node->get_outputs()[0]->uses_[0].second.get_shared();
            sc_op_ptr pre_node = next_node;
            std::vector<sc_op_ptr> node_to_remove;
            while (next_node->isa<tensor_view_op_t>()
                    && is_single_use(next_node)) {
                merge_dispatch_key_sets(node, next_node);
                node_to_remove.push_back(next_node);
                pre_node = next_node;
                next_node = next_node->get_outputs()[0]
                                    ->uses_[0]
                                    .second.get_shared();
            }
            if (next_node->isa<tensor_view_op_t>()) { pre_node = next_node; }
            if (pre_node != next_node || pre_node->isa<tensor_view_op_t>()) {
                if (pre_node == next_node) {
                    merge_dispatch_key_sets(node, pre_node);
                }
                node->get_outputs()[0]->details_
                        = pre_node->get_outputs()[0]->details_;
                std::vector<std::pair<int, sc_op_weak_ptr_t>> uses
                        = pre_node->get_outputs()[0]->uses_;
                for (size_t i = 0; i < uses.size(); i++) {
                    int pre_idx = uses[i].first;
                    uses[i].second->replace_input(
                            pre_idx, node->get_outputs()[0]);
                }
                for (auto &del_node : node_to_remove) {
                    del_node->remove();
                }
                if (pre_node->isa<tensor_view_op_t>()) { pre_node->remove(); }
                if (node->attrs_.has_key("order")) {
                    node->attrs_.remove("order");
                }
            }
            vis->update_state_for_visited(node);
        }
    });
    graph.reset_op_ids();
}

static bool can_simplify(
        const sc_op_ptr &node, const constant_op_t *in_const_op) {
    auto const_dtype = in_const_op->get_constant_dtype();
    if (in_const_op->get_constant_plain_dims() != sc_dims {1}) { return false; }
    if (const_dtype == datatypes::f32) {
        float constant_val = reinterpret_cast<float *>(
                in_const_op->get_constant_values()->data_)[0];
        if ((constant_val == 0.f
                    && (node->isa<add_op_t>() || node->isa<sub_op_t>()))
                || (constant_val == 1.f
                        && (node->isa<mul_op_t>() || node->isa<div_op_t>())))
            return true;
    } else if (const_dtype == datatypes::s32) {
        int constant_val = reinterpret_cast<int *>(
                in_const_op->get_constant_values()->data_)[0];
        if ((constant_val == 0
                    && (node->isa<add_op_t>() || node->isa<sub_op_t>()))
                || (constant_val == 1
                        && (node->isa<mul_op_t>() || node->isa<div_op_t>())))
            return true;
    } else if (const_dtype == datatypes::bf16) {
        bf16_t constant_val = reinterpret_cast<bf16_t *>(
                in_const_op->get_constant_values()->data_)[0];
        if ((constant_val == bf16_t(0)
                    && (node->isa<add_op_t>() || node->isa<sub_op_t>()))
                || (constant_val == bf16_t(1)
                        && (node->isa<mul_op_t>() || node->isa<div_op_t>())))
            return true;
    } else if (const_dtype == datatypes::f16) {
        fp16_t constant_val = reinterpret_cast<fp16_t *>(
                in_const_op->get_constant_values()->data_)[0];
        if ((constant_val == fp16_t(0)
                    && (node->isa<add_op_t>() || node->isa<sub_op_t>()))
                || (constant_val == fp16_t(1)
                        && (node->isa<mul_op_t>() || node->isa<div_op_t>())))
            return true;
    }
    return false;
}

// eliminate redundant binary op
void redundant_binary_op_elimination(
        sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        vis->update_state_for_visited(node);
        // x + 0 or 0 + x or x * 1 or 1 * x
        if (node->isa<add_op_t>() || node->isa<mul_op_t>()) {
            for (size_t i = 0; i < node->get_inputs().size(); i++) {
                if (node->get_inputs()[i]
                                ->producer_owner_->isa<constant_op_t>()) {
                    auto in_const_op = node->get_inputs()[i]
                                               ->producer_owner_
                                               ->dyn_cast<constant_op_t>();
                    bool do_simplify = can_simplify(node, in_const_op);
                    if (do_simplify) {
                        size_t use_size = node->get_outputs()[0]->uses_.size();
                        int use_idx = 0;
                        for (size_t j = 0; j < use_size; j++) {
                            sc_op_ptr next_node = node->get_outputs()[0]
                                                          ->uses_.at(use_idx)
                                                          .second.get_shared();
                            int idx = node->get_outputs()[0]
                                              ->uses_.at(use_idx)
                                              .first;
                            if (next_node->isa<output_op>()
                                    && node->get_inputs()[1 - i]
                                               ->producer_owner_
                                               ->isa<input_op>()) {
                                insert_tensor_view_op(graph,
                                        node->get_inputs()[1 - i], idx,
                                        next_node);
                            } else {
                                next_node->replace_input(
                                        idx, node->get_inputs()[1 - i]);
                                use_idx--;
                            }
                            use_idx++;
                        }
                        node->remove();
                        if (in_const_op->get_outputs()[0]->uses_.empty()) {
                            in_const_op->remove();
                        }
                        break;
                    }
                }
            }
        }
        // x - 0 or x / 1
        else if (node->isa<sub_op_t>() || node->isa<div_op_t>()) {
            if (node->get_inputs()[1]->producer_owner_->isa<constant_op_t>()) {
                auto in_const_op
                        = node->get_inputs()[1]
                                  ->producer_owner_->dyn_cast<constant_op_t>();
                bool do_simplify = can_simplify(node, in_const_op);
                if (do_simplify) {
                    size_t use_size = node->get_outputs()[0]->uses_.size();
                    int use_idx = 0;
                    for (size_t j = 0; j < use_size; j++) {
                        sc_op_ptr next_node = node->get_outputs()[0]
                                                      ->uses_.at(use_idx)
                                                      .second.get_shared();
                        int idx = node->get_outputs()[0]
                                          ->uses_.at(use_idx)
                                          .first;
                        if (next_node->isa<output_op>()
                                && node->get_inputs()[0]
                                           ->producer_owner_->isa<input_op>()) {
                            insert_tensor_view_op(graph, node->get_inputs()[0],
                                    idx, next_node);
                        } else {
                            next_node->replace_input(
                                    idx, node->get_inputs()[0]);
                            use_idx--;
                        }
                        use_idx++;
                    }
                    node->remove();
                    if (in_const_op->get_outputs()[0]->uses_.empty()) {
                        in_const_op->remove();
                    }
                }
            }
        }
    });
    graph.reset_op_ids();
}

static bool is_basic_binary_calculate_op(const sc_op_ptr &node) {
    return node->isa<add_op_t>() || node->isa<sub_op_t>()
            || node->isa<mul_op_t>() || node->isa<div_op_t>();
}

static bool is_high_priority_op(const sc_op_ptr &node) {
    return node->isa<mul_op_t>() || node->isa<div_op_t>();
}

static bool is_lower_priority_op(const sc_op_ptr &node) {
    return node->isa<add_op_t>() || node->isa<sub_op_t>();
}

static bool is_constant_op(const sc_op *node) {
    return node->isa<constant_op_t>()
            || node->attrs_.get_or_else("constant", const_kind::not_const)
            != const_kind::not_const;
}

static bool is_all_positive(const sc_op *node) {
    return node->attrs_.get_or_else("all_positive", false);
}

// a add/mul op with two inputs: in0 and in1, when in0 is a constant and in1
// not, exchange them.
static void exchange_binary_const_ops(
        sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::dfs_topology_sort();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<mul_op_t>() || node->isa<add_op_t>()) {
            if (node->get_inputs()[0]->producer_owner_->attrs_.get_or_else(
                        "constant", const_kind::not_const)
                            != const_kind::not_const
                    && node->get_inputs()[1]
                                    ->producer_owner_->attrs_.get_or_else(
                                            "constant", const_kind::not_const)
                            == const_kind::not_const) {
                auto new_node = graph.make(node->op_name_,
                        {node->get_inputs()[1], node->get_inputs()[0]}, {},
                        node->attrs_);
                node->replace_uses_with_and_remove(new_node);
                vis->update_state_for_visited(new_node);
            }
        }
    });
    graph.reset_op_ids();
}

// For case like: mul + add + relu + (mul/div), change to mul + add + (mul/div)
// + relu for more folding opportunities.
static void push_relu_back(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::dfs_topology_sort_unchecked();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<relu_op_t>()) {
            sc_op_ptr cur_node = node, pre_node = node;
            while (cur_node->is_single_output_single_use()
                    && utils::is_one_of(cur_node->get_outputs()[0]
                                                ->uses_[0]
                                                .second->op_name_,
                            std::string("mul"), std::string("div"))) {
                cur_node = cur_node->get_outputs()[0]->uses_[0].second;
                auto inp1 = cur_node->get_inputs()[1]->producer_owner_;
                if (!(is_constant_op(inp1) && is_all_positive(inp1))) {
                    cur_node = pre_node;
                    break;
                }
                if (pre_node == node) {
                    cur_node->replace_input(
                            node->get_outputs()[0]->uses_[0].first,
                            node->get_inputs()[0]);
                }
                pre_node = cur_node;
            }
            if (cur_node != node) {
                auto &out_tsr = cur_node->get_outputs()[0];
                auto uses = out_tsr->uses_;
                for (auto &use : uses) {
                    use.second->replace_input(
                            use.first, node->get_outputs()[0]);
                }
                node->replace_input(0, out_tsr);
            }
        }
    });
    graph.reset_op_ids();
}

static bool is_same_priority(const sc_op_ptr &op0, const sc_op_ptr &op1,
        const sc_data_type_t &dtype, std::string &elt_type) {
    std::string add = "add", sub = "sub", mul = "mul", div = "div";
    bool ret = false;
    if (utils::is_one_of(op0->op_name_, add, sub)
            && utils::is_one_of(op1->op_name_, add, sub)) {
        if (op0->op_name_ == sub) {
            elt_type = op1->op_name_ == add ? sub : add;
        } else {
            elt_type = op1->op_name_;
        }
        return true;
    }
    if (dtype == datatypes::f32) {
        if (utils::is_one_of(op0->op_name_, mul, div)
                && utils::is_one_of(op1->op_name_, mul, div)) {
            ret = true;
        }
    } else if (dtype == datatypes::s32) {
        if ((op0->op_name_ == mul && op1->op_name_ == mul)
                || (op0->op_name_ == div && op1->op_name_ == div)) {
            ret = true;
        }
    }
    if (ret) {
        if (op0->op_name_ == div) {
            elt_type = op1->op_name_ == mul ? div : mul;
        } else {
            elt_type = op1->op_name_;
        }
    }
    return ret;
}

// Process the pattern like "(a + b) + c" => "a + (b + c)" where b and c are
// constant. Return the new folded node if processing is successful.
static sc_op_ptr same_priority_pattern_fold(
        sc_graph_t &graph, const sc_op_ptr &node) {
    if (is_basic_binary_calculate_op(node)
            && node->is_single_output_single_use()) {
        auto &out_tsr = node->get_outputs()[0];
        auto next_node = out_tsr->uses_[0].second;
        std::string elt_type;
        if (utils::is_one_of(
                    out_tsr->details_.dtype_, datatypes::f32, datatypes::s32)
                && is_same_priority(
                        node, next_node, out_tsr->details_.dtype_, elt_type)) {
            auto *cur_inp1 = node->get_inputs()[1]->producer_owner_;
            auto *next_inp1 = next_node->get_inputs()[1]->producer_owner_;
            if (is_constant_op(cur_inp1) && is_constant_op(next_inp1)
                    && next_inp1 != node.get()) {
                sc_op_ptr cal_const;
                cal_const = graph.make(elt_type,
                        {cur_inp1->get_outputs()[0],
                                next_inp1->get_outputs()[0]},
                        {}, {});
                cal_const->attrs_.set("constant", const_kind::local_const);
                cal_const->attrs_.set("all_positive",
                        is_all_positive(cur_inp1)
                                && is_all_positive(next_inp1));
                auto new_node = graph.make(node->op_name_,
                        {node->get_inputs()[0], cal_const->get_outputs()[0]},
                        {}, node->attrs_);
                new_node->copy_dispatch_key_set_from_op(node);
                node->replace_uses_with_and_remove(new_node);
                auto uses = next_node->get_outputs()[0]->uses_;
                for (auto &use : uses) {
                    use.second->replace_input(
                            use.first, new_node->get_outputs()[0]);
                }
                next_node->remove();
                return new_node;
            }
        }
    }
    return nullptr;
}

// process the partten like "(a * b + c) * d => (a * b * d + c * d)" where b, c,
// d are constant. Return the new folded node if processing is successful.
static sc_op_ptr diff_priority_pattern_fold(
        sc_graph_t &graph, const sc_op_ptr &node) {
    if (is_high_priority_op(node) && node->is_single_output_single_use()) {
        auto &out_tsr = node->get_outputs()[0];
        auto &dtype = out_tsr->details_.dtype_;
        auto next_node = out_tsr->uses_[0].second;
        auto *cur_inp1 = node->get_inputs()[1]->producer_owner_;
        if (!is_constant_op(cur_inp1)) { return nullptr; }
        if (is_lower_priority_op(next_node)
                && node->is_single_output_single_use()) {
            auto nnext_node = next_node->get_outputs()[0]->uses_[0].second;
            auto *next_inp1 = next_node->get_inputs()[1]->producer_owner_;
            if (!is_constant_op(next_inp1)) { return nullptr; }
            if (is_high_priority_op(nnext_node)
                    && next_node->is_single_output_single_use()) {
                auto nnext_inp1 = nnext_node->get_inputs()[1]->producer_owner_;
                if (!is_constant_op(nnext_inp1)) { return nullptr; }
                std::string cur_elt_type = "mul",
                            next_elt_type = nnext_node->op_name_;
                if ((dtype == datatypes::s32 && node->isa<mul_op_t>()
                            && nnext_node->isa<mul_op_t>())
                        || (dtype == datatypes::f32
                                && is_same_priority(node, nnext_node, dtype,
                                        cur_elt_type))) {
                    auto cur_cal_const = graph.make(cur_elt_type,
                            {cur_inp1->get_outputs()[0],
                                    nnext_inp1->get_outputs()[0]},
                            {}, {});
                    cur_cal_const->attrs_.set(
                            "constant", const_kind::local_const);
                    cur_cal_const->attrs_.set("all_positive",
                            is_all_positive(cur_inp1)
                                    && is_all_positive(nnext_inp1));
                    auto new_node = graph.make(node->op_name_,
                            {node->get_inputs()[0],
                                    cur_cal_const->get_outputs()[0]},
                            {}, node->attrs_);
                    new_node->copy_dispatch_key_set_from_op(node);
                    node->replace_uses_with_and_remove(new_node);
                    auto next_cal_const = graph.make(next_elt_type,
                            {next_inp1->get_outputs()[0],
                                    nnext_inp1->get_outputs()[0]},
                            {}, {});
                    next_cal_const->attrs_.set(
                            "constant", const_kind::local_const);
                    next_cal_const->attrs_.set("all_positive",
                            is_all_positive(next_inp1)
                                    && is_all_positive(nnext_inp1));
                    auto new_next_node = graph.make(next_node->op_name_,
                            {next_node->get_inputs()[0],
                                    next_cal_const->get_outputs()[0]},
                            {}, next_node->attrs_);
                    new_next_node->copy_dispatch_key_set_from_op(next_node);
                    next_node->replace_uses_with_and_remove(new_next_node);
                    auto uses = nnext_node->get_outputs()[0]->uses_;
                    auto last_out = new_next_node->get_outputs()[0];
                    for (auto &use : uses) {
                        use.second->replace_input(use.first, last_out);
                    }
                    nnext_node->remove();
                    return new_node;
                }
            }
        }
    }
    return nullptr;
}

static void fold_polynomial(sc_graph_t &graph, const context_ptr &ctx) {
    auto vis = op_visitor_t::bfs();
    constexpr const int MAX_TRY_TIMES = 100;
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        sc_op_ptr last_folded = node, pre_folded = last_folded;
        for (int i = 0; i < MAX_TRY_TIMES; i++) {
            auto same_folded = same_priority_pattern_fold(graph, last_folded);
            if (same_folded) { last_folded = same_folded; }
            auto diff_folded = diff_priority_pattern_fold(graph, last_folded);
            if (diff_folded) { last_folded = diff_folded; }
            if (pre_folded == last_folded) { break; }
            pre_folded = last_folded;
        }
        vis->update_state_for_visited(last_folded);
    });
    graph.reset_op_ids();
}

void graph_constant_folding(sc_graph_t &graph, const context_ptr &ctx) {
    exchange_binary_const_ops(graph, ctx);
    push_relu_back(graph, ctx);
    fold_polynomial(graph, ctx);
}

void graph_simplify(sc_graph_t &graph, const context_ptr &ctx) {
    redundant_binary_op_elimination(graph, ctx);
    excess_tensor_view_elimination(graph, ctx);
    useless_tensor_view_elimination(graph, ctx);
    horizontal_same_op_elimination(graph, ctx);
    graph_constant_folding(graph, ctx);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
