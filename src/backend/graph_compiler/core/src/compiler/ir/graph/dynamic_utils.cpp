/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include "dynamic_utils.hpp"
#include <algorithm>
#include <utility>
#include "dynamic_dispatch_key.hpp"
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/hash_dispatch_table.hpp>
#include <util/utils.hpp>
namespace sc {
void initialize_format_table_with_op(
        const sc_op_ptr &op, op_dispatch_tables_ptr &tb) {
    uint32_t inp_size = op->get_inputs().size();
    auto dispatch_keys = op->get_dispatch_key_set();
    if (op->isa<tunable_op_t>()) {
        auto set_format_by_keys
                = [&](const op_dispatch_key_base_t *dispatch_key) {
                      std::vector<runtime::dispatch_key> all_formats
                              = dispatch_key->convert_to_runtime_format_vec();
                      std::vector<runtime::dispatch_key> keys(
                              all_formats.begin(), all_formats.end() - 1);
                      runtime::dispatch_key value
                              = all_formats[all_formats.size() - 1];
                      tb->format_table_[keys] = {value};
                  };
        dispatch_keys->for_each_key_process(set_format_by_keys);
    } else {
        uint64_t unknown_fmt = 0;
        std::vector<runtime::dispatch_key> keys(inp_size, unknown_fmt);
        auto set_format_by_key
                = [&](const op_dispatch_key_base_t *dispatch_key) {
                      std::vector<runtime::dispatch_key> values
                              = dispatch_key->convert_to_runtime_format_vec();
                      // only one input format known.
                      for (uint32_t i = 0; i < inp_size; i++) {
                          keys.clear();
                          keys.resize(inp_size, unknown_fmt);
                          keys[i] = values[i];
                          tb->format_table_[keys] = values;
                      }
                      // all input format known
                      for (uint32_t i = 0; i < inp_size; i++) {
                          keys[i] = values[i];
                      }
                      tb->format_table_[keys] = values;
                  };
        dispatch_keys->for_each_key_process(set_format_by_key);
    }
}

void add_dispatch_symbol_to_kernel_table(op_dispatch_tables_ptr &tb,
        const op_dispatch_key_base_t *key, const std::string &value) {
    std::vector<runtime::dispatch_key> runtime_keys
            = key->convert_to_runtime_format_vec();
    tb->kernel_table_.insert(std::make_pair(runtime_keys, value));
}

bool can_op_be_dispatched(const sc_op_ptr &op) {
    return op->op_name_ != "input" && op->op_name_ != "output"
            && op->op_name_ != "constant"
            && op->get_dispatch_key_set()->size() > 1;
}

std::vector<dispatch_set_ptr> get_dispatch_set_vec_from_ops(
        const std::vector<sc_op_ptr> &ops) {
    std::vector<dispatch_set_ptr> ret;
    ret.reserve(ops.size());
    for (auto &op : ops) {
        ret.emplace_back(op->get_dispatch_key_set());
    }
    return ret;
}

sc_op_ptr find_parent_dispatch_node(const graph_tensor_ptr &in) {
    auto cur_op = in->producer_owner_;
    while (!(cur_op->isa<tunable_op_t>() || cur_op->isa<input_op>()
            || cur_op->isa<reorder_op_t>())) {
        int layout_input_idx = cur_op->attrs_.get_or_else(
                op_attr_key::layout_input_index, 0);
        cur_op = cur_op->get_inputs()[layout_input_idx]->producer_owner_;
    };
    return cur_op->shared_from_this();
}

sc_op_ptr find_output_linked_tunable_op(const graph_tensor_ptr &in) {
    for (auto &use : in->uses_) {
        auto op = use.second.lock();
        if (op->isa<tunable_op_t>() || op->isa<reorder_op_t>()
                || op->isa<output_op>()) {
            continue;
        }
        if (op->isa<binary_elementwise_op_t>()) {
            auto parent_node = find_parent_dispatch_node(op->get_inputs()[0]);
            if (parent_node->isa<tunable_op_t>()) { return parent_node; }
            parent_node = find_parent_dispatch_node(op->get_inputs()[1]);
            if (parent_node->isa<tunable_op_t>()) { return parent_node; }
        }
        auto ret = find_output_linked_tunable_op(op->get_outputs()[0]);
        if (ret) { return ret; }
    }
    return nullptr;
}

static std::pair<int, int> make_linked_op_output_pair(
        const std::vector<dispatch_set_ptr> &dispatch_keys, int op_idx) {
    assert(op_idx >= 0 && op_idx < static_cast<int>(dispatch_keys.size()));
    return std::make_pair(op_idx,
            static_cast<int>(dispatch_keys[op_idx]
                                     ->get_inner_set()
                                     .begin()
                                     ->in_out_formats_.size())
                    - 1);
}

op_layout_link_vec_t get_op_layout_link_relationships(
        const std::vector<std::shared_ptr<sc_op>> &ops,
        const std::vector<dispatch_set_ptr> &dispatch_keys,
        const sc_op_ptr &modified_inp) {
    op_layout_link_vec_t ret;
    ret.resize(ops.size());
    for (size_t i = 0; i < ops.size(); i++) {
        assert(!dispatch_keys[i]
                        ->get_inner_set()
                        .begin()
                        ->in_out_formats_.empty());
        ret[i].resize(dispatch_keys[i]
                              ->get_inner_set()
                              .begin()
                              ->in_out_formats_.size(),
                std::make_pair(no_link_idx, no_link_idx));
    }
    for (size_t i = 0; i < ops.size(); i++) {
        auto &op = ops[i];
        if (op->isa<reorder_op_t>()) {
            auto parent_inp = find_parent_dispatch_node(op->get_inputs()[0]);
            int linked_op_idx = no_link_idx;
            int op_layout_idx = no_link_idx;
            if (parent_inp->isa<tunable_op_t>() || parent_inp == modified_inp) {
                // input linked with tuanble op output
                if (parent_inp == modified_inp) {
                    linked_op_idx = 0;
                } else {
                    auto it = std::find(ops.begin(), ops.end(), parent_inp);
                    COMPILE_ASSERT(it != ops.end(), "Wrong graph pattern!");
                    linked_op_idx = std::distance(ops.begin(), it);
                }
                op_layout_idx = 0;
            } else {
                // output may be linked with tunable op output
                auto tunable_op
                        = find_output_linked_tunable_op(op->get_outputs()[0]);
                op_layout_idx = 1;
                if (tunable_op) {
                    auto it = std::find(ops.begin(), ops.end(), tunable_op);
                    COMPILE_ASSERT(it != ops.end(), "Wrong graph pattern!");
                    linked_op_idx = std::distance(ops.begin(), it);
                }
            }
            if (linked_op_idx != no_link_idx) {
                auto linked_layout_idx
                        = static_cast<int>(dispatch_keys[linked_op_idx]
                                                   ->get_inner_set()
                                                   .begin()
                                                   ->in_out_formats_.size())
                        - 1;
                if (linked_op_idx < static_cast<int>(i)) {
                    ret[i][op_layout_idx]
                            = std::make_pair(linked_op_idx, linked_layout_idx);
                } else {
                    ret[linked_op_idx][linked_layout_idx]
                            = std::make_pair(i, op_layout_idx);
                }
            }
        } else {
            // tunable_op link
            assert(op->isa<tunable_op_t>());
            for (size_t j = 0; j < op->get_inputs().size(); j++) {
                auto parent_node
                        = find_parent_dispatch_node(op->get_inputs()[j]);
                if (parent_node->isa<tunable_op_t>()
                        || parent_node->isa<reorder_op_t>()) {
                    auto it = std::find(ops.begin(), ops.end(), parent_node);
                    COMPILE_ASSERT(it != ops.end(), "Wrong graph pattern!");
                    int op_idx = std::distance(ops.begin(), it);
                    auto linked_layout_idx
                            = static_cast<int>(dispatch_keys[op_idx]
                                                       ->get_inner_set()
                                                       .begin()
                                                       ->in_out_formats_.size())
                            - 1;
                    if (op_idx < static_cast<int>(i)) {
                        ret[i][j] = std::make_pair(op_idx, linked_layout_idx);
                    } else {
                        ret[op_idx][linked_layout_idx] = std::make_pair(i, j);
                    }
                }
            }
        }
    }
    return ret;
}

bool is_linked_layout(
        const sc_data_format_t &layout1, const sc_data_format_t &layout2) {
    if (!(layout1.is_blocking() && layout2.is_blocking())) {
        return layout1 == layout2;
    }
    // both blocking
    auto block_num1 = layout1.get_blocks_size();
    auto block_num2 = layout2.get_blocks_size();
    auto offset1 = block_num1 < block_num2 ? 0 : block_num1 - block_num2;
    auto offset2 = block_num1 < block_num2 ? block_num2 - block_num1 : 0;
    auto block_num = std::min(block_num1, block_num2);
    for (int i = 0; i < block_num; i++) {
        if (!(layout1.blocks_[offset1 + i] == layout2.blocks_[offset2 + i]
                    || layout1.blocks_[offset1 + i] == 1
                    || layout2.blocks_[offset2 + i] == 1)) {
            return false;
        }
    }
    return true;
}

runtime::dynamic_tensor_t convert_graph_tensor_to_dynamic_tensor(
        const graph_tensor_ptr &in, void *data_ptr, sc_dim *shape_ptr) {
    runtime::dynamic_tensor_t ret;
    auto &plain_dims = in->details_.get_plain_dims();
    ret.data_ = data_ptr;
    ret.dims_ = shape_ptr;
    ret.ndims_ = static_cast<int>(plain_dims.size());
    ret.dtype_ = static_cast<uint32_t>(in->details_.dtype_.type_code_);
    ret.dyn_mask_ = 0;
    for (int i = 0; i < static_cast<int>(plain_dims.size()); i++) {
        if (is_dynamic_dim(plain_dims[i])) { ret.dyn_mask_ |= (1 << i); }
    }
    return ret;
}

static bool need_query_next_first(const sc_op_ptr &node) {
    bool has_tail_reorder = node->isa<reorder_op_t>();
    std::vector<sc_op_ptr> out_ops;
    if (node->isa<fused_op_t>()) {
        out_ops = node->stc_cast<fused_op_t>()
                          ->mgr_->get_graph()
                          .get_output_ops();
    } else if (node->isa<mixed_fuse_op_t>()) {
        out_ops = node->stc_cast<mixed_fuse_op_t>()
                          ->sub_graph_.get_output_ops();
    }
    for (auto &op : out_ops) {
        if (op->get_inputs()[0]->producer_owner_->isa<reorder_op_t>()) {
            has_tail_reorder = true;
            break;
        }
    }
    return has_tail_reorder
            && can_op_be_dispatched(
                    node->get_outputs()[0]->uses_[0].second.lock());
}

void lower_query_function(std::vector<bool> &visited, const sc_op_ptr &node,
        const std::function<void(const sc_op_ptr &)> &callback) {
    if (visited[node->logical_op_id_]) { return; }
    if (node->attrs_.get_or_else("constant", const_kind::not_const)
                    == const_kind::not_const
            && need_query_next_first(node)) {
        for (size_t i = 0; i < node->get_outputs().size(); i++) {
            auto query_node = node->get_outputs()[i]->uses_[0].second.lock();
            lower_query_function(visited, query_node, callback);
        }
    }
    callback(node);
    visited[node->logical_op_id_] = true;
}

void visit_fused_graph_by_query_order(sc_graph_t &graph,
        const std::function<void(const sc_op_ptr &)> &callback) {
    for (auto &op : graph.ops_) {
        if (!op->isa<reorder_op_t>()) { callback(op); }
    }
    for (auto &op : graph.ops_) {
        if (op->isa<reorder_op_t>()) { callback(op); }
    }
}
std::vector<sc_op_ptr> get_graph_inner_dispatch_ops(
        sc_graph_t &graph, int *total_key_num) {
    std::vector<sc_op_ptr> ret;
    // first add tunable_op dispatch key then reorder op
    auto get_inner_ops = [&ret, &total_key_num](const sc_op_ptr &op) {
        if (op->isa<tunable_op_t>() || op->isa<reorder_op_t>()) {
            ret.emplace_back(op);
            if (total_key_num) {
                *total_key_num += static_cast<int>(
                        op->get_inputs().size() + op->get_outputs().size());
            }
        }
    };
    visit_fused_graph_by_query_order(graph, get_inner_ops);
    return ret;
}

void update_graph_format_by_key(const sc_op_ptr &fused_op, sc_graph_t &graph,
        const combined_op_dispatch_key_t &key, int &key_idx,
        size_t node_input_offset, size_t graph_input_offset,
        const sc_op_ptr &modified_inp) {
    auto &node_inputs = fused_op->get_inputs();
    std::vector<bool> visited(graph.ops_.size(), false);
    // currently we store dispatch key of tunable op and reorder only in
    // fused op.
    auto update_format = [&](const sc_op_ptr &node) {
        if (node->isa<tunable_op_t>() || node->isa<reorder_op_t>()) {
            auto &cur_key = key[key_idx++];
            assert(cur_key.in_out_formats_.size() == 2
                    || cur_key.in_out_formats_.size() == 3);
            // update format
            for (size_t i = 0; i < cur_key.in_out_formats_.size() - 1; i++) {
                node->get_inputs()[i]->details_.set_format(
                        cur_key.in_out_formats_[i]);
            }
            if (node->isa<tunable_op_t>()) {
                node->stc_cast<tunable_op_t>()->set_config_by_key(cur_key);
            }
            if (node->isa<reorder_op_t>()
                    && cur_key.in_out_formats_[0].is_blocking()) {
                auto inp_parent
                        = find_parent_dispatch_node(node->get_inputs()[0]);
                if (inp_parent->isa<input_op>() && inp_parent != modified_inp) {
                    inp_parent->get_outputs()[0]->details_.set_format(
                            cur_key.in_out_formats_[0]);
                }
            }
            node->get_outputs()[0]->details_.set_format(
                    cur_key.in_out_formats_[cur_key.in_out_formats_.size()
                            - 1]);
            // update impl alg
            node->info_.cur_impl_ = cur_key.impl_;
        }
    };
    visit_fused_graph_by_query_order(graph, update_format);
    assert(key_idx == static_cast<int>(key.size()));
    graph.attrs_.set("insert_reorder", false);
    graph.attrs_.set("is_output_plain", false);
    layout_propagation(graph);

    // sync fused op's input/output format with inner graph
    auto graph_inputs = graph.get_input_ops();
    assert(node_inputs.size() + graph_input_offset
            == graph_inputs.size() + node_input_offset);
    for (size_t i = 0; i + graph_input_offset < graph_inputs.size(); i++) {
        node_inputs[i + node_input_offset]->details_.set_format(
                graph_inputs[i + graph_input_offset]
                        ->get_outputs()[0]
                        ->details_.get_format());
    }
    // update fused op output format
    auto &node_outputs = fused_op->get_outputs();
    auto graph_outputs = graph.get_output_ops();
    assert(graph_outputs.size() == node_outputs.size());
    for (size_t i = 0; i < graph_outputs.size(); i++) {
        node_outputs[i]->details_.set_format(
                graph_outputs[i]->get_inputs()[0]->details_.get_format());
    }
}

int count_dynamic_dims(const sc_dims &in) {
    int ret = 0;
    for (auto &dim : in) {
        if (is_dynamic_dim(dim)) { ret++; }
    }
    return ret;
}

expr divide_and_ceil(const expr &v, const expr &d) {
    return do_cast_and_fold((v + d - 1) / d);
}
} // namespace sc
