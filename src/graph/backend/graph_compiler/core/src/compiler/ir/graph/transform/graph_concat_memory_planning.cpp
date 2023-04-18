/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#include "../visitor.hpp"
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/transform/concat_memory_planning.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.graph_concat_memory_planning);

/*
For the following graph:
    op0   op1   op2
      \    |    /
       \   |   /
         concat0    op3
           \        /
            \      /
            concat1
Merge the consecutive two concats into one, i.e., merge concat0 into concat1:
    op0    op1    op2
      \     |     /
       \    |    /
        \   |   / op3
         \  |  /  /
          \ | /  /
          concat1
Constraints:
0. the output of concat0 is only use by concat1
1. concat dim must be equal.
*/
SC_INTERNAL_API void merge_concats(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context()) {
    op_visitor_t vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &op) {
        if (!op->isa<concat_op_t>()) { return; }
        auto concat = op->stc_cast<concat_op_t>();
        std::vector<graph_tensor_ptr> ori_inputs = concat->info_.inputs_;
        std::vector<graph_tensor_ptr> new_inputs;
        std::unordered_map<int, int> new_idx_to_ori_idx;
        std::unordered_set<sc_op *> ops_to_remove;
        for (size_t i = 0; i < ori_inputs.size(); ++i) {
            auto &ori_input = ori_inputs[i];
            sc_op *parent_op = ori_input->producer_owner_;
            // 1. parent op is concat and its only output is only used by
            // current op.
            // 2. the concat dim must be equal for current op and parent op
            if (parent_op->isa<concat_op_t>()
                    && parent_op->stc_cast<concat_op_t>()->get_axis()
                            == concat->get_axis()
                    && parent_op->get_outputs()[0]->uses_.size() == 1) {
                SC_MODULE_INFO << "Meets two consecutive concat ops: "
                               << parent_op->logical_op_id_ << " --> "
                               << concat->logical_op_id_
                               << ", same concat dim: " << concat->get_axis();
                // the inputs of parent op are now inputs of current op
                new_inputs.insert(new_inputs.end(),
                        parent_op->info_.inputs_.begin(),
                        parent_op->info_.inputs_.end());
                ops_to_remove.insert(parent_op);
            } else {
                new_idx_to_ori_idx[new_inputs.size()] = i;
                new_inputs.push_back(ori_input);
            }
        }
        if (ori_inputs != new_inputs) {
            op->info_.inputs_ = new_inputs;
            concat->is_input_valid_
                    = std::vector<bool>(new_inputs.size(), true);
            for (size_t j = 0; j < op->info_.inputs_.size(); ++j) {
                // can not call replace_input here because the input idx is
                // changed, and detach_use and attach_use uses different index
                op->info_.inputs_[j]->detach_use(op, new_idx_to_ori_idx[j]);
                new_inputs[j]->attach_use(op, j);
            }
            for (auto &op_to_remove : ops_to_remove) {
                SC_MODULE_INFO << "Remove op: " << op_to_remove->op_name_
                               << op_to_remove->logical_op_id_;
                op_to_remove->remove();
            }
            vis->update_state_for_visited(op);
        }
    });
    graph.reset_op_ids();
}

// Calc multi-dimensional offset of each input to concat.
// Each input has specific offset to the output buffer of concat.
static std::vector<sc_dims> calc_offsets(
        std::vector<sc_dims> &inputs_dims, unsigned concat_dim) {
    size_t num_dims = inputs_dims[0].size();
    std::vector<sc_dims> offsets(inputs_dims.size(), sc_dims(num_dims, 0));
    sc_dim offset = 0;
    // the offset of the first tensor is 0
    for (size_t i = 1; i < inputs_dims.size(); ++i) {
        offset += inputs_dims[i - 1][concat_dim];
        offsets[i][concat_dim] = offset;
    }
    return offsets;
}

/* This function is the preparation for TensorIR pass concat_memory_planning_t.
 * We calculate the offset for each input of concat op and set strides of output
 * of parent op. For detailed explanation, please refer to
 * concat_memory_planning_t. */
static bool set_offsets_and_strides_for_op(
        const std::unordered_set<sc_op_ptr> &ops, sc_op *op,
        const sc_dims &strides,
        std::unordered_map<graph_tensor_ptr,
                std::pair<graph_tensor_ptr, sc_dims>> &concat_in_out) {
    auto concat = op->stc_cast<concat_op_t>();
    unsigned concat_dim = concat->get_axis();
    SC_MODULE_INFO << "Set offset and strides for concat_op: " << op->op_name_
                   << op->logical_op_id_;
    std::vector<graph_tensor_ptr> &inputs = concat->info_.inputs_;
    std::unordered_set<graph_tensor_ptr> inputs_set(
            inputs.begin(), inputs.end());
    if (inputs_set.size() != inputs.size()) {
        SC_MODULE_INFO
                << "There are duplicate inputs to current concat, cannot "
                   "do memory planning, donot set offset and strides";
        return false;
    }
    std::vector<sc_dims> inputs_dims(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs_dims[i] = inputs[i]->details_.get_blocking_dims();
    }
    std::vector<sc_dims> offsets = calc_offsets(inputs_dims, concat_dim);

    // The input of concat is the output of its parent op, so we need to
    // set strides of the output of the parent op.
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto parent_op = inputs[i]->producer_owner_->shared_from_this();
        if (parent_op->isa<input_op>() || parent_op->isa<tensor_view_op_t>()) {
            continue;
        }
        if (ops.count(parent_op)) {
            if (parent_op->isa<mixed_fuse_op_t>()
                    && parent_op->stc_cast<mixed_fuse_op_t>()
                                    ->parti_list_.size()
                            != 1) {
                SC_MODULE_INFO << "Do not support mixed fuse op with "
                                  "multiple partitions";
                continue;
            }

            auto it = find(parent_op->info_.outputs_.begin(),
                    parent_op->info_.outputs_.end(), inputs[i]);
            COMPILE_ASSERT(it != parent_op->info_.outputs_.end(),
                    "Cannot find input tensor");
            size_t output_idx = it - parent_op->info_.outputs_.begin();

            concat_in_out[inputs[i]] = {concat->info_.outputs_[0], offsets[i]};
            inputs[i]->details_.set_strides(strides);
            concat->is_input_valid_[i] = false;
            if (parent_op->isa<mixed_fuse_op_t>()) {
                auto parti = parent_op->stc_cast<mixed_fuse_op_t>()
                                     ->parti_list_[0]
                                     .get();
                std::vector<expr> strides_expr;
                for (auto o : strides) {
                    strides_expr.emplace_back(uint64_t(o));
                }
                parti->func_->params_[output_idx].checked_as<tensor>()->strides_
                        = strides_expr;
            }
            SC_MODULE_INFO << "Set strides of tensor: #"
                           << it - parent_op->info_.outputs_.begin()
                           << " output of op " << parent_op->op_name_
                           << parent_op->logical_op_id_
                           << " to : " << utils::print_vector(strides);
        }
    }

    concat->info_.outputs_[0]->details_.set_strides(strides);

    for (auto &id_op_pair : concat->info_.outputs_[0]->uses_) {
        auto child_op = id_op_pair.second.lock();
        if (ops.count(child_op)) {
            if (child_op->isa<mixed_fuse_op_t>()) {
                auto child_mixed = child_op->stc_cast<mixed_fuse_op_t>();
                // TODO(niuxiaoguang): to support this case
                COMPILE_ASSERT(child_mixed->parti_list_.size() == 1,
                        "The child op of concat should not have multi "
                        "partitions: "
                                << child_mixed->parti_list_.size());
                auto parti = child_mixed->parti_list_[0].get();
                std::vector<expr> strides_expr;
                for (auto o : strides) {
                    strides_expr.emplace_back(uint64_t(o));
                }
                int input_idx
                        = id_op_pair.first + child_op->info_.outputs_.size();
                parti->func_->params_[input_idx].checked_as<tensor>()->strides_
                        = strides_expr;
                SC_MODULE_INFO << "Set strides of tensor: #" << id_op_pair.first
                               << " input of op " << child_op->op_name_
                               << child_op->logical_op_id_
                               << " to : " << utils::print_vector(strides);
            }
        }
    }
    return true;
}

static void find_final_tensor_and_offset(std::unordered_map<graph_tensor_ptr,
        std::pair<graph_tensor_ptr, sc_dims>> &concat_in_out) {
    for (auto &pair : concat_in_out) {
        graph_tensor_ptr curr = pair.first;
        size_t n_dims = curr->details_.get_blocking_dims().size();
        sc_dims final_offset(n_dims, 0);
        while (concat_in_out.find(curr) != concat_in_out.end()) {
            auto parent = concat_in_out[curr].first;
            auto offset = concat_in_out[curr].second;
            curr = parent;
            for (size_t i = 0; i < n_dims; ++i) {
                final_offset[i] = final_offset[i] + offset[i];
            }
        }
        pair.second = {curr, final_offset};
        curr->producer_owner_->attrs_[concat_optim_attr_keys::is_final_concat]
                = true;
    }
}

static void set_final_offsets(sc_op *op,
        std::unordered_map<graph_tensor_ptr,
                std::pair<graph_tensor_ptr, sc_dims>> &concat_in_out) {
    auto concat = op->stc_cast<concat_op_t>();
    std::vector<graph_tensor_ptr> &inputs = concat->info_.inputs_;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (concat_in_out.find(inputs[i]) != concat_in_out.end()) {
            SC_MODULE_INFO << "Set final offset for concat_op: " << op->op_name_
                           << op->logical_op_id_ << " #" << i << " input";
            inputs[i]->attrs_[concat_optim_attr_keys::graph_memory_offset_to]
                    = concat_in_out[inputs[i]].first;
            std::vector<expr> offset_expr;
            for (auto o : concat_in_out[inputs[i]].second) {
                offset_expr.emplace_back(uint64_t(o));
            }
            inputs[i]->attrs_[concat_optim_attr_keys::graph_memory_offset]
                    = offset_expr;
        }
    }
}

bool set_offsets_and_strides_recursively(std::vector<sc_op_ptr> &ops) {
    std::unordered_set<sc_op_ptr> ops_set(ops.begin(), ops.end());
    std::vector<std::vector<sc_op *>> concats_seqs;
    for (sc_op_ptr &op : ops) {
        if (!op->isa<concat_op_t>()) { continue; }
        auto concat = op->stc_cast<concat_op_t>();
        SC_MODULE_INFO << "Meet concat_op: " << op->op_name_
                       << op->logical_op_id_;
        bool new_seq = true;
        // We put the directly connected concats into one sequence.
        // TODO(niuxiaoguang): support the topo that multiple parents are
        // concat.
        for (const auto &inp_lt : concat->info_.inputs_) {
            sc_op *parent_op = inp_lt->producer_owner_;
            if (parent_op->isa<tensor_view_op_t>()) {
                // tensor_view_op_t will be skiped when lowering, so the concat
                // op cannot get memory_offset attr from its new input
                SC_MODULE_INFO << "Do not optimize curr input because it is "
                                  "from a tensor_view_op_t";
                continue;
            }
            if (parent_op->isa<concat_op_t>()
                    && parent_op->stc_cast<concat_op_t>()->get_axis()
                            == concat->get_axis()) {
                SC_MODULE_INFO
                        << "Current concat has a concat parent with same dim";
                if (ops_set.count(parent_op->shared_from_this()) == 0) {
                    SC_MODULE_INFO << "The parent op is not in current "
                                      "partition, so start a new seq.";
                    continue;
                }
                new_seq = false;
                for (auto &seq : concats_seqs) {
                    if (seq.back() == parent_op) {
                        seq.push_back(op.get());
                        break;
                    }
                }
            }
        }
        if (new_seq) { concats_seqs.push_back({op.get()}); }
    }

    SC_MODULE_INFO << "There are " << concats_seqs.size()
                   << " sequences of concat ops.";
    if (!concats_seqs.empty()) {
        std::unordered_map<graph_tensor_ptr,
                std::pair<graph_tensor_ptr, sc_dims>>
                concat_in_out;
        for (auto &concat_seq : concats_seqs) {
            SC_MODULE_INFO << "Process concats seq with " << concat_seq.size()
                           << " concat ops.";
            sc_dims strides = concat_seq.back()
                                      ->info_.outputs_[0]
                                      ->details_.get_strides();
            for (int i = concat_seq.size() - 1; i >= 0; --i) {
                set_offsets_and_strides_for_op(
                        ops_set, concat_seq[i], strides, concat_in_out);
            }
        }

        find_final_tensor_and_offset(concat_in_out);

        for (auto &concat_seq : concats_seqs) {
            for (int i = concat_seq.size() - 1; i >= 0; --i) {
                set_final_offsets(concat_seq[i], concat_in_out);
            }
        }
    }
    return !concats_seqs.empty();
}

SC_INTERNAL_API void graph_concat_memory_planning(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context()) {
    SC_MODULE_INFO << "Run graph concat memory planning on graph with "
                   << graph.ops_.size() << " ops.";
    if (graph.ops_.size() < 2) { return; }
    set_offsets_and_strides_recursively(graph.ops_);
}

SC_INTERNAL_API bool concat_memory_planning_on_graph(sc_graph_t &graph) {
    return set_offsets_and_strides_recursively(graph.ops_);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
