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

#include <unordered_map>

#include "../graph.hpp"
#include "../visitor.hpp"
#include "pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

bool compare_graph(sc_op_ptr &first_diff_lhs, sc_op_ptr &first_diff_rhs,
        const sc_graph_t &lhs, const sc_graph_t &rhs,
        const std::unordered_map<int, int> &lhs_rhs_input_mapping,
        const std::function<bool(const sc_op *, const std::string &)> &filter) {
    std::vector<int> op_id_map(lhs.ops_.size(), -1);
    std::unordered_map<graph_tensor *, graph_tensor *> tsr_map;
    if (lhs_rhs_input_mapping.empty()) {
        auto inops_lhs = lhs.get_input_or_const_ops();
        auto inops_rhs = rhs.get_input_or_const_ops();
        if (inops_lhs.size() != inops_rhs.size()) { return false; }
        for (size_t i = 0; i < inops_lhs.size(); i++) {
            auto &l = inops_lhs[i];
            auto &r = inops_rhs[i];
            op_id_map.at(l->logical_op_id_) = r->logical_op_id_;
        }
    } else {
        for (auto &lhs_rhs_id_map : lhs_rhs_input_mapping) {
            op_id_map.at(lhs_rhs_id_map.first) = lhs_rhs_id_map.second;
        }
    }
    bool not_same = false;
    op_visitor_t visitor {/*selector with early stopping*/
            [&not_same](op_visitor_t *vis) {
                auto ret = op_visitor_t::pop_back_selector(vis);
                return not_same ? sc_op_ptr() : ret;
            },
            op_visitor_t::create_DAG_updater(lhs.ops_.size()), false};

    // a utility macro to return from the visit function and tell the visitor to
    // stop early
#define DO_RETURN \
    { \
        not_same = true; \
        first_diff_lhs = lnode; \
        first_diff_rhs = rnode; \
        return; \
    }

    visitor.visit_graph(
            lhs, [&](op_visitor_t *visitor, const sc_op_ptr &lnode) {
                sc_op_ptr rnode;
                int rhs_node_idx = op_id_map[lnode->logical_op_id_];
                // if cannot find RHS
                if (rhs_node_idx == -1) { DO_RETURN; }
                rnode = rhs.ops_.at(rhs_node_idx);
                // check the name and attrs
                if (!lnode->compare_contents(rnode.get(), filter)) {
                    DO_RETURN;
                }
                // check the mapping of input tensors. because the input nodes
                // are already visited (DAG_updater), we just need to check if
                // the LHS tensor is correctly mapped to RHS tensor
                auto &lhs_inputs = lnode->get_inputs();
                auto &rhs_inputs = rnode->get_inputs();
                if (lhs_inputs.size() != rhs_inputs.size()) { DO_RETURN; }
                for (size_t i = 0; i < lhs_inputs.size(); i++) {
                    auto lhs_tsr = lhs_inputs[i].get();
                    auto rhs_tsr = rhs_inputs[i].get();
                    auto itr = tsr_map.find(lhs_tsr);
                    if (itr == tsr_map.end() || itr->second != rhs_tsr) {
                        DO_RETURN;
                    }
                }

                // the output tensors are owned by the current node, so they
                // should have not been visited. For each output tensor, we need
                // to:
                // 1. check if the logical tensor contents are exactly the same
                // (e.g. shape, dtype, etc.)
                // 2. set the lhs_tsr => rhs_tsr mapping (the mapping will be
                // checked when the tsr is an input of other ops)
                // 3. check if the sizes of uses_ matches for LHS and RHS
                // 4. for each consumer Op of the tensor (which we call "use"),
                // set the op mapping from LHS to RHS

                auto &lhs_outputs = lnode->get_outputs();
                auto &rhs_outputs = rnode->get_outputs();
                if (lhs_outputs.size() != rhs_outputs.size()) { DO_RETURN; }
                for (size_t i = 0; i < lhs_outputs.size(); i++) {
                    auto lhs_tsr = lhs_outputs[i].get();
                    auto rhs_tsr = rhs_outputs[i].get();
                    // 1. contents
                    if (!(lhs_tsr->details_ == rhs_tsr->details_)) {
                        DO_RETURN;
                    }
                    // 2. tsr mapping
                    auto itr = tsr_map.find(lhs_tsr);
                    // if lhs_tsr is already mapped to a rhs, return false
                    if (itr != tsr_map.end()) { DO_RETURN; }
                    tsr_map.insert(std::make_pair(lhs_tsr, rhs_tsr));
                    // 3. size of uses
                    auto &lhs_uses = lhs_tsr->uses_;
                    auto &rhs_uses = rhs_tsr->uses_;
                    if (lhs_uses.size() != rhs_uses.size()) { DO_RETURN; }
                    // 4. op mapping
                    for (size_t u = 0; u < lhs_uses.size(); u++) {
                        auto &lhs_use = lhs_uses[u];
                        auto &rhs_use = rhs_uses[u];
                        if (lhs_use.first != rhs_use.first) { DO_RETURN; }
                        int &mapped_id
                                = op_id_map.at(lhs_use.second->logical_op_id_);
                        // if there is already a mapping in the op map, and the
                        // mapped RHS does not match rhs_use's Op id, return
                        // false
                        if (mapped_id != -1
                                && mapped_id
                                        != rhs_use.second->logical_op_id_) {
                            DO_RETURN;
                        }
                        // set the mapping. When visiting the node, we need this
                        // mapping to find the RHS
                        mapped_id = rhs_use.second->logical_op_id_;
                    }
                }
            });
    return !not_same;
}

bool compare_graph(const sc_graph_t &lhs, const sc_graph_t &rhs,
        const std::unordered_map<int, int> &lhs_rhs_input_mapping,
        const std::function<bool(const sc_op *, const std::string &)> &filter) {
    sc_op_ptr first_diff_lhs, first_diff_rhs;
    return compare_graph(first_diff_lhs, first_diff_rhs, lhs, rhs,
            lhs_rhs_input_mapping, filter);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
