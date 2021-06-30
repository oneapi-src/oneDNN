/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#ifndef BACKEND_FAKE_PATTERN_UTILS_HPP
#define BACKEND_FAKE_PATTERN_UTILS_HPP

#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "backend/fake/fake_backend.hpp"
#include "backend/fake/fake_partition_impl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace fake_impl {

/*!
* \brief Function to do comparison between a graph
         and a pattern. It will search from a graph op,
         and compare its inputs / outputs with the ops in
         the pattern, until all the ops in the pattern are
         exhausted.
* \param graph_op the op in the graph to compare from
* \param selected the set stores the ops have been selected
* \tparam the matched op
*/
inline op_t *per_op_comp(op_t *graph_op, std::unordered_set<op_t *> &selected) {
    // check if it's have already selected and if it's assigned to a partition
    if (selected.count(graph_op) != 0 || graph_op->is_assigned_to_partition()) {
        return nullptr;
    }

    return graph_op;
}

class pattern_utils {
public:
    inline void match(dnnl::graph::impl::graph_t &backend_graph,
            std::vector<op_t *> &fusion_ops);
    inline void fuse(dnnl::graph::impl::graph_t &backend_graph,
            std::vector<op_t *> &fusion_ops);
    pattern_utils() = default;
    pattern_utils(const pattern_utils &) = delete;
    pattern_utils(pattern_utils &&) = delete;
    pattern_utils &operator=(const pattern_utils &) = delete;
};

// function to do pattern matching
inline void pattern_utils::match(dnnl::graph::impl::graph_t &backend_graph,
        std::vector<op_t *> &fusion_ops) {
    std::unordered_set<op_t *> selected;
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        op_t *candidate_op = per_op_comp(cur_op, selected);
        if (!candidate_op) { return impl::status::success; }
        fusion_ops.emplace_back(candidate_op);
        selected.insert(candidate_op);
        return impl::status::success;
    });
}

// function to do fusion but not rewrite the graph
inline void pattern_utils::fuse(dnnl::graph::impl::graph_t &backend_graph,
        std::vector<op_t *> &fusion_ops) {
    std::unordered_set<op_t *> fusion_ops_set;
    for (auto &cur_op : fusion_ops) {
        fusion_ops_set.clear();

        std::shared_ptr<op_t> fused_op(new op_t(cur_op->get_kind()));
        fused_op->merge_attributes(cur_op->get_attributes());

        fusion_ops_set.insert(cur_op);

        // merge the attrs and op ids
        fused_op->merge_attributes(cur_op->get_attributes());
        fused_op->add_op_ids(cur_op->get_op_ids());

        // merge the input tensor
        // FIXME(qun) Here is a potential bug: We assume that the input
        // tensors which have producer will be in prior to the input
        // tensors which have no producer, but this assumption is not
        // always true. However, Above buggy pattern will not be matched
        // by pattern matcher now, because of another bug in our current
        // pattern matcher. We will fix all these bugs in new pattern
        // matcher
        for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
            auto in_value = cur_op->get_input_value(j);
            // if in_value has no producer or its producer isn't in pattern,
            // add this input value to fused op
            if (!in_value->has_producer()
                    || !fusion_ops_set.count(&in_value->get_producer())) {
                auto copied_in_value = std::make_shared<value_t>(
                        in_value->get_logical_tensor(), /*internal*/ true);
                fused_op->add_input(copied_in_value);
            }
        }

        // merge the output tensor
        for (size_t k = 0; k < cur_op->num_outputs(); ++k) {
            auto out_value = cur_op->get_output_value(k);
            auto copied_out_value = std::make_shared<value_t>(
                    out_value->get_logical_tensor(), /*internal*/ true);
            fused_op->add_output(copied_out_value);
        }

        auto pimpl = std::make_shared<fake_partition_impl_t>(
                backend_graph.get_engine_kind());

        // use the fused op to initialize the partition_impl, and merge the
        // informations to it.
        pimpl->init(fused_op.get());

        // transfer the ownership of fusion op from graph to partition
        // note: the fusion op will not be removed from the graph
        pimpl->add_op(cur_op->shared_from_this());
        // claim the op belong to the partition
        cur_op->set_partition(pimpl.get());

        backend_graph.add_partition(pimpl);
    }
}

} // namespace fake_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
