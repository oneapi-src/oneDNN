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
#ifndef BACKEND_DNNL_PATTERNS_PATTERN_UTILS_HPP
#define BACKEND_DNNL_PATTERNS_PATTERN_UTILS_HPP

#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "utils/compatible.hpp"

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"

#include "utils/pm/nested_matcher.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

class pattern_utils_t {
public:
    inline void match(dnnl::graph::impl::graph_t &backend_graph,
            std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph,
            std::vector<std::vector<op_t *>> &fusion_ops);

    inline void rewrite(dnnl::graph::impl::graph_t &backend_graph,
            op_t *origin_pattern, op_t *optimized_pattern,
            std::vector<std::vector<op_t *>> &fusion_ops);

    inline void fuse(dnnl::graph::impl::graph_t &backend_graph,
            std::vector<std::vector<op_t *>> &fusion_ops,
            op_t &op_with_backend);

    pattern_utils_t() = default;
    pattern_utils_t(const pattern_utils_t &) = delete;
    pattern_utils_t(pattern_utils_t &&) = delete;
    pattern_utils_t &operator=(const pattern_utils_t &) = delete;
};

// function to do v2 pattern matching
inline void pattern_utils_t::match(dnnl::graph::impl::graph_t &backend_graph,
        std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        std::vector<op_t *> candidate_fusion;
        if (!impl::utils::pm::match_pattern(cur_op, pgraph, candidate_fusion)) {
            return status::success;
        }
        fusion_ops.emplace_back(candidate_fusion);
        return status::success;
    });
}

//do fuse with v2 pattern language
inline void pattern_utils_t::fuse(dnnl::graph::impl::graph_t &backend_graph,
        std::vector<std::vector<op_t *>> &fusion_ops, op_t &fused_op) {
    std::vector<op_t *> fusion_ops_set;
    std::unordered_set<op_t *> visit;

    for (auto &pairs : fusion_ops) {
        fusion_ops_set.clear();
        visit.clear();
        std::shared_ptr<op_t> partition_fused_op(new op_t(fused_op.get_kind()));
        partition_fused_op->merge_attributes(fused_op.get_attributes());
        for (size_t i = 0; i < pairs.size(); ++i) {
            visit.insert(pairs[i]);
            fusion_ops_set.push_back(pairs[i]);
        }

        for (auto &cur_op : fusion_ops_set) {
            // merge the attrs and op ids
            partition_fused_op->merge_attributes(cur_op->get_attributes());
            partition_fused_op->add_op_ids(cur_op->get_op_ids());

            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                std::shared_ptr<value_t> in_value = cur_op->get_input_value(j);
                // if in_value has no producer or its producer isn't in pattern,
                // add this input value to fused op
                if (!in_value->has_producer()
                        || !visit.count(&in_value->get_producer())) {
                    std::shared_ptr<value_t> copied_in_value
                            = std::make_shared<value_t>(
                                    in_value->get_logical_tensor(),
                                    /*internal*/ true);
                    partition_fused_op->add_input(copied_in_value);
                }
            }

            // find an end_op to start reorder, find an op whose output isn't in
            // matched_pairs, can't use pb_op's output == 0, because ops defined
            // in subgraph also has 0 output but isn't the end of the pattern.
            bool is_end = true;
            for (auto &output : cur_op->get_output_values()) {
                for (auto &consumer : output->get_consumers()) {
                    if (visit.count(&(consumer.get_op()))) {
                        is_end = false;
                        break;
                    }
                }
                if (!is_end) { break; }
            }

            // merge the output tensor if current op is an end op
            if (is_end) {
                // it's the end op of pattern
                for (size_t k = 0; k < cur_op->num_outputs(); ++k) {
                    std::shared_ptr<value_t> out_value
                            = cur_op->get_output_value(k);
                    std::shared_ptr<value_t> copied_out_value
                            = std::make_shared<value_t>(
                                    out_value->get_logical_tensor(),
                                    /*internal*/ true);
                    partition_fused_op->add_output(copied_out_value);
                }
            }
        }

        std::shared_ptr<dnnl_partition_impl_t> pimpl
                = std::make_shared<dnnl_partition_impl_t>(
                        backend_graph.get_engine_kind());

        // use the fused op to initialize the partition_impl, and merge the
        // informations to it.
        pimpl->init(partition_fused_op.get());

        // transfer the ownership of fusion op from graph to partition
        // note: the fusion op will not be removed from the graph
        for (size_t i = 0; i < pairs.size(); ++i) {
            pimpl->add_op(pairs[i]->shared_from_this());
            // claim the op belong to the partition
            pairs[i]->set_partition(pimpl.get());
        }

        backend_graph.add_partition(pimpl);
    }
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
