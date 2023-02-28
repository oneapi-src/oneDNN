/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_PATTERN_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_PATTERN_UTILS_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "graph/interface/graph.hpp"
#include "graph/interface/partition.hpp"

#include "graph/backend/graph_compiler/compiler_backend.hpp"
#include "graph/backend/graph_compiler/compiler_partition_impl.hpp"

#include "graph/utils/pm/nested_matcher.hpp"
#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

class pattern_utils_t {
public:
    inline void match(graph::graph_t &backend_graph,
            std::shared_ptr<graph::utils::pm::pb_graph_t> pgraph,
            std::vector<std::vector<op_t *>> &fusion_ops);
    inline void set_partitions(graph::graph_t &backend_graph,
            std::vector<std::vector<op_t *>> &fusion_ops,
            graph::partition_kind_t pkind, std::string pname);

    pattern_utils_t() = default;
    pattern_utils_t(const pattern_utils_t &) = delete;
    pattern_utils_t(pattern_utils_t &&) = delete;
    pattern_utils_t &operator=(const pattern_utils_t &) = delete;
};

inline void pattern_utils_t::match(graph::graph_t &backend_graph,
        std::shared_ptr<graph::utils::pm::pb_graph_t> pgraph,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        std::vector<op_t *> candidate_fusion;
        if (!graph::utils::pm::match_pattern(
                    cur_op, pgraph, candidate_fusion)) {
            return status::success;
        }
        fusion_ops.emplace_back(candidate_fusion);
        return status::success;
    });
}

inline void pattern_utils_t::set_partitions(graph::graph_t &backend_graph,
        std::vector<std::vector<op_t *>> &fusion_ops,
        graph::partition_kind_t pkind, std::string pname) {
    std::vector<op_t *> fusion_ops_set;
    std::unordered_set<op_t *> visit;

    for (auto &pairs : fusion_ops) {
        fusion_ops_set.clear();
        visit.clear();
        auto pimpl = std::make_shared<compiler_partition_impl_t>(
                backend_graph.get_engine_kind(),
                backend_graph.get_fpmath_mode(), pkind, pname);

        for (size_t i = 0; i < pairs.size(); ++i) {
            visit.insert(pairs[i]);
            fusion_ops_set.push_back(pairs[i]);
        }

        for (auto &cur_op : fusion_ops_set) {
            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                auto in_value = cur_op->get_input_value(j);
                if (!in_value->has_producer()
                        || !visit.count(&in_value->get_producer())) {
                    pimpl->add_input_tensor(in_value);
                }
            }

            for (size_t j = 0; j < cur_op->num_outputs(); ++j) {
                auto out_value = cur_op->get_output_value(j);
                // if out_value has no consumer
                // OR any of its consumers are not inside the pattern
                // it is output tensor
                bool is_output = out_value->get_consumers().empty();
                for (auto &consumer : out_value->get_consumers()) {
                    if (!visit.count(&consumer.get_op())) {
                        is_output = true;
                        break;
                    }
                }
                if (is_output) { pimpl->add_output_tensor(out_value); }
            }
        }

        // transfer the matched op's ownership from graph to partition
        for (size_t i = 0; i < pairs.size(); ++i) {
            pimpl->add_op(pairs[i]->shared_from_this());
            // claim the op belong to the partition
            pairs[i]->set_partition(pimpl.get());
        }
        backend_graph.add_partition(pimpl);
    }
}

} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
