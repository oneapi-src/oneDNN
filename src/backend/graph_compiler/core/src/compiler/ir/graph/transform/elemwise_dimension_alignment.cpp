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
#include <unordered_set>

#include "../fusible_op.hpp"
#include "../visitor.hpp"
#include <compiler/ir/graph/fusible_op.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/ternary_elemwise.hpp>

namespace sc {

static void infer_aligned_shape(const logical_tensor_t &a,
        const logical_tensor_t &b, const std::vector<int> &plain_bc_axis,
        sc_dims &aligned_shape, std::vector<int> &aligned_axis,
        sc_data_format_t &new_format) {
    assert(a.get_plain_dims().size() > b.get_plain_dims().size());
    if (plain_bc_axis.size() != b.get_plain_dims().size()
            || plain_bc_axis == std::vector<int> {-1}) {
        aligned_shape = sc_dims {};
        aligned_axis = {};
        new_format = sc_data_format_t();
        return;
    }

    int dim_difference = a.get_plain_dims().size() - b.get_plain_dims().size();
    sc_data_format_t b_format = b.get_format();
    if (b_format.is_any()) {
        b_format = sc_data_format_t::get_plain_by_dims(
                b.get_plain_dims().size());
    }
    // b's new shape --> extend to a's number of dimension
    aligned_shape.resize(a.get_plain_dims().size(), 1);
    // start infer extended shape
    for (size_t i = 0; i < plain_bc_axis.size(); ++i) {
        aligned_shape[plain_bc_axis[i]] = b.get_plain_dims()[i];
    }
    for (size_t i = 0; i < a.get_plain_dims().size(); ++i) {
        if (std::find(plain_bc_axis.begin(), plain_bc_axis.end(), (int)i)
                == plain_bc_axis.end()) {
            aligned_axis.push_back(i);
        }
    }
    // start infer extended format
    // the logic below wish to let blocking shape having extended dims
    // as its leading dims (having limitation on batch format)
    // e.g.
    // plain_shape: [3, 5] --> [1, 3, 1, 5]
    // blocking_shape: [5, 3] --> [1, 1, 5, 3]
    // format: BA --> xxDB --> ACDB
    // plain_shape: [3, 4, 5] --> [3, 1, 4, 5]
    // blocking_shape: [3, 5, 4] --> [3, 1, 5, 4]
    // format: X_BA --> X_xCB --> X_ACB
    std::vector<int> storage_args(
            b_format.format_code_.ndims() + dim_difference, -1);
    int batch_dim
            = b.get_plain_dims().size() - b_format.format_code_.norig_dims();
    std::unordered_set<int> axis;
    for (int i = 0; i < b_format.format_code_.ndims(); ++i) {
        int original_axis = b_format.format_code_.get(i);
        storage_args[i + dim_difference]
                = plain_bc_axis[original_axis + batch_dim] - batch_dim;
        axis.insert(storage_args[i + dim_difference]);
    }

    int sequential_fill_up = 0;
    for (size_t i = 0; i < storage_args.size(); ++i) {
        if (storage_args[i] == -1) {
            while (axis.find(sequential_fill_up) != axis.end()) {
                sequential_fill_up++;
            }
            storage_args[i] = sequential_fill_up++;
        }
    }
    new_format = sc_data_format_t(storage_args, b_format.blocks_);
}

/*
(x,y,z,w)[v0] [v1](y,w)  <-- shorter side
           \   /
            add
             |
   (x,y,z,w)[v3]
===============
               [v1](y,w)  <-- shorter side
                |
            tensor_view
                |
(x,y,z,w)[v0] [v4](1,y,1,w)
           \   /
            add
             |
   (x,y,z,w)[v3]
*/
void elemwise_dimension_alignment(sc_graph_t &graph, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto binary_node = node->dyn_cast<binary_elementwise_op_t>()) {
            COMPILE_ASSERT(binary_node->info_.inputs_.size() == 2,
                    "Wrong number of inputs for binary_elementwise_op");
            const auto &lhs = binary_node->info_.inputs_[0]->details_;
            const auto &rhs = binary_node->info_.inputs_[1]->details_;
            sc_dims shape;
            std::vector<int> aligned_axis;
            sc_data_format_t format;
            if (lhs.get_plain_dims().size() < rhs.get_plain_dims().size()) {
                infer_aligned_shape(rhs, lhs, binary_node->get_plain_bc_axis(),
                        shape, aligned_axis, format);
                if (!shape.empty()) {
                    // insert tensor view
                    auto ret = graph.make("tensor_view",
                            {binary_node->info_.inputs_[0]}, {},
                            {{"shape", shape}, {"format", format},
                                    {"expand_dim", aligned_axis}});
                    node->replace_input(0, ret->get_outputs()[0]);
                }
            } else if (lhs.get_plain_dims().size()
                    > rhs.get_plain_dims().size()) {
                infer_aligned_shape(lhs, rhs, binary_node->get_plain_bc_axis(),
                        shape, aligned_axis, format);
                if (!shape.empty()) {
                    // insert tensor view
                    auto ret = graph.make("tensor_view",
                            {binary_node->info_.inputs_[1]}, {},
                            {{"shape", shape}, {"format", format},
                                    {"expand_dim", aligned_axis}});
                    node->replace_input(1, ret->get_outputs()[0]);
                }
            }
        } else if (auto select_node = node->dyn_cast<select_op_t>()) {
            COMPILE_ASSERT(select_node->info_.inputs_.size() == 3,
                    "Wrong number of inputs for select_op");
            const auto &cond = select_node->info_.inputs_[0]->details_;
            const auto &els = select_node->info_.inputs_[2]->details_;
            sc_dims shape;
            std::vector<int> aligned_axis;
            sc_data_format_t format;
            if (cond.get_plain_dims().size() < els.get_plain_dims().size()) {
                infer_aligned_shape(els, cond, select_node->get_plain_bc_axis(),
                        shape, aligned_axis, format);
                if (!shape.empty()) {
                    // insert tensor view
                    auto ret = graph.make("tensor_view",
                            {select_node->info_.inputs_[0]}, {},
                            {{"shape", shape}, {"format", format},
                                    {"expand_dim", true}});
                    node->replace_input(0, ret->get_outputs()[0]);
                }
            } else if (cond.get_plain_dims().size()
                    > els.get_plain_dims().size()) {
                infer_aligned_shape(cond, els, select_node->get_plain_bc_axis(),
                        shape, aligned_axis, format);
                if (!shape.empty()) {
                    // insert tensor view
                    auto ret = graph.make("tensor_view",
                            {select_node->info_.inputs_[2]}, {},
                            {{"shape", shape}, {"format", format},
                                    {"expand_dim", true}});
                    node->replace_input(2, ret->get_outputs()[0]);
                }
            }
        }
    });
    graph.reset_op_ids();
}
} // namespace sc
