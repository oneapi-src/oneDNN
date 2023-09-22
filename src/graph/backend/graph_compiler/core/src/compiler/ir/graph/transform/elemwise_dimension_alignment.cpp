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
#include <ops/fusible/broadcast.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static void infer_aligned_shape(const logical_tensor_t &a,
        const logical_tensor_t &b, const std::vector<int> &plain_bc_axis,
        sc_dims &aligned_shape, std::vector<int> &aligned_axis,
        sc_data_format_t &new_format) {
    assert(a.get_plain_dims().size() > b.get_plain_dims().size());
    // skip the case where b's shape == {1}
    if (plain_bc_axis == std::vector<int> {-1}) {
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
    // get dimension binding relation of b to plain_bc_axis index
    std::unordered_map<int, int> bc_axis_bind;
    int bc_axis_idx = 0;
    for (size_t i = 0; i < b.get_plain_dims().size(); ++i) {
        if (b.get_plain_dims()[i] == 1
                && a.get_plain_dims()[plain_bc_axis[bc_axis_idx]] != 1) {
            continue;
        } else if (b.get_plain_dims()[i] != 1
                && a.get_plain_dims()[plain_bc_axis[bc_axis_idx]] == 1) {
            COMPILE_ASSERT(0, "Invalid bc_axis found for broadcastable ops.");
        } else {
            COMPILE_ASSERT(is_dynamic_dim(b.get_plain_dims()[i])
                            || is_dynamic_dim(
                                    a.get_plain_dims()
                                            [plain_bc_axis[bc_axis_idx]])
                            || b.get_plain_dims()[i]
                                    == a.get_plain_dims()
                                               [plain_bc_axis[bc_axis_idx]],
                    "Invalid bc_axis found for broadcastable ops.");
            bc_axis_bind[i] = bc_axis_idx;
            bc_axis_idx++;
            if (bc_axis_idx >= static_cast<int>(plain_bc_axis.size())) {
                break;
            }
        }
    }
    COMPILE_ASSERT(bc_axis_idx == static_cast<int>(plain_bc_axis.size()),
            "Invalid bc_axis found for broadcastable ops.");
    // b's new shape --> extend to a's number of dimension
    aligned_shape.resize(a.get_plain_dims().size(), 1);
    // start infer extended shape
    for (const auto &bind : bc_axis_bind) {
        aligned_shape[plain_bc_axis[bind.second]]
                = b.get_plain_dims()[bind.first];
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
    // plain_shape: [3, 1, 5] --> [1, 3, 1, 5] ({1, 3})
    // format: ABC --> xBxD --> ABCD
    // format: CAB ([5, 3, 1])--> xDBx --> ADBC --> [1, 5, 3, 1]
    std::vector<int> storage_args(
            b_format.format_code_.ndims() + dim_difference, -1);
    std::unordered_set<int> axis;
    for (int i = 0; i < b_format.format_code_.ndims(); ++i) {
        int original_axis = b_format.format_code_.get(i);
        // original_axis is the axis of the b's plain_dims
        // if plain_dims[original_axis]
        // the issue is we don't know whether original_axis is
        if (bc_axis_bind.find(original_axis) != bc_axis_bind.end()) {
            storage_args[i + dim_difference]
                    = plain_bc_axis[bc_axis_bind[original_axis]];
            axis.insert(storage_args[i + dim_difference]);
        } else {
            COMPILE_ASSERT(b.get_plain_dims()[original_axis] == 1,
                    "Axis not found in bc_axis_bind map's corresponding dim "
                    "must be 1.");
        }
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
        if (auto may_broadcast_node
                = node->dyn_cast<op_traits::may_broadcast_t>()) {
            COMPILE_ASSERT(
                    !may_broadcast_node->get_non_broadcast_input_index(false)
                             .empty(),
                    "elemwise_dimension_alignment requires the broadcast-able "
                    "op to have at least 1 non-broadcast input.");
            for (size_t i = 0; i < node->get_inputs().size(); ++i) {
                auto cur_in_lt = node->get_inputs()[i]->details_;
                auto cur_out_lt = node->get_outputs()[0]->details_;
                if (cur_in_lt.get_plain_dims().size()
                        < cur_out_lt.get_plain_dims().size()) {
                    const auto &plain_bc_axis
                            = may_broadcast_node->get_plain_bc_axis()[i];
                    sc_dims shape;
                    std::vector<int> aligned_axis;
                    sc_data_format_t format;
                    infer_aligned_shape(cur_out_lt, cur_in_lt, plain_bc_axis,
                            shape, aligned_axis, format);
                    if (!shape.empty()) {
                        // insert tensor view
                        auto ret = graph.make("tensor_view",
                                {node->info_.inputs_[i]}, {},
                                {{"shape", shape}, {"format", format},
                                        {"expand_dim", aligned_axis}});
                        node->replace_input(i, ret->get_outputs()[0]);
                    }
                }
            }
        } else if (auto broadcast_op = node->dyn_cast<broadcast_op_t>()) {
            auto cur_in_lt = node->get_inputs()[0]->details_;
            auto cur_out_lt = node->get_outputs()[0]->details_;
            if (cur_in_lt.get_plain_dims().size()
                    < cur_out_lt.get_plain_dims().size()) {
                const auto &plain_bc_axis = broadcast_op->get_plain_bc_axis();
                sc_dims shape;
                std::vector<int> aligned_axis;
                sc_data_format_t format;
                infer_aligned_shape(cur_out_lt, cur_in_lt, plain_bc_axis, shape,
                        aligned_axis, format);
                if (!shape.empty()) {
                    // insert tensor view
                    auto ret = graph.make("tensor_view",
                            {node->info_.inputs_[0]}, {},
                            {{"shape", shape}, {"format", format},
                                    {"expand_dim", aligned_axis}});
                    node->replace_input(0, ret->get_outputs()[0]);
                }
            }
        }
    });
    graph.reset_op_ids();
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
