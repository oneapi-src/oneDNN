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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "anchor_loop_generator.hpp"
#include "fusible_op_utils.hpp"
#include "fusion_anchor.hpp"
#include "mixed_partition.hpp"
#include "utils.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/cpu/local_tensor_lower.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/pooling.hpp>
#include <ops/fusible/reduce.hpp>
#include <runtime/config.hpp>
#include <util/utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.outer_loop_gen);

for_loop get_next_inner_loop(const for_loop &cur_loop) {
    if (cur_loop->body_.isa<for_loop>()) {
        return cur_loop->body_.checked_as<for_loop>();
    } else if (cur_loop->body_.isa<stmts>()
            && cur_loop->body_.checked_as<stmts>()->seq_.size() == 1
            && cur_loop->body_.checked_as<stmts>()->seq_[0].isa<for_loop>()) {
        return cur_loop->body_.checked_as<stmts>()
                ->seq_[0]
                .checked_as<for_loop>();
    }
    return for_loop();
}

static bool axis_can_be_sort(sc_graph_t &graph) {
    return is_optimized_sub_graph(graph)
            && std::all_of(graph.ops_.begin(), graph.ops_.end(),
                    [](const sc_op_ptr &op) {
                        return (!op->isa<reorder_op_t>()
                                       && !op->isa<tensor_view_op_t>())
                                || op->attrs_.get_or_else(
                                        op_attr_key::no_fuse, false);
                    });
}

typedef std::vector<int> (*loop_sort_rule_func)(const context_ptr &,
        const std::vector<int> &, sc_graph_t &, const graph_tensor_ptr &);

/**
 * Move loop axis of reduce axis to inner.
 *
 * E.g. loop axis is {0, 1, 2, 3}, rd_axis is {1, 2}, after func, we get loop
 * axis {0, 3, 1, 2}
 * */
static std::vector<int> move_reduce_axis_to_inner(const context_ptr &ctx,
        const std::vector<int> &in_axis, sc_graph_t &graph,
        const graph_tensor_ptr &base_gt) {
    if (graph.is_dynamic() || !axis_can_be_sort(graph)) { return in_axis; }
    auto run_threads = runtime_config_t::get().get_num_threads();
    std::vector<int> out_axis(in_axis.begin(), in_axis.end());
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    bool can_move = true;
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        std::vector<int> reduce_axis;
        if (auto reduce_node = node->dyn_cast<reduce_op_t>()) {
            reduce_axis = reduce_node->get_rd_axis();
        } else if (auto reduce_node = node->dyn_cast<reduce_impl_op_t>()) {
            reduce_axis = reduce_node->get_rd_axis();
        } else {
            return;
        }
        // if using single core and the reduce op is the only op in this
        // partition, don't reorder the loops
        // we move the reduce axis for parallelism with a cost of losing
        // cache locality. For single core configuration, we don't do this
        // moving
        if (run_threads == 1 && graph.ops_.size() == 3) {
            // 3 ops for input, reduce and output
            can_move = false;
            return;
        }
        std::sort(reduce_axis.begin(), reduce_axis.end());
        if (reduce_axis.back() >= static_cast<int>(in_axis.size())) {
            can_move = false;
            return;
        }
        for (auto raxis : reduce_axis) {
            auto rend = std::remove(out_axis.begin(), out_axis.end(), raxis);
            assert(rend + 1 == out_axis.end());
            *rend = raxis;
        }
    });
    return can_move ? out_axis : in_axis;
}

/**
 * Move last channel axis to outer if pooling format is NXC, in order to acheive
 * better loop parallelism
 * */
static std::vector<int> move_pooling_axis_to_outer(const context_ptr &ctx,
        const std::vector<int> &in_axis, sc_graph_t &graph,
        const graph_tensor_ptr &base_gt) {
    if (graph.is_dynamic() && !axis_can_be_sort(graph)) { return in_axis; }

    if (!is_optimized_sub_graph(graph)) return in_axis;
    auto run_threads = runtime_config_t::get().get_num_threads();
    // auto skip
    if (run_threads == 1) return in_axis;
    std::vector<int> out_axis(in_axis.begin(), in_axis.end());
    bool use_vectorized = false;
    if (std::any_of(graph.ops_.begin(), graph.ops_.end(),
                [&ctx, &run_threads, &use_vectorized](const sc_op_ptr &node) {
                    if (!node->isa<pooling_op_t>()) return false;
                    auto &detail = node->get_inputs()[0]->details_;
                    // skip if not a plain format
                    if (detail.get_format().is_blocking()) return false;
                    auto shape = detail.get_blocking_dims();
                    auto pool = node->dyn_cast<pooling_op_t>();
                    auto channel_axis = pool->get_channel_axis();
                    COMPILE_ASSERT(channel_axis.size() == 1,
                            "plain format is expected")
                    // skip if not channel last
                    if (pool->get_channel_axis()[0]
                            != (static_cast<int>(shape.size()) - 1))
                        return false;
                    auto &last_dim = shape.back();
                    auto vector_lanes
                            = vectorize_step(ctx, detail.dtype_.type_code_);
                    use_vectorized = (last_dim / vector_lanes
                            && last_dim % vector_lanes == 0);
                    if (use_vectorized) {
                        shape.back() = std::max(
                                (int64_t)1, shape.back() / vector_lanes);
                    }
                    auto pool_axis = pool->get_real_pooling_axis();
                    int parallel_num = 1;
                    // calculate possible parallelism
                    for (int i = 0; i < static_cast<int>(shape.size()); i++) {
                        if (std::find(pool_axis.begin(), pool_axis.end(), i)
                                != pool_axis.end())
                            continue;
                        parallel_num *= shape[i];
                    }
                    // check parallel_num
                    return parallel_num >= run_threads;
                })) {
        // move last channel axis to outer
        auto &last_channel_ax = out_axis.back();
        out_axis.insert(out_axis.begin() + 1, last_channel_ax);
        if (!use_vectorized) out_axis.pop_back();
        return out_axis;
    } else {
        return in_axis;
    }
}

/**
 * Satisfy continuous access of input tensor include vectorization on last axis
 * and ensure size of each load is more than cache line.
 *
 * E.g. loop axis = {1, 3, 4, 0, 2}
 *
 * IF input tensor(origin shape) is f32(32, 4, 16, 8, 16), last axis is 16
 * which fills up a cache line, after func we get loop axis = {1, 3, 0, 2, 4}.
 * IF input tensor(origin shape) is f32{32, 4, 16, 8, 8}, after func we get loop
 * axis = {1, 0, 2, 3, 4}
 * */
static std::vector<int> continuous_access_satisfaction(const context_ptr &ctx,
        const std::vector<int> &in_axis, sc_graph_t &graph,
        const graph_tensor_ptr &base_gt) {
    if (!axis_can_be_sort(graph)) { return in_axis; }
    auto base_dims = base_gt->details_.get_blocking_dims_expr(graph);
    assert(in_axis.size() == base_dims.size());
    constexpr int cache_line_size = 64;
    int fill_up_dim = static_cast<int>(base_dims.size()) - 1;
    int dtype_size = utils::get_sizeof_type(base_gt->details_.dtype_);
    int cur_load_size = base_dims[fill_up_dim].isa<constant>()
            ? get_expr_as_int(base_dims[fill_up_dim])
            : 1;
    while (fill_up_dim > 0 && cur_load_size * dtype_size < cache_line_size) {
        fill_up_dim--;
        cur_load_size = cur_load_size
                * (base_dims[fill_up_dim].isa<constant>()
                                ? get_expr_as_int(base_dims[fill_up_dim])
                                : 1);
    }
    // input tensor is too small that can not fill up a cache line.
    // No need to change loop axis.
    if (fill_up_dim == 0) {
        if (!graph.is_dynamic()) {
            return in_axis;
        } else {
            fill_up_dim = static_cast<int>(base_dims.size()) - 1;
        }
    }
    std::vector<int> out_axis(in_axis.begin(), in_axis.end());
    for (int i = fill_up_dim; i < static_cast<int>(base_dims.size()); i++) {
        auto rend = std::remove(out_axis.begin(), out_axis.end(), i);
        *rend = i;
    }
    return out_axis;
}

static std::vector<loop_sort_rule_func> loop_sort_rules
        = {move_reduce_axis_to_inner, continuous_access_satisfaction,
                move_pooling_axis_to_outer};

anchor_loop_generator_t::anchor_loop_generator_t(
        const graph_tensor_ptr &base_gt)
    : body_generator_base_t(nullptr, {}, {}), base_gt_(base_gt) {}

sc_graph_t &get_owner_graph_from_gt(const graph_tensor_ptr &gt) {
    // search producer firstly
    auto op = gt->producer_owner_;
    if (!op->owner_graph_) {
        // if not found, search in users
        for (auto &user : gt->uses_) {
            op = user.second.get();
            if (op->owner_graph_) { break; }
        }
    }
    COMPILE_ASSERT(op->owner_graph_, "No owner graph found, please check")
    return op->get_owner_graph();
}

bool anchor_loop_generator_t::create_outer_loop_anchor(
        fusion_anchor_mgr_t *fmgr, const context_ptr &ctx) const {
    COMPILE_ASSERT(fmgr, "fusion anchor mgr should not be null")
    auto &g = get_owner_graph_from_gt(base_gt_);
    // query binding axis
    query_binding_axis(g);
    auto base_dims = base_gt_->details_.get_blocking_dims_expr(g);
    auto numdims = base_dims.size();
    assert(numdims > 0);
    std::vector<expr> loop_vars;
    slice_range cur_tsr_slice;
    std::vector<int> loop_axis;
    loop_vars.reserve(numdims);
    cur_tsr_slice.reserve(numdims);
    loop_axis.reserve(numdims);

    auto bld = builder::get_current_builder();
    // will create numdims loop vars but uses numdims - 1 because user may sort
    // loop axis
    for (size_t i = 0; i < numdims; i++) {
        bld->push_scope();
        loop_vars.emplace_back(builder::make_var(
                datatypes::index, std::string("__itr_") + std::to_string(i)));
        // outer loops should have tensor slice of length=1
        cur_tsr_slice.emplace_back(std::make_pair(loop_vars.back(), expr(1)));
        loop_axis.push_back(static_cast<int>(i));
    }
    // sort loop axis with rules
    for (auto &sort_rule : loop_sort_rules) {
        loop_axis = sort_rule(ctx, loop_axis, g, base_gt_);
    }

    // generate anchors from inner to outer
    if (numdims > 1) {
        // get ax for last dim
        int max_ax = *std::max_element(loop_axis.begin(), loop_axis.end());
        // if duplicated ax found, it means that loop generator needs to split
        // loop for last dim vectorization
        bool need_split_loop
                = (std::count(loop_axis.begin(), loop_axis.end(), max_ax) == 2);
        auto lanes = vectorize_step(ctx, base_gt_->details_.dtype_.type_code_);
        // set last dim slice range for lanes
        if (need_split_loop) {
            cur_tsr_slice[max_ax]
                    = std::make_pair(loop_vars[max_ax] * lanes, lanes);
        }
        for (size_t i = 0; i < numdims - 1; i++) {
            // loop num is current dimension index
            auto loop_num = loop_axis[numdims - i - 1];
            // upper loop num
            auto upper_loop_num = loop_axis[numdims - i - 2];
            // set full tensor range for loop_num dimension
            cur_tsr_slice[loop_num] = std::make_pair(0, base_dims[loop_num]);
            fmgr->create_fusion_anchor(
                    slice_map {{base_gt_.get(), {cur_tsr_slice}}});
            auto body = bld->pop_scope();
            auto loop = bld->push_for_loop(loop_vars[upper_loop_num], 0,
                    (need_split_loop && upper_loop_num == max_ax)
                            ? do_cast_and_fold(base_dims[max_ax] / lanes)
                            : base_dims[upper_loop_num],
                    1, body, true, for_type::NORMAL);
            // bind outer loops with axis hint
            bind_loop_axis(base_gt_, loop, upper_loop_num, true);
        }
    } else {
        COMPILE_ASSERT(numdims == 1, "only 1 dims is expected")
        if (base_dims[0].isa<constant>()) {
            auto only_dim
                    = get_const_as_int(base_dims[0].static_as<constant>());
            auto lanes
                    = vectorize_step(ctx, base_gt_->details_.dtype_.type_code_);
            if ((only_dim % lanes == 0) && (only_dim > lanes)) {
                cur_tsr_slice[0].second = expr((int)lanes);
                fmgr->create_fusion_anchor(
                        slice_map {{base_gt_.get(), {cur_tsr_slice}}});
                auto body = bld->pop_scope();
                auto loop = bld->push_for_loop(loop_vars[0], 0, base_dims[0],
                        expr((int)lanes), body, true, for_type::NORMAL);
                // bind outer loops with axis hint
                bind_loop_axis(base_gt_, loop, 0, true);
            }
        }
    }
    // create outer-most anchor
    cur_tsr_slice[loop_axis[0]] = std::make_pair(0, base_dims[loop_axis[0]]);
    fmgr->create_fusion_anchor(slice_map {{base_gt_.get(), {cur_tsr_slice}}});
    return true;
}

bool anchor_loop_generator_t::create_inner_loop_anchor(
        fusion_anchor_mgr_t *fmgr,
        const fusion_anchor_ptr &parent_fanchor) const {
    COMPILE_ASSERT(parent_fanchor, "parent anchor must be set")
    // do not support multi-slice
    if (parent_fanchor->fsmap_.get(base_gt_).size() != 1) return false;
    auto bld = builder::get_current_builder();
    std::vector<expr> loop_vars;
    std::vector<int> inner_anchor_axis;
    slice_range inner_slice;
    // will create numdims loop vars but uses numdims - 1 because user may sort
    // loop axis
    auto &range = parent_fanchor->fsmap_.get(base_gt_)[0];
    auto is_valid_constant_range = [](const std::pair<expr, expr> &range_i) {
        return (range_i.second.isa<constant_c>()
                && get_expr_as_int(range_i.second) > 1);
    };
    size_t valid_range_end = 0;
    for (int64_t i = range.size() - 1; i >= 0; --i) {
        if (is_valid_constant_range(range[i])) {
            valid_range_end = i;
            break;
        }
    }
    for (size_t i = 0; i < range.size(); i++) {
        if (is_valid_constant_range(range[i]) && (i < valid_range_end)) {
            bld->push_scope();
            loop_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("__inner_itr_") + std::to_string(i)));
            inner_slice.emplace_back(std::make_pair(loop_vars.back(), expr(1)));
            inner_anchor_axis.emplace_back(i);
        } else {
            inner_slice.emplace_back(std::make_pair(expr(0), range[i].second));
        }
    }

    auto inner_anchor_num = inner_anchor_axis.size();
    if (!inner_anchor_num) return false;
    // generate anchors from inner to outer
    for (int64_t i = static_cast<int64_t>(inner_anchor_num) - 1; i >= 0; i--) {
        auto loop_num = inner_anchor_axis[i];
        fmgr->create_fusion_anchor(
                slice_map {{base_gt_.get(), {inner_slice}}}, parent_fanchor);
        auto body = bld->pop_scope();
        auto loop = bld->push_for_loop(loop_vars[i], 0, range[loop_num].second,
                1, body, true, for_type::NORMAL);
        if (i == 0) {
            loop->attr()[stmt_attr_key::merge_loop] = true;
            // bind outer loops with axis hint
            bind_loop_axis(base_gt_, loop, loop_num, true);
        }
        inner_slice[loop_num] = range[loop_num];
    }
    return true;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
