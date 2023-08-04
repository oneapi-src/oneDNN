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

#include "outer_loop_generator.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "fusible_op_utils.hpp"
#include "fusion_mgr.hpp"
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

static int64_t get_loop_range(const for_loop &loop) {
    if (!loop->iter_begin_.isa<constant>() || !loop->iter_end_.isa<constant>()
            || !loop->step_.isa<constant>()) {
        return dimensions::dynamic_any;
    }
    return (get_const_as_int(loop->iter_end_.checked_as<constant>())
                   - get_const_as_int(loop->iter_begin_.checked_as<constant>()))
            / get_const_as_int(loop->step_.checked_as<constant>());
}

static void fuse_outer_loops(for_loop outer_loop, bool is_dynamic = false) {
    assert(outer_loop.defined());
    const int max_fused_number = runtime_config_t::get().get_num_threads() * 10;
    for_loop cur_loop = std::move(outer_loop);
    std::vector<for_loop> loops;
    while (cur_loop.defined()) {
        loops.push_back(cur_loop);
        cur_loop = get_next_inner_loop(cur_loop);
    }
    if (!loop_can_be_fused(loops[0])) { return; }
    int64_t fused_number = get_loop_range(loops[0]);
    size_t end = loops.size() - 1;
    if (is_dynamic) {
        // todo: for vnni reorder
        if (loops.size() >= 4) { end = loops.size() - 2; }
        if (loops.back()->iter_end_.isa<constant>()
                && loops.back()->iter_begin_.isa<constant>()
                && (get_expr_as_int(loops.back()->iter_end_)
                           - get_expr_as_int(loops.back()->iter_begin_))
                        < 16) {
            end = 3;
        }
    }
    for (size_t i = 1; i < end; i++) {
        if (fused_number != dimensions::dynamic_any
                && fused_number >= max_fused_number) {
            break;
        }
        if (!loop_can_be_fused(loops[i])) { break; }
        if (fused_number != dimensions::dynamic_any) {
            fused_number = fused_number * get_loop_range(loops[i]);
        }
        loops[0]->fuse(loops[i]);
    }
}

outer_loop_generator_t::outer_loop_generator_t(
        size_t base_tsr_idx, bool use_output_mode)
    : body_generator_base_t(nullptr, {}, {})
    , base_tsr_idx_(base_tsr_idx)
    , use_output_mode_(use_output_mode) {}

typedef std::vector<int> (*loop_sort_rule_func)(const context_ptr &,
        const std::vector<int> &, sc_graph_t &, const tensor &);

/**
 * @param forced: If True, axis needs to be sorted forcedly to ensure
 * parallelism which means some op may break fusion. Default is false.
 * */
static bool axis_can_be_sort(sc_graph_t &graph, bool forced = false) {
    bool res = is_optimized_sub_graph(graph)
            || std::all_of(graph.ops_.begin(), graph.ops_.end(),
                    [&forced](const sc_op_ptr &op) {
                        if (op->isa<reorder_op_t>()
                                || op->isa<tensor_view_op_t>()) {
                            if (forced) {
                                op->attrs_.set(op_attr_key::no_fuse, true);
                            }
                            return false;
                        }
                        return true;
                    });
    // if forced, the given graph need repartition
    if (!res && forced) graph.attrs_["temp.need_repartition"] = true;
    return res;
}

/**
 * Currently, outloop generator can only generate output anchor based on input
 * shape, However some ops like reorder may have different shape between input
 * and output which may cause conflict. We need to enhance this in the future.
 * */
static bool detect_loop_conflict(fusion_manager *fmgr) {
    auto &graph = fmgr->get_graph();
    return std::any_of(
            graph.ops_.begin(), graph.ops_.end(), [&fmgr](const sc_op_ptr &op) {
                if (op->isa<reorder_op_t>()) {
                    if (op->get_inputs()[0]
                                    ->producer_owner_->dyn_cast<input_op>()
                            == fmgr->get_first_input()) {
                        op->attrs_.set(op_attr_key::no_fuse, true);
                        return true;
                    }
                }
                return false;
            });
}

/**
 * Move loop axis of reduce axis to inner.
 *
 * E.g. loop axis is {0, 1, 2, 3}, rd_axis is {1, 2}, after func, we get loop
 * axis {0, 3, 1, 2}
 * */
static std::vector<int> move_reduce_axis_to_inner(const context_ptr &ctx,
        const std::vector<int> &in_axis, sc_graph_t &graph, const tensor &tsr) {
    if (graph.is_dynamic() && !axis_can_be_sort(graph)) { return in_axis; }
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
        auto shape = node->get_inputs()[0]->details_.get_blocking_dims();
        int parallel_num = 1;
        for (int i = 0; i < *reduce_axis.begin(); i++) {
            parallel_num *= shape[i];
        }
        /* Due to loop order not only affect outer-loop parallelism,
         * but also inner-loop fusion, which will affect local buffer size(
         * sensitive to cache line size). Further, more performance data
         * maybe required and analyzed to decide which strategy shuold be
         * applied to achieve best performance*/

        // need check parallel_num
        if (!graph.is_dynamic()
                && !axis_can_be_sort(graph,
                        (parallel_num < run_threads)
                                && !node->attrs_.get_or_else(
                                        op_attr_key::bwise_fuse, false))) {
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
        const std::vector<int> &in_axis, sc_graph_t &graph, const tensor &tsr) {
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
        const std::vector<int> &in_axis, sc_graph_t &graph, const tensor &tsr) {
    assert(in_axis.size() == tsr->dims_.size());
    if (!axis_can_be_sort(graph)) { return in_axis; }
    constexpr int cache_line_size = 64;
    int fill_up_dim = static_cast<int>(tsr->dims_.size()) - 1;
    int dtype_size = utils::get_sizeof_type(tsr->elem_dtype_);
    int cur_load_size = tsr->dims_[fill_up_dim].isa<constant>()
            ? get_expr_as_int(tsr->dims_[fill_up_dim])
            : 1;
    while (fill_up_dim > 0 && cur_load_size * dtype_size < cache_line_size) {
        fill_up_dim--;
        cur_load_size = cur_load_size
                * (tsr->dims_[fill_up_dim].isa<constant>()
                                ? get_expr_as_int(tsr->dims_[fill_up_dim])
                                : 1);
    }
    // input tensor is too small that can not fill up a cache line.
    // No need to change loop axis.
    if (fill_up_dim == 0) {
        if (!graph.is_dynamic()) {
            return in_axis;
        } else {
            fill_up_dim = static_cast<int>(tsr->dims_.size()) - 1;
        }
    }
    std::vector<int> out_axis(in_axis.begin(), in_axis.end());
    for (int i = fill_up_dim; i < static_cast<int>(tsr->dims_.size()); i++) {
        auto rend = std::remove(out_axis.begin(), out_axis.end(), i);
        *rend = i;
    }
    return out_axis;
}

static std::vector<loop_sort_rule_func> loop_sort_rules
        = {move_reduce_axis_to_inner, continuous_access_satisfaction,
                move_pooling_axis_to_outer};
bool outer_loop_generator_t::generate(context_ptr ctx, const void *config,
        fusion_manager *fusion, const std::vector<expr> &inputs,
        const std::vector<expr> &outputs, std::vector<for_loop> &loops) const {
    if (!use_output_mode_) {
        COMPILE_ASSERT(inputs.size() > base_tsr_idx_,
                "Expecting at least " << base_tsr_idx_
                                + 1 << " input(s) for outer_loop_generator_t");
    } else {
        COMPILE_ASSERT(outputs.size() > base_tsr_idx_,
                "Expecting at least " << base_tsr_idx_
                                + 1 << " output(s) for outer_loop_generator_t");
    }
    // If loop conflict found, return.
    if (!is_optimized_sub_graph(fusion->get_graph())
            && detect_loop_conflict(fusion))
        return false;
    tensor base_tsr = (use_output_mode_ ? outputs[base_tsr_idx_]
                                        : inputs[base_tsr_idx_])
                              .as<tensor>();
    COMPILE_ASSERT(base_tsr.defined(), "Expecting a tensor");
    auto bld = builder::get_current_builder();
    auto numdims = base_tsr->dims_.size();
    auto lanes = vectorize_step(ctx, base_tsr->elem_dtype_.type_code_);
    assert(numdims > 0);
    std::vector<expr> loop_vars;
    slice_range cur_tsr_slice;
    std::vector<int> loop_axis;
    loop_vars.reserve(numdims);
    cur_tsr_slice.reserve(numdims);
    loop_axis.reserve(numdims);
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
    for (auto sort_rule : loop_sort_rules) {
        loop_axis = sort_rule(ctx, loop_axis, fusion->get_graph(), base_tsr);
        // check whether need to repartition
        if (fusion->get_graph().attrs_.get_or_else(
                    "temp.need_repartition", false)) {
            fusion->get_graph().attrs_.remove("temp.need_repartition");
            return false;
        }
    }

    // generate anchors from inner to outer
    if (numdims > 1) {
        // get ax for last dim
        int max_ax = *std::max_element(loop_axis.begin(), loop_axis.end());
        // if duplicated ax found, it means that loop generator needs to split
        // loop for last dim vectorization
        bool need_split_loop
                = (std::count(loop_axis.begin(), loop_axis.end(), max_ax) == 2);
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
            cur_tsr_slice[loop_num]
                    = std::make_pair(0, base_tsr->dims_[loop_num]);
            fusion->create_output_fusion_anchor(
                    {tensor_slice(base_tsr, slice_range(cur_tsr_slice))});
            auto body = bld->pop_scope();
            auto loop = bld->push_for_loop(loop_vars[upper_loop_num], 0,
                    (need_split_loop && upper_loop_num == max_ax)
                            ? do_cast_and_fold(base_tsr->dims_[max_ax] / lanes)
                            : base_tsr->dims_[upper_loop_num],
                    1, body, true, for_type::NORMAL);
            loops.emplace_back(loop.checked_as<for_loop>());
        }
    } else {
        COMPILE_ASSERT(numdims == 1, "only 1 dims is expected")
        if (base_tsr->dims_[0].isa<constant>()) {
            auto only_dim = get_const_as_int(
                    base_tsr->dims_[0].static_as<constant>());
            if ((only_dim % lanes == 0) && (only_dim > lanes)) {
                cur_tsr_slice[0].second = expr((int)lanes);
                fusion->create_output_fusion_anchor(
                        {tensor_slice(base_tsr, slice_range(cur_tsr_slice))});
                auto body = bld->pop_scope();
                auto loop = bld->push_for_loop(loop_vars[0], 0,
                        base_tsr->dims_[0], expr((int)lanes), body, true,
                        for_type::NORMAL);
                loops.emplace_back(loop.checked_as<for_loop>());
            }
        }
    }

    if (!loops.empty()) {
        bound_axis bd_axis(loops.size());
        std::transform(loop_axis.begin(), loop_axis.begin() + loops.size(),
                bd_axis.begin(),
                [](const int &ax) { return std::vector<int> {ax}; });
        loops[0]->attr()[stmt_attr_key::loop_axis_hint] = bd_axis;
    }

    cur_tsr_slice[loop_axis[0]]
            = std::make_pair(0, base_tsr->dims_[loop_axis[0]]);
    fusion->create_output_fusion_anchor(
            {tensor_slice(base_tsr, slice_range(cur_tsr_slice))});
    return true;
}

anchor_loop_generator_t::anchor_loop_generator_t(
        const graph_tensor_ptr &gt, const fuse_anchor_map_ptr &parent_fanchor)
    : body_generator_base_t(nullptr, {}, {})
    , gt_(gt)
    , parent_fanchor_(parent_fanchor) {}

std::vector<fuse_anchor_map_ptr>
anchor_loop_generator_t::create_inner_anchor() {
    COMPILE_ASSERT(parent_fanchor_, "parent anchor could not be null")
    if (parent_fanchor_->fsmap_.get(gt_).size() != 1) return {};
    auto bld = builder::get_current_builder();
    std::vector<expr> loop_vars;
    std::vector<int> inner_anchor_axis;
    slice_range inner_slice;
    // will create numdims loop vars but uses numdims - 1 because user may sort
    // loop axis
    auto range = parent_fanchor_->fsmap_.get(gt_)[0];
    size_t valid_range_end = 0;
    for (int64_t i = range.size() - 1; i >= 0; --i) {
        if (range[i].second.isa<constant_c>()
                && get_expr_as_int(range[i].second) > 1) {
            valid_range_end = i;
            break;
        }
    }
    for (size_t i = 0; i < (range.size() - 1); i++) {
        if (range[i].second.isa<constant_c>()
                && get_expr_as_int(range[i].second) > 1
                && (i < valid_range_end)) {
            bld->push_scope();
            loop_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("__inner_itr_") + std::to_string(i)));
            inner_slice.emplace_back(
                    std::make_pair(range[i].first + loop_vars.back(), expr(1)));
            inner_anchor_axis.emplace_back(i);
        } else {
            inner_slice.emplace_back(range[i]);
        }
    }
    inner_slice.emplace_back(range.back());

    auto inner_anchor_num = inner_anchor_axis.size();
    if (!inner_anchor_num) return {};
    std::vector<fuse_anchor_map_ptr> fanchor_map;
    // generate anchors from inner to outer
    for (int64_t i = static_cast<int64_t>(inner_anchor_num) - 1; i >= 0; i--) {
        auto loop_num = inner_anchor_axis[i];
        fslice_map fsmap;
        fsmap.get(gt_) = slice_range_list {inner_slice};
        auto s = bld->push_anchor();
        fanchor_map.emplace_back(
                std::make_shared<fuse_anchor_map_t>(s, fsmap, parent_fanchor_));
        auto body = bld->pop_scope();
        auto loop = bld->push_for_loop(loop_vars[i], 0, range[loop_num].second,
                1, body, true, for_type::NORMAL);
        if (i == 0) loop->attr()[stmt_attr_key::merge_loop] = true;
        inner_slice[loop_num] = range[loop_num];
    }
    return fanchor_map;
}

static void schedule_outer_anchor_loops(
        const stmt &body, const for_loop &main_loop) {
    ir_comparer cmper;
    if (body.isa<stmts>()) {
        for (auto &st : body.checked_as<stmts>()->seq_) {
            if (st.isa<for_loop>()) {
                auto cur_for = st.checked_as<for_loop>();
                // if loop is not from outmost anchor
                if (main_loop.defined() && cur_for->equals(main_loop, cmper)) {
                    continue;
                }
                cur_for->kind_ = for_type::PARALLEL;
                // if outmost anchor step is larger than 1
                if (cur_for->step_.isa<constant>()
                        && get_expr_as_int(cur_for->step_) > 1)
                    continue;
                auto body = cur_for->body_;
                assert(!body.defined() || body.isa<for_loop>()
                        || body.isa<stmts>());
                // next body has for loop at first
                for_loop next_for;
                while (body.defined()
                        && (body.isa<for_loop>()
                                || (!body.static_as<stmts>()->seq_.empty()
                                        && body.static_as<stmts>()
                                                   ->seq_[0]
                                                   .isa<for_loop>()))) {
                    if (body.isa<for_loop>()) {
                        next_for = body.static_as<for_loop>();
                    } else {
                        next_for = body.static_as<stmts>()
                                           ->seq_[0]
                                           .checked_as<for_loop>();
                    }
                    if (next_for->step_.isa<constant>()
                            && get_expr_as_int(next_for->step_) == 1) {
                        cur_for->fuse(next_for);
                    } else {
                        break;
                    }
                    body = next_for->body_;
                }
                // for input shape with only one dimension.
                if (!main_loop.defined() && !next_for.defined()) {
                    cur_for->kind_ = for_type::NORMAL;
                }
            }
        }
    }
}

void outer_loop_generator_t::schedule_loops(context_ptr ctx, const void *config,
        stmt body, std::vector<for_loop> &fors) const {
    for_loop l0;
    if (!fors.empty()) {
        l0 = fors.back();
        l0->kind_ = for_type::PARALLEL;
        // satisfies the vectorization when there is a reduce op inside when
        // dynamic.
        size_t offset = 0;
        auto is_dynamic = body->attr().get<bool>("temp.is_dynamic");
        if (is_dynamic) {
            auto *fusion = body->attr().get<fusion_manager *>(
                    "temp.fusion_manager_pointer");
            for (auto &op : fusion->get_graph().ops_) {
                if (op->isa<reduce_op_t>()) {
                    auto rd_axis = op->stc_cast<reduce_op_t>()->get_rd_axis();
                    if (!std::all_of(rd_axis.begin(), rd_axis.end(),
                                [&fors](const int &x) {
                                    return x != static_cast<int>(fors.size());
                                })) {
                        offset = std::max(offset, rd_axis.size());
                    }
                }
            }
        }
        int num_threads = runtime_config_t::get().get_num_threads();
        if (num_threads > 1 && (fors.size() - offset > 0)) {
            for (auto itr = fors.rbegin() + 1; itr != fors.rend() - offset;
                    ++itr) {
                l0->fuse(*itr);
            }
        }
    }
    // For anchor outside fors
    schedule_outer_anchor_loops(body, l0);
}

bool top_level_anchor_generator_t::generate(context_ptr ctx, const void *config,
        fusion_manager *fusion, const std::vector<expr> &inputs,
        const std::vector<expr> &outputs, std::vector<for_loop> &loops) const {
    slice_range ranges;
    COMPILE_ASSERT(!inputs.empty(),
            "Expecting at least 1 input for top_level_anchor_generator_t");
    tensor in_tsr = inputs[0].as<tensor>();
    COMPILE_ASSERT(in_tsr.defined(), "Expecting a tensor");
    for (auto &dim : in_tsr->dims_) {
        ranges.emplace_back(std::make_pair(0, dim));
    }
    fusion->create_output_fusion_anchor(
            {tensor_slice(inputs[0], std::move(ranges))});
    return true;
}

void top_level_anchor_generator_t::schedule_loops(context_ptr ctx,
        const void *config, stmt body, std::vector<for_loop> &fors) const {
    bool is_dynamic = body->attr().get<bool>("temp.is_dynamic");
    if (body.isa<stmts>()) {
        auto body_seqs = body.checked_as<stmts>()->seq_;
        for (size_t i = 0; i < body_seqs.size(); i++) {
            if (body_seqs[i].isa<for_loop>()) {
                body_seqs[i].static_as<for_loop>()->kind_ = for_type::PARALLEL;
                fuse_outer_loops(
                        body_seqs[i].checked_as<for_loop>(), is_dynamic);
            }
        }
    } else if (body.isa<for_loop>()) {
        body.checked_as<for_loop>()->kind_ = for_type::PARALLEL;
        fuse_outer_loops(body.checked_as<for_loop>(), is_dynamic);
    }
}

ir_module_ptr try_lower_fusion_manager(const context_ptr &ctx,
        outer_loop_generator_t *gen, sc_op *op, fusion_manager *fmgr,
        bool check_parallel, bool just_check,
        std::vector<sc_op_ptr> &out_failed) {
    auto modu = std::make_shared<ir_module_t>(ctx);

    std::vector<expr> ins;
    // real_outs are the output tensors in the function arguments
    std::vector<expr> real_outs;
    auto func = graph::create_func_decl_for_op(op, ins, real_outs);
    // finds if an output can be computed in-place on an "input" of the fusion
    // graph
    auto inplacemap = fmgr->query_inplace();
    // todo: check inplace
    auto main_op_input_size = op->get_inputs().size();
    COMPILE_ASSERT(!op->get_inputs().empty(), "Expecting at least 1 input");
    assert(op->get_inputs().size() == (size_t)fmgr->get_input_op_count());
    assert(op->get_outputs().size() == (size_t)fmgr->get_output_op_count());

    // =======================
    // Start of building function body
    // =======================
    builder::ir_builder_t bld;
    bld.push_scope();

    std::vector<for_loop> loops;
    bool status = gen->generate(ctx, nullptr, fmgr, ins, real_outs, loops);
    // if status is false, also return and add failed ops
    if (!status) {
        for (auto &op : fmgr->get_graph().ops_) {
            if (op->attrs_.get_or_else(op_attr_key::no_fuse, false))
                out_failed.emplace_back(op);
        }
        fmgr->clear_anchor();
        return nullptr;
    }
    bld.push_returns(true);
    auto body = bld.pop_scope();

    // =======================
    // End of building function body
    // =======================
    // the additional arguments for fmgr, according base_tsr_idx_ of gen
    auto base_tsr_idx = gen->get_base_tsr_idx();
    std::vector<expr> additional_args;
    for (size_t i = 0; i < ins.size(); i++) {
        if (i == base_tsr_idx) continue;
        additional_args.emplace_back(ins[i]);
    }
    if (!just_check) { fmgr->transform_graph(ctx, false); }
    fuse_state_t fstate;
    out_failed = fmgr->prepare_and_check(ctx, fstate);
    if (!out_failed.empty()) {
        fmgr->clear_anchor();
        return nullptr;
    }
    if (just_check) {
        fmgr->clear_anchor();
        return nullptr;
    }
    fmgr->commit(modu, fstate, real_outs, additional_args);

    func->body_ = std::move(body);
    // To get inside ops during schedule loops
    func->body_->attr().set("temp.is_dynamic", fmgr->get_graph().is_dynamic());
    func->body_->attr().set("temp.fusion_manager_pointer", fmgr);
    gen->schedule_loops(ctx, nullptr, func->body_, loops);
    // check that if we are using the outer most anchor. If so, print a warning.

    if (check_parallel && !loops.empty()) {
        auto l0 = loops.back();
        auto &seq = func->body_.checked_as<stmts>()->seq_;
        for (size_t idx = 0; idx < seq.size(); idx++) {
            if (seq[idx].ptr_same(l0)) {
                if (idx != seq.size() - 2) {
                    if (l0->kind_ == for_type::NORMAL) {
                        SC_MODULE_WARN
                                << "Using non-parallel generator. This may "
                                   "lead to bad performance. Op name="
                                << op->op_name_;
                    }
                }
            }
        }
    }
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

ir_module_ptr lower_fusion_manager(const context_ptr &ctx,
        outer_loop_generator_t *gen, sc_op *op, fusion_manager *fmgr,
        bool check_parallel) {
    std::vector<sc_op_ptr> out_failed;
    auto ret = try_lower_fusion_manager(
            ctx, gen, op, fmgr, check_parallel, false, out_failed);
    COMPILE_ASSERT(ret, "Fusible Op generation failed");
    return ret;
}

ir_module_ptr inplaced_reorder_get_func(sc_op *op, const context_ptr &ctx) {
    auto modu = std::make_shared<ir_module_t>(ctx);

    std::vector<expr> ins;
    // real_outs are the output tensors in the function arguments
    std::vector<expr> real_outs;
    auto func = graph::create_func_decl_for_op(op, ins, real_outs);
    builder::ir_builder_t bld;
    bld.push_scope();
    bld.push_evaluate(builder::make_write_struct(real_outs[0],
            builder::make_read_struct(ins[0], dyn_tsr_struct_t::name,
                    dyn_tsr_struct_t::fields::data_ptr),
            dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::data_ptr));
    bld.push_returns(true);
    auto body = bld.pop_scope();
    func->body_ = std::move(body);
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
