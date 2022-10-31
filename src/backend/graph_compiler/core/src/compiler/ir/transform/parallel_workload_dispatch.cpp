/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#include "parallel_workload_dispatch.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/config.hpp>
#include <unordered_map>

namespace sc {

SC_DECL_PASS_INFO(parallel_workload_dispatcher,
        SC_PASS_DEPENDS_ON(
                tensor_init, constant_folder, nested_parallel_flattener),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

// the workload allocated to a thread can not exceed
// threshold_per_thread * coefficient
static float threshold_coefficient = 1.f;

// workload should be marked on stmts or for_loop
static inline size_t extract_workload_from_stmt(const stmt &v) {
    if (!v.defined()) return 0UL;
    if (v->attr().has_key(op_traits::workload_computable_t::workload_number)) {
        return v->attr().get<size_t>(
                op_traits::workload_computable_t::workload_number);
    } else if (v.isa<evaluate>()) {
        // Disable workload dispatch on brgemm
        // TODO(zhennan): give brgemm a resonable workload
        const auto &eval = v.as<evaluate>();
        if (eval->value_.isa<intrin_call>()) {
            const auto &intrin = eval->value_.as<intrin_call>();
            if (intrin->type_ == intrin_type::brgemm
                    || intrin->type_ == intrin_type::list_brgemm) {
                return memory_access_threshold_per_thread;
            }
        }
    }
    return 0UL;
}

static void split_parallel_loop(const for_loop &v, size_t wkld) {
    bool need_split_parallel = v->kind_ == for_type::PARALLEL
            && wkld < memory_access_threshold_per_thread && wkld;
    if (need_split_parallel) {
        size_t block = 1UL;
        size_t best_block = 1UL;
        size_t new_wkld = wkld;
        assert(v->iter_begin_.isa<constant>() && v->iter_end_.isa<constant>()
                && v->step_.isa<constant>());
        size_t begin = get_expr_as_int(v->iter_begin_);
        size_t end = get_expr_as_int(v->iter_end_);
        size_t step = get_expr_as_int(v->step_);
        size_t loop_len = end - begin;
        while (block <= loop_len
                && static_cast<float>(new_wkld)
                        < static_cast<float>(memory_access_threshold_per_thread)
                                * threshold_coefficient) {
            best_block = block;
            block++;
            while (block <= loop_len && loop_len % block > 0UL) {
                block++;
            }
            if (block > loop_len) { break; }
            new_wkld = wkld * block;
        }
        if (best_block > 1UL && step == 1) { v->split(best_block); }
    }
}
class workload_accumulator_t : public ir_visitor_t {
public:
    workload_accumulator_t(bool record, std::unordered_map<stmt_c, size_t> &map)
        : record_workload(record), stmt_workload_map(map) {}
    bool record_workload;
    std::unordered_map<stmt_c, size_t> &stmt_workload_map;
    size_t cur_workload = 0UL;
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    stmt_c dispatch(stmt_c v) override {
        size_t stmt_workload = extract_workload_from_stmt(v.remove_const());
        cur_workload = 0UL;
        auto newv = ir_visitor_t::dispatch(v);
        cur_workload = stmt_workload + cur_workload;
        return newv;
    }
    stmt_c visit(stmts_c v) override {
        bool changed = false;
        size_t total_wkld = 0UL;
        std::vector<stmt_c> seq;
        seq.reserve(v->seq_.size());
        for (auto &s : v->seq_) {
            auto newstmt = dispatch(s);
            total_wkld = total_wkld + cur_workload;
            changed |= !newstmt.ptr_same(s);
            seq.emplace_back(std::move(newstmt));
        }
        cur_workload = total_wkld;
        changed |= v->seq_.size() != seq.size();

        if (changed) {
            stmt newv = copy_attr(*v, builder::make_stmts_unattached(seq));
            if (record_workload) { stmt_workload_map[newv] = total_wkld; }
            return std::move(newv);
        }
        if (record_workload) { stmt_workload_map[v] = total_wkld; }
        return std::move(v);
    }
    stmt_c visit(if_else_c v) override {
        auto cond = dispatch(v->condition_);
        size_t then_wkld = cur_workload;
        auto thencase = dispatch(v->then_case_);
        size_t else_wkld = cur_workload;

        stmt_c elsecase;
        if (v->else_case_.defined()) elsecase = dispatch(v->else_case_);
        size_t total_wkld = std::max(then_wkld, else_wkld);
        cur_workload = total_wkld;
        bool changed = !cond.ptr_same(v->condition_)
                || !elsecase.ptr_same(v->else_case_)
                || !thencase.ptr_same(v->then_case_);
        if (changed) {
            stmt newv = copy_attr(*v,
                    builder::make_if_else_unattached(cond, thencase, elsecase));
            if (record_workload) { stmt_workload_map[newv] = total_wkld; }
            return std::move(newv);
        }
        if (record_workload) { stmt_workload_map[v] = total_wkld; }
        return std::move(v);
    }
    stmt_c visit(for_loop_c v) override {
        size_t total_wkld = 0UL;
        auto var = dispatch(v->var_);
        auto begin = dispatch(v->iter_begin_);
        auto end = dispatch(v->iter_end_);
        auto step = dispatch(v->step_);
        auto body = dispatch(v->body_);
        size_t body_wkld = cur_workload;
        assert(body.isa<stmts_c>() || body.isa<for_loop_c>());
        bool changed = !(var.ptr_same(v->var_) && begin.ptr_same(v->iter_begin_)
                && end.ptr_same(v->iter_end_) && step.ptr_same(v->step_)
                && body.ptr_same(v->body_));
        if (begin.isa<constant>() && end.isa<constant>()) {
            total_wkld = total_wkld
                    + (get_expr_as_int(end) - get_expr_as_int(begin))
                            * body_wkld;
        } else {
            total_wkld = memory_access_threshold_per_thread;
        }
        cur_workload = total_wkld;
        changed |= (body_wkld > 0UL);
        if (changed) {
            stmt_c newv = copy_attr(*v,
                    builder::make_for_loop_unattached(var, begin, end, step,
                            body, v->incremental_, v->kind_, v->num_threads_));
            // todo: for dynamic boundary cases, try split of quotient and
            // remainder process with if-else.
            if (v->kind_ == for_type::PARALLEL && begin.isa<constant>()
                    && end.isa<constant>() && step.isa<constant>()) {
                // copy whole for loop as split is inplace
                std::unordered_map<expr_c, expr> rmap;
                ir_copier_t cpier(rmap, false);
                newv = cpier(newv);
                split_parallel_loop(newv.checked_as<for_loop>(), body_wkld);
            }
            if (record_workload) { stmt_workload_map[newv] = total_wkld; }
            return newv;
        }
        if (record_workload) { stmt_workload_map[v] = total_wkld; }
        return std::move(v);
    }
};

func_c parallel_workload_dispatcher_t::operator()(func_c f) {
    workload_accumulator_t vis(record_workload_, stmt_workload_map_);
    return vis.dispatch(f);
}

stmt_c parallel_workload_dispatcher_t::operator()(stmt_c f) {
    workload_accumulator_t vis(record_workload_, stmt_workload_map_);
    return vis.dispatch(std::move(f));
}
} // namespace sc
