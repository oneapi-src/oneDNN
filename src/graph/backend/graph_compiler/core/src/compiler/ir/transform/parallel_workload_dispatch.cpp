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
#include "parallel_workload_dispatch.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/config.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(parallel_workload_dispatcher,
        SC_PASS_DEPENDS_ON(tensor_init, constant_folder),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

// workload should be marked on stmts or for_loop
static inline size_t extract_workload_from_stmt(
        const stmt &v, int runtime_num_threads, bool &is_brgemm) {
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
                is_brgemm = true;
                return memory_access_threshold_per_thread * runtime_num_threads;
            }
        }
    }
    return 0UL;
}

class tid_intrin_replacer_t : public ir_visitor_t {
public:
    expr tid_;
    tid_intrin_replacer_t(const expr &tid) : tid_ {tid} {}
    tid_intrin_replacer_t() = default;
    expr_c visit(intrin_call_c v) override {
        if (v->type_ == intrin_type::get_group_thread_id) {
            auto level = v->args_.at(0).as<constant>();
            if (get_const_as_int(level) == -1) {
                return builder::make_cast(datatypes::s32, tid_);
            }
        }
        return ir_visitor_t::visit(v);
    }
};

static stmt split_parallel_loop(
        const for_loop &v, size_t wkld, int runtime_num_threads) {
    if (v->kind_ != for_type::PARALLEL || runtime_num_threads == 1) {
        return v;
    }
    bool need_split_parallel = wkld != 0;
    if (wkld < (unsigned)runtime_num_threads
                    * memory_access_threshold_per_thread) {
        runtime_num_threads = utils::divide_and_ceil(
                wkld, memory_access_threshold_per_thread);
    }
    if (need_split_parallel) {
        auto tid = builder::make_var(datatypes::index, "tid");
        auto seq = builder::make_stmts_unattached({}).static_as<stmts>();
        auto thread_for = builder::make_for_loop_unattached(tid, UINT64_C(0),
                uint64_t(runtime_num_threads), UINT64_C(1), seq, true,
                for_type::PARALLEL, 0);
        expr start, end;
        builtin::generate_balance211(runtime_num_threads, v->iter_begin_,
                v->iter_end_, v->step_, tid, nullptr, &start, nullptr, &end,
                &seq->seq_);
        v->iter_begin_ = start;
        v->iter_end_ = end;
        v->num_threads_ = 0;
        v->kind_ = for_type::NORMAL;

        tid_intrin_replacer_t replacer {tid};
        seq->seq_.emplace_back(replacer.dispatch(v).remove_const());
        return thread_for;
    }
    return v;
}
class workload_accumulator_t : public ir_visitor_t {
public:
    workload_accumulator_t(bool record, std::unordered_map<stmt_c, size_t> &map)
        : record_workload(record), stmt_workload_map(map) {}
    bool record_workload;
    std::unordered_map<stmt_c, size_t> &stmt_workload_map;
    size_t cur_workload = 0UL;
    const int runtime_num_threads = runtime_config_t::get().get_num_threads();
    // if the current parallel for contains complex operations like brgemm, the
    // function may has complex body. The parallel workload dispatch may break
    // buffer scheduling inside the function. We may skip this pass on this
    // case.
    bool is_complex_pfor_ = false;
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    expr_c dispatch(expr_c v) override { return v; }

    stmt_c dispatch(stmt_c v) override {
        bool is_brgemm = false;
        size_t stmt_workload = extract_workload_from_stmt(
                v.remove_const(), runtime_num_threads, is_brgemm);
        is_complex_pfor_ |= is_brgemm;
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
        bool is_normal_parallel
                = v->kind_ == for_type::PARALLEL && v->num_threads_ == 0;
        bool old_is_complex = is_complex_pfor_;
        if (is_normal_parallel) { is_complex_pfor_ = false; }
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
            total_wkld
                    = memory_access_threshold_per_thread * runtime_num_threads;
        }
        cur_workload = total_wkld;
        changed |= (body_wkld > 0UL);
        if (changed) {
            stmt_c newv = copy_attr(*v,
                    builder::make_for_loop_unattached(var, begin, end, step,
                            body, v->incremental_, v->kind_, v->num_threads_));
            if (is_normal_parallel) {
                if (!is_complex_pfor_) {
                    newv = split_parallel_loop(newv.checked_as<for_loop>(),
                            total_wkld, runtime_num_threads);
                }
                is_complex_pfor_ = old_is_complex;
            }
            if (record_workload) { stmt_workload_map[newv] = total_wkld; }
            return newv;
        }
        if (is_normal_parallel) { is_complex_pfor_ = old_is_complex; }
        if (record_workload) { stmt_workload_map[v] = total_wkld; }
        return v;
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
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
