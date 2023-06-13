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
#include "buffer_schedule.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include "constant_fold.hpp"
#include "pointer_alias_info.hpp"
#include "static_memory_planner.hpp"
#include "tensor_inplace_info.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/intrinsics.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/visitor.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/optional_find.hpp>

SC_MODULE(pass.buffer_schedule);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(buffer_scheduler,
        SC_PASS_DEPENDS_ON(tensor_shrinker, tensor_inplace, tensor2var,
                tensor_init, index_flattener, dead_write_eliminator,
                nested_parallel_flattener),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

using namespace special_ticks;

// a visitor which has "instruction counter". Every visit on any expr will
// increase the tick_ by 1. Note that by default, all tick_visitor visits exprs
// in the order where they are executed at the run time. So the tick_ is the
// execution order of each expr. If an expr has less tick than another, then it
// is executed after the other.
class tick_visitor_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    uint32_t tick_ = 0;
    expr_c dispatch(expr_c e) override {
        tick_++;
        return ir_visitor_t::dispatch(std::move(e));
    }

    virtual void enter_complex_scope(const stmt_c &cur) {}
    virtual void leave_complex_scope(const stmt_c &cur) {}

    // for-loop is a special case: we can not really order the body exprs and
    // iter_end_. So before we enter the body, we call enter_complex_scope(),
    // after the body + iter_end_, we call leave_complex_scope(). Users can do
    // specialized operation on the tick_
    stmt_c visit(for_loop_c v) override {
        auto var = dispatch(v->var_);
        auto begin = dispatch(v->iter_begin_);
        auto step = dispatch(v->step_);
        enter_complex_scope(v);
        // different visitor order than ir_visitor_t:
        // iter_end_ will be executed after the loop body at the run time
        auto body = dispatch(v->body_);
        auto end = dispatch(v->iter_end_);
        leave_complex_scope(v);

        bool changed = !(var.ptr_same(v->var_) && begin.ptr_same(v->iter_begin_)
                && end.ptr_same(v->iter_end_) && step.ptr_same(v->step_)
                && body.ptr_same(v->body_));
        if (changed) {
            return copy_attr(*v,
                    builder::make_for_loop_unattached(var, begin, end, step,
                            body, v->incremental_, v->kind_, v->num_threads_));
        } else {
            return std::move(v);
        }
    }

    stmt_c visit(if_else_c v) override {
        auto cond = dispatch(v->condition_);
        enter_complex_scope(v);
        auto then = dispatch(v->then_case_);
        stmt_c else_case;
        if (v->else_case_.defined()) { else_case = dispatch(v->else_case_); }
        leave_complex_scope(v);
        if (!cond.ptr_same(v->condition_) || !then.ptr_same(v->then_case_)
                || !else_case.ptr_same(v->else_case_)) {
            return copy_attr(*v,
                    builder::make_if_else_unattached(cond, then, else_case));
        }
        return v;
    }
};

// the struct to track the call site which may have inplace reuse
struct call_site_info_t {
    func_c func_;
    // the tensors passed to the function on the caller side. It has length of
    // func->args_. If an arg is not tensor related, the element in this array
    // will be empty.
    std::vector<expr_c> tensors_passed_;
};

// the tensor is not thread local
static constexpr uint64_t NOT_THREAD_LOCAL = 0;
struct tensor_tick_info_t {
    // first read/write tick, will not be reset by complex scopes, useful for
    // sorting the tensors
    int64_t real_first_access_ = TICK_NOT_EXIST;
    int64_t first_access_ = TICK_NOT_EXIST; // first read/write tick
    int64_t last_read_ = TICK_NOT_EXIST; // last read tick
    std::set<int64_t> writes_; // all write ticks
    int64_t create_ = TICK_NOT_EXIST; // tensor creation tick
    int64_t delete_ = TICK_NOT_EXIST; // the tick that the tensor scope is done
    bool is_arg_ = false; // if is the tensor defined in function args
    // if the tensor is already scheduled, infered from the define stmt's
    // init_. If the tensor is arg tensor, already_scheduled_base_ is the tensor
    // itself
    expr_c already_scheduled_base_;
    uint64_t scope_ = NOT_THREAD_LOCAL; // parallel scope id
    bool has_hint_ = false; // if the tensor has hint tick info
    // the tensors that the current tensor can inplace reuse buffer with. Not
    // all of the tensors in this set is valid. Only if last_access_tick of the
    // tensor in the set == the first_access_tick of the current tensor can
    // the tensor be reused.
    std::vector<std::pair<expr_c, inplace_kind>> inplace_reuse_;
    // the call site of the inplace function call. may be null. It is used to
    // back-propagate the inplace result to the function to be called, to inform
    // it that some of the args has pointer alias
    std::shared_ptr<call_site_info_t> inplace_call_site_;
    // Only valid when this tensor is an argument to list_brgemm calls. The set
    // is used to record the A and B data tensors that the address list tensor
    // of list_brgemm points to.
    std::unique_ptr<std::unordered_set<expr_c>> list_brgemm_tensors_;

    bool is_already_scheduled() const {
        return already_scheduled_base_.defined();
    }
    int64_t get_last_access() const {
        int64_t last_access = last_read_;
        if (!writes_.empty()) {
            last_access = std::max(last_access, *writes_.rbegin());
        }
        if (last_access == TICK_NOT_EXIST) { last_access = first_access_; }
        return last_access;
    }
};

static bool is_parallel_for(const stmt_c &v) {
    if (v.isa<for_loop_c>()) {
        auto loop = v.static_as<for_loop_c>();
        if (loop->kind_ == for_type::PARALLEL) { return true; }
        if (v->attr_
                && v->attr_->get_or_else(
                        attr_keys::buf_sched_top_scope, false)) {
            return true;
        }
    }
    return false;
}

class reference_tick_finder_t : public tick_visitor_t {
public:
    using tick_visitor_t::dispatch;
    using tick_visitor_t::visit;
    // tensor -> tensor_tick_info
    std::unordered_map<expr_c, tensor_tick_info_t> &out_;
    // memorize all tensors that are read/written in for-loop/if-else. Will
    // later reset their tick to the tick of the end of the loop, the values are
    // bit masks: the tensor is read if (mask & READ_MASK != 0) the tensor is
    // written if (mask & WRITE_MASK != 0)
    // scope may be nested in parallel for, we use a stack to record inner
    // ticks.
    std::vector<std::unordered_map<expr_c, char>> ticks_in_scope_;
    static constexpr char READ_MASK = 1;
    static constexpr char WRITE_MASK = 1 << 1;
    // the depth of nested for we are currently visiting
    int for_depth_ = 0;
    // A stack recording the start tick in loop scopes, for those tensors who
    // are defined in loops or accessed in loops.
    std::vector<int64_t> for_start_ticks_;
    // the depth of in for loop for the outmost level of parallel for.
    // When parallel_depth_ > 0, the current scope is in parallel.
    int parallel_depth_ = -1;
    bool in_parallel_for_ = false;
    // the number of "top-level" stmts that contains buffers to reschedule.
    // The top-level stmts of a function has scope_id=NOT_THREAD_LOCAL(0) and
    // each parallel-for has an unique scope id. It will be used to set scope_
    // in tick_info.
    uint64_t scope_id_ = NOT_THREAD_LOCAL;
    // record scope id of stmts for base tensor replacement
    std::unordered_map<stmt_c, uint64_t> stmts_to_scope_id_;

    // the stack for tensors in nested scopes
    std::vector<std::vector<expr_c>> tensor_in_scope_;
    // the list of all defined tensors, by tensor creation order
    std::vector<expr_c> &defined_tensors_;
    // the map of tensor identity to the tensor
    std::unordered_map<alias_info::tensor_alias_identity_t *, expr_c>
            identity_to_tensor_;
    reference_tick_finder_t(std::unordered_map<expr_c, tensor_tick_info_t> &out,
            std::vector<expr_c> &defined_tensors)
        : out_(out), defined_tensors_(defined_tensors) {}

    // if a visit to the tensor_node means that the tensor use is complicated.
    // If the parent node is call/indexing, this variable should be true
    bool good_tensor_ = false;
    enum {
        READ,
        WRITE,
        TENSOR_PTR,
    } indexing_parent_
            = READ;

    // update the ticks of tensor `t` when leaving the complex scope, according
    // to three parameters:
    // for start tick: default the tick entered the outmost loop. If we got
    // hint tick, add it with default tick.
    // for end tick: default the tick leaved the outmost loop. If we got
    // hint tick, add it with default tick.
    // read/write mask: mark tensor is read/written in loop.
    void update_complex_scope_ticks(const expr_c &t, int64_t for_start_tick,
            int64_t for_end_tick, char rw_mask) {
        auto itr = out_.find(t);
        if (itr != out_.end()) {
            if (itr->second.already_scheduled_base_.defined()
                    && !t.ptr_same(itr->second.already_scheduled_base_)) {
                update_complex_scope_ticks(itr->second.already_scheduled_base_,
                        for_start_tick, for_end_tick, rw_mask);
            }
            assert(for_start_tick != TICK_NOT_EXIST
                    && for_start_tick != COMPLICATED_ACCESS);
            assert(itr->second.first_access_ != TICK_NOT_EXIST);
            // first used out of loop
            itr->second.first_access_
                    = std::min(itr->second.first_access_, for_start_tick);
            if (itr->second.last_read_ == COMPLICATED_ACCESS) { return; }
            auto first_access_tick = for_start_tick;
            auto last_access_tick = for_end_tick;
            // if tensor has additional hint tick info
            if (t->attr_
                    && t->attr_->has_key(attr_keys::hint_first_access_tick)) {
                itr->second.has_hint_ = true;
                auto hint_first_access_tick = t->attr_->get_or_else(
                        attr_keys::hint_first_access_tick, TICK_NOT_EXIST);
                auto hint_last_access_tick = t->attr_->get_or_else(
                        attr_keys::hint_last_access_tick, TICK_NOT_EXIST);
                // update tick
                if (hint_first_access_tick == HINT_IN_LOOP
                        || hint_last_access_tick == HINT_IN_LOOP) {
                    if (itr->second.first_access_ >= for_start_tick) {
                        itr->second.first_access_ = for_start_tick;
                    }
                    last_access_tick = for_end_tick;
                } else if (hint_first_access_tick != TICK_NOT_EXIST
                        && hint_first_access_tick != COMPLICATED_ACCESS) {
                    assert(hint_last_access_tick != TICK_NOT_EXIST
                            && hint_last_access_tick != COMPLICATED_ACCESS);
                    assert(itr->second.first_access_ >= 0);
                    // first access in loop, reset its first access
                    if (itr->second.first_access_ >= for_start_tick) {
                        itr->second.first_access_
                                = for_start_tick + hint_first_access_tick;
                    }
                    last_access_tick = for_start_tick + hint_last_access_tick;
                }
            }
            if (rw_mask & READ_MASK) {
                itr->second.last_read_ = last_access_tick;
            }
            if (rw_mask & WRITE_MASK) {
                itr->second.writes_.insert(last_access_tick);
            }
        }
    }

    void set_read_tick(const expr_c &t, int64_t tick) {
        auto itr = out_.find(t);
        if (itr != out_.end()) {
            if (itr->second.already_scheduled_base_.defined()
                    && !t.ptr_same(itr->second.already_scheduled_base_)) {
                set_read_tick(itr->second.already_scheduled_base_, tick);
            }
            if (itr->second.last_read_ == TICK_NOT_EXIST
                    && itr->second.writes_.empty()) {
                itr->second.first_access_ = tick;
                itr->second.real_first_access_ = tick;
            }
            // if the tensor has complicated use, don't optimize it
            if (itr->second.last_read_ == COMPLICATED_ACCESS) { return; }
            itr->second.last_read_ = tick;
            // record ticks in current scope; if the tensor is defined in parent
            // scope but used in parallel scope, add it to first scope(not
            // thread local) in ticks_in_scope_
            if (for_depth_ > 0) {
                assert(!ticks_in_scope_.empty());
                if (itr->second.scope_ != scope_id_) {
                    // non_thread_local in parallel, push to parent scope
                    ticks_in_scope_[0][t] |= READ_MASK;
                } else if (for_depth_ != parallel_depth_) {
                    // push to current scope
                    ticks_in_scope_.back()[t] |= READ_MASK;
                }
            }
        }
        // else, the tensor is not defined in the current scope
    }

    void set_write_tick(const expr_c &t, int64_t tick) {
        auto itr = out_.find(t);
        if (itr != out_.end()) {
            if (itr->second.already_scheduled_base_.defined()
                    && !t.ptr_same(itr->second.already_scheduled_base_)) {
                set_write_tick(itr->second.already_scheduled_base_, tick);
            }
            if (itr->second.last_read_ == TICK_NOT_EXIST
                    && itr->second.writes_.empty()) {
                itr->second.first_access_ = tick;
                itr->second.real_first_access_ = tick;
            }
            if (itr->second.last_read_ == COMPLICATED_ACCESS) { return; }
            // record ticks in current scope; if the tensor is defined in parent
            // scope but used in parallel scope, add it to first scope(not
            // thread local) in ticks_in_scope_
            if (for_depth_ > 0) {
                assert(!ticks_in_scope_.empty());
                if (itr->second.scope_ != scope_id_) {
                    ticks_in_scope_[0][t] |= WRITE_MASK;
                    return;
                } else if (for_depth_ != parallel_depth_) {
                    ticks_in_scope_.back()[t] |= WRITE_MASK;
                    return;
                }
            }
            itr->second.writes_.insert(tick);
        }
        // else, the tensor is not defined in the current scope
    }

    void set_both_tick(const expr_c &t, int64_t tick) {
        auto itr = out_.find(t);
        if (itr != out_.end()) {
            if (itr->second.already_scheduled_base_.defined()
                    && !t.ptr_same(itr->second.already_scheduled_base_)) {
                set_both_tick(itr->second.already_scheduled_base_, tick);
            }
            if (tick == COMPLICATED_ACCESS) {
                itr->second.last_read_ = tick;
                itr->second.writes_.clear();
                return;
            }
            if (itr->second.last_read_ == TICK_NOT_EXIST
                    && itr->second.writes_.empty()) {
                itr->second.first_access_ = tick;
                itr->second.real_first_access_ = tick;
            }

            if (itr->second.last_read_ == COMPLICATED_ACCESS) { return; }
            itr->second.last_read_ = tick;
            // record ticks in current scope; if the tensor is defined in parent
            // scope but used in parallel scope, add it to first scope(not
            // thread local) in ticks_in_scope_
            if (for_depth_ > 0) {
                assert(!ticks_in_scope_.empty());
                if (itr->second.scope_ != scope_id_) {
                    ticks_in_scope_[0][t] = WRITE_MASK | READ_MASK;
                    return;
                } else if (for_depth_ != parallel_depth_) {
                    ticks_in_scope_.back()[t] = WRITE_MASK | READ_MASK;
                    return;
                }
            }
            itr->second.writes_.insert(tick);
        }
        // else, the tensor is not defined in the current scope
    }

    // called when we are entering the body of a complex scope like for-loop
    void enter_complex_scope(const stmt_c &v) override {
        in_parallel_for_ |= is_parallel_for(v);
        for_depth_++;
        if (in_parallel_for_ && parallel_depth_ == -1) {
            scope_id_++;
            stmts_to_scope_id_[v.checked_as<for_loop>()->body_] = scope_id_;
            parallel_depth_ = for_depth_;
        }
        // ticks_in_scope has at most 2 scopes, non thread local/thread
        // local
        if (for_depth_ == 1 || parallel_depth_ + 1 == for_depth_) {
            for_start_ticks_.emplace_back(tick_);
            ticks_in_scope_.emplace_back();
            assert(for_start_ticks_.size() <= 2);
            COMPILE_ASSERT(ticks_in_scope_.size() <= 2UL,
                    "Buffer scheduling currently only supports 2 level scopes");
        }
    }

    // called when we are leaving a complex scope like for-loop
    void leave_complex_scope(const stmt_c &cur) override {
        for_depth_--;
        // if we are in the most outer loop or inner loop in parallel loop
        if (for_depth_ == 0 || parallel_depth_ == for_depth_) {
            assert(!for_start_ticks_.empty());
            assert(!ticks_in_scope_.empty());
            // reset all referenced tensor to the current tick
            for (auto &itr : ticks_in_scope_.back()) {
                update_complex_scope_ticks(
                        itr.first, for_start_ticks_.back(), tick_, itr.second);
            }
            // exit inner thread local scope
            for_start_ticks_.pop_back();
            ticks_in_scope_.pop_back();
        }
        // we assume that there is no nested parallel for
        if (is_parallel_for(cur)) { in_parallel_for_ = false; }
        if (parallel_depth_ == for_depth_ + 1) { parallel_depth_ = -1; }
    }

    stmt_c visit(assign_c v) override {
        // we allow getting tensor address for "A_list[...]=&A", when A_list is
        // a list_brgemm_arg
        if (v->var_.isa<indexing>()) {
            auto tsr = v->var_.static_as<indexing>()->ptr_;
            if (tsr->attr_
                    && tsr->attr_->get_or_else("list_brgemm_arg", false)) {
                // make accesses on the real buffer not "complicated"
                good_tensor_ = true;
                auto itr = out_.find(tsr);
                if (itr != out_.end()) {
                    if (!itr->second.list_brgemm_tensors_) {
                        itr->second.list_brgemm_tensors_ = utils::make_unique<
                                std::unordered_set<expr_c>>();
                    }
                    auto list_brg_real_tsr = get_base_tensor_of(v->value_);
                    if (list_brg_real_tsr.defined()) {
                        itr->second.list_brgemm_tensors_->insert(
                                list_brg_real_tsr);
                    }
                }
            }
        }
        // read first, then write
        dispatch(v->value_);
        if (v->var_.isa<indexing>()) indexing_parent_ = WRITE;
        dispatch(v->var_);
        return v;
    }

    stmt_c visit(stmts_c v) override {
        if (stmts_to_scope_id_.empty()) {
            assert(scope_id_ == 0);
            stmts_to_scope_id_[v] = scope_id_;
        }
        tensor_in_scope_.emplace_back(std::vector<expr_c>());
        tick_visitor_t::visit(v);
        for (auto &t : tensor_in_scope_.back()) {
            out_[t].delete_ = tick_;
        }
        tensor_in_scope_.pop_back();
        return v;
    }

    stmt_c visit(define_c v) override {
        if (v->var_.isa<tensor>()) {
            expr already_scheduled_base;
            // only process local tensors here
            if (v->linkage_ == linkage::local
                    && (!v->attr_
                            || !v->attr_->get_or_else(
                                    attr_keys::tsr_dont_buf_sched, false))) {
                bool flag = false;
                if (!v->init_.defined()) {
                    // e.g., tensor c : [u8 * 32]. This is the default case.
                    flag = true;
                } else if (v->init_.isa<tensorptr_c>()) {
                    // e.g., tensor A_list : [pointer * 8] =
                    // &__rescheduled_0[0UL] which is built in former
                    // buffer_schedule process
                    const auto &tsr
                            = v->init_.static_as<tensorptr>()->base_->ptr_;
                    if (tsr->attr_
                            && tsr->attr_->get_or_else(
                                    attr_keys::can_be_scheduled, false)) {
                        flag = true;
                        already_scheduled_base = tsr;
                    }
                } else {
                    flag = false;
                }

                if (flag) {
                    out_[v->var_].create_ = tick_;
                    // set scope id for thread local tensor
                    if (in_parallel_for_) {
                        assert(scope_id_ >= 1);
                        out_[v->var_].scope_ = scope_id_;
                    }
                    if (v->init_.defined()) {
                        out_[v->var_].already_scheduled_base_
                                = already_scheduled_base;
                    }
                    if (auto identity = alias_info::get_alias_info(*v->var_)) {
                        auto itr = identity_to_tensor_.find(identity);
                        if (itr != identity_to_tensor_.end()) {
                            SC_MODULE_WARN << "Multiple tensors uses the same "
                                              "tensor identity: "
                                           << v->var_;
                            itr->second = expr_c();
                        } else {
                            identity_to_tensor_[identity] = v->var_;
                        }
                    }
                    defined_tensors_.emplace_back(v->var_);
                    tensor_in_scope_.back().emplace_back(v->var_);
                    good_tensor_ = true;
                    dispatch(v->var_);
                    if (v->init_.defined()) {
                        good_tensor_ = true;
                        dispatch(v->init_);
                        good_tensor_ = false;
                    }
                    return v;
                }
            }
        }
        tick_visitor_t::visit(v);
        return v;
    }

    expr_c visit(indexing_c v) override {
        // if the parent of the indexing is tensorptr, good_tensor_ remains
        if (indexing_parent_ != TENSOR_PTR) { good_tensor_ = true; }
        auto local_indexing_parent = indexing_parent_;
        indexing_parent_ = READ;

        tick_visitor_t::visit(v);
        switch (local_indexing_parent) {
            case READ:
                set_read_tick(v->ptr_.checked_as<tensor_c>(), tick_);
                break;
            case WRITE:
                set_write_tick(v->ptr_.checked_as<tensor_c>(), tick_);
                break;
            case TENSOR_PTR:
                if (v->ptr_.isa<tensor_c>()) {
                    auto tsr = v->ptr_.checked_as<tensor_c>();
                    if (tsr->attr_
                            && tsr->attr_->get_or_else(
                                    attr_keys::can_be_scheduled, false)) {
                        set_read_tick(tsr, tick_);
                    }
                }
                break;
            default: assert(0 && "Bad indexing parent");
        }
        return v;
    }
    expr_c visit(tensor_c v) override {
        tick_visitor_t::visit(v);
        if (!good_tensor_) {
            // complex use of tensor. Normal use of tensor in
            // indexing/function call will not go here
            set_both_tick(v, COMPLICATED_ACCESS);
        }
        good_tensor_ = false;
        return v;
    }

    expr_c visit(tensorptr_c v) override {
        indexing_parent_ = TENSOR_PTR;
        tick_visitor_t::visit(v);
        return v;
    }

    void dispatch_args(const std::vector<expr> &args) {
        bool old_good = good_tensor_;
        // first calculate the tick after evaluating the args
        for (unsigned i = 0; i < args.size(); i++) {
            auto &p = args[i];
            if (old_good || p.isa<tensor>() || p.isa<tensorptr>()) {
                good_tensor_ = true; // good to use tensorptr in args
            }
            dispatch(p);
        }
    }

    // if a tensor/tensorptr is passed in function args, set the r/w ticks
    expr_c visit(call_c v) override {
        using hint_t = std::vector<
                std::pair<int, std::vector<tensor_inplace_info_t>>>;
        hint_t *inplace_hint = (v->func_->attr_)
                ? (v->func_->attr_->get_or_null<hint_t>(
                        function_attrs::inplace_hint))
                : nullptr;
        if (v->func_ == builtin::get_mem_set_func()
                || v->func_ == builtin::get_brgemm_init_func()) {
            good_tensor_ = true;
        }
        dispatch_args(v->args_);
        if (auto ex = std::dynamic_pointer_cast<expr_base>(v->func_)) {
            dispatch(expr_c(ex));
        }
        // now tick_ is the tick when the last parameter is calculated. set the
        // tick for referenced tensors
        auto prototype = v->get_prototype();
        auto call_site = std::make_shared<call_site_info_t>();
        std::vector<expr_c> &arg_tensors = call_site->tensors_passed_;
        call_site->func_ = prototype;
        arg_tensors.resize(v->args_.size());
        for (unsigned i = 0; i < v->args_.size(); i++) {
            auto &p = v->args_[i];
            auto &funcp = prototype->params_[i];
            tensor_c tsr = get_base_tensor_of(p);
            if (tsr.defined()) {
                bool has_read
                        = funcp->attr_ && funcp->attr_->has_key("read_buffer");
                if (has_read) { set_read_tick(tsr, tick_); }
                bool has_write
                        = funcp->attr_ && funcp->attr_->has_key("write_buffer");
                if (has_write) { set_write_tick(tsr, tick_); }
                if (!has_read && !has_write) {
                    // by default the buffer is r/w
                    set_both_tick(tsr, tick_);
                }
                if (inplace_hint) { arg_tensors[i] = tsr; }
            }
        }
        if (inplace_hint) {
            for (auto &kv : *inplace_hint) {
                auto &ths_tsr = arg_tensors.at(kv.first);
                auto itr = out_.find(ths_tsr);
                if (itr == out_.end()) { continue; }
                // if the tensor is not firstly accessed at current time, the
                // inplace hint is meaningless for the tensor, because it is
                // already created
                if (tick_ != itr->second.first_access_) { continue; }
                auto &can_inplace_tsrs = kv.second;
                COMPILE_ASSERT(ths_tsr.defined(),
                        "Bad inplace hint index: " << kv.first);
                for (auto idx : can_inplace_tsrs) {
                    auto &inplace_tsr = arg_tensors.at(idx.used_arg_idx_);
                    COMPILE_ASSERT(inplace_tsr.defined(),
                            "Bad inplace hint index: " << idx.used_arg_idx_);
                    itr->second.inplace_reuse_.emplace_back(
                            inplace_tsr, idx.kind_);
                }
                itr->second.inplace_call_site_ = call_site;
            }
        }
        return v;
    }

    // if a tensor/tensorptr is passed in function args, set the r/w ticks
    expr_c visit(intrin_call_c v) override {
        if (v->type_ == intrin_type::brgemm
                || v->type_ == intrin_type::list_brgemm) {
            good_tensor_ = true;
            dispatch_args(v->args_);
            // now tick_ is the tick when the last parameter is calculated. set
            // the tick for referenced tensors
            assert((v->type_ == intrin_type::brgemm
                           && v->check_brgemm_arg_size(
                                   brgemm_args::NUM_FULL_ARGS_STRIDE))
                    || (v->type_ == intrin_type::list_brgemm
                            && v->check_brgemm_arg_size(
                                    brgemm_args::NUM_FULL_ARGS_LIST)));
            for (int i = 0; i < brgemm_args::C + 1; i++) {
                auto &p = v->args_[i];
                tensor_c tsr = get_base_tensor_of(p);
                if (tsr.defined()) {
                    switch (i) {
                        case brgemm_args::A: // fall through
                        case brgemm_args::B:
                            set_read_tick(tsr, tick_);
                            // set the ticks of the real tensors
                            if (v->type_ == intrin_type::list_brgemm) {
                                auto itr = out_.find(tsr);
                                if (itr != out_.end()
                                        && itr->second.list_brgemm_tensors_) {
                                    for (auto &linked_tsr :
                                            *itr->second.list_brgemm_tensors_) {
                                        set_read_tick(linked_tsr, tick_);
                                    }
                                }
                            }
                            break;
                        case brgemm_args::C: set_write_tick(tsr, tick_); break;
                        default: break;
                    }
                }
            }
        } else {
            ir_visitor_t::visit(v);
        }
        return v;
    }

    void parse_func_params(const std::vector<expr> &params) {
        for (auto &p : params) {
            if (p.isa<tensor>() && p->attr_
                    && p->attr_->has_key("write_buffer")) {
                auto &info = (out_[p] = {});
                info.create_ = 0;
                info.delete_ = std::numeric_limits<int64_t>::max();
                info.is_arg_ = true;
                info.already_scheduled_base_ = p;
            }
        }
    }

    func_c dispatch(func_c v) override {
        parse_func_params(v->params_);
        dispatch(v->body_);
        return v;
    }
};

class dead_tsr_write_remover_t : public tick_visitor_t {
public:
    using tick_visitor_t::dispatch;
    using tick_visitor_t::visit;
    // update stmts pointer in this pass.
    std::unordered_map<stmt_c, uint64_t> &stmts_to_scope_id_;
    // tensor -> tensor_tick_info
    std::unordered_map<expr_c, tensor_tick_info_t> &out_;
    std::unordered_set<expr_c> removed_in_for;
    dead_tsr_write_remover_t(
            std::unordered_map<stmt_c, uint64_t> &stmts_to_scope_id,
            std::unordered_map<expr_c, tensor_tick_info_t> &out)
        : stmts_to_scope_id_(stmts_to_scope_id), out_(out) {}

    int for_depth_ = 0;
    void enter_complex_scope(const stmt_c &v) override { for_depth_++; }

    // called when we are leaving a for-loop, remove the end-of-loop write
    void leave_complex_scope(const stmt_c &v) override {
        for_depth_--;
        // if we are in the most outer loop
        if (for_depth_ == 0) {
            for (auto &itr : removed_in_for) {
                auto infoitr = out_.find(itr);
                assert(infoitr != out_.end());
                auto &writes = infoitr->second.writes_;
                if (writes.find(tick_) != writes.end()) { writes.erase(tick_); }
            }
            removed_in_for.clear();
        }
    }

    stmt_c visit(assign_c v) override {
        // read first, then write
        dispatch(v->value_);
        dispatch(v->var_);
        if (v->var_.isa<indexing>()) {
            auto idxing = v->var_.static_as<indexing>();
            auto tsr = idxing->ptr_.checked_as<tensor>();
            auto itr = out_.find(tsr);
            if (itr != out_.end()) {
                auto last_read = itr->second.last_read_;
                bool should_remove = !itr->second.is_arg_
                        && !itr->second.is_already_scheduled()
                        && !itr->second.scope_ && !itr->second.has_hint_
                        && last_read != COMPLICATED_ACCESS
                        && tick_ > itr->second.last_read_;
                if (should_remove) {
                    auto &writes = itr->second.writes_;
                    if (for_depth_ == 0) {
                        // not in for, directly remove the write
                        assert(writes.find(tick_) != writes.end());
                        writes.erase(tick_);
                    } else {
                        // will be removed after for
                        removed_in_for.insert(tsr);
                    }
                    return builder::make_stmts_unattached({});
                }
            }
        }
        return v;
    }

    stmt_c visit(stmts_c v) override {
        auto newv = ir_visitor_t::visit(v);
        if (!newv.ptr_same(v)) {
            auto itr = stmts_to_scope_id_.find(v);
            if (itr != stmts_to_scope_id_.end()) {
                stmts_to_scope_id_[newv] = itr->second;
                stmts_to_scope_id_.erase(itr);
            }
        }
        return newv;
    }

    func_c dispatch(func_c v) override {
        auto body = dispatch(v->body_);
        if (!body.ptr_same(v->body_)) {
            return copy_attr(*v,
                    builder::make_func(v->name_, v->params_,
                            body.remove_const(), v->ret_type_));
        }
        return v;
    }
};

static bool check_if_tensor_valid(
        const std::pair<const expr_c, tensor_tick_info_t> &itr,
        bool &need_remove) {
    int64_t lastread = itr.second.last_read_;
    if (lastread == TICK_NOT_EXIST && itr.second.writes_.empty()
            && !itr.second.is_arg_ && !itr.second.is_already_scheduled()) {
        // the tensor is neither read or written, remove the tensor
        SC_MODULE_INFO << "Removing unused " << itr.first;
        need_remove = true;
        return false;
    }
    if (!itr.first.checked_as<tensor_c>()->dims_[0].isa<constant>()) {
        // only when the candidate has const shape can we reuse it
        SC_MODULE_INFO << "The tensor " << itr.first
                       << " has non-constant shape";
        return false;
    }
    SC_MODULE_INFO << "tsr: " << itr.first << ", LRT=" << itr.second.last_read_
                   << ", LWT="
                   << (itr.second.writes_.empty()
                                      ? TICK_NOT_EXIST
                                      : *itr.second.writes_.rbegin())
                   << ", FAT=" << itr.second.first_access_;
    return true;
}

static void schedule_tensors(
        std::unordered_map<expr_c, tensor_tick_info_t> &ticks,
        std::vector<expr_c> &defined_tensors,
        // the old_tensor -> reuse target tensor map
        std::unordered_map<expr_c, expr_c> &replace_map,
        // the reuse target tensor -> extended new tensor map
        std::unordered_map<expr_c, expr_c> &extend_map) {
    // last read tick => tensor, decending order
    std::multimap<int64_t, expr_c, std::greater<int64_t>> last_read_tensor;
    for (auto &itr : ticks) {
        bool need_remove = false;
        int64_t lastread = itr.second.last_read_;
        if (!check_if_tensor_valid(itr, need_remove)) {
            if (need_remove) { replace_map[itr.first] = expr_c(); }
            continue;
        }
        // if lastread == TICK_NOT_EXIST, we are good. Because TICK_NOT_EXIST is
        // a nagative number
        if (lastread != COMPLICATED_ACCESS) {
            last_read_tensor.insert(std::make_pair(lastread, itr.first));
        }
    }

    // there will be some cases that buffer A decides to reuse buffer B, which
    // is previously defined, but B has smaller size than A. We need to extend B
    // this map memorize all tensors that needs to be extended
    std::unordered_map<expr_c, int64_t> tensors_to_extend;
    for (auto &tsr : defined_tensors) {
        auto itr = ticks.find(tsr);
        assert(itr != ticks.end());
        const auto &cur_info = itr->second;
        // SC_MODULE_INFO << "Tsr " << tsr << " LRT= " << cur_info.last_read_;
        int64_t my_last_read = cur_info.last_read_;
        tensor_c cur_tensor = tsr.static_as<tensor_c>();
        assert(cur_tensor->dims_.size() == 1);
        if (!cur_tensor->dims_[0].isa<constant>()) {
            SC_MODULE_INFO << "The tensor " << cur_tensor
                           << " has non-constant shape";
            continue;
        }
        int64_t cur_tsr_size = get_const_as_int(
                cur_tensor->dims_[0].static_as<constant_c>());

        if (my_last_read == TICK_NOT_EXIST && cur_info.writes_.empty()) {
            // the tensor is neither read or written, remove the tensor
            SC_MODULE_WARN << "Removing " << tsr;
            replace_map[cur_tensor] = expr_c();
            continue;
        }
        if (my_last_read == COMPLICATED_ACCESS) {
            // it has complicated access pattern, skip
            SC_MODULE_INFO << "Complex access on " << tsr;
            continue;
        }

        // find an avaliable tensor, which:
        // 1. its last-read-tick is less than current first_access tick
        // 2. it is in tensor_alive_ set

        // upper_bound returns the first element which is less than cur tick
        auto titr = last_read_tensor.upper_bound(cur_info.first_access_);
        while (titr != last_read_tensor.end()) {
            auto new_tensor = titr->second.checked_as<tensor_c>();
            if (new_tensor.ptr_same(cur_tensor)) {
                // found myself as in the map, skip
                // this can occur when a tensor is written, but is never read
                ++titr;
                continue;
            }
            auto new_tensor_itr = ticks.find(new_tensor);
            if (new_tensor_itr == ticks.end()) {
                // the new_tensor is erased (reusing another tensor). Erase
                // it from the lookup map
                titr = last_read_tensor.erase(titr);
                continue;
            }
            auto &new_info = new_tensor_itr->second;
            // check if the candidate:
            // 1) is defined before cur tensor's first use
            // 2) dies after cur tensor dies
            // 3) its dtype is the same as cur tensor
            if (new_info.create_ <= cur_info.first_access_
                    && new_info.delete_ >= cur_info.delete_
                    && new_tensor->elem_dtype_ == cur_tensor->elem_dtype_) {
                // check that the candidate has no writes during the time range
                // when cur_tensor is in use: [cur_info.first_access_,
                // cur_info.last_read_]
                if (cur_info.last_read_ != TICK_NOT_EXIST) {
                    auto lower = new_info.writes_.lower_bound(
                            cur_info.first_access_);
                    auto upper
                            = new_info.writes_.upper_bound(cur_info.last_read_);
                    // lower: the first element >= first_access_
                    // upper: the first element > last_read_
                    if (lower != upper) {
                        // there are writes between first_access and last_read
                        SC_MODULE_INFO << "Write after read: Failed "
                                       << cur_tensor->name_ << "->"
                                       << new_tensor->name_ << "@" << *lower;
                        ++titr;
                        continue;
                    }
                }

                // if candidate is out_buffer from arg, we must not write to it
                // after its first original access
                if (new_info.is_arg_ && !cur_info.writes_.empty()
                        && *cur_info.writes_.rbegin()
                                >= new_info.first_access_) {
                    SC_MODULE_INFO << "Write final buf: Failed "
                                   << cur_tensor->name_ << "->"
                                   << new_tensor->name_;
                    ++titr;
                    continue;
                }

                // an avaliable tensor is found, we can reuse it!
                // now, check if the reused tensor: new_tensor is large
                // enough
                assert(new_tensor->dims_.size() == 1);
                // all tensors in last_read_tensor are ensured to have const
                // shape
                int64_t new_tsr_size = get_const_as_int(
                        new_tensor->dims_[0].static_as<constant_c>());
                // if the candidate is too small, need to extend it
                if (cur_tsr_size > new_tsr_size) {
                    // cannot extend an argument buffer: it is given by the
                    // caller
                    if (new_info.is_arg_) {
                        SC_MODULE_INFO << cur_tensor->name_ << " cannot reuse "
                                       << new_tensor->name_
                                       << ": cannot extend";
                        ++titr;
                        continue;
                    }
                    auto &sz = tensors_to_extend[new_tensor];
                    sz = std::max(sz, cur_tsr_size);
                }
                // push to replace map
                replace_map.insert(std::make_pair(cur_tensor, new_tensor));
                // update the reused tensor's last-read-tick
                // if my_last_read is TICK_NOT_EXIST, make sure the new tick
                // be new_tensor's last read tick
                int64_t newtick = std::max(my_last_read, new_info.last_read_);
                new_info.last_read_ = newtick;
                // merge the write traces
                new_info.writes_.insert(
                        cur_info.writes_.begin(), cur_info.writes_.end());
                last_read_tensor.insert(std::make_pair(newtick, new_tensor));
                last_read_tensor.erase(titr);
                // cur_tensor is removed
                ticks.erase(cur_tensor);

                SC_MODULE_INFO << "Reuse " << tsr << "->" << new_tensor;
                break;
            }
            ++titr;
        }
    }
    // fix the recursion in replace map: a replaced by b, and b replaced by c
    // then a should be replaced by c
    constexpr int MAX_REPLACE_RECURSION = 100;
    bool need_replace = true;
    for (int recursion = 0; need_replace; recursion++) {
        COMPILE_ASSERT(recursion < MAX_REPLACE_RECURSION,
                "MAX_REPLACE_RECURSION Reached, a loop in buffer schedule?");
        need_replace = false;
        for (auto &v : replace_map) {
            auto itr2 = replace_map.find(v.second);
            if (itr2 != replace_map.end()) {
                v.second = itr2->second;
                need_replace = true;
            }
        }
    }

    // replace the tensors to extend
    for (auto &v : tensors_to_extend) {
        SC_MODULE_INFO << "Extend " << v.first;
        auto newtsr = v.first->remake().checked_as<tensor>();
        newtsr->dims_[0] = builder::make_constant((uint64_t)v.second);
        extend_map.insert(std::make_pair(v.first, std::move(newtsr)));
    }
    // some of the reused tensors are extended, update them
    for (auto &v : replace_map) {
        // if the new value is also replaced
        auto itr2 = extend_map.find(v.second);
        if (itr2 != extend_map.end()) { v.second = itr2->second; }
    }
}

// if tsr has in-place hint and the hint contains only one target and the target
// has the same size & dtype of tsr
static const tensor_node *get_inplace_target_if_single_inplace(
        const std::unordered_map<alias_info::tensor_alias_identity_t *, expr_c>
                &identity_to_tensor,
        const tensor_node *tsr, const tensor_node *base) {
    return some_opt(tsr->attr_.get())
            .map([](any_map_t *v) {
                return v->get_or_null<std::vector<temp_tensor_inplace_info_t>>(
                        attr_keys::tensor_inplace_hint);
            })
            .filter([](std::vector<temp_tensor_inplace_info_t> *v) {
                return v->size() == 1UL
                        && (*v)[0].kind_ == inplace_kind::ZERO_OFFSET;
            })
            .flat_map([&identity_to_tensor](
                              std::vector<temp_tensor_inplace_info_t> *v) {
                return utils::find_map_value(
                        identity_to_tensor, (*v)[0].to_reuse_.get());
            })
            .map([](const expr_c *v) { return v->as<tensor_c>().get(); })
            .filter([base](const tensor_node *v) {
                return base->elem_dtype_ == v->elem_dtype_
                        && get_const_as_int(
                                   base->dims_.at(0).static_as<constant>())
                        >= get_const_as_int(
                                v->dims_.at(0).static_as<constant>());
            })
            .get_or_else(nullptr);
}

static std::vector<size_t> schedule_tensor_memory_planner(
        std::unordered_map<expr_c, tensor_tick_info_t> &ticks,
        const std::unordered_map<alias_info::tensor_alias_identity_t *, expr_c>
                &identity_to_tensor,
        std::vector<expr_c> &defined_tensors,
        // the old_tensor -> (scope id, start offset)
        std::unordered_map<expr_c, std::pair<uint64_t, size_t>> &replace_map,
        std::unordered_map<expr_c, expr> &replace_map_in_output, bool hot_first,
        uint64_t scopes, bool inplace) {
    std::vector<size_t> total_list(scopes, 0);
    // tick->trace map, outer map for different parallel scope.
    std::vector<std::multimap<int64_t, memory_optim::memory_alloc_trace_t>>
            traces(scopes);
    std::vector<std::pair<const expr_c, tensor_tick_info_t> *> sorted_ticks;
    sorted_ticks.reserve(ticks.size());
    for (auto &itr : ticks) {
        sorted_ticks.push_back(&itr);
    }

    memory_optim::inplace_info_map inplace_map;
    // unordered ticks may cause unordered multimap
    std::sort(sorted_ticks.begin(), sorted_ticks.end(),
            [](const std::pair<const expr_c, tensor_tick_info_t> *x,
                    const std::pair<const expr_c, tensor_tick_info_t> *y) {
                if (x->second.real_first_access_
                        != y->second.real_first_access_) {
                    return x->second.real_first_access_
                            < y->second.real_first_access_;
                }
                return x->first.checked_as<tensor>()->name_
                        < y->first.checked_as<tensor>()->name_;
            });
    for (auto &itr_ptr : sorted_ticks) {
        auto &itr = *itr_ptr;
        auto tsr = itr.first.checked_as<tensor>();
        if (itr.second.is_arg_) {
            auto inplace_parent = get_inplace_target_if_single_inplace(
                    identity_to_tensor, tsr.get(), tsr.get());
            while (inplace_parent) {
                if (auto pinfo = utils::find_map_value(
                            ticks, inplace_parent->node_ptr_from_this())
                                         .get_or_else(nullptr)) {
                    bool need_remove = false;
                    int64_t lastread = itr.second.last_read_;
                    if (!check_if_tensor_valid(itr, need_remove)) {
                        if (need_remove) {
                            replace_map[itr.first]
                                    = std::make_pair(itr.second.scope_,
                                            std::numeric_limits<size_t>::max());
                        }
                    } else {
                        SC_MODULE_INFO << "In-place reusing output buffer"
                                       << inplace_parent << " -> " << tsr;
                        pinfo->is_arg_ = true;
                        replace_map_in_output[inplace_parent
                                                      ->node_ptr_from_this()]
                                = builder::tensor_ptr(tsr, {UINT64_C(0)});
                    }
                    inplace_parent = get_inplace_target_if_single_inplace(
                            identity_to_tensor, inplace_parent, tsr.get());
                } else {
                    inplace_parent = nullptr;
                }
            }
        }
    }
    for (auto &itr_ptr : sorted_ticks) {
        auto &itr = *itr_ptr;
        auto tsr = itr.first.checked_as<tensor>();
        if (itr.second.is_arg_ || itr.second.is_already_scheduled()) {
            // skip arg tensor for now
            // skip already scheduled tensors
            continue;
        }
        bool need_remove = false;
        int64_t lastread = itr.second.last_read_;
        if (!check_if_tensor_valid(itr, need_remove)) {
            if (need_remove) {
                replace_map[itr.first] = std::make_pair(
                        itr.second.scope_, std::numeric_limits<size_t>::max());
            }
            continue;
        }
        if (lastread == COMPLICATED_ACCESS) { continue; }

        assert(tsr->dims_.size() == 1);
        // all tensors in last_read_tensor are ensured to have const
        // shape
        int64_t tsr_size = utils::get_sizeof_type(tsr->elem_dtype_)
                * get_const_as_int(tsr->dims_[0].static_as<constant_c>());
        traces[itr.second.scope_].insert(
                std::make_pair(itr.second.first_access_ * 2,
                        memory_optim::memory_alloc_trace_t {
                                (uintptr_t)tsr.get(), (size_t)tsr_size}));
        auto last_access = itr.second.get_last_access();
        // make sure last access > first_access_
        traces[itr.second.scope_].insert(std::make_pair(last_access * 2 + 1,
                memory_optim::memory_alloc_trace_t {(uintptr_t)tsr.get(), 0}));
        if (inplace) {
            std::unordered_map<uintptr_t, inplace_kind> inplace_tsrs;
            for (auto &inp_tsr : itr.second.inplace_reuse_) {
                // check if the tensor in the inplace hint is the last use
                if (ticks[inp_tsr.first].get_last_access()
                        == itr.second.first_access_) {
                    inplace_tsrs.insert(std::make_pair(
                            (uintptr_t)inp_tsr.first.get(), inp_tsr.second));
                }
            }
            if (tsr->attr_) {
                if (auto inplace_hints
                        = tsr->attr_->get_or_null<
                                std::vector<temp_tensor_inplace_info_t>>(
                                attr_keys::tensor_inplace_hint)) {
                    for (auto &hint : *inplace_hints) {
                        auto hint_itr
                                = identity_to_tensor.find(hint.to_reuse_.get());
                        if (hint_itr == identity_to_tensor.end()) {
                            SC_MODULE_WARN
                                    << "Bad temp tensor in-place identity:"
                                    << tsr;
                        } else {
                            auto &victim = hint_itr->second;
                            auto lastuse = ticks[victim].get_last_access();

                            inplace_tsrs.insert(std::make_pair(
                                    (uintptr_t)victim.get(), hint.kind_));
                        }
                    }
                }
            }
            if (!inplace_tsrs.empty()) {
                inplace_map[(uintptr_t)tsr.get()]
                        = std::vector<std::pair<uintptr_t, inplace_kind>>(
                                inplace_tsrs.begin(), inplace_tsrs.end());
            }
        }
    }

    for (size_t i = 0; i < traces.size(); i++) {
        auto &trace = traces[i];
        std::vector<memory_optim::memory_alloc_trace_t> in_trace;
        in_trace.reserve(trace.size());
        for (auto &kv : trace) {
            in_trace.emplace_back(kv.second);
        }

        std::unordered_map<uintptr_t, size_t> out;

        std::unordered_map<uintptr_t, std::vector<uintptr_t>>
                out_inplace_result;
        total_list[i] = memory_optim::schedule_memory_allocations(in_trace,
                /*512-bit alignment*/ 64, hot_first, inplace_map, out,
                out_inplace_result);
        for (auto &kv : out) {
            auto p = reinterpret_cast<expr_base *>(kv.first)
                             ->node_ptr_from_this();
            replace_map[p] = std::make_pair(i, kv.second);
            SC_MODULE_INFO << p << " -> " << kv.second;
        }
        // set alias info to the callee function
        for (auto &kv : out_inplace_result) {
            auto p = reinterpret_cast<expr_base *>(kv.first)
                             ->node_ptr_from_this();
            // p is a tensor which inplace reuses others
            if (kv.second.empty()) { continue; }
            auto &tinfo = ticks[p];
            const auto &callsite = tinfo.inplace_call_site_;
            // update the alias info. If a tensor inplace reuses another one in
            // func parameters, we need to inform the function implementation
            // that there is pointer alias in arguments
            if (callsite) {
                expr arg;
                auto aliasinfo = std::make_shared<alias_info::alias_set_t>();
                // first step, find the first usage of tensor p in the call
                for (size_t pidx = 0; pidx < callsite->tensors_passed_.size();
                        pidx++) {
                    if (callsite->tensors_passed_[pidx].ptr_same(p)) {
                        auto cur_arg = callsite->func_->params_.at(pidx);
                        auto cur_aliasinfo
                                = alias_info::get_or_create_alias_info(
                                        *cur_arg);
                        if (!arg.defined()) { arg = cur_arg; }
                        cur_aliasinfo->add_to_clique(aliasinfo);
                    }
                }
                assert(arg.defined());
                // second, find all func args that may have alias
                for (size_t pidx = 0; pidx < callsite->tensors_passed_.size();
                        pidx++) {
                    for (auto inplace_id : kv.second) {
                        auto inp = reinterpret_cast<expr_base *>(inplace_id);
                        if (callsite->tensors_passed_[pidx].get() == inp) {
                            auto cur_arg = callsite->func_->params_.at(pidx);
                            auto cur_aliasinfo
                                    = alias_info::get_or_create_alias_info(
                                            *cur_arg);
                            cur_aliasinfo->add_to_clique(aliasinfo);
                        }
                    }
                }
            }
        }
        SC_MODULE_INFO << "Scope: " << i << ",Total: " << total_list[i];
    }
    return total_list;
}

class buffer_replacer_t : public ir_visitor_t {
public:
    std::unordered_map<expr_c, expr_c> &replace_map_;
    std::unordered_map<expr_c, expr_c> &extend_map_;

    buffer_replacer_t(std::unordered_map<expr_c, expr_c> &replace_map,
            std::unordered_map<expr_c, expr_c> &extend_map)
        : replace_map_(replace_map), extend_map_(extend_map) {}

    stmt_c visit(define_c v) override {
        if (v->var_.isa<tensor>()) {
            auto itr = replace_map_.find(v->var_);
            if (itr != replace_map_.end()) {
                // the tensor is completely replaced
                return builder::make_stmts_unattached({});
            }
            itr = extend_map_.find(v->var_);
            if (itr != extend_map_.end()) {
                // the tensor is extended
                return copy_attr(*v,
                        builder::make_var_tensor_def_unattached(
                                itr->second, v->linkage_, v->init_));
            }
            return std::move(v);
        } else {
            return ir_visitor_t::visit(v);
        }
    }

    expr_c visit(tensor_c v) override {
        auto itr = replace_map_.find(v);
        if (itr != replace_map_.end()) {
            // the tensor is completely replaced
            return itr->second;
        }
        itr = extend_map_.find(v);
        if (itr != extend_map_.end()) {
            // the tensor is extended
            return itr->second;
        }
        return std::move(v);
    }
};

class buffer_replacer_memory_planner_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    // tsr -> (scope id, start offset)
    std::unordered_map<expr_c, std::pair<uint64_t, size_t>> &replace_map_;
    std::unordered_map<expr_c, expr> &replace_map_in_output_;
    // top-level stmts -> scope id
    std::unordered_map<stmt_c, uint64_t> &stmts_to_scope_id_;
    std::vector<size_t> total_list_;
    std::vector<expr> base_list_;
    buffer_replacer_memory_planner_t(
            std::unordered_map<expr_c, std::pair<uint64_t, size_t>>
                    &replace_map,
            std::unordered_map<expr_c, expr> &replace_map_in_output,
            std::unordered_map<stmt_c, uint64_t> &stmts_to_scope_id,
            const std::vector<size_t> &total_list)
        : replace_map_(replace_map)
        , replace_map_in_output_(replace_map_in_output)
        , stmts_to_scope_id_(stmts_to_scope_id)
        , total_list_(total_list)
        , base_list_(std::vector<expr>(total_list.size())) {}

    stmt_c visit(define_c v) override {
        if (v->var_.isa<tensor>() && !(v->init_.defined())) {
            auto itr = replace_map_.find(v->var_);
            if (itr != replace_map_.end()) {
                auto cur_scope = itr->second.first;
                // if need_remove
                if (itr->second.second == std::numeric_limits<size_t>::max()) {
                    return builder::make_stmts_unattached({});
                }
                assert(base_list_[cur_scope].defined());
                return copy_attr(*v,
                        builder::make_var_tensor_def_unattached(v->var_,
                                v->linkage_,
                                builder::tensor_ptr(base_list_[cur_scope],
                                        {itr->second.second})));
            }
            auto itr2 = replace_map_in_output_.find(v->var_);
            if (itr2 != replace_map_in_output_.end()) {
                return copy_attr(*v,
                        builder::make_var_tensor_def_unattached(
                                v->var_, v->linkage_, itr2->second));
            }
            return std::move(v);
        } else {
            return ir_visitor_t::visit(v);
        }
    }

    stmt_c visit(stmts_c v) override {
        auto stmts_to_scope_itr = stmts_to_scope_id_.find(v);
        bool is_top_level = stmts_to_scope_itr != stmts_to_scope_id_.end();
        if (is_top_level) {
            auto cur_scope = stmts_to_scope_itr->second;
            if (total_list_[cur_scope]) {
                auto rescheduled_tsr = builder::make_tensor(
                        "__rescheduled_" + std::to_string(cur_scope),
                        {total_list_[cur_scope]}, datatypes::s8);
                rescheduled_tsr->attr()[attr_keys::can_be_scheduled] = true;
                base_list_[cur_scope] = rescheduled_tsr;
            }
        }
        auto ret = ir_visitor_t::visit(v);
        if (is_top_level) {
            auto cur_scope = stmts_to_scope_itr->second;
            if (total_list_[cur_scope]) {
                auto newret = ret.checked_as<stmts>()->seq_;
                newret.insert(newret.begin(),
                        builder::make_var_tensor_def_unattached(
                                base_list_[cur_scope]));
                return make_stmt<stmts_node_t>(std::move(newret));
            }
        }
        return ret;
    }

    func_c dispatch(func_c v) override {
        auto ret = ir_visitor_t::dispatch(v);
        if (ret != v) {
            std::const_pointer_cast<func_base>(ret)
                    ->attr()[attr_keys::already_buf_sched]
                    = true;
        }
        return ret;
    }
};

template <typename T1>
static T1 run(const context_ptr &ctx, T1 f, const func_c &toplevel,
        bool remove_dead, bool inplace) {
    int type = ctx->flags_.buffer_schedule_;
    if (f->attr_) {
        if (f->attr_->has_key(attr_keys::buf_sched_type)) {
            int localtype
                    = f->attr_->template get<int>(attr_keys::buf_sched_type);
            if (localtype < 0 || localtype > 3) {
                SC_MODULE_WARN
                        << "The attr pass.buf_sched_type should be >0 and <3";
            } else {
                type = localtype;
            }
        }
        if (f->attr_->get_or_else(attr_keys::already_buf_sched, false)) {
            return f;
        }
    }
    if (type == attr_keys::BUF_SCHED_NONE) { return f; }
    bool aggresive = (type > 1);
    std::unordered_map<expr_c, tensor_tick_info_t> tick_out;
    std::vector<expr_c> defined;
    reference_tick_finder_t finder(tick_out, defined);
    if (toplevel) { finder.parse_func_params(toplevel->params_); }
    finder.dispatch(f);
    // if no local tensor defined, shortcut
    if (defined.empty()) { return f; }
    if (remove_dead) {
        dead_tsr_write_remover_t dtwr(finder.stmts_to_scope_id_, tick_out);
        f = dtwr.dispatch(f);
        assert(dtwr.tick_ == finder.tick_);
    }
    if (aggresive) {
        std::unordered_map<expr_c, std::pair<uint64_t, size_t>> replacer_list;
        std::unordered_map<expr_c, expr> replacer_list_output;
        auto total_list = schedule_tensor_memory_planner(tick_out,
                finder.identity_to_tensor_, defined, replacer_list,
                replacer_list_output, type == attr_keys::BUF_SCHED_HOT,
                finder.scope_id_ + 1, inplace);
        if (replacer_list.size() <= 1UL && replacer_list_output.empty()) {
            return f;
        }
        buffer_replacer_memory_planner_t rep {replacer_list,
                replacer_list_output, finder.stmts_to_scope_id_, total_list};
        return rep.dispatch(f);
    } else {
        std::unordered_map<expr_c, expr_c> replacer, extender;
        schedule_tensors(tick_out, defined, replacer, extender);
        // if not replace, shortcut
        if (replacer.empty() && extender.empty()) { return f; }
        buffer_replacer_t rep(replacer, extender);
        return rep.dispatch(f);
    }
}

func_c buffer_scheduler_t::operator()(func_c f) {
    if (f->attr_ && f->attr_->get_or_else(function_attrs::low_level, false)) {
        return f;
    }
    return run(ctx_, f, nullptr, eliminate_dead_writes_, do_inplace_opt_);
}

stmt_c buffer_scheduler_t::operator()(stmt_c f) const {
    return run(ctx_, std::move(f), top_level_, eliminate_dead_writes_,
            do_inplace_opt_);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
