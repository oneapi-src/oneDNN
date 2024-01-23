/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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
#include "dynamic_parallel_transform.hpp"
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/tir_pos_trace.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/cpu/closurize.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/util_module_passes.hpp>
#include <compiler/ir/viewer.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/config.hpp>
#include <runtime/dynamic_threadpool_c.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/assert.hpp>
#include <util/optional_find.hpp>
#include <util/weakptr_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(dynamic_parallel_transform, SC_PASS_DEPENDS_ON(),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

func_t get_dyn_threadpool_shared_buffer_func() {
    static func_t f = builder::_decl_func("sc_dyn_threadpool_shared_buffer",
            datatypes::u8.get_pointerof(), {_arg_("size", datatypes::index)});
    return f;
}

func_t get_dyn_threadpool_loop_end_func() {
    static func_t f = builder::_decl_func("sc_dyn_threadpool_loop_end",
            datatypes::index,
            {_arg_("scope", datatypes::index),
                    _arg_("size", datatypes::index)});
    return f;
}

func_t get_dyn_threadpool_submit_func() {
    static func_t f = builder::_decl_func("sc_dyn_threadpool_create_work_items",
            datatypes::index,
            {_arg_("pfunc", datatypes::pointer),
                    _arg_("iter", datatypes::pointer),
                    _arg_("iter_num", datatypes::index),
                    _arg_("loop_len", datatypes::index),
                    _arg_("num_blocks", datatypes::index),
                    _arg_("tid_hint", datatypes::index),
                    _arg_("buffer_num", datatypes::index),
                    _arg_("buffers", datatypes::pointer),
                    _arg_("flags", datatypes::index)});
    return f;
}

func_t get_dyn_threadpool_run_func() {
    static func_t f = builder::_decl_func(
            "sc_dyn_threadpool_run", datatypes::void_t, {});
    return f;
}

func_t get_dyn_threadpool_destroy_func() {
    static func_t f = builder::_decl_func(
            "sc_dyn_threadpool_sched_destroy", datatypes::void_t, {});
    return f;
}

func_t get_dyn_threadpool_init_func() {
    static func_t f = builder::_decl_func("sc_dyn_threadpool_sched_init",
            datatypes::void_t,
            {_arg_("stream", datatypes::pointer),
                    _arg_("module_data", datatypes::pointer),
                    _arg_("args", datatypes::generic.get_pointerof()),
                    _arg_("num_roots", datatypes::index),
                    _arg_("queue_size", datatypes::index),
                    _arg_("num_threads", datatypes::index)});
    return f;
}

// what a parallel-for scope defines/captures(from parent parallel-fors)
struct parallel_for_scope_t {
    std::vector<expr_c> iters_;
    std::unordered_set<expr_c> defined_;
    std::unordered_set<expr_c> captured_;
    std::vector<expr_c> ordered_captured_;
    std::weak_ptr<parallel_for_scope_t> inlined_to_;
    for_loop_c loop_;
    uint64_t nested_level_;
    // if this scope is the start of an original parallel for
    bool is_start_;
    // if this scope is the end of an original parallel for
    bool is_end_ = true;
    bool in_if_scope_ = false;
    bool in_for_scope_ = false;
    uint64_t tid_step_;

    parallel_for_scope_t(const for_loop_c &loop, uint64_t nested_level,
            bool is_start, uint64_t tid_step,
            const std::vector<expr_c> *parent_iters, const expr_c &cur_iter)
        : loop_ {loop}
        , nested_level_ {nested_level}
        , is_start_ {is_start}
        , tid_step_ {tid_step} {
        if (parent_iters) {
            iters_ = *parent_iters;
            if (cur_iter.defined()) { iters_.emplace_back(cur_iter); }
            defined_ = std::unordered_set<expr_c>(iters_.begin(), iters_.end());
        }
    }
};

struct dyn_parallel_analysis_result_t {
    // the collection of all captured var/tensor of nested parallel fors
    std::unordered_set<expr_c> captured_;
    std::vector<expr_c> ordered_captured_;

    std::vector<std::shared_ptr<parallel_for_scope_t>> pfor_chain_ {
            std::make_shared<parallel_for_scope_t>(
                    for_loop_c(), 0, true, 0, nullptr, expr_c())};
};

/**
 * this sub-pass traverses the nested parallel-for bodies to find 0)
 * parallel-for loop scopes 1) captured var/tensors in main_thread 2) captured
 * tensors in parent parallel-for scopes
 * */
class dyn_parallel_analysis_impl_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    const std::vector<std::unordered_set<expr_c>> &main_thread_defined_;
    const std::unordered_set<expr_c> &globals_;

    dyn_parallel_analysis_result_t &result_;
    tir_pos_tracer pass_error_tracer_;
    uint64_t num_threads_;

    void assert_value_is(const expr &v, int64_t val) const {
        COMPILE_ASSERT_POS(v.isa<constant>()
                        && get_const_as_int(v.static_as<constant_c>()) == val,
                "Expecting " << val << " for the boundary of the loop, got "
                             << v);
    }

    dyn_parallel_analysis_impl_t(uint64_t num_threads,
            const std::vector<std::unordered_set<expr_c>> &main_thread_defined,
            const std::unordered_set<expr_c> &globals,
            dyn_parallel_analysis_result_t &result)
        : main_thread_defined_(main_thread_defined)
        , globals_(globals)
        , result_(result)
        , num_threads_ {num_threads} {}

    expr_c dispatch(expr_c v) override {
        TIR_ERROR_TRACE(v);
        ir_viewer_t::dispatch(v);
        return v;
    }

    stmt_c dispatch(stmt_c v) override {
        TIR_ERROR_TRACE(v);
        ir_viewer_t::dispatch(v);
        return v;
    }

    bool try_capture_main_thread(const expr_c &v) {
        if (result_.captured_.count(v)) { return true; }
        for (auto &d : main_thread_defined_) {
            if (d.count(v)) {
                result_.captured_.insert(v);
                result_.ordered_captured_.emplace_back(v);
                return true;
            }
        }
        return false;
    }

    bool try_capture_in_parallel(const expr_c &v) {
        int64_t found = -1;
        for (int64_t i = result_.pfor_chain_.size() - 1; i >= 0; i--) {
            auto &cur_scope = result_.pfor_chain_[i];
            if (cur_scope->captured_.count(v)) {
                found = i;
                break;
            }
            if (cur_scope->defined_.count(v)) {
                found = i;
                break;
            }
        }
        if (found == -1) { return false; }
        for (uint64_t i = found + 1; i < result_.pfor_chain_.size(); i++) {
            auto &cur_scope = result_.pfor_chain_[i];
            cur_scope->captured_.insert(v);
            cur_scope->ordered_captured_.emplace_back(v);
        }
        return true;
    }

    void view(define_c v) override {
        result_.pfor_chain_.back()->defined_.insert(v->var_);
        ir_viewer_t::view(std::move(v));
    }

    void view(tensor_c v) override {
        if (!try_capture_in_parallel(v)) {
            auto defined_in_main = try_capture_main_thread(v);
            if (!defined_in_main) {
                COMPILE_ASSERT_POS(
                        globals_.count(v), "Use of undefined tensor " << v);
            }
        }
        ir_viewer_t::view(std::move(v));
    }

    void view(var_c v) override {
        // we only allow using variables in the current parallel-for scope, or
        // in main-thread captures, or in module globals
        if (!result_.pfor_chain_.back()->defined_.count(v)) {
            auto defined_in_main = try_capture_main_thread(v);
            if (!defined_in_main) {
                COMPILE_ASSERT_POS(
                        globals_.count(v), "Use of undefined variable " << v);
            }
        }
        // ir_viewer_t::view(std::move(v));
    }

    void view(if_else_c v) override {
        auto &chain_data = result_.pfor_chain_.back()->in_if_scope_;
        auto old_chain_data = chain_data;
        chain_data = true;
        ir_viewer_t::view(std::move(v));
        chain_data = old_chain_data;
    }

    void view(for_loop_c v) override {
        auto &chain_data = result_.pfor_chain_.back()->in_for_scope_;
        auto old_chain_data = chain_data;
        if (v->kind_ == for_type::NORMAL) {
            chain_data = true;
            result_.pfor_chain_.back()->defined_.insert(v->var_);
            ir_viewer_t::view(v);
            result_.pfor_chain_.back()->defined_.erase(v->var_);
        } else if (v->kind_ == for_type::PARALLEL) {
            COMPILE_ASSERT_POS(
                    !chain_data && !result_.pfor_chain_.back()->in_if_scope_,
                    "Nested parallel cannot be in if or normal for");
            assert_value_is(v->iter_begin_, 0);
            assert_value_is(v->step_, 1);
            ir_viewer_t::dispatch(v->iter_begin_);
            ir_viewer_t::dispatch(v->iter_end_);
            ir_viewer_t::dispatch(v->step_);

            auto current_scope = result_.pfor_chain_.back();
            // the current scope has another parallel-for, so it is not the end
            current_scope->is_end_ = false;
            auto old_num_threads = num_threads_;
            uint64_t thread_id_step = 1;
            if (v->num_threads_ > 0) {
                thread_id_step = old_num_threads / v->num_threads_;
                num_threads_ = thread_id_step;
            }
            // dispatch body in a new scope
            auto ptr = std::make_shared<parallel_for_scope_t>(v,
                    current_scope->nested_level_ + 1, true, thread_id_step,
                    &current_scope->iters_, v->var_);
            COMPILE_ASSERT(ptr->nested_level_ <= 6,
                    "At most 6 nested parallel loops is supported");
            result_.pfor_chain_.emplace_back(ptr);
            // dispatch into the loop body
            ir_viewer_t::dispatch(v->body_);
            num_threads_ = old_num_threads;
            // push a new scope for the remainder of the current pfor. is_start
            // should be false
            auto new_scope = std::make_shared<parallel_for_scope_t>(
                    current_scope->loop_, current_scope->nested_level_, false,
                    current_scope->tid_step_, &current_scope->iters_, expr_c());
            result_.pfor_chain_.emplace_back(new_scope);
        } else {
            throw std::runtime_error("Bad loop type");
        }
        chain_data = old_chain_data;
    }
};

// transform the nested parallel for scope bodies into functions
static int transform_scopes_to_funcs(const std::string &name,
        const stmts_c &body, const dyn_parallel_analysis_result_t &result,
        int idx_in_result, const std::unordered_set<expr_c> &globals,
        std::vector<func_t> &out_funcs) {
    auto current_scope = result.pfor_chain_.at(idx_in_result);
    COMPILE_ASSERT(body.get() == current_scope->loop_->body_.get(),
            "Bad parallel for scope state");
    stmts new_body;
    std::unordered_map<expr_c, expr> *replace_map = nullptr;
    std::unique_ptr<ir_copier_t> cpyer;
    auto prepare_new_var_tensor
            = [&replace_map, &globals, &result, &current_scope]() {
                  // make global vars unchanged
                  for (auto &k : globals) {
                      (*replace_map)[k] = k.remove_const();
                  }
                  std::vector<expr> new_args;
                  // remake iterators
                  for (auto &k : current_scope->iters_) {
                      new_args.emplace_back(k->remake());
                      (*replace_map)[k] = new_args.back();
                  }
                  // remake thread captured
                  for (auto &k : current_scope->ordered_captured_) {
                      new_args.emplace_back(k->remake());
                      (*replace_map)[k] = new_args.back();
                  }
                  // remake main-thread captured
                  for (auto &k : result.ordered_captured_) {
                      new_args.emplace_back(k->remake());
                      (*replace_map)[k] = new_args.back();
                  }
                  return new_args;
              };
    auto prepare_new_out_func = [&replace_map, &cpyer, &idx_in_result, &name,
                                        &new_body, &prepare_new_var_tensor,
                                        &out_funcs]() {
        new_body = builder::make_stmts_unattached({}).static_as<stmts>();
        new_body->temp_data() = std::unordered_map<expr_c, expr>();
        replace_map = &(
                new_body->temp_data().get<std::unordered_map<expr_c, expr>>());
        cpyer = utils::make_unique<ir_copier_t>(*replace_map, true);
        std::vector<expr> new_args = prepare_new_var_tensor();
        auto ret = builder::make_func(
                name + "_0_closure_N" + std::to_string(idx_in_result), new_args,
                new_body, datatypes::void_t);
        ret->attr()[function_attrs::private_] = true;
        ret->attr()[function_attrs::no_parallel] = true;
        ret->decl_->attr()[function_attrs::private_] = true;
        ret->decl_->attr()[function_attrs::no_parallel] = true;
        out_funcs.emplace_back(ret);
        return ret;
    };
    auto current_new_func = prepare_new_out_func();
    for (auto &s : body->seq_) {
        COMPILE_ASSERT(!s.isa<stmts>(),
                "Expecting nested stmts to be flattened, got " << body);
        auto captured_def
                = s.cast<define>()
                          .flat_map([](const define &v) {
                              return v->var_.cast<tensor>();
                          })
                          .filter([&](const tensor &v) {
                              return (uint64_t)idx_in_result + 1
                                      < result.pfor_chain_.size()
                                      && result.pfor_chain_[idx_in_result + 1]
                                                 ->captured_.count(v);
                          });
        if (captured_def.has_value()) {
            // if "s" is a define node and the tensor is captured by the next
            // scope, need special allocation for this tensor
            auto def = s.static_as<define>();
            COMPILE_ASSERT(!def->init_.defined(),
                    "Captured tensors should not have init pointer" << def);
            auto newdef = (*cpyer)(s).static_as<define>();
            auto old_tsr = captured_def.get();
            newdef->init_ = builder::make_call(
                    get_dyn_threadpool_shared_buffer_func(),
                    {do_cast_and_fold((*cpyer)(old_tsr->dims_.at(0))
                            * utils::get_sizeof_type(old_tsr->elem_dtype_))});
            new_body->seq_.emplace_back(newdef);
            continue;
        }

        auto loop = s.cast<for_loop_c>()
                            .filter([](const for_loop_c &l) {
                                return l->kind_ == for_type::PARALLEL;
                            })
                            .get_or_else(for_loop_c());
        if (!loop.defined()) {
            // if it is not a parallel-for
            new_body->seq_.emplace_back((*cpyer)(s).remove_const());
            continue;
        }
        // "s" is a parallel for
        idx_in_result = transform_scopes_to_funcs(name, loop->body_.as<stmts>(),
                                result, idx_in_result + 1, globals, out_funcs)
                + 1;
        // switch the local states to the new scope
        current_scope = result.pfor_chain_.at(idx_in_result);
        COMPILE_ASSERT(body.get() == current_scope->loop_->body_.get(),
                "Bad parallel for scope state");
        current_new_func = prepare_new_out_func();
    }
    return idx_in_result;
}

static void finalize_body_funcs(const dyn_parallel_analysis_result_t &result,
        std::vector<func_t> &body_funcs, std::vector<func_t> &out_funcs) {
    assert(body_funcs.size() == result.pfor_chain_.size() - 1);
    // generate place holders for wrappers
    out_funcs.reserve(body_funcs.size());
    for (size_t i = 0; i < body_funcs.size(); i++) {
        auto &scope = result.pfor_chain_.at(i + 1);
        uint64_t depth = scope->nested_level_;
        uint64_t num_buffers = scope->ordered_captured_.size();
        uint64_t num_args = result.ordered_captured_.size();
        auto func = builder::make_func(body_funcs[i]->name_ + "_wrapper",
                {builder::make_tensor("itr", {depth}, datatypes::index),
                        builder::make_tensor(
                                "buffers", {num_buffers}, datatypes::generic),
                        builder::make_tensor(
                                "args", {num_args}, datatypes::generic)},
                builder::make_stmts_unattached({}), datatypes::void_t);
        func->attr()[function_attrs::private_] = true;
        func->decl_->attr()[function_attrs::private_] = true;
        out_funcs.emplace_back(func);
    }
    auto find_parent_scope
            = [&body_funcs, &result](const for_loop_c &v,
                      size_t idx_in_body_funcs, size_t &out_idx_in_body_funcs) {
                  uint64_t diff = 0;
                  for (int64_t i = (int64_t)idx_in_body_funcs; i >= 0; i--) {
                      // skip dummy droped for loop bodies
                      if (!body_funcs[i]) { continue; }
                      auto &scope = result.pfor_chain_.at(i + 1);
                      if (scope->is_start_ && v.ptr_same(scope->loop_)) {
                          out_idx_in_body_funcs = i;
                          return diff;
                      }
                      if (utils::is_uninitialized_weakptr(scope->inlined_to_)) {
                          // if the scope is inlined to another, don't need to
                          // add to the diff because it is not visible by the
                          // runtime
                          diff++;
                      }
                  }
                  throw std::runtime_error("Bad dyn_parallel_analysis_result");
              };
    for (size_t i = 0; i < body_funcs.size(); i++) {
        auto &cur_func = body_funcs[i];
        if (!cur_func) {
            out_funcs[i] = nullptr;
            continue;
        }
        auto &cur_func_body = cur_func->body_.static_as<stmts>()->seq_;
        std::vector<stmt> *cur_body = &cur_func->body_.static_as<stmts>()->seq_;
        auto &cur_scope = result.pfor_chain_.at(i + 1);
        // generate code for 1) count down barrier 2) submit the next loop
        // workload

        /*
        1) if the current scope is at the end of a parallel-for, count down
        barrier like:
        u64 cur_scope_1 = sc_dyn_threadpool_loop_end(nullptr, 0);
        if(cur_scope_1) {
            sc_dyn_threadpool_create_work_items(...);
        }
        2) an optimization, if there are consecutive multiple empty scopes that
        ends a parallel-for after the current scope, fold them in a nested "if"
        TIR:
        pfor(t) {
            pfor(i) {
                pfor(j) {
                    pfor(k) {
                        code();
                    } // <<<< loop_end here
                } // <<<< an empty scope and loop_end here
            } // <<<< an empty scope and loop_end here
            code2() // non empty scope
        }
        The tail of loop k can be transformed to:
        // count down the barrier for loop k
        u64 cur_scope_1 = sc_dyn_threadpool_loop_end(0, 0);
        // if the current thread is the last one on loop k
        if(cur_scope_1) {
            // go up "1" parent scope and count down the barrier for loop j
            u64 cur_scope_2 = sc_dyn_threadpool_loop_end(cur_scope_1, 1);
            // if the current thread is the last one on loop j
            if (cur_scope_2) {
                // go up "1" parent scope and count down the barrier for loop i
                u64 cur_scope_3 = sc_dyn_threadpool_loop_end(cur_scope_2, 1);
                // if the current thread is the last one on loop j
                if(cur_scope_3) {
                    // submit workload for "code2"
                    sc_dyn_threadpool_create_work_items(...);
                }
            }
        }
        */
        auto cur_tail_scope = cur_scope;
        int scope_count = 0;
        expr current_scope_handle = UINT64_C(0);
        uint64_t last_parent_scope_idx = i;
        auto cur_tail_scope_idx = i + 1;
        bool has_barrier = false;
        while (cur_tail_scope->is_end_) {
            if (cur_tail_scope != cur_scope) {
                // if the scope's body is not empty, we cannot optimize it.
                // break.
                auto &cur_scope_func = body_funcs.at(cur_tail_scope_idx - 1);
                if (!cur_scope_func->body_.static_as<stmts>()->seq_.empty()) {
                    break;
                }
                // remove the empty scopy body
                cur_scope_func->temp_data_ = nullptr;
                cur_scope_func = nullptr;
            }
            has_barrier = true;
            auto scope_ret = builder::make_var(datatypes::index,
                    "__scope_handle_" + std::to_string(scope_count));
            scope_count++;
            uint64_t new_scope = 0;
            auto scope_diff = find_parent_scope(
                    cur_tail_scope->loop_, last_parent_scope_idx, new_scope);
            last_parent_scope_idx = new_scope;
            cur_body->emplace_back(builder::make_var_tensor_def_unattached(
                    scope_ret, linkage::local,
                    get_dyn_threadpool_loop_end_func()(
                            current_scope_handle, scope_diff)));
            auto if_body
                    = builder::make_stmts_unattached({}).static_as<stmts>();
            cur_body->emplace_back(builder::make_if_else_unattached(
                    scope_ret != UINT64_C(0), if_body, stmt_c()));
            cur_body = &if_body->seq_;
            // now check the next scope
            cur_tail_scope_idx++;
            if (cur_tail_scope_idx >= result.pfor_chain_.size()) { break; }
            cur_tail_scope = result.pfor_chain_[cur_tail_scope_idx];
            current_scope_handle = scope_ret;
        }
        // now submit the next scope's body
        // first find the next body to submit
        size_t next_func_idx = 0;
        for (size_t j = i + 1; j < body_funcs.size(); j++) {
            if (body_funcs[j]) {
                next_func_idx = j;
                break;
            }
        }
        if (!next_func_idx) {
            if (has_barrier) {
                // if the current scope is the last parallel-for body
                // remove the barrier code (if any)
                // delele the var definition
                cur_func_body.pop_back();
                // delele the "if"
                cur_func_body.pop_back();
            }
            body_funcs[i]->temp_data_ = nullptr;
            // no function to submit, skip
            continue;
        }
        // add the job submission code in cur_body
        const auto &next_scope_info = result.pfor_chain_.at(next_func_idx + 1);
        expr loop_len;
        uint64_t num_iter = 0;
        uint64_t num_threads = 1;
        uint64_t tid_step = next_scope_info->tid_step_;
        auto &replace = body_funcs[i]
                                ->body_->temp_data()
                                .get<std::unordered_map<expr_c, expr>>();
        if (next_scope_info->is_start_) {
            num_iter = next_scope_info->nested_level_ - 1;
            num_threads
                    = std::max((uint64_t)next_scope_info->loop_->num_threads_,
                            UINT64_C(1));
            // if the next scope is the begining of a loop
            ir_copier_t cpy {replace, true};
            loop_len = cpy(next_scope_info->loop_->iter_end_).remove_const();
        } else {
            // if the next scope returns to a parent loop
            num_iter = next_scope_info->nested_level_;
            loop_len = UINT64_C(1);
            auto &nextfunc = body_funcs.at(next_func_idx);
            if (nextfunc->body_.static_as<stmts>()->seq_.empty()) {
                /* an optimization: if the next scope is empty, directly call
                   the body:
                   pfor() {
                    pfor() {
                        // current scope
                    }
                    // <<< next scope here. It only submits the next pfor
                    pfor() {

                    }
                   }
                   */
                std::vector<expr> args;
                for (auto &itr : next_scope_info->iters_) {
                    args.emplace_back(
                            *utils::find_map_value(replace, itr).get());
                }
                for (auto &itr : next_scope_info->ordered_captured_) {
                    args.emplace_back(
                            *utils::find_map_value(replace, itr).get());
                }
                for (auto &itr : result.ordered_captured_) {
                    args.emplace_back(
                            *utils::find_map_value(replace, itr).get());
                }
                auto thecall = builder::make_call(nextfunc->decl_, args);
                thecall->attr()["inline_level"] = 2;
                cur_body->emplace_back(
                        builder::make_evaluate_unattached(thecall));
                assert(utils::is_uninitialized_weakptr(
                        next_scope_info->inlined_to_));
                next_scope_info->inlined_to_ = cur_scope;
                continue;
            }
        }
        uint64_t num_shared_buffers = next_scope_info->ordered_captured_.size();
        expr buffers;
        if (num_shared_buffers) {
            // an optimization, if the current scope has the same share buffers
            // of the next scope, just pass nullptr
            std::vector<expr_c> *cur_buffers = &cur_scope->ordered_captured_;
            if (!utils::is_uninitialized_weakptr(cur_scope->inlined_to_)) {
                auto scope = cur_scope->inlined_to_.lock();
                assert(scope);
                cur_buffers = &scope->ordered_captured_;
            }
            bool fast_path = false;
            auto &next_buffers = next_scope_info->ordered_captured_;
            // check that every buffer in next buffer is in the current buffers
            if (cur_buffers->size() >= next_buffers.size()) {
                fast_path = true;
                for (size_t bid = 0; bid < next_buffers.size(); bid++) {
                    if (!next_buffers[bid].ptr_same((*cur_buffers)[bid])) {
                        fast_path = false;
                        break;
                    }
                }
            }
            if (fast_path) {
                buffers = get_ir_null();
            } else {
                buffers = builder::make_tensor("_shared_buf",
                        {num_shared_buffers}, datatypes::generic);
                cur_body->emplace_back(
                        builder::make_var_tensor_def_unattached(buffers));
                for (size_t idx = 0; idx < next_buffers.size(); idx++) {
                    auto &buf = next_buffers[idx];
                    auto itr = replace.find(buf);
                    COMPILE_ASSERT(itr != replace.end(),
                            "Bad capture state, cannot find the captured "
                            "buffer");
                    cur_body->emplace_back(
                            builder::make_assign_unattached(buffers[idx],
                                    builder::make_cast(
                                            datatypes::generic, itr->second)));
                }
                buffers = builder::make_cast(datatypes::pointer, buffers);
            }
        } else {
            buffers = get_ir_null();
        }
        // clear temp data
        body_funcs[i]->temp_data_ = nullptr;
        cur_body->emplace_back(builder::make_evaluate_unattached(
                get_dyn_threadpool_submit_func()(
                        builder::make_func_addr(
                                out_funcs[next_func_idx]->decl_),
                        get_ir_null(), num_iter, loop_len, num_threads,
                        /*fix-me: (yijie) outer_loop_hash*/ UINT64_C(0),
                        num_shared_buffers, buffers,
                        /*flags*/ tid_step
                                | runtime::dynamic_threadpool::work_item_flags::
                                        bind_last_level)));
    }
}

// finally, fill in the body of the placeholders for wrappers
static void generate_wrappers(const dyn_parallel_analysis_result_t &result,
        std::vector<func_t> &body_funcs, std::vector<func_t> &wrapper_funcs) {
    for (size_t i = 0; i < wrapper_funcs.size(); i++) {
        if (!wrapper_funcs[i]) { continue; }
        auto &wrapper = wrapper_funcs[i];
        auto &bodyfunc = body_funcs[i];
        auto &current_scope = result.pfor_chain_.at(i + 1);
        // args of wrapper:
        // uint64_t *itr, generic_val* shared_buffers, generic_val *args
        auto &body = wrapper->body_.static_as<stmts>()->seq_;
        auto &itr = wrapper->params_[0];
        auto &shared_buffers = wrapper->params_[1];
        auto &args = wrapper->params_[2];
        std::vector<expr> funcargs;
        for (uint64_t j = 0; j < current_scope->iters_.size(); j++) {
            funcargs.emplace_back(itr[j]);
        }
        for (uint64_t j = 0; j < current_scope->ordered_captured_.size(); j++) {
            auto dtype = current_scope->ordered_captured_[j]->dtype_;
            funcargs.emplace_back(builder::make_cast(dtype, shared_buffers[j]));
        }
        for (uint64_t j = 0; j < result.ordered_captured_.size(); j++) {
            auto dtype = result.ordered_captured_[j]->dtype_;
            funcargs.emplace_back(builder::make_cast(dtype, args[j]));
        }
        body.emplace_back(builder::make_evaluate_unattached(
                builder::make_call(bodyfunc->decl_, funcargs)));
    }
}

class dyn_parallel_transform_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    bool always_transform_;
    std::vector<std::unordered_set<expr_c>> defined_;
    ir_module_ptr mod_;
    std::unordered_set<expr_c> globals_;
    func_c cur_func_;
    int capture_count_ = 0;
    int num_threads_;

    dyn_parallel_transform_impl_t(
            bool always_transform, const ir_module_ptr &mod, int num_threads)
        : always_transform_(always_transform)
        , mod_(mod)
        , num_threads_(num_threads) {
        for (auto &v : mod->get_module_vars()) {
            globals_.insert(v->var_);
        }
    }

    stmt transform_loop(for_loop_c v) {
        v = normalize_parallel_for_loop(v);
        dyn_parallel_analysis_result_t result;
        dyn_parallel_analysis_impl_t ana {
                (uint64_t)num_threads_, defined_, globals_, result};
        ana.pass_error_tracer_.cur_func_ = cur_func_.get();
        ana.view(v);
        // remove the dummy tail scope
        result.pfor_chain_.pop_back();
        std::vector<func_t> body_funcs, wrapper_funcs;
        transform_scopes_to_funcs(cur_func_->name_, v->body_.as<stmts>(),
                result, 1, globals_, body_funcs);
        finalize_body_funcs(result, body_funcs, wrapper_funcs);
        generate_wrappers(result, body_funcs, wrapper_funcs);
        // the parallel-for bodies are ready. Now try to submit the root jobs
        auto retstmts = builder::make_stmts_unattached({}).static_as<stmts>();
        auto captures = builder::make_tensor(
                "__captures" + std::to_string(capture_count_),
                {(uint64_t)result.ordered_captured_.size()},
                datatypes::generic);
        capture_count_++;
        retstmts->seq_.emplace_back(
                builder::make_var_tensor_def_unattached(captures));
        for (uint64_t i = 0; i < result.ordered_captured_.size(); i++) {
            auto &cap = result.ordered_captured_[i];
            retstmts->seq_.emplace_back(builder::make_assign_unattached(
                    captures[i], builder::make_cast(datatypes::generic, cap)));
        }
        using namespace runtime::dynamic_threadpool;
        auto null_for_stream = get_ir_null();
        null_for_stream->attr()["auto_fill_stream"] = true;
        auto null_for_module_data = get_ir_null();
        null_for_module_data->attr()["auto_fill_module_data"] = true;
        retstmts->seq_.emplace_back(builder::make_evaluate_unattached(
                get_dyn_threadpool_init_func()(null_for_stream,
                        null_for_module_data, captures, UINT64_C(1),
                        /*queue size*/ UINT64_C(256), uint64_t(num_threads_))));
        uint64_t num_threads = v->num_threads_;
        if (num_threads == 0) {
            // normal parallel for
            num_threads = num_threads_;
        }
        retstmts->seq_.emplace_back(builder::make_evaluate_unattached(
                get_dyn_threadpool_submit_func()(
                        builder::make_func_addr(wrapper_funcs.front()->decl_),
                        get_ir_null(), UINT64_C(0), v->iter_end_, num_threads,
                        /*fix-me: (yijie) outer_loop_hash*/ UINT64_C(0),
                        /*num_shared_buffers*/ UINT64_C(0), get_ir_null(),
                        /*flags*/
                        uint64_t(work_item_flags::is_root
                                | work_item_flags::bind_last_level
                                | result.pfor_chain_.at(1)->tid_step_))));
        retstmts->seq_.emplace_back(builder::make_evaluate_unattached(
                get_dyn_threadpool_run_func()()));
        retstmts->seq_.emplace_back(builder::make_evaluate_unattached(
                get_dyn_threadpool_destroy_func()()));
        for (auto &f : body_funcs) {
            if (f) { mod_->add_func({f}); }
        }
        for (auto &f : wrapper_funcs) {
            if (f) { mod_->add_func({f}); }
        }
        return retstmts;
    }

    func_c dispatch(func_c f) override {
        cur_func_ = f;
        capture_count_ = 0;
        defined_.emplace_back(std::unordered_set<expr_c>(
                f->params_.begin(), f->params_.end()));
        auto ret = ir_visitor_t::dispatch(f);
        defined_.pop_back();
        return ret;
    }

    expr_c dispatch(expr_c v) override { return v; }

    stmt_c visit(define_c v) override {
        defined_.back().insert(v->var_);
        return v;
    }

    stmt_c visit(for_loop_c v) override {
        defined_.back().insert(v->var_);
        auto ret = ir_visitor_t::visit(v);
        defined_.back().erase(v->var_);
        return ret;
    }

    stmt_c visit(stmts_c v) override {
        std::vector<stmt_c> ret;
        defined_.emplace_back();
        bool changed = false;
        for (auto &s : v->seq_) {
            auto loop = s.cast<for_loop_c>()
                                .filter([](const for_loop_c &l) {
                                    return l->kind_ == for_type::PARALLEL;
                                })
                                .get_or_else(for_loop_c());
            if (loop.defined()) {
                bool should_transform = always_transform_
                        || any_map_t::fetch_or_else(loop->attr_.get(),
                                attr_keys::dynamic_parallel, false);
                if (should_transform) {
                    changed = true;
                    ret.emplace_back(transform_loop(loop));
                } else {
                    // don't need to dispatch on statically dispatched
                    // parallel-for
                    ret.emplace_back(dispatch(s));
                }
            } else {
                auto news = dispatch(s);
                changed |= !news.ptr_same(s);
                ret.emplace_back(news);
            }
        }
        defined_.pop_back();
        if (!changed) { return v; }
        return copy_attr(*v, builder::make_stmts_unattached(ret));
    }
};

const_ir_module_ptr dynamic_parallel_transform_t::operator()(
        const_ir_module_ptr f) {
    auto ret = std::make_shared<ir_module_t>(*f);
    auto threads = runtime_config_t::get().get_num_threads();
    dyn_parallel_transform_impl_t impl {always_transform_, ret, threads};
    auto &contents = ret->get_contents();
    size_t sz = contents.size();
    for (size_t i = 0; i < sz; i++) {
        if (threads == 1) {
            contents[i] = std::const_pointer_cast<func_base>(
                    remove_parallel_on_func(contents[i]));
        } else {
            contents[i] = std::const_pointer_cast<func_base>(
                    impl.dispatch(contents[i]));
        }
    }
    ret->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL]
            = thread_pool_mode_t::DYNAMIC;
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
