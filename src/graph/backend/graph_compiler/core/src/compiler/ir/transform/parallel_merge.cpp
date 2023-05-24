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
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../util_module_passes.hpp"
#include "parallel_merge.hpp"
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/pass_id.hpp>
#include <runtime/config.hpp>
#include <runtime/trace.hpp>
#include <unordered_map>
#include <util/optional_find.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
SC_DECL_PASS_INFO(parallel_merge,
        SC_PASS_DEPENDS_ON(
                nested_parallel_flattener, parallel_workload_dispatcher),
        SC_PASS_REQUIRE_STATE(CONST_FOLDED), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

static func_base *get_callee(const stmt &s) {
    return s.cast<evaluate>()
            .map([](const evaluate &v) { return v->value_.as<call>(); })
            .map([](const call &v) {
                return dynamic_cast<func_base *>(v->func_.get());
            })
            .get_or_else(nullptr);
}

static constexpr bool allow_buffer_def_in_merge = false;
static bool is_tensor_def(const stmt &s) {
    return allow_buffer_def_in_merge
            && s.cast<define>()
                       .filter([](const define &v) {
                           if (v->var_.isa<tensor>()) {
                               if (!v->init_.defined()
                                       || v->init_.isa<tensorptr>()) {
                                   return true;
                               }
                           }
                           return false;
                       })
                       .has_value();
}

static bool is_func_ok_to_merge(func_base *f, uint64_t &num_threads) {
    // the the function call is marked no_dep_prev_op, need to check the callee
    // func body
    if (!f->body_.defined()) { return false; }
    int num_pfor = 0;
    for (auto &s : f->body_.checked_as<stmts>()->seq_) {
        if (auto func = get_callee(s)) {
            // if the statement is a call, and the callee is trace-related
            // function, we can still merge the loop
            if (any_map_t::fetch_or_else(func->attr_.get(),
                        function_attrs::is_trace_func, false)) {
                continue;
            }
            // if the callee is barrier init func, we can still merge the loop
            if (allow_buffer_def_in_merge
                    && func == builtin::get_init_barrier_func().get()) {
                continue;
            }
            // otherwise, there is complex stmt in the function. We can't
            // parallel-merge it
            return false;
        } else if (is_tensor_def(s)) {
            // if is a tensor definition
            continue;
        } else if (s.cast<returns>()
                           .filter([](const returns &v) {
                               return v->value_.isa<constant>();
                           })
                           .has_value()) {
            // if is return-const
            continue;
        } else if (s.cast<for_loop>()
                           .filter([&num_pfor, &num_threads](
                                           const for_loop &v) {
                               if (v->var_->dtype_ != datatypes::index) {
                                   return false;
                               }
                               if (v->kind_ != for_type::PARALLEL) {
                                   return false;
                               }
                               num_pfor++;
                               if (num_pfor > 1) { return false; }
                               if (!v->iter_begin_.isa<constant>()
                                       || !v->iter_end_.isa<constant>()
                                       || !v->step_.isa<constant>()) {
                                   return false;
                               }
                               auto v1 = get_const_as_int(
                                       v->iter_begin_.static_as<constant>());
                               if (v1 != 0) { return false; }
                               auto v2 = get_const_as_int(
                                       v->iter_end_.static_as<constant>());
                               if (v2 > runtime_config_t::get()
                                                .get_num_threads()) {
                                   return false;
                               }
                               auto v3 = get_const_as_int(
                                       v->step_.static_as<constant>());
                               if (v3 != 1) { return false; }
                               num_threads = v2;
                               return true;
                           })
                           .has_value()) {
            // if is parallel-for
            continue;
        } else {
            return false;
        }
    }
    return true;
}

static func_base *get_func_to_merge(const ir_module_t &mod, const stmt &s,
        uint64_t &num_threads, const std::string *&out_expected_next_op) {
    auto ret = get_callee(s);
    if (!ret) { return nullptr; }
    out_expected_next_op = any_map_t::fetch_or_null<std::string>(
            s->attr_.get(), attr_keys::no_post_barrier);
    if (!out_expected_next_op) { return nullptr; }
    ret = mod.get_func(ret->name_).get();
    return is_func_ok_to_merge(ret, num_threads) ? ret : nullptr;
}

struct func_call_numthreads_t {
    uint64_t threads_;
    func_t f_;
    call c_;
    func_call_numthreads_t(uint64_t threads, const func_t &f, const call &c)
        : threads_(threads), f_(f), c_(c) {}
};

// sort the func calls in the parallel section by the num threads needed by each
// func. Try to balance each thread with similar size of workload
static void sort_funcs(
        std::vector<func_call_numthreads_t> &funcs, uint64_t total_threads) {
    std::vector<func_call_numthreads_t *> func_ref;
    func_ref.reserve(funcs.size());
    for (auto &f : funcs) {
        func_ref.emplace_back(&f);
    }
    // sort by num threads in descending order
    std::stable_sort(func_ref.begin(), func_ref.end(),
            [](func_call_numthreads_t *v1, func_call_numthreads_t *v2) {
                return v1->threads_ > v2->threads_;
            });

    std::vector<func_call_numthreads_t> outfuncs;
    outfuncs.reserve(funcs.size());
    assert(!funcs.empty());
    // example, total_threads=16, threads per func: 10,2,3,10,3,16
    // sorted -> 16,10,10,3,3,2
    // first pick 16, finish. First round = [16]
    // pick 10, then find that 3 and 3 is ok to run in parallel. [10,3,3]
    // pick 10, 2 is ok to run in parallel (3 is already picked). [10,2]
    // Done!

    // first pick the largest num_threads
    for (size_t i = 0; i < funcs.size(); i++) {
        // if the func is already picked, skip
        if (!func_ref[i]) { continue; }
        auto f = func_ref[i];
        func_ref[i] = nullptr;
        outfuncs.emplace_back(*f);
        uint64_t remaining_threads = total_threads - f->threads_;
        if (remaining_threads == 0) { continue; }
        for (size_t j = i + 1; j < funcs.size(); j++) {
            if (!func_ref[j]) { continue; }
            if (remaining_threads >= func_ref[j]->threads_) {
                remaining_threads -= func_ref[j]->threads_;
                outfuncs.emplace_back(*func_ref[j]);
                func_ref[j] = nullptr;
            }
            if (remaining_threads == 0) { break; }
        }
    }
    funcs = std::move(outfuncs);
}

static void rename_var(const expr &v, int cnt) {
    if (v.isa<var>()) {
        v.static_as<var>()->name_ += "_";
        v.static_as<var>()->name_ += std::to_string(cnt);
    } else if (v.isa<tensor>()) {
        v.static_as<tensor>()->name_ += "_";
        v.static_as<tensor>()->name_ += std::to_string(cnt);
    }
}

static func_t merge_funcs(std::vector<func_call_numthreads_t> &funcs) {
    auto funcbody = builder::make_stmts_unattached({}).static_as<stmts>();
    auto loop_body = builder::make_stmts_unattached({});
    std::vector<stmt> &retseq = loop_body.static_as<stmts>()->seq_;
    std::vector<expr> retargs;
    std::string name = "parallel";
    uint64_t remaining_threads = runtime_config_t::get().get_num_threads();
    uint64_t total_threads = remaining_threads;
    sort_funcs(funcs, total_threads);
    bool all_private = true;
    for (auto &f : funcs) {
        name += '_';
        name += '_';
        name += f.f_->name_;
        all_private = all_private
                && any_map_t::fetch_or_else(
                        f.f_->attr_.get(), function_attrs::private_, false);
    }
    auto loop_iter = builder::make_var(datatypes::index, "merged_tid");
    bool alloc_thread_from_0 = true;
    int func_id = 0;
    for (auto &kv : funcs) {
        const auto &f = kv.f_;
        auto num_threads = kv.threads_;
        int trace_id = 0;
        bool tracing = false;

        const auto &seq = f->body_.static_as<stmts>()->seq_;
        std::unordered_map<expr_c, expr> replace_map;
        for (auto &s : seq) {
            if (auto callee = get_callee(s)) {
                if (any_map_t::fetch_or_else(callee->attr_.get(),
                            function_attrs::is_trace_func, false)) {
                    continue;
                }
                // if the callee is barrier init func, we can still merge the
                // loop
                if (allow_buffer_def_in_merge
                        && callee == builtin::get_init_barrier_func().get()) {
                    ir_copier_t cpyer {replace_map};
                    funcbody->seq_.emplace_back(cpyer(s).remove_const());
                    continue;
                }
            } else if (is_tensor_def(s)) {
                ir_copier_t cpyer {replace_map};
                funcbody->seq_.emplace_back(cpyer(s).remove_const());
                continue;
            } else if (s.cast<returns>()
                               .filter([](const returns &v) {
                                   return v->value_.isa<constant>();
                               })
                               .has_value()) {
                // if is return-const
                continue;
            } else if (s.isa<for_loop>()) {
                auto loop = s.static_as<for_loop>();
                std::vector<stmt> *target_body = &retseq;
                /*
                if the thread num of the loop is less than threads available
                we can rewrite the code to
                parallel-for(tid, 0, num_threads) {
                    ....
                    if(tid>=TID_OFFSET && tid<TID_OFFSET+THREADS1) {
                        real_tid = tid-TID_OFFSET
                        // old code depending on real_tid
                    }
                    ...
                }
                */
                uint64_t tid_offset = 0;
                if (num_threads > remaining_threads) {
                    remaining_threads = total_threads;
                    alloc_thread_from_0 = !alloc_thread_from_0;
                }
                if (num_threads <= remaining_threads
                        && num_threads != total_threads) {
                    if (alloc_thread_from_0) {
                        /*
                        |<   offset  >|<num_threads>|
                                      |< remaining_threads  >|
                        |<           total_threads          >|
                        */
                        tid_offset = total_threads - remaining_threads;
                    } else {
                        /*
                        |<offset>|<num_threads>|
                        |< remaining_threads  >|
                        |<           total_threads          >|
                        */
                        tid_offset = remaining_threads - num_threads;
                    }
                    remaining_threads -= num_threads;
                }
                if (num_threads != total_threads) {
                    auto dispath_body = builder::make_stmts_unattached({});
                    target_body = &dispath_body.static_as<stmts>()->seq_;
                    retseq.emplace_back(builder::make_if_else_unattached(
                            loop_iter >= tid_offset
                                    && loop_iter < tid_offset + num_threads,
                            dispath_body, stmt()));
                }
                if (runtime_config_t::get().trace_mode_
                        >= runtime_config_t::trace_mode_t::MULTI_THREAD) {
                    trace_id = register_traced_func(f->name_);
                    tracing = true;
                    target_body->emplace_back(builder::make_evaluate_unattached(
                            builtin::make_trace(trace_id, 0, 0)));
                }
                if (tid_offset) {
                    replace_map[loop->var_] = loop_iter - tid_offset;
                } else {
                    replace_map[loop->var_] = loop_iter;
                }
                ir_copier_t cpyer {replace_map};
                auto copied_loop_body = cpyer(loop->body_).as<stmts_c>();
                auto &copied = copied_loop_body->seq_;
                target_body->insert(
                        target_body->end(), copied.begin(), copied.end());

                if (tracing) {
                    target_body->emplace_back(builder::make_evaluate_unattached(
                            builtin::make_trace(trace_id, 1, 0)));
                }
            } else {
                throw std::runtime_error("Bad body in parallel_merge");
            }
        }

        for (auto &p : f->params_) {
            retargs.emplace_back(
                    utils::find_map_value(replace_map, p)
                            .map([](expr *v) { return *v; })
                            .get_or_else(copy_attr(*p, p->remake())));
            rename_var(retargs.back(), func_id);
        }
        func_id++;
    }
    auto forbody = builder::make_for_loop_unattached(loop_iter, UINT64_C(0),
            (uint64_t)runtime_config_t::get().get_num_threads(), UINT64_C(1),
            loop_body, true, for_type::PARALLEL);
    funcbody->seq_.emplace_back(forbody);
    auto ret = builder::make_func(name, retargs, funcbody, datatypes::void_t);
    if (all_private) { ret->attr()[function_attrs::private_] = true; }
    return ret;
}

static bool is_idle_func_setter(const stmt &s) {
    return s.cast<evaluate>()
            .map([](const evaluate &v) { return v->value_.as<intrin_call>(); })
            .filter([](const intrin_call &v) {
                return v->type_ == intrin_type::set_thread_idle_func;
            })
            .has_value();
}

const_ir_module_ptr parallel_merge_t::operator()(const_ir_module_ptr f) {
    if (runtime_config_t::get().get_num_threads() == 1) { return f; }
    auto mainf = f->get_entry_func();
    if (!mainf) { return f; }
    ir_module_ptr ret;
    auto old_seq = mainf->body_.checked_as<stmts>()->seq_;
    optional<size_t> last_idle_func_setter;
    for (size_t i = 0; i < old_seq.size(); i++) {
        auto &s = old_seq[i];
        if (!s.defined()) { continue; }
        uint64_t num_threads = 0;
        const std::string *next_expected_func = nullptr;
        if (auto callee
                = get_func_to_merge(*f, s, num_threads, next_expected_func)) {
            // look forward to find the funcs to merge
            std::vector<func_call_numthreads_t> to_merge {
                    {num_threads, callee->shared_from_this(),
                            s.static_as<evaluate>()->value_.static_as<call>()}};
            size_t insert_point = 0;
            std::vector<size_t> to_remove_idx;
            for (size_t j = i + 1; j < old_seq.size(); j++) {
                auto sj = old_seq[j];
                // we only allow define nodes between merged functions
                if (!sj.defined()) {
                    continue;
                } else if (sj.isa<define>()) {
                    continue;
                } else if (auto callee2 = get_callee(sj)) {
                    callee2 = f->get_func(callee2->name_).get();
                    bool ok_to_merge = callee2
                            && callee2->name_ == *next_expected_func
                            && is_func_ok_to_merge(callee2, num_threads);
                    if (ok_to_merge) {
                        to_merge.emplace_back(num_threads,
                                callee2->shared_from_this(),
                                sj.static_as<evaluate>()
                                        ->value_.static_as<call>());
                        old_seq[j] = stmt();
                        insert_point = j;

                        next_expected_func
                                = any_map_t::fetch_or_null<std::string>(
                                        sj->attr_.get(),
                                        attr_keys::no_post_barrier);
                        if (next_expected_func) { continue; }
                    }
                } else if (is_idle_func_setter(sj)) {
                    // remove prefetcher between merged ops
                    to_remove_idx.emplace_back(j);
                    continue;
                }
                // met unallowed stmt, break
                break;
            }
            if (to_merge.size() > 1UL) {
                if (!ret) { ret = std::make_shared<ir_module_t>(*f); }
                old_seq[i] = stmt();
                for (auto j : to_remove_idx) {
                    old_seq[j] = stmt();
                }
                auto newf = merge_funcs(to_merge);
                ret->add_func({newf});
                std::vector<expr> args;
                for (auto &ca : to_merge) {
                    args.insert(args.end(), ca.c_->args_.begin(),
                            ca.c_->args_.end());
                }
                // remove the idle func (if any), because the barrier the idle
                // func may use is very likely removed
                if (last_idle_func_setter.has_value()) {
                    old_seq[last_idle_func_setter.get()] = stmt();
                }
                old_seq[insert_point] = builder::make_evaluate_unattached(
                        builder::make_call(newf->decl_, args));
            }
            last_idle_func_setter = none_opt {};
        } else if (is_idle_func_setter(s)) {
            // record the previous idle_func_setter, so that when a op is
            // merged, we can remove this idle func
            last_idle_func_setter = i;
        } else if (get_callee(s)) {
            last_idle_func_setter = none_opt {};
        }
    }
    if (!ret) { return f; }
    std::vector<stmt_c> seq;
    for (size_t i = 0; i < old_seq.size(); i++) {
        auto &s = old_seq[i];
        if (!s.defined()) { continue; }
        seq.emplace_back(s);
    }
    auto newmain = copy_attr(*mainf,
            builder::make_func(mainf->name_, mainf->params_,
                    builder::make_stmts_unattached(seq), mainf->ret_type_));
    ret->get_contents()[ret->get_entry_func_idx()] = newmain;
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
