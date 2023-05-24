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
#include "closurize.hpp"
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/closurize_impl.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/viewer.hpp>
#include <runtime/config.hpp>
#include <runtime/thread_pool_flags.hpp>
#include <runtime/trace.hpp>
#include <unordered_map>
#include <util/any_map.hpp>
#include <util/optional.hpp>
#include <util/optional_find.hpp>
#include <util/utils.hpp>

SC_MODULE(pass.closurize)
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(closurizer_cpu,
        SC_PASS_DEPENDS_ON(nested_parallel_flattener,
                parallel_workload_dispatcher, validator, trace_inserter,
                parallel_merge, tensor_init),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

func_t get_parallel_call_with_env_func(bool managed) {
    static func_t f = builder::_decl_func("sc_parallel_call_cpu_with_env",
            datatypes::void_t,
            {_arg_("func", datatypes::pointer),
                    _arg_("flags", datatypes::index),
                    _arg_("stream", datatypes::pointer),
                    _arg_("env", datatypes::s8.get_pointerof()),
                    _arg_("begin", datatypes::index),
                    _arg_("end", datatypes::index),
                    _arg_("step", datatypes::index),
                    _arg_("args", datatypes::generic.get_pointerof())});
    static func_t f_managed
            = builder::_decl_func("sc_parallel_call_managed", datatypes::void_t,
                    {_arg_("func", datatypes::pointer),
                            _arg_("flags", datatypes::index),
                            _arg_("stream", datatypes::pointer),
                            _arg_("env", datatypes::s8.get_pointerof()),
                            _arg_("begin", datatypes::index),
                            _arg_("end", datatypes::index),
                            _arg_("step", datatypes::index),
                            _arg_("args", datatypes::generic.get_pointerof())});
    return managed ? f_managed : f;
}

// the sub-pass to collect the tensor definitions in a function. It is used in
// last-barrier safe removal to find the captured base tensors
class tensor_def_collector_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    struct tensor_def_info_t {
        expr_c init_;
        bool is_arg_;
    };
    using result_t = std::unordered_map<expr_c, tensor_def_info_t>;
    static expr_c find_base_tensor(const expr_c &v,
            const tensor_def_collector_t::result_t &scope, bool &out_is_arg) {
        expr_c cur = v;
        for (;;) {
            auto base = utils::find_map_value(scope, cur).get_or_else(nullptr);
            // global tensor
            if (!base) { return cur; }
            if (base->is_arg_) {
                // it is arg tensor
                out_is_arg = true;
                return v;
            }
            if (!base->init_.defined()) {
                // itself is a base tensor
                out_is_arg = false;
                return cur;
            }
            if (!base->init_.isa<tensor>()) {
                // base is not a tensor, failed
                out_is_arg = false;
                return expr_c();
            }
            cur = base->init_;
        }
    }

    result_t result_;

    expr_c dispatch(expr_c v) override { return v; }
    func_c dispatch(func_c v) override {
        for (auto &arg : v->params_) {
            if (arg.isa<tensor>()) {
                result_[arg] = tensor_def_info_t {expr_c(), true};
            }
        }
        return ir_viewer_t::dispatch(v);
    }

    void view(define_c v) override {
        if (v->var_.isa<tensor>()) {
            expr_c base;
            if (v->init_.defined()) {
                auto tsr = get_base_tensor_of(v->init_);
                if (tsr.defined()) {
                    base = tsr;
                } else {
                    base = v->init_;
                }
            }
            result_[v->var_] = tensor_def_info_t {base, false};
        }
    }

    // link the args of a function into the current function's tensor info. The
    // call should happen in the current func body
    void link_call_args(const call &thecall, const func_t &callee) {
        for (size_t i = 0; i < callee->params_.size(); i++) {
            auto &param = callee->params_[i];
            auto &arg = thecall->args_[i];
            if (param.isa<tensor>()) {
                result_[param]
                        = tensor_def_info_t {get_base_tensor_of(arg), false};
            }
        }
    }
};

namespace parallel_call_args {
enum { ADDR = 0, FLAGS, STREAM, MODULE_DATA, BEGIN, END, STEP, ARGS };
}

class closurize_cpu_impl_t : public closurize_impl_t {
    int rename_counter_ = 0;
    bool use_managed_thread_pool_;
    std::vector<call> out_calls;
    func_t the_last_op_;
    // we use pointers instead of reference to bypass a g++4.8 bug. If we pass
    // result_t by reference, the address will be wrong in g++4.8
    const tensor_def_collector_t::result_t *main_tensor_def_info_;
    const tensor_def_collector_t::result_t *lastop_tensor_def_info_;
    bool has_idle_func_ = false;
    // makes the closure function and its generic wrapper
    func_t make_closure_func(const std::string &name,
            std::vector<expr_c> &&params, stmt_c body,
            const std::vector<call_node::parallel_attr_t> &para_attr) override {
        bool need_trace = runtime_config_t::get().trace_mode_
                == runtime_config_t::trace_mode_t::MULTI_THREAD;
        COMPILE_ASSERT(
                para_attr.size() == 1, "CPU does not support grouped parallel");
        func_t closure
                = builder::make_func(name, params, body, datatypes::void_t);
        // make the wrapper function
        stmts seq = make_stmt<stmts_node_t>(std::vector<stmt>());
        int func_id = -1;
        if (need_trace) {
            func_id = register_traced_func(name);
            seq->seq_.insert(seq->seq_.begin(),
                    builder::make_evaluate_unattached(
                            builtin::make_trace(func_id, 0, 0)));
        }
        expr itervar = builder::make_var(datatypes::index, "i");
        expr args = builder::make_tensor(
                "args", {(int)params.size() - 1}, datatypes::generic);
        func_t ret = builder::make_func(name + "_0wrapper",
                std::vector<expr> {itervar, args}, seq, datatypes::void_t);
        assert(!modu_->get_func(ret->name_));
        assert(!modu_->get_func(closure->name_));

        closure->attr()[function_attrs::private_] = true;
        ret->attr()[function_attrs::private_] = true;
        ret->decl_->attr() = ret->attr();
        closure->decl_->attr() = closure->attr();
        std::vector<expr> fargs;
        fargs.reserve(params.size());
        // params[0] is the itervar
        fargs.emplace_back(itervar);
        for (uint64_t idx = 1; idx < params.size(); idx++) {
            auto &param = closure->params_[idx];
            assert(param->dtype_.lanes_ == 1);
            if (param->dtype_ == datatypes::generic) {
                fargs.emplace_back(args[idx - 1]);
            } else {
                fargs.emplace_back(
                        builder::make_cast(param->dtype_, args[idx - 1]));
            }
        }
        seq->seq_.emplace_back(builder::make_evaluate_unattached(
                builder::make_call(closure->decl_, fargs)));
        if (need_trace) {
            seq->seq_.emplace_back(builder::make_evaluate_unattached(
                    builtin::make_trace(func_id, 1, 0)));
        }
        modu_->add_func({closure, ret});
        return ret;
    }
    stmt make_parallel_call(func_t target, std::vector<expr> &captures,
            std::vector<call_node::parallel_attr_t> &&para_attr) override {
        // we now need to prepare the arguments.
        // 1. Allcate a generic value array
        // 2. For each captured variable except the first one, put them
        // in the array. The first capture is the itervar, we should
        // ignore it
        stmts seq = make_stmt<stmts_node_t>(std::vector<stmt>());
        std::string argname = "__tempargs";
        argname += std::to_string(rename_counter_++);
        auto argsbuf = builder::make_tensor(
                argname, {captures.size() - 1}, datatypes::generic);
        seq->seq_.emplace_back(
                builder::make_var_tensor_def_unattached(argsbuf));

        for (uint64_t argidx = 1; argidx < captures.size(); argidx++) {
            // push "argsbuf[i] = (generic)captures_i"
            seq->seq_.emplace_back(builder::make_assign_unattached(
                    argsbuf[argidx - 1],
                    builder::make_cast(datatypes::generic, captures[argidx])));
        }
        assert(para_attr.size() == 1);
        constant_folder_t folder {};
        expr_c begin_v = para_attr[0].begin_;
        cast_to(begin_v, datatypes::index, begin_v);
        begin_v = folder(begin_v);

        expr_c end_v = para_attr[0].end_;
        cast_to(end_v, datatypes::index, end_v);
        end_v = folder(end_v);

        expr_c step_v = para_attr[0].step_;
        cast_to(step_v, datatypes::index, step_v);
        step_v = folder(step_v);
        uint64_t flag = has_idle_func_
                ? runtime::thread_pool_flags::THREAD_POOL_RUN_IDLE_FUNC
                : 0;
        auto ret_call = builder::make_call(
                get_parallel_call_with_env_func(use_managed_thread_pool_),
                std::vector<expr_c> {builder::make_func_addr(std::move(target)),
                        /*flags*/ make_expr<constant_node>(flag),
                        /*stream*/
                        make_expr<constant_node>(
                                UINT64_C(0), datatypes::pointer),
                        /*env*/
                        make_expr<constant_node>(
                                UINT64_C(0), datatypes::s8.get_pointerof()),
                        begin_v, end_v, step_v, argsbuf});
        ret_call->temp_data() = captures;
        out_calls.emplace_back(ret_call.static_as<call>());
        seq->seq_.emplace_back(make_stmt<evaluate_node_t>(std::move(ret_call)));
        has_idle_func_ = false;
        return seq;
    }

public:
    using closurize_impl_t::dispatch;
    using closurize_impl_t::visit;

    expr_c visit(call_c v) override {
        if (v->func_ == builtin::get_set_idle_func_managed_func()
                && !in_parallel_for) {
            has_idle_func_ = true;
        }
        return closurize_impl_t::visit(std::move(v));
    }

    uint64_t &get_last_parallel_call_flag() {
        return out_calls.back()
                ->args_.at(parallel_call_args::FLAGS)
                .checked_as<constant>()
                ->value_.at(0)
                .u64;
    }

    optional<std::vector<expr_c>> collect_captured_base_tensors(
            const std::vector<expr> &captured) {
        std::vector<expr_c> ret;
        for (auto &v : captured) {
            if (!v.isa<tensor>()) {
                if (v.isa<var>() && v->dtype_.is_etype_pointer()) {
                    return none_opt();
                }
                continue;
            }
            if (v.static_as<tensor>()->elem_dtype_.is_pointer()) {
                return none_opt();
            }
            bool is_arg = false;
            expr_c base = tensor_def_collector_t::find_base_tensor(
                    v, *lastop_tensor_def_info_, is_arg);
            if (!base.defined()) {
                // failed to find the captured base tensor
                return none_opt();
            }
            if (is_arg) {
                base = tensor_def_collector_t::find_base_tensor(
                        base, *main_tensor_def_info_, is_arg);
                if (!base.defined()) {
                    // failed to find the captured base tensor
                    return none_opt();
                }
            }
            ret.emplace_back(base);
        }
        return ret;
    }

    func_c dispatch(func_c f) override {
        out_calls.clear();
        has_idle_func_ = false;
        auto ret = closurize_impl_t::dispatch(f);
        if (!out_calls.empty() && f->attr_
                && f->attr_->get_or_else(
                        function_attrs::has_idle_func, false)) {
            auto &the_flag = get_last_parallel_call_flag();
            the_flag |= runtime::thread_pool_flags::THREAD_POOL_RUN_IDLE_FUNC;
            the_flag |= runtime::thread_pool_flags::THREAD_POOL_DISABLE_ROLLING;
        }
        if (f == the_last_op_ && !out_calls.empty()) {
            // try to remove the last barrier
            auto &seq = f->body_.checked_as<stmts>()->seq_;
            for (auto itr = seq.rbegin(); itr != seq.rend(); ++itr) {
                auto &s = *itr;
                // allow return const after the last parallel call
                if (s.cast<returns>()
                                .filter([](const returns &v) {
                                    return v->value_.isa<constant>();
                                })
                                .has_value()) {
                    continue;
                }
                // if the statement is a parallel-for
                if (s.cast<for_loop>()
                                .filter([](const for_loop &v) {
                                    return v->kind_ == for_type::PARALLEL;
                                })
                                .has_value()) {
                    // we need to pin the captured tensors to runtime stack, we
                    // first collect the base tensors
                    auto captured_base = collect_captured_base_tensors(
                            out_calls.back()
                                    ->get_temp_data()
                                    .get<std::vector<expr>>());
                    // if the captured tensors are based on complex
                    // tensors/pointers, we can't optimize it
                    if (!captured_base.has_value()) {
                        SC_MODULE_WARN
                                << "Cannot optimize the last barrier in "
                                   "function "
                                << f->name_
                                << " because it captures complex pointers.";
                        break;
                    }
                    for (auto &base : captured_base.get()) {
                        base.remove_const()
                                ->attr()[attr_keys::runtime_stack_alloc]
                                = true;
                    }
                    auto &the_flag = get_last_parallel_call_flag();
                    the_flag |= runtime::thread_pool_flags::THREAD_POOL_EXIT;
                    // the closure args buffer must be allocated via runtime
                    // allocator instead of the native heap. So that when the
                    // main thread exits from the kernel and waiting for the
                    // worker threads, the args are still valid.
                    auto closure_arg
                            = out_calls.back()
                                      ->args_.at(parallel_call_args::ARGS)
                                      .as<tensor>();
                    COMPILE_ASSERT(closure_arg.defined(), "Bad closure arg");
                    closure_arg->attr()[attr_keys::runtime_stack_alloc] = true;
                    break;
                }

                // else, there are some complex statements after the last
                // parallel-for, do not optimize
                SC_MODULE_WARN
                        << "Cannot optimize the last barrier in function "
                        << f->name_
                        << " because there are complex code after the last "
                           "parallel-for.";
                break;
            }
        }
        return ret;
    }
    closurize_cpu_impl_t(const ir_module_ptr &m, bool use_managed_thread_pool,
            const func_t &the_last_op,
            const tensor_def_collector_t::result_t &main_tensor_def_info,
            const tensor_def_collector_t::result_t &lastop_tensor_def_info)
        : closurize_impl_t(m->get_module_vars(), m)
        , use_managed_thread_pool_(use_managed_thread_pool)
        , the_last_op_ {the_last_op}
        , main_tensor_def_info_ {&main_tensor_def_info}
        , lastop_tensor_def_info_ {&lastop_tensor_def_info} {}
};

class single_core_remove_parallel_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    expr_c dispatch(expr_c v) override { return v; }
    stmt_c visit(for_loop_c v) override {
        auto ret = ir_visitor_t::visit(v).checked_as<for_loop>();
        if (ret->kind_ == for_type::PARALLEL) {
            if (ret.get() != v.get()) {
                // the for loop is remade, just change it
                ret->kind_ = for_type::NORMAL;
                ret->attr()[attr_keys::buf_sched_top_scope] = true;
            } else {
                // remake a new IR node
                auto retnode = copy_attr(*v,
                        builder::make_for_loop_unattached(ret->var_,
                                ret->iter_begin_, ret->iter_end_, ret->step_,
                                ret->body_, ret->incremental_,
                                for_type::NORMAL));
                retnode->attr()[attr_keys::buf_sched_top_scope] = true;
                return retnode;
            }
        }
        return ret;
    }
};

const_ir_module_ptr closurizer_cpu_t::operator()(const_ir_module_ptr inmod) {
    float gflop
            = inmod->attr_.get_or_else(ir_module_t::attr_key_t::GFLOP, 0.0f);

    bool use_managed_thread_pool = inmod->attr_.get_or_else(
            ir_module_t::attr_key_t::MANAGED_THREAD_POOL, false);

    SC_MODULE_INFO << "Use managed thread pool? " << use_managed_thread_pool
                   << ". Module gflops = " << gflop;

    func_t the_last_op;
    tensor_def_collector_t::result_t main_tensor_def_info;
    tensor_def_collector_t::result_t lastop_tensor_def_info;
    if (!single_core_ && use_managed_thread_pool) {
        // find the last op in main entry
        auto main_entry_func = inmod->get_entry_func();
        if (main_entry_func && main_entry_func->attr_
                && main_entry_func->attr_->get_or_else(
                        function_attrs::is_main, false)) {
            auto &seq = main_entry_func->body_.checked_as<stmts>()->seq_;
            for (auto itr = seq.rbegin(); itr != seq.rend(); ++itr) {
                auto &s = *itr;
                // allow return const after the last op call
                if (s.cast<returns>()
                                .filter([](const returns &v) {
                                    return v->value_.isa<constant>();
                                })
                                .has_value()) {
                    continue;
                }
                auto f = s.cast<evaluate>()
                                 .map([](const evaluate &v) {
                                     return v->value_.as<call>();
                                 })
                                 .map([](const call &v) {
                                     return std::dynamic_pointer_cast<
                                             func_base>(v->func_);
                                 });
                if (f.has_value()) {
                    the_last_op = inmod->get_func(f.get()->name_);
                    if (the_last_op) {
                        assert(the_last_op->name_ == f.get()->name_);
                        tensor_def_collector_t collector;
                        collector.dispatch(main_entry_func);
                        auto last_call = s.static_as<evaluate>()
                                                 ->value_.static_as<call>();
                        collector.link_call_args(last_call, the_last_op);
                        main_tensor_def_info = std::move(collector.result_);

                        tensor_def_collector_t collector2;
                        collector2.dispatch(the_last_op);
                        lastop_tensor_def_info = std::move(collector2.result_);
                    }
                    break;
                }
                // else, there are some complex statements after the last
                // parallel-for, do not optimize
                SC_MODULE_WARN
                        << "Cannot optimize the last barrier in main function "
                        << main_entry_func->name_
                        << " because there are complex code after the last "
                           "parallel-for.";
                break;
            }
        }
    }

    auto ret = inmod->copy();
    ir_visitor_t *the_pass;
    closurize_cpu_impl_t pass(ret, use_managed_thread_pool, the_last_op,
            main_tensor_def_info, lastop_tensor_def_info);
    single_core_remove_parallel_t singlepass {};
    if (single_core_) {
        the_pass = &singlepass;
    } else {
        the_pass = &pass;
    }
    auto &funcs = ret->get_contents();
    auto sz = funcs.size();
    for (unsigned i = 0; i < sz; i++) {
        auto f = std::const_pointer_cast<func_base>(
                the_pass->dispatch(funcs[i]));
        funcs[i] = std::move(f);
    }

    ret->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL]
            = use_managed_thread_pool;
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
