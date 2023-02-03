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
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/closurize_impl.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <runtime/config.hpp>
#include <runtime/thread_pool_flags.hpp>
#include <runtime/trace.hpp>
#include <util/any_map.hpp>
#include <util/utils.hpp>

SC_MODULE(pass.closurize)
namespace sc {

SC_DECL_PASS_INFO(closurizer_cpu,
        SC_PASS_DEPENDS_ON(nested_parallel_flattener,
                parallel_workload_dispatcher, validator, trace_inserter,
                tensor_init),
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

class closurize_cpu_impl_t : public closurize_impl_t {
    int rename_counter_ = 0;
    bool use_managed_thread_pool_;
    std::vector<call> out_calls;
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
        auto ret_call = builder::make_call(
                get_parallel_call_with_env_func(use_managed_thread_pool_),
                std::vector<expr_c> {builder::make_func_addr(std::move(target)),
                        /*flags*/ make_expr<constant_node>(UINT64_C(0)),
                        /*stream*/
                        make_expr<constant_node>(
                                UINT64_C(0), datatypes::pointer),
                        /*env*/
                        make_expr<constant_node>(
                                UINT64_C(0), datatypes::s8.get_pointerof()),
                        begin_v, end_v, step_v, argsbuf});
        out_calls.emplace_back(ret_call.static_as<call>());
        seq->seq_.emplace_back(make_stmt<evaluate_node_t>(std::move(ret_call)));
        return seq;
    }

public:
    using closurize_impl_t::dispatch;

    func_c dispatch(func_c f) override {
        out_calls.clear();
        auto ret = closurize_impl_t::dispatch(f);
        if (!out_calls.empty() && f->attr_
                && f->attr_->get_or_else(
                        function_attrs::has_idle_func, false)) {
            auto &the_flag = out_calls.back()
                                     ->args_.at(1)
                                     .checked_as<constant>()
                                     ->value_.at(0)
                                     .u64;
            the_flag |= runtime::thread_pool_flags::THREAD_POOL_RUN_IDLE_FUNC;
            the_flag |= runtime::thread_pool_flags::THREAD_POOL_DISABLE_ROLLING;
        }
        return ret;
    }
    closurize_cpu_impl_t(const ir_module_ptr &m, bool use_managed_thread_pool)
        : closurize_impl_t(m->get_module_vars(), m)
        , use_managed_thread_pool_(use_managed_thread_pool) {}
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

    auto ret = inmod->copy();
    ir_visitor_t *the_pass;
    closurize_cpu_impl_t pass(ret, use_managed_thread_pool);
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

} // namespace sc
