/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/closurize_impl.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <util/any_map.hpp>
#include <util/utils.hpp>

namespace sc {

func_t get_parallel_call_func() {
    static func_t f
            = builder::_decl_func("sc_parallel_call_cpu", datatypes::void_t,
                    {_arg_("func", datatypes::pointer),
                            _arg_("begin", datatypes::index),
                            _arg_("end", datatypes::index),
                            _arg_("step", datatypes::index),
                            _arg_("args", datatypes::pointer)});
    return f;
}

func_t get_parallel_call_with_env_func() {
    static func_t f = builder::_decl_func("sc_parallel_call_cpu_with_env",
            datatypes::void_t,
            {_arg_("func", datatypes::pointer),
                    _arg_("stream", datatypes::pointer),
                    _arg_("env", datatypes::s8.get_pointerof()),
                    _arg_("begin", datatypes::index),
                    _arg_("end", datatypes::index),
                    _arg_("step", datatypes::index),
                    _arg_("args", datatypes::generic.get_pointerof())});
    return f;
}

class closurize_cpu_impl_t : public closurize_impl_t {
    int rename_counter_ = 0;
    // makes the closure function and its generic wrapper
    func_t make_closure_func(const std::string &name,
            std::vector<expr_c> &&params, stmt_c body,
            const std::vector<call_node::parallel_attr_t> &para_attr) override {
        COMPILE_ASSERT(
                para_attr.size() == 1, "CPU does not support grouped parallel");
        func_t closure
                = builder::make_func(name, params, body, datatypes::void_t);
        // make the wrapper function
        stmts seq = make_stmt<stmts_node_t>(std::vector<stmt>());
        expr itervar = builder::make_var(datatypes::index, "i");
        expr args = builder::make_tensor(
                "args", {(int)params.size() - 1}, datatypes::generic);
        func_t ret = builder::make_func(name + "_0wrapper",
                std::vector<expr> {itervar, args}, seq, datatypes::void_t);
        assert(!modu_->get_func(ret->name_));
        assert(!modu_->get_func(closure->name_));

        closure->attr()["private"] = true;
        ret->attr()["private"] = true;
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
        auto ret_call = builder::make_call(get_parallel_call_func(),
                std::vector<expr_c> {builder::make_func_addr(std::move(target)),
                        begin_v, end_v, step_v, argsbuf});
        seq->seq_.emplace_back(make_stmt<evaluate_node_t>(std::move(ret_call)));
        return seq;
    }

public:
    using closurize_impl_t::dispatch;
    closurize_cpu_impl_t(const ir_module_ptr &m)
        : closurize_impl_t(m->get_module_vars(), m) {}
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
            } else {
                // remake a new IR node
                return copy_attr(*v,
                        builder::make_for_loop_unattached(ret->var_,
                                ret->iter_begin_, ret->iter_end_, ret->step_,
                                ret->body_, ret->incremental_,
                                for_type::NORMAL));
            }
        }
        return ret;
    }
};

const_ir_module_ptr closurizer_cpu_t::operator()(const_ir_module_ptr inmod) {
    std::vector<call> out_closures;
    auto ret = inmod->copy();
    ir_visitor_t *the_pass;
    closurize_cpu_impl_t pass(ret);
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
    return ret;
}

} // namespace sc
