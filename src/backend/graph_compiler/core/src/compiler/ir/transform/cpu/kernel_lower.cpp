/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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
#include <unordered_map>

#include <string>
#include <utility>
#include <vector>
#include "../../builder.hpp"
#include "../../content_hash.hpp"
#include "../../visitor.hpp"
#include "kernel_lower.hpp"
#include <compiler/ir/easy_build.hpp>
#include <microkernel/builtin.hpp>

SC_MODULE(pass.kernel_lowering_cpu)
namespace sc {
using namespace builtin;

static bool check_and_push_const_range(const std::vector<expr> &args,
        std::vector<constant_c> &out_consts, int start, int end) {
    out_consts.clear();
    out_consts.reserve(end - start);
    // check the parameters, if they are all constants,
    // we can cache the kernel
    for (int i = start; i < end; i++) {
        if (!args[i].isa<constant>()) {
            // not const, don't cache
            return false;
        }
        out_consts.emplace_back(args[i].static_as<constant_c>());
    }
    return true;
}

static expr brgemm_init_kernel_cache(brgemm_mode mode,
        scflags_t::brgemm_t backend, const std::vector<expr> &args,
        std::vector<constant_c> &out_consts, float beta) {
    if (mode == brgemm_mode::stride) {
        // +2 for dtypes
        const int expected_num_args = brgemm_args::NUM_ARGS_CPU + 2;
        // +1 for context_t pointer
        assert(args.size() == expected_num_args + 1);
        // check the parameters from M to stride_b
        if (!check_and_push_const_range(
                    args, out_consts, brgemm_args::M, expected_num_args)) {
            return expr();
        }
        return get_brgemm_creator_and_call_func(mode, backend)
                .first(args[brgemm_args::M], args[brgemm_args::N],
                        args[brgemm_args::K], args[brgemm_args::LDA],
                        args[brgemm_args::LDB], args[brgemm_args::LDC],
                        args[brgemm_args::STRIDE_A],
                        args[brgemm_args::STRIDE_B], beta,
                        args[brgemm_args::NUM_ARGS_CPU],
                        args[brgemm_args::NUM_ARGS_CPU + 1]);
    } else {
        // +2 for dtypes
        const int expected_num_args = brgemm_args::NUM_ARGS_LIST + 2;
        // +1 for context_t pointer
        assert(args.size() == expected_num_args + 1);
        // check the parameters from M to LDC
        if (!check_and_push_const_range(
                    args, out_consts, brgemm_args::M, expected_num_args)) {
            return expr();
        }
        return get_brgemm_creator_and_call_func(mode, backend)
                .first(args[brgemm_args::M], args[brgemm_args::N],
                        args[brgemm_args::K], args[brgemm_args::LDA],
                        args[brgemm_args::LDB], args[brgemm_args::LDC], beta,
                        args[brgemm_args::NUM_ARGS_LIST],
                        args[brgemm_args::NUM_ARGS_LIST + 1]);
    }
}

static expr brgemm_run(brgemm_mode mode, scflags_t::brgemm_t backend,
        const expr &cache, const std::vector<expr> &args) {
    if (mode == brgemm_mode::stride) {
        const int expected_num_args = brgemm_args::NUM_ARGS_CPU + 2;
        assert(args.size() == expected_num_args + 1);
        return get_brgemm_creator_and_call_func(mode, backend)
                .second(cache, args[brgemm_args::A], args[brgemm_args::B],
                        args[brgemm_args::C], args[brgemm_args::NUM],
                        /*ctx*/ args.back());
    } else {
        const int expected_num_args = brgemm_args::NUM_ARGS_LIST + 2;
        assert(args.size() == expected_num_args + 1);
        return get_brgemm_creator_and_call_func(mode, backend)
                .second(cache, args[brgemm_args::A], args[brgemm_args::B],
                        args[brgemm_args::C], args[brgemm_args::NUM],
                        args[brgemm_args::STRIDE_A],
                        args[brgemm_args::STRIDE_B], args[brgemm_args::LEN],
                        args[brgemm_args::NUM_ARGS_LIST],
                        args[brgemm_args::NUM_ARGS_LIST + 1],
                        /*ctx*/ args.back());
    }
}

class kernel_lower_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    // the kernel parameters => kernel pointer mapping
    using param_cache_table = content_hash_map<std::vector<constant_c>, expr>;
    ir_module_ptr mod_;
    bool optimize_;
    // the kernel name => param_cache_table mapping
    std::unordered_map<std::string, param_cache_table> kernel_cache;
    typedef expr (*init_func_t)(brgemm_mode mode, scflags_t::brgemm_t backend,
            const std::vector<expr> &args, std::vector<constant_c> &out_consts,
            float beta);
    using run_func_t = expr (*)(brgemm_mode, scflags_t::brgemm_t, const expr &,
            const std::vector<expr> &);
    expr_c optimize_kernel_call(brgemm_mode mode, scflags_t::brgemm_t backend,
            expr_c v, const std::vector<expr> &args, const std::string &name,
            init_func_t init_func, run_func_t run_func, float beta) {
        std::vector<constant_c> out_consts;
        expr cachev = init_func(mode, backend, args, out_consts, beta);
        // check if the kernel lowerer agrees to optimize the kernel call
        if (!cachev.defined()) {
            SC_MODULE_INFO << "Cannot optimize the kernel call: " << v;
            return v;
        }
        // find the param_cache_table in the kernel cache
        auto cache_itr = kernel_cache.find(name);
        if (cache_itr != kernel_cache.end()) {
            auto &entry = cache_itr->second;
            auto itr = entry.find(out_consts);
            if (itr != entry.end()) {
                // if the same parameters are cached in the kernel_cache,
                // reuse the cached kernel pointer
                return run_func(mode, backend, itr->second, args);
            }
        }
        // Make kernel pointer global var. will be auto-renamed
        expr cache = mod_->make_global_var(cachev->dtype_, "__sc_kernel_cache",
                linkage::private_global, cachev);
        // put the var to the kernel_cache
        kernel_cache[name][out_consts] = cache;
        return run_func(mode, backend, cache, args);
    }

    expr_c visit(intrin_call_c v) override {
        brgemm_args::extra_args_t *extras;
        sc_data_type_t dtypeA, dtypeB;
        brgemm_mode mode;
        v = ir_visitor_t::visit(std::move(v)).checked_as<intrin_call_c>();
        if (v->type_ == intrin_type::brgemm) {
            mode = brgemm_mode::stride;
        } else if (v->type_ == intrin_type::list_brgemm) {
            mode = brgemm_mode::addr_list;
        } else {
            return v;
        }
        extras = &v->intrin_attrs_->get<brgemm_args::extra_args_t>(
                intrin_attr::brgemm_extras);
        dtypeA = extras->dtype_A_;
        dtypeB = extras->dtype_B_;
        COMPILE_ASSERT(extras->is_cpu_, "Found non-CPU brgemm: " << v);
        scflags_t::brgemm_t backend = mod_->ctx_->flags_.brgemm_backend_;

        auto fpair = get_brgemm_update_funcs(mode, backend);
        func_t f;
        if (extras->cpu_.init_) {
            f = fpair.second;
        } else {
            f = fpair.first;
        }
        assert(f);
        std::vector<expr> dtype_args {v->args_.begin(), v->args_.end()};
        dtype_args.emplace_back(dtypeA.as_etype_int());
        dtype_args.emplace_back(dtypeB.as_etype_int());
        // the context_t (as null)
        dtype_args.emplace_back(
                make_expr<constant_node>(0UL, datatypes::pointer));
        bool optimized = optimize_;
        if (!optimized) {
            return builder::make_call(f, dtype_args);
        } else {
            auto ret = optimize_kernel_call(mode, backend, v, dtype_args,
                    f->name_, brgemm_init_kernel_cache, brgemm_run,
                    extras->cpu_.init_ ? 0.0f : 1.0f);
            if (ret.ptr_same(v)) { return builder::make_call(f, dtype_args); }
            return ret;
        }
    }

    kernel_lower_impl_t(ir_module_ptr mod, bool optimize)
        : mod_(std::move(mod)), optimize_(optimize) {}
}; // namespace sc

const_ir_module_ptr kernel_lowering_cpu_t::operator()(const_ir_module_ptr m) {
    auto ret = m->copy();
    kernel_lower_impl_t pass(ret, optimize_);
    auto old_gval_size = ret->get_module_vars().size();
    for (auto &f : ret->get_contents()) {
        f = std::const_pointer_cast<func_base>(pass.dispatch(f));
    }
    if (old_gval_size != ret->get_module_vars().size()) {
        if (auto initf = ret->get_func("__sc_init__")) {
            for (size_t i = old_gval_size; i < ret->get_module_vars().size();
                    i++) {
                auto pvar = ret->get_module_vars()[i];
                COMPILE_ASSERT(pvar->var_.isa<var>(),
                        "Expecting var def in kernel_lowering_cpu_t");
                initf->body_.checked_as<stmts>()->seq_.emplace_back(
                        builder::make_assign_unattached(
                                pvar->var_, pvar->init_));
            }
        } else {
            ret->add_func({ret->make_init_func()});
        }
    }

    return ret;
}

} // namespace sc
