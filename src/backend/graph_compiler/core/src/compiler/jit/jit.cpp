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

#include "jit.hpp"
#if SC_CFAKE_JIT_ENABLED
#include "cfake/cfake_jit.hpp"
#endif
#include <atomic>
#include <chrono>
#include <stdio.h>
#include "llvm/llvm_jit.hpp"
#include <compiler/ir/pass/ir_copy.hpp>
#include <runtime/config.hpp>
#include <util/scoped_timer.hpp>

namespace sc {

std::shared_ptr<jit_function_t> jit_engine_t::get_entry_func(
        const ir_module_ptr &ir_mod, bool generic) {
    auto jm = make_jit_module(ir_mod, generic);
    COMPILE_ASSERT(ir_mod->get_entry_func(),
            "Expecting an ir_module with entry function");
    return jm->get_function(ir_mod->get_entry_func()->name_);
}

std::unique_ptr<jit_engine_t> jit_engine_t::make(const context_ptr &ctx) {
    switch (ctx->flags_.jit_kind_) {
#if SC_CFAKE_JIT_ENABLED
        case jit_kind::cfake: return utils::make_unique<cfake_jit>(ctx);
#endif
        case jit_kind::llvm: return utils::make_unique<llvm_jit>(ctx);
        default:
            assert(0 && "Bad JIT type");
            return nullptr;
            break;
    }
}

void jit_engine_t::set_target_machine(jit_kind kind, target_machine_t &tm) {
    switch (kind) {
        case jit_kind::cfake:
#if SC_CFAKE_JIT_ENABLED == 0
            return;
#else
            return cfake_jit::set_target_machine(tm);
#endif

        case jit_kind::llvm:
#if SC_LLVM_BACKEND <= 8
            tm.cpu_flags_.fAVX512BF16 = false;
#endif
            return;
        default: assert(0 && "Bad JIT type"); break;
    }
}

static std::atomic<size_t> module_id {0};

jit_module::jit_module(bool managed_thread_pool)
    : module_id_(module_id++), managed_thread_pool_(managed_thread_pool) {}
jit_module::jit_module(statics_table_t &&globals, bool managed_thread_pool)
    : globals_(std::move(globals))
    , module_id_(module_id++)
    , managed_thread_pool_(managed_thread_pool) {}

template <bool execution_verbose>
struct jit_timer_t {
    jit_timer_t(const general_jit_function_t *) {}
};

template <>
struct jit_timer_t<true> {
    struct callback_t {
        const general_jit_function_t *ths;
        void operator()(utils::time_duration dur) const {
            using namespace std::chrono;
            double duration = static_cast<double>(
                                      duration_cast<nanoseconds>(dur).count())
                    / 1e6;
            printf("Entry point: %s@%zu. Time elapsed: %lf ms\n",
                    ths->fname_.c_str(), ths->module_->module_id_, duration);
        }
    };
    utils::scoped_timer<callback_t> timer_;
    jit_timer_t(const general_jit_function_t *ths)
        : timer_ {true, callback_t {ths}} {}
};

using functype = void (*)(runtime::stream_t *, void *, generic_val *);

template <bool thread_pool_init>
struct thread_pool_caller_t {
    static void call(functype f, runtime::stream_t *stream, void *module_data,
            generic_val *args) {
        f(stream, module_data, args);
    }
};

template <bool thread_pool_init, bool execution_verbose>
class injected_general_jit_function_t : public general_jit_function_t {
    void call_generic(
            runtime::stream_t *stream, generic_val *args) const override {
        injected_general_jit_function_t::call_generic(
                stream, module_->globals_.data_.data_, args);
    }

    void call_generic(runtime::stream_t *stream, void *module_data,
            generic_val *args) const override {
        jit_timer_t<execution_verbose> timer(this);
        assert(wrapper_ && "Trying to call 'call_generic' \
            on a jit funciton with no wrapper.");
        functype f = reinterpret_cast<functype>(wrapper_);
        thread_pool_caller_t<thread_pool_init>::call(
                f, stream, module_data, args);
    }

public:
    using general_jit_function_t::general_jit_function_t;
    friend class general_jit_function_t;
};

void general_jit_function_t::call_generic(
        runtime::stream_t *stream, generic_val *args) const {
    general_jit_function_t::call_generic(
            stream, module_->globals_.data_.data_, args);
}

void general_jit_function_t::call_generic(
        runtime::stream_t *stream, void *module_data, generic_val *args) const {
    assert(wrapper_ && "Trying to call 'call_generic' \
            on a jit funciton with no wrapper.");
    functype f = reinterpret_cast<functype>(wrapper_);
    f(stream, module_data, args);
}

std::shared_ptr<jit_function_t> general_jit_function_t::make(
        const std::shared_ptr<jit_module> &module, void *funcptr, void *wrapper,
        const std::string &name, bool managed_thread_pool) {
    auto &runtime_cfg = runtime_config_t::get();
    if (managed_thread_pool) {
        if (runtime_cfg.execution_verbose_) {
            return std::shared_ptr<general_jit_function_t>(
                    new injected_general_jit_function_t<true, true>(
                            module, funcptr, wrapper, name));
        } else {
            return std::shared_ptr<general_jit_function_t>(
                    new injected_general_jit_function_t<true, false>(
                            module, funcptr, wrapper, name));
        }
    } else {
        if (runtime_cfg.execution_verbose_) {
            return std::shared_ptr<general_jit_function_t>(
                    new injected_general_jit_function_t<false, true>(
                            module, funcptr, wrapper, name));
        } else {
            return std::shared_ptr<general_jit_function_t>(
                    new general_jit_function_t(module, funcptr, wrapper, name));
        }
    }
}

} // namespace sc
