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
#include "xbyak/xbyak_jit_engine.hpp"
#include <compiler/ir/pass/ir_copy.hpp>
#include <runtime/config.hpp>
#include <util/math_utils.hpp>
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
        case jit_kind::xbyak:
            return utils::make_unique<sc_xbyak::xbyak_jit_engine>(ctx);
        default:
            assert(0 && "Bad JIT type");
            return nullptr;
            break;
    }
}

void jit_engine_t::set_target_machine(
        jit_kind kind, runtime::target_machine_t &tm) {
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
        case jit_kind::xbyak:
            return sc_xbyak::xbyak_jit_engine::set_target_machine(tm);
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

void jit_module::update_runtime_op_tables(const const_ir_module_ptr &ir_mod) {
    constexpr size_t capacity_coefficient = 2;
    auto compiler_tables = ir_mod->get_op_table_map();
    if (compiler_tables.empty()) { return; }
    runtime::dispatch_table_map_t ret;
    ret.reserve(compiler_tables.size());
    for (auto &kv : compiler_tables) {
        const std::string &symbol = kv.first;
        auto &compiler_format_table = kv.second->format_table_;
        auto &compiler_kernel_table = kv.second->kernel_table_;
        runtime::op_dispatch_tables_ptr runtime_table
                = utils::make_unique<runtime::op_dispatch_tables_t>();
        if (!compiler_format_table.empty()) {
            uint32_t format_num_keys
                    = compiler_format_table.begin()->first.size();
            size_t format_capacity = math_utils::nearest_power_of_2(
                                             compiler_format_table.size())
                    * capacity_coefficient;
            runtime_table->format_table_
                    = utils::make_unique<runtime::hash_dispatch_table_t>(
                            format_num_keys, format_capacity);
        }
        if (!compiler_kernel_table.empty()) {
            uint32_t kernel_num_keys
                    = compiler_kernel_table.begin()->first.size();
            size_t kernel_capacity = math_utils::nearest_power_of_2(
                                             compiler_kernel_table.size())
                    * capacity_coefficient;
            // Currently we use hash dispatch table for kernel dispatch
            runtime_table->kernel_table_
                    = utils::make_unique<runtime::hash_dispatch_table_t>(
                            kernel_num_keys, kernel_capacity);
            runtime_table->kernel_dispatch_func_
                    = runtime_table->kernel_table_->get_dispatch_func();
        }

        // initialize format table
        for (auto &format_kv : compiler_format_table) {
            runtime_table->set_format_table_keys(
                    reinterpret_cast<uint64_t *>(
                            const_cast<runtime::dispatch_key *>(
                                    format_kv.first.data())),
                    format_kv.first.size(),
                    reinterpret_cast<uint64_t *>(
                            const_cast<runtime::dispatch_key *>(
                                    format_kv.second.data())),
                    format_kv.second.size());
        }
        // initialize kernel table
        for (auto &kernel_kv : compiler_kernel_table) {
            void *func_addr = get_address_of_symbol(kernel_kv.second);
            assert(func_addr);
            runtime_table->kernel_table_->set(
                    reinterpret_cast<uint64_t *>(
                            const_cast<runtime::dispatch_key *>(
                                    kernel_kv.first.data())),
                    kernel_kv.first.size(), func_addr);
        }
        // update global table vars' pointer
        auto var_name = kv.first;
        void **value = reinterpret_cast<void **>(globals_.get(var_name));
        *value = runtime_table.get();
        ret[var_name] = std::move(runtime_table);
    }
    op_tables_ = std::move(ret);
}

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
