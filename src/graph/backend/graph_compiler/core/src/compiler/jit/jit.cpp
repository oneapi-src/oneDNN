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

#include "jit.hpp"
#if SC_CFAKE_JIT_ENABLED
#include "cfake/cfake_jit.hpp"
#endif
#if defined(SC_LLVM_BACKEND)
#include "llvm/llvm_jit.hpp"
#endif
#include <atomic>
#include <chrono>
#include <stdio.h>
#if SC_BUILTIN_JIT_ENABLED
#include "xbyak/xbyak_jit.hpp"
#endif
#include <compiler/ir/graph/dynamic_utils.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <runtime/config.hpp>
#include <runtime/managed_thread_pool.hpp>
#include <runtime/microkernel/cpu/brgemm_range_handle.hpp>
#include <util/math_utils.hpp>
#include <util/scoped_timer.hpp>

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#include <common/stream.hpp>
#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

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
#if defined(SC_LLVM_BACKEND)
        case jit_kind::llvm: return utils::make_unique<llvm_jit>(ctx);
#endif
#if SC_BUILTIN_JIT_ENABLED
        case jit_kind::xbyak: return utils::make_unique<xbyak_jit>(ctx);
#endif
        default:
            assert(0 && "Bad JIT type");
            return nullptr;
            break;
    }
}

void jit_engine_t::set_target_machine(
        jit_kind kind, scflags_t &sc_flags, runtime::target_machine_t &tm) {
    switch (kind) {
        case jit_kind::cfake:
#if SC_CFAKE_JIT_ENABLED == 0
            return;
#else
        {
            auto flags = cfake_jit::get_compiler_flags();
            if (flags.fAVX512AMXBF16 && flags.fAVX512AMXTILE
                    && flags.fAVX512AMXINT8) {
                sc_flags.jit_support_amx_intrinsics_ = true;
            } else {
                sc_flags.jit_support_amx_intrinsics_ = false;
            }
            return cfake_jit::set_target_machine(tm);
        }
#endif

#if defined(SC_LLVM_BACKEND)
        case jit_kind::llvm:
#if SC_LLVM_BACKEND <= 10
            tm.cpu_flags_.fAVX512BF16 = false;
#endif
#if SC_LLVM_BACKEND >= 12
            sc_flags.jit_support_amx_intrinsics_ = true;
#else
            sc_flags.jit_support_amx_intrinsics_ = false;
#endif
            return;
#endif

#if SC_BUILTIN_JIT_ENABLED
        case jit_kind::xbyak:
            sc_flags.jit_support_amx_intrinsics_ = true;
            return xbyak_jit::set_target_machine(tm);
#endif
        default: assert(0 && "Bad JIT type"); break;
    }
}

static std::atomic<size_t> module_id {0};

jit_module_code::jit_module_code(bool managed_thread_pool)
    : module_id_(module_id++), managed_thread_pool_(managed_thread_pool) {}

void jit_module_code::postprocess(
        const const_ir_module_ptr &ir_mod, statics_table_t &globals) {
    update_runtime_data(ir_mod, globals);
    if (ir_mod->get_entry_func()) {
        entry_func_name_ = ir_mod->get_entry_func()->name_;
    }
}
void jit_module_code::update_op_dispatch_table(
        const const_ir_module_ptr &ir_mod, statics_table_t &globals) {
    constexpr size_t capacity_coefficient = 2;
    auto compiler_tables = ir_mod->get_op_table_map();
    if (compiler_tables.empty()) { return; }
    runtime::dispatch_table_map_t ret;
    ret.reserve(compiler_tables.size());
    for (auto &kv : compiler_tables) {
        const std::string &symbol = kv.first;
        auto &compiler_format_table = kv.second->format_table_;
        auto &compiler_impl_kind_table = kv.second->impl_kind_table_;
        auto &compiler_kernel_table = kv.second->kernel_table_;
        auto &compiler_op_info = kv.second->op_info_;
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
        if (!compiler_impl_kind_table.empty()) {
            uint32_t impl_num_keys
                    = compiler_impl_kind_table.begin()->first.size();
            size_t impl_capacity = math_utils::nearest_power_of_2(
                                           compiler_impl_kind_table.size())
                    * capacity_coefficient;
            runtime_table->impl_kind_table_
                    = utils::make_unique<runtime::hash_dispatch_table_t>(
                            impl_num_keys, impl_capacity);
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
        // initialize impl kind table
        for (auto &impl_kv : compiler_impl_kind_table) {
            runtime_table->set_impl_kind_table_keys(
                    const_cast<uint64_t *>(impl_kv.first.data()),
                    impl_kv.first.size(), impl_kv.second);
        }
        // initialize kernel table
        for (auto &kernel_kv : compiler_kernel_table) {
            void *kernel_addr = kernel_kv.second.already_compiled()
                    ? get_address_of_symbol(kernel_kv.second.name_or_postfix_)
                    : nullptr;
            runtime_table->kernel_table_->set(
                    reinterpret_cast<uint64_t *>(
                            const_cast<runtime::dispatch_key *>(
                                    kernel_kv.first.data())),
                    kernel_kv.first.size(), kernel_addr);
        }
        // update op info
        runtime_table->op_info_ = compiler_op_info;
        // update global table vars' pointer
        auto var_name = kv.first;
        void **value = reinterpret_cast<void **>(globals.get(var_name));
        *value = runtime_table.get();
        ret[var_name] = std::move(runtime_table);
    }
    op_tables_ = std::move(ret);
}

void jit_module_code::update_runtime_data(
        const const_ir_module_ptr &ir_mod, statics_table_t &globals) {
    // update op dispatch table
    update_op_dispatch_table(ir_mod, globals);
    // update brgemm range handler.
    auto brg_handles = ir_mod->get_brg_range_handle_vec();
    brg_handles_.insert(
            brg_handles_.end(), brg_handles.begin(), brg_handles.end());
}

void *jit_module::get_address_of_symbol(const std::string &name) {
    void *global_var = globals_.get_or_null(name);
    if (global_var) { return global_var; }
    return code_->get_address_of_symbol(name);
}
std::shared_ptr<jit_function_t> jit_module::get_function(
        const std::string &name) {
    void *wrapper = nullptr;
    auto func = code_->get_function(name, wrapper);
    if (func || wrapper) {
        if (runtime_config_t::get().execution_verbose_) {
            return general_jit_function_t::make(shared_from_this(), func,
                    wrapper, name, code_->managed_thread_pool_);
        } else {
            return general_jit_function_t::make(shared_from_this(), func,
                    wrapper, std::string(), code_->managed_thread_pool_);
        }
    } else {
        return nullptr;
    }
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
                    ths->fname_.c_str(), ths->module_->code_->module_id_,
                    duration);
        }
    };
    utils::scoped_timer<callback_t> timer_;
    jit_timer_t(const general_jit_function_t *ths)
        : timer_ {true, callback_t {ths}} {}
};

using functype = runtime::thread_manager::main_func_t;

template <bool thread_pool_init>
struct thread_pool_caller_t {
    static void call(functype f, runtime::stream_t *stream, void *module_data,
            generic_val *args) {
        f(stream, module_data, args);
    }
};

template <>
struct thread_pool_caller_t<true> {
    static void call(functype f, runtime::stream_t *stream, void *module_data,
            generic_val *args) {
        runtime::thread_manager::cur_mgr.run_main_function(
                f, stream, module_data, args);
    }
};

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#define before_kernel_run() \
    bool already_in_tp = threadpool_utils::get_active_threadpool(); \
    if (!already_in_tp) { \
        dnnl::threadpool_interop::threadpool_iface *tp = nullptr; \
        stream->vtable()->stream->get_threadpool(&tp); \
        threadpool_utils::activate_threadpool(tp); \
    }
#define after_kernel_run() \
    if (!already_in_tp) { threadpool_utils::deactivate_threadpool(); }
#else
#define before_kernel_run()
#define after_kernel_run()
#endif

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
        before_kernel_run();
        thread_pool_caller_t<thread_pool_init>::call(
                f, stream, module_data, args);
        after_kernel_run()
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

    before_kernel_run();
    f(stream, module_data, args);
    after_kernel_run();
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

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
