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

#include "jit.hpp"
#if SC_CFAKE_JIT_ENABLED
#include "cfake/cfake_jit.hpp"
#endif
#include "llvm/llvm_jit.hpp"
#include <compiler/ir/pass/ir_copy.hpp>

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
#ifdef SC_CFAKE_JIT_ENABLED
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

} // namespace sc
