/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_XBYAK_JIT_ENGINE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_XBYAK_JIT_ENGINE_HPP

#include <memory>
#include <string>
#include <compiler/jit/jit.hpp>

namespace sc {
namespace sc_xbyak {

class xbyak_jit_module;

class SC_INTERNAL_API xbyak_jit_engine : public jit_engine_t {
public:
    std::function<void *(const std::string &)> external_symbol_resolver_;
    xbyak_jit_engine(context_ptr context = get_default_context());
    virtual ~xbyak_jit_engine();

    std::shared_ptr<jit_module> make_jit_module(
            const_ir_module_ptr ir_mod, bool generate_wrapper);
    static void set_target_machine(runtime::target_machine_t &tm) {
        // TODO(XXX): add checks in tm
    }
};

} // namespace sc_xbyak
} // namespace sc

#endif
