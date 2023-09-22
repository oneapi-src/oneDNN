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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_XBYAK_JIT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_XBYAK_JIT_HPP

#include <compiler/jit/jit.hpp>
#include <compiler/jit/xbyak/backend/xbyak_jit_generator.hpp>
#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>

#include <memory>
#include <string>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class xbyak_jit;
/**
 * @class xbyak_jit_module
 *
 * Explanation of the inheritence hierarchy:
 *
 * - The two public base classes are required to fit into the Graphcompiler JIT
 *   design.
 *
 * - We want SOME class to inherit from both jit_generator and ir_viewer_t,
 *   as a matter of coding convenience: we can directly call Xbyak-related
 *   codegen functions from the ir_handler callback methods.
 *
 * - We want THIS class to inherit from jit_generator because it's a simple
 *   way to ensure that the memory allocated for the JIT'ed code has the
 *   same lifespan as this xbyak_jit_module.
 */
class SC_INTERNAL_API xbyak_jit_module_code : public jit_module_code {
public:
    virtual ~xbyak_jit_module_code() = default;

private:
    // NOTE: It may be okay to actually provide these. I just haven't given it
    // much consideration yet. -cconvey
    xbyak_jit_module_code(xbyak_jit_module_code &&other) = delete;
    xbyak_jit_module_code(const xbyak_jit_module_code &other) = delete;

    // xbyak_jit is this object's factory class.
    friend class xbyak_jit;

    /**
     * @param jit_output - Describes the xbyak jit result.
     * @param managed_thread_pool - Whether to use managed thread pool
     */
    xbyak_jit_module_code(
            std::shared_ptr<xbyak::xbyak_jit_generator> jit_output,
            bool managed_thread_pool);

    std::shared_ptr<xbyak::xbyak_jit_generator> jit_output_;

public:
    void *get_address_of_symbol(const std::string &name) override;

    void *get_function(const std::string &name, void *&wrapper) override;
};

class SC_INTERNAL_API xbyak_jit : public jit_engine_t {
public:
    std::function<void *(const std::string &)> external_symbol_resolver_;
    xbyak_jit(context_ptr context = get_default_context());
    virtual ~xbyak_jit();

    std::shared_ptr<jit_module> make_jit_module(
            const_ir_module_ptr ir_mod, bool generate_wrapper);
    static void set_target_machine(runtime::target_machine_t &tm) {
        // TODO(XXX): add checks in tm
    }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
