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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_LLVM_LLVM_JIT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_LLVM_LLVM_JIT_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/sc_function.hpp>
#include <compiler/jit/jit.hpp>
#include <runtime/generic_val.hpp>

namespace llvm {
class ExecutionEngine;
class LLVMContext;
} // namespace llvm

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class llvm_jit;
struct llvm_jit_listeners;
class SC_INTERNAL_API llvm_jit_module_code : public jit_module_code {
    llvm_jit_module_code(llvm_jit_module_code &&other) = delete;
    llvm_jit_module_code(const llvm_jit_module_code &other) = delete;

public:
    llvm_jit_module_code(std::unique_ptr<llvm::ExecutionEngine> engine,
            std::unique_ptr<llvm::LLVMContext> llvm_ctx,
            std::shared_ptr<llvm_jit_listeners> &&listeners,
            bool managed_thread_pool, const std::string &source_path);
    // listeners_ reference will be destructed after engine_, to make sure
    // jit_listeners are still alive when ExecutionEngine is destroyed
    std::shared_ptr<llvm_jit_listeners> listeners_;
    std::unique_ptr<llvm::LLVMContext> llvm_ctx_;
    std::unique_ptr<llvm::ExecutionEngine> engine_;
    std::string source_path_;
    ~llvm_jit_module_code();

    void *get_address_of_symbol(const std::string &name) override;
    void *get_function(const std::string &name, void *&wrapper) override;

    std::vector<std::string> get_temp_filenames() const override;
};

class SC_INTERNAL_API llvm_jit : public jit_engine_t {
public:
    llvm_jit(context_ptr ctx = get_default_context())
        : jit_engine_t(std::move(ctx)) {
        opt_level_ = context_->flags_.backend_opt_level_;
    }
    unsigned opt_level_;
    std::shared_ptr<jit_module> make_jit_module(
            const_ir_module_ptr module, bool generate_wrapper) override;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
