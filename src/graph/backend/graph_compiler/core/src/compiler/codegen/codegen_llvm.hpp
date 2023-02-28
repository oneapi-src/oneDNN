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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_CODEGEN_LLVM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_CODEGEN_LLVM_HPP

#include <memory>
#include <ostream>

#include <string>
#include <compiler/ir/module_pass.hpp>
#include <llvm/IR/Module.h>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class SC_INTERNAL_API llvm_generator_pass : public module_pass_t {
public:
    llvm::LLVMContext &llvm_ctx_;
    std::unique_ptr<llvm::Module> &out_module_;
    bool gen_wrapper_;
    std::string out_source_path_;
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    llvm_generator_pass(llvm::LLVMContext &llvm_ctx,
            std::unique_ptr<llvm::Module> &out_module, bool gen_wrapper)
        : llvm_ctx_(llvm_ctx)
        , out_module_(out_module)
        , gen_wrapper_(gen_wrapper) {}
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
