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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_LLVM_LLVM_JIT_RESOLVER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_LLVM_LLVM_JIT_RESOLVER_HPP

#include <string>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class sc_llvm_jit_resolver : public llvm::SectionMemoryManager {
    sc_llvm_jit_resolver(const sc_llvm_jit_resolver &) = delete;
    void operator=(const sc_llvm_jit_resolver &) = delete;

public:
    sc_llvm_jit_resolver();
    virtual ~sc_llvm_jit_resolver();
    virtual uint64_t getSymbolAddress(const std::string &name) override;
    uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
            unsigned SectionID, llvm::StringRef SectionName) override;

    uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
            unsigned SectionID, llvm::StringRef SectionName,
            bool isReadOnly) override;
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
