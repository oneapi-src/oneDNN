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
#include <string>

#include "llvm_jit_resolver.hpp"
#include <compiler/jit/symbol_resolver.hpp>
#include <util/utils.hpp>

SC_MODULE(jit.llvm_resolver)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

sc_llvm_jit_resolver::sc_llvm_jit_resolver() = default;
sc_llvm_jit_resolver::~sc_llvm_jit_resolver() = default;
uint64_t sc_llvm_jit_resolver::getSymbolAddress(const std::string &name) {
    uint64_t ret = SectionMemoryManager::getSymbolAddress(name);
    if (ret) return ret;
    return reinterpret_cast<uint64_t>(default_external_symbol_resolve(name));
}

uint8_t *sc_llvm_jit_resolver::allocateCodeSection(uintptr_t Size,
        unsigned Alignment, unsigned SectionID, llvm::StringRef SectionName) {
    SC_MODULE_INFO << "allocateCodeSection, Size=" << Size << ", SectionName"
                   << std::string(SectionName);
    return SectionMemoryManager::allocateCodeSection(
            Size, Alignment, SectionID, SectionName);
}

uint8_t *sc_llvm_jit_resolver::allocateDataSection(uintptr_t Size,
        unsigned Alignment, unsigned SectionID, llvm::StringRef SectionName,
        bool isReadOnly) {
    SC_MODULE_INFO << "allocateDataSection, Size=" << Size << ", SectionName"
                   << std::string(SectionName);
    return SectionMemoryManager::allocateDataSection(
            Size, Alignment, SectionID, SectionName, isReadOnly);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
