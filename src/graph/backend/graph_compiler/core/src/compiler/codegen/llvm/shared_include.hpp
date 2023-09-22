/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_LLVM_SHARED_INCLUDE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_LLVM_SHARED_INCLUDE_HPP

#include <compiler/codegen/codegen_llvm.hpp>
#include <compiler/codegen/precodegen_passes.hpp>
#include <compiler/ir/pass/printer.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/viewer.hpp>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#if SC_LLVM_BACKEND > 16
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#if SC_LLVM_BACKEND > 13
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <util/any_map.hpp>
#include <util/bf16.hpp>
#include <util/file.hpp>
#include <util/scoped_timer.hpp>
#include <util/utils.hpp>

#if SC_LLVM_BACKEND > 15
#if SC_LLVM_BACKEND < 17
#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>
#endif
#include <llvm/Support/ModRef.h>
#endif

#if SC_LLVM_BACKEND > 8
#include <llvm/IR/IntrinsicsX86.h>
#endif

#include "llvm_visitor.hpp"

SC_MODULE(codegen.llvm);

#if SC_LLVM_BACKEND > 8
#define SC_LLVM_ALIGN(a) Align(a)
#else
#define SC_LLVM_ALIGN(a) (a)
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

#if SC_LLVM_BACKEND > 16
// starting from LLVM17, they use STL's optional container
template <typename T>
using Optional = std::optional<T>;
#endif

#if SC_LLVM_BACKEND > 10
using shuffle_idx_t = int;
#else
using shuffle_idx_t = uint32_t;
#endif

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
