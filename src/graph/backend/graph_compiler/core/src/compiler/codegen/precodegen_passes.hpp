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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_PRECODEGEN_PASSES_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_PRECODEGEN_PASSES_HPP

#include <compiler/ir/util_module_passes.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

sequential_module_pass_t get_default_precodegen_passes(
        const context_ptr &ctx, bool gen_wrapper);

const_ir_module_ptr run_precodegen_passes(
        module_pass_t &pass, const_ir_module_ptr f);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
