/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_PASS_UNROLL_HPP
#define GPU_INTEL_JIT_PASS_UNROLL_HPP

#include "gpu/intel/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Eliminates let statements from the outer loops to be able to unroll loop
// nest for SLM buffering or prefetch injection. Example:
// Before:
//     for (int i = 0; i < I; i++) {
//         int tmp = TMP;
//         for (int j = 0; j < J; j++) {
//            ...
//         }
//     }
// After:
//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             int tmp = TMP;
//             ...
//         }
//     }
stmt_t update_loops_for_unrolling(const stmt_t &s, ir_context_t &ir_ctx);

// Unrolls loops according to their unroll attribute.
// Before:
//     for (int i = 0; i < 2; i++) [unroll: 2] {
//         body(i);
//     }
// After:
//     body(0);
//     body(1);
stmt_t unroll_loops(const stmt_t &s, ir_context_t &ir_ctx);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
