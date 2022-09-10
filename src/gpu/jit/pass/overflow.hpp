/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_JIT_PASS_OVERFLOW_HPP
#define GPU_JIT_PASS_OVERFLOW_HPP

#include "gpu/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Detects and fixes overflows of operations with 32-bit integers.
// Before (a * b can overflow):
//     c.u64 = u64(c_ptr) + a.s32 * b.s32
// After:
//     c.u64 = u64(c_ptr) + s64(a.s32) * b.s32
stmt_t fix_int32_overflow(const stmt_t &s, ir_context_t &ir_ctx);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
