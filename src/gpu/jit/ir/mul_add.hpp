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

#ifndef GPU_JIT_IR_MUL_ADD_HPP
#define GPU_JIT_IR_MUL_ADD_HPP

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Performs the following operation:
//     buf = alpha * buf + beta
stmt_t create_mul_add_stmt(ir_context_t &ir_ctx, const expr_t &buf, int size,
        const type_t &type, float alpha, float beta);

inline stmt_t create_zero_out_stmt(
        ir_context_t &ir_ctx, const expr_t &buf, int size) {
    return create_mul_add_stmt(ir_ctx, buf, size, type_t::f32(), 0, 0);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
