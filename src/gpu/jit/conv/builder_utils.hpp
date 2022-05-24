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

#ifndef GPU_JIT_CONV_BUILDER_UTILS_HPP
#define GPU_JIT_CONV_BUILDER_UTILS_HPP

#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Trace for debugging purposes.
inline void trace_pass(const char *pass_name, const stmt_t &stmt) {
    ir_trace() << "=== After " << pass_name << std::endl;
    ir_trace() << stmt << std::endl;
}

// Performs the following operation:
//     buf = alpha * buf + beta
stmt_t create_mul_add_stmt(ngen::HW hw, const expr_t &buf, int size,
        const type_t &type, float alpha, float beta);

inline stmt_t create_zero_out_stmt(ngen::HW hw, const expr_t &buf, int size) {
    return create_mul_add_stmt(hw, buf, size, type_t::f32(), 0, 0);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
