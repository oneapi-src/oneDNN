/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_JIT_V2_CONV_KERNEL_HPP
#define GPU_JIT_V2_CONV_KERNEL_HPP

#include "common/cpp_compat.hpp"

#include "gpu/jit/codegen/codegen.hpp"
#include "gpu/jit/codegen/kernel.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/v2/conv/ir_builder.hpp"
#include "gpu/jit/v2/conv/kernel_desc.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

template <ngen::HW hw>
class kernel_t : public ir_kernel_t<hw> {
public:
    IR_KERNEL_FORWARD(hw)

    kernel_t(const kernel_desc_base_t &_desc, const kernel_info_t &kernel_info);
};

template <ngen::HW hw>
kernel_t<hw>::kernel_t(
        const kernel_desc_base_t &_desc, const kernel_info_t &kernel_info)
    : ir_kernel_t<hw>(_desc, kernel_info) {

    auto &desc = static_cast<const kernel_desc_t &>(_desc);

    this->require_signal_header_ = true;

    // Build IR for the kernel.
    grid_context_t grid_ctx;
    stmt_t body = build_ir(desc, kernel_info, grid_ctx);

    alloc_manager_t alloc_mgr(body);
    setup_interface(body);

    generate_prologue();

    // Bind "external" variables.
    expr_binding_t expr_binding(hw);
    bind_external_vars(body, grid_ctx, expr_binding);

    // Generate assembly from IR.
    convert_ir_to_ngen<hw>(body, this, expr_binding);

    generate_epilogue();
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
