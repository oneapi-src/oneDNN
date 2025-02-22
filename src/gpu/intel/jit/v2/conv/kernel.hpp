/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_KERNEL_HPP
#define GPU_INTEL_JIT_V2_CONV_KERNEL_HPP

#include "common/cpp_compat.hpp"

#include "gpu/intel/jit/codegen/codegen.hpp"
#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/v2/conv/builder.hpp"
#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"
#include "gpu/intel/jit/v2/ir/builder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

template <ngen::HW hw>
class kernel_t : public ir_kernel_t<hw> {
public:
    IR_KERNEL_FORWARD(hw)

    kernel_t(const kernel_desc_base_t &_desc, const impl::engine_t *engine);
};

template <ngen::HW hw>
kernel_t<hw>::kernel_t(
        const kernel_desc_base_t &_desc, const impl::engine_t *engine)
    : ir_kernel_t<hw>(_desc, engine, {GENERATOR_NAME, GENERATOR_LINE}) {

    auto &desc = static_cast<const kernel_desc_t &>(_desc);

    this->require_signal_header_ = true;

    // Build IR for the kernel.
    var_manager_t var_mgr(kernel_iface());
    stmt_t body = build_ir(exec_cfg(), desc, var_mgr);

    alloc_manager_t alloc_mgr(body);
    setup_interface(body);

    generate_prologue();

    // Bind "external" variables.
    expr_binding_t expr_binding(hw);
    bind_external_vars(body, expr_binding);

    // Generate assembly from IR.
    convert_ir_to_ngen<hw>(body, this, expr_binding);

    generate_epilogue();
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
