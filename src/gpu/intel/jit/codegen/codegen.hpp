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

#ifndef GPU_INTEL_JIT_CODEGEN_CODEGEN_HPP
#define GPU_INTEL_JIT_CODEGEN_CODEGEN_HPP

#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <ngen::HW hw>
void convert_ir_to_ngen(const stmt_t &body, ir_kernel_t<hw> *host,
        const expr_binding_t &expr_binding);

REG_GEN9_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::Gen9> *host, const expr_binding_t &expr_binding));
REG_GEN11_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::Gen11> *host,
        const expr_binding_t &expr_binding));
REG_XELP_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::XeLP> *host, const expr_binding_t &expr_binding));
REG_XEHP_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::XeHP> *host, const expr_binding_t &expr_binding));
REG_XEHPG_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::XeHPG> *host,
        const expr_binding_t &expr_binding));
REG_XEHPC_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::XeHPC> *host,
        const expr_binding_t &expr_binding));
REG_XE2_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::Xe2> *host, const expr_binding_t &expr_binding));
REG_XE3_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::Xe3> *host, const expr_binding_t &expr_binding));

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
