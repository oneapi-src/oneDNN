/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_REORDER_REORDER_KERNEL_HPP
#define GPU_JIT_REORDER_REORDER_KERNEL_HPP

#include "gpu/jit/codegen/codegen.hpp"
#include "gpu/jit/codegen/kernel.hpp"
#include "gpu/jit/codegen/ngen_helpers.hpp"
#include "gpu/jit/codegen/register_scope.hpp"
#include "gpu/jit/ir/ir_builder.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/reorder/ir_builder.hpp"
#include "gpu/jit/utils/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <ngen::HW hw = ngen::HW::Unknown>
class reorder_kernel_t : public ir_kernel_t<hw> {
public:
    IR_KERNEL_FORWARD(hw)

    reorder_kernel_t(const reorder_config_t &cfg,
            const std::string &kernel_name, const kernel_info_t &kernel_info,
            bool require_dpas, grf_mode_t grf_mode,
            const primitive_desc_t *pd = nullptr)
        : ir_kernel_t<hw>(kernel_name, cfg.exec_cfg(), kernel_info,
                kernel_info.nd_range(), require_dpas, grf_mode) {
        const primitive_attr_t *attr = (pd) ? pd->attr() : nullptr;
        const memory_desc_t *dst_md = (pd) ? pd->dst_md() : nullptr;
        reorder_ir_builder_t builder(cfg, kernel_info, attr, dst_md);
        stmt_t body = builder.stmt();
        setup_interface(body);
        generate_prologue();
        expr_binding_t expr_binding(hw);
        bind_external_vars(
                body, builder.kernel_grid(), builder.local_id(), expr_binding);

        // Generate assembly from IR.
        convert_ir_to_ngen<hw>(body, this, expr_binding);

        generate_epilogue();
    }

    static compute::nd_range_t nd_range(const exec_config_t &exec_cfg,
            const layout_t &src, const layout_t &dst) {
        return reorder_ir_builder_t::nd_range(exec_cfg, src, dst);
    }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
