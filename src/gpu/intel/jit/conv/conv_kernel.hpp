/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CONV_CONV_KERNEL_HPP
#define GPU_INTEL_JIT_CONV_CONV_KERNEL_HPP

#include "common/cpp_compat.hpp"

#include "gpu/intel/jit/codegen/codegen.hpp"
#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/reduce.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"

#include "gpu/intel/jit/conv/config.hpp"
#include "gpu/intel/jit/conv/grf_usage.hpp"
#include "gpu/intel/jit/conv/ir_builder.hpp"
#include "gpu/intel/jit/conv/plan.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <ngen::HW hw>
class conv_kernel_t : public ir_kernel_t<hw> {
public:
    IR_KERNEL_FORWARD(hw)

    conv_kernel_t(const conv_config_t &cfg, const kernel_info_t &kernel_info,
            const compute::range_t &local_range, const layout_t &zp_dst);

private:
    const conv_problem_t &prb_;
    const conv_config_t &cfg_;
};

template <ngen::HW hw>
conv_kernel_t<hw>::conv_kernel_t(const conv_config_t &cfg,
        const kernel_info_t &kernel_info, const compute::range_t &local_range,
        const layout_t &zp_dst)
    : ir_kernel_t<hw>("gen_conv", cfg.exec_cfg(), local_range,
            utils::one_of(cfg.fma_kind(), fma_kind_t::dpas, fma_kind_t::dpasw),
            {GENERATOR_NAME, GENERATOR_LINE})
    , prb_(cfg.prb())
    , cfg_(cfg) {

    set_kernel_iface(kernel_info.iface());

    // XXX: BWD_W does 32x32 multiplication in the inner loop which may cause
    // hangs when using with split barrier. Switch to emulation to work around
    // the issue.
    if (prb_.is_bwd_w && hw < ngen::HW::XeHPC) emu_strategy.emulate64 = true;

    ir_utils::debug_profiler_t profile("Conv Kernel Construction Profile");
    // Build IR for the kernel.
    conv_ir_builder_t builder(cfg, kernel_info, zp_dst);
    stmt_t body = builder.stmt();
    profile.stamp("Kernel Builder");

    alloc_manager_t alloc_mgr(body);
    profile.stamp("Alloc_Mgr Construct");

    setup_interface(body);
    profile.stamp("Setup Interface");

    this->require_signal_header_ = true;
    generate_prologue();

    profile.stamp("Prologue");

    // Bind "external" variables.
    expr_binding_t expr_binding(hw);
    bind_external_vars(body, cfg_.plan().gemm_schedule.kernel_grid_walk_order(),
            expr_binding);
    profile.stamp("Bind Variables");

#ifdef DNNL_DEV_MODE
    profile.stop();
    verify_grf_usage(cfg, body, ra_.get_alloced_regs());
    profile.start();
#endif

    // Generate assembly from IR.
    convert_ir_to_ngen<hw>(body, this, expr_binding);
    profile.stamp("Generate Assembly");

    generate_epilogue();
    profile.stop("Epilogue");

#ifdef DNNL_DEV_MODE
    ir_perf_no_trace() << profile << "\n";

    ir_trace() << "Actual register usage:           " << ra_.get_peak_regs()
               << std::endl;
    int estimated_peak_regs = estimate_register_count(cfg_);
    if (ra_.get_peak_regs() > estimated_peak_regs) {
        ir_warning()
                << "conv_kernel_t register usage underestimated: estimate = "
                << estimated_peak_regs << ", actual = " << ra_.get_peak_regs()
                << "\n";
    }
#endif
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
