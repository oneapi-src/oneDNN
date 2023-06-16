/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_JIT_CONV_CONV_KERNEL_HPP
#define GPU_JIT_CONV_CONV_KERNEL_HPP

#include "common/cpp_compat.hpp"

#include "gpu/jit/codegen/codegen.hpp"
#include "gpu/jit/codegen/kernel.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reduce.hpp"
#include "gpu/jit/ir/reorder.hpp"

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/grf_usage.hpp"
#include "gpu/jit/conv/ir_builder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <ngen::HW hw>
class conv_kernel_t : public ir_kernel_t<hw> {
public:
    IR_KERNEL_FORWARD(hw)

    conv_kernel_t(const conv_config_t &cfg, const kernel_info_t &kernel_info,
            const compute::nd_range_t &nd_range,
            grf_mode_t grf_mode = grf_mode_t::any);

private:
    const conv_problem_t &prb_;
    const conv_config_t &cfg_;
};

template <ngen::HW hw>
conv_kernel_t<hw>::conv_kernel_t(const conv_config_t &cfg,
        const kernel_info_t &kernel_info, const compute::nd_range_t &nd_range,
        grf_mode_t grf_mode)
    : ir_kernel_t<hw>("gen_conv", cfg.exec_cfg(), kernel_info, nd_range,
            utils::one_of(cfg.fma_kind(), fma_kind_t::dpas, fma_kind_t::dpasw),
            grf_mode)
    , prb_(cfg.prb())
    , cfg_(cfg) {

    // XXX: BWD_W does 32x32 multiplication in the inner loop which may cause
    // hangs when using with split barrier. Switch to emulation to work around
    // the issue.
    if (prb_.is_bwd_w && hw < ngen::HW::XeHPC) emu_strategy.emulate64 = true;

    ir_utils::debug_profiler_t profile("Conv Kernel Construction Profile");
    // Build IR for the kernel.
    conv_ir_builder_t builder(cfg, kernel_info);
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
    bind_external_vars(
            body, cfg_.kernel_grid(), builder.local_id(), expr_binding);
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
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
