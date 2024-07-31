/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_SIMPLE_ZERO_PAD_HPP
#define GPU_INTEL_OCL_SIMPLE_ZERO_PAD_HPP

#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_zero_pad_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

#define VDISPATCH_ZERO_PAD(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, zero_pad, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

struct simple_zero_pad_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_zero_pad_pd_t {
        using gpu_zero_pad_pd_t::gpu_zero_pad_pd_t;

        DECLARE_COMMON_PD_T("ocl:simple:any", simple_zero_pad_t);
        status_t init(impl::engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            VDISPATCH_ZERO_PAD(compute_engine->mayiuse_sub_group(16),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroup(16)");
            return status::success;
        }
    };

    ;

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        std::vector<compute::kernel_t> kernels {};
        CHECK(create_kernels(engine, &kernels,
                {"simple_zero_pad", "simple_zero_pad_subg_16",
                        "simple_zero_pad_subg_16_mask_and_clear_dt_1b"},
                kernel_ctx));

        kernel_ = kernels[0];
        kernel_subg16_ = kernels[1];
        kernel_subg16_mask_and_clear_dt_1b_ = kernels[2];

        if (!kernel_ || !kernel_subg16_ || !kernel_subg16_mask_and_clear_dt_1b_)
            return status::runtime_error;
        return status::success;
    }
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_simple(const exec_ctx_t &ctx) const;
    status_t execute_subg_16(const exec_ctx_t &ctx,
            const memory_desc_wrapper &mdw,
            const blocking_desc_t &blocking_desc) const;
    status_t execute_subg_16_mask_and_clear_dt_1B(const exec_ctx_t &ctx,
            const memory_desc_wrapper &mdw,
            const blocking_desc_t &blocking_desc) const;
    compute::kernel_t kernel_;
    compute::kernel_t kernel_subg16_;
    compute::kernel_t kernel_subg16_mask_and_clear_dt_1b_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
