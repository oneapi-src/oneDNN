/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REDUCTION_COMBINED_REDUCTION_HPP
#define GPU_INTEL_OCL_REDUCTION_COMBINED_REDUCTION_HPP

#include "common/primitive.hpp"
#include "gpu/gpu_reduction_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/reduction/reduction_utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct reduction_phase_conf_t : public reduction_subproblem_t {
    reduction_phase_conf_t(const reduction_subproblem_t &subprb,
            data_type_t src_type, data_type_t dst_type,
            const compute::compute_engine_t *compute_engine,
            bool large_grf_mode);
    bool can_use_block_reads();
    data_type_t src_type, dst_type;
    compute::nd_range_t nd_range;

    int outer_tile_size, slm_reductions;
    bool is_final, is_first;
    int subgroup_size;
    bool with_block_reads;
};

struct combined_reduction_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_reduction_pd_t {
        using gpu_reduction_pd_t::gpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("ocl:combined", combined_reduction_t);

        status_t init(impl::engine_t *engine) {
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto attr_skip_mask = smask_t::post_ops | smask_t::gpu_attr;
            VDISPATCH_REDUCTION_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_REDUCTION(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REDUCTION(!memory_desc_ndims_ok(src_md(), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
            VDISPATCH_REDUCTION(
                    post_ops_with_binary_ok(attr(), dst_md()->data_type, 5),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_REDUCTION_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_REDUCTION_SC(init_conf(engine), "init_conf");
            init_scratchpad();

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
                const reduction_phase_conf_t &phase) const;
        void init_scratchpad();

        reduction_conf_t conf;
        std::vector<reduction_phase_conf_t> phases;
    };

    status_t init(impl::engine_t *engine) override {
        auto &phases = pd()->phases;

        status_t status;
        for (auto &phase : phases) {
            compute::kernel_ctx_t kernel_ctx(pd()->attr());
            status = pd()->init_kernel_ctx(kernel_ctx, phase);
            CHECK(status);
            compute::kernel_t kernel;
            status = create_kernel(
                    engine, &kernel, "combined_reduce", kernel_ctx);
            CHECK(status);
            kernels_.push_back(kernel);
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_combined(ctx);
    }

private:
    status_t execute_combined(const exec_ctx_t &ctx) const;
    const pd_t *pd() const {
        return reinterpret_cast<const pd_t *>(primitive_t::pd().get());
    }

    std::vector<compute::kernel_t> kernels_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
