/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_REDUCTION_HPP
#define GPU_INTEL_JIT_REDUCTION_HPP

// A small wrapper on the reduction_generator_t, used to test its functionality.
// Only valid in dev mode for now, until performance is improved.
#ifdef DNNL_DEV_MODE

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reduction_pd.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_reduction_pd.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/jit/reduction_generator.hpp"
#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct reduction_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_reduction_pd_t {
        using gpu_reduction_pd_t::gpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("jit:ref", reduction_t);

        status_t init(impl::engine_t *engine) {
            // Require the corresponding environment variable - skip this impl
            // unless requested (do not report this skip to verbose)
            bool enable_jit_reduction
                    = gpu_utils::dev_getenv("enable_jit_reduction", false);
            if (!enable_jit_reduction) return status::unimplemented;

            using smask_t = primitive_attr_t::skip_mask_t;
            const auto attr_skip_mask = smask_t::gpu_attr;
            VDISPATCH_REDUCTION_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_REDUCTION(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REDUCTION(!memory_desc_ndims_ok(src_md(), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
            VDISPATCH_REDUCTION_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_TAG);
            // Only f32 supported for now
            VDISPATCH_REDUCTION(
                    utils::everyone_is(data_type::f32, src_md()->data_type,
                            dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            // Make sure we can use the injector for this problem
            VDISPATCH_REDUCTION(
                    reduction_injector_f32_is_supported(desc()->alg_kind),
                    VERBOSE_BAD_ALGORITHM);

            VDISPATCH_REDUCTION_SC(init_conf(engine), "init_conf");

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        dim_t reduction_size = 0;
        dim_t reduction_stride = 0;
        int nregs = 1;
        compute::nd_range_t nd_range;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto *gpu_engine = utils::downcast<ocl::engine_t *>(engine);
        if (!gpu_engine) return status::runtime_error;

        const compute::device_info_t &device_info = *gpu_engine->device_info();
        kernel_ = make_kernel<reduction_generator_t>(this, engine, device_info,
                pd()->desc()->alg_kind, pd()->reduction_stride,
                pd()->reduction_size, pd()->nregs);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return reinterpret_cast<const pd_t *>(primitive_t::pd().get());
    }

    compute::kernel_t kernel_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_DEV_MODE
#endif
