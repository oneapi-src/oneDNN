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

#ifndef GPU_INTEL_OCL_VECTORIZED_RESAMPLING_HPP
#define GPU_INTEL_OCL_VECTORIZED_RESAMPLING_HPP

#include "gpu/gpu_resampling_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct vectorized_resampling_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_resampling_bwd_pd_t {
        using gpu_resampling_bwd_pd_t::gpu_resampling_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:vectorized", vectorized_resampling_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);

            VDISPATCH_RESAMPLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_RESAMPLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_RESAMPLING_SC(init_conf(engine), "init_conf()");
            return status::success;
        }
        resampling_conf_t conf;

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
    };

    status_t init(impl::engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "vectorized_resampling_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
