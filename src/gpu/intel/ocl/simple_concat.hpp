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

#ifndef GPU_INTEL_OCL_SIMPLE_CONCAT_HPP
#define GPU_INTEL_OCL_SIMPLE_CONCAT_HPP

#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_concat_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct simple_concat_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_concat_pd_t {
        using gpu_concat_pd_t::gpu_concat_pd_t;

        DECLARE_CONCAT_PD_T("simple:any", simple_concat_t);

        status_t init(impl::engine_t *engine) {
            VDISPATCH_CONCAT(n_inputs() <= 64, VERBOSE_BAD_PARAM, "n_inputs");
            VDISPATCH_CONCAT(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONCAT_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_CONCAT_SC(init_conf(engine),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "concat");

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        concat_conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(engine, &kernel_, "simple_concat", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_concat(ctx);
    }

private:
    status_t execute_concat(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
