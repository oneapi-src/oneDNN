/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REF_RESAMPLING_HPP
#define GPU_INTEL_OCL_REF_RESAMPLING_HPP

#include "gpu/gpu_resampling_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_resampling_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_resampling_fwd_pd_t {
        using gpu_resampling_fwd_pd_t::gpu_resampling_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_resampling_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            using sm = primitive_attr_t::skip_mask_t;
            const auto attr_skip_mask = sm::post_ops;

            VDISPATCH_RESAMPLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_RESAMPLING(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_RESAMPLING(
                    post_ops_with_binary_ok(attr(), dst_md()->data_type, 5),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_RESAMPLING_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_RESAMPLING_SC(init_conf(engine), "init_conf()");
            return status::success;
        }
        compute::dispatch_t dispatch;
        resampling_conf_t conf;

        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        status_t init_conf(impl::engine_t *engine);
    };

    status_t init(impl::engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "ref_resampling_fwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_resampling_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_resampling_bwd_pd_t {
        using gpu_resampling_bwd_pd_t::gpu_resampling_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_resampling_bwd_t);

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
                engine, &kernel_, "ref_resampling_bwd", kernel_ctx));
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
