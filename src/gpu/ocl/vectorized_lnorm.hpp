/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef GPU_OCL_VECTORIZED_LNORM_HPP
#define GPU_OCL_VECTORIZED_LNORM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_layer_normalization_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct vectorized_lnorm_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_layer_normalization_fwd_pd_t {
        using gpu_layer_normalization_fwd_pd_t::
                gpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:vectorized", vectorized_lnorm_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;

            bool ok = is_fwd() && !has_zero_dim_memory()
                    && (utils::everyone_is(f16, src_data_t, dst_data_t)
                            || utils::everyone_is(bf16, src_data_t, dst_data_t)
                            || utils::everyone_is(f32, src_data_t, dst_data_t))
                    && IMPLICATION(f16 == src_data_t,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && !memory_desc_ndims_ok(src_md(), dst_md(), stat_md())
                    && stat_md()->data_type == f32
                    && check_scale_shift_data_type()
                    && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        lnorm_conf_t conf;
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "vectorized_lnorm_fwd", kernel_ctx));
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

struct vectorized_lnorm_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_layer_normalization_bwd_pd_t {
        using gpu_layer_normalization_bwd_pd_t::
                gpu_layer_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:vectorized", vectorized_lnorm_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto src_dt = src_md()->data_type;
            auto diff_dst_dt = diff_dst_md()->data_type;
            auto diff_src_dt = diff_src_md()->data_type;

            bool ok = !is_fwd() && !has_zero_dim_memory()
                    && (utils::everyone_is(
                                f32, src_dt, diff_dst_dt, diff_src_dt)
                            || utils::everyone_is(
                                    bf16, src_dt, diff_dst_dt, diff_src_dt)
                            || utils::everyone_is(
                                    f16, src_dt, diff_dst_dt, diff_src_dt))
                    && IMPLICATION(f16 == src_dt,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && stat_md()->data_type == f32
                    && check_scale_shift_data_type()
                    && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));
            init_scratchpad();
            return status::success;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        lnorm_conf_t conf;
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        if (pd()->conf.use_fused) {
            CHECK(create_kernel(engine, &kernel_fused_,
                    "vectorized_lnorm_bwd_fused", kernel_ctx));
            if (!kernel_fused_) return status::runtime_error;
        }
        CHECK(create_kernel(
                engine, &kernel_, "vectorized_lnorm_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;
        if (pd()->conf.use_scale || pd()->conf.use_shift) {
            CHECK(create_kernel(engine, &kernel_scaleshift_,
                    "vectorized_lnorm_bwd_scaleshift", kernel_ctx));
            if (!kernel_scaleshift_) return status::runtime_error;
            CHECK(create_kernel(engine, &kernel_scaleshift_finalize_,
                    "vectorized_lnorm_bwd_scaleshift_final", kernel_ctx));
            if (!kernel_scaleshift_finalize_) return status::runtime_error;
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_scaleshift_;
    compute::kernel_t kernel_scaleshift_finalize_;
    compute::kernel_t kernel_;
    compute::kernel_t kernel_fused_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
