/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_OCL_REF_LAYER_NORMALIZATION_HPP
#define GPU_OCL_REF_LAYER_NORMALIZATION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_layer_normalization_pd.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_layer_normalization_fwd_t : public primitive_impl_t {
    struct pd_t : public gpu_layer_normalization_fwd_pd_t {
        using gpu_layer_normalization_fwd_pd_t::
                gpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("lnorm_ref:any", ref_layer_normalization_fwd_t);

        status_t init() {
            using namespace data_type;

            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;

            bool ok = is_fwd()
                    && (utils::everyone_is(f16, src_data_t, dst_data_t)
                            || utils::everyone_is(bf16, src_data_t, dst_data_t)
                            || utils::everyone_is(f32, src_data_t, dst_data_t))
                    && IMPLICATION(src_data_t == f16, !is_training())
                    && stat_md()->data_type == f32
                    && IMPLICATION(
                            use_scaleshift(), weights_md()->data_type == f32)
                    && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            return init_conf();
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        lnorm_conf_t conf;
    };

    ref_layer_normalization_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        compute_engine->create_kernel(&kernel_, "ref_lnorm_fwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    compute::kernel_t kernel_;
};

struct ref_layer_normalization_bwd_t : public primitive_impl_t {
    struct pd_t : public gpu_layer_normalization_bwd_pd_t {
        using gpu_layer_normalization_bwd_pd_t::
                gpu_layer_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("lnorm_ref:any", ref_layer_normalization_bwd_t);

        status_t init() {
            using namespace data_type;

            auto src_data_t = src_md()->data_type;
            auto diff_dst_data_t = diff_dst_md()->data_type;

            auto wei_data_t = weights_md()->data_type;
            auto diff_wei_data_t = diff_weights_md()->data_type;

            bool ok = is_bwd()
                    && (utils::everyone_is(f32, src_data_t, diff_dst_data_t)
                            || utils::everyone_is(
                                    bf16, src_data_t, diff_dst_data_t))
                    && IMPLICATION(use_scaleshift(),
                            utils::everyone_is(
                                    f32, wei_data_t, diff_wei_data_t))
                    && set_default_formats_common()
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return init_conf();
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        lnorm_conf_t conf;
    };

    ref_layer_normalization_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        compute_engine->create_kernel(&kernel_, "ref_lnorm_bwd", kernel_ctx);
        if (pd()->conf.use_scaleshift) {
            compute_engine->create_kernel(&kernel_scaleshift_,
                    "ref_lnorm_bwd_scaleshift", kernel_ctx);
            if (!kernel_scaleshift_) return status::runtime_error;
        }
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    compute::kernel_t kernel_scaleshift_;
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
