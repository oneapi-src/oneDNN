/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_OCL_REF_BATCH_NORMALIZATION_HPP
#define GPU_OCL_REF_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_batch_normalization_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_batch_normalization_init_conf(
        bnorm_conf_t &conf, const batch_normalization_pd_t *pd, offsets &off);
status_t ref_batch_normalization_init_const_def(
        compute::kernel_ctx_t &kernel_ctx, const bnorm_conf_t &conf,
        const offsets &off);
void ref_batch_normalization_init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const bnorm_conf_t &conf);

struct ref_batch_normalization_fwd_t : public primitive_impl_t {
    struct pd_t : public ocl_batch_normalization_fwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : ocl_batch_normalization_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conf_()
            , off_() {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_batch_normalization_fwd_t);

        status_t init() {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());
            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = true && is_fwd()
                    && (utils::everyone_is(f16, src_data_t, dst_data_t)
                            || utils::everyone_is(bf16, src_data_t, dst_data_t)
                            || utils::everyone_is(f32, src_data_t, dst_data_t))
                    && attr()->has_default_values(attr_skip_mask)
                    && IMPLICATION(!attr()->has_default_values(),
                            attr()->post_ops_.len_ == 1 && with_relu_post_op())
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups);
            if (!ok) return status::unimplemented;

            if (src_data_t == s8 && !stats_is_src())
                return status::unimplemented;

            if (is_training() && fuse_norm_relu()) init_default_ws(8);

            status_t status
                    = ref_batch_normalization_init_conf(conf_, this, off_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            ref_batch_normalization_init_scratchpad(scratchpad, conf_);
            return status::success;
        }

        bnorm_conf_t conf_;
        offsets off_;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        ref_batch_normalization_init_const_def(
                kernel_ctx, pd()->conf_, pd()->off_);

        std::vector<const char *> kernel_names
                = {"ref_bnorm_fwd", nullptr, nullptr, nullptr, nullptr};
        if (pd()->conf_.use_16mb_unroll && pd()->conf_.calculate_stats) {
            kernel_names[1] = "calculate_mean";
            kernel_names[2] = "calculate_variance";
            kernel_names[3] = "reduce_mean";
            kernel_names[4] = "reduce_variance";
        }

        std::vector<compute::kernel_t> kernels;
        auto status = compute_engine->create_kernels(
                &kernels, kernel_names, kernel_ctx);
        CHECK(status);

        kernel_ = kernels[0];
        calculate_mean_kernel_ = kernels[1];
        calculate_variance_kernel_ = kernels[2];
        reduce_mean_kernel_ = kernels[3];
        reduce_variance_kernel_ = kernels[4];

        return status::success;
    }

    ref_batch_normalization_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
    compute::kernel_t calculate_mean_kernel_;
    compute::kernel_t reduce_mean_kernel_;
    compute::kernel_t calculate_variance_kernel_;
    compute::kernel_t reduce_variance_kernel_;
};

struct ref_batch_normalization_bwd_t : public primitive_impl_t {
    struct pd_t : public ocl_batch_normalization_bwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : ocl_batch_normalization_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conf_()
            , off_() {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_batch_normalization_bwd_t);

        status_t init() {
            using namespace data_type;
            bool ok = true && is_bwd() && set_default_formats_common()
                    && (utils::everyone_is(f32, src_md()->data_type,
                                diff_src_md()->data_type)
                            || utils::everyone_is(bf16, src_md()->data_type,
                                    diff_src_md()->data_type))
                    && IMPLICATION(use_scaleshift(),
                            utils::everyone_is(f32, weights_md()->data_type,
                                    diff_weights_md()->data_type))
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (fuse_norm_relu()) {
                init_default_ws(8);
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            status_t status
                    = ref_batch_normalization_init_conf(conf_, this, off_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            ref_batch_normalization_init_scratchpad(scratchpad, conf_);
            return status::success;
        }

        bnorm_conf_t conf_;
        offsets off_;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        status_t status = ref_batch_normalization_init_const_def(
                kernel_ctx, pd()->conf_, pd()->off_);
        CHECK(status);

        std::vector<const char *> kernel_names
                = {"ref_bnorm_bwd", nullptr, nullptr};

        if (pd()->conf_.use_16mb_unroll) {
            kernel_names[1] = "calculate_stats";
            kernel_names[2] = "reduce_stats";
        }

        std::vector<compute::kernel_t> kernels;
        status = compute_engine->create_kernels(
                &kernels, kernel_names, kernel_ctx);
        CHECK(status);

        kernel_ = kernels[0];
        calculate_stats_kernel_ = kernels[1];
        reduce_stats_kernel_ = kernels[2];

        return status::success;
    }

    ref_batch_normalization_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
    compute::kernel_t calculate_stats_kernel_;
    compute::kernel_t reduce_stats_kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
