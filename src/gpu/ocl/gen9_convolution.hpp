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

#ifndef GPU_OCL_GEN9_CONVOLUTION_HPP
#define GPU_OCL_GEN9_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gen9_convolution_fwd_init_conf(
        conv_conf_t &conf, const convolution_pd_t *pd);
status_t gen9_convolution_fwd_init_const_def(
        compute::kernel_ctx_t &kernel_ctx, const conv_conf_t &conf);

status_t gen9_convolution_bwd_data_init_conf(
        conv_conf_t &conf, const convolution_pd_t *pd);
status_t gen9_convolution_bwd_data_init_const_def(
        compute::kernel_ctx_t &kernel_ctx, const conv_conf_t &conf);

status_t gen9_convolution_bwd_weights_init_conf(
        conv_conf_t &conf, const convolution_pd_t *pd);
status_t gen9_convolution_bwd_weights_init_const_def(
        compute::kernel_ctx_t &kernel_ctx, const conv_conf_t &conf);

struct gen9_convolution_fwd_t : public primitive_impl_t {
    struct pd_t : public gpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conf_() {}

        DECLARE_COMMON_PD_T("ocl:gen9:blocked", gen9_convolution_fwd_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            auto src_data_t = this->desc()->src_desc.data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                            forward_inference)
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(true,
                            expect_data_types(f32, f32, f32, f32, f32),
                            expect_data_types(f16, f16, f16, f16, f16))
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(src_data_t == f16,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short))
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_ok(attr());
            if (!ok) return status::unimplemented;

            status_t status = gen9_convolution_fwd_init_conf(conf_, this);
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    conf_.src_tag, conf_.wei_tag, conf_.dst_tag);
            return ok ? status::success : status::unimplemented;
        }
        conv_conf_t conf_;
    };

    status_t init() override {
        const char *kernel_name = nullptr;
        if (pd()->conf_.is_depthwise)
            kernel_name = "gen9_conv_dw_fwd";
        else if (pd()->desc()->src_desc.data_type == data_type::f16)
            kernel_name = "gen9_conv_fwd_f16";
        else if (pd()->desc()->src_desc.data_type == data_type::f32)
            kernel_name = "gen9_conv_fwd_f32";
        else
            assert(!"not expected");

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());

        compute::kernel_ctx_t kernel_ctx;
        status_t status
                = gen9_convolution_fwd_init_const_def(kernel_ctx, pd()->conf_);
        if (status != status::success) return status;

        compute_engine->create_kernel(&kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    gen9_convolution_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

struct gen9_convolution_bwd_data_t : public primitive_impl_t {
    struct pd_t : public gpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ncsp:any", gen9_convolution_bwd_data_t);

        status_t init() {
            using namespace data_type;
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && this->desc()->prop_kind == backward_data
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(true,
                            expect_data_types(
                                    f32, f32, data_type::undef, f32, f32),
                            expect_data_types(
                                    f16, f16, data_type::undef, f16, f16))
                    && (IMPLICATION(this->with_bias()
                                        && this->desc()->diff_dst_desc.data_type
                                                != f16,
                                this->desc()->bias_desc.data_type == f32)
                            || IMPLICATION(this->with_bias()
                                            && this->desc()->diff_dst_desc
                                                            .data_type
                                                    == f16,
                                    this->desc()->bias_desc.data_type == f16))
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && !has_zero_dim_memory() && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            status_t status = gen9_convolution_bwd_data_init_conf(conf_, this);
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    conf_.src_tag, conf_.wei_tag, conf_.dst_tag);
            return ok ? status::success : status::unimplemented;
        }
        conv_conf_t conf_;
    };

    status_t init() override {
        const char *kernel_name = nullptr;
        if (pd()->conf_.is_depthwise)
            kernel_name = "gen9_conv_dw_bwd_data";
        else
            kernel_name = "gen9_conv_bwd_data";

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());

        compute::kernel_ctx_t kernel_ctx;
        status_t status = gen9_convolution_bwd_data_init_const_def(
                kernel_ctx, pd()->conf_);
        if (status != status::success) return status;

        compute_engine->create_kernel(&kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    gen9_convolution_bwd_data_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

struct gen9_convolution_bwd_weights_t : public primitive_impl_t {
    struct pd_t : public gpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_weights_pd_t(
                    engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ncsp:any", gen9_convolution_bwd_weights_t);

        status_t init() {
            using namespace data_type;
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            bool ok = set_default_alg_kind(alg_kind::convolution_direct)
                    && this->desc()->prop_kind == backward_weights
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && expect_data_types(f32, f32, f32, f32, f32)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::khr_int64_base_atomics)
                    && !has_zero_dim_memory() && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            status_t status
                    = gen9_convolution_bwd_weights_init_conf(conf_, this);
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    conf_.src_tag, conf_.wei_tag, conf_.dst_tag);

            return ok ? status::success : status::unimplemented;
        }
        conv_conf_t conf_;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());

        compute::kernel_ctx_t kernel_ctx;
        status_t status = gen9_convolution_bwd_weights_init_const_def(
                kernel_ctx, pd()->conf_);
        if (status != status::success) return status;

        compute_engine->create_kernel(
                &kernel_, "gen9_conv_bwd_weights", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    gen9_convolution_bwd_weights_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
