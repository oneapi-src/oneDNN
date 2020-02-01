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

#ifndef GPU_OCL_JIT_GEN9_COMMON_CONVOLUTION_HPP
#define GPU_OCL_JIT_GEN9_COMMON_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/jit_primitive_conf.hpp"
#include "gpu/ocl/ocl_convolution_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t jit_gen9_convolution_fwd_init_conf(
        jit_conv_conf_t &jcp, const convolution_pd_t *pd);
status_t jit_gen9_convolution_fwd_init_const_def(
        compute::kernel_ctx_t &kernel_ctx, const jit_conv_conf_t &jcp);

status_t jit_gen9_convolution_bwd_data_init_conf(
        jit_conv_conf_t &jcp, const convolution_pd_t *pd);
status_t jit_gen9_convolution_bwd_data_init_const_def(
        compute::kernel_ctx_t &kernel_ctx, const jit_conv_conf_t &jcp);

status_t jit_gen9_convolution_bwd_weights_init_conf(
        jit_conv_conf_t &jcp, const convolution_pd_t *pd);
status_t jit_gen9_convolution_bwd_weights_init_const_def(
        compute::kernel_ctx_t &kernel_ctx, const jit_conv_conf_t &jcp);

struct jit_gen9_common_convolution_fwd_t : public primitive_impl_t {
    struct pd_t : public ocl_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                "ocl:gen9:blocked", jit_gen9_common_convolution_fwd_t);

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

            status_t status = jit_gen9_convolution_fwd_init_conf(jcp_, this);
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    jcp_.src_tag, jcp_.wei_tag, jcp_.dst_tag);
            return ok ? status::success : status::unimplemented;
        }
        jit_conv_conf_t jcp_;
    };

    status_t init() override {
        const char *kernel_name = nullptr;
        if (pd()->jcp_.is_depthwise)
            kernel_name = "gen9_common_conv_dw_fwd";
        else if (pd()->desc()->src_desc.data_type == data_type::f16)
            kernel_name = "gen9_common_conv_fwd_f16";
        else if (pd()->desc()->src_desc.data_type == data_type::f32)
            kernel_name = "gen9_common_conv_fwd_f32";
        else
            assert(!"not expected");

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());

        compute::kernel_ctx_t kernel_ctx;
        status_t status = jit_gen9_convolution_fwd_init_const_def(
                kernel_ctx, pd()->jcp_);
        if (status != status::success) return status;

        compute_engine->create_kernel(&kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    jit_gen9_common_convolution_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

struct jit_gen9_common_convolution_bwd_data_t : public primitive_impl_t {
    struct pd_t : public ocl_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                "ocl:ncsp:any", jit_gen9_common_convolution_bwd_data_t);

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

            status_t status
                    = jit_gen9_convolution_bwd_data_init_conf(jcp_, this);
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    jcp_.src_tag, jcp_.wei_tag, jcp_.dst_tag);
            return ok ? status::success : status::unimplemented;
        }
        jit_conv_conf_t jcp_;
    };

    status_t init() override {
        const char *kernel_name = nullptr;
        if (pd()->jcp_.is_depthwise)
            kernel_name = "gen9_common_conv_dw_bwd_data";
        else
            kernel_name = "gen9_common_conv_bwd_data";

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());

        compute::kernel_ctx_t kernel_ctx;
        status_t status = jit_gen9_convolution_bwd_data_init_const_def(
                kernel_ctx, pd()->jcp_);
        if (status != status::success) return status;

        compute_engine->create_kernel(&kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    jit_gen9_common_convolution_bwd_data_t(const pd_t *apd)
        : primitive_impl_t(apd) {
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

struct jit_gen9_common_convolution_bwd_weights_t : public primitive_impl_t {
    struct pd_t : public ocl_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : ocl_convolution_bwd_weights_pd_t(
                    engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                "ocl:ncsp:any", jit_gen9_common_convolution_bwd_weights_t);

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
                    = jit_gen9_convolution_bwd_weights_init_conf(jcp_, this);
            if (status != status::success) return status;

            ok = set_default_formats_common(
                    jcp_.src_tag, jcp_.wei_tag, jcp_.dst_tag);

            return ok ? status::success : status::unimplemented;
        }
        jit_conv_conf_t jcp_;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());

        compute::kernel_ctx_t kernel_ctx;
        status_t status = jit_gen9_convolution_bwd_weights_init_const_def(
                kernel_ctx, pd()->jcp_);
        if (status != status::success) return status;

        compute_engine->create_kernel(
                &kernel_, "gen9_common_conv_bwd_weights", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    jit_gen9_common_convolution_bwd_weights_t(const pd_t *apd)
        : primitive_impl_t(apd) {
    }

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
