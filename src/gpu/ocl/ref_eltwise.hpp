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

#ifndef GPU_OCL_REF_ELTWISE_HPP
#define GPU_OCL_REF_ELTWISE_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_eltwise_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_eltwise_init_conf(
        eltwise_conf_t &conf, const eltwise_pd_t *pd, offsets &off);
status_t ref_eltwise_init_const_def(compute::kernel_ctx_t &kernel_ctx,
        const eltwise_conf_t &conf, const offsets &off);

struct ref_eltwise_fwd_t : public primitive_impl_t {
    struct pd_t : public ocl_eltwise_fwd_pd_t {
        using ocl_eltwise_fwd_pd_t::ocl_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_eltwise_fwd_t);

        status_t init() {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            using namespace alg_kind;
            bool ok = true
                    && utils::one_of(desc()->prop_kind,
                            prop_kind::forward_training,
                            prop_kind::forward_inference)
                    && utils::one_of(desc()->alg_kind, eltwise_relu,
                            eltwise_linear, eltwise_bounded_relu, eltwise_abs,
                            eltwise_tanh, eltwise_elu, eltwise_square,
                            eltwise_sqrt, eltwise_soft_relu, eltwise_logistic,
                            eltwise_exp, eltwise_gelu, eltwise_swish,
                            eltwise_log, eltwise_clip, eltwise_pow,
                            eltwise_relu_use_dst_for_bwd,
                            eltwise_logistic_use_dst_for_bwd,
                            eltwise_tanh_use_dst_for_bwd,
                            eltwise_elu_use_dst_for_bwd,
                            eltwise_sqrt_use_dst_for_bwd,
                            eltwise_exp_use_dst_for_bwd)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::f16, data_type::bf16,
                            data_type::s32, data_type::s8)
                    && attr()->has_default_values()
                    && IMPLICATION(utils::one_of(desc()->data_desc.data_type,
                                           data_type::s32, data_type::s8),
                            desc()->alg_kind == eltwise_relu
                                    && desc()->alpha == 0)
                    && IMPLICATION(
                            desc()->data_desc.data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16));
            if (!ok) return status::unimplemented;

            return ref_eltwise_init_conf(conf_, this, off_);
        }

        eltwise_conf_t conf_;
        offsets off_;
    };

    virtual status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        status_t status = ref_eltwise_init_const_def(
                kernel_ctx, pd()->conf_, pd()->off_);

        if (status != status::success) return status;

        compute_engine->create_kernel(&kernel_, "ref_eltwise_fwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    ref_eltwise_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward_dense(ctx);
    }

private:
    status_t execute_forward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

struct ref_eltwise_bwd_t : public primitive_impl_t {
    struct pd_t : public ocl_eltwise_bwd_pd_t {
        pd_t(engine_t *engine, const eltwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : ocl_eltwise_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_eltwise_bwd_t);

        status_t init() {
            using namespace prop_kind;
            using namespace utils;
            assert(engine()->kind() == engine_kind::gpu);

            using namespace alg_kind;
            bool ok = true && desc()->prop_kind == backward_data
                    && utils::one_of(desc()->alg_kind, eltwise_relu,
                            eltwise_linear, eltwise_bounded_relu, eltwise_abs,
                            eltwise_tanh, eltwise_elu, eltwise_square,
                            eltwise_sqrt, eltwise_soft_relu, eltwise_logistic,
                            eltwise_exp, eltwise_gelu, eltwise_swish,
                            eltwise_log, eltwise_clip, eltwise_pow,
                            eltwise_relu_use_dst_for_bwd,
                            eltwise_logistic_use_dst_for_bwd,
                            eltwise_tanh_use_dst_for_bwd,
                            eltwise_elu_use_dst_for_bwd,
                            eltwise_sqrt_use_dst_for_bwd,
                            eltwise_exp_use_dst_for_bwd)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::bf16)
                    && set_default_formats_common()
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return ref_eltwise_init_conf(conf_, this, off_);
        }

        eltwise_conf_t conf_;
        offsets off_;
        bool use_dense_;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        status_t status = ref_eltwise_init_const_def(
                kernel_ctx, pd()->conf_, pd()->off_);
        if (status != status::success) return status;

        compute_engine->create_kernel(&kernel_, "ref_eltwise_bwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    ref_eltwise_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    ~ref_eltwise_bwd_t() {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_dense(ctx);
    }

private:
    status_t execute_backward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
