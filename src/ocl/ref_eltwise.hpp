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

#ifndef OCL_REF_ELTWISE_HPP
#define OCL_REF_ELTWISE_HPP

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_ref_eltwise_common_kernel.hpp"
#include "ocl/ocl_eltwise_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

extern const char *ref_eltwise_kernel;

namespace dnnl {
namespace impl {
namespace ocl {

struct ref_eltwise_fwd_t : public primitive_impl_t {
    struct pd_t : public ocl_eltwise_fwd_pd_t {
        using ocl_eltwise_fwd_pd_t::ocl_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_eltwise_fwd_t);

        status_t init() {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            bool ok = true
                    && utils::one_of(desc()->prop_kind,
                            prop_kind::forward_training,
                            prop_kind::forward_inference)
                    && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu,
                            alg_kind::eltwise_linear,
                            alg_kind::eltwise_bounded_relu,
                            alg_kind::eltwise_abs, alg_kind::eltwise_tanh,
                            alg_kind::eltwise_elu, alg_kind::eltwise_square,
                            alg_kind::eltwise_sqrt, alg_kind::eltwise_soft_relu,
                            alg_kind::eltwise_logistic, alg_kind::eltwise_exp,
                            alg_kind::eltwise_gelu, alg_kind::eltwise_swish,
                            alg_kind::eltwise_log)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::f16, data_type::bf16,
                            data_type::s32, data_type::s8)
                    && attr()->has_default_values()
                    && IMPLICATION(utils::one_of(desc()->data_desc.data_type,
                                           data_type::s32, data_type::s8),
                            desc()->alg_kind == alg_kind::eltwise_relu
                                    && desc()->alpha == 0)
                    && IMPLICATION(
                            desc()->data_desc.data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16));
            if (!ok) return status::unimplemented;

            return jit_ref_eltwise_common_kernel::init_conf(jel_, data_md_,
                    glob_zero_md, jit_off_, desc()->alg_kind, true);
        }

        jit_eltwise_conf_t jel_;
        jit_offsets jit_off_;
    };

    virtual status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        status_t status = jit_ref_eltwise_common_kernel::init_const_def(
                kernel_ctx, pd()->jel_, pd()->jit_off_);

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

            bool ok = true && desc()->prop_kind == backward_data
                    && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu,
                            alg_kind::eltwise_linear,
                            alg_kind::eltwise_bounded_relu,
                            alg_kind::eltwise_abs, alg_kind::eltwise_tanh,
                            alg_kind::eltwise_elu, alg_kind::eltwise_square,
                            alg_kind::eltwise_sqrt, alg_kind::eltwise_soft_relu,
                            alg_kind::eltwise_logistic, alg_kind::eltwise_exp,
                            alg_kind::eltwise_gelu, alg_kind::eltwise_swish,
                            alg_kind::eltwise_log)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::bf16)
                    && set_default_formats_common()
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return jit_ref_eltwise_common_kernel::init_conf(jel_, data_md_,
                    diff_data_md_, jit_off_, desc()->alg_kind, false);
        }

        jit_eltwise_conf_t jel_;
        jit_offsets jit_off_;
        bool use_dense_;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        status_t status = jit_ref_eltwise_common_kernel::init_const_def(
                kernel_ctx, pd()->jel_, pd()->jit_off_);
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
} // namespace impl
} // namespace dnnl

#endif
