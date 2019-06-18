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

#ifndef OCL_REF_INNER_PRODUCT_HPP
#define OCL_REF_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "ocl/cl_engine.hpp"
#include "ocl/jit_ref_inner_product_common_kernel.hpp"
#include "ocl/ocl_inner_product_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

extern const char *ref_inner_product_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type,
        impl::data_type_t acc_type = dst_type>
struct ref_inner_product_fwd_t : public primitive_t {
    struct pd_t : public ocl_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : ocl_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jit_off_() {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_inner_product_fwd_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine()->kind() == engine_kind::gpu);
            auto *cl_engine = utils::downcast<cl_engine_t *>(engine());

            bool ok = true
                    && utils::one_of(desc()->prop_kind, forward_training,
                               forward_inference)
                    && set_default_params() == status::success
                    && desc()->src_desc.data_type == src_type
                    && desc()->weights_desc.data_type == wei_type
                    && desc()->accum_data_type == acc_type
                    && desc()->dst_desc.data_type == dst_type
                    && utils::everyone_is(desc()->src_desc.data_type,
                               desc()->weights_desc.data_type,
                               desc()->accum_data_type,
                               desc()->dst_desc.data_type)
                    && utils::one_of(desc()->src_desc.data_type, f16, f32)
                    && IMPLICATION(with_bias(),
                               utils::one_of(
                                       desc()->bias_desc.data_type, f32, f16))
                    && attr()->output_scales_.has_default_values()
                    && dense_consitency_check(
                               src_md(), weights_md(), dst_md())
                    && IMPLICATION(src_type == data_type::f16,
                               cl_engine->mayiuse(cl_device_ext_t::khr_fp16));
            if (!ok)
                return status::unimplemented;

            return jit_ref_inner_product_fwd_kernel::init_conf(jip_, desc_,
                    src_md(), weights_md(), dst_md(), *this->attr(), jit_off_);
        }
        bool with_eltwise() const {
            return attr()->post_ops_.find(primitive_kind::eltwise) != -1;
        }

        bool with_sum() const {
            return attr()->post_ops_.find(primitive_kind::sum) != -1;
        }

        float eltwise_alpha() const {
            const int eltwise_idx =
                attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx =
                attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                : 0.0f;
        }

        float sum_scale() const {
            const int sum_idx =
                attr()->post_ops_.find(primitive_kind::sum);
            return with_sum()
                ? attr()->post_ops_.entry_[sum_idx].sum.scale
                : 1.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx =
                attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                : mkldnn_alg_kind_undef;
        }

        jit_inner_product_conf_t jip_;
        jit_offsets jit_off_;
    };

    status_t init() override {
        auto jit = ocl_jit_t(ref_inner_product_kernel);
        jit_ref_inner_product_fwd_kernel::init_const_def(
                jit, pd()->jip_, pd()->jit_off_, pd()->with_eltwise(),
                pd()->with_sum(), pd()->eltwise_alg_kind());

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_inner_product_fwd_kernel");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    ref_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {
        ker_ = new jit_ref_inner_product_fwd_kernel(pd()->jip_);
    }
    ~ref_inner_product_fwd_t() { delete ker_; }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_ref_inner_product_fwd_kernel *ker_;
    ocl_kernel_t kernel_;
};

template <impl::data_type_t diff_src_type, impl::data_type_t wei_type,
        impl::data_type_t diff_dst_type,
        impl::data_type_t acc_type = diff_src_type>
struct ref_inner_product_bwd_data_t : public primitive_t {
    struct pd_t : public ocl_inner_product_bwd_data_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : ocl_inner_product_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jit_off_() {}

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_data_t);

        status_t init() {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::gpu);
            bool ok = true
                    && utils::one_of(
                               this->desc()->prop_kind, backward, backward_data)
                    && this->set_default_params() == status::success
                    && desc()->diff_src_desc.data_type == diff_src_type
                    && desc()->weights_desc.data_type == wei_type
                    && desc()->accum_data_type == acc_type
                    && desc()->diff_dst_desc.data_type == diff_dst_type
                    && attr()->has_default_values();
            if (!ok)
                return status::unimplemented;

            return jit_ref_inner_product_fwd_kernel::init_conf(jip_, desc_,
                    diff_src_md(), weights_md(), diff_dst_md(), *this->attr(),
                    jit_off_);
        }
        jit_inner_product_conf_t jip_;
        jit_offsets jit_off_;
    };

    virtual status_t init() override {
        auto jit = ocl_jit_t(ref_inner_product_kernel);
        jit_ref_inner_product_fwd_kernel::init_const_def(
                jit, pd()->jip_, pd()->jit_off_, false, false,
                mkldnn_alg_kind_undef);

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_inner_product_bwd_data_kernel");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    ref_inner_product_bwd_data_t(const pd_t *apd) : primitive_t(apd) {
        ker_ = new jit_ref_inner_product_fwd_kernel(pd()->jip_);
    }
    ~ref_inner_product_bwd_data_t() { delete ker_; }

    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_ref_inner_product_fwd_kernel *ker_;
    ocl_kernel_t kernel_;
};

template <impl::data_type_t data_type>
struct ref_inner_product_bwd_weights_t : public primitive_t {
    struct pd_t : public ocl_inner_product_bwd_weights_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : ocl_inner_product_bwd_weights_pd_t(
                      engine, adesc, attr, hint_fwd_pd)
            , jit_off_() {}

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_weights_t);

        status_t init() {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::gpu);
            bool ok = true
                    && utils::one_of(this->desc()->prop_kind, backward,
                               backward_weights)
                    && this->set_default_params() == status::success
                    && utils::everyone_is(data_type,
                               this->desc()->src_desc.data_type,
                               this->desc()->diff_dst_desc.data_type,
                               this->desc()->diff_weights_desc.data_type)
                    && IMPLICATION(this->with_bias(),
                               data_type
                                       == this->desc()
                                                  ->diff_bias_desc.data_type)
                    && attr()->has_default_values();
            if (!ok)
                return status::unimplemented;

            return jit_ref_inner_product_fwd_kernel::init_conf(jip_, desc_,
                    src_md(), diff_weights_md(), diff_dst_md(), *this->attr(),
                    jit_off_);
        }
        jit_inner_product_conf_t jip_;
        jit_offsets jit_off_;
    };

    status_t init() override {
        auto jit = ocl_jit_t(ref_inner_product_kernel);
        jit_ref_inner_product_fwd_kernel::init_const_def(
                jit, pd()->jip_, pd()->jit_off_, false, false,
                mkldnn_alg_kind_undef);

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_inner_product_bwd_weights_kernel");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    ref_inner_product_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {
        ker_ = new jit_ref_inner_product_fwd_kernel(pd()->jip_);
    }
    ~ref_inner_product_bwd_weights_t() { delete ker_; }

    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_ref_inner_product_fwd_kernel *ker_;
    ocl_kernel_t kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
