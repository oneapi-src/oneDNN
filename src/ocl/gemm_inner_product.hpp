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

#ifndef OCL_GEMM_INNER_PRODUCT_HPP
#define OCL_GEMM_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "ocl/ocl_inner_product_pd.hpp"

extern const char *gemm_inner_product_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

namespace {
status_t create_gemm_pd(primitive_desc_t **gemm_pd, engine_t *engine,
        transpose_t transa, transpose_t transb, int m, int n, int k,
        int lda, int ldb, int ldc, data_type_t a_dt, data_type_t b_dt,
        data_type_t c_dt, float alpha, float beta,
        const primitive_attr_t &attr) {
    gemm_desc_t gemm_desc;
    gemm_desc.primitive_kind = primitive_kind::gemm;
    gemm_desc.transa = transa;
    gemm_desc.transb = transb;
    gemm_desc.m = m;
    gemm_desc.n = n;
    gemm_desc.k = k;
    gemm_desc.lda = lda;
    gemm_desc.ldb = ldb;
    gemm_desc.ldc = ldc;
    gemm_desc.alpha = alpha;
    gemm_desc.beta = beta;
    gemm_desc.a_type = a_dt;
    gemm_desc.b_type = b_dt;
    gemm_desc.c_type = c_dt;

    op_desc_t op_desc(gemm_desc);

    return mkldnn_primitive_desc_create(gemm_pd, &op_desc, &attr, engine,
            nullptr);
}
}

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type,
        impl::data_type_t acc_type = dst_type>
struct gemm_inner_product_fwd_t : public primitive_t {
    struct pd_t : public ocl_inner_product_fwd_pd_t {
        using ocl_inner_product_fwd_pd_t::ocl_inner_product_fwd_pd_t;

        pd_t(const pd_t &rhs): ocl_inner_product_fwd_pd_t(rhs) {
            gemm_pd_ = rhs.gemm_pd_->clone();
        }
        ~pd_t() { delete gemm_pd_; }

        pd_t &operator=(const pd_t &rhs) {
            MKLDNN_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            delete gemm_pd_;
            gemm_pd_ = rhs.gemm_pd_->clone();
            return *this;
        }

        DECLARE_COMMON_PD_T("ocl:gemm", gemm_inner_product_fwd_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);

            bool with_eltwise =  true
                && attr()->output_scales_.has_default_values()
                && attr()->post_ops_.find(primitive_kind::eltwise) != -1;
            bool with_sum = attr()->post_ops_.find(primitive_kind::sum) != -1;

            bool ok = true
                && set_default_params() == status::success
                && is_fwd()
                && !has_zero_dim_memory()
                && src_md()->data_type == src_type
                && weights_md()->data_type == wei_type
                && dst_md()->data_type == dst_type
                && (attr()->has_default_values()
                        || IMPLICATION(with_eltwise, !with_bias()))
                && IMPLICATION(src_type == f16, !with_eltwise)
                && !with_sum
                && dense_consitency_check(src_md(), weights_md(), dst_md())
                && dense_gemm_consitency_check(src_md(), weights_md(),
                        dst_md());
            if (!ok)
                return status::unimplemented;

            const auto &wmd = *this->weights_md();
            bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = status::success == create_gemm_pd(&gemm_pd_,
                    this->engine(),
                    wei_tr ? transpose::trans : transpose::notrans,
                    transpose::notrans, oc, mb, ic_total,
                    wei_tr ? ic_total : oc, ic_total, oc, wei_type,
                    src_type, dst_type, 1.0, 0.0, *attr());
            if (!gemm_ok)
                return status::unimplemented;

            return status::success;
        }

        primitive_desc_t *gemm_pd_ = nullptr;
    };

    status_t init() override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(&gemm_);
        if (gemm_status != status::success)
            return gemm_status;

        if (pd()->with_bias()) {
            auto jit = ocl_jit_t(gemm_inner_product_kernel);

            jit.set_data_type(src_type);
            jit.define_int("MB", pd()->MB());
            jit.define_int("OC", pd()->OC());

            status_t bias_kernel_status = jit.build(engine());
            if (bias_kernel_status != status::success)
                return bias_kernel_status;

            bias_kernel_
                = jit.get_kernel("gemm_inner_product_forward_bias");
            if (!bias_kernel_)
                return status::runtime_error;
        }

        return status::success;
    }

    gemm_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    ~gemm_inner_product_fwd_t() { delete gemm_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    primitive_t *gemm_ = nullptr;
    ocl_kernel_t bias_kernel_;
};

template <impl::data_type_t diff_src_type,
        impl::data_type_t wei_type = diff_src_type,
        impl::data_type_t diff_dst_type = diff_src_type,
        impl::data_type_t acc_type = diff_src_type>
struct gemm_inner_product_bwd_data_t : public primitive_t {
    struct pd_t : public ocl_inner_product_bwd_data_pd_t {
        using ocl_inner_product_bwd_data_pd_t::ocl_inner_product_bwd_data_pd_t;

        pd_t(const pd_t &rhs): ocl_inner_product_bwd_data_pd_t(rhs) {
            gemm_pd_ = rhs.gemm_pd_->clone();
        }
        ~pd_t() { delete gemm_pd_; }

        pd_t &operator=(const pd_t &rhs) {
            MKLDNN_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            delete gemm_pd_;
            gemm_pd_ = rhs.gemm_pd_->clone();
            return *this;
        }

        DECLARE_COMMON_PD_T("ocl:gemm", gemm_inner_product_bwd_data_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);

            bool ok = true
                && set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && !has_zero_dim_memory()
                && diff_src_md()->data_type == diff_src_type
                && weights_md()->data_type == wei_type
                && diff_dst_md()->data_type == diff_dst_type
                && attr()->has_default_values()
                && dense_consitency_check(diff_src_md(), weights_md(),
                        diff_dst_md())
                && dense_gemm_consitency_check(diff_src_md(), weights_md(),
                        diff_dst_md());
            if (!ok)
                return status::unimplemented;

            const auto &wmd = *this->weights_md();
            bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = status::success == create_gemm_pd(&gemm_pd_,
                    this->engine(),
                    wei_tr ? transpose::trans : transpose::notrans,
                    transpose::notrans, ic_total, mb, oc,
                    wei_tr ? oc : ic_total, oc, ic_total, wei_type,
                    diff_src_type, diff_dst_type, 1.0, 0.0, *attr());
            if (!gemm_ok)
                return status::unimplemented;

            return status::success;
        }

        primitive_desc_t *gemm_pd_ = nullptr;
    };

    status_t init() override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(&gemm_);
        if (gemm_status != status::success)
            return gemm_status;

        return status::success;
    }

    gemm_inner_product_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}
    ~gemm_inner_product_bwd_data_t() { delete gemm_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    primitive_t *gemm_ = nullptr;
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_weights_t : public primitive_t {
    using ocl_ip_bwd_weights_pd_t = ocl_inner_product_bwd_weights_pd_t;
    struct pd_t : public ocl_ip_bwd_weights_pd_t {
        using ocl_ip_bwd_weights_pd_t::ocl_ip_bwd_weights_pd_t;

        pd_t(const pd_t &rhs): ocl_ip_bwd_weights_pd_t(rhs) {
            gemm_pd_ = rhs.gemm_pd_->clone();
        }
        ~pd_t() { delete gemm_pd_; }

        pd_t &operator=(const pd_t &rhs) {
            MKLDNN_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            delete gemm_pd_;
            gemm_pd_ = rhs.gemm_pd_->clone();
            return *this;
        }

        DECLARE_COMMON_PD_T("gemm:ocl", gemm_inner_product_bwd_weights_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);

            bool ok = true
                && set_default_params() == status::success
                && this->desc()->prop_kind == backward_weights
                && !has_zero_dim_memory()
                && src_md()->data_type == data_type
                && diff_weights_md()->data_type == data_type
                && diff_dst_md()->data_type == data_type
                && attr()->has_default_values()
                && dense_consitency_check(src_md(), diff_weights_md(),
                        diff_dst_md())
                && dense_gemm_consitency_check(src_md(), diff_weights_md(),
                        diff_dst_md());
            if (!ok)
                return status::unimplemented;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = false;
            if (wei_tr()) {
                gemm_ok = create_gemm_pd(&gemm_pd_, this->engine(),
                        transpose::notrans, transpose::trans, oc, ic_total, mb,
                        oc, ic_total, oc, data_type, data_type, data_type,
                        1.0, 0.0, *attr()) == status::success;
            } else {
                gemm_ok = create_gemm_pd(&gemm_pd_, this->engine(),
                        transpose::notrans, transpose::trans, ic_total, oc, mb,
                        ic_total, oc, ic_total, data_type, data_type,
                        data_type, 1.0, 0.0, *attr()) == status::success;
            }
            if (!gemm_ok)
                return status::unimplemented;

            return status::success;
        }

        bool wei_tr() const {
            const auto &wmd = *this->diff_weights_md();
            return wmd.format_desc.blocking.strides[0] == 1;
        }

        primitive_desc_t *gemm_pd_ = nullptr;
    };

    status_t init() override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(&gemm_);
        if (gemm_status != status::success)
            return gemm_status;

        if (pd()->with_bias()) {
            auto jit = ocl_jit_t(gemm_inner_product_kernel);

            jit.set_data_type(data_type);
            jit.define_int("MB", pd()->MB());
            jit.define_int("OC", pd()->OC());

            status_t bias_kernel_status = jit.build(engine());
            if (bias_kernel_status != status::success)
                return bias_kernel_status;

            bias_kernel_
                = jit.get_kernel("gemm_inner_product_backward_weights_bias");
            if (!bias_kernel_)
                return status::runtime_error;
        }

        return status::success;
    }

    gemm_inner_product_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}
    ~gemm_inner_product_bwd_weights_t() { delete gemm_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *gemm_ = nullptr;
    ocl_kernel_t bias_kernel_;
};

}
}
}

#endif
