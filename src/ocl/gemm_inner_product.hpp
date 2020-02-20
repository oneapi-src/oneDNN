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

#ifndef OCL_GEMM_INNER_PRODUCT_HPP
#define OCL_GEMM_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/ocl_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

namespace {
status_t create_gemm_pd(primitive_desc_t **gemm_pd, engine_t *engine,
        transpose_t transa, transpose_t transb, int m, int n, int k, int lda,
        int ldb, int ldc, data_type_t a_dt, data_type_t b_dt, data_type_t c_dt,
        float alpha, float beta, const primitive_attr_t &attr) {
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
    gemm_desc.acc_type = c_dt;

    return dnnl_primitive_desc_create(
            gemm_pd, (op_desc_t *)&gemm_desc, &attr, engine, nullptr);
}
} // namespace

struct gemm_inner_product_fwd_t : public primitive_impl_t {
    struct pd_t : public ocl_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : ocl_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : ocl_inner_product_fwd_pd_t(rhs) {
            gemm_pd_ = rhs.gemm_pd_->clone();
        }
        ~pd_t() { delete gemm_pd_; }

        pd_t &operator=(const pd_t &rhs) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            ocl_inner_product_fwd_pd_t::operator=(rhs);
            delete gemm_pd_;
            gemm_pd_ = rhs.gemm_pd_->clone();
            return *this;
        }

        DECLARE_COMMON_PD_T("ocl:gemm", gemm_inner_product_fwd_t);

        status_t init() {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool with_eltwise
                    = attr()->post_ops_.find(primitive_kind::eltwise) != -1;
            bool with_sum = attr()->post_ops_.find(primitive_kind::sum) != -1;

            bool ok = true && is_fwd()
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && utils::one_of(true,
                            expect_data_types(f16, f16, f16, f16, f16),
                            expect_data_types(f32, f32, f32, f32, f32))
                    && attr()->has_default_values(attr_skip_mask)
                    && attr()->post_ops_.len_ <= 1
                    && IMPLICATION(with_eltwise, !with_bias()) && !with_sum
                    && dense_consitency_check(src_md(), weights_md(), dst_md())
                    && dense_gemm_consitency_check(
                            src_md(), weights_md(), dst_md());
            if (!ok) return status::unimplemented;

            const auto &wmd = *this->weights_md();
            bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = status::success
                    == create_gemm_pd(&gemm_pd_, this->engine(),
                            wei_tr ? transpose::trans : transpose::notrans,
                            transpose::notrans, oc, mb, ic_total,
                            wei_tr ? ic_total : oc, ic_total, oc,
                            weights_md()->data_type, src_md()->data_type,
                            dst_md()->data_type, 1.0, 0.0, *attr());
            if (!gemm_ok) return status::unimplemented;

            return status::success;
        }

        primitive_desc_t *gemm_pd_ = nullptr;
    };

    status_t init() override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(&gemm_);
        if (gemm_status != status::success) return gemm_status;

        if (pd()->with_bias()) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());
            compute::kernel_ctx_t kernel_ctx;

            kernel_ctx.set_data_type(pd()->src_md()->data_type);
            kernel_ctx.define_int("MB", pd()->MB());
            kernel_ctx.define_int("OC", pd()->OC());

            compute_engine->create_kernel(&bias_kernel_,
                    "gemm_inner_product_forward_bias", kernel_ctx);
            if (!bias_kernel_) return status::runtime_error;
        }

        return status::success;
    }

    gemm_inner_product_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}
    ~gemm_inner_product_fwd_t() { gemm_->release(); }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    primitive_t *gemm_ = nullptr;
    compute::kernel_t bias_kernel_;
};

struct gemm_inner_product_bwd_data_t : public primitive_impl_t {
    struct pd_t : public ocl_inner_product_bwd_data_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : ocl_inner_product_bwd_data_pd_t(
                    engine, adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : ocl_inner_product_bwd_data_pd_t(rhs) {
            gemm_pd_ = rhs.gemm_pd_->clone();
        }
        ~pd_t() { delete gemm_pd_; }

        pd_t &operator=(const pd_t &rhs) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            ocl_inner_product_bwd_data_pd_t::operator=(rhs);
            delete gemm_pd_;
            gemm_pd_ = rhs.gemm_pd_->clone();
            return *this;
        }

        DECLARE_COMMON_PD_T("ocl:gemm", gemm_inner_product_bwd_data_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);

            bool ok = true && this->desc()->prop_kind == backward_data
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && expect_data_types(f32, f32, data_type::undef, f32, f32)
                    && attr()->has_default_values()
                    && dense_consitency_check(
                            diff_src_md(), weights_md(), diff_dst_md())
                    && dense_gemm_consitency_check(
                            diff_src_md(), weights_md(), diff_dst_md());
            if (!ok) return status::unimplemented;

            const auto &wmd = *this->weights_md();
            bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = status::success
                    == create_gemm_pd(&gemm_pd_, this->engine(),
                            wei_tr ? transpose::trans : transpose::notrans,
                            transpose::notrans, ic_total, mb, oc,
                            wei_tr ? oc : ic_total, oc, ic_total,
                            weights_md()->data_type, diff_src_md()->data_type,
                            diff_dst_md()->data_type, 1.0, 0.0, *attr());
            if (!gemm_ok) return status::unimplemented;

            return status::success;
        }

        primitive_desc_t *gemm_pd_ = nullptr;
    };

    status_t init() override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(&gemm_);
        if (gemm_status != status::success) return gemm_status;

        return status::success;
    }

    gemm_inner_product_bwd_data_t(const pd_t *apd) : primitive_impl_t(apd) {}
    ~gemm_inner_product_bwd_data_t() { gemm_->release(); }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    primitive_t *gemm_ = nullptr;
};

struct gemm_inner_product_bwd_weights_t : public primitive_impl_t {
    using ocl_ip_bwd_weights_pd_t = ocl_inner_product_bwd_weights_pd_t;
    struct pd_t : public ocl_ip_bwd_weights_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : ocl_ip_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : ocl_ip_bwd_weights_pd_t(rhs) {
            gemm_pd_ = rhs.gemm_pd_->clone();
        }
        ~pd_t() { delete gemm_pd_; }

        pd_t &operator=(const pd_t &rhs) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            ocl_ip_bwd_weights_pd_t::operator=(rhs);
            delete gemm_pd_;
            gemm_pd_ = rhs.gemm_pd_->clone();
            return *this;
        }

        DECLARE_COMMON_PD_T("gemm:ocl", gemm_inner_product_bwd_weights_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);

            bool ok = true && this->desc()->prop_kind == backward_weights
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && expect_data_types(f32, f32, f32, f32, f32)
                    && attr()->has_default_values()
                    && dense_consitency_check(
                            src_md(), diff_weights_md(), diff_dst_md())
                    && dense_gemm_consitency_check(
                            src_md(), diff_weights_md(), diff_dst_md());
            if (!ok) return status::unimplemented;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = false;
            if (wei_tr()) {
                gemm_ok = create_gemm_pd(&gemm_pd_, this->engine(),
                                  transpose::notrans, transpose::trans, oc,
                                  ic_total, mb, oc, ic_total, oc,
                                  src_md()->data_type, src_md()->data_type,
                                  src_md()->data_type, 1.0, 0.0, *attr())
                        == status::success;
            } else {
                gemm_ok = create_gemm_pd(&gemm_pd_, this->engine(),
                                  transpose::notrans, transpose::trans,
                                  ic_total, oc, mb, ic_total, oc, ic_total,
                                  src_md()->data_type, src_md()->data_type,
                                  src_md()->data_type, 1.0, 0.0, *attr())
                        == status::success;
            }
            if (!gemm_ok) return status::unimplemented;

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
        if (gemm_status != status::success) return gemm_status;

        if (pd()->with_bias()) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());
            compute::kernel_ctx_t kernel_ctx;

            kernel_ctx.set_data_type(pd()->src_md()->data_type);
            kernel_ctx.define_int("MB", pd()->MB());
            kernel_ctx.define_int("OC", pd()->OC());

            compute_engine->create_kernel(&bias_kernel_,
                    "gemm_inner_product_backward_weights_bias", kernel_ctx);
            if (!bias_kernel_) return status::runtime_error;
        }

        return status::success;
    }

    gemm_inner_product_bwd_weights_t(const pd_t *apd) : primitive_impl_t(apd) {}
    ~gemm_inner_product_bwd_weights_t() { gemm_->release(); }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    primitive_t *gemm_ = nullptr;
    compute::kernel_t bias_kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif
