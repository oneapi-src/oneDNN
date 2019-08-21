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

#ifndef OCL_GEMM_X8S8S32X_INNER_PRODUCT_HPP
#define OCL_GEMM_X8S8S32X_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/gemm_types.hpp"
#include "compute/compute.hpp"
#include "ocl/ocl_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

namespace {
// FIXME: should be common for all data types
inline status_t create_gemm_x8s8s32x_pd(primitive_desc_t **gemm_pd,
        engine_t *engine, transpose_t transa, transpose_t transb, int m, int n,
        int k, int lda, int ldb, int ldc, data_type_t a_dt, data_type_t b_dt,
        data_type_t c_dt, float alpha, float beta,
        const primitive_attr_t &attr) {
    gemm_desc_t gemm_desc;
    gemm_desc.primitive_kind = primitive_kind::gemm;
    gemm_desc.transa = transa;
    gemm_desc.transb = transb;
    gemm_desc.offsetc = offsetc::fixed; // FIXME
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
    gemm_desc.ao = 0; // FIXME
    gemm_desc.bo = 0; // FIXME

    return dnnl_primitive_desc_create(
            gemm_pd, (op_desc_t *)&gemm_desc, &attr, engine, nullptr);
}
} // namespace

struct gemm_x8s8s32x_inner_product_fwd_t : public primitive_impl_t {
    struct pd_t : public ocl_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : ocl_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : ocl_inner_product_fwd_pd_t(rhs) {
            if (rhs.gemm_pd_) gemm_pd_ = rhs.gemm_pd_->clone();
        }
        ~pd_t() { delete gemm_pd_; }

        pd_t &operator=(const pd_t &rhs) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            delete gemm_pd_;
            if (rhs.gemm_pd_) gemm_pd_ = rhs.gemm_pd_->clone();
            return *this;
        }

        DECLARE_COMMON_PD_T(
                "ocl:gemm_x8s8s32x", gemm_x8s8s32x_inner_product_fwd_t);

        status_t init() {
            using namespace status;
            using namespace utils;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);

            bool ok = true && set_default_params() == success && is_fwd()
                    && one_of(src_md()->data_type, s8, u8)
                    && weights_md()->data_type == s8
                    && !with_bias() // enable in post proccess kernel
                    && IMPLICATION(with_bias(),
                            one_of(weights_md(1)->data_type, s8, u8, f32, s32))
                    && one_of(dst_md()->data_type, /* u8, s8, */ f32, s32)
                    && dense_consitency_check(src_md(), weights_md(), dst_md())
                    && dense_gemm_consitency_check(
                            src_md(), weights_md(), dst_md());
            if (!ok) return unimplemented;

            const auto &wmd = *this->weights_md();
            bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = status::success
                    == create_gemm_x8s8s32x_pd(&gemm_pd_, this->engine(),
                            wei_tr ? transpose::trans : transpose::notrans,
                            transpose::notrans, oc, mb, ic_total,
                            wei_tr ? ic_total : oc, ic_total, oc,
                            weights_md()->data_type, src_md()->data_type,
                            dst_md()->data_type, 1.0, 0.0, *attr());
            if (!gemm_ok) return status::unimplemented;

            // TODO: book a buffer for scratchpad memory
            // bool use_scratchpad = !one_of(dst_md()->data_type, s32, f32);
            return success;
        }

        bool with_post_proccess() const { return false; }

        primitive_desc_t *gemm_pd_ = nullptr;
    };

    status_t init() override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(&gemm_);
        if (gemm_status != status::success) return gemm_status;

        // TODO: create post processing kernel

        return status::success;
    }

    gemm_x8s8s32x_inner_product_fwd_t(const pd_t *apd)
        : primitive_impl_t(apd) {}
    ~gemm_x8s8s32x_inner_product_fwd_t() { delete gemm_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    primitive_t *gemm_ = nullptr;
    compute::kernel_t post_proccess_;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif
