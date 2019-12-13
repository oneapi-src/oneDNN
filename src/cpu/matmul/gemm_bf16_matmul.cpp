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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "gemm_bf16_matmul.hpp"

#include "gemm/gemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;

template <impl::data_type_t dst_type>
status_t gemm_bf16_matmul_t<dst_type>::pd_t::init() {
    auto check_bias = [&]() -> bool {
        return !with_bias()
                || (utils::one_of(weights_md(1)->data_type, f32, bf16)
                        && is_bias_1xN());
    };

    auto can_use_gemm = [&]() -> bool { return mayiuse(avx512_core); };

    bool ok = src_md()->data_type == src_type
            && weights_md()->data_type == weights_type
            && desc()->accum_data_type == acc_type
            && dst_md()->data_type == dst_type && can_use_gemm() && batch() == 1
            && check_bias()
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::post_ops);
    if (!ok) return status::unimplemented;

    // set state
    params_.dst_is_acc_ = dst_type == data_type::f32;

    status_t status = check_and_configure_attributes();
    if (status != status::success) return status;

    if (!set_default_formats()) return status::unimplemented;

    gemm_based::book_acc_scratchpad(*this, params_, sizeof(acc_data_t));

    return status::success;
}

template <impl::data_type_t dst_type>
status_t gemm_bf16_matmul_t<dst_type>::pd_t::check_and_configure_attributes() {
    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return oscale.mask_ == 0
                || (oscale.mask_ == (1 << 1) && batched() == false);
    };

    auto check_attr_post_ops = [&]() -> bool {
        using namespace primitive_kind;
        const auto &p = attr()->post_ops_;
        auto check_sum = [&](int idx) -> bool {
            return p.contain(sum, idx) && params_.gemm_applies_output_scales_;
        };
        switch (p.len_) {
            case 0: return true;
            case 1: return check_sum(0) || p.contain(eltwise, 0);
            case 2: return check_sum(0) && p.contain(eltwise, 1);
            default: return false;
        }
    };

    // check basic attributes
    if (!check_attr_oscale()) return status::unimplemented;

    // set state
    params_.pp_attr_ = *attr();
    params_.gemm_applies_output_scales_
            = attr()->output_scales_.mask_ == 0 && !with_bias();
    if (params_.gemm_applies_output_scales_)
        params_.pp_attr_.output_scales_.set(1.f);

    // check post-ops
    if (check_attr_post_ops()) {
        auto &po = params_.pp_attr_.post_ops_;
        const int sum_idx = 0;
        bool with_sum = po.len_ > 0 && po.contain(primitive_kind::sum, sum_idx);
        if (with_sum && params_.dst_is_acc_) {
            // set state
            params_.gemm_beta_ = po.entry_[sum_idx].sum.scale;
            // drop sum from pp_attributes, as it will be applied by gemm
            for (int i = 0; i < po.len_ - 1; ++i)
                po.entry_[i] = po.entry_[i + 1];
            po.len_ -= 1;
        }
    } else {
        return status::unimplemented;
    }

    // set state
    params_.has_pp_kernel_ = !params_.dst_is_acc_ || with_bias()
            || !params_.pp_attr_.has_default_values();

    return status::success;
}

template <impl::data_type_t dst_type>
status_t gemm_bf16_matmul_t<dst_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const weights_data_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    DEFINE_SCALES_BUFFER(scales);

    const gemm_based::params_t &params = pd()->params();

    acc_data_t *acc = params.dst_is_acc_
            ? (acc_data_t *)dst
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    const auto &dst_bd = dst_d.blocking_desc();

    const bool batched = pd()->batched();

    const dim_t batch = batched ? dst_d.dims()[0] : 1;
    const dim_t M = dst_d.dims()[batched + 0];
    const dim_t N = dst_d.dims()[batched + 1];
    const dim_t K = src_d.dims()[batched + 1];

    // case: dynamic sizes
    bool need_free_acc = false;
    if (acc == nullptr) {
        acc = (acc_data_t *)malloc(sizeof(acc_data_t) * batch * M * N, 64);
        need_free_acc = true;
    }

    // gemm section
    {
        const auto &src_strides = &src_d.blocking_desc().strides[batched];
        const auto &weights_strides
                = &weights_d.blocking_desc().strides[batched];

        const char *transA
                = src_strides[1] == 1 && src_d.dims()[batched + 0] > 1 ? "N"
                                                                       : "T";
        const char *transB
                = weights_strides[1] == 1 && weights_d.dims()[batched + 0] > 1
                ? "N"
                : "T";

        const int M_s32 = (int)M;
        const int N_s32 = (int)N;
        const int K_s32 = (int)K;

        const int lda = (int)src_strides[*transA == 'N' ? 0 : 1];
        const int ldb = (int)weights_strides[*transB == 'N' ? 0 : 1];
        const int ldc = (int)dst_bd.strides[batched + 0];

        const float alpha = params.get_gemm_alpha(scales);
        const float beta = params.gemm_beta_;

        status_t status = gemm_bf16bf16f32(transB, transA, &N_s32, &M_s32,
                &K_s32, &alpha, weights, &ldb, src, &lda, &beta, acc, &ldc);
        if (status != status::success) {
            if (need_free_acc) free(acc);
            return status;
        }
    }

    if (params.has_pp_kernel_) {
        const bool force_sequential = pp_kernel_->sequential_kernel();
        const float *pp_scales = params.get_post_processing_scales(scales);
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start {}, end {};
            balance211((size_t)(M * N), nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, pp_scales, start, end, (size_t)N);
        });
    }

    if (need_free_acc) free(acc);

    return status::success;
}

using namespace data_type;
template struct gemm_bf16_matmul_t<data_type::f32>;
template struct gemm_bf16_matmul_t<data_type::bf16>;

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
