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

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "type_helpers.hpp"

#include "bfloat16.hpp"
#include "gemm_bf16_inner_product.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::primitive_kind;
using namespace memory_tracking::names;
using namespace dnnl::impl::cpu::bf16_support;

template <data_type_t dst_data_type>
void gemm_bf16_inner_product_fwd_t<dst_data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const int M = pd()->OC();
    const int N = pd()->MB();
    const int K = pd()->IC_total_padded();

    const auto &wmd = *pd()->weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

    acc_data_t *acc = pd()->dst_is_acc_
            ? (acc_data_t *)dst
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0;
    gemm_bf16bf16f32(wei_tr ? "T" : "N", "N", &M, &N, &K, &alpha, weights,
            wei_tr ? &K : &M, src, &K, &beta_, acc, &M);

    const float *scales = pd()->attr()->output_scales_.scales_;
    if (postops_in_ip_) {
        const bool force_sequential = pp_kernel_->sequential_kernel();
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start = 0, end = 0;
            size_t work_size = M * N;
            balance211(work_size, nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, scales, start, end);
        });
    }
}

template <data_type_t diff_src_data_type>
void gemm_bf16_inner_product_bwd_data_t<diff_src_data_type>::
        execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const int M = pd()->IC_total_padded();
    const int N = pd()->MB();
    const int K = pd()->OC();

    const auto &wmd = *pd()->weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

    acc_data_t *acc = pd()->diff_src_is_acc_
            ? (acc_data_t *)diff_src
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    gemm_bf16bf16f32(wei_tr ? "T" : "N", "N", &M, &N, &K, &alpha, weights,
            wei_tr ? &K : &M, diff_dst, &K, &beta, acc, &M);

    if (!pd()->diff_src_is_acc_) {
        parallel(0, [&](int ithr, int nthr) {
            size_t start = 0, end = 0;
            size_t work_size = M * N;
            balance211(work_size, nthr, ithr, start, end);
            if (end > start)
                cvt_float_to_bfloat16((bfloat16_t *)&diff_src[start],
                        (const float *)&acc[start], end - start);
        });
    }
}

template <data_type_t diff_wei_data_type>
void gemm_bf16_inner_product_bwd_weights_t<diff_wei_data_type>::
        execute_backward_weights(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    diff_dst += diff_dst_d.offset0();

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int IC = pd()->IC_total_padded();

    const auto &wmd = *pd()->diff_weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

    const int M = wei_tr ? OC : IC;
    const int N = wei_tr ? IC : OC;
    const int K = MB;

    acc_data_t *acc = pd()->diff_wei_is_acc_
            ? (acc_data_t *)diff_weights
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    gemm_bf16bf16f32("N", "T", &M, &N, &K, &alpha, wei_tr ? diff_dst : src, &M,
            wei_tr ? src : diff_dst, &N, &beta, acc, &M);

    if (!pd()->diff_wei_is_acc_) {
        parallel(0, [&](int ithr, int nthr) {
            size_t start = 0, end = 0;
            size_t work_size = M * N;
            balance211(work_size, nthr, ithr, start, end);
            if (end > start)
                cvt_float_to_bfloat16((bfloat16_t *)&diff_weights[start],
                        (const float *)&acc[start], end - start);
        });
    }

    if (pd()->with_bias()) {
        const size_t bias_dt_size
                = types::data_type_size(pd()->diff_weights_md(1)->data_type);
        diff_bias += bias_dt_size * diff_bias_d.offset0();
        constexpr int blksize = 16;
        const int OC_blocks = utils::div_up(OC, blksize);
        float *diff_bias_acc = pd()->diff_bias_is_acc_
                ? (float *)diff_bias
                : (float *)ctx.get_scratchpad_grantor()
                          .template get<acc_data_t>(
                                  key_iprod_bias_bf16_convert_wsp);
        parallel(0, [&](const int ithr, const int nthr) {
            int oc_s {0}, oc_e {0};
            balance211(OC_blocks, nthr, ithr, oc_s, oc_e);
            oc_s = std::min(oc_s * blksize, OC);
            oc_e = std::min(oc_e * blksize, OC);
            auto len = oc_e - oc_s;

            if (len > 0) {
                cvt_bfloat16_to_float(&diff_bias_acc[oc_s],
                        &((bfloat16_t *)diff_dst)[oc_s], len);

                for (int mb = 1; mb < MB; ++mb)
                    cvt_bfloat16_and_add_to_float(&diff_bias_acc[oc_s],
                            &((bfloat16_t *)diff_dst)[mb * OC + oc_s],
                            &diff_bias_acc[oc_s], len);

                if (!pd()->diff_bias_is_acc_)
                    cvt_float_to_bfloat16(&((bfloat16_t *)diff_bias)[oc_s],
                            &diff_bias_acc[oc_s], len);
            }
        });
    }
}

template struct gemm_bf16_inner_product_fwd_t<data_type::f32>;
template struct gemm_bf16_inner_product_fwd_t<data_type::bf16>;
template struct gemm_bf16_inner_product_bwd_data_t<data_type::f32>;
template struct gemm_bf16_inner_product_bwd_data_t<data_type::bf16>;
template struct gemm_bf16_inner_product_bwd_weights_t<data_type::f32>;
template struct gemm_bf16_inner_product_bwd_weights_t<data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
