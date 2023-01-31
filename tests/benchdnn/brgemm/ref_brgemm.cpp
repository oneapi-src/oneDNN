/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "utils/parallel.hpp"

#include "brgemm/brgemm.hpp"

namespace brgemm {

#if defined(DNNL_X64) && DNNL_X64 == 1 && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE

int64_t src_off_f(const prb_t *prb, int64_t bs, int64_t m, int64_t k) {
    return (m * prb->batch_size + bs) * prb->k + k;
}

int64_t wei_off_f(const prb_t *prb, int64_t bs, int64_t k, int64_t n) {
    return (bs * prb->k + k) * prb->n + n;
}

int64_t dst_off_f(const prb_t *prb, int64_t m, int64_t n) {
    return m * prb->n + n;
}

void compute_ref_brgemm(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &bia_m = args.find(DNNL_ARG_BIAS);
    const dnn_mem_t &acc_m = args.find(DNNL_ARG_SRC_1);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
    const dnn_mem_t &ws_m = args.find(DNNL_ARG_WORKSPACE);
    const int64_t BS = prb->batch_size;
    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;

    // Using workspace memory as a method to get brgemm attributes.
    using brgemm_attr_t = dnnl::impl::cpu::x64::brgemm_attr_t;
    brgemm_attr_t *brgemm_attr = (brgemm_attr_t *)ws_m;

    const int wei_zero_point = prb->attr.zero_points[DNNL_ARG_WEIGHTS];

    dnn_mem_t dst_tmp(dst_m, dnnl_f32, tag::abx, dst_m.engine());

    const auto alpha = prb->alpha;
    const auto beta = prb->beta;

    if (!brgemm_attr->generate_skip_accumulation) {
        benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
            auto src = (const float *)src_m;
            auto wei = (const float *)wei_m;

            float res = 0;
            for_(int64_t bs = 0; bs < BS; bs++)
            for (int64_t k = 0; k < K; ++k) {
                auto s = src[src_off_f(prb, bs, m, k)];
                maybe_zero_point(prb->attr, s, prb->src_zp, k, DNNL_ARG_SRC);
                auto w = wei[wei_off_f(prb, bs, k, n)] - wei_zero_point;
                res += alpha * s * w;
            }
            float &dst = ((float *)dst_tmp)[dst_off_f(prb, m, n)];
            float acc = ((float *)acc_m)[dst_off_f(prb, m, n)];
            dst = res + (beta != 0 ? beta * acc : 0);
        });
    } else {
        benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
            float &dst = ((float *)dst_tmp)[dst_off_f(prb, m, n)];
            float acc = ((float *)acc_m)[dst_off_f(prb, m, n)];
            dst = beta * acc;
        });
    }

    auto wei_scale = prb->attr.scales.get(DNNL_ARG_WEIGHTS);
    auto attr_scale_arg = !wei_scale.is_def() ? DNNL_ARG_WEIGHTS : DNNL_ARG_SRC;

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    static constexpr int bias_broadcast_mask = 2;
    benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
        size_t dst_off = dst_off_f(prb, m, n);
        float &dst = ((float *)dst_m)[dst_off];

        float tmp = ((float *)dst_tmp)[dst_off];
        maybe_scale(prb->attr, tmp, prb->scales, n, attr_scale_arg);
        if (prb->bia_dt != dnnl_data_type_undef) {
            int64_t bia_off = dst_m.get_scale_idx(dst_off, bias_broadcast_mask);
            float *bia_ptr = (float *)bia_m;
            tmp += bia_ptr[bia_off];
        }

        const auto v_po_vals
                = prepare_po_vals(dst_m, args, v_po_masks, dst_off);

        maybe_post_ops(prb->attr, tmp, dst, v_po_vals);

        maybe_scale(prb->attr, tmp, prb->dst_scales, n, DNNL_ARG_DST, true);
        maybe_zero_point(prb->attr, tmp, prb->dst_zp, n, DNNL_ARG_DST, true);
        dst = tmp;
    });
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    compute_ref_brgemm(prb, args);
}

#else

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {}

#endif

} // namespace brgemm
