/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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
    // Accumulator values are passed for `generate_skip_accumulation` check.
    const dnn_mem_t &acc_m = args.find(DNNL_ARG_DST_1);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
    const dnn_mem_t &ws_m = args.find(DNNL_ARG_WORKSPACE);

    const dnn_mem_t &src_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const dnn_mem_t &wei_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    const dnn_mem_t &src_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const dnn_mem_t &wei_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_src_scale, src_scales.nelems() == 1));
    assert(IMPLICATION(has_dst_scale, dst_scales.nelems() == 1));
    float src_scale = has_src_scale ? src_scales.get_elem(0) : 1.f;
    float dst_scale = has_dst_scale ? 1.f / dst_scales.get_elem(0) : 1.f;
    const int wei_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_matmul, wei_m.ndims());

    const bool has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_zp
            = !prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();

    const int src_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_SRC).policy);
    const int wei_zp_mask = prb->attr.zero_points.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_matmul, wei_m.ndims());
    const int dst_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_DST).policy);

    const int64_t BS = prb->batch_size;
    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;

    // Using workspace memory as a method to get brgemm attributes.
    bool generate_skip_accumulation = ws_m && *((bool *)ws_m);

    const bool wei_zp_per_n = wei_zp_mask & (1 << (wei_m.ndims() - 1));
    const bool wei_zp_per_k = wei_zp_mask & (1 << (wei_m.ndims() - 2));
    const int64_t wei_zp_stride_n = wei_zp_per_n ? 1 : 0;
    const int64_t wei_zp_stride_k = wei_zp_per_k ? wei_zp_per_n ? N : 1 : 0;
    const auto wei_zp_groups
            = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).groups;
    const int64_t wei_zp_group_k
            = !wei_zp_groups.empty() ? wei_zp_groups[0] : 1;

    dnn_mem_t dst_tmp(dst_m, dnnl_f32, tag::abx, dst_m.engine());

    const auto alpha = prb->alpha;
    const auto beta = prb->beta;

    if (!generate_skip_accumulation) {
        benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
            auto src = (const float *)src_m;
            auto wei = (const float *)wei_m;

            float res = 0;
            for_(int64_t bs = 0; bs < BS; bs++)
            for (int64_t k = 0; k < K; ++k) {
                int src_zp = has_src_zp
                        ? src_zps.get_elem(src_zp_mask > 0 ? k : 0)
                        : 0;
                int wei_zp = has_wei_zp ? wei_zps.get_elem(
                                     wei_zp_stride_k * (k / wei_zp_group_k)
                                     + wei_zp_stride_n * n)
                                        : 0;

                auto s = src[src_off_f(prb, bs, m, k)] - src_zp;
                auto w = wei[wei_off_f(prb, bs, k, n)] - wei_zp;
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

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    static constexpr int bias_broadcast_mask = 2;
    benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
        size_t dst_off = dst_off_f(prb, m, n);
        float &dst = ((float *)dst_m)[dst_off];

        float wei_scale = 1.f;
        if (has_wei_scale)
            wei_scale = wei_scales.get_elem(wei_scale_mask > 0 ? n : 0);
        float tmp = ((float *)dst_tmp)[dst_off] * src_scale * wei_scale;

        if (prb->bia_dt != dnnl_data_type_undef) {
            int64_t bia_off = dst_m.get_scale_idx(dst_off, bias_broadcast_mask);
            float *bia_ptr = (float *)bia_m;
            tmp += bia_ptr[bia_off];
        }

        const auto v_po_vals
                = prepare_po_vals(dst_m, args, v_po_masks, dst_off);

        maybe_post_ops(prb->attr, tmp, dst, v_po_vals);

        int dst_zp = has_dst_zp ? dst_zps.get_elem(dst_zp_mask > 0 ? n : 0) : 0;
        dst = tmp * dst_scale + dst_zp;
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

} // namespace brgemm
