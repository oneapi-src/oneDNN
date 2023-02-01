/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "matmul/matmul.hpp"

namespace matmul {

void compute_ref_matmul(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &bia_m = args.find(DNNL_ARG_BIAS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
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
    const int wei_scale_mask
            = prb->attr.scales.get_mask(DNNL_ARG_WEIGHTS, dnnl_matmul);

    const bool has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_zp
            = !prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_wei_zp, wei_zps.nelems() == 1));
    const int wei_zp = has_wei_zp ? wei_zps.get_elem(0) : 0;
    const int src_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_SRC).policy);
    const int dst_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_DST).policy);

    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;
    const int64_t MB = prb->mb;
    const int batch_ndims = dst_m.ndims() - 2;

    // Fast return if any dim is zero. Common logic doesn't apply because of
    // broadcast semantics.
    for (int d = 0; d < dst_m.ndims(); d++) {
        if (prb->src_dims()[d] == 0 || prb->weights_dims()[d] == 0) return;
    }

    dnn_mem_t dst_tmp(dst_m, dnnl_f32, tag::abx, dst_m.engine());

    const auto src_broadcast_mask = prb->src_broadcast_mask();
    const auto wei_broadcast_mask = prb->weights_broadcast_mask();

    benchdnn_parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        auto src = (const float *)src_m;
        auto wei = (const float *)wei_m;

        float dst = 0;
        const int64_t src_mb
                = dst_m.get_scale_idx(mb, src_broadcast_mask, batch_ndims);
        const int64_t wei_mb
                = dst_m.get_scale_idx(mb, wei_broadcast_mask, batch_ndims);
        for (int64_t k = 0; k < K; ++k) {
            int src_zp = has_src_zp ? src_zps.get_elem(src_zp_mask > 0 ? k : 0)
                                    : 0;
            auto s = src[src_off_f(prb, src_mb, m, k)] - src_zp;
            auto w = wei[wei_off_f(prb, wei_mb, k, n)] - wei_zp;
            dst += s * w;
        }
        ((float *)dst_tmp)[dst_off_f(prb, mb, m, n)] = dst;
    });

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    const auto bias_broadcast_mask = prb->bias_broadcast_mask();
    benchdnn_parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        size_t dst_off = dst_off_f(prb, mb, m, n);
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

#ifdef DNNL_EXPERIMENTAL_SPARSE
void compute_ref_matmul_csr(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;

    // Batch is not supported.
    const int64_t mb = 0;

    float *dst = dst_m.get_mapped_pointer<float>();

    benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
        dst[dst_off_f(prb, mb, m, n)] = 0.0f;
    });

    if (prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS) == dnnl_csr) {
        const float *src = src_m.get_mapped_pointer<float>();
        const float *wei_values = wei_m.get_mapped_pointer<float>(0);
        const int32_t *wei_indices = wei_m.get_mapped_pointer<int32_t>(1);
        const int32_t *wei_pointers = wei_m.get_mapped_pointer<int32_t>(2);

        benchdnn_parallel_nd(M, [&](int64_t m) {
            for (int64_t k = 0; k < K; k++) {
                const int64_t row_start = wei_pointers[k];
                const int64_t row_end = wei_pointers[k + 1];
                for (int64_t n = row_start; n < row_end; n++) {
                    const int64_t src_idx = src_off_f(prb, mb, m, k);
                    const int64_t dst_idx
                            = dst_off_f(prb, mb, m, wei_indices[n]);
                    dst[dst_idx] = dst[dst_idx] + src[src_idx] * wei_values[n];
                }
            }
        });
    } else if (prb->sparse_options.get_encoding(DNNL_ARG_SRC) == dnnl_csr) {
        const float *weights = wei_m.get_mapped_pointer<float>();
        const float *src_values = src_m.get_mapped_pointer<float>(0);
        const int32_t *src_indices = src_m.get_mapped_pointer<int32_t>(1);
        const int32_t *src_pointers = src_m.get_mapped_pointer<int32_t>(2);

        benchdnn_parallel_nd(M, [&](int64_t m) {
            const int64_t row_start = src_pointers[m];
            const int64_t row_end = src_pointers[m + 1];
            for (int64_t k = row_start; k < row_end; k++) {
                for (int64_t n = 0; n < N; n++) {
                    const int64_t dst_idx = dst_off_f(prb, mb, m, n);
                    const int64_t wei_idx
                            = wei_off_f(prb, mb, src_indices[k], n);
                    dst[dst_idx]
                            = dst[dst_idx] + src_values[k] * weights[wei_idx];
                }
            }
        });
    }
}
#endif

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

#ifdef DNNL_EXPERIMENTAL_SPARSE
    const auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

    if (src_encoding == dnnl_csr || wei_encoding == dnnl_csr) {
        compute_ref_matmul_csr(prb, args);
    } else {
        compute_ref_matmul(prb, args);
    }
#else
    compute_ref_matmul(prb, args);
#endif
}

} // namespace matmul
