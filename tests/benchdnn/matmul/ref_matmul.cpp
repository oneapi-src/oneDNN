/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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
    const dnn_mem_t &dropout = args.find(DNNL_ARG_ATTR_DROPOUT_MASK);

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_dst_scale, dst_scales.nelems() == 1));
    float dst_scale = has_dst_scale ? 1.f / dst_scales.get_elem(0) : 1.f;
    const int src_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_SRC, dnnl_matmul, src_m.ndims());
    const int wei_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_matmul, wei_m.ndims());

    const bool has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_zp
            = !prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).is_def();
    const bool has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();

    const int src_zp_mask = prb->attr.zero_points.get_mask(
            DNNL_ARG_SRC, dnnl_matmul, src_m.ndims());
    const int wei_zp_mask = prb->attr.zero_points.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_matmul, wei_m.ndims());
    const int dst_zp_mask = attr_t::get_default_mask(
            prb->attr.zero_points.get(DNNL_ARG_DST).policy);

    const auto &src_scale_groups = prb->attr.scales.get(DNNL_ARG_SRC).groups;
    const auto &wei_scale_groups
            = prb->attr.scales.get(DNNL_ARG_WEIGHTS).groups;
    const auto &src_zp_groups = prb->attr.zero_points.get(DNNL_ARG_SRC).groups;
    const auto &wei_zp_groups
            = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).groups;

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

    const auto src_broadcast_mask = prb->src_broadcast_mask();
    const auto wei_broadcast_mask = prb->weights_broadcast_mask();
    const auto bias_broadcast_mask = prb->bias_broadcast_mask();
    auto v_po_masks = prb->attr.post_ops.get_po_masks();

    benchdnn_parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        float dst = 0;
        const int64_t src_mb
                = dst_m.get_idx(mb, src_broadcast_mask, batch_ndims);
        const int64_t wei_mb
                = dst_m.get_idx(mb, wei_broadcast_mask, batch_ndims);

        for (int64_t k = 0; k < K; ++k) {
            const auto src_off = src_off_f(prb, src_mb, m, k);
            const auto wei_off = wei_off_f(prb, wei_mb, k, n);

            int src_zp = 0;
            if (has_src_zp) {
                const auto src_zp_idx = src_m.get_idx(
                        src_off, src_zp_mask, src_m.ndims(), src_zp_groups);
                src_zp = src_zps.get_elem(src_zp_idx);
            }
            int wei_zp = 0;
            if (has_wei_zp) {
                const auto wei_zp_idx = wei_m.get_idx(
                        wei_off, wei_zp_mask, wei_m.ndims(), wei_zp_groups);
                wei_zp = wei_zps.get_elem(wei_zp_idx);
            }

            float src_scale = 1.f;
            if (has_src_scale) {
                const auto src_scale_idx = src_m.get_idx(src_off,
                        src_scale_mask, src_m.ndims(), src_scale_groups);
                src_scale = src_scales.get_elem(src_scale_idx);
            }
            float wei_scale = 1.f;
            if (has_wei_scale) {
                const auto wei_scale_idx = wei_m.get_idx(wei_off,
                        wei_scale_mask, wei_m.ndims(), wei_scale_groups);
                wei_scale = wei_scales.get_elem(wei_scale_idx);
            }

            auto s = src_scale * (src_m.get_elem(src_off) - src_zp);
            auto w = wei_scale * (wei_m.get_elem(wei_off) - wei_zp);

            dst += s * w;
        }

        const auto dst_off = dst_off_f(prb, mb, m, n);
        if (prb->bia_dt != dnnl_data_type_undef) {
            const auto bia_idx = dst_m.get_idx(dst_off, bias_broadcast_mask);
            dst += bia_m.get_elem(bia_idx);
        }

        const auto v_po_vals
                = prepare_po_vals(dst_m, args, v_po_masks, dst_off);
        maybe_dropout(prb->attr, dst, dst_off, dropout);
        const auto sum_val = dst_m.get_elem(dst_off);
        maybe_post_ops(prb->attr, dst, sum_val, v_po_vals);

        int dst_zp = 0;
        if (has_dst_zp) {
            const auto dst_zp_idx = dst_m.get_idx(dst_off, dst_zp_mask);
            dst_zp = dst_zps.get_elem(dst_zp_idx);
        }
        float dst_val = dst_scale * dst + dst_zp;
        maybe_round(prb->attr, DNNL_ARG_DST, dst_val, dst_off, prb->dst_dt());
        dst_m.set_elem(dst_off, dst_val);
    });
}

void cvt_coo_indices_to_csr_pointers(const int32_t *indices, int32_t *pointers,
        const int nnz, const int nrows) {
    for (int i = 0; i < nnz; ++i) {
        ++pointers[indices[i] + 1];
    }
    for (int i = 0; i < nrows; ++i) {
        pointers[i + 1] += pointers[i];
    }
}

void compute_ref_sparse_matmul(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);

    const auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

    const bool is_src_sparse
            = src_encoding == dnnl_csr || src_encoding == dnnl_coo;
    const bool is_wei_sparse
            = wei_encoding == dnnl_csr || wei_encoding == dnnl_coo;
    auto encoding = is_src_sparse ? src_encoding : wei_encoding;

    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;

    // TODO: Depending on the matrix dimensions the pointer buffer may take
    // up a significant amount of memory. This wil require a mechanism to
    // register the memory needed for the current scratchpad during
    // COO-to-CSR format conversion.
    std::vector<int32_t> pointer_buffer(1 + (is_src_sparse ? M : K), 0);

    // Batch is not supported.
    const int64_t mb = 0;
    benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
        dst_m.set_elem(dst_off_f(prb, mb, m, n), 0.0f);
    });

    if (is_wei_sparse) {
        int32_t *wei_indices = wei_m.get_mapped_pointer<int32_t>(
                encoding == dnnl_csr ? 1 : 2);
        int32_t *wei_pointers = wei_m.get_mapped_pointer<int32_t>(2);

        if (encoding == dnnl_coo) {
            int32_t *wei_row_indices = wei_m.get_mapped_pointer<int32_t>(1);
            const int64_t nnz = query_md_nnz(wei_m.md_);

            benchdnn_parallel_nd(
                    K + 1, [&](int64_t i) { pointer_buffer[i] = 0; });
            cvt_coo_indices_to_csr_pointers(
                    wei_row_indices, pointer_buffer.data(), nnz, K);
            wei_pointers = pointer_buffer.data();
        }

        benchdnn_parallel_nd(M, [&](int64_t m) {
            for (int64_t k = 0; k < K; k++) {
                const int64_t row_start = wei_pointers[k];
                const int64_t row_end = wei_pointers[k + 1];
                for (int64_t n = row_start; n < row_end; n++) {
                    const int64_t src_idx = src_off_f(prb, mb, m, k);
                    const int64_t dst_idx
                            = dst_off_f(prb, mb, m, wei_indices[n]);
                    const float src_val = src_m.get_elem(src_idx);
                    const float wei_val = wei_m.get_elem(n, 0);
                    float dst_val = dst_m.get_elem(dst_idx);
                    dst_val += src_val * wei_val;
                    dst_m.set_elem(dst_idx, dst_val);
                }
            }
        });
    } else if (is_src_sparse) {
        int32_t *src_indices = src_m.get_mapped_pointer<int32_t>(
                encoding == dnnl_csr ? 1 : 2);
        int32_t *src_pointers = src_m.get_mapped_pointer<int32_t>(2);

        if (encoding == dnnl_coo) {
            int32_t *src_row_indices = src_m.get_mapped_pointer<int32_t>(1);
            const int64_t nnz = query_md_nnz(src_m.md_);
            cvt_coo_indices_to_csr_pointers(
                    src_row_indices, pointer_buffer.data(), nnz, M);
            src_pointers = pointer_buffer.data();
        }

        benchdnn_parallel_nd(M, [&](int64_t m) {
            const int64_t row_start = src_pointers[m];
            const int64_t row_end = src_pointers[m + 1];
            for (int64_t n = 0; n < N; n++) {
                const int64_t dst_idx = dst_off_f(prb, mb, m, n);
                float dst_val = dst_m.get_elem(dst_idx);

                for (int64_t k = row_start; k < row_end; k++) {
                    const int64_t wei_idx
                            = wei_off_f(prb, mb, src_indices[k], n);
                    const float src_val = src_m.get_elem(k, 0);
                    const float wei_val = wei_m.get_elem(wei_idx);
                    dst_val += src_val * wei_val;
                }
                dst_m.set_elem(dst_idx, dst_val);
            }
        });
    }
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    const auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

    if (src_encoding == dnnl_csr || wei_encoding == dnnl_csr
            || src_encoding == dnnl_coo || wei_encoding == dnnl_coo) {
        compute_ref_sparse_matmul(prb, args);
    } else {
        compute_ref_matmul(prb, args);
    }
}

} // namespace matmul
