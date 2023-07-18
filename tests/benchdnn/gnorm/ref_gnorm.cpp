/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gnorm/gnorm.hpp"

namespace gnorm {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &mean = args.find(DNNL_ARG_MEAN);
    const dnn_mem_t &var = args.find(DNNL_ARG_VARIANCE);
    const dnn_mem_t &sc = args.find(DNNL_ARG_SCALE);
    const dnn_mem_t &sh = args.find(DNNL_ARG_SHIFT);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);
    const dnn_mem_t &src_scale = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const dnn_mem_t &dst_scale = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    float *dst_ptr = (float *)dst;

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_src_scale, src_scale.nelems() == 1));
    assert(IMPLICATION(has_dst_scale, dst_scale.nelems() == 1));

    const float src_scale_val = has_src_scale ? src_scale.get_elem(0) : 1.f;
    const float dst_scale_val = has_dst_scale ? dst_scale.get_elem(0) : 1.f;
    const float r_dst_scale_val = 1.0f / dst_scale_val;

    const int64_t MB = prb->mb;
    const int64_t G = prb->g;
    const int64_t D = prb->id;
    const int64_t H = prb->ih;
    const int64_t W = prb->iw;
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    auto v_po_masks = prb->attr.post_ops.get_po_masks();

    benchdnn_parallel_nd(MB, G, [&](int64_t n, int64_t g) {
        float smean = mean.get_elem(n * G + g);
        float svar = var.get_elem(n * G + g);
        float sqrt_var = sqrtf(svar + prb->eps);
        float rcp_denom = 1.f / sqrt_var;

        for_(int64_t c = prb->get_c_start(g); c < prb->get_c_start(g + 1); ++c)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(prb, n, c, d, h, w);
            float x_hat = (src.get_elem(off) - smean) * rcp_denom;
            float gamma = use_sc ? sc.get_elem(c) : 1.f;
            float beta = use_sh ? sh.get_elem(c) : 0.f;
            float res = gamma * x_hat + beta;
            const auto v_po_vals = prepare_po_vals(dst, args, v_po_masks, off);
            res *= src_scale_val;
            maybe_post_ops(prb->attr, res, 0.f, v_po_vals);
            dst_ptr[off] = res * r_dst_scale_val;
        }
    });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &mean = args.find(DNNL_ARG_MEAN);
    const dnn_mem_t &var = args.find(DNNL_ARG_VARIANCE);
    const dnn_mem_t &d_dst = args.find(DNNL_ARG_DIFF_DST);
    const dnn_mem_t &sc = args.find(DNNL_ARG_SCALE);
    const dnn_mem_t &d_src = args.find(DNNL_ARG_DIFF_SRC);
    const dnn_mem_t &d_sc = args.find(DNNL_ARG_DIFF_SCALE);
    const dnn_mem_t &d_sh = args.find(DNNL_ARG_DIFF_SHIFT);

    const int64_t MB = prb->mb;
    const int64_t G = prb->g;
    const int64_t C = prb->ic;
    const int64_t D = prb->id;
    const int64_t H = prb->ih;
    const int64_t W = prb->iw;
    const bool glob_stats = prb->flags & GLOB_STATS;
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    const int C_PER_G = C / G;
    const float CSP = C_PER_G * D * H * W;

    benchdnn_parallel_nd(C, [&](int64_t c) {
        int64_t g = c / C_PER_G;

        float gamma = use_sc ? sc.get_elem(c) : 1.f;

        float d_gamma = 0;
        float d_beta = 0;

        for (int64_t mb = 0; mb < MB; ++mb) {
            int64_t stat_off = mb * G + g;
            float smean = mean.get_elem(stat_off);
            float svar = var.get_elem(stat_off);
            float rcp_denom = 1.f / sqrtf(svar + prb->eps);

            for_(int64_t d = 0; d < D; ++d)
            for_(int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w) {
                auto off = data_off(prb, mb, c, d, h, w);
                float dd = d_dst.get_elem(off);
                d_gamma += dd * (src.get_elem(off) - smean) * rcp_denom;
                d_beta += dd;
            }
        }

        if (use_sc && (prb->dir & FLAG_WEI)) d_sc.set_elem(c, d_gamma);
        if (use_sh && (prb->dir & FLAG_WEI)) d_sh.set_elem(c, d_beta);

        for (int64_t mb = 0; mb < MB; ++mb) {
            int64_t stat_off = mb * G + g;
            float smean = mean.get_elem(stat_off);
            float svar = var.get_elem(stat_off);
            float rcp_denom = 1.f / sqrtf(svar + prb->eps);

            for_(int64_t d = 0; d < D; ++d)
            for_(int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w) {
                auto off = data_off(prb, mb, c, d, h, w);
                float dd = d_dst.get_elem(off);
                float ds = dd;

                if (!glob_stats) {
                    float x_hat = (src.get_elem(off) - smean) * d_gamma;
                    ds -= (d_beta + x_hat * rcp_denom) / CSP;
                }

                d_src.set_elem(off, rcp_denom * ds * gamma);
            }
        }
    });
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prb->dir & FLAG_FWD)
        compute_ref_fwd(prb, args);
    else
        compute_ref_bwd(prb, args);
}

} // namespace gnorm
