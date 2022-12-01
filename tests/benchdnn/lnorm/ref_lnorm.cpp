/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "lnorm/lnorm.hpp"

namespace lnorm {

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

    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_src_scale, src_scale.nelems() == 1));
    assert(IMPLICATION(has_dst_scale, dst_scale.nelems() == 1));

    const float src_scale_val = has_src_scale ? src_scale.get_elem(0) : 1.f;
    const float dst_scale_val = has_dst_scale ? dst_scale.get_elem(0) : 1.f;
    const float output_scale = src_scale_val / dst_scale_val;

    benchdnn_parallel_nd(prb->n, [&](int64_t n) {
        float smean = mean.get_elem(n);
        float svar = var.get_elem(n);
        float sqrt_var = sqrtf(svar + prb->eps);

        for (int64_t c = 0; c < prb->c; ++c) {
            float gamma = (use_sc ? sc.get_elem(c) : 1.0f) / sqrt_var;
            float beta = use_sh ? sh.get_elem(c) : 0;
            auto off = n * prb->c + c;
            float res = gamma * (src.get_elem(off) - smean) + beta;
            dst_ptr[off] = res * output_scale;
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

    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    if ((use_sc || use_sh) && (prb->dir & FLAG_WEI)) {
        benchdnn_parallel_nd(prb->c, [&](int64_t c) {
            float d_gamma = 0;
            float d_beta = 0;

            for (int64_t n = 0; n < prb->n; ++n) {
                float smean = mean.get_elem(n);
                float svar = var.get_elem(n);
                float rcp_denom = 1.f / sqrtf(svar + prb->eps);
                auto off = n * prb->c + c;
                float dd = d_dst.get_elem(off);
                d_gamma += dd * (src.get_elem(off) - smean) * rcp_denom;
                d_beta += dd;
            }

            if (use_sc && (prb->dir & FLAG_WEI)) d_sc.set_elem(c, d_gamma);
            if (use_sh && (prb->dir & FLAG_WEI)) d_sh.set_elem(c, d_beta);
        });
    }

    benchdnn_parallel_nd(prb->n, [&](int64_t n) {
        float smean = mean.get_elem(n);
        float svar = var.get_elem(n);
        float rcp_denom = 1.f / sqrtf(svar + prb->eps);
        float dd_gamma = 0, dd_gamma_x = 0;
        if (!(prb->flags & GLOB_STATS)) {
            for (int64_t c = 0; c < prb->c; ++c) {
                auto off = n * prb->c + c;
                float ds = d_dst.get_elem(off);
                const float x = src.get_elem(off) - smean;
                float gamma = use_sc ? sc.get_elem(c) : 1;
                dd_gamma += gamma * ds;
                dd_gamma_x += gamma * ds * x;
            }
            dd_gamma_x *= rcp_denom;
        }
        for (int64_t c = 0; c < prb->c; ++c) {
            float gamma = use_sc ? sc.get_elem(c) : 1;
            auto off = n * prb->c + c;
            float ds = d_dst.get_elem(off) * gamma;
            if (!(prb->flags & GLOB_STATS)) {
                const float x = src.get_elem(off) - smean;
                ds -= (dd_gamma + x * dd_gamma_x * rcp_denom) / prb->c;
            }

            d_src.set_elem(off, rcp_denom * ds);
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

} // namespace lnorm
