/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#include "bnorm/bnorm.hpp"

namespace bnorm {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &src_add = args.find(DNNL_ARG_SRC_1);
    const dnn_mem_t &mean = args.find(DNNL_ARG_MEAN);
    const dnn_mem_t &var = args.find(DNNL_ARG_VARIANCE);
    const dnn_mem_t &sc = args.find(DNNL_ARG_SCALE);
    const dnn_mem_t &sh = args.find(DNNL_ARG_SHIFT);
    const dnn_mem_t &ws = args.find(DNNL_ARG_WORKSPACE);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);
    const dnn_mem_t &src_hat = args.find(DNNL_ARG_DST_1);

    uint8_t *ws_ptr = (uint8_t *)ws;
    float *dst_ptr = (float *)dst;

    const int64_t MB = prb->mb;
    const int64_t C = prb->ic;
    const int64_t D = prb->id;
    const int64_t H = prb->ih;
    const int64_t W = prb->iw;
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();
    const bool fuse_relu = prb->fuse_relu();
    const bool fuse_add_relu = prb->fuse_add_relu();
    const bool need_ws = prb->need_ws();
    const auto &attr = prb->attr;

    benchdnn_parallel_nd(C, [&](int64_t c) {
        float smean = mean.get_elem(c);
        float svar = var.get_elem(c);
        float sqrt_var = sqrtf(svar + prb->eps);
        float rcp_denom = 1.f / sqrt_var;
        float gamma = use_sc ? sc.get_elem(c) : 1.f;
        float beta = use_sh ? sh.get_elem(c) : 0.f;

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(prb, mb, c, d, h, w);
            float x_hat = (src.get_elem(off) - smean) * rcp_denom;
            float res = gamma * x_hat + beta;
            if (fuse_add_relu) res += src_add.get_elem(off);
            if (fuse_relu && res < 0) res = 0;
            if (need_ws) ws_ptr[off] = !!res;
            maybe_post_ops(attr, res);
            dst_ptr[off] = res;
            if (prb->dir & FLAG_BWD) src_hat.set_elem(off, x_hat);
        }
    });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_hat = args.find(DNNL_ARG_DST_1);
    const dnn_mem_t &var = args.find(DNNL_ARG_VARIANCE);
    const dnn_mem_t &d_dst = args.find(DNNL_ARG_DIFF_DST);
    const dnn_mem_t &sc = args.find(DNNL_ARG_SCALE);
    const dnn_mem_t &ws = args.find(DNNL_ARG_WORKSPACE);
    const dnn_mem_t &d_src = args.find(DNNL_ARG_DIFF_SRC);
    const dnn_mem_t &d_src_add = args.find(DNNL_ARG_DIFF_SRC_1);
    const dnn_mem_t &d_sc = args.find(DNNL_ARG_DIFF_SCALE);
    const dnn_mem_t &d_sh = args.find(DNNL_ARG_DIFF_SHIFT);

    const int64_t MB = prb->mb;
    const int64_t C = prb->ic;
    const int64_t D = prb->id;
    const int64_t H = prb->ih;
    const int64_t W = prb->iw;
    const bool glob_stats = prb->flags & GLOB_STATS;
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();
    const bool fuse_relu = prb->fuse_relu();
    const bool fuse_add_relu = prb->fuse_add_relu();

    const float MB_SP = MB * D * H * W;

    benchdnn_parallel_nd(C, [&](int64_t c) {
        float rcp_denom = 1.f / sqrtf(var.get_elem(c) + prb->eps);
        float gamma = use_sc ? sc.get_elem(c) : 1.f;

        float d_gamma = 0;
        float d_beta = 0;

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(prb, mb, c, d, h, w);
            float dd = d_dst.get_elem(off);
            if (fuse_relu && ws.get_elem(off) == 0) dd = 0;
            d_gamma += dd * src_hat.get_elem(off);
            d_beta += dd;
        }

        if (use_sc && (prb->dir & FLAG_WEI)) d_sc.set_elem(c, d_gamma);
        if (use_sh && (prb->dir & FLAG_WEI)) d_sh.set_elem(c, d_beta);

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(prb, mb, c, d, h, w);
            float dd = d_dst.get_elem(off);
            if (fuse_relu && ws.get_elem(off) == 0) dd = 0;
            if (fuse_add_relu) d_src_add.set_elem(off, dd);
            float ds = dd;

            if (!glob_stats)
                ds -= (d_beta + src_hat.get_elem(off) * d_gamma) / MB_SP;

            d_src.set_elem(off, rcp_denom * ds * gamma);
        }
    });
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    compute_ref_fwd(prb, args);
    if (prb->dir & FLAG_BWD) compute_ref_bwd(prb, args);
}

} // namespace bnorm
