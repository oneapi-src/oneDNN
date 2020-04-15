/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "src/common/dnnl_thread.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src,
        const dnn_mem_t &mean, const dnn_mem_t &var, const dnn_mem_t &ss,
        dnn_mem_t &dst) {
    const int64_t MB = p->mb;
    const int64_t C = p->ic;
    const int64_t D = p->id;
    const int64_t H = p->ih;
    const int64_t W = p->iw;
    const bool use_scale_shift = p->flags & USE_SCALESHIFT;
    const bool fuse_relu = p->flags & FUSE_NORM_RELU;

    const auto dt = p->dt;
    const auto &attr = p->attr;

    dnnl::impl::parallel_nd(C, [&](int64_t c) {
        float smean = mean.get_elem(c);
        float svar = var.get_elem(c);
        float sqrt_var = sqrtf(svar + p->eps);
        float rcp_denom = 1.f / sqrt_var;
        float gamma = use_scale_shift ? ss.get_elem(c) : 1.f;
        float beta = use_scale_shift ? ss.get_elem(C + c) : 0;

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(p, mb, c, d, h, w);
            float x_hat = (src.get_elem(off) - smean) * rcp_denom;
            float res = gamma * x_hat + beta;
            float &D = ((float *)dst)[off];
            if (fuse_relu && res < 0) res = 0;
            maybe_post_ops(res, D, attr);
            D = maybe_saturate(dt, res);
        }
    });
}

void compute_ref_bwd(const prb_t *p, const dnn_mem_t &src,
        const dnn_mem_t &mean, const dnn_mem_t &var, const dnn_mem_t &d_dst,
        const dnn_mem_t &ss, const dnn_mem_t &ws, dnn_mem_t &d_src,
        dnn_mem_t &d_ss) {
    const int64_t MB = p->mb;
    const int64_t C = p->ic;
    const int64_t D = p->id;
    const int64_t H = p->ih;
    const int64_t W = p->iw;
    const bool glob_stats = p->flags & GLOB_STATS;
    const bool use_scale_shift = p->flags & USE_SCALESHIFT;
    const bool fuse_relu = p->flags & FUSE_NORM_RELU;

    const float MB_SP = MB * D * H * W;

    dnnl::impl::parallel_nd(C, [&](int64_t c) {
        float smean = mean.get_elem(c);
        float rcp_denom = 1.f / sqrtf(var.get_elem(c) + p->eps);
        float gamma = use_scale_shift ? ss.get_elem(c) : 1.f;

        float d_gamma = 0;
        float d_beta = 0;

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(p, mb, c, d, h, w);
            float dd = d_dst.get_elem(off);
            if (fuse_relu && ws.get_elem(off) == 0) dd = 0;

            float x_hat = (src.get_elem(off) - smean) * rcp_denom;
            d_gamma += dd * x_hat;
            d_beta += dd;
        }

        if (use_scale_shift && (p->dir & FLAG_WEI)) {
            d_ss.set_elem(c, d_gamma);
            d_ss.set_elem(C + c, d_beta);
        }

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(p, mb, c, d, h, w);
            float dd = d_dst.get_elem(off);
            if (fuse_relu && ws.get_elem(off) == 0) dd = 0;
            float ds = dd;

            if (!glob_stats) {
                float x_hat = (src.get_elem(off) - smean) * rcp_denom;
                ds -= (d_beta + x_hat * d_gamma) / MB_SP;
            }

            d_src.set_elem(off, rcp_denom * ds * gamma);
        }
    });
}

} // namespace bnorm
