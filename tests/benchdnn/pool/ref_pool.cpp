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

#include "tests/test_thread.hpp"

#include "pool/pool.hpp"

namespace pool {

void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst,
        dnn_mem_t &ws) {
    std::vector<int> v_bin_po_mask = p->attr.post_ops.get_binary_po_masks();
    auto ker = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
        const int64_t ID = p->id, IH = p->ih, IW = p->iw;
        const int64_t KD = p->kd, KH = p->kh, KW = p->kw;
        const int64_t PD = p->pd, PH = p->ph, PW = p->pw;
        const int64_t SD = p->sd, SH = p->sh, SW = p->sw;

        // XXX: this is a hack to let tests with padded area to pass for bf16
        // dt due to the library initialize values with -max_dt, but not -INF.
        float max_value = -max_dt(p->cfg[DST].dt);
        float avg_value = 0.;
        int ws_off = INT_MAX;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t id = od * SD - PD + kd;
            if (id < 0 || id >= ID) continue;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH - PH + kh;
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW - PW + kw;
                    if (iw < 0 || iw >= IW) continue;

                    float s = src.get_elem(src_off_f(p, mb, ic, id, ih, iw));
                    if (s > max_value) {
                        max_value = s;
                        ws_off = ker_off_f(p, kd, kh, kw);
                    }
                    avg_value += s;
                }
            }
        }

        const auto dst_off = dst_off_f(p, mb, ic, od, oh, ow);
        float res = 0.f;
        if (p->alg == MAX) {
            res = max_value;
            if (!(p->dir & FLAG_INF)) ws.set_elem(dst_off, ws_off);
        } else if (p->alg == AVG_NP || p->alg == AVG_P) {
            res = avg_value / get_num_summands(p, od, oh, ow);
        }

        std::vector<float> v_binary_vals;
        for (size_t d = 0; d < v_bin_po_mask.size(); ++d) {
            auto bin_po_offset = dst.get_scale_idx(dst_off, v_bin_po_mask[d]);
            float binary_val = binary_po[d].get_elem(bin_po_offset);
            v_binary_vals.push_back(binary_val);
        }
        maybe_post_ops(p->attr, res, 0.f, v_binary_vals);
        dst.set_elem(dst_off, res);
    };

    dnnl::impl::parallel_nd(p->mb, p->ic, p->od, p->oh, p->ow,
            [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
                ker(mb, ic, od, oh, ow);
            });
}

void compute_ref_bwd(const prb_t *p, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, const dnn_mem_t &ws) {
    auto zero_diff_src = [&](int64_t mb, int64_t ic) {
        for (int64_t id = 0; id < p->id; ++id)
            for (int64_t ih = 0; ih < p->ih; ++ih)
                for (int64_t iw = 0; iw < p->iw; ++iw)
                    diff_src.set_elem(src_off_f(p, mb, ic, id, ih, iw), 0.);
    };

    auto ker = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
        const auto diff_dst_off = dst_off_f(p, mb, ic, od, oh, ow);
        float diff_dst_val = diff_dst.get_elem(diff_dst_off);
        int ws_off = (p->alg == MAX) ? ws.get_elem(diff_dst_off) : 0;

        const int64_t ID = p->id, IH = p->ih, IW = p->iw;
        const int64_t KD = p->kd, KH = p->kh, KW = p->kw;
        const int64_t PD = p->pd, PH = p->ph, PW = p->pw;
        const int64_t SD = p->sd, SH = p->sh, SW = p->sw;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t id = od * SD - PD + kd;
            if (id < 0 || id >= ID) continue;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH - PH + kh;
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW - PW + kw;
                    if (iw < 0 || iw >= IW) continue;

                    float &S = ((
                            float *)diff_src)[src_off_f(p, mb, ic, id, ih, iw)];
                    if (p->alg == MAX) {
                        if (ws_off == ker_off_f(p, kd, kh, kw))
                            S += diff_dst_val;
                    } else if (p->alg == AVG_NP || p->alg == AVG_P)
                        S += diff_dst_val / get_num_summands(p, od, oh, ow);
                }
            }
        }
    };

    dnnl::impl::parallel_nd(p->mb, p->ic, [&](int64_t mb, int64_t ic) {
        zero_diff_src(mb, ic);
        for (int64_t od = 0; od < p->od; ++od)
            for (int64_t oh = 0; oh < p->oh; ++oh)
                for (int64_t ow = 0; ow < p->ow; ++ow)
                    ker(mb, ic, od, oh, ow);
    });
}

} // namespace pool
