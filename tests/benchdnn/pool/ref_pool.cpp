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

#include "pool/pool.hpp"

namespace pool {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);
    const dnn_mem_t &ws = args.find(DNNL_ARG_WORKSPACE);

    float *dst_ptr = (float *)dst;

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    auto ker = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
        const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
        const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
        const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
        const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;
        const int64_t DD = prb->dd, DH = prb->dh, DW = prb->dw;

        // XXX: this is a hack to let tests with padded area to pass for bf16
        // dt due to the library initialize values with -max_dt, but not -INF.
        float max_value = lowest_dt(prb->dst_dt());
        float avg_value = 0.;
        // Set initial value based on ws data type
        int ws_off = prb->kernel_size() <= UINT8_MAX ? UINT8_MAX : INT_MAX;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t id = od * SD - PD + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH - PH + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW - PW + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    float s = src.get_elem(src_off_f(prb, mb, ic, id, ih, iw));
                    if (s > max_value) {
                        max_value = s;
                        ws_off = ker_off_f(prb, kd, kh, kw);
                    }
                    avg_value += s;
                }
            }
        }

        const auto dst_off = dst_off_f(prb, mb, ic, od, oh, ow);
        float res = 0.f;
        if (prb->alg == max) {
            res = max_value;
            if (!(prb->dir & FLAG_INF)) ws.set_elem(dst_off, ws_off);
        } else if (prb->alg == avg_np || prb->alg == avg_p) {
            res = avg_value / get_num_summands(prb, od, oh, ow);
        }

        const auto v_po_vals = prepare_po_vals(dst, args, v_po_masks, dst_off);

        maybe_post_ops(prb->attr, res, 0.f, v_po_vals);
        dst_ptr[dst_off] = res;
    };

    benchdnn_parallel_nd(prb->mb, prb->ic, prb->od, prb->oh, prb->ow,
            [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
                ker(mb, ic, od, oh, ow);
            });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &d_dst = args.find(DNNL_ARG_DIFF_DST);
    const dnn_mem_t &ws = args.find(DNNL_ARG_WORKSPACE);
    const dnn_mem_t &d_src = args.find(DNNL_ARG_DIFF_SRC);

    float *d_src_ptr = (float *)d_src;

    auto zero_d_src = [&](int64_t mb, int64_t ic) {
        for_(int64_t id = 0; id < prb->id; ++id)
        for_(int64_t ih = 0; ih < prb->ih; ++ih)
        for (int64_t iw = 0; iw < prb->iw; ++iw)
            d_src_ptr[src_off_f(prb, mb, ic, id, ih, iw)] = 0.f;
    };

    auto ker = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
        const auto d_dst_off = dst_off_f(prb, mb, ic, od, oh, ow);
        float d_dst_val = d_dst.get_elem(d_dst_off);
        int ws_off = (prb->alg == max) ? ws.get_elem(d_dst_off) : 0;

        const int64_t ID = prb->id, IH = prb->ih, IW = prb->iw;
        const int64_t KD = prb->kd, KH = prb->kh, KW = prb->kw;
        const int64_t DD = prb->dd, DH = prb->dh, DW = prb->dw;
        const int64_t PD = prb->pd, PH = prb->ph, PW = prb->pw;
        const int64_t SD = prb->sd, SH = prb->sh, SW = prb->sw;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t id = od * SD - PD + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH - PH + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW - PW + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    float &S = d_src_ptr[src_off_f(prb, mb, ic, id, ih, iw)];
                    if (prb->alg == max) {
                        if (ws_off == ker_off_f(prb, kd, kh, kw))
                            S += d_dst_val;
                    } else if (prb->alg == avg_np || prb->alg == avg_p)
                        S += d_dst_val / get_num_summands(prb, od, oh, ow);
                }
            }
        }
    };

    benchdnn_parallel_nd(prb->mb, prb->ic, [&](int64_t mb, int64_t ic) {
        zero_d_src(mb, ic);
        for_(int64_t od = 0; od < prb->od; ++od)
        for_(int64_t oh = 0; oh < prb->oh; ++oh)
        for (int64_t ow = 0; ow < prb->ow; ++ow)
            ker(mb, ic, od, oh, ow);
    });
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    compute_ref_fwd(prb, args);
    if (prb->dir & FLAG_BWD) compute_ref_bwd(prb, args);
}

} // namespace pool
