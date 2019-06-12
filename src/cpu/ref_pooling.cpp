/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "ref_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace nstl;

template <data_type_t data_type, data_type_t acc_type>
void ref_pooling_fwd_t<data_type, acc_type>::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);
    auto ws = CTX_OUT_MEM(unsigned char *, MKLDNN_ARG_WORKSPACE);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper ws_d(pd()->workspace_md());

    auto alg = pd()->desc()->alg_kind;
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    if (ws)
        assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);

    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const bool is_3d = pd()->desc()->src_desc.ndims == 5;

    auto set_ws = [=](int mb, int oc, int od, int oh, int ow, int value) {
        if (ws) {
            const auto off = is_3d
                ? ws_d.off(mb, oc, od, oh, ow)
                : ws_d.off(mb, oc, oh, ow);
            if (ws_dt == data_type::u8) {
                assert(0 <= value && value <= numeric_limits<typename
                        prec_traits<data_type::u8>::type>::max());
                ws[off] = value;
            } else
                reinterpret_cast<int *>(ws)[off] = value;
        }
    };

    auto ker_max = [=](data_t *d, int mb, int oc, int od, int oh, int ow) {
        for (int kd = 0; kd < KD; ++kd) {
            const int id = od * SD - padF + kd;
            if (id < 0 || id >= ID) continue;
            for (int kh = 0; kh < KH; ++kh) {
                const int ih = oh * SH - padT + kh;
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    const int iw = ow * SW - padL + kw;
                    if (iw < 0 || iw >= IW) continue;

                    const auto off = is_3d ? src_d.off(mb, oc, id, ih, iw)
                                           : src_d.off(mb, oc, ih, iw);
                    auto s = src[off];
                    if (s > d[0]) {
                        d[0] = s;
                        set_ws(mb, oc, od, oh, ow, (kd * KH + kh) * KW + kw);
                    }
                }
            }
        }
    };

    auto ker_avg = [=](data_t *d, int mb, int oc, int od, int oh, int ow) {
        auto id_start = max(od * SD - padF, 0);
        auto ih_start = max(oh * SH - padT, 0);
        auto iw_start = max(ow * SW - padL, 0);
        auto id_end = min(od * SD - padF + KD, ID);
        auto ih_end = min(oh * SH - padT + KH, IH);
        auto iw_end = min(ow * SW - padL + KW, IW);

        auto num_summands = (alg == alg_kind::pooling_avg_include_padding)
            ? KW * KH * KD
            : (id_end - id_start) * (ih_end - ih_start) * (iw_end - iw_start);

        acc_data_t dst = 0;
        for (int id = id_start; id < id_end; ++id)
        for (int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            const auto off = is_3d ? src_d.off(mb, oc, id, ih, iw)
                                   : src_d.off(mb, oc, ih, iw);
            dst += src[off];
        }

        d[0] = math::out_round<data_t>((float)dst / num_summands);
    };

    const int MB = pd()->MB();
    const int OC = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    if (alg == alg_kind::pooling_max) {
        parallel_nd(MB, OC, OD, OH, OW,
            [&](int mb, int oc, int od, int oh, int ow) {
            data_t *d = is_3d
                ? &dst[dst_d.off(mb, oc, od, oh, ow)]
                : &dst[dst_d.off(mb, oc, oh, ow)];
            d[0] = numeric_limits<data_t>::lowest();
            set_ws(mb, oc, od, oh, ow, 0);
            ker_max(d, mb, oc, od, oh, ow);
        });
    } else {
        parallel_nd(MB, OC, OD, OH, OW,
            [&](int mb, int oc, int od, int oh, int ow) {
            data_t *d = is_3d
                ? &dst[dst_d.off(mb, oc, od, oh, ow)]
                : &dst[dst_d.off(mb, oc, oh, ow)];
            d[0] = 0;
            ker_avg(d, mb, oc, od, oh, ow);
        });
    }
}

template <data_type_t data_type>
void ref_pooling_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {

    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const unsigned char *, MKLDNN_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;
    auto alg = pd()->desc()->alg_kind;

    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    auto ker_zero = [=](int mb, int oc) {
        for (int id = 0; id < ID; ++id)
        for (int ih = 0; ih < IH; ++ih)
        for (int iw = 0; iw < IW; ++iw) {
            const auto off = is_3d
                ? diff_src_d.off(mb, oc, id, ih, iw)
                : diff_src_d.off(mb, oc, ih, iw);
            diff_src[off] = data_type_t(0);
        }
    };

    auto ker_max = [=](const data_t *d, int mb, int oc, int od, int oh,
            int ow) {
        const auto ws_off = is_3d
            ? ws_d.off(mb, oc, od, oh, ow)
            : ws_d.off(mb, oc, oh, ow);
        const int index = ws_d.data_type() == data_type::u8
            ? (int)ws[ws_off]
            : ((int *)ws)[ws_off];
        const int kd = (index / KW) / KH;
        const int kh = (index / KW) % KH;
        const int kw = index % KW;
        const int id = od * SD - padF + kd;
        const int ih = oh * SH - padT + kh;
        const int iw = ow * SW - padL + kw;

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID) return;
        if (ih < 0 || ih >= IH) return;
        if (iw < 0 || iw >= IW) return;

        const auto off = is_3d ? diff_src_d.off(mb, oc, id, ih, iw)
                               : diff_src_d.off(mb, oc, ih, iw);
        diff_src[off] += d[0];
    };

    auto ker_avg = [=](
            const data_t *d, int mb, int oc, int od, int oh, int ow) {
        auto id_start = max(od * SD - padF, 0);
        auto ih_start = max(oh * SH - padT, 0);
        auto iw_start = max(ow * SW - padL, 0);
        auto id_end = min(od * SD - padF + KD, ID);
        auto ih_end = min(oh * SH - padT + KH, IH);
        auto iw_end = min(ow * SW - padL + KW, IW);

        auto num_summands = (alg == alg_kind::pooling_avg_include_padding)
            ? KW * KH * KD
            : (id_end - id_start) * (ih_end - ih_start) * (iw_end - iw_start);

        for (int id = id_start; id < id_end; ++id)
        for (int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            const auto off = is_3d ? diff_src_d.off(mb, oc, id, ih, iw)
                                   : diff_src_d.off(mb, oc, ih, iw);
            diff_src[off] += d[0] / num_summands;
        }
    };

    const int MB = pd()->MB();
    const int OC = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    int ow_start = max(0, utils::div_up(padL - KW + 1, SW));
    int ow_end = min(OW, 1 + (padL + IW - 1) / SW);

    int oh_start = max(0, utils::div_up(padT - KH + 1, SH));
    int oh_end = min(OH, 1 + (padT + IH - 1) / SH);

    int od_start = max(0, utils::div_up(padF - KD + 1, SD));
    int od_end = min(OD, 1 + (padF + ID - 1) / SD);

    if (alg == alg_kind::pooling_max) {
        parallel_nd(MB, OC, [&](int mb, int oc) {
            ker_zero(mb, oc);
            for (int od = od_start; od < od_end; ++od)
            for (int oh = oh_start; oh < oh_end; ++oh)
            for (int ow = ow_start; ow < ow_end; ++ow) {
                const data_t *d = is_3d
                        ? &diff_dst[diff_dst_d.off(mb, oc, od, oh, ow)]
                        : &diff_dst[diff_dst_d.off(mb, oc, oh, ow)];
                ker_max(d, mb, oc, od, oh, ow);
            }
        });
    } else {
        parallel_nd(MB, OC, [&](int mb, int oc) {
            ker_zero(mb, oc);
            for (int od = od_start; od < od_end; ++od)
            for (int oh = oh_start; oh < oh_end; ++oh)
            for (int ow = ow_start; ow < ow_end; ++ow) {
                const data_t *d = is_3d
                        ? &diff_dst[diff_dst_d.off(mb, oc, od, oh, ow)]
                        : &diff_dst[diff_dst_d.off(mb, oc, oh, ow)];
                ker_avg(d, mb, oc, od, oh, ow);
            }
        });
    }
}

template struct ref_pooling_fwd_t<data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s32>;
template struct ref_pooling_fwd_t<data_type::bf16, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s8, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::u8, data_type::s32>;

template struct ref_pooling_bwd_t<data_type::f32>;
template struct ref_pooling_bwd_t<data_type::s32>;
template struct ref_pooling_bwd_t<data_type::bf16>;
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
