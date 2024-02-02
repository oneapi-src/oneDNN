/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"
#include "cpu/ref_pooling.hpp"
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static inline dim_t get_offset(const memory_desc_wrapper &mdw, dim_t n, dim_t c,
        dim_t d, dim_t h, dim_t w) {
    switch (mdw.ndims()) {
        case 3: return mdw.off(n, c, w);
        case 4: return mdw.off(n, c, h, w);
        case 5: return mdw.off(n, c, d, h, w);
        default: assert(!"Invalid tensor dimension in pooling");
    }
    return 0;
}

using namespace nstl;

template <data_type_t data_type, data_type_t acc_type>
status_t ref_pooling_fwd_t<data_type, acc_type>::execute_forward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);
    auto ws = CTX_OUT_CLEAN_MEM(unsigned char *, DNNL_ARG_WORKSPACE, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;
    if (ws) assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);

    const auto alg = pd()->desc()->alg_kind;
    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();
    const dim_t DD = pd()->KDD();
    const dim_t DH = pd()->KDH();
    const dim_t DW = pd()->KDW();

    auto set_ws = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow,
                          dim_t value) {
        if (ws) {
            const auto off = get_offset(ws_d, mb, oc, od, oh, ow);
            if (ws_dt == data_type::u8) {
                assert(0 <= value
                        && value <= numeric_limits<typename prec_traits<
                                        data_type::u8>::type>::max());
                ws[off] = value;
            } else
                reinterpret_cast<int *>(ws)[off] = value;
        }
    };

    auto ker_max = [=](float &d, dim_t mb, dim_t oc, dim_t od, dim_t oh,
                           dim_t ow) {
        set_ws(mb, oc, od, oh, ow, 0);
        for (dim_t kd = 0; kd < KD; ++kd) {
            const dim_t id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto off = get_offset(src_d, mb, oc, id, ih, iw);
                    auto s = src[off];
                    if (s > d) {
                        d = s;
                        set_ws(mb, oc, od, oh, ow, (kd * KH + kh) * KW + kw);
                    }
                }
            }
        }
    };

    auto ker_avg = [=](float &d, dim_t mb, dim_t oc, dim_t od, dim_t oh,
                           dim_t ow) {
        for (dim_t kd = 0; kd < KD; ++kd) {
            const dim_t id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto off = get_offset(src_d, mb, oc, id, ih, iw);
                    d += src[off];
                }
            }
        }
        int num_summands;
        if (alg == alg_kind::pooling_avg_include_padding)
            num_summands = KW * KH * KD;
        else {
            auto id_start = od * SD - padF;
            auto ih_start = oh * SH - padT;
            auto iw_start = ow * SW - padL;
            auto id_end = od * SD - padF + (KD - 1) * DD + KD;
            auto ih_end = oh * SH - padT + (KH - 1) * DH + KH;
            auto iw_end = ow * SW - padL + (KW - 1) * DW + KW;

            auto id_start_excluded
                    = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
            auto ih_start_excluded
                    = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
            auto iw_start_excluded
                    = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
            auto id_end_excluded
                    = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
            auto ih_end_excluded
                    = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
            auto iw_end_excluded
                    = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

            num_summands = (KD - id_start_excluded - id_end_excluded)
                    * (KH - ih_start_excluded - ih_end_excluded)
                    * (KW - iw_start_excluded - iw_end_excluded);
        }
        d /= num_summands;
    };

    const bool is_max_pool = alg == alg_kind::pooling_max;

    float base_res
            = is_max_pool ? (float)numeric_limits<data_t>::lowest() : 0.f;
    using ker_t
            = std::function<void(float &, dim_t, dim_t, dim_t, dim_t, dim_t)>;
    ker_t kernel = is_max_pool ? (ker_t)ker_max : (ker_t)ker_avg;

    parallel_nd(MB, OC, OD, OH, OW,
            [&](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
                auto data_p_off = get_offset(dst_d, mb, oc, od, oh, ow);
                auto data_l_off
                        = (((mb * OC + oc) * OD + od) * OH + oh) * OW + ow;
                float res = base_res;
                kernel(res, mb, oc, od, oh, ow);

                ref_post_ops_t::args_t args;
                args.ctx = &ctx;
                args.l_offset = data_l_off;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(res, args);

                dst[data_p_off] = cpu::q10n::saturate_and_round<data_t>(res);
            });

    return status::success;
}

status_t ref_pooling_bwd_t::execute(const exec_ctx_t &ctx) const {
    status_t status = status::success;

    const auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    const auto ws = CTX_IN_MEM(const void *, DNNL_ARG_WORKSPACE);
    auto diff_src_ptr = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper ws_d(pd()->workspace_md());

    auto scratchpad = ctx.get_scratchpad_grantor();
    float *cvt_src = scratchpad.template get<float>(
            memory_tracking::names::key_pool_src_bf16cvt);
    void *diff_src = (diff_src_d.data_type() != data_type::f32)
            ? cvt_src
            : reinterpret_cast<float *>(diff_src_ptr);

    const auto alg = pd()->desc()->alg_kind;
    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();
    const dim_t DD = pd()->KDD();
    const dim_t DH = pd()->KDH();
    const dim_t DW = pd()->KDW();

    auto ker_max = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
        const auto ws_off = get_offset(ws_d, mb, oc, od, oh, ow);
        const dim_t index = io::load_int_value(ws_d.data_type(), ws, ws_off);
        const dim_t kd = (index / KW) / KH;
        const dim_t kh = (index / KW) % KH;
        const dim_t kw = index % KW;
        const dim_t id = od * SD - padF + kd * (DD + 1);
        const dim_t ih = oh * SH - padT + kh * (DH + 1);
        const dim_t iw = ow * SW - padL + kw * (DW + 1);

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID) return;
        if (ih < 0 || ih >= IH) return;
        if (iw < 0 || iw >= IW) return;

        const auto diff_src_off = get_offset(diff_src_d, mb, oc, id, ih, iw);
        const auto diff_dst_off = get_offset(diff_dst_d, mb, oc, od, oh, ow);
        const float dd = io::load_float_value(
                diff_dst_d.data_type(), diff_dst, diff_dst_off);
        const float ds
                = io::load_float_value(data_type::f32, diff_src, diff_src_off);
        io::store_float_value(data_type::f32, ds + dd, diff_src, diff_src_off);
    };

    auto ker_avg = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
        dim_t num_summands = KW * KH * KD;
        if (alg != alg_kind::pooling_avg_include_padding) {
            auto id_start = od * SD - padF;
            auto ih_start = oh * SH - padT;
            auto iw_start = ow * SW - padL;
            auto id_end = od * SD - padF + (KD - 1) * DD + KD;
            auto ih_end = oh * SH - padT + (KH - 1) * DH + KH;
            auto iw_end = ow * SW - padL + (KW - 1) * DW + KW;

            auto id_start_excluded
                    = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
            auto ih_start_excluded
                    = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
            auto iw_start_excluded
                    = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
            auto id_end_excluded
                    = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
            auto ih_end_excluded
                    = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
            auto iw_end_excluded
                    = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

            num_summands = (KD - id_start_excluded - id_end_excluded)
                    * (KH - ih_start_excluded - ih_end_excluded)
                    * (KW - iw_start_excluded - iw_end_excluded);
        }

        for (dim_t kd = 0; kd < KD; ++kd) {
            const dim_t id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto diff_src_off
                            = get_offset(diff_src_d, mb, oc, id, ih, iw);
                    const auto diff_dst_off
                            = get_offset(diff_dst_d, mb, oc, od, oh, ow);
                    const float dd = io::load_float_value(
                            diff_dst_d.data_type(), diff_dst, diff_dst_off);
                    const float ds = io::load_float_value(
                            data_type::f32, diff_src, diff_src_off);
                    io::store_float_value(data_type::f32,
                            ds + (dd / num_summands), diff_src, diff_src_off);
                }
            }
        }
    };

    dim_t ow_start
            = max(dim_t(0), utils::div_up(padL - ((KW - 1) * DW + KW) + 1, SW));
    dim_t ow_end = min(OW, 1 + (padL + IW - 1) / SW);

    dim_t oh_start
            = max(dim_t(0), utils::div_up(padT - ((KH - 1) * DH + KH) + 1, SH));
    dim_t oh_end = min(OH, 1 + (padT + IH - 1) / SH);

    dim_t od_start
            = max(dim_t(0), utils::div_up(padF - ((KD - 1) * DD + KD) + 1, SD));
    dim_t od_end = min(OD, 1 + (padF + ID - 1) / SD);

    using ker_t = std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t)>;
    ker_t kernel
            = alg == alg_kind::pooling_max ? (ker_t)ker_max : (ker_t)ker_avg;

    const int nthr = pd()->nthr_;
    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t start = 0, end = 0;
        balance211(diff_src_d.nelems(true), nthr, ithr, start, end);
        if (start == end) return;

        for (int i = start; i < end; i++)
            io::store_float_value(data_type::f32, 0, diff_src, i);
    });

    parallel_nd_ext(nthr, MB, OC, [&](int, int, dim_t mb, dim_t oc) {
        for_(dim_t od = od_start; od < od_end; ++od)
        for_(dim_t oh = oh_start; oh < oh_end; ++oh)
        for (dim_t ow = ow_start; ow < ow_end; ++ow) {
            kernel(mb, oc, od, oh, ow);
        }
    });

    if (diff_src_d.data_type() != data_type::f32) {
        parallel(nthr, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(diff_src_d.nelems(true), nthr, ithr, start, end);
            if (start == end) return;

            const auto diff_src_dt_size = diff_src_d.data_type_size();
            const auto in_ptr = reinterpret_cast<float *>(diff_src) + start;
            auto out_ptr = reinterpret_cast<char *>(diff_src_ptr)
                    + start * diff_src_dt_size;

            types::cvt_from_float(
                    diff_src_d.data_type(), out_ptr, in_ptr, end - start);
        });
    }

    return status::success;
}

template struct ref_pooling_fwd_t<data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s32>;
template struct ref_pooling_fwd_t<data_type::bf16, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::f16, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s8, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::u8, data_type::s32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
