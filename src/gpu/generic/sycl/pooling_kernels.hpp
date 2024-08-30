/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_POOLING_KERNELS_HPP
#define GPU_GENERIC_SYCL_POOLING_KERNELS_HPP

#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/nstl.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace nstl;
struct pooling_fwd_kernel_vec_t {
    pooling_fwd_kernel_vec_t(const sycl_pooling_fwd_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , dst_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , ws_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_WORKSPACE))
        , po_args_(cgh, ctx) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t src_mem(src_, conf_.src_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);

        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        const bool is_max_pool = conf_.alg == alg_kind::pooling_max;
        float base_res = is_max_pool ? data_conv() : 0.f;
        dim_t MB = conf_.MB;
        dim_t OC = conf_.OC;
        dim_t OD = conf_.OD;
        dim_t OH = conf_.OH;
        dim_t OW = conf_.OW;
        const dim_t work_amount = MB * OC * OD * OH * OW;
        if (work_amount == 0) return;

        dims_t dst_dims;
        for (int i = 0; i < xpu::sycl::md_t::max_dims; i++) {
            if (i < dst_mem.md().ndims()) {
                dst_dims[i] = dst_mem.md().dims()[i];
            } else {
                dst_dims[i] = 1;
            }
        }

        dim_t start {0}, end {0};
        balance211(work_amount, conf_.n_thr, ithr, start, end);
        dim_t mb {0}, oc {0}, od {0}, oh {0}, ow {0};
        utils::nd_iterator_init(start, mb, MB, oc, OC, od, OD, oh, OH, ow, OW);
        for (dim_t iwork = start; iwork < end; ++iwork) {
            auto data_p_off = get_offset(dst_md(), mb, oc, od, oh, ow);
            auto data_l_off = (((mb * OC + oc) * OD + od) * OH + oh) * OW + ow;
            float res = base_res;

            if (is_max_pool) {
                ker_max(src_mem, res, mb, oc, od, oh, ow);
            } else {
                ker_avg(src_mem, res, mb, oc, od, oh, ow);
            }

            dims_t off;
            utils::l_dims_by_l_offset(
                    off, data_l_off, dst_dims, dst_md().ndims());
            res = conf_.post_ops.apply(res, po_args_, off);

            dst_mem.store(res, data_p_off);
            utils::nd_iterator_step(mb, MB, oc, OC, od, OD, oh, OH, ow, OW);
        }
    }

private:
    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    const xpu::sycl::md_t &ws_md() const { return conf_.ws_md; }

    void *gen_ptr(xpu::sycl::in_memory_arg_t gen_) const {
        return gen_.get_pointer();
    }
    void *ws_ptr() const { return ws_.get_pointer(); }

    static dim_t get_offset(const xpu::sycl::md_t &mdw, dim_t n, dim_t c,
            dim_t d, dim_t h, dim_t w) {
        switch (mdw.ndims()) {
            case 3: return mdw.off(n, c, w);
            case 4: return mdw.off(n, c, h, w);
            case 5: return mdw.off(n, c, d, h, w);
            default: return 0;
        }
        return 0;
    }
    float data_conv() const {
        switch (src_md().data_type()) {
            case data_type::bf16:
                return (float)
                        std::numeric_limits<xpu::sycl::bfloat16_t>::lowest();
            case data_type::s8:
                return (float)numeric_limits<typename xpu::sycl::prec_traits<
                        data_type::s8>::type>::lowest();
            case data_type::f16:
                return (float)
                        std::numeric_limits<typename xpu::sycl::prec_traits<
                                data_type::f16>::type>::lowest();
            case data_type::s32:
                return (float)numeric_limits<typename xpu::sycl::prec_traits<
                        data_type::s32>::type>::lowest();
            case data_type::u8:
                return (float)numeric_limits<typename xpu::sycl::prec_traits<
                        data_type::u8>::type>::lowest();
            default:
                return (float)numeric_limits<typename xpu::sycl::prec_traits<
                        data_type::f32>::type>::lowest();
        }
    }

    void set_ws(dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow,
            dim_t value) const {
        if (ws_ptr()) {
            memory_tensor_t ws_mem(ws_, conf_.ws_md);
            const auto off = get_offset(ws_md(), mb, oc, od, oh, ow);
            ws_mem.store(value, off);
        }
    }

    void ker_max(const in_memory_tensor_t &src_mem, float &d, dim_t mb,
            dim_t oc, dim_t od, dim_t oh, dim_t ow) const {
        set_ws(mb, oc, od, oh, ow, 0);
        for (dim_t kd = 0; kd < conf_.KD; ++kd) {
            const dim_t id = od * conf_.SD - conf_.padF + kd * (conf_.DD + 1);
            if (id < 0 || id >= conf_.ID) continue;
            for (dim_t kh = 0; kh < conf_.KH; ++kh) {
                const dim_t ih
                        = oh * conf_.SH - conf_.padT + kh * (conf_.DH + 1);
                if (ih < 0 || ih >= conf_.IH) continue;
                for (dim_t kw = 0; kw < conf_.KW; ++kw) {
                    const dim_t iw
                            = ow * conf_.SW - conf_.padL + kw * (conf_.DW + 1);
                    if (iw < 0 || iw >= conf_.IW) continue;

                    const auto off = get_offset(src_md(), mb, oc, id, ih, iw);
                    auto s = src_mem.load(off);

                    if (s > d) {
                        d = s;
                        set_ws(mb, oc, od, oh, ow,
                                (kd * conf_.KH + kh) * conf_.KW + kw);
                    }
                }
            }
        }
    }

    void ker_avg(const in_memory_tensor_t &src_mem, float &d, dim_t mb,
            dim_t oc, dim_t od, dim_t oh, dim_t ow) const {
        for (dim_t kd = 0; kd < conf_.KD; ++kd) {
            const dim_t id = od * conf_.SD - conf_.padF + kd * (conf_.DD + 1);
            if (id < 0 || id >= conf_.ID) continue;
            for (dim_t kh = 0; kh < conf_.KH; ++kh) {
                const dim_t ih
                        = oh * conf_.SH - conf_.padT + kh * (conf_.DH + 1);
                if (ih < 0 || ih >= conf_.IH) continue;
                for (dim_t kw = 0; kw < conf_.KW; ++kw) {
                    const dim_t iw
                            = ow * conf_.SW - conf_.padL + kw * (conf_.DW + 1);
                    if (iw < 0 || iw >= conf_.IW) continue;

                    const auto off = get_offset(src_md(), mb, oc, id, ih, iw);
                    float s = src_mem.load(off);
                    d += s;
                }
            }
        }
        int num_summands;
        if (conf_.alg == alg_kind::pooling_avg_include_padding)
            num_summands = conf_.KW * conf_.KH * conf_.KD;
        else {
            auto id_start = od * conf_.SD - conf_.padF;
            auto ih_start = oh * conf_.SH - conf_.padT;
            auto iw_start = ow * conf_.SW - conf_.padL;
            auto id_end = od * conf_.SD - conf_.padF + (conf_.KD - 1) * conf_.DD
                    + conf_.KD;
            auto ih_end = oh * conf_.SH - conf_.padT + (conf_.KH - 1) * conf_.DH
                    + conf_.KH;
            auto iw_end = ow * conf_.SW - conf_.padL + (conf_.KW - 1) * conf_.DW
                    + conf_.KW;

            auto id_start_excluded = id_start < 0
                    ? (0 - id_start - 1) / (conf_.DD + 1) + 1
                    : 0;
            auto ih_start_excluded = ih_start < 0
                    ? (0 - ih_start - 1) / (conf_.DH + 1) + 1
                    : 0;
            auto iw_start_excluded = iw_start < 0
                    ? (0 - iw_start - 1) / (conf_.DW + 1) + 1
                    : 0;
            auto id_end_excluded = id_end > conf_.ID
                    ? (id_end - conf_.ID - 1) / (conf_.DD + 1) + 1
                    : 0;
            auto ih_end_excluded = ih_end > conf_.IH
                    ? (ih_end - conf_.IH - 1) / (conf_.DH + 1) + 1
                    : 0;
            auto iw_end_excluded = iw_end > conf_.IW
                    ? (iw_end - conf_.IW - 1) / (conf_.DW + 1) + 1
                    : 0;

            num_summands = (conf_.KD - id_start_excluded - id_end_excluded)
                    * (conf_.KH - ih_start_excluded - ih_end_excluded)
                    * (conf_.KW - iw_start_excluded - iw_end_excluded);
        }
        d /= num_summands;
    }

    sycl_pooling_fwd_conf_t conf_;

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    xpu::sycl::out_memory_arg_t ws_;
    post_op_input_args po_args_;
};

struct pooling_bwd_kernel_vec_t {
    pooling_bwd_kernel_vec_t(const sycl_pooling_bwd_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , diff_dst_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST))
        , diff_src_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC))
        , ws_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WORKSPACE)) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t diff_src_mem(diff_src_, conf_.diff_src_md);
        memory_tensor_t diff_dst_mem(diff_dst_, conf_.diff_dst_md);

        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();

        dim_t ow_start = max(dim_t(0),
                math::div_up(
                        conf_.padL - ((conf_.KW - 1) * conf_.DW + conf_.KW) + 1,
                        conf_.SW));
        dim_t ow_end
                = min(conf_.OW, 1 + (conf_.padL + conf_.IW - 1) / conf_.SW);

        dim_t oh_start = max(dim_t(0),
                math::div_up(
                        conf_.padT - ((conf_.KH - 1) * conf_.DH + conf_.KH) + 1,
                        conf_.SH));
        dim_t oh_end
                = min(conf_.OH, 1 + (conf_.padT + conf_.IH - 1) / conf_.SH);

        dim_t od_start = max(dim_t(0),
                math::div_up(
                        conf_.padF - ((conf_.KD - 1) * conf_.DD + conf_.KD) + 1,
                        conf_.SD));
        dim_t od_end
                = min(conf_.OD, 1 + (conf_.padF + conf_.ID - 1) / conf_.SD);

        const bool is_max_pool = conf_.alg == alg_kind::pooling_max;
        dim_t MB = conf_.MB;
        dim_t OC = conf_.OC;
        const dim_t work_amount = MB * OC;
        if (work_amount == 0) return;
        dim_t start {0}, end {0};
        balance211(work_amount, conf_.n_thr, ithr, start, end);
        dim_t mb {0}, oc {0};
        utils::nd_iterator_init(start, mb, MB, oc, OC);
        for (dim_t iwork = start; iwork < end; ++iwork) {
            ker_zero(diff_src_mem, mb, oc);
            for_(dim_t od = od_start; od < od_end; ++od)
            for_(dim_t oh = oh_start; oh < oh_end; ++oh)
            for (dim_t ow = ow_start; ow < ow_end; ++ow) {
                if (is_max_pool) {
                    ker_max(diff_src_mem, diff_dst_mem, mb, oc, od, oh, ow);
                } else {
                    ker_avg(diff_src_mem, diff_dst_mem, mb, oc, od, oh, ow);
                }
            }
            utils::nd_iterator_step(mb, MB, oc, OC);
        }
    }

private:
    const xpu::sycl::md_t &diff_src_md() const { return conf_.diff_src_md; }
    const xpu::sycl::md_t &diff_dst_md() const { return conf_.diff_dst_md; }
    const xpu::sycl::md_t &ws_md() const { return conf_.ws_md; }

    static dim_t get_offset(const xpu::sycl::md_t &mdw, dim_t n, dim_t c,
            dim_t d, dim_t h, dim_t w) {
        switch (mdw.ndims()) {
            case 3: return mdw.off(n, c, w);
            case 4: return mdw.off(n, c, h, w);
            case 5: return mdw.off(n, c, d, h, w);
            default: return 0;
        }
        return 0;
    }

    void ker_zero(out_memory_tensor_t &diff_src_mem, dim_t mb, dim_t oc) const {
        for_(dim_t id = 0; id < conf_.ID; ++id)
        for_(dim_t ih = 0; ih < conf_.IH; ++ih)
        for (dim_t iw = 0; iw < conf_.IW; ++iw) {
            const auto off = get_offset(diff_src_md(), mb, oc, id, ih, iw);
            diff_src_mem.store(0, off);
        }
    }

    void ker_max(out_memory_tensor_t &diff_src_mem,
            const in_memory_tensor_t &diff_dst_mem, dim_t mb, dim_t oc,
            dim_t od, dim_t oh, dim_t ow) const {
        memory_tensor_t ws_mem(ws_, conf_.ws_md);
        const auto ws_off = get_offset(ws_md(), mb, oc, od, oh, ow);

        const int index = ws_mem.load(ws_off);
        const dim_t kd = (index / conf_.KW) / conf_.KH;
        const dim_t kh = (index / conf_.KW) % conf_.KH;
        const dim_t kw = index % conf_.KW;
        const dim_t id = od * conf_.SD - conf_.padF + kd * (conf_.DD + 1);
        const dim_t ih = oh * conf_.SH - conf_.padT + kh * (conf_.DH + 1);
        const dim_t iw = ow * conf_.SW - conf_.padL + kw * (conf_.DW + 1);
        if (id < 0 || id >= conf_.ID) return;
        if (ih < 0 || ih >= conf_.IH) return;
        if (iw < 0 || iw >= conf_.IW) return;

        const auto d_src_off = get_offset(diff_src_md(), mb, oc, id, ih, iw);
        const auto d_dst_off = get_offset(diff_dst_md(), mb, oc, od, oh, ow);
        float v_src = diff_src_mem.load(d_src_off);
        float v_dst = diff_dst_mem.load(d_dst_off);
        v_src += v_dst;
        diff_src_mem.store(v_src, d_src_off);
    }

    void ker_avg(out_memory_tensor_t &diff_src_mem,
            const in_memory_tensor_t &diff_dst_mem, dim_t mb, dim_t oc,
            dim_t od, dim_t oh, dim_t ow) const {
        int num_summands;
        if (conf_.alg == alg_kind::pooling_avg_include_padding)
            num_summands = conf_.KW * conf_.KH * conf_.KD;
        else {
            auto id_start = od * conf_.SD - conf_.padF;
            auto ih_start = oh * conf_.SH - conf_.padT;
            auto iw_start = ow * conf_.SW - conf_.padL;
            auto id_end = od * conf_.SD - conf_.padF + (conf_.KD - 1) * conf_.DD
                    + conf_.KD;
            auto ih_end = oh * conf_.SH - conf_.padT + (conf_.KH - 1) * conf_.DH
                    + conf_.KH;
            auto iw_end = ow * conf_.SW - conf_.padL + (conf_.KW - 1) * conf_.DW
                    + conf_.KW;

            auto id_start_excluded = id_start < 0
                    ? (0 - id_start - 1) / (conf_.DD + 1) + 1
                    : 0;
            auto ih_start_excluded = ih_start < 0
                    ? (0 - ih_start - 1) / (conf_.DH + 1) + 1
                    : 0;
            auto iw_start_excluded = iw_start < 0
                    ? (0 - iw_start - 1) / (conf_.DW + 1) + 1
                    : 0;
            auto id_end_excluded = id_end > conf_.ID
                    ? (id_end - conf_.ID - 1) / (conf_.DD + 1) + 1
                    : 0;
            auto ih_end_excluded = ih_end > conf_.IH
                    ? (ih_end - conf_.IH - 1) / (conf_.DH + 1) + 1
                    : 0;
            auto iw_end_excluded = iw_end > conf_.IW
                    ? (iw_end - conf_.IW - 1) / (conf_.DW + 1) + 1
                    : 0;

            num_summands = (conf_.KD - id_start_excluded - id_end_excluded)
                    * (conf_.KH - ih_start_excluded - ih_end_excluded)
                    * (conf_.KW - iw_start_excluded - iw_end_excluded);
        }
        for (dim_t kd = 0; kd < conf_.KD; ++kd) {
            const dim_t id = od * conf_.SD - conf_.padF + kd * (conf_.DD + 1);
            if (id < 0 || id >= conf_.ID) continue;
            for (dim_t kh = 0; kh < conf_.KH; ++kh) {
                const dim_t ih
                        = oh * conf_.SH - conf_.padT + kh * (conf_.DH + 1);
                if (ih < 0 || ih >= conf_.IH) continue;
                for (dim_t kw = 0; kw < conf_.KW; ++kw) {
                    const dim_t iw
                            = ow * conf_.SW - conf_.padL + kw * (conf_.DW + 1);
                    if (iw < 0 || iw >= conf_.IW) continue;

                    const auto d_src_off
                            = get_offset(diff_src_md(), mb, oc, id, ih, iw);
                    const auto d_dst_off
                            = get_offset(diff_dst_md(), mb, oc, od, oh, ow);
                    float v_src = diff_src_mem.load(d_src_off);
                    ;
                    float v_dst = diff_dst_mem.load(d_dst_off);
                    v_src += v_dst / num_summands;
                    diff_src_mem.store(v_src, d_src_off);
                }
            }
        }
    }

    sycl_pooling_bwd_conf_t conf_;
    xpu::sycl::in_memory_arg_t diff_dst_;
    xpu::sycl::out_memory_arg_t diff_src_;
    xpu::sycl::in_memory_arg_t ws_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
