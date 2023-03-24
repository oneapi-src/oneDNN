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

#ifndef GPU_SYCL_POOLING_KERNELS_HPP
#define GPU_SYCL_POOLING_KERNELS_HPP

#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/nstl.hpp"
#include "common/primitive_attr.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {
using namespace nstl;
struct pooling_fwd_kernel_vec_t {
    pooling_fwd_kernel_vec_t(const sycl_pooling_conf_t &conf,
            sycl_in_memory_arg_t &src, sycl_out_memory_arg_t &dst,
            sycl_out_memory_arg_t &ws, sycl_in_memory_arg_t &src_1,
            sycl_in_memory_arg_t &src_2, sycl_in_memory_arg_t &src_3,
            sycl_in_memory_arg_t &src_4, sycl_in_memory_arg_t &src_5)
        : conf_(conf)
        , src_(src)
        , dst_(dst)
        , ws_(ws)
        , src_1_(src_1)
        , src_2_(src_2)
        , src_3_(src_3)
        , src_4_(src_4)
        , src_5_(src_5) {}

    void operator()(::sycl::nd_item<1> item) const {
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
        dim_t start {0}, end {0};
        balance211(work_amount, conf_.n_thr, ithr, start, end);
        dim_t mb {0}, oc {0}, od {0}, oh {0}, ow {0};
        utils::nd_iterator_init(start, mb, MB, oc, OC, od, OD, oh, OH, ow, OW);
        for (dim_t iwork = start; iwork < end; ++iwork) {
            auto data_p_off = get_offset(dst_md(), mb, oc, od, oh, ow);
            auto data_l_off = (((mb * OC + oc) * OD + od) * OH + oh) * OW + ow;
            float res = base_res;

            if (is_max_pool) {
                ker_max(res, mb, oc, od, oh, ow);
            } else {
                ker_avg(res, mb, oc, od, oh, ow);
            }

            ::sycl::vec<float, 8> dst_arr;

            for (int idx = 0; idx < conf_.po_len; ++idx) {
                float r = 0.0f;
                if (conf_.post_ops.get_post_op_kind(idx)
                        == primitive_kind::binary) {
                    if (idx == 0) { r = dst_Value(src_1_, idx, data_l_off); }
                    if (idx == 1) { r = dst_Value(src_2_, idx, data_l_off); }
                    if (idx == 2) { r = dst_Value(src_3_, idx, data_l_off); }
                    if (idx == 3) { r = dst_Value(src_4_, idx, data_l_off); }
                    if (idx == 4) { r = dst_Value(src_5_, idx, data_l_off); }
                    dst_arr[idx] = r;
                }
            }
            res = conf_.post_ops.apply(res, dst_arr);

            store_float_value(dst_md().data_type(), res, dst_ptr(), data_p_off);
            utils::nd_iterator_step(mb, MB, oc, OC, od, OD, oh, OH, ow, OW);
        }
    }

private:
    const sycl_md_t &src_md() const { return conf_.src_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }
    const sycl_md_t &ws_md() const { return conf_.ws_md; }

    void *src_ptr() const { return src_.get_pointer(); }
    void *gen_ptr(sycl_in_memory_arg_t gen_) const {
        return gen_.get_pointer();
    }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *ws_ptr() const { return ws_.get_pointer(); }

    static dim_t get_offset(
            const sycl_md_t &mdw, dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
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
                return (float)std::numeric_limits<bfloat16_t>::lowest();
            case data_type::s8:
                return (float)numeric_limits<
                        typename prec_traits<data_type::s8>::type>::lowest();
            case data_type::f16:
                return (float)numeric_limits<
                        typename prec_traits<data_type::f16>::type>::lowest();
            case data_type::s32:
                return (float)numeric_limits<
                        typename prec_traits<data_type::s32>::type>::lowest();
            case data_type::u8:
                return (float)numeric_limits<
                        typename prec_traits<data_type::u8>::type>::lowest();
            default:
                return (float)numeric_limits<
                        typename prec_traits<data_type::f32>::type>::lowest();
        }
    }

    float dst_Value(sycl_in_memory_arg_t arr, int idx, int offset) const {
        auto src1_desc = conf_.src1_md[idx];
        dim_t src_dim[DNNL_MAX_NDIMS];
        auto src_dim_ = src1_desc.dims();

        for (int j = 0; j < src1_desc.ndims(); j++) {
            src_dim[j] = src_dim_[j];
        }
        const auto off = get_binary_src1_off(
                src1_desc, src_dim, offset, conf_.dst_dims, conf_.dst_ndims);
        auto dst = load_float_value(src1_desc.data_type(), gen_ptr(arr), off);
        return dst;
    }

    dim_t get_binary_src1_off(const sycl_md_t &src1_md, const dim_t *src_dim,
            const dim_t l_offset, const dim_t *dst_dims,
            const int dst_ndims) const {

        const int mask_binary_po
                = utils::get_dims_mask(dst_dims, src_dim, dst_ndims);

        return get_po_tensor_off(
                src1_md, l_offset, dst_dims, dst_ndims, mask_binary_po);
    }

    dim_t get_po_tensor_off(const sycl_md_t &tensor_md, const dim_t l_offset,
            const dim_t *dst_dims, const int dst_ndims, int mask) const {

        dims_t l_dims_po {};
        get_l_dims_po(l_dims_po, l_offset, dst_dims, dst_ndims, mask);

        return tensor_md.off_v(l_dims_po);
    }

    void get_l_dims_po(dims_t &l_dims_po, const dim_t l_offset,
            const dim_t *dst_dims, const int dst_ndims, int mask) const {
        utils::l_dims_by_l_offset(l_dims_po, l_offset, dst_dims, dst_ndims);
        utils::apply_mask_on_dims(l_dims_po, dst_ndims, mask);
    }

    void set_ws(dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow,
            dim_t value) const {
        if (ws_ptr()) {
            const auto off = get_offset(ws_md(), mb, oc, od, oh, ow);
            const data_type_t ws_dt
                    = ws_ptr() ? ws_md().data_type() : data_type::undef;
            if (ws_dt == data_type::u8) {
                store_float_value(ws_dt, value, ws_ptr(), off);
            } else
                store_float_value(ws_dt, value, ws_ptr(), off);
        }
    }

    void ker_max(
            float &d, dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) const {
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
                    auto s = load_float_value(
                            src_md().data_type(), src_ptr(), off);

                    if (s > d) {
                        d = s;
                        set_ws(mb, oc, od, oh, ow,
                                (kd * conf_.KH + kh) * conf_.KW + kw);
                    }
                }
            }
        }
    }

    void ker_avg(
            float &d, dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) const {
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
                    float s = load_float_value(
                            src_md().data_type(), src_ptr(), off);
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

    sycl_pooling_conf_t conf_;

    sycl_in_memory_arg_t src_;
    sycl_out_memory_arg_t dst_;
    sycl_out_memory_arg_t ws_;
    sycl_in_memory_arg_t src_1_;
    sycl_in_memory_arg_t src_2_;
    sycl_in_memory_arg_t src_3_;
    sycl_in_memory_arg_t src_4_;
    sycl_in_memory_arg_t src_5_;
};

struct pooling_bwd_kernel_vec_t {
    pooling_bwd_kernel_vec_t(const sycl_pooling_conf_t &conf,
            sycl_in_memory_arg_t &diff_dst, sycl_out_memory_arg_t &diff_src,
            sycl_in_memory_arg_t &ws)
        : conf_(conf), diff_dst_(diff_dst), diff_src_(diff_src), ws_(ws) {}

    void operator()(::sycl::nd_item<1> item) const {
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
            ker_zero(mb, oc);
            for_(dim_t od = od_start; od < od_end; ++od)
            for_(dim_t oh = oh_start; oh < oh_end; ++oh)
            for (dim_t ow = ow_start; ow < ow_end; ++ow) {
                if (is_max_pool) {
                    ker_max(mb, oc, od, oh, ow);
                } else {
                    ker_avg(mb, oc, od, oh, ow);
                }
            }
            utils::nd_iterator_step(mb, MB, oc, OC);
        }
    }

private:
    const sycl_md_t &diff_src_md() const { return conf_.diff_src_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }
    const sycl_md_t &ws_md() const { return conf_.ws_md; }

    void *diff_src_ptr() const { return diff_src_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }
    void *ws_ptr() const { return ws_.get_pointer(); }

    static dim_t get_offset(
            const sycl_md_t &mdw, dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
        switch (mdw.ndims()) {
            case 3: return mdw.off(n, c, w);
            case 4: return mdw.off(n, c, h, w);
            case 5: return mdw.off(n, c, d, h, w);
            default: return 0;
        }
        return 0;
    }

    void ker_zero(dim_t mb, dim_t oc) const {
        for_(dim_t id = 0; id < conf_.ID; ++id)
        for_(dim_t ih = 0; ih < conf_.IH; ++ih)
        for (dim_t iw = 0; iw < conf_.IW; ++iw) {
            const auto off = get_offset(diff_src_md(), mb, oc, id, ih, iw);
            store_float_value(
                    diff_src_md().data_type(), 0, diff_src_ptr(), off);
        }
    }

    void ker_max(dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) const {
        const auto ws_off = get_offset(ws_md(), mb, oc, od, oh, ow);

        const int index = ws_md().data_type() == data_type::u8
                ? (int)load_float_value(ws_md().data_type(), ws_ptr(), ws_off)
                : load_float_value(ws_md().data_type(), ws_ptr(), ws_off);
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
        float v_src = load_float_value(
                diff_src_md().data_type(), diff_src_ptr(), d_src_off);
        float v_dst = load_float_value(
                diff_dst_md().data_type(), diff_dst_ptr(), d_dst_off);
        v_src += v_dst;
        store_float_value(
                diff_src_md().data_type(), v_src, diff_src_ptr(), d_src_off);
    }

    void ker_avg(dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) const {
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
                    float v_src = load_float_value(diff_src_md().data_type(),
                            diff_src_ptr(), d_src_off);
                    float v_dst = load_float_value(diff_dst_md().data_type(),
                            diff_dst_ptr(), d_dst_off);
                    v_src += v_dst / num_summands;
                    store_float_value(diff_src_md().data_type(), v_src,
                            diff_src_ptr(), d_src_off);
                }
            }
        }
    }

    sycl_pooling_conf_t conf_;
    sycl_in_memory_arg_t diff_dst_;
    sycl_out_memory_arg_t diff_src_;
    sycl_in_memory_arg_t ws_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
