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

#ifndef GPU_SYCL_RESAMPLING_KERNELS_HPP
#define GPU_SYCL_RESAMPLING_KERNELS_HPP

#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/sycl/resampling_utils.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct resampling_kernel_fwd_vec_t {
    resampling_kernel_fwd_vec_t(const sycl_resampling_conf_t &conf,
            sycl_in_memory_arg_t &src, sycl_out_memory_arg_t &dst,
            sycl_in_memory_arg_t &src_1, sycl_in_memory_arg_t &src_2,
            sycl_in_memory_arg_t &src_3, sycl_in_memory_arg_t &src_4,
            sycl_in_memory_arg_t &src_5)
        : conf_(conf)
        , src_(src)
        , dst_(dst)
        , src_1_(src_1)
        , src_2_(src_2)
        , src_3_(src_3)
        , src_4_(src_4)
        , src_5_(src_5) {}

    void operator()(::sycl::nd_item<1> item) const {
        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dim_t MB = conf_.MB;
        dim_t C = conf_.C;

        dim_t ID = conf_.ID;
        dim_t IH = conf_.IH;
        dim_t IW = conf_.IW;

        dim_t OD = conf_.OD;
        dim_t OH = conf_.OH;
        dim_t OW = conf_.OW;

        auto lin_interp = [&](float c0, float c1, float w) {
            return c0 * w + c1 * (1 - w);
        };
        auto bilin_interp = [&](float c00, float c01, float c10, float c11,
                                    float w0, float w1) {
            return lin_interp(
                    lin_interp(c00, c10, w0), lin_interp(c01, c11, w0), w1);
        };
        auto trilin_interp = [&](float c000, float c001, float c010, float c011,
                                     float c100, float c101, float c110,
                                     float c111, float w0, float w1, float w2) {
            return lin_interp(bilin_interp(c000, c010, c100, c110, w0, w1),
                    bilin_interp(c001, c011, c101, c111, w0, w1), w2);
        };

        const dim_t work_amount = MB * C * OD * OH * OW;
        if (work_amount == 0) return;

        dim_t start {0}, end {0};
        balance211(work_amount, conf_.n_thr, ithr, start, end);
        dim_t mb {0}, c {0}, od {0}, oh {0}, ow {0};
        utils::nd_iterator_init(start, mb, MB, c, C, od, OD, oh, OH, ow, OW);
        for (dim_t iwork = start; iwork < end; ++iwork) {

            const dim_t data_p_off = get_offset(dst_md(), mb, c, od, oh, ow);

            const dim_t data_l_off
                    = (((mb * C + c) * OD + od) * OH + oh) * OW + ow;

            float dst = 0.f;

            if (conf_.alg == alg_kind::resampling_nearest) {
                const dim_t id = resampling_utils::nearest_idx(od, OD, ID);
                const dim_t ih = resampling_utils::nearest_idx(oh, OH, IH);
                const dim_t iw = resampling_utils::nearest_idx(ow, OW, IW);

                dst = load_float_value(src_md().data_type(), src_ptr(),
                        get_offset(src_md(), mb, c, id, ih, iw));

            } else if (conf_.alg == alg_kind::resampling_linear) {

                auto id = resampling_utils::linear_coeffs_t(od, OD, ID);
                auto iw = resampling_utils::linear_coeffs_t(ow, OW, IW);
                auto ih = resampling_utils::linear_coeffs_t(oh, OH, IH);
                float src_l[8] = {0};
                for_(int i = 0; i < 2; i++)
                for_(int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    src_l[4 * i + 2 * j + k]
                            = load_float_value(src_md().data_type(), src_ptr(),
                                    get_offset(src_md(), mb, c, id.idx[i],
                                            ih.idx[j], iw.idx[k]));
                }
                dst = trilin_interp(src_l[0], src_l[1], src_l[2], src_l[3],
                        src_l[4], src_l[5], src_l[6], src_l[7], id.wei[0],
                        ih.wei[0], iw.wei[0]);
            }

            ::sycl::vec<float, 8> dst_arr;
            for (int idx = 0; idx < conf_.po_len; ++idx) {
                float r = 0.0f;
                if (conf_.post_ops.get_post_op_kind(idx)
                        == primitive_kind::binary) {
                    if (idx == 0) { r = dst_value(src_1_, idx, data_l_off); }
                    if (idx == 1) { r = dst_value(src_2_, idx, data_l_off); }
                    if (idx == 2) { r = dst_value(src_3_, idx, data_l_off); }
                    if (idx == 3) { r = dst_value(src_4_, idx, data_l_off); }
                    if (idx == 4) { r = dst_value(src_5_, idx, data_l_off); }
                    dst_arr[idx] = r;
                }
            }
            auto dst_sum = load_float_value(
                    dst_md().data_type(), dst_ptr(), data_p_off);

            dst = conf_.post_ops.apply(dst, dst_sum, dst_arr);
            store_float_value(dst_md().data_type(), dst, dst_ptr(), data_p_off);
            utils::nd_iterator_step(mb, MB, c, C, od, OD, oh, OH, ow, OW);
        }
    }

private:
    const sycl_md_t &src_md() const { return conf_.src_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    void *src_ptr() const { return src_.get_pointer(); }
    void *src_1_ptr() const { return src_1_.get_pointer(); }
    void *src_2_ptr() const { return src_2_.get_pointer(); }
    void *src_3_ptr() const { return src_3_.get_pointer(); }
    void *src_4_ptr() const { return src_4_.get_pointer(); }
    void *src_5_ptr() const { return src_5_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    void *gen_ptr(sycl_in_memory_arg_t gen_) const {
        return gen_.get_pointer();
    }

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

    float dst_value(sycl_in_memory_arg_t arr, int idx, int offset) const {
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

    sycl_resampling_conf_t conf_;

    sycl_in_memory_arg_t src_;
    sycl_out_memory_arg_t dst_;
    sycl_in_memory_arg_t src_1_;
    sycl_in_memory_arg_t src_2_;
    sycl_in_memory_arg_t src_3_;
    sycl_in_memory_arg_t src_4_;
    sycl_in_memory_arg_t src_5_;
};

struct resampling_kernel_bwd_vec_t {
    resampling_kernel_bwd_vec_t(const sycl_resampling_conf_t &conf,
            sycl_in_memory_arg_t &diff_dst, sycl_out_memory_arg_t &diff_src)
        : conf_(conf), diff_dst_(diff_dst), diff_src_(diff_src) {}

    void operator()(::sycl::nd_item<1> item) const {

        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dim_t MB = conf_.MB;
        dim_t C = conf_.C;

        dim_t ID = conf_.ID;
        dim_t IH = conf_.IH;
        dim_t IW = conf_.IW;

        dim_t OD = conf_.OD;
        dim_t OH = conf_.OH;
        dim_t OW = conf_.OW;

        const dim_t work_amount = MB * C * ID * IH * IW;
        if (work_amount == 0) return;

        dim_t start {0}, end {0};
        balance211(work_amount, conf_.n_thr, ithr, start, end);
        dim_t mb {0}, c {0}, id {0}, ih {0}, iw {0};
        utils::nd_iterator_init(start, mb, MB, c, C, id, ID, ih, IH, iw, IW);
        for (dim_t iwork = start; iwork < end; ++iwork) {

            if (conf_.alg == alg_kind::resampling_nearest) {
                const dim_t od_start = resampling_utils::ceil_idx(
                        ((float)id * OD / ID) - 0.5f);
                const dim_t oh_start = resampling_utils::ceil_idx(
                        ((float)ih * OH / IH) - 0.5f);
                const dim_t ow_start = resampling_utils::ceil_idx(
                        ((float)iw * OW / IW) - 0.5f);
                const dim_t od_end = resampling_utils::ceil_idx(
                        ((id + 1.f) * OD / ID) - 0.5f);
                const dim_t oh_end = resampling_utils::ceil_idx(
                        ((ih + 1.f) * OH / IH) - 0.5f);
                const dim_t ow_end = resampling_utils::ceil_idx(
                        ((iw + 1.f) * OW / IW) - 0.5f);

                float ds = 0;
                for_(dim_t od = od_start; od < od_end; od++)
                for_(dim_t oh = oh_start; oh < oh_end; oh++)
                for (dim_t ow = ow_start; ow < ow_end; ow++)
                    ds += load_float_value(diff_dst_md().data_type(),
                            diff_dst_ptr(),
                            get_offset(diff_dst_md(), mb, c, od, oh, ow));
                store_float_value(diff_src_md().data_type(), ds, diff_src_ptr(),
                        get_offset(diff_src_md(), mb, c, id, ih, iw));
            }
            utils::nd_iterator_step(mb, MB, c, C, id, ID, ih, IH, iw, IW);
        }
    }

private:
    const sycl_md_t &diff_src_md() const { return conf_.diff_src_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }

    void *diff_src_ptr() const { return diff_src_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }

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

    sycl_resampling_conf_t conf_;
    sycl_in_memory_arg_t diff_dst_;
    sycl_out_memory_arg_t diff_src_;
};

struct resampling_kernel_bwd_vec1_t {
    resampling_kernel_bwd_vec1_t(const sycl_resampling_conf_t &conf,
            sycl_in_memory_arg_t &diff_dst, sycl_out_memory_arg_t &diff_src)
        : conf_(conf), diff_dst_(diff_dst), diff_src_(diff_src) {}

    void operator()(::sycl::nd_item<1> item) const {
        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dim_t MB = conf_.MB;
        dim_t C = conf_.C;

        dim_t ID = conf_.ID;
        dim_t IH = conf_.IH;
        dim_t IW = conf_.IW;

        dim_t OD = conf_.OD;
        dim_t OH = conf_.OH;
        dim_t OW = conf_.OW;

        const dim_t work_amount = MB * C * ID * IH * IW;
        if (work_amount == 0) return;
        dim_t start {0}, end {0};
        balance211(work_amount, conf_.n_thr, ithr, start, end);
        dim_t mb {0}, c {0}, id {0}, ih {0}, iw {0};
        utils::nd_iterator_init(start, mb, MB, c, C, id, ID, ih, IH, iw, IW);
        for (dim_t iwork = start; iwork < end; ++iwork) {
            resampling_utils::bwd_linear_coeffs_t d(id, OD, ID);
            resampling_utils::bwd_linear_coeffs_t h(ih, OH, IH);
            resampling_utils::bwd_linear_coeffs_t w(iw, OW, IW);

            float ds = 0;
            for_(int i = 0; i < 2; i++)
            for_(int j = 0; j < 2; j++)
            for_(int k = 0; k < 2; k++)
            for_(dim_t od = d.start[i]; od < d.end[i]; od++)
            for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
            for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                const float weight_d
                        = resampling_utils::linear_weight(i, od, OD, ID);
                const float weight_h
                        = resampling_utils::linear_weight(j, oh, OH, IH);
                const float weight_w
                        = resampling_utils::linear_weight(k, ow, OW, IW);

                float dd = load_float_value(diff_dst_md().data_type(),
                        diff_dst_ptr(),
                        get_offset(diff_dst_md(), mb, c, od, oh, ow));
                ds += dd * weight_d * weight_h * weight_w;
            }
            store_float_value(diff_src_md().data_type(), ds, diff_src_ptr(),
                    get_offset(diff_src_md(), mb, c, id, ih, iw));
            utils::nd_iterator_step(mb, MB, c, C, id, ID, ih, IH, iw, IW);
        }
    }

private:
    const sycl_md_t &diff_src_md() const { return conf_.diff_src_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }

    void *diff_src_ptr() const { return diff_src_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }

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

    sycl_resampling_conf_t conf_;
    sycl_in_memory_arg_t diff_dst_;
    sycl_out_memory_arg_t diff_src_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
