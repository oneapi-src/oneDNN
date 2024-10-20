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

#ifndef GPU_GENERIC_SYCL_RESAMPLING_KERNELS_HPP
#define GPU_GENERIC_SYCL_RESAMPLING_KERNELS_HPP

#include "common/dnnl_thread.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/resampling_utils.hpp"
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

struct resampling_kernel_fwd_vec_t {
    resampling_kernel_fwd_vec_t(const sycl_resampling_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , dst_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , po_args_(cgh, ctx, conf_.post_ops) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t src_mem(src_, conf_.src_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);

        size_t ithr = item.get_global_id(0);

        const auto &src_ndims = conf_.src_md.ndims();
        const auto &src_dims = conf_.src_md.dims();
        dim_t MB = src_dims[0];
        dim_t C = src_dims[1];
        dim_t ID = src_ndims >= 5 ? src_dims[src_ndims - 3] : 1;
        dim_t IH = src_ndims >= 4 ? src_dims[src_ndims - 2] : 1;
        dim_t IW = src_ndims >= 3 ? src_dims[src_ndims - 1] : 1;

        const auto &dst_ndims = conf_.dst_md.ndims();
        const auto &dst_dims = conf_.dst_md.dims();
        dim_t OD = dst_ndims >= 5 ? dst_dims[dst_ndims - 3] : 1;
        dim_t OH = dst_ndims >= 4 ? dst_dims[dst_ndims - 2] : 1;
        dim_t OW = dst_ndims >= 3 ? dst_dims[dst_ndims - 1] : 1;

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

            float dst = 0.f;

            if (conf_.alg == alg_kind::resampling_nearest) {
                const dim_t id = resampling_utils::nearest_idx(od, OD, ID);
                const dim_t ih = resampling_utils::nearest_idx(oh, OH, IH);
                const dim_t iw = resampling_utils::nearest_idx(ow, OW, IW);

                dst = src_mem.load(get_offset(src_md(), mb, c, id, ih, iw));

            } else if (conf_.alg == alg_kind::resampling_linear) {

                auto id = resampling_utils::linear_coeffs_t(od, OD, ID);
                auto iw = resampling_utils::linear_coeffs_t(ow, OW, IW);
                auto ih = resampling_utils::linear_coeffs_t(oh, OH, IH);
                float src_l[8] = {0};
                for_(int i = 0; i < 2; i++)
                for_(int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    src_l[4 * i + 2 * j + k] = src_mem.load(get_offset(
                            src_md(), mb, c, id.idx[i], ih.idx[j], iw.idx[k]));
                }
                dst = trilin_interp(src_l[0], src_l[1], src_l[2], src_l[3],
                        src_l[4], src_l[5], src_l[6], src_l[7], id.wei[0],
                        ih.wei[0], iw.wei[0]);
            }

            dims_t off {mb, c, od, oh, ow};
            if (dst_ndims == 3) {
                off[2] = ow;
                off[3] = 0;
                off[4] = 0;
            } else if (dst_ndims == 4) {
                off[2] = oh;
                off[3] = ow;
                off[4] = 0;
            }
            dst = conf_.post_ops.apply(dst, dst_, data_p_off, po_args_, off);
            dst_mem.store(dst, data_p_off);
            utils::nd_iterator_step(mb, MB, c, C, od, OD, oh, OH, ow, OW);
        }
    }

private:
    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    void *gen_ptr(xpu::sycl::in_memory_arg_t gen_) const {
        return gen_.get_pointer();
    }

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

    sycl_resampling_conf_t conf_;

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::inout_memory_arg_t dst_;
    post_op_input_args po_args_;
};

struct resampling_kernel_bwd_vec_t {
    resampling_kernel_bwd_vec_t(const sycl_resampling_conf_t &conf,
            xpu::sycl::in_memory_arg_t &diff_dst,
            xpu::sycl::out_memory_arg_t &diff_src)
        : conf_(conf), diff_dst_(diff_dst), diff_src_(diff_src) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t diff_src_mem(diff_src_, conf_.diff_src_md);
        memory_tensor_t diff_dst_mem(diff_dst_, conf_.diff_dst_md);

        size_t ithr = item.get_global_id(0);

        const auto &diff_src_ndims = conf_.diff_src_md.ndims();
        const auto &diff_src_dims = conf_.diff_src_md.dims();
        dim_t MB = diff_src_dims[0];
        dim_t C = diff_src_dims[1];

        dim_t ID = diff_src_ndims >= 5 ? diff_src_dims[diff_src_ndims - 3] : 1;
        dim_t IH = diff_src_ndims >= 4 ? diff_src_dims[diff_src_ndims - 2] : 1;
        dim_t IW = diff_src_ndims >= 3 ? diff_src_dims[diff_src_ndims - 1] : 1;

        const auto &diff_dst_ndims = conf_.diff_dst_md.ndims();
        const auto &diff_dst_dims = conf_.diff_dst_md.dims();
        dim_t OD = diff_dst_ndims >= 5 ? diff_dst_dims[diff_dst_ndims - 3] : 1;
        dim_t OH = diff_dst_ndims >= 4 ? diff_dst_dims[diff_dst_ndims - 2] : 1;
        dim_t OW = diff_dst_ndims >= 3 ? diff_dst_dims[diff_dst_ndims - 1] : 1;

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
                    ds += diff_dst_mem.load(
                            get_offset(diff_dst_md(), mb, c, od, oh, ow));
                diff_src_mem.store(
                        ds, get_offset(diff_src_md(), mb, c, id, ih, iw));
            }
            utils::nd_iterator_step(mb, MB, c, C, id, ID, ih, IH, iw, IW);
        }
    }

private:
    const xpu::sycl::md_t &diff_src_md() const { return conf_.diff_src_md; }
    const xpu::sycl::md_t &diff_dst_md() const { return conf_.diff_dst_md; }

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

    sycl_resampling_conf_t conf_;
    xpu::sycl::in_memory_arg_t diff_dst_;
    xpu::sycl::out_memory_arg_t diff_src_;
};

struct resampling_kernel_bwd_vec1_t {
    resampling_kernel_bwd_vec1_t(const sycl_resampling_conf_t &conf,
            xpu::sycl::in_memory_arg_t &diff_dst,
            xpu::sycl::out_memory_arg_t &diff_src)
        : conf_(conf), diff_dst_(diff_dst), diff_src_(diff_src) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t diff_src_mem(diff_src_, conf_.diff_src_md);
        memory_tensor_t diff_dst_mem(diff_dst_, conf_.diff_dst_md);

        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();

        const auto &diff_src_ndims = conf_.diff_src_md.ndims();
        const auto &diff_src_dims = conf_.diff_src_md.dims();
        dim_t MB = diff_src_dims[0];
        dim_t C = diff_src_dims[1];

        dim_t ID = diff_src_ndims >= 5 ? diff_src_dims[diff_src_ndims - 3] : 1;
        dim_t IH = diff_src_ndims >= 4 ? diff_src_dims[diff_src_ndims - 2] : 1;
        dim_t IW = diff_src_ndims >= 3 ? diff_src_dims[diff_src_ndims - 1] : 1;

        const auto &diff_dst_ndims = conf_.diff_dst_md.ndims();
        const auto &diff_dst_dims = conf_.diff_dst_md.dims();
        dim_t OD = diff_dst_ndims >= 5 ? diff_dst_dims[diff_dst_ndims - 3] : 1;
        dim_t OH = diff_dst_ndims >= 4 ? diff_dst_dims[diff_dst_ndims - 2] : 1;
        dim_t OW = diff_dst_ndims >= 3 ? diff_dst_dims[diff_dst_ndims - 1] : 1;

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

                float dd = diff_dst_mem.load(
                        get_offset(diff_dst_md(), mb, c, od, oh, ow));
                ds += dd * weight_d * weight_h * weight_w;
            }
            diff_src_mem.store(
                    ds, get_offset(diff_src_md(), mb, c, id, ih, iw));
            utils::nd_iterator_step(mb, MB, c, C, id, ID, ih, IH, iw, IW);
        }
    }

private:
    const xpu::sycl::md_t &diff_src_md() const { return conf_.diff_src_md; }
    const xpu::sycl::md_t &diff_dst_md() const { return conf_.diff_dst_md; }

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

    sycl_resampling_conf_t conf_;
    xpu::sycl::in_memory_arg_t diff_dst_;
    xpu::sycl::out_memory_arg_t diff_src_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
