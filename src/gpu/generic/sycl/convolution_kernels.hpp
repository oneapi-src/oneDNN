/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_SYCL_CONVOLUTION_KERNELS_HPP
#define GPU_SYCL_CONVOLUTION_KERNELS_HPP

#include "common/primitive_exec_types.hpp"
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

struct convolution_kernel_fwd_t {
    static constexpr int max_supported_ndims = 6;

    convolution_kernel_fwd_t(const sycl_convolution_fwd_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , data_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_0))
        , weights_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WEIGHTS))
        , bias_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_BIAS))
        , dst_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , data_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0))
        , weights_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS))
        , dst_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST))
        , data_zeropoints_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC_0))
        , dst_zeropoints_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST))
        , scales_data_dt_(conf_.do_scale_data
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , scales_weights_dt_(conf_.do_scale_weights
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , zeropoints_data_dt_(conf_.use_data_zeropoints
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_ZERO_POINTS
                                       | DNNL_ARG_SRC_0)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , zeropoints_dst_dt_(conf_.use_dst_zeropoints
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST)
                                    .data_type()
                          : data_type_t::dnnl_f32) {}

    void operator()(::sycl::nd_item<1> item) const {
        const float sm_data = (conf_.do_scale_data
                        ? load_float_value(scales_data_dt_, data_scale_ptr(), 0)
                        : 1.f);

        float sm_weights = (conf_.do_scale_weights && conf_.single_weight_scale
                        ? load_float_value(
                                scales_weights_dt_, weights_scale_ptr(), 0)
                        : 1.f);

        const float sm_dst = (conf_.do_scale_dst
                        ? load_float_value(data_type::f32, dst_scale_ptr(), 0)
                        : 1.f);

        dims_t data_dims, weights_dims, dst_dims, dst_strides, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            data_dims[i] = (i < data_md().ndims()) ? data_md().dims()[i] : 1;
            weights_dims[i]
                    = (i < weights_md().ndims()) ? weights_md().dims()[i] : 1;
            dst_dims[i] = (i < dst_md().ndims()) ? dst_md().dims()[i] : 1;
            dst_strides[i]
                    = (i < dst_md().ndims()) ? dst_md().strides()[i] : INT_MAX;
        }

        bool no_groups = weights_md().ndims() == data_md().ndims();

        const int SD = conf_.strides[0];
        const int SH = conf_.strides[1];
        const int SW = conf_.strides[2];

        //per group
        int OC = weights_dims[1];
        int IC = weights_dims[2];

        int KD = weights_dims[3];
        int KH = weights_dims[4];
        int KW = weights_dims[5];
        if (no_groups) {
            OC = weights_dims[0];
            IC = weights_dims[1];
            KD = weights_dims[2];
            KH = weights_dims[3];
            KW = weights_dims[4];
        }

        const int PD = conf_.padding[0];
        const int PH = conf_.padding[1];
        const int PW = conf_.padding[2];

        const int DD = conf_.dilation[0];
        const int DH = conf_.dilation[1];
        const int DW = conf_.dilation[2];

        for (int idx = item.get_global_id(0); idx < conf_.wk_size;
                idx += item.get_global_range(0)) {
            for (int i = 0; i < max_supported_ndims; i++) {
                off[i] = idx / dst_strides[i] % dst_dims[i];
            }

            const int n = off[0];
            const int oc_tot = off[1];
            const int oc = oc_tot % OC;
            const int g = oc_tot / OC;

            const int od = off[2];
            const int oh = off[3];
            const int ow = off[4];

            float accumulator = 0;
            for (int ic = 0; ic < IC; ++ic) {
                for (int kd = 0; kd < KD; ++kd) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            const int id = od * SD - PD + kd * (1 + DD);
                            const int ih = oh * SH - PH + kh * (1 + DH);
                            const int iw = ow * SW - PW + kw * (1 + DW);

                            if (id < 0 || id >= data_dims[2] || ih < 0
                                    || ih >= data_dims[3] || iw < 0
                                    || iw >= data_dims[4]) {
                                continue;
                            }

                            dims_t off_data {n, g * IC + ic, id, ih, iw};
                            const int data_idx = data_md().off_v(off_data);
                            dims_t off_weights {g, oc, ic, kd, kh, kw};
                            dims_t off_weights_no_groups {oc, ic, kd, kh, kw};
                            const int weights_idx = weights_md().off_v(no_groups
                                            ? off_weights_no_groups
                                            : off_weights);

                            auto data = load_float_value(data_md().data_type(),
                                    data_ptr(), data_idx);
                            auto weight
                                    = load_float_value(weights_md().data_type(),
                                            weights_ptr(), weights_idx);

                            if (conf_.use_data_zeropoints) {
                                int zpoint_idx = conf_.single_data_zeropoint
                                        ? 0
                                        : g * IC + ic;
                                auto data_zeropoint = load_float_value(
                                        zeropoints_data_dt_,
                                        data_zeropoint_ptr(), zpoint_idx);
                                data -= data_zeropoint;
                            }
                            accumulator += data * weight;
                        }
                    }
                }
            }
            if (conf_.do_scale_data) { accumulator *= sm_data; }
            if (conf_.do_scale_weights) {
                if (!conf_.single_weight_scale) {
                    sm_weights = load_float_value(
                            scales_weights_dt_, weights_scale_ptr(), oc_tot);
                }
                accumulator *= sm_weights;
            }

            if (conf_.has_bias) {
                auto bias = load_float_value(conf_.bias_dt, bias_ptr(), oc_tot);
                accumulator += bias;
            }

            accumulator = conf_.post_ops.apply(accumulator, dst_, idx);

            if (conf_.do_scale_dst) { accumulator /= sm_dst; }
            if (conf_.use_dst_zeropoints) {
                int zpoint_idx = conf_.single_dst_zeropoint ? 0 : oc_tot;
                auto dst_zeropoint = load_float_value(
                        zeropoints_dst_dt_, dst_zeropoint_ptr(), zpoint_idx);
                accumulator += dst_zeropoint;
            }
            store_float_value(
                    dst_md().data_type(), accumulator, dst_ptr(), idx);
        }
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.data_md; }
    const xpu::sycl::md_t &weights_md() const { return conf_.weights_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *weights_ptr() const { return weights_.get_pointer(); }
    void *bias_ptr() const { return bias_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *data_scale_ptr() const { return data_scale_.get_pointer(); }
    void *weights_scale_ptr() const { return weights_scale_.get_pointer(); }
    void *dst_scale_ptr() const { return dst_scale_.get_pointer(); }
    void *data_zeropoint_ptr() const { return data_zeropoints_.get_pointer(); }
    void *dst_zeropoint_ptr() const { return dst_zeropoints_.get_pointer(); }

    sycl_convolution_fwd_conf_t conf_;

    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::in_memory_arg_t weights_;
    xpu::sycl::in_memory_arg_t bias_;
    xpu::sycl::inout_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t data_scale_;
    xpu::sycl::in_memory_arg_t weights_scale_;
    xpu::sycl::in_memory_arg_t dst_scale_;
    xpu::sycl::in_memory_arg_t data_zeropoints_;
    xpu::sycl::in_memory_arg_t dst_zeropoints_;
    data_type_t scales_data_dt_;
    data_type_t scales_weights_dt_;
    data_type_t zeropoints_data_dt_;
    data_type_t zeropoints_dst_dt_;
};

struct convolution_kernel_bwd_data_t {
    static constexpr int max_supported_ndims = 6;

    convolution_kernel_bwd_data_t(const sycl_convolution_bwd_data_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , diff_data_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC))
        , weights_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WEIGHTS))
        , bias_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_BIAS))
        , diff_dst_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST))
        , data_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0))
        , weights_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS))
        , dst_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST))
        , data_zeropoints_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC_0))
        , dst_zeropoints_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST))
        , scales_data_dt_(conf_.do_scale_data
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , scales_weights_dt_(conf_.do_scale_weights
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , zeropoints_data_dt_(conf_.use_data_zeropoints
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_ZERO_POINTS
                                       | DNNL_ARG_SRC_0)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , zeropoints_dst_dt_(conf_.use_dst_zeropoints
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST)
                                    .data_type()
                          : data_type_t::dnnl_f32) {}

    void operator()(::sycl::nd_item<1> item) const {
        const float sm_data = (conf_.do_scale_data
                        ? load_float_value(scales_data_dt_, data_scale_ptr(), 0)
                        : 1.f);

        float sm_weights = (conf_.do_scale_weights && conf_.single_weight_scale
                        ? load_float_value(
                                scales_weights_dt_, weights_scale_ptr(), 0)
                        : 1.f);

        const float sm_dst = (conf_.do_scale_dst
                        ? load_float_value(data_type::f32, dst_scale_ptr(), 0)
                        : 1.f);

        dims_t data_dims, weights_dims, dst_dims, data_strides, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            data_dims[i] = (i < diff_data_md().ndims())
                    ? diff_data_md().dims()[i]
                    : 1;
            weights_dims[i]
                    = (i < weights_md().ndims()) ? weights_md().dims()[i] : 1;
            dst_dims[i]
                    = (i < diff_dst_md().ndims()) ? diff_dst_md().dims()[i] : 1;
            data_strides[i] = (i < diff_data_md().ndims())
                    ? diff_data_md().strides()[i]
                    : INT_MAX;
        }

        bool no_groups = weights_md().ndims() == diff_data_md().ndims();

        const int SD = ::sycl::max(conf_.strides[0], 1);
        const int SH = ::sycl::max(conf_.strides[1], 1);
        const int SW = ::sycl::max(conf_.strides[2], 1);

        //per group
        int OC = weights_dims[1];
        int IC = weights_dims[2];

        int KD = weights_dims[3];
        int KH = weights_dims[4];
        int KW = weights_dims[5];
        if (no_groups) {
            OC = weights_dims[0];
            IC = weights_dims[1];
            KD = weights_dims[2];
            KH = weights_dims[3];
            KW = weights_dims[4];
        }

        const int PD = conf_.padding[0];
        const int PH = conf_.padding[1];
        const int PW = conf_.padding[2];

        const int DD = conf_.dilation[0];
        const int DH = conf_.dilation[1];
        const int DW = conf_.dilation[2];

        for (int idx = item.get_global_id(0); idx < conf_.wk_size;
                idx += item.get_global_range(0)) {
            for (int i = 0; i < max_supported_ndims; i++) {
                off[i] = idx / data_strides[i] % data_dims[i];
            }

            const int n = off[0];
            const int ic_tot = off[1];
            const int ic = ic_tot % IC;
            const int g = ic_tot / IC;

            const int id = off[2];
            const int ih = off[3];
            const int iw = off[4];

            float accumulator = 0;
            for (int oc = 0; oc < OC; ++oc) {
                for (int kd = 0; kd < KD; ++kd) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int ow = iw - kw * (1 + DW) + PW;
                            int oh = ih - kh * (1 + DH) + PH;
                            int od = id - kd * (1 + DD) + PD;

                            if (od < 0 || oh < 0 || ow < 0) { continue; }

                            if (ow % SW != 0 || oh % SH != 0 || od % SD != 0) {
                                continue;
                            }
                            ow /= SW;
                            oh /= SH;
                            od /= SD;

                            if (od >= dst_dims[2] || oh >= dst_dims[3]
                                    || ow >= dst_dims[4]) {
                                continue;
                            }

                            dims_t off_dst {n, g * OC + oc, od, oh, ow};
                            const int dst_idx = diff_dst_md().off_v(off_dst);
                            dims_t off_weights {g, oc, ic, kd, kh, kw};
                            dims_t off_weights_no_groups {oc, ic, kd, kh, kw};
                            const int weights_idx = weights_md().off_v(no_groups
                                            ? off_weights_no_groups
                                            : off_weights);

                            auto diff_dst = load_float_value(
                                    diff_dst_md().data_type(), diff_dst_ptr(),
                                    dst_idx);
                            auto weight
                                    = load_float_value(weights_md().data_type(),
                                            weights_ptr(), weights_idx);

                            if (conf_.use_data_zeropoints) {
                                /* 
                                Zeropoints are only used when this kernel is used to implement fwd pass of deconvolution.
                                In that case diff_dst is actually data of deconvolution. So in that case data zeropoint goes with diff_dst.
                                Done to be consistent with OpenCL backend, so both can use the same deconvolution implementation.
                                */
                                int zpoint_idx = conf_.single_data_zeropoint
                                        ? 0
                                        : ic_tot;
                                auto data_zeropoint = load_float_value(
                                        zeropoints_data_dt_,
                                        data_zeropoint_ptr(), zpoint_idx);
                                diff_dst -= data_zeropoint;
                            }
                            accumulator += diff_dst * weight;
                        }
                    }
                }
            }
            if (conf_.do_scale_data) { accumulator *= sm_data; }
            if (conf_.do_scale_weights) {
                if (!conf_.single_weight_scale) {
                    sm_weights = load_float_value(
                            scales_weights_dt_, weights_scale_ptr(), ic_tot);
                }
                accumulator *= sm_weights;
            }

            if (conf_.has_bias) {
                auto bias = load_float_value(conf_.bias_dt, bias_ptr(), ic_tot);
                accumulator += bias;
            }

            accumulator = conf_.post_ops.apply(accumulator, diff_data_, idx);

            if (conf_.do_scale_dst) { accumulator /= sm_dst; }
            if (conf_.use_dst_zeropoints) {
                int zpoint_idx = conf_.single_dst_zeropoint ? 0 : g * IC + ic;
                auto dst_zeropoint = load_float_value(
                        zeropoints_dst_dt_, dst_zeropoint_ptr(), zpoint_idx);
                accumulator += dst_zeropoint;
            }
            store_float_value(diff_data_md().data_type(), accumulator,
                    diff_data_ptr(), idx);
        }
    }

private:
    const xpu::sycl::md_t &diff_data_md() const { return conf_.diff_data_md; }
    const xpu::sycl::md_t &weights_md() const { return conf_.weights_md; }
    const xpu::sycl::md_t &diff_dst_md() const { return conf_.diff_dst_md; }

    void *diff_data_ptr() const { return diff_data_.get_pointer(); }
    void *weights_ptr() const { return weights_.get_pointer(); }
    void *bias_ptr() const { return bias_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }
    void *data_scale_ptr() const { return data_scale_.get_pointer(); }
    void *weights_scale_ptr() const { return weights_scale_.get_pointer(); }
    void *dst_scale_ptr() const { return dst_scale_.get_pointer(); }
    void *data_zeropoint_ptr() const { return data_zeropoints_.get_pointer(); }
    void *dst_zeropoint_ptr() const { return dst_zeropoints_.get_pointer(); }

    sycl_convolution_bwd_data_conf_t conf_;

    xpu::sycl::inout_memory_arg_t diff_data_;
    xpu::sycl::in_memory_arg_t weights_;
    xpu::sycl::in_memory_arg_t bias_;
    xpu::sycl::in_memory_arg_t diff_dst_;
    xpu::sycl::in_memory_arg_t data_scale_;
    xpu::sycl::in_memory_arg_t weights_scale_;
    xpu::sycl::in_memory_arg_t dst_scale_;
    xpu::sycl::in_memory_arg_t data_zeropoints_;
    xpu::sycl::in_memory_arg_t dst_zeropoints_;
    data_type_t scales_data_dt_;
    data_type_t scales_weights_dt_;
    data_type_t zeropoints_data_dt_;
    data_type_t zeropoints_dst_dt_;
};

struct convolution_kernel_bwd_weights_t {
    static constexpr int max_supported_ndims = 6;

    convolution_kernel_bwd_weights_t(
            const sycl_convolution_bwd_weights_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx, int data_arg,
            int diff_dst_arg)
        : conf_(conf)
        , data_(CTX_IN_SYCL_KERNEL_MEMORY(data_arg))
        , diff_weights_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_WEIGHTS))
        , diff_bias_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_BIAS))
        , diff_dst_(CTX_IN_SYCL_KERNEL_MEMORY(diff_dst_arg)) {}

    void operator()(::sycl::nd_item<1> item) const {
        dims_t data_dims, weights_dims, dst_dims, weights_strides, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            data_dims[i] = (i < data_md().ndims()) ? data_md().dims()[i] : 1;
            weights_dims[i] = (i < diff_weights_md().ndims())
                    ? diff_weights_md().dims()[i]
                    : 1;
            dst_dims[i]
                    = (i < diff_dst_md().ndims()) ? diff_dst_md().dims()[i] : 1;
            weights_strides[i] = (i < diff_weights_md().ndims())
                    ? diff_weights_md().strides()[i]
                    : INT_MAX;
        }

        bool no_groups = diff_weights_md().ndims() == data_md().ndims();

        const int SD = ::sycl::max(conf_.strides[0], 1);
        const int SH = ::sycl::max(conf_.strides[1], 1);
        const int SW = ::sycl::max(conf_.strides[2], 1);

        //per group
        int OC = weights_dims[1];
        int IC = weights_dims[2];
        if (no_groups) {
            OC = weights_dims[0];
            IC = weights_dims[1];
        }

        int MB = data_dims[0];
        int ID = data_dims[2];
        int IH = data_dims[3];
        int IW = data_dims[4];
        int OD = dst_dims[2];
        int OH = dst_dims[3];
        int OW = dst_dims[4];

        const int PD = conf_.padding[0];
        const int PH = conf_.padding[1];
        const int PW = conf_.padding[2];

        const int DD = conf_.dilation[0];
        const int DH = conf_.dilation[1];
        const int DW = conf_.dilation[2];

        for (int idx = item.get_global_id(0); idx < conf_.wk_size;
                idx += item.get_global_range(0)) {
            for (int i = 0; i < max_supported_ndims; i++) {
                off[i] = idx / weights_strides[i] % weights_dims[i];
            }

            int g = off[0];
            int oc = off[1];
            int ic = off[2];
            int kd = off[3];
            int kh = off[4];
            int kw = off[5];
            if (no_groups) {
                g = 0;
                oc = off[0];
                ic = off[1];
                kd = off[2];
                kh = off[3];
                kw = off[4];
            }

            auto bias_backprop_lambda = [&](int D, int H, int W, int OC, int ic,
                                                int oc, void *diff_ptr,
                                                xpu::sycl::md_t diff_md) {
                if (ic == 0 && kh == 0 && kw == 0 & kd == 0) {
                    float accumulator_bias = 0;
                    for (int n = 0; n < MB; ++n) {
                        for (int od = 0; od < D; ++od) {
                            for (int oh = 0; oh < H; ++oh) {
                                for (int ow = 0; ow < W; ++ow) {
                                    dims_t off_dst {n, g * OC + oc, od, oh, ow};
                                    const int dst_idx = diff_md.off_v(off_dst);
                                    auto diff_dst = load_float_value(
                                            diff_md.data_type(), diff_ptr,
                                            dst_idx);
                                    accumulator_bias += diff_dst;
                                }
                            }
                        }
                    }
                    store_float_value(conf_.bias_dt, accumulator_bias,
                            diff_bias_ptr(), g * OC + oc);
                }
            };
            if (conf_.is_deconvolution) {
                bias_backprop_lambda(
                        ID, IH, IW, IC, oc, ic, data_ptr(), data_md());
            } else {
                bias_backprop_lambda(
                        OD, OH, OW, OC, ic, oc, diff_dst_ptr(), diff_dst_md());
            }

            float accumulator_weights = 0;
            for (int n = 0; n < MB; ++n) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            int id = od * SD - PD + kd * (1 + DD);
                            int ih = oh * SH - PH + kh * (1 + DH);
                            int iw = ow * SW - PW + kw * (1 + DW);

                            if (id >= ID || ih >= IH || iw >= IW || id < 0
                                    || ih < 0 || iw < 0) {
                                continue;
                            }

                            dims_t off_dst {n, g * OC + oc, od, oh, ow};
                            const int dst_idx = diff_dst_md().off_v(off_dst);
                            dims_t off_data {n, g * IC + ic, id, ih, iw};
                            const int data_idx = data_md().off_v(off_data);

                            auto diff_dst = load_float_value(
                                    diff_dst_md().data_type(), diff_dst_ptr(),
                                    dst_idx);
                            auto data = load_float_value(data_md().data_type(),
                                    data_ptr(), data_idx);

                            accumulator_weights += diff_dst * data;
                        }
                    }
                }
            }
            store_float_value(diff_weights_md().data_type(),
                    accumulator_weights, diff_weights_ptr(), idx);
        }
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.data_md; }
    const xpu::sycl::md_t &diff_weights_md() const {
        return conf_.diff_weights_md;
    }
    const xpu::sycl::md_t &diff_dst_md() const { return conf_.diff_dst_md; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *diff_weights_ptr() const { return diff_weights_.get_pointer(); }
    void *diff_bias_ptr() const { return diff_bias_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }

    sycl_convolution_bwd_weights_conf_t conf_;

    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::out_memory_arg_t diff_weights_;
    xpu::sycl::out_memory_arg_t diff_bias_;
    xpu::sycl::in_memory_arg_t diff_dst_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
