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

#ifndef GPU_SYCL_LRN_KERNELS_HPP
#define GPU_SYCL_LRN_KERNELS_HPP

#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct lrn_fwd_kernel_vec_t {
    lrn_fwd_kernel_vec_t(const sycl_lrn_conf_t &conf, sycl_in_memory_arg_t &src,
            sycl_out_memory_arg_t &dst, const format_tag_t &tag)
        : conf_(conf), src_(src), dst_(dst), tag_(tag) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;
        size_t base_idx = offset_t * conf_.block_size;

        auto data_off = [=](dim_t &mb, dim_t &c, dim_t &d, dim_t &h, dim_t &w) {
            switch (tag_) {
                case format_tag::nchw:
                    return mb * conf_.stride_mb + c * conf_.h * conf_.w
                            + h * conf_.w + w;
                case format_tag::nhwc:
                    return mb * conf_.stride_mb + h * conf_.w * conf_.c
                            + w * conf_.c + c;
                default:
                    if (conf_.ndims >= 5) return src_md().off(mb, c, d, h, w);
                    if (conf_.ndims >= 4) return src_md().off(mb, c, h, w);
                    if (conf_.ndims >= 3) return src_md().off(mb, c, w);
                    return src_md().off(mb, c);
            }
        };

        auto ker = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
            float sum = 0;
            const dim_t half_size = (conf_.size - 1) / 2;
            if (conf_.alg_kind == alg_kind::lrn_across_channels) {
                const dim_t c_st = nstl::max(oc - half_size + 0, (dim_t)0);
                const dim_t c_en = nstl::min(oc + half_size + 1, conf_.c);

                for (dim_t c = c_st; c < c_en; ++c) {
                    const auto s_off = data_off(mb, c, od, oh, ow);
                    const auto s = load_float_value(
                            src_md().data_type(), src_ptr(), s_off);
                    sum += s * s;
                }
            } else {
                dim_t d_st = nstl::max(od - half_size + 0, (dim_t)0);
                dim_t d_en = nstl::min(od + half_size + 1, conf_.d);
                dim_t h_st = nstl::max(oh - half_size + 0, (dim_t)0);
                dim_t h_en = nstl::min(oh + half_size + 1, conf_.h);
                dim_t w_st = nstl::max(ow - half_size + 0, (dim_t)0);
                dim_t w_en = nstl::min(ow + half_size + 1, conf_.w);
                for_(dim_t d = d_st; d < d_en; ++d)
                for_(dim_t h = h_st; h < h_en; ++h)
                for (dim_t w = w_st; w < w_en; ++w) {
                    const auto s_off = data_off(mb, oc, d, h, w);
                    const auto s = load_float_value(
                            src_md().data_type(), src_ptr(), s_off);
                    sum += s * s;
                }
            }
            sum = conf_.k + conf_.alpha * sum / conf_.compute_n_summands;
            const auto s_off = data_off(mb, oc, od, oh, ow);
            const auto s
                    = load_float_value(src_md().data_type(), src_ptr(), s_off);
            return (s * fast_negative_powf(sum, conf_.beta));
        };

        auto operation = [=](dim_t &mb, dim_t &c, dim_t &d, dim_t &h,
                                 dim_t &w) {
            if (format_tag::nhwc == tag_) {
                const dim_t off = mb * conf_.stride_mb + h * conf_.w * conf_.c
                        + w * conf_.c + c;
                auto val = ker(mb, c, 0, h, w);
                store_float_value(dst_md().data_type(), val, dst_ptr(), off);
            } else {
                const dim_t off = data_off(mb, c, d, h, w);
                auto val = ker(mb, c, d, h, w);
                store_float_value(dst_md().data_type(), val, dst_ptr(), off);
            }
        };

        for (dim_t blk_idx = 0; blk_idx < conf_.block_size; blk_idx++) {
            dim_t idx = base_idx + blk_idx;
            if (idx < conf_.wk_size) {
                dim_t N = conf_.mb;
                dim_t C = conf_.c;
                dim_t D = conf_.d;
                dim_t H = conf_.h;
                dim_t W = conf_.w;

                dim_t n = (idx / (C * D * H * W)) % N;
                dim_t c = (idx / (D * H * W)) % C;
                dim_t d = (idx / (H * W)) % D;
                dim_t h = (idx / (W)) % H;
                dim_t w = (idx / (1)) % W;

                operation(n, c, d, h, w);
            }
        }
    }

    inline float fast_negative_powf(float omega, float beta) const {
        float Y;
        if (beta == 0.75f) {
            Y = ::sycl::sqrt(1.0f / (::sycl::sqrt(omega) * omega));
        } else {
            Y = 1.0f / ::sycl::pow(omega, beta);
        }
        return Y;
    }

private:
    const sycl_md_t &src_md() const { return conf_.src_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    sycl_lrn_conf_t conf_;
    sycl_in_memory_arg_t src_;
    sycl_out_memory_arg_t dst_;
    format_tag_t tag_;
};

struct lrn_bwd_kernel_vec_t {
    lrn_bwd_kernel_vec_t(const sycl_lrn_conf_t &conf, sycl_in_memory_arg_t &src,
            sycl_in_memory_arg_t &diff_dst, sycl_out_memory_arg_t &diff_src,
            const format_tag_t &tag)
        : conf_(conf)
        , src_(src)
        , diff_dst_(diff_dst)
        , diff_src_(diff_src)
        , tag_(tag) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;
        size_t base_idx = offset_t * conf_.block_size;

        auto data_off = [&](dim_t &mb, dim_t &c, dim_t &d, dim_t &h,
                                dim_t &w) -> dim_t {
            switch (tag_) {
                case format_tag::nchw:
                    return mb * conf_.stride_mb + c * conf_.h * conf_.w
                            + h * conf_.w + w;
                case format_tag::nhwc:
                    return mb * conf_.stride_mb + h * conf_.w * conf_.c
                            + w * conf_.c + c;
                default:
                    if (conf_.ndims >= 5) return src_md().off(mb, c, d, h, w);
                    if (conf_.ndims >= 4) return src_md().off(mb, c, h, w);
                    if (conf_.ndims >= 3) return src_md().off(mb, c, w);
                    return src_md().off(mb, c);
            }
        };

        auto get_omega = [=](dim_t &mb, dim_t &oc, dim_t &od, dim_t &oh,
                                 dim_t &ow) {
            auto sum = 0;
            const dim_t half_size = (conf_.size - 1) / 2;
            if (conf_.alg_kind == alg_kind::lrn_across_channels) {
                const dim_t c_st = nstl::max(oc - half_size + 0, (dim_t)0);
                const dim_t c_en = nstl::min(oc + half_size + 1, conf_.c);

                for (dim_t c = c_st; c < c_en; ++c) {
                    const auto s_off = data_off(mb, c, od, oh, ow);
                    const auto s = load_float_value(
                            src_md().data_type(), src_ptr(), s_off);
                    sum += s * s;
                }
            } else {
                dim_t d_st = nstl::max(od - half_size + 0, (dim_t)0);
                dim_t d_en = nstl::min(od + half_size + 1, conf_.d);
                dim_t h_st = nstl::max(oh - half_size + 0, (dim_t)0);
                dim_t h_en = nstl::min(oh + half_size + 1, conf_.h);
                dim_t w_st = nstl::max(ow - half_size + 0, (dim_t)0);
                dim_t w_en = nstl::min(ow + half_size + 1, conf_.w);
                for_(dim_t d = d_st; d < d_en; ++d)
                for_(dim_t h = h_st; h < h_en; ++h)
                for (dim_t w = w_st; w < w_en; ++w) {
                    const auto s_off = data_off(mb, oc, d, h, w);
                    const auto s = load_float_value(
                            src_md().data_type(), src_ptr(), s_off);
                    sum += s * s;
                }
            }
            return (conf_.k + conf_.alpha * sum / conf_.compute_n_summands);
        };

        auto ker = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
            float A = 0, B = 0;
            const dim_t half_size = (conf_.size - 1) / 2;
            if (conf_.alg_kind == alg_kind::lrn_across_channels) {
                const dim_t c_st = nstl::max(oc - half_size + 0, (dim_t)0);
                const dim_t c_en = nstl::min(oc + half_size + 1, conf_.c);

                for (dim_t c = c_st; c < c_en; c++) {
                    const auto off = data_off(mb, c, od, oh, ow);
                    const auto omega = get_omega(mb, c, od, oh, ow);
                    const auto omega_in_beta
                            = fast_negative_powf(omega, conf_.beta);

                    const auto dst_val = load_float_value(
                            diff_dst_md().data_type(), diff_dst_ptr(), off);
                    const auto tmp = omega_in_beta * dst_val;
                    if (c == oc) A = tmp;
                    const auto src_val = load_float_value(
                            src_md().data_type(), src_ptr(), off);
                    B += (src_val * tmp / omega);
                }
            } else {
                dim_t d_st = nstl::max(od - half_size + 0, (dim_t)0);
                dim_t d_en = nstl::min(od + half_size + 1, conf_.d);
                dim_t h_st = nstl::max(oh - half_size + 0, (dim_t)0);
                dim_t h_en = nstl::min(oh + half_size + 1, conf_.h);
                dim_t w_st = nstl::max(ow - half_size + 0, (dim_t)0);
                dim_t w_en = nstl::min(ow + half_size + 1, conf_.w);
                for_(dim_t d = d_st; d < d_en; ++d)
                for_(dim_t h = h_st; h < h_en; ++h)
                for (dim_t w = w_st; w < w_en; ++w) {
                    const auto off = data_off(mb, oc, d, h, w);
                    const auto omega = get_omega(mb, oc, d, h, w);
                    const auto omega_in_beta
                            = fast_negative_powf(omega, conf_.beta);

                    const auto dst_val = load_float_value(
                            diff_dst_md().data_type(), diff_dst_ptr(), off);
                    const auto tmp = omega_in_beta * dst_val;
                    if (d == od && h == oh && w == ow) A = tmp;
                    const auto src_val = load_float_value(
                            src_md().data_type(), src_ptr(), off);
                    B += (src_val * tmp / omega);
                }
            }
            const auto off = data_off(mb, oc, od, oh, ow);
            const auto src_val
                    = load_float_value(src_md().data_type(), src_ptr(), off);
            B *= (2.0f * conf_.alpha * conf_.beta * src_val
                    / conf_.compute_n_summands);
            return (A - B);
        };

        auto operation
                = [=](dim_t &mb, dim_t &c, dim_t &d, dim_t &h, dim_t &w) {
                      if (format_tag::nhwc == tag_) {
                          const dim_t off = mb * conf_.stride_mb
                                  + h * conf_.w * conf_.c + w * conf_.c + c;
                          auto val = ker(mb, c, 0, h, w);
                          store_float_value(diff_src_md().data_type(), val,
                                  diff_src_ptr(), off);
                      } else {
                          const dim_t off = data_off(mb, c, d, h, w);
                          auto val = ker(mb, c, d, h, w);
                          store_float_value(diff_src_md().data_type(), val,
                                  diff_src_ptr(), off);
                      }
                  };

        for (dim_t blk_idx = 0; blk_idx < conf_.block_size; blk_idx++) {
            dim_t idx = base_idx + blk_idx;
            if (idx < conf_.wk_size) {
                dim_t N = conf_.mb;
                dim_t C = conf_.c;
                dim_t D = conf_.d;
                dim_t H = conf_.h;
                dim_t W = conf_.w;

                dim_t n = (idx / (C * D * H * W)) % N;
                dim_t c = (idx / (D * H * W)) % C;
                dim_t d = (idx / (H * W)) % D;
                dim_t h = (idx / (W)) % H;
                dim_t w = (idx / (1)) % W;

                operation(n, c, d, h, w);
            }
        }
    }

    inline float fast_negative_powf(float omega, float beta) const {
        float Y;
        if (beta == 0.75f) {
            Y = ::sycl::sqrt(1.0f / (::sycl::sqrt(omega) * omega));
        } else {
            Y = 1.0f / ::sycl::pow(omega, beta);
        }
        return Y;
    }

private:
    const sycl_md_t &src_md() const { return conf_.src_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }
    const sycl_md_t &diff_src_md() const { return conf_.diff_src_md; }

    void *src_ptr() const { return src_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }
    void *diff_src_ptr() const { return diff_src_.get_pointer(); }

    sycl_lrn_conf_t conf_;
    sycl_in_memory_arg_t src_;
    sycl_in_memory_arg_t diff_dst_;
    sycl_out_memory_arg_t diff_src_;
    format_tag_t tag_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
