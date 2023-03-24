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

#ifndef GPU_SYCL_ELTWISE_KERNELS_HPP
#define GPU_SYCL_ELTWISE_KERNELS_HPP

#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_math_utils.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct eltwise_fwd_kernel_vec_t {
    eltwise_fwd_kernel_vec_t(const sycl_eltwise_conf_t &conf,
            sycl_in_memory_arg_t &src, sycl_out_memory_arg_t &dst,
            sycl_in_memory_arg_t &srcOp1, sycl_in_memory_arg_t &srcOp2,
            sycl_in_memory_arg_t &srcOp3, sycl_in_memory_arg_t &srcOp4,
            sycl_in_memory_arg_t &srcOp5)
        : conf_(conf)
        , src_(src)
        , srcOp1_(srcOp1)
        , srcOp2_(srcOp2)
        , srcOp3_(srcOp3)
        , srcOp4_(srcOp4)
        , srcOp5_(srcOp5)
        , dst_(dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;

        size_t base_idx = offset_t * conf_.block_size;

        auto operation = [=](dim_t &idx, dim_t &n, dim_t &c, dim_t &d, dim_t &h,
                                 dim_t &w) {
            dim_t src_offset = data_offset(src_md(), n, c, d, h, w);

            auto src = load_float_value(
                    src_md().data_type(), src_ptr(), src_offset);
            auto dst = load_float_value(
                    dst_md().data_type(), dst_ptr(), src_offset);

            dim_t data_l_off = (((n * conf_.c + c) * conf_.d + d) * conf_.h + h)
                            * conf_.w
                    + w;

            ::sycl::vec<float, 16> post_po_sr = post_op_src_val(data_l_off);

            dst = compute_alg_n(src, conf_.alpha, conf_.beta, conf_.alg_kind);
            dst = conf_.post_ops.apply(dst, post_po_sr);
            store_float_value(dst_md().data_type(), dst, dst_ptr(), src_offset);
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
                operation(idx, n, c, d, h, w);
            }
        }
    }

private:
    const sycl_md_t &src_md() const { return conf_.src_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    float compute_alg_n(const float &s, const float &alpha, const float &beta,
            const alg_kind_t &alg) const {
        switch (alg) {

            case alg_kind::eltwise_abs:
                return (float)(dnnl::impl::math::abs_fwd((float)s));

            case alg_kind::eltwise_clip:
                return (float)(dnnl::impl::math::clip_fwd(
                        (float)s, alpha, beta));

            case alg_kind::eltwise_clip_v2:
                return (float)(dnnl::impl::math::clip_v2_fwd(
                        (float)s, alpha, beta));

            case alg_kind::eltwise_elu:
                return (float)(math::elu_fwd((float)s, alpha));

            case alg_kind::eltwise_exp: return (float)(math::exp_fwd((float)s));

            case alg_kind::eltwise_gelu_erf:
                return (float)(math::gelu_erf_fwd(s));

            case alg_kind::eltwise_gelu_tanh:
                return (float)(math::gelu_tanh_fwd((float)s));

            case alg_kind::eltwise_hardsigmoid:
                return (float)(dnnl::impl::math::hardsigmoid_fwd(
                        (float)s, alpha, beta));

            case alg_kind::eltwise_hardswish:
                return (float)(dnnl::impl::math::hardswish_fwd(
                        (float)s, alpha, beta));

            case alg_kind::eltwise_linear:
                return (float)(dnnl::impl::math::linear_fwd(s, alpha, beta));

            case alg_kind::eltwise_log: return (float)(math::log_fwd((float)s));

            case alg_kind::eltwise_logistic:
                return (float)(math::logistic_fwd((float)s));

            case alg_kind::eltwise_mish:
                return (float)(math::mish_fwd((float)s));

            case alg_kind::eltwise_pow:
                return (float)(math::pow_fwd((float)s, alpha, beta));

            case alg_kind::eltwise_relu:
                return (float)(math::relu_fwd((float)s, alpha));

            case alg_kind::eltwise_round:
                return (float)(dnnl::impl::math::round_fwd((float)s));

            case alg_kind::eltwise_soft_relu:
                return (float)(math::soft_relu_fwd((float)s, alpha));

            case alg_kind::eltwise_sqrt:
                return (float)(math::sqrt_fwd((float)s));

            case alg_kind::eltwise_square:
                return (float)(dnnl::impl::math::square_fwd((float)s));

            case alg_kind::eltwise_swish:
                return (float)(math::swish_fwd(s, alpha));

            case alg_kind::eltwise_tanh:
                return (float)(math::tanh_fwd((float)s));

            case alg_kind::eltwise_clip_v2_use_dst_for_bwd:
                return (float)(dnnl::impl::math::clip_v2_fwd(
                        (float)s, alpha, beta));

            case alg_kind::eltwise_elu_use_dst_for_bwd:
                return (float)(math::elu_fwd((float)s, alpha));

            case alg_kind::eltwise_exp_use_dst_for_bwd:
                return (float)(math::exp_fwd((float)s));

            case alg_kind::eltwise_logistic_use_dst_for_bwd:
                return (float)(math::logistic_fwd((float)s));

            case alg_kind::eltwise_relu_use_dst_for_bwd:
                return (float)(math::relu_fwd((float)s, alpha));

            case alg_kind::eltwise_sqrt_use_dst_for_bwd:
                return (float)(math::sqrt_fwd((float)s));

            case alg_kind::eltwise_tanh_use_dst_for_bwd:
                return (float)(math::tanh_fwd((float)s));

            default: return (float)(NAN);
        }
    }

    inline ::sycl::vec<float, 16> post_op_src_val(dim_t &data_l_off) const {
        ::sycl::vec<float, 16> post_po_sr;
        const auto maxPostPo = conf_.post_po_len;

        for (dim_t po_idx = 0; po_idx < maxPostPo; po_idx++) {
            float res = 0.0f;
            if (po_idx == 0)
                res = get_post_op_val(srcOp1_, po_idx, data_l_off);
            else if (po_idx == 1)
                res = get_post_op_val(srcOp2_, po_idx, data_l_off);
            else if (po_idx == 2)
                res = get_post_op_val(srcOp3_, po_idx, data_l_off);
            else if (po_idx == 3)
                res = get_post_op_val(srcOp4_, po_idx, data_l_off);
            else if (po_idx == 4)
                res = get_post_op_val(srcOp5_, po_idx, data_l_off);

            post_po_sr[po_idx] = res;
        }
        return post_po_sr;
    }

    inline dim_t data_offset(const sycl_md_t &mem, dim_t &n, dim_t &c, dim_t &d,
            dim_t &h, dim_t &w) const {
        const auto ndims = mem.ndims();
        switch (ndims) {
            case 1: return mem.off(n);
            case 2: return mem.off(n, c);
            case 3: return mem.off(n, c, w);
            case 4: return mem.off(n, c, h, w);
            case 5: return mem.off(n, c, d, h, w);
            default: return -1;
        }
        return -1;
    }

    float get_post_op_val(const sycl_in_memory_arg_t &bin_src_op, dim_t &idx,
            dim_t &offset) const {
        auto src1_desc = conf_.binary_src_arr[idx];

        const auto off = get_binary_src1_off(
                src1_desc, offset, dst_md().dims(), dst_md().ndims());

        auto dst = load_float_value(
                src1_desc.data_type(), bin_src_op.get_pointer(), off);
        return dst;
    }

    dim_t get_binary_src1_off(const sycl_md_t &src1_md, const dim_t &l_offset,
            const sycl_md_t::dims32_t &dst_dims,
            const sycl_md_t::dim32_t &dst_ndims) const {
        const dim_t mask_binary_po
                = get_dims_mask(dst_dims, src1_md.dims(), dst_ndims);
        return get_po_tensor_off(
                src1_md, l_offset, dst_dims, dst_ndims, mask_binary_po);
    }

    inline dim_t get_dims_mask(const sycl_md_t::dims32_t &dims1,
            const sycl_md_t::dims32_t &dims2, const dim_t &ndims,
            bool skip_dim_of_one = false) const {
        dim_t mask = 0;
        for (dim_t d = 0; d < ndims; ++d) {
            // Disable mask_bit for dimensions of `1` by request.
            dim_t mask_bit = skip_dim_of_one && dims1[d] == 1 ? 0 : (1 << d);
            mask += dims1[d] == dims2[d] ? mask_bit : 0;
        }
        return mask;
    }

    inline dim_t get_po_tensor_off(const sycl_md_t &tensor_md,
            const dim_t &l_offset, const sycl_md_t::dims32_t &dst_dims,
            const dim_t &dst_ndims, const dim_t &mask) const {
        dims_t l_dims_po {};
        get_l_dims_po(l_dims_po, l_offset, dst_dims, dst_ndims, mask);

        return tensor_md.off_v(l_dims_po);
    }

    inline void get_l_dims_po(dims_t l_dims_po, dim_t l_offset,
            const sycl_md_t::dims32_t &dst_dims, const dim_t &dst_ndims,
            const dim_t &mask) const {

        l_dims_by_l_offset(l_dims_po, l_offset, dst_dims, dst_ndims);
        utils::apply_mask_on_dims(l_dims_po, dst_ndims, mask);
    }

    inline void l_dims_by_l_offset(dims_t dims_pos, dim_t l_offset,
            const sycl_md_t::dims32_t &dims, const dim_t &ndims) const {
        for (dim_t rd = 0; rd < ndims; ++rd) {
            const dim_t d = ndims - 1 - rd;
            /* switch to faster 32-bit division when possible. */
            if (l_offset <= INT32_MAX && dims[d] <= INT32_MAX) {
                dims_pos[d] = (int32_t)l_offset % (int32_t)dims[d];
                l_offset = (int32_t)l_offset / (int32_t)dims[d];
            } else {
                dims_pos[d] = l_offset % dims[d];
                l_offset /= dims[d];
            }
        }
    }

    sycl_eltwise_conf_t conf_;
    sycl_in_memory_arg_t src_;
    sycl_in_memory_arg_t srcOp1_;
    sycl_in_memory_arg_t srcOp2_;
    sycl_in_memory_arg_t srcOp3_;
    sycl_in_memory_arg_t srcOp4_;
    sycl_in_memory_arg_t srcOp5_;
    sycl_out_memory_arg_t dst_;
};

struct eltwise_bwd_kernel_vec_t {
    eltwise_bwd_kernel_vec_t(const sycl_eltwise_conf_t &conf,
            sycl_in_memory_arg_t &diff_src, sycl_in_memory_arg_t &src,
            sycl_out_memory_arg_t &diff_dst)
        : conf_(conf), src_(src), diff_src_(diff_src), diff_dst_(diff_dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;
        size_t base_idx = offset_t * conf_.block_size;

        for (dim_t i = 0; i < conf_.block_size; i++) {
            dim_t idx = base_idx + i;
            if (idx < conf_.wk_size) {
                auto diff_src = load_float_value(
                        diff_src_md().data_type(), diff_src_ptr(), idx);
                auto src = load_float_value(
                        src_md().data_type(), src_ptr(), idx);

                auto dst = compute_alg_n(
                        diff_src, src, conf_.alpha, conf_.beta, conf_.alg_kind);
                store_float_value(
                        diff_dst_md().data_type(), dst, diff_dst_ptr(), idx);
            }
        }
    }

private:
    const sycl_md_t &src_md() const { return conf_.src_md; }
    const sycl_md_t &diff_src_md() const { return conf_.diff_src_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }

    void *src_ptr() const { return src_.get_pointer(); }
    void *diff_src_ptr() const { return diff_src_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }

    inline float compute_alg_n(const float &dd, const float &s,
            const float &alpha, const float &beta,
            const alg_kind_t &alg) const {
        switch (alg) {
            case alg_kind::eltwise_abs:
                return (float)(math::abs_bwd((float)dd, (float)s));

            case alg_kind::eltwise_clip:
                return (float)(dnnl::impl::math::clip_bwd(
                        (float)dd, (float)s, alpha, beta));

            case alg_kind::eltwise_clip_v2:
                return (float)(dnnl::impl::math::clip_v2_bwd(
                        (float)dd, (float)s, (float)alpha, (float)beta));

            case alg_kind::eltwise_clip_v2_use_dst_for_bwd:
                return (float)(dnnl::impl::math::clip_v2_bwd_use_dst(
                        (float)dd, (float)s, (float)alpha, (float)beta));

            case alg_kind::eltwise_elu:
                return (float)(math::elu_bwd((float)dd, (float)s, alpha));

            case alg_kind::eltwise_elu_use_dst_for_bwd:
                return (float)(math::elu_bwd_use_dst(
                        (float)dd, (float)s, alpha));

            case alg_kind::eltwise_exp:
                return (float)(math::exp_bwd((float)dd, (float)s));

            case alg_kind::eltwise_exp_use_dst_for_bwd:
                return (float)(dnnl::impl::math::exp_bwd_use_dst(
                        (float)dd, (float)s));

            case alg_kind::eltwise_gelu_erf:
                return (float)(math::gelu_erf_bwd((float)dd, (float)s));

            case alg_kind::eltwise_gelu_tanh:
                return (float)(math::gelu_tanh_bwd((float)dd, (float)s));

            case alg_kind::eltwise_hardsigmoid:
                return (float)(math::hardsigmoid_bwd(
                        (float)dd, (float)s, alpha, beta));

            case alg_kind::eltwise_hardswish:
                return (float)(dnnl::impl::math::hardswish_bwd(
                        (float)dd, (float)s, alpha, beta));

            case alg_kind::eltwise_linear:
                return (float)(dnnl::impl::math::linear_bwd(
                        (float)dd, (float)s, alpha, beta));

            case alg_kind::eltwise_log:
                return (float)(math::log_bwd((float)dd, (float)s));

            case alg_kind::eltwise_logistic:
                return (float)(math::logistic_bwd((float)dd, (float)s));

            case alg_kind::eltwise_logistic_use_dst_for_bwd:
                return (float)(dnnl::impl::math::logistic_bwd_use_dst(
                        (float)dd, (float)s));

            case alg_kind::eltwise_mish:
                return (float)(math::mish_bwd((float)dd, (float)s));

            case alg_kind::eltwise_pow:
                return (float)(math::pow_bwd((float)dd, (float)s, alpha, beta));

            case alg_kind::eltwise_relu:
                return (float)(dnnl::impl::math::relu_bwd(
                        (float)dd, (float)s, alpha));

            case alg_kind::eltwise_relu_use_dst_for_bwd:
                return (float)(math::relu_bwd_use_dst(
                        (float)dd, (float)s, alpha));

            case alg_kind::eltwise_soft_relu:
                return (float)(math::soft_relu_bwd((float)dd, (float)s, alpha));

            case alg_kind::eltwise_sqrt:
                return (float)(dnnl::impl::math::sqrt_bwd((float)dd, (float)s));

            case alg_kind::eltwise_sqrt_use_dst_for_bwd:
                return (float)(dnnl::impl::math::sqrt_bwd_use_dst(
                        (float)dd, (float)s));

            case alg_kind::eltwise_square:
                return (float)(dnnl::impl::math::square_bwd(
                        (float)dd, (float)s));

            case alg_kind::eltwise_swish:
                return (float)(math::swish_bwd(dd, s, alpha));

            case alg_kind::eltwise_tanh:
                return (float)(math::tanh_bwd((float)dd, (float)s));

            case alg_kind::eltwise_tanh_use_dst_for_bwd:
                return (float)(math::tanh_bwd_use_dst((float)dd, (float)s));

            default: return (float)(NAN);
        }
    }

    sycl_eltwise_conf_t conf_;
    sycl_in_memory_arg_t src_;
    sycl_in_memory_arg_t diff_src_;
    sycl_out_memory_arg_t diff_dst_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
