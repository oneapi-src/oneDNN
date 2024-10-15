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

#ifndef GPU_GENERIC_SYCL_ELTWISE_KERNELS_HPP
#define GPU_GENERIC_SYCL_ELTWISE_KERNELS_HPP

#include "common/primitive_exec_types.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct eltwise_fwd_kernel_vec_t {
    eltwise_fwd_kernel_vec_t(const sycl_eltwise_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , po_args_(cgh, ctx, conf_.post_ops)
        , dst_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST)) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t src_mem(src_, conf_.src_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);

        auto operation = [&](dim_t &idx, dim_t &n, dim_t &c, dim_t &d, dim_t &h,
                                 dim_t &w) {
            dim_t src_offset = data_offset(src_mem.md(), n, c, d, h, w);
            auto src = src_mem.load(src_offset);

            float acc = compute_alg_n(
                    src, conf_.alpha, conf_.beta, conf_.alg_kind);

            dims_t po_off {n, c, d, h, w};
            switch (src_mem.md().ndims()) {
                case 3: po_off[2] = w; break;
                case 4:
                    po_off[2] = h;
                    po_off[3] = w;
                    break;
            }
            acc = conf_.post_ops.apply(acc, dst_, src_offset, po_args_, po_off);

            dst_mem.store(acc, src_offset);
        };

        for (dim_t idx = item.get_global_id(0); idx < conf_.wk_size;
                idx += item.get_global_range(0)) {
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

private:
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

    inline dim_t data_offset(const xpu::sycl::md_t &mem, dim_t &n, dim_t &c,
            dim_t &d, dim_t &h, dim_t &w) const {
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

    sycl_eltwise_conf_t conf_;
    xpu::sycl::in_memory_arg_t src_;
    post_op_input_args po_args_;
    xpu::sycl::inout_memory_arg_t dst_;
};

struct eltwise_bwd_kernel_vec_t {
    eltwise_bwd_kernel_vec_t(const sycl_eltwise_conf_t &conf,
            ::sycl::handler &cgh, const exec_ctx_t &ctx, bool use_dst)
        : conf_(conf)
        , src_(use_dst ? CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DST)
                       : CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , diff_src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST))
        , diff_dst_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC)) {}

    void operator()(::sycl::nd_item<1> item) const {
        memory_tensor_t src_mem(src_, conf_.src_md);
        memory_tensor_t diff_src_mem(diff_src_, conf_.diff_src_md);
        memory_tensor_t diff_dst_mem(diff_dst_, conf_.diff_dst_md);

        for (dim_t idx = item.get_global_id(0); idx < conf_.wk_size;
                idx += item.get_global_range(0)) {
            auto diff_src = diff_src_mem.load(idx);
            auto src = src_mem.load(idx);

            auto dst = compute_alg_n(
                    diff_src, src, conf_.alpha, conf_.beta, conf_.alg_kind);
            diff_dst_mem.store(dst, idx);
        }
    }

private:
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
    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::in_memory_arg_t diff_src_;
    xpu::sycl::out_memory_arg_t diff_dst_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
