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

#ifndef GPU_GENERIC_SYCL_BATCH_NORMALIZATION_KERNELS_HPP
#define GPU_GENERIC_SYCL_BATCH_NORMALIZATION_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct batch_normalization_fwd_kernel_vec_t {
    batch_normalization_fwd_kernel_vec_t(
            const sycl_batch_normalization_conf_t &conf,
            xpu::sycl::in_memory_arg_t &data, xpu::sycl::in_memory_arg_t &scale,
            xpu::sycl::in_memory_arg_t &shift, xpu::sycl::in_memory_arg_t &stat,
            xpu::sycl::in_memory_arg_t &var, xpu::sycl::out_memory_arg_t &dst,
            xpu::sycl::out_memory_arg_t &ws, xpu::sycl::in_memory_arg_t &src1)
        : conf_(conf)
        , data_(data)
        , scale_(scale)
        , shift_(shift)
        , stat_(stat)
        , var_(var)
        , dst_(dst)
        , ws_(ws)
        , src1_(src1) {}

    void operator()(::sycl::nd_item<1> item) const {
        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dim_t start_c, end_c;
        balance211(conf_.C, conf_.n_thr, ithr, start_c, end_c);
        compute_fwd(start_c, end_c);
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.data_md; }
    const xpu::sycl::md_t &src1_md() const { return conf_.src1_md; }
    const data_type_t &ws_dt() const { return conf_.ws_dt; }
    const xpu::sycl::md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const xpu::sycl::md_t &var_md() const { return conf_.var_md; }
    const xpu::sycl::md_t &stat_d() const { return conf_.stat_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    const float epsilon() const { return conf_.batch_norm_epsilon; }

    inline static dim_t DATA_OFF(const xpu::sycl::md_t &mdw, dim_t n, dim_t c,
            dim_t d, dim_t h, dim_t w) {
        switch (mdw.ndims()) {
            case 1: return mdw.off(n);
            case 2: return mdw.off(n, c);
            case 3: return mdw.off(n, c, w);
            case 4: return mdw.off(n, c, h, w);
            case 5: return mdw.off(n, c, d, h, w);
            default: return 0;
        }
        return 0;
    }

    void compute_fwd(dim_t start_c, dim_t end_c) const {
        memory_tensor_t data_mem(data_, conf_.data_md);
        memory_tensor_t scale_mem(scale_, conf_.data_scaleshift_md);
        memory_tensor_t shift_mem(shift_, conf_.data_scaleshift_md);
        memory_tensor_t stat_mem(stat_, conf_.stat_md);
        memory_tensor_t var_mem(var_, conf_.var_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);
        memory_plain_t ws_mem(ws_, conf_.ws_dt);
        memory_tensor_t src1_mem(src1_, conf_.src1_md);

        const bool with_relu = conf_.with_relu;
        auto maybe_post_op = [&](float res) {
            if (with_relu) { return math::relu_fwd(res, conf_.alpha); }
            return res;
        };

        for (dim_t c = start_c; c < end_c; c++) {
            float v_mean = stat_mem.load(c);
            float v_variance = var_mem.load(c);
            float sqrt_variance = sqrtf(v_variance + conf_.batch_norm_epsilon);
            float sm = (conf_.use_scale ? scale_mem.load(c) : 1.0f)
                    / sqrt_variance;
            float sv = conf_.use_shift ? shift_mem.load(c) : 0;

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                auto d_off = DATA_OFF(data_md(), n, c, d, h, w);
                float bn_res = (sm * (data_mem.load(d_off) - v_mean)) + sv;
                if (conf_.fuse_norm_relu) {
                    if (bn_res <= 0) {
                        bn_res = 0;
                        if (conf_.is_training) ws_mem.store(0, d_off);
                    } else {
                        if (conf_.is_training) ws_mem.store(1, d_off);
                    }
                }

                if (conf_.fuse_norm_add_relu) {
                    float v = src1_mem.load(d_off);
                    bn_res = bn_res + v;
                    if (bn_res <= 0) {
                        bn_res = 0;
                        if (conf_.is_training) ws_mem.store(0, d_off);
                    } else {
                        if (conf_.is_training) ws_mem.store(1, d_off);
                    }
                }

                if (data_md().data_type() == data_type::s8) {
                    bn_res = gpu::generic::sycl::qz_a1b0<float,
                            xpu::sycl::prec_traits<data_type::s8>::type>()(
                            maybe_post_op(bn_res));
                    dst_mem.store(bn_res, d_off);
                } else {
                    bn_res = maybe_post_op(bn_res);
                    dst_mem.store(bn_res, d_off);
                }
            }
        }
    }

    sycl_batch_normalization_conf_t conf_;

    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::in_memory_arg_t scale_;
    xpu::sycl::in_memory_arg_t shift_;
    xpu::sycl::in_memory_arg_t stat_;
    xpu::sycl::in_memory_arg_t var_;
    xpu::sycl::out_memory_arg_t dst_;
    xpu::sycl::out_memory_arg_t ws_;
    xpu::sycl::in_memory_arg_t src1_;
};

struct batch_normalization_fwd_kernel_vec_t1 {
    batch_normalization_fwd_kernel_vec_t1(
            const sycl_batch_normalization_conf_t &conf,
            xpu::sycl::in_memory_arg_t &data, xpu::sycl::in_memory_arg_t &scale,
            xpu::sycl::in_memory_arg_t &shift, xpu::sycl::out_memory_arg_t &dst,
            xpu::sycl::out_memory_arg_t &mean_out,
            xpu::sycl::out_memory_arg_t &var_out,
            xpu::sycl::out_memory_arg_t &ws, xpu::sycl::in_memory_arg_t &src1)
        : conf_(conf)
        , data_(data)
        , scale_(scale)
        , shift_(shift)
        , dst_(dst)
        , mean_out_(mean_out)
        , var_out_(var_out)
        , ws_(ws)
        , src1_(src1) {}

    void operator()(::sycl::nd_item<1> item) const {
        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dim_t start_c, end_c;
        balance211(conf_.C, conf_.n_thr, ithr, start_c, end_c);
        compute_fwd(start_c, end_c);
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.data_md; }
    const xpu::sycl::md_t &src1_md() const { return conf_.src1_md; }
    const data_type_t &ws_dt() const { return conf_.ws_dt; }
    const xpu::sycl::md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const xpu::sycl::md_t &var_md() const { return conf_.var_md; }
    const xpu::sycl::md_t &stat_d() const { return conf_.stat_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    const float epsilon() const { return conf_.batch_norm_epsilon; }

    inline static dim_t DATA_OFF(const xpu::sycl::md_t &mdw, dim_t n, dim_t c,
            dim_t d, dim_t h, dim_t w) {
        switch (mdw.ndims()) {
            case 1: return mdw.off(n);
            case 2: return mdw.off(n, c);
            case 3: return mdw.off(n, c, w);
            case 4: return mdw.off(n, c, h, w);
            case 5: return mdw.off(n, c, d, h, w);
            default: return 0;
        }
        return 0;
    }

    void compute_fwd(dim_t start_c, dim_t end_c) const {
        memory_tensor_t data_mem(data_, conf_.data_md);
        memory_tensor_t scale_mem(scale_, conf_.data_scaleshift_md);
        memory_tensor_t shift_mem(shift_, conf_.data_scaleshift_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);
        memory_tensor_t stat_mem(mean_out_, conf_.stat_md);
        memory_tensor_t var_mem(var_out_, conf_.var_md);
        memory_tensor_t src1_mem(src1_, conf_.src1_md);
        memory_plain_t ws_mem(ws_, conf_.ws_dt);

        if (conf_.zero_dims && conf_.calculate_stats && conf_.save_stats) {
            for (dim_t i = 0; i < conf_.C; i++) {
                stat_mem.store(0, i);
                var_mem.store(0, i);
            }
        }

        const bool with_relu = conf_.with_relu;
        auto maybe_post_op = [&](float res) {
            if (with_relu) { return math::relu_fwd(res, conf_.alpha); }
            return res;
        };

        for (dim_t c = start_c; c < end_c; c++) {

            float v_mean = 0;
            float v_variance = 0;
            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                auto data_off = DATA_OFF(data_md(), n, c, d, h, w);
                v_mean += data_mem.load(data_off);
            }
            v_mean /= (conf_.W * conf_.N * conf_.H * conf_.D);

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                auto data_off = DATA_OFF(data_md(), n, c, d, h, w);
                auto m = data_mem.load(data_off) - v_mean;
                v_variance += m * m;
            }
            v_variance /= (conf_.W * conf_.H * conf_.N * conf_.D);

            float sqrt_variance = sqrtf(v_variance + conf_.batch_norm_epsilon);
            float sm = (conf_.use_scale ? scale_mem.load(c) : 1.0f)
                    / sqrt_variance;
            float sv = conf_.use_shift ? shift_mem.load(c) : 0;

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                auto d_off = DATA_OFF(data_md(), n, c, d, h, w);
                float bn_res = (sm * (data_mem.load(d_off) - v_mean)) + sv;
                if (conf_.fuse_norm_relu) {
                    if (bn_res <= 0) {
                        bn_res = 0;
                        if (conf_.is_training) ws_mem.store(0, d_off);
                    } else {
                        if (conf_.is_training) ws_mem.store(1, d_off);
                    }
                }

                if (conf_.fuse_norm_add_relu) {
                    float v = src1_mem.load(d_off);
                    bn_res = bn_res + v;
                    if (bn_res <= 0) {
                        bn_res = 0;
                        if (conf_.is_training) ws_mem.store(0, d_off);
                    } else {
                        if (conf_.is_training) ws_mem.store(1, d_off);
                    }
                }

                if (data_md().data_type() == data_type::s8) {
                    bn_res = gpu::generic::sycl::qz_a1b0<float,
                            xpu::sycl::prec_traits<data_type::s8>::type>()(
                            maybe_post_op(bn_res));
                    dst_mem.store(bn_res, d_off);
                } else {
                    bn_res = maybe_post_op(bn_res);
                    dst_mem.store(bn_res, d_off);
                }
            }
            if (conf_.calculate_stats) {
                if (conf_.save_stats) {
                    stat_mem.store(v_mean, c);
                    var_mem.store(v_variance, c);
                }
            }
        }
    }

    sycl_batch_normalization_conf_t conf_;

    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::in_memory_arg_t scale_;
    xpu::sycl::in_memory_arg_t shift_;
    xpu::sycl::out_memory_arg_t dst_;
    xpu::sycl::out_memory_arg_t mean_out_;
    xpu::sycl::out_memory_arg_t var_out_;
    xpu::sycl::out_memory_arg_t ws_;
    xpu::sycl::in_memory_arg_t src1_;
};

struct batch_normalization_bwd_kernel_vec_t {
    batch_normalization_bwd_kernel_vec_t(
            const sycl_batch_normalization_conf_t &conf,
            xpu::sycl::in_memory_arg_t &data,
            xpu::sycl::out_memory_arg_t &diff_data,
            xpu::sycl::in_memory_arg_t &scale,
            xpu::sycl::out_memory_arg_t &diff_scale,
            xpu::sycl::out_memory_arg_t &diff_shift,
            xpu::sycl::in_memory_arg_t &stat, xpu::sycl::in_memory_arg_t &var,
            xpu::sycl::in_memory_arg_t &diff_dst,
            xpu::sycl::in_memory_arg_t &dst, xpu::sycl::in_memory_arg_t &ws,
            xpu::sycl::out_memory_arg_t &diff_src1)
        : conf_(conf)
        , data_(data)
        , diff_data_(diff_data)
        , scale_(scale)
        , diff_scale_(diff_scale)
        , diff_shift_(diff_shift)
        , stat_(stat)
        , var_(var)
        , diff_dst_(diff_dst)
        , dst_(dst)
        , ws_(ws)
        , diff_src1_(diff_src1) {}

    void operator()(::sycl::nd_item<1> item) const {
        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dim_t start_c, end_c;
        balance211(conf_.C, conf_.n_thr, ithr, start_c, end_c);
        compute_bwd(start_c, end_c);
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.data_md; }
    const xpu::sycl::md_t &diff_data_md() const { return conf_.diff_data_md; }
    const data_type_t &diff_src1_dt() const { return conf_.diff_src1_dt; }
    const xpu::sycl::md_t &stat_d() const { return conf_.stat_md; }
    const data_type_t &ws_dt() const { return conf_.ws_dt; }
    const xpu::sycl::md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const xpu::sycl::md_t &diff_data_scaleshift_md() const {
        return conf_.diff_data_scaleshift_md;
    }
    const xpu::sycl::md_t &var_md() const { return conf_.var_md; }
    const xpu::sycl::md_t &diff_dst_md() const { return conf_.diff_dst_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    const unsigned flags() const { return conf_.flags; }
    const float epsilon() const { return conf_.batch_norm_epsilon; }

    static dim_t DATA_OFF(const xpu::sycl::md_t &mdw, dim_t n, dim_t c, dim_t d,
            dim_t h, dim_t w) {
        switch (mdw.ndims()) {
            case 1: return mdw.off(n);
            case 2: return mdw.off(n, c);
            case 3: return mdw.off(n, c, w);
            case 4: return mdw.off(n, c, h, w);
            case 5: return mdw.off(n, c, d, h, w);
            default: return 0;
        }
        return 0;
    }

    void compute_bwd(dim_t start_c, dim_t end_c) const {
        memory_tensor_t data_mem(data_, conf_.data_md);
        memory_tensor_t diff_data_mem(diff_data_, conf_.diff_data_md);
        memory_tensor_t scale_mem(scale_, conf_.data_scaleshift_md);
        memory_tensor_t diff_scale_mem(
                diff_scale_, conf_.diff_data_scaleshift_md);
        memory_tensor_t diff_shift_mem(
                diff_shift_, conf_.diff_data_scaleshift_md);
        memory_tensor_t stat_mem(stat_, conf_.stat_md);
        memory_tensor_t var_mem(var_, conf_.var_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);
        memory_tensor_t diff_dst_mem(diff_dst_, conf_.diff_dst_md);
        memory_plain_t diff_src1_mem(diff_src1_, conf_.diff_src1_dt);
        memory_plain_t ws_mem(ws_, conf_.ws_dt);

        if (conf_.zero_dims) {
            if (conf_.use_scale) {
                for (dim_t c = 0; c < conf_.C; ++c) {
                    diff_scale_mem.store(0.0f, c);
                }
            }
            if (conf_.use_shift) {
                for (dim_t c = 0; c < conf_.C; ++c) {
                    diff_shift_mem.store(0.0f, c);
                }
            }
        }

        for (dim_t c = start_c; c < end_c; c++) {
            float v_mean = stat_mem.load(c);
            float v_variance = var_mem.load(c);
            float sqrt_variance = static_cast<float>(
                    1.0f / sqrtf(v_variance + conf_.batch_norm_epsilon));
            float gamma = conf_.use_scale ? scale_mem.load(c) : 1.0f;
            float diff_gamma = 0;
            float diff_beta = 0;

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                const size_t s_off = DATA_OFF(data_md(), n, c, d, h, w);

                float dd = diff_dst_mem.load(
                        DATA_OFF(diff_data_md(), n, c, d, h, w));
                if (conf_.fuse_norm_relu && !ws_mem.load(s_off)) dd = 0;

                if (conf_.fuse_norm_add_relu) {
                    dd = ::dnnl::impl::math::relu_bwd(
                            dd, ws_mem.load(s_off), conf_.alpha);
                    diff_src1_mem.store(dd, s_off);
                }

                diff_gamma += (data_mem.load(s_off) - v_mean) * dd;
                diff_beta += dd;
            }
            diff_gamma *= sqrt_variance;

            if (conf_.use_scale) { diff_scale_mem.store(diff_gamma, c); }

            if (conf_.use_shift) { diff_shift_mem.store(diff_beta, c); }

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                const size_t s_off = DATA_OFF(data_md(), n, c, d, h, w);
                const size_t dd_off = DATA_OFF(diff_data_md(), n, c, d, h, w);
                float dd = diff_dst_mem.load(dd_off);

                if (conf_.fuse_norm_relu && !ws_mem.load(s_off)) dd = 0;

                if (conf_.fuse_norm_add_relu) {
                    dd = ::dnnl::impl::math::relu_bwd(
                            dd, ws_mem.load(s_off), conf_.alpha);
                    diff_src1_mem.store(dd, s_off);
                }

                float v_diff_src = dd;
                if (conf_.calculate_diff_stats) {
                    v_diff_src -= diff_beta
                                    / (conf_.D * conf_.W * conf_.H * conf_.N)
                            + (data_mem.load(s_off) - v_mean) * diff_gamma
                                    * sqrt_variance
                                    / (conf_.D * conf_.W * conf_.H * conf_.N);
                }
                v_diff_src *= gamma * sqrt_variance;
                diff_data_mem.store(v_diff_src, dd_off);
            }
        } //end of main loop

    } //end of compute

    sycl_batch_normalization_conf_t conf_;

    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::out_memory_arg_t diff_data_;
    xpu::sycl::in_memory_arg_t scale_;
    xpu::sycl::out_memory_arg_t diff_scale_;
    xpu::sycl::out_memory_arg_t diff_shift_;
    xpu::sycl::in_memory_arg_t stat_;
    xpu::sycl::in_memory_arg_t var_;
    xpu::sycl::in_memory_arg_t diff_dst_;
    xpu::sycl::in_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t ws_;
    xpu::sycl::out_memory_arg_t diff_src1_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
