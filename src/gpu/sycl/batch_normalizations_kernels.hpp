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

#ifndef GPU_SYCL_BATCH_NORMALIZATION_KERNELS_HPP
#define GPU_SYCL_BATCH_NORMALIZATION_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

namespace {

using acc_data_t = float;

template <typename T>
inline float maybe_up_convert(T x) {
    return x;
}

template <>
inline float maybe_up_convert<bfloat16_t>(bfloat16_t x) {
    return (float)x;
}

} // namespace

struct batch_normalization_fwd_kernel_vec_t {
    batch_normalization_fwd_kernel_vec_t(
            const sycl_batch_normalization_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_in_memory_arg_t &scale,
            sycl_in_memory_arg_t &shift, sycl_in_memory_arg_t &stat,
            sycl_in_memory_arg_t &var, sycl_out_memory_arg_t &dst,
            sycl_out_memory_arg_t &ws, sycl_in_memory_arg_t &src1)
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
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &src1_md() const { return conf_.src1_md; }
    const sycl_md_t &ws_md() const { return conf_.ws_md; }
    const sycl_md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const sycl_md_t &var_md() const { return conf_.var_md; }
    const sycl_md_t &stat_d() const { return conf_.stat_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }
    const float epsilon() const { return conf_.batch_norm_epsilon; }

    inline static dim_t DATA_OFF(
            const sycl_md_t &mdw, dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
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
        const bool with_relu = conf_.with_relu;
        auto maybe_post_op = [&](float res) {
            if (with_relu) { return math::relu_fwd(res, conf_.alpha); }
            return res;
        };

        for (dim_t c = start_c; c < end_c; c++) {
            float v_mean
                    = load_float_value(stat_d().data_type(), stat_ptr(), c);
            float v_variance
                    = load_float_value(var_md().data_type(), var_ptr(), c);
            float sqrt_variance = sqrtf(v_variance + conf_.batch_norm_epsilon);
            float sm = (conf_.use_scale ? load_float_value(
                                data_scaleshift_md().data_type(), scale_ptr(),
                                data_scaleshift_md().off(c))
                                        : 1.0f)
                    / sqrt_variance;
            float sv = conf_.use_shift
                    ? load_float_value(data_scaleshift_md().data_type(),
                            shift_ptr(), data_scaleshift_md().off(c))
                    : 0;

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                auto d_off = DATA_OFF(data_md(), n, c, d, h, w);
                float bn_res = (sm
                                       * (maybe_up_convert(load_float_value(
                                                  data_md().data_type(),
                                                  data_ptr(), d_off))
                                               - v_mean))
                        + sv;
                if (conf_.fuse_norm_relu) {
                    if (bn_res <= 0) {
                        bn_res = 0;
                        if (conf_.is_training)
                            store_float_value(
                                    ws_md().data_type(), 0, ws_ptr(), d_off);
                    } else {
                        if (conf_.is_training)
                            store_float_value(
                                    ws_md().data_type(), 1, ws_ptr(), d_off);
                    }
                }

                if (conf_.fuse_norm_add_relu) {
                    float v = load_float_value(
                            src1_md().data_type(), src1_ptr(), d_off);
                    bn_res = bn_res + v;
                    if (bn_res <= 0) {
                        bn_res = 0;
                        if (conf_.is_training)
                            store_float_value(
                                    ws_md().data_type(), 0, ws_ptr(), d_off);
                    } else {
                        if (conf_.is_training)
                            store_float_value(
                                    ws_md().data_type(), 1, ws_ptr(), d_off);
                    }
                }

                if (data_md().data_type() == data_type::s8) {
                    bn_res = ::dnnl::impl::sycl::qz_a1b0<float,
                            sycl_prec_traits<data_type::s8>::type>()(
                            maybe_post_op(bn_res));
                    store_float_value(
                            dst_md().data_type(), bn_res, dst_ptr(), d_off);
                } else {
                    bn_res = maybe_post_op(bn_res);
                    store_float_value(
                            dst_md().data_type(), bn_res, dst_ptr(), d_off);
                }
            }
        }
    }

    void *data_ptr() const { return data_.get_pointer(); }
    void *src1_ptr() const { return src1_.get_pointer(); }
    void *scale_ptr() const { return scale_.get_pointer(); }
    void *shift_ptr() const { return shift_.get_pointer(); }
    void *stat_ptr() const { return stat_.get_pointer(); }
    void *var_ptr() const { return var_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *ws_ptr() const { return ws_.get_pointer(); }

    sycl_batch_normalization_conf_t conf_;

    sycl_in_memory_arg_t data_;
    sycl_in_memory_arg_t scale_;
    sycl_in_memory_arg_t shift_;
    sycl_in_memory_arg_t stat_;
    sycl_in_memory_arg_t var_;
    sycl_out_memory_arg_t dst_;
    sycl_out_memory_arg_t ws_;
    sycl_in_memory_arg_t src1_;
};

struct batch_normalization_fwd_kernel_vec_t1 {
    batch_normalization_fwd_kernel_vec_t1(
            const sycl_batch_normalization_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_in_memory_arg_t &scale,
            sycl_in_memory_arg_t &shift, sycl_out_memory_arg_t &dst,
            sycl_out_memory_arg_t &mean_out, sycl_out_memory_arg_t &var_out,
            sycl_out_memory_arg_t &ws, sycl_in_memory_arg_t &src1)
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
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &src1_md() const { return conf_.src1_md; }
    const sycl_md_t &ws_md() const { return conf_.ws_md; }
    const sycl_md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const sycl_md_t &var_md() const { return conf_.var_md; }
    const sycl_md_t &stat_d() const { return conf_.stat_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }
    const float epsilon() const { return conf_.batch_norm_epsilon; }

    inline static dim_t DATA_OFF(
            const sycl_md_t &mdw, dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
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

        if (conf_.zero_dims && conf_.calculate_stats && conf_.save_stats) {
            for (dim_t i = 0; i < conf_.C; i++) {
                store_float_value(stat_d().data_type(), 0, stat_out_ptr(), i);
                store_float_value(var_md().data_type(), 0, var_out_ptr(), i);
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
                v_mean += maybe_up_convert(load_float_value(
                        data_md().data_type(), data_ptr(), data_off));
            }
            v_mean /= (conf_.W * conf_.N * conf_.H * conf_.D);

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                auto data_off = DATA_OFF(data_md(), n, c, d, h, w);
                auto m = load_float_value(
                                 data_md().data_type(), data_ptr(), data_off)
                        - v_mean;
                v_variance += m * m;
            }
            v_variance /= (conf_.W * conf_.H * conf_.N * conf_.D);

            float sqrt_variance = sqrtf(v_variance + conf_.batch_norm_epsilon);
            float sm = (conf_.use_scale ? load_float_value(
                                data_scaleshift_md().data_type(), scale_ptr(),
                                data_scaleshift_md().off(c))
                                        : 1.0f)
                    / sqrt_variance;
            float sv = conf_.use_shift
                    ? load_float_value(data_scaleshift_md().data_type(),
                            shift_ptr(), data_scaleshift_md().off(c))
                    : 0;

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                auto d_off = DATA_OFF(data_md(), n, c, d, h, w);
                float bn_res = (sm
                                       * (maybe_up_convert(load_float_value(
                                                  data_md().data_type(),
                                                  data_ptr(), d_off))
                                               - v_mean))
                        + sv;
                if (conf_.fuse_norm_relu) {
                    if (bn_res <= 0) {
                        bn_res = 0;
                        if (conf_.is_training)
                            store_float_value(
                                    ws_md().data_type(), 0, ws_ptr(), d_off);
                    } else {
                        if (conf_.is_training)
                            store_float_value(
                                    ws_md().data_type(), 1, ws_ptr(), d_off);
                    }
                }

                if (conf_.fuse_norm_add_relu) {
                    float v = load_float_value(
                            src1_md().data_type(), src1_ptr(), d_off);
                    bn_res = bn_res + v;
                    if (bn_res <= 0) {
                        bn_res = 0;
                        if (conf_.is_training)
                            store_float_value(
                                    ws_md().data_type(), 0, ws_ptr(), d_off);
                    } else {
                        if (conf_.is_training)
                            store_float_value(
                                    ws_md().data_type(), 1, ws_ptr(), d_off);
                    }
                }

                if (data_md().data_type() == data_type::s8) {
                    bn_res = ::dnnl::impl::sycl::qz_a1b0<float,
                            sycl_prec_traits<data_type::s8>::type>()(
                            maybe_post_op(bn_res));
                    store_float_value(
                            dst_md().data_type(), bn_res, dst_ptr(), d_off);
                } else {
                    bn_res = maybe_post_op(bn_res);
                    store_float_value(
                            dst_md().data_type(), bn_res, dst_ptr(), d_off);
                }
            }
            if (conf_.calculate_stats) {
                if (conf_.save_stats) {
                    store_float_value(
                            stat_d().data_type(), v_mean, stat_out_ptr(), c);
                    store_float_value(
                            var_md().data_type(), v_variance, var_out_ptr(), c);
                }
            }
        }
    }

    void *data_ptr() const { return data_.get_pointer(); }
    void *src1_ptr() const { return src1_.get_pointer(); }
    void *scale_ptr() const { return scale_.get_pointer(); }
    void *shift_ptr() const { return shift_.get_pointer(); }
    void *stat_out_ptr() const { return mean_out_.get_pointer(); }
    void *var_out_ptr() const { return var_out_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *ws_ptr() const { return ws_.get_pointer(); }

    sycl_batch_normalization_conf_t conf_;

    sycl_in_memory_arg_t data_;
    sycl_in_memory_arg_t scale_;
    sycl_in_memory_arg_t shift_;
    sycl_out_memory_arg_t dst_;
    sycl_out_memory_arg_t mean_out_;
    sycl_out_memory_arg_t var_out_;
    sycl_out_memory_arg_t ws_;
    sycl_in_memory_arg_t src1_;
};

struct batch_normalization_bwd_kernel_vec_t {
    batch_normalization_bwd_kernel_vec_t(
            const sycl_batch_normalization_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_out_memory_arg_t &diff_data,
            sycl_in_memory_arg_t &scale, sycl_out_memory_arg_t &diff_scale,
            sycl_out_memory_arg_t &diff_shift, sycl_in_memory_arg_t &stat,
            sycl_in_memory_arg_t &var, sycl_in_memory_arg_t &diff_dst,
            sycl_in_memory_arg_t &dst, sycl_in_memory_arg_t &ws,
            sycl_in_memory_arg_t &diff_src1)
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
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &diff_data_md() const { return conf_.diff_data_md; }
    const sycl_md_t &diff_src1_md() const { return conf_.diff_src1_md; }
    const sycl_md_t &stat_d() const { return conf_.stat_md; }
    const sycl_md_t &ws_md() const { return conf_.ws_md; }
    const sycl_md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const sycl_md_t &diff_data_scaleshift_md() const {
        return conf_.diff_data_scaleshift_md;
    }
    const sycl_md_t &var_md() const { return conf_.var_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }
    const unsigned flags() const { return conf_.flags; }
    const float epsilon() const { return conf_.batch_norm_epsilon; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *scale_ptr() const { return scale_.get_pointer(); }
    void *diff_data_ptr() const { return diff_data_.get_pointer(); }
    void *diff_src1_ptr() const { return diff_src1_.get_pointer(); }
    void *diff_scale_ptr() const { return diff_scale_.get_pointer(); }
    void *diff_shift_ptr() const { return diff_shift_.get_pointer(); }
    void *stat_ptr() const { return stat_.get_pointer(); }
    void *var_ptr() const { return var_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *ws_ptr() const { return ws_.get_pointer(); }

    static dim_t DATA_OFF(
            const sycl_md_t &mdw, dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
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
        if (conf_.zero_dims) {
            if (conf_.use_scale) {
                for (dim_t c = 0; c < conf_.C; ++c) {
                    store_float_value(diff_data_scaleshift_md().data_type(),
                            0.0f, diff_scale_ptr(),
                            diff_data_scaleshift_md().off(c));
                }
            }
            if (conf_.use_shift) {
                for (dim_t c = 0; c < conf_.C; ++c) {
                    store_float_value(diff_data_scaleshift_md().data_type(),
                            0.0f, diff_shift_ptr(),
                            diff_data_scaleshift_md().off(c));
                }
            }
        }

        for (dim_t c = start_c; c < end_c; c++) {

            float v_mean
                    = load_float_value(stat_d().data_type(), stat_ptr(), c);
            float v_variance
                    = load_float_value(var_md().data_type(), var_ptr(), c);
            float sqrt_variance = static_cast<float>(
                    1.0f / sqrtf(v_variance + conf_.batch_norm_epsilon));
            float gamma = conf_.use_scale
                    ? load_float_value(data_scaleshift_md().data_type(),
                            scale_ptr(), data_scaleshift_md().off(c))
                    : 1.0f;
            float diff_gamma = 0;
            float diff_beta = 0;

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                const size_t s_off = DATA_OFF(data_md(), n, c, d, h, w);

                float dd = maybe_up_convert(load_float_value(
                        diff_dst_md().data_type(), diff_dst_ptr(),
                        DATA_OFF(diff_data_md(), n, c, d, h, w)));
                if (conf_.fuse_norm_relu
                        && !load_float_value(
                                ws_md().data_type(), ws_ptr(), s_off))
                    dd = 0;

                if (conf_.fuse_norm_add_relu) {
                    dd = ::dnnl::impl::math::relu_bwd(dd,
                            load_float_value(
                                    ws_md().data_type(), ws_ptr(), s_off),
                            conf_.alpha);
                    store_float_value(diff_src1_md().data_type(), dd,
                            diff_src1_ptr(), s_off);
                }

                diff_gamma
                        += (maybe_up_convert(load_float_value(
                                    data_md().data_type(), data_ptr(), s_off))
                                   - v_mean)
                        * dd;
                diff_beta += dd;
            }
            diff_gamma *= sqrt_variance;

            if (conf_.use_scale) {
                store_float_value(diff_data_scaleshift_md().data_type(),
                        diff_gamma, diff_scale_ptr(),
                        diff_data_scaleshift_md().off(c));
            }

            if (conf_.use_shift) {
                store_float_value(diff_data_scaleshift_md().data_type(),
                        diff_beta, diff_shift_ptr(),
                        diff_data_scaleshift_md().off(c));
            }

            for_(dim_t n = 0; n < conf_.N; ++n)
            for_(dim_t d = 0; d < conf_.D; ++d)
            for_(dim_t h = 0; h < conf_.H; ++h)
            for (dim_t w = 0; w < conf_.W; ++w) {
                const size_t s_off = DATA_OFF(data_md(), n, c, d, h, w);
                const size_t dd_off = DATA_OFF(diff_data_md(), n, c, d, h, w);
                float dd = maybe_up_convert(load_float_value(
                        diff_dst_md().data_type(), diff_dst_ptr(), dd_off));

                if (conf_.fuse_norm_relu
                        && !load_float_value(
                                ws_md().data_type(), ws_ptr(), s_off))
                    dd = 0;

                if (conf_.fuse_norm_add_relu) {
                    dd = ::dnnl::impl::math::relu_bwd(dd,
                            load_float_value(
                                    ws_md().data_type(), ws_ptr(), s_off),
                            conf_.alpha);
                    store_float_value(diff_src1_md().data_type(), dd,
                            diff_src1_ptr(), s_off);
                }

                float v_diff_src = dd;
                if (conf_.calculate_diff_stats) {
                    v_diff_src -= diff_beta
                                    / (conf_.D * conf_.W * conf_.H * conf_.N)
                            + (maybe_up_convert(
                                       load_float_value(data_md().data_type(),
                                               data_ptr(), s_off))
                                      - v_mean)
                                    * diff_gamma * sqrt_variance
                                    / (conf_.D * conf_.W * conf_.H * conf_.N);
                }
                v_diff_src *= gamma * sqrt_variance;
                store_float_value(diff_data_md().data_type(), v_diff_src,
                        diff_data_ptr(), dd_off);
            }
        } //end of main loop

    } //end of compute

    sycl_batch_normalization_conf_t conf_;

    sycl_in_memory_arg_t data_;
    sycl_out_memory_arg_t diff_data_;
    sycl_in_memory_arg_t scale_;
    sycl_out_memory_arg_t diff_scale_;
    sycl_out_memory_arg_t diff_shift_;
    sycl_in_memory_arg_t stat_;
    sycl_in_memory_arg_t var_;
    sycl_in_memory_arg_t diff_dst_;
    sycl_in_memory_arg_t dst_;
    sycl_in_memory_arg_t ws_;
    sycl_in_memory_arg_t diff_src1_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
