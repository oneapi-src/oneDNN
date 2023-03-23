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

#ifndef GPU_SYCL_LAYER_NORMALIZATION_KERNELS_HPP
#define GPU_SYCL_LAYER_NORMALIZATION_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct layer_normalization_fwd_kernel_vec_t {
    layer_normalization_fwd_kernel_vec_t(
            const sycl_layer_normalization_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_in_memory_arg_t &scale,
            sycl_in_memory_arg_t &shift, sycl_in_memory_arg_t &stat,
            sycl_in_memory_arg_t &var, sycl_out_memory_arg_t &dst,
            sycl_in_memory_arg_t &rt_scale, sycl_in_memory_arg_t &dst_scale)
        : conf_(conf)
        , data_(data)
        , scale_(scale)
        , shift_(shift)
        , stat_(stat)
        , var_(var)
        , dst_(dst)
        , rt_scale_(rt_scale)
        , dst_scale_(dst_scale) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;
        size_t base_idx = offset_t * conf_.block_size;
        for (int i = 0; i < conf_.block_size; i++) {
            dim_t idx = base_idx + i;
            if (idx < conf_.N) { compute_alg_n(idx); }
        }
    }

private:
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &stat_md() const { return conf_.stat_md; }
    const sycl_md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const sycl_md_t &var_md() const { return conf_.var_md; }
    const sycl_md_t &stat_d() const { return conf_.stat_d; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    const unsigned flags() const { return conf_.flags; }
    const float epsilon() const { return conf_.layer_norm_epsilon; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *scale_ptr() const { return scale_.get_pointer(); }
    void *shift_ptr() const { return shift_.get_pointer(); }
    void *stat_ptr() const { return stat_.get_pointer(); }
    void *var_ptr() const { return var_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    void *rt_oscale_ptr() const { return rt_scale_.get_pointer(); }
    void *dst_oscale_ptr() const { return dst_scale_.get_pointer(); }

    inline void compute_alg_n(int idx) const {
        float eps = epsilon();
        const size_t s_off = conf_.stat_md.off_l(idx);
        auto v_mean
                = load_float_value(stat_md().data_type(), stat_ptr(), s_off);
        auto v_variance
                = load_float_value(var_md().data_type(), var_ptr(), s_off);
        dim_t C = conf_.C;

        float sqrt_variance = sqrtf(v_variance + eps);

        if (idx < conf_.N) {
            for (dim_t c = 0; c < C; ++c) {
                const float sm = ((conf_.use_scale) ? load_float_value(
                                          data_scaleshift_md().data_type(),
                                          scale_ptr(), c)
                                                    : 1.f)
                        / sqrt_variance;
                const float sv = conf_.use_shift
                        ? load_float_value(data_scaleshift_md().data_type(),
                                shift_ptr(), c)
                        : 0;
                dim_t index = idx * C + c;
                const auto src_off = data_md().off_l(index);
                const auto d_off = dst_md().off_l(index);
                float s = load_float_value(
                        data_md().data_type(), data_ptr(), src_off);
                float d = sm * (s - v_mean) + sv;

                float sr = conf_.src_def
                        ? 1.f
                        : load_float_value(data_scaleshift_md().data_type(),
                                rt_oscale_ptr(), 0);
                float ds = conf_.dst_def
                        ? 1.f
                        : load_float_value(data_scaleshift_md().data_type(),
                                dst_oscale_ptr(), 0);
                d = (d * sr * (1.f / ds));
                store_float_value(dst_md().data_type(), d, dst_ptr(), d_off);
            }
        }
    }

    sycl_layer_normalization_conf_t conf_;
    sycl_in_memory_arg_t data_;
    sycl_in_memory_arg_t scale_;
    sycl_in_memory_arg_t shift_;
    sycl_in_memory_arg_t stat_;
    sycl_in_memory_arg_t var_;
    sycl_out_memory_arg_t dst_;
    sycl_in_memory_arg_t rt_scale_;
    sycl_in_memory_arg_t dst_scale_;
};

struct layer_normalization_fwd_kernel_vec1_t {
    layer_normalization_fwd_kernel_vec1_t(
            const sycl_layer_normalization_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_in_memory_arg_t &scale,
            sycl_in_memory_arg_t &shift, sycl_out_memory_arg_t &dst,
            sycl_out_memory_arg_t &mean_out, sycl_out_memory_arg_t &var_out,
            sycl_in_memory_arg_t &rt_scale, sycl_in_memory_arg_t &dst_scale)
        : conf_(conf)
        , data_(data)
        , scale_(scale)
        , shift_(shift)
        , dst_(dst)
        , mean_out_(mean_out)
        , var_out_(var_out)
        , rt_scale_(rt_scale)
        , dst_scale_(dst_scale) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;

        size_t base_idx = offset_t * conf_.block_size;
        for (int i = 0; i < conf_.block_size; i++) {
            dim_t idx = base_idx + i;
            if (idx < conf_.N) { compute_alg_n(idx); }
        }
    }

private:
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &stat_md() const { return conf_.stat_md; }
    const sycl_md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const sycl_md_t &var_md() const { return conf_.var_md; }
    const sycl_md_t &stat_d() const { return conf_.stat_d; }
    const sycl_md_t &dst_md() const { return conf_.dst_md; }

    const unsigned flags() const { return conf_.flags; }
    const float epsilon() const { return conf_.layer_norm_epsilon; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *scale_ptr() const { return scale_.get_pointer(); }
    void *shift_ptr() const { return shift_.get_pointer(); }
    void *stat_out_ptr() const { return mean_out_.get_pointer(); }
    void *var_out_ptr() const { return var_out_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *rt_oscale_ptr() const { return rt_scale_.get_pointer(); }
    void *dst_oscale_ptr() const { return dst_scale_.get_pointer(); }

    inline void compute_alg_n(int idx) const {
        if (conf_.zero_dims && conf_.calculate_stats && conf_.save_stats) {
            store_float_value(stat_md().data_type(), 0, stat_out_ptr(), idx);
            store_float_value(var_md().data_type(), 0, var_out_ptr(), idx);
        }
        float eps = epsilon();
        const size_t s_off = conf_.stat_md.off_l(idx);
        float v_mean = 0.f;
        float v_variance = 0.f;
        dim_t C = conf_.C;

        if (conf_.calculate_stats) {
            for (dim_t c = 0; c < C; ++c) {
                dim_t index = idx * C + c;
                const auto sd_off = data_md().off_l(index);
                float s = load_float_value(
                        data_md().data_type(), data_ptr(), sd_off);
                v_mean += s;
            }
            v_mean /= C;
            for (dim_t c = 0; c < C; ++c) {
                dim_t index = idx * C + c;
                const auto s_off = data_md().off_l(index);
                float sc = load_float_value(
                        data_md().data_type(), data_ptr(), s_off);
                float m = sc - v_mean;
                v_variance += m * m;
            }
            v_variance /= C;
        }

        float sqrt_variance = sqrtf(v_variance + eps);

        if (idx < conf_.N) {
            for (dim_t c = 0; c < C; ++c) {
                const float sm = ((conf_.use_scale) ? load_float_value(
                                          data_scaleshift_md().data_type(),
                                          scale_ptr(), c)
                                                    : 1.f)
                        / sqrt_variance;
                const float sv = conf_.use_shift
                        ? load_float_value(data_scaleshift_md().data_type(),
                                shift_ptr(), c)
                        : 0;
                dim_t index = idx * C + c;
                const auto src_off = data_md().off_l(index);
                const auto d_off = dst_md().off_l(index);
                float s = load_float_value(
                        data_md().data_type(), data_ptr(), src_off);
                float d = sm * (s - v_mean) + sv;

                float sr = conf_.src_def
                        ? 1.f
                        : load_float_value(data_scaleshift_md().data_type(),
                                rt_oscale_ptr(), 0);
                float ds = conf_.dst_def
                        ? 1.f
                        : load_float_value(data_scaleshift_md().data_type(),
                                dst_oscale_ptr(), 0);
                d = (d * sr * (1.f / ds));

                store_float_value(dst_md().data_type(), d, dst_ptr(), d_off);
            }
        }

        if (conf_.calculate_stats && conf_.save_stats) {
            store_float_value(
                    stat_md().data_type(), v_mean, stat_out_ptr(), s_off);
            store_float_value(
                    var_md().data_type(), v_variance, var_out_ptr(), s_off);
        }
    }

    sycl_layer_normalization_conf_t conf_;
    sycl_in_memory_arg_t data_;
    sycl_in_memory_arg_t scale_;
    sycl_in_memory_arg_t shift_;
    sycl_out_memory_arg_t dst_;
    sycl_out_memory_arg_t mean_out_;
    sycl_out_memory_arg_t var_out_;
    sycl_in_memory_arg_t rt_scale_;
    sycl_in_memory_arg_t dst_scale_;
};

struct layer_normalization_bwd_kernel_vec_t {
    layer_normalization_bwd_kernel_vec_t(
            const sycl_layer_normalization_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_out_memory_arg_t &diff_data,
            sycl_in_memory_arg_t &scale, sycl_out_memory_arg_t &diff_scale,
            sycl_out_memory_arg_t &diff_shift, sycl_in_memory_arg_t &stat,
            sycl_in_memory_arg_t &var, sycl_in_memory_arg_t &diff_dst)
        : conf_(conf)
        , data_(data)
        , diff_data_(diff_data)
        , scale_(scale)
        , diff_scale_(diff_scale)
        , diff_shift_(diff_shift)
        , stat_(stat)
        , var_(var)
        , diff_dst_(diff_dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dim_t start_c, end_c;
        balance211(conf_.C, conf_.n_thr, ithr, start_c, end_c);
        compute_alg_bwd(start_c, end_c, ithr);
    }

private:
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &diff_data_md() const { return conf_.diff_data_md; }
    const sycl_md_t &stat_d() const { return conf_.stat_d; }
    const sycl_md_t &stat_md() const { return conf_.stat_md; }
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
    const float epsilon() const { return conf_.layer_norm_epsilon; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *scale_ptr() const { return scale_.get_pointer(); }
    void *diff_data_ptr() const { return diff_data_.get_pointer(); }
    void *diff_scale_ptr() const { return diff_scale_.get_pointer(); }
    void *diff_shift_ptr() const { return diff_shift_.get_pointer(); }
    void *stat_ptr() const { return stat_.get_pointer(); }
    void *var_ptr() const { return var_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }

    inline void compute_alg_bwd(dim_t start_c, dim_t end_c, size_t ithr) const {
        if ((dim_t)ithr >= conf_.C) return;
        float eps = epsilon();
        if (conf_.zero_dims) {
            if (conf_.use_scale) {
                for (dim_t c = start_c; c < end_c; ++c) {
                    store_float_value(diff_data_scaleshift_md().data_type(), 0,
                            diff_scale_ptr(), diff_data_scaleshift_md().off(c));
                }
            }
            if (conf_.use_shift) {
                for (dim_t c = end_c; c < end_c; ++c) {
                    store_float_value(diff_data_scaleshift_md().data_type(), 0,
                            diff_shift_ptr(), diff_data_scaleshift_md().off(c));
                }
            }
        }

        if (conf_.use_scale || conf_.use_shift) {
            for (dim_t c = start_c; c < end_c; ++c) {
                float diff_gamma = 0.f;
                float diff_beta = 0.f;

                for (dim_t n = 0; n < (conf_.N); ++n) {
                    const size_t index = (n * conf_.C) + c;
                    const auto src_off = data_md().off_l(index),
                               diff_dst_off = diff_dst_md().off_l(index),
                               s_off = stat_md().off_l(n);

                    float inv_sqrt_variance = 1.f
                            / sqrtf(load_float_value(var_md().data_type(),
                                            var_ptr(), s_off)
                                    + eps); //stat
                    float s = load_float_value(
                            data_md().data_type(), data_ptr(), src_off);
                    auto dd = load_float_value(diff_dst_md().data_type(),
                            diff_dst_ptr(), diff_dst_off);
                    auto stat_v = load_float_value(
                            stat_md().data_type(), stat_ptr(), s_off);
                    diff_gamma += (s - stat_v) * dd * inv_sqrt_variance;
                    diff_beta += dd;
                }
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
            }
        }
    }

    sycl_layer_normalization_conf_t conf_;

    sycl_in_memory_arg_t data_;
    sycl_out_memory_arg_t diff_data_;
    sycl_in_memory_arg_t scale_;
    sycl_out_memory_arg_t diff_scale_;
    sycl_out_memory_arg_t diff_shift_;
    sycl_in_memory_arg_t stat_;
    sycl_in_memory_arg_t var_;
    sycl_in_memory_arg_t diff_dst_;
};

struct layer_normalization_bwd_kernel_vec2_t {
    layer_normalization_bwd_kernel_vec2_t(
            const sycl_layer_normalization_conf_t &conf,
            sycl_in_memory_arg_t &data, sycl_out_memory_arg_t &diff_data,
            sycl_in_memory_arg_t &scale, sycl_out_memory_arg_t &diff_scale,
            sycl_out_memory_arg_t &diff_shift, sycl_in_memory_arg_t &stat,
            sycl_in_memory_arg_t &var, sycl_in_memory_arg_t &diff_dst)
        : conf_(conf)
        , data_(data)
        , diff_data_(diff_data)
        , scale_(scale)
        , diff_scale_(diff_scale)
        , diff_shift_(diff_shift)
        , stat_(stat)
        , var_(var)
        , diff_dst_(diff_dst) {}

    void operator()(::sycl::nd_item<1> item) const {
        size_t ithr = item.get_group(0) * conf_.wg_size + item.get_local_id();
        dim_t start_n, end_n;
        balance211(conf_.N, conf_.n_thr, ithr, start_n, end_n);
        compute_alg_bwd2(start_n, end_n, ithr);
    }

private:
    const sycl_md_t &data_md() const { return conf_.data_md; }
    const sycl_md_t &diff_data_md() const { return conf_.diff_data_md; }
    const sycl_md_t &stat_d() const { return conf_.stat_d; }
    const sycl_md_t &stat_md() const { return conf_.stat_md; }
    const sycl_md_t &data_scaleshift_md() const {
        return conf_.data_scaleshift_md;
    }
    const sycl_md_t &diff_data_scaleshift_md() const {
        return conf_.diff_data_scaleshift_md;
    }
    const sycl_md_t &var_md() const { return conf_.var_md; }
    const sycl_md_t &diff_dst_md() const { return conf_.diff_dst_md; }

    const unsigned flags() const { return conf_.flags; }
    const float epsilon() const { return conf_.layer_norm_epsilon; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *scale_ptr() const { return scale_.get_pointer(); }
    void *diff_data_ptr() const { return diff_data_.get_pointer(); }
    void *diff_scale_ptr() const { return diff_scale_.get_pointer(); }
    void *diff_shift_ptr() const { return diff_shift_.get_pointer(); }
    void *stat_ptr() const { return stat_.get_pointer(); }
    void *var_ptr() const { return var_.get_pointer(); }
    void *diff_dst_ptr() const { return diff_dst_.get_pointer(); }

    inline void compute_alg_bwd2(
            dim_t start_n, dim_t end_n, size_t ithr) const {

        dim_t C = conf_.C;
        if ((dim_t)ithr >= conf_.N) return;
        float eps = epsilon();

        for (dim_t n = start_n; n < end_n; ++n) {
            const size_t s_off = stat_md().off_l(n);
            float inv_sqrt_variance = 1.f
                    / sqrtf(load_float_value(
                                    var_md().data_type(), var_ptr(), s_off)
                            + eps);
            float dd_gamma = 0.f;
            float dd_gamma_x = 0.f;
            if (conf_.calculate_diff_stats) {

                for (dim_t c = 0; c < conf_.C; ++c) {

                    float gamma = (conf_.use_scale)
                            ? load_float_value(data_scaleshift_md().data_type(),
                                    scale_ptr(), data_scaleshift_md().off(c))
                            : 1.f;
                    const size_t src_off = data_md().off_l(n * conf_.C + c),
                                 diff_dst_off
                            = diff_dst_md().off_l(n * conf_.C + c);
                    auto dd = load_float_value(diff_dst_md().data_type(),
                            diff_dst_ptr(), diff_dst_off);
                    dd_gamma += (dd * gamma);
                    dd_gamma_x += (dd * gamma
                            * (load_float_value(data_md().data_type(),
                                       data_ptr(), src_off)
                                    - load_float_value(stat_md().data_type(),
                                            stat_ptr(), s_off)));
                }
                dd_gamma_x *= inv_sqrt_variance;
            }

            for (dim_t c = 0; c < C; ++c) {
                float gamma = (conf_.use_scale)
                        ? load_float_value(data_scaleshift_md().data_type(),
                                scale_ptr(), data_scaleshift_md().off(c))
                        : 1.f;
                const size_t src_off
                        = data_md().off_l(n * conf_.C + c),
                        diff_src_off = diff_data_md().off_l(n * conf_.C + c),
                        diff_dst_off = diff_dst_md().off_l(n * conf_.C + c);
                float v_diff_src = load_float_value(diff_dst_md().data_type(),
                                           diff_dst_ptr(), diff_dst_off)
                        * gamma;
                if (conf_.calculate_diff_stats) {
                    v_diff_src -= dd_gamma / C;
                    v_diff_src
                            -= (load_float_value(data_md().data_type(),
                                        data_ptr(), src_off)
                                       - load_float_value(stat_md().data_type(),
                                               stat_ptr(), s_off))
                            * dd_gamma_x * inv_sqrt_variance / C;
                }
                v_diff_src *= inv_sqrt_variance;
                store_float_value(diff_data_md().data_type(), v_diff_src,
                        diff_data_ptr(), diff_src_off);
            }
        }
    }

    sycl_layer_normalization_conf_t conf_;

    sycl_in_memory_arg_t data_;
    sycl_out_memory_arg_t diff_data_;
    sycl_in_memory_arg_t scale_;
    sycl_out_memory_arg_t diff_scale_;
    sycl_out_memory_arg_t diff_shift_;
    sycl_in_memory_arg_t stat_;
    sycl_in_memory_arg_t var_;
    sycl_in_memory_arg_t diff_dst_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
