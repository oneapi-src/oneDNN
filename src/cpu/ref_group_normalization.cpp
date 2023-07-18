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
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_group_normalization.hpp"
#include "cpu/ref_io_helper.hpp"

#define DATA_OFF(f, n, c, d, h, w) \
    (ndims == 2) ? (f).off(n, c) \
                 : ((ndims == 3) ? (f).off(n, c, w) \
                                 : ((ndims == 4) ? (f).off(n, c, h, w) \
                                                 : (f).off(n, c, d, h, w)))

namespace dnnl {
namespace impl {
namespace cpu {

status_t ref_group_normalization_fwd_t::execute(const exec_ctx_t &ctx) const {
    status_t status = status::success;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper ss_d(pd()->weights_md());

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);
    auto mean = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
            : CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_MEAN, status);
    CHECK(status);
    auto variance = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
            : CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_VARIANCE, status);
    CHECK(status);

    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const auto ndims = src_d.ndims();
    const auto N = pd()->MB();

    const auto C = pd()->C();
    const auto D = pd()->D();
    const auto H = pd()->H();
    const auto W = pd()->W();

    const auto G = pd()->desc()->groups;
    const auto eps = pd()->desc()->group_norm_epsilon;
    const auto calculate_stats = !pd()->stats_is_src();
    const auto save_stats = pd()->is_training();

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) {
        if (calculate_stats && save_stats)
            for (dim_t n = 0; n < N; n++) {
                for (dim_t g = 0; g < G; g++) {
                    size_t off = n * G + g;
                    mean[off] = 0;
                    variance[off] = 0;
                }
            }
        return status::success;
    }

    const auto C_PER_G = C / G;
    auto get_c_start = [&C_PER_G](int64_t g) { return g * C_PER_G; };

    parallel_nd(N, G, [&](dim_t n, dim_t g) {
        size_t stat_off = n * G + g;
        float v_mean = calculate_stats ? 0 : mean[stat_off];
        float v_variance = calculate_stats ? 0 : variance[stat_off];

        if (calculate_stats) {
            for_(int c = get_c_start(g); c < get_c_start(g + 1); ++c)
            for_(int d = 0; d < D; ++d)
            for_(int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const auto s_off = DATA_OFF(src_d, n, c, d, h, w);
                float s = io::load_float_value(src_d.data_type(), src, s_off);
                v_mean += s;
            }
            v_mean /= C_PER_G * W * H * D;

            for_(int c = get_c_start(g); c < get_c_start(g + 1); ++c)
            for_(int d = 0; d < D; ++d)
            for_(int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const auto s_off = DATA_OFF(src_d, n, c, d, h, w);
                float s = io::load_float_value(src_d.data_type(), src, s_off);
                float m = s - v_mean;
                v_variance += m * m;
            }
            v_variance /= C_PER_G * W * H * D;
        }
        float sqrt_variance = sqrtf(v_variance + eps);

        for (int c = get_c_start(g); c < get_c_start(g + 1); ++c) {
            float sm = (scale ? scale[ss_d.off(c)] : 1.0f) / sqrt_variance;
            float sv = shift ? shift[ss_d.off(c)] : 0;

            for_(dim_t d = 0; d < D; ++d)
            for_(dim_t h = 0; h < H; ++h)
            for (dim_t w = 0; w < W; ++w) {
                auto s_off = DATA_OFF(src_d, n, c, d, h, w);
                auto d_off = DATA_OFF(dst_d, n, c, d, h, w);

                float s = io::load_float_value(src_d.data_type(), src, s_off);
                float val = sm * (s - v_mean) + sv;
                val *= src_scales[0];

                // post-ops
                ref_post_ops_t::args_t args;
                args.ctx = &ctx;
                args.l_offset = n * C * D * H * W + c * D * H * W + d * H * W
                        + h * W + w;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(val, args);
                val *= dst_scales[0];
                io::store_float_value(dst_d.data_type(), val, dst, d_off);
            }
        }

        if (calculate_stats) {
            if (save_stats) {
                mean[stat_off] = v_mean;
                variance[stat_off] = v_variance;
            }
        }
    });
    return status::success;
}

status_t ref_group_normalization_bwd_t::execute(const exec_ctx_t &ctx) const {
    status_t status = status::success;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper ss_d(pd()->weights_md());
    const memory_desc_wrapper diff_ss_d(pd()->diff_weights_md());

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
    auto variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);

    auto diff_src = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    auto scale = CTX_IN_MEM(float *, DNNL_ARG_SCALE);
    auto diff_scale = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SCALE, status);
    CHECK(status);
    auto diff_shift = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SHIFT, status);
    CHECK(status);

    const auto ndims = src_d.ndims();
    const auto N = pd()->MB();
    const auto C = pd()->C();
    const auto D = pd()->D();
    const auto H = pd()->H();
    const auto W = pd()->W();

    const auto G = pd()->desc()->groups;
    const auto eps = pd()->desc()->group_norm_epsilon;
    const auto calculate_diff_stats = !pd()->stats_is_src();

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) {
        if (diff_scale) {
            for (dim_t c = 0; c < C; ++c) {
                diff_scale[diff_ss_d.off(c)] = 0.0f;
            }
        }
        if (diff_shift) {
            for (dim_t c = 0; c < C; ++c) {
                diff_shift[diff_ss_d.off(c)] = 0.0f;
            }
        }
        return status::success;
    }

    const auto C_PER_G = C / G;
    const auto CSP = C_PER_G * D * H * W;

    parallel_nd(C, [&](dim_t c) {
        int64_t g = c / C_PER_G;

        float gamma = scale ? scale[ss_d.off(c)] : 1.0f;
        float diff_gamma = 0;
        float diff_beta = 0;

        for (dim_t n = 0; n < N; ++n) {
            size_t stat_off = n * G + g;
            float v_mean = mean[stat_off];
            float v_variance = variance[stat_off];
            float sqrt_variance = 1.0f / sqrtf(v_variance + eps);

            for_(dim_t d = 0; d < D; ++d)
            for_(dim_t h = 0; h < H; ++h)
            for (dim_t w = 0; w < W; ++w) {
                const size_t s_off = DATA_OFF(src_d, n, c, d, h, w);
                const size_t dd_off = DATA_OFF(diff_dst_d, n, c, d, h, w);
                float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, dd_off);
                float s = io::load_float_value(src_d.data_type(), src, s_off);
                diff_gamma += (s - v_mean) * dd * sqrt_variance;
                diff_beta += dd;
            }
        }

        if (diff_scale) diff_scale[diff_ss_d.off(c)] = diff_gamma;
        if (diff_shift) diff_shift[diff_ss_d.off(c)] = diff_beta;

        for (dim_t n = 0; n < N; ++n) {
            size_t stat_off = n * G + g;
            float v_mean = mean[stat_off];
            float v_variance = variance[stat_off];
            float sqrt_variance = 1.0f / sqrtf(v_variance + eps);

            for_(dim_t d = 0; d < D; ++d)
            for_(dim_t h = 0; h < H; ++h)
            for (dim_t w = 0; w < W; ++w) {
                const size_t s_off = DATA_OFF(src_d, n, c, d, h, w);
                const size_t dd_off = DATA_OFF(diff_dst_d, n, c, d, h, w);
                const size_t ds_off = DATA_OFF(diff_src_d, n, c, d, h, w);
                float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, dd_off);
                float s = io::load_float_value(src_d.data_type(), src, s_off);
                float v_diff_src = dd;
                if (calculate_diff_stats) {
                    v_diff_src -= diff_beta / CSP
                            + (s - v_mean) * diff_gamma * sqrt_variance / CSP;
                }

                v_diff_src *= gamma * sqrt_variance;
                io::store_float_value(
                        diff_src_d.data_type(), v_diff_src, diff_src, ds_off);
            }
        }
    });
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
