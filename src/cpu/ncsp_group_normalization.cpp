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

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"

#include "cpu/ncsp_group_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

status_t ncsp_group_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    const bool save_stats = pd()->is_training();
    const bool calculate_stats = !pd()->stats_is_src();

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto src_dt = pd()->src_md()->data_type;
    const auto dst_dt = pd()->dst_md()->data_type;

    const auto use_scale = pd()->use_scale();
    const auto use_shift = pd()->use_shift();

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();
    float *__restrict cvt_scratch
            = scratchpad.template get<float>(key_gnorm_cvt);

    float *mean {nullptr}, *variance {nullptr};
    if (!calculate_stats) {
        mean = const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN));
        variance = const_cast<float *>(
                CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE));
    } else {
        if (save_stats) {
            mean = CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
            variance = CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
        }
    }

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);
    const float combined_scale = src_scales[0] * dst_scales[0];

    const dim_t N = pd()->MB();
    const dim_t G = pd()->desc()->groups;
    const dim_t C = pd()->C();
    const dim_t SP = pd()->D() * pd()->H() * pd()->W();

    const float eps = pd()->desc()->group_norm_epsilon;

    const dim_t C_PER_G = C / G;
    auto get_c_start = [&C_PER_G](dim_t g) { return g * C_PER_G; };

    const dim_t sp_block_nelems
            = static_cast<dim_t>(pd_t::cvt_per_thread_size_);
    const dim_t n_sp_block = SP / sp_block_nelems;
    const dim_t sp_block_reminder = SP % sp_block_nelems;

    const int nthr = pd()->nthr_;

    auto kernel = [&](int ithr, int, const dim_t n, const dim_t g) {
        float m = 0.0f;
        float v = 0.0f;
        if (calculate_stats) {
            for (dim_t c = get_c_start(g); c < get_c_start(g + 1); ++c) {
                const size_t s_off = (size_t)n * C * SP + c * SP;
                const char *__restrict _src
                        = reinterpret_cast<const char *>(src)
                        + s_off * src_d.data_type_size();
                for (dim_t sp_block = 0; sp_block < n_sp_block; sp_block++) {
                    const float *__restrict src_f32 {nullptr};
                    if (src_dt != data_type::f32) {
                        float *tmp = cvt_scratch + sp_block_nelems * ithr;
                        for (dim_t sp = 0; sp < sp_block_nelems; ++sp)
                            tmp[sp] = io::load_float_value(
                                    src_d.data_type(), _src, sp);
                        src_f32 = tmp;
                    } else {
                        src_f32 = reinterpret_cast<const float *__restrict>(
                                _src);
                    }
                    PRAGMA_OMP_SIMD(reduction(+ : m))
                    for (dim_t sp = 0; sp < sp_block_nelems; sp++) {
                        float s = src_f32[sp];
                        m += s;
                    }
                    _src += sp_block_nelems * src_d.data_type_size();
                }
                if (sp_block_reminder) {
                    const float *__restrict src_f32 {nullptr};
                    if (src_dt != data_type::f32) {
                        float *tmp = cvt_scratch + sp_block_nelems * ithr;
                        for (dim_t sp = 0; sp < sp_block_reminder; ++sp)
                            tmp[sp] = io::load_float_value(
                                    src_d.data_type(), _src, sp);
                        src_f32 = tmp;
                    } else {
                        src_f32 = reinterpret_cast<const float *__restrict>(
                                _src);
                    }
                    PRAGMA_OMP_SIMD(reduction(+ : m))
                    for (dim_t sp = 0; sp < sp_block_reminder; sp++) {
                        float s = src_f32[sp];
                        m += s;
                    }
                }
            }
            m /= SP * C_PER_G;

            for (dim_t c = get_c_start(g); c < get_c_start(g + 1); ++c) {
                const size_t s_off = (size_t)n * C * SP + c * SP;
                const char *__restrict _src
                        = reinterpret_cast<const char *>(src)
                        + s_off * src_d.data_type_size();
                for (dim_t sp_block = 0; sp_block < n_sp_block; sp_block++) {
                    const float *__restrict src_f32 {nullptr};
                    if (src_dt != data_type::f32) {
                        float *tmp = cvt_scratch + sp_block_nelems * ithr;
                        for (dim_t sp = 0; sp < sp_block_nelems; ++sp)
                            tmp[sp] = io::load_float_value(
                                    src_d.data_type(), _src, sp);
                        src_f32 = tmp;
                    } else {
                        src_f32 = reinterpret_cast<const float *__restrict>(
                                _src);
                    }
                    PRAGMA_OMP_SIMD(reduction(+ : v))
                    for (dim_t sp = 0; sp < sp_block_nelems; sp++) {
                        float s = src_f32[sp];
                        float s0 = s - m;
                        v += s0 * s0;
                    }
                    _src += sp_block_nelems * src_d.data_type_size();
                }
                if (sp_block_reminder) {
                    const float *__restrict src_f32 {nullptr};
                    if (src_dt != data_type::f32) {
                        float *tmp = cvt_scratch + sp_block_nelems * ithr;
                        for (dim_t sp = 0; sp < sp_block_reminder; ++sp)
                            tmp[sp] = io::load_float_value(
                                    src_d.data_type(), _src, sp);
                        src_f32 = tmp;
                    } else {
                        src_f32 = reinterpret_cast<const float *__restrict>(
                                _src);
                    }
                    PRAGMA_OMP_SIMD(reduction(+ : v))
                    for (dim_t sp = 0; sp < sp_block_reminder; sp++) {
                        float s = src_f32[sp];
                        float s0 = s - m;
                        v += s0 * s0;
                    }
                }
            }
            v /= SP * C_PER_G;
        } else {
            m = mean[n * G + g];
            v = variance[n * G + g];
        }

        const float sqrt_variance = sqrtf(v + eps);

        for (dim_t c = get_c_start(g); c < get_c_start(g + 1); ++c) {
            const size_t s_off = (size_t)n * C * SP + c * SP;
            const char *__restrict _src = reinterpret_cast<const char *>(src)
                    + s_off * src_d.data_type_size();
            char *__restrict _dst = reinterpret_cast<char *>(dst)
                    + s_off * dst_d.data_type_size();

            float sm = (use_scale ? (float)scale[c] : (float)1.0f)
                    / sqrt_variance;
            float sv = use_shift ? (float)shift[c] : (float)0;

            for (dim_t sp_block = 0; sp_block < n_sp_block; sp_block++) {
                const float *__restrict src_f32 {nullptr};
                if (src_dt != data_type::f32) {
                    float *tmp = cvt_scratch + sp_block_nelems * ithr;
                    for (dim_t sp = 0; sp < sp_block_nelems; ++sp)
                        tmp[sp] = io::load_float_value(
                                src_d.data_type(), _src, sp);
                    src_f32 = tmp;
                } else {
                    src_f32 = reinterpret_cast<const float *__restrict>(_src);
                }
                float *__restrict dst_f32 {nullptr};
                if (dst_dt != data_type::f32) {
                    dst_f32 = cvt_scratch + sp_block_nelems * ithr;
                } else {
                    dst_f32 = reinterpret_cast<float *__restrict>(_dst);
                }
                PRAGMA_OMP_SIMD()
                for (dim_t sp = 0; sp < sp_block_nelems; sp++) {
                    float s = src_f32[sp];
                    float gn_res = sm * (s - m) + sv;
                    dst_f32[sp] = gn_res * combined_scale;
                }
                if (dst_dt != data_type::f32) {
                    for (dim_t sp = 0; sp < sp_block_nelems; ++sp)
                        io::store_float_value(
                                dst_d.data_type(), dst_f32[sp], _dst, sp);
                }
                _src += sp_block_nelems * src_d.data_type_size();
                _dst += sp_block_nelems * dst_d.data_type_size();
            }
            if (sp_block_reminder) {
                const float *__restrict src_f32 {nullptr};
                if (src_dt != data_type::f32) {
                    float *tmp = cvt_scratch + sp_block_nelems * ithr;
                    for (dim_t sp = 0; sp < sp_block_reminder; ++sp)
                        tmp[sp] = io::load_float_value(
                                src_d.data_type(), _src, sp);
                    src_f32 = tmp;
                } else {
                    src_f32 = reinterpret_cast<const float *__restrict>(_src);
                }
                float *__restrict dst_f32 {nullptr};
                if (dst_dt != data_type::f32) {
                    dst_f32 = cvt_scratch + sp_block_nelems * ithr;
                } else {
                    dst_f32 = reinterpret_cast<float *__restrict>(_dst);
                }
                PRAGMA_OMP_SIMD()
                for (dim_t sp = 0; sp < sp_block_reminder; sp++) {
                    float s = src_f32[sp];
                    float gn_res = sm * (s - m) + sv;
                    dst_f32[sp] = gn_res * combined_scale;
                }
                if (dst_dt != data_type::f32) {
                    for (dim_t sp = 0; sp < sp_block_reminder; ++sp)
                        io::store_float_value(
                                dst_d.data_type(), dst_f32[sp], _dst, sp);
                }
            }
        }

        if (calculate_stats && save_stats) {
            mean[n * G + g] = m;
            variance[n * G + g] = v;
        }
    };

    parallel_nd_ext(nthr, N, G, kernel);
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
