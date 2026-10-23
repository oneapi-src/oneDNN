/******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
******************************************************************************/

#include <assert.h>
#include <math.h>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rv64/jit_uni_batch_normalization.hpp"
#include "cpu/rv64/jit_uni_batch_normalization_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

static inline void bn_fwd_kernel(const void *s_base, void *d_base, size_t len,
        const float *mean, const float *sm, const float *sv,
        data_type_t data_type, bool per_elem_params, bool with_relu) {
    jit_uni_batch_normalization_apply(s_base, d_base, static_cast<dim_t>(len),
            mean, sm, sv, data_type, per_elem_params, with_relu);
}

} // namespace

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper data_d(pd()->src_md());
    const auto dtsrc = pd()->src_md()->data_type;
    const int ndims = data_d.ndims();

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();

    const float eps = pd()->desc()->batch_norm_epsilon;
    const size_t data_size = types::data_type_size(dtsrc);

    void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const float *scale = pd()->use_scale()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SCALE)
            : nullptr;
    const float *shift = pd()->use_shift()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SHIFT)
            : nullptr;

    const bool with_relu = pd()->fused_relu_in_kernel()
            || pd()->attr()->post_ops_.find(primitive_kind::eltwise) != -1;

    auto off = [&](dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) -> size_t {
        switch (ndims) {
            case 3: return data_d.off(n, c, w);
            case 4: return data_d.off(n, c, h, w);
            case 5: return data_d.off(n, c, d, h, w);
            default: assert(!"unsupported ndims"); return dim_t(0);
        }
    };

    const bool channels_dense = data_d.blocking_desc().strides[1] == 1;

    // Mean/variance: either supplied by the user or computed. Computed
    // statistics are outputs for training and scratchpad for inference.
    const float *mean, *var;
    if (pd()->use_global_stats()) {
        mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
        var = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    } else {
        assert(utils::one_of(dtsrc, data_type::f32, data_type::f16)
                && !channels_dense);
        const dim_t SP = D * H * W;
        const dim_t count = N * SP;
        float *mean_out, *var_out;
        if (pd()->is_training()) {
            mean_out = CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
            var_out = CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
        } else {
            mean_out = ctx.get_scratchpad_grantor().template get<float>(
                    memory_tracking::names::key_bnorm_tmp_stats);
            var_out = mean_out + C;
        }
        const auto &mean_kernel
                = get_jit_uni_batch_normalization_fwd_stat_kernel(
                        dtsrc, /*calculate_variance=*/false);
        const auto &variance_kernel
                = get_jit_uni_batch_normalization_fwd_stat_kernel(
                        dtsrc, /*calculate_variance=*/true);

        parallel_nd(C, [&](dim_t c) {
            auto reduce
                    = [&](const jit_uni_batch_normalization_fwd_stat_kernel_t
                                      &kernel,
                              float reduction_mean, float &result) {
                for (dim_t n = 0; n < N; ++n) {
                    const size_t base = off(n, c, 0, 0, 0);
                    const jit_uni_batch_normalization_fwd_stat_kernel_t::
                            call_params_t p {static_cast<const char *>(src)
                                            + base * data_size,
                                    SP, &result, reduction_mean};
                    kernel(&p);
                }
            };

            float sum = 0.f;
            reduce(mean_kernel, 0.f, sum);
            mean_out[c] = sum / static_cast<float>(count);

            float squared_difference_sum = 0.f;
            reduce(variance_kernel, mean_out[c], squared_difference_sum);
            var_out[c] = squared_difference_sum / static_cast<float>(count);
        });
        mean = mean_out;
        var = var_out;
    }

    if (!channels_dense && pd()->spatial_dense(data_d)) {
        // abx data tag with contiguous spatial dims: vectorize over the D*H*W
        // row of each (n, c) so small spatial maps fill the vector. When there
        // are few rows, split each row into chunks so that every thread gets
        // work; chunks stay long enough to keep the vector unit busy.
        const dim_t SP = D * H * W;
        const dim_t min_chunk = 1024;
        const dim_t rows = N * C;
        const dim_t nchunks = nstl::max(dim_t(1),
                nstl::min(
                        utils::div_up(dim_t(4) * dnnl_get_max_threads(), rows),
                        SP / min_chunk));
        const dim_t chunk = utils::div_up(SP, nchunks);
        parallel_nd(N, C, nchunks, [&](dim_t n, dim_t c, dim_t k) {
            const dim_t sp_start = k * chunk;
            const dim_t len = nstl::min(chunk, SP - sp_start);
            if (len <= 0) return;
            const float inv_std = 1.0f / sqrtf(var[c] + eps);
            const float vscale = scale ? scale[c] : 1.0f;
            const float vshift = shift ? shift[c] : 0.0f;
            const float mean_b[1] = {mean[c]};
            const float sm_b[1] = {vscale * inv_std};
            const float sv_b[1] = {vshift};
            const size_t base_off = off(n, c, 0, 0, 0) + sp_start;
            const void *s_ptr = reinterpret_cast<const void *>(
                    reinterpret_cast<const char *>(src) + base_off * data_size);
            void *d_ptr = reinterpret_cast<void *>(
                    reinterpret_cast<char *>(dst) + base_off * data_size);
            bn_fwd_kernel(s_ptr, d_ptr, static_cast<size_t>(len), mean_b, sm_b,
                    sv_b, dtsrc, /*per_elem_params=*/false, with_relu);
        });
    } else if (!channels_dense) {
        // abx data tag: vectorize over W for fixed channel
        parallel_nd(C, N, D, H, [&](dim_t c, dim_t n, dim_t d, dim_t h) {
            const float vmean = mean[c];
            const float inv_std = 1.0f / sqrtf(var[c] + eps);
            const float vscale = scale ? scale[c] : 1.0f;
            const float vshift = shift ? shift[c] : 0.0f;
            const float sm = vscale * inv_std;
            const float sv = vshift;
            size_t base_off = off(n, c, d, h, 0);

            const void *s_ptr = reinterpret_cast<const void *>(
                    reinterpret_cast<const char *>(src) + base_off * data_size);
            void *d_ptr = reinterpret_cast<void *>(
                    reinterpret_cast<char *>(dst) + base_off * data_size);
            const float mean_b[1] = {vmean};
            const float sm_b[1] = {sm};
            const float sv_b[1] = {sv};
            bn_fwd_kernel(s_ptr, d_ptr, static_cast<size_t>(W), mean_b, sm_b,
                    sv_b, dtsrc, /*per_elem_params=*/false, with_relu);
        });
    } else {
        // axb data tag: vectorize across channels
        auto &grantor = ctx.get_scratchpad_grantor();
        float *sm_arr = grantor.template get<float>(
                memory_tracking::names::key_bnorm_tmp_mean);
        float *sv_arr = grantor.template get<float>(
                memory_tracking::names::key_bnorm_tmp_var);
        for (dim_t c = 0; c < C; ++c) {
            const float inv_std = 1.0f / sqrtf(var[c] + eps);
            const float vscale = scale ? scale[c] : 1.0f;
            const float vshift = shift ? shift[c] : 0.0f;
            sm_arr[static_cast<size_t>(c)] = vscale * inv_std;
            sv_arr[static_cast<size_t>(c)] = vshift;
        }

        parallel_nd(N, D, H, W, [&](dim_t n, dim_t d, dim_t h, dim_t w) {
            const size_t base_off = off(n, 0, d, h, w);
            const void *s_ptr = reinterpret_cast<const void *>(
                    reinterpret_cast<const char *>(src) + base_off * data_size);
            void *d_ptr = reinterpret_cast<void *>(
                    reinterpret_cast<char *>(dst) + base_off * data_size);
            bn_fwd_kernel(s_ptr, d_ptr, static_cast<size_t>(C), mean, sm_arr,
                    sv_arr, dtsrc, /*per_elem_params=*/true, with_relu);
        });
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::execute_backward(
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper data_d(pd()->src_md());
    const data_type_t data_type = pd()->src_md()->data_type;
    const size_t data_size = types::data_type_size(data_type);
    const int ndims = data_d.ndims();

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const dim_t M = N * D * H * W;
    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool calculate_diff_stats = !pd()->use_global_stats();

    const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    void *diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
    const float *mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
    const float *variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    const float *scale = pd()->use_scale()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SCALE)
            : nullptr;
    float *diff_scale_out = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SCALE);
    float *diff_shift_out = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SHIFT);

    auto &grantor = ctx.get_scratchpad_grantor();
    float *tmp = grantor.template get<float>(
            memory_tracking::names::key_bnorm_tmp_diff_ss);
    float *reduction = grantor.template get<float>(
            memory_tracking::names::key_bnorm_reduction);
    float *diff_scale = diff_scale_out ? diff_scale_out : tmp;
    float *diff_shift = diff_shift_out ? diff_shift_out : tmp + C;
    float *scale_mul = tmp + 2 * C;
    float *diff_scale_mul = tmp + 3 * C;
    float *diff_shift_add = tmp + 4 * C;

    auto off = [&](dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) -> size_t {
        switch (ndims) {
            case 3: return data_d.off(n, c, w);
            case 4: return data_d.off(n, c, h, w);
            case 5: return data_d.off(n, c, d, h, w);
            default: assert(!"unsupported ndims"); return dim_t(0);
        }
    };
    auto data_ptr = [&](const void *base, size_t elem_off) {
        return reinterpret_cast<const void *>(
                reinterpret_cast<const char *>(base) + elem_off * data_size);
    };
    auto mutable_data_ptr = [&](void *base, size_t elem_off) {
        return reinterpret_cast<void *>(
                reinterpret_cast<char *>(base) + elem_off * data_size);
    };

    const bool channels_dense = data_d.blocking_desc().strides[1] == 1;
    if (!channels_dense) {
        parallel_nd(C, [&](dim_t c) {
            diff_scale[c] = 0.0f;
            diff_shift[c] = 0.0f;
            for (dim_t n = 0; n < N; ++n) {
                for (dim_t d = 0; d < D; ++d) {
                    for (dim_t h = 0; h < H; ++h) {
                        const size_t base_off = off(n, c, d, h, 0);
                        jit_uni_batch_normalization_bwd_reduce(
                                data_ptr(src, base_off),
                                data_ptr(diff_dst, base_off), W, mean + c,
                                diff_scale + c, diff_shift + c, data_type,
                                /*per_elem_params=*/false);
                    }
                }
            }
        });
    } else {
        const dim_t rows = N * D * H * W;
        const int nthr = pd()->nthr();
        for (dim_t i = 0; i < 2 * C * nthr; ++i)
            reduction[i] = 0.0f;
        parallel(nthr, [&](int ithr, int nthr) {
            float *local_diff_scale = reduction + 2 * C * ithr;
            float *local_diff_shift = local_diff_scale + C;
            for (dim_t c = 0; c < C; ++c) {
                local_diff_scale[c] = 0.0f;
                local_diff_shift[c] = 0.0f;
            }

            dim_t start = 0, end = 0;
            balance211(rows, nthr, ithr, start, end);
            for (dim_t row = start; row < end; ++row) {
                dim_t q = row;
                const dim_t w = q % W;
                q /= W;
                const dim_t h = q % H;
                q /= H;
                const dim_t d = q % D;
                const dim_t n = q / D;
                const size_t base_off = off(n, 0, d, h, w);
                jit_uni_batch_normalization_bwd_reduce(data_ptr(src, base_off),
                        data_ptr(diff_dst, base_off), C, mean, local_diff_scale,
                        local_diff_shift, data_type,
                        /*per_elem_params=*/true);
            }
        });

        parallel_nd(C, [&](dim_t c) {
            float dg = 0.0f;
            float db = 0.0f;
            for (int ithr = 0; ithr < pd()->nthr(); ++ithr) {
                dg += reduction[2 * C * ithr + c];
                db += reduction[2 * C * ithr + C + c];
            }
            diff_scale[c] = dg;
            diff_shift[c] = db;
        });
    }

    parallel_nd(C, [&](dim_t c) {
        const float inv_std = 1.0f / sqrtf(variance[c] + eps);
        diff_scale[c] *= inv_std;
        scale_mul[c] = (scale ? scale[c] : 1.0f) * inv_std;
        diff_scale_mul[c] = calculate_diff_stats
                ? diff_scale[c] * inv_std / static_cast<float>(M)
                : 0.0f;
        diff_shift_add[c] = calculate_diff_stats
                ? diff_shift[c] / static_cast<float>(M)
                : 0.0f;
    });

    if (!channels_dense) {
        parallel_nd(C, N, D, H, [&](dim_t c, dim_t n, dim_t d, dim_t h) {
            const size_t base_off = off(n, c, d, h, 0);
            jit_uni_batch_normalization_bwd_apply(data_ptr(src, base_off),
                    data_ptr(diff_dst, base_off),
                    mutable_data_ptr(diff_src, base_off), W, mean + c,
                    scale_mul + c, diff_scale_mul + c, diff_shift_add + c,
                    data_type,
                    /*per_elem_params=*/false);
        });
    } else {
        parallel_nd(N, D, H, W, [&](dim_t n, dim_t d, dim_t h, dim_t w) {
            const size_t base_off = off(n, 0, d, h, w);
            jit_uni_batch_normalization_bwd_apply(data_ptr(src, base_off),
                    data_ptr(diff_dst, base_off),
                    mutable_data_ptr(diff_src, base_off), C, mean, scale_mul,
                    diff_scale_mul, diff_shift_add, data_type,
                    /*per_elem_params=*/true);
        });
    }

    return status::success;
}

template struct jit_uni_batch_normalization_fwd_t<v>;
template struct jit_uni_batch_normalization_fwd_t<zvfh>;
template struct jit_uni_batch_normalization_fwd_t<zvfbfwma>;
template struct jit_uni_batch_normalization_bwd_t<v>;
template struct jit_uni_batch_normalization_bwd_t<zvfh>;
template struct jit_uni_batch_normalization_bwd_t<zvfbfwma>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
