/******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 SpacemiT Corporation
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

#include <math.h>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/nstl.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_rvv_softmax_kernel.hpp"
#include "cpu/rv64/rvv_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

rvv_softmax_fwd_t::rvv_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {
    if (pd()->use_jit_ && !pd()->use_inner_jit_) {
        affine_kernel_.reset(new jit_rvv_softmax_affine_kernel_t());
    }
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    // Resolve the process-wide kernels during primitive creation so the
    // strided hot path can bypass the function-local static guards.
    if (utils::one_of(pd()->rsp_.data_type, data_type::f16, data_type::bf16)
            && pd()->rsp_.inner_size > 1) {
        gather_kernel_ = &get_xf16_strided_kernel<true>();
        scatter_kernel_ = &get_xf16_strided_kernel<false>();
    }
#endif
}

status_t rvv_softmax_fwd_t::init(engine_t *engine) {
    UNUSED(engine);
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    if (pd()->use_inner_jit_) {
        const bool inf_as_zero
                = pd()->alg_kind() == alg_kind::softmax_accurate_inf_as_zero;
        const auto &reduce = get_softmax_inner_kernel(false,
                pd()->rsp_.is_logsoftmax, inf_as_zero, pd()->inner_lmul_,
                pd()->use_axis_jit_, pd()->pack_axis_);
        CHECK(reduce.status());
        const auto &apply = get_softmax_inner_kernel(true,
                pd()->rsp_.is_logsoftmax, inf_as_zero, pd()->inner_lmul_,
                pd()->use_axis_jit_, pd()->pack_axis_);
        CHECK(apply.status());
        inner_reduce_kernel_ = &reduce;
        inner_apply_kernel_ = &apply;
    }
#endif
    return status::success;
}

namespace {

// f32 compute kernel. jit_exp selects the vectorized exp/sub/sum kernel for the
// reduction; the caller gates it on axis length and stride so that very short
// or memory-bound (large-stride) reductions keep the scalar exp path.
void compute_softmax_f32_rvv(const float *src, float *dst, dim_t len,
        bool is_logsoftmax, bool is_softmax_inf_as_zero,
        const jit_rvv_softmax_affine_kernel_t *affine_kernel, bool jit_exp,
        bool jit_reduce_max) {
    float max_val = -INFINITY;
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    if (jit_reduce_max) {
        jit_rvv_softmax_f32_reduce_max(src, len, &max_val);
    } else
#else
    MAYBE_UNUSED(jit_reduce_max);
#endif
    {
        for (dim_t i = 0; i < len; ++i)
            max_val = src[i] > max_val ? src[i] : max_val;
    }

    if (is_logsoftmax) {
        float sum_exp = 0.f;
        if (jit_exp) {
            jit_rvv_softmax_f32_exp_sub_sum(
                    src, dst, len, max_val, &sum_exp, false);
        } else {
            for (dim_t i = 0; i < len; ++i) {
                sum_exp += expf(src[i] - max_val);
            }
        }
        const float log_sum = logf(sum_exp);

        if (affine_kernel) {
            jit_rvv_softmax_affine_kernel_t::call_params_t p;
            p.src = src;
            p.dst = dst;
            p.len = len;
            p.sub = max_val + log_sum;
            p.mul = 1.f;
            (*affine_kernel)(&p);
        } else {
            for (dim_t i = 0; i < len; ++i)
                dst[i] = src[i] - max_val - log_sum;
        }
    } else {
        float sum_exp = 0.f;
        const bool all_minus_inf
                = is_softmax_inf_as_zero && (max_val == -INFINITY);
        if (jit_exp) {
            if (all_minus_inf) {
                for (dim_t i = 0; i < len; ++i)
                    dst[i] = 0.f;
            } else {
                jit_rvv_softmax_f32_exp_sub_sum(
                        src, dst, len, max_val, &sum_exp, true);
            }
        } else {
            for (dim_t i = 0; i < len; ++i) {
                float e = all_minus_inf ? 0.f : expf(src[i] - max_val);
                dst[i] = e;
                sum_exp += e;
            }
        }

        const float inv_sum = sum_exp ? (1.0f / sum_exp) : 1.0f;
        if (affine_kernel) {
            jit_rvv_softmax_affine_kernel_t::call_params_t p;
            p.src = dst;
            p.dst = dst;
            p.len = len;
            p.sub = 0.f;
            p.mul = inv_sum;
            (*affine_kernel)(&p);
        } else {
            for (dim_t i = 0; i < len; ++i)
                dst[i] *= inv_sum;
        }
    }
}

#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
template <typename T>
void compute_softmax_xf16_scalar(
        const T *src, T *dst, dim_t len, bool is_logsoftmax, float max_val) {
    float sum_exp = 0.f;

    for (dim_t i = 0; i < len; ++i) {
        const float D = (float)src[i] - max_val;
        sum_exp += expf(D);
    }

    if (is_logsoftmax) {
        const float log_sum = logf(sum_exp);
        for (dim_t i = 0; i < len; ++i)
            dst[i] = (float)src[i] - max_val - log_sum;
    } else {
        const float inv_sum = sum_exp ? (1.0f / sum_exp) : 1.0f;
        for (dim_t i = 0; i < len; ++i)
            dst[i] = expf((float)src[i] - max_val) * inv_sum;
    }
}

// 16-bit float compute kernel, shared by f16 (T = float16_t) and bf16
// (T = bfloat16_t). Both compute at f32; only the convert instructions the
// kernels emit and the NaN/lowest constants differ.
template <typename T, data_type_t dt>
void compute_softmax_xf16_rvv(const T *src, T *dst, dim_t len,
        bool is_logsoftmax, bool is_softmax_inf_as_zero, float *scratchpad) {

    constexpr dim_t reduce_max_jit_min_len = 16;
    // A 16-bit float is NaN iff (raw & 0x7fff) exceeds its +inf encoding.
    constexpr uint16_t nan_thresh = dt == data_type::bf16 ? 0x7f80u : 0x7c00u;
    float max_val = (float)nstl::numeric_limits<T>::lowest();
    bool has_nan = false;
    if (len >= reduce_max_jit_min_len) {
        // The vectorized pass widens to f32 and performs a vector max
        // reduction. Its seed and NaN semantics reproduce the scalar loop.
        uint32_t jit_has_nan = 0;
        jit_rvv_softmax_xf16_reduce_max(dt, src, len, &max_val, &jit_has_nan);
        has_nan = jit_has_nan;
    } else {
        const uint16_t *raw = reinterpret_cast<const uint16_t *>(src);
        for (dim_t i = 0; i < len; ++i) {
            has_nan = has_nan || ((raw[i] & 0x7fffu) > nan_thresh);
            const float val = (float)src[i];
            if (val > max_val) max_val = val;
        }
    }
    const bool max_is_pos_inf = max_val == INFINITY;

    if (len == 1 && isfinite((float)src[0])) {
        dst[0] = is_logsoftmax ? 0.f : 1.f;
        return;
    }

    if (has_nan || max_is_pos_inf) {
        compute_softmax_xf16_scalar(src, dst, len, is_logsoftmax, max_val);
        return;
    }

    if (is_logsoftmax) {
        float sum_exp = 0.f;

        jit_rvv_softmax_xf16_exp_sub_sum(
                dt, src, scratchpad, len, max_val, &sum_exp);
        const float log_sum = logf(sum_exp);

        const float sub = max_val + log_sum;
        jit_rvv_softmax_xf16_affine_from_xf16(dt, src, dst, len, sub, 1.0f);
    } else {
        float sum_exp = 0.f;
        const bool all_minus_inf
                = is_softmax_inf_as_zero && (max_val == -INFINITY);

        if (all_minus_inf) {
            for (dim_t i = 0; i < len; ++i)
                scratchpad[i] = 0.f;
        } else {
            jit_rvv_softmax_xf16_exp_sub_sum(
                    dt, src, scratchpad, len, max_val, &sum_exp);
        }
        const float inv_sum = sum_exp ? (1.0f / sum_exp) : 1.0f;

        jit_rvv_softmax_xf16_affine_from_f32(
                dt, scratchpad, dst, len, 0.0f, inv_sum);
    }
}

// Driver for the 16-bit float path: contiguous axis runs the compute kernel in
// place; a strided axis gathers into a packed buffer first. `scratch` is the
// packed xf16 buffer (only used when inner_size > 1), `reduction` the per-thread
// f32 exp buffer.
template <typename T, data_type_t dt>
void execute_xf16(const void *src, void *dst, const rvv_softmax_conf_t &rsp,
        dim_t outer_stride, int nthr, bool is_softmax_inf_as_zero,
        float *reduction, char *scratch,
        const jit_rvv_softmax_xf16_strided_kernel_t *gather_kernel,
        const jit_rvv_softmax_xf16_strided_kernel_t *scatter_kernel) {
    const T *src_p = static_cast<const T *>(src);
    T *dst_p = static_cast<T *>(dst);

    if (rsp.inner_size == 1) {
        parallel(nthr, [&](int ithr, int nthr) {
            float *tmp = reduction
                    + static_cast<size_t>(ithr)
                            * static_cast<size_t>(rsp.axis_size);
            dim_t start {0}, end {0};
            balance211(rsp.outer_size, nthr, ithr, start, end);
            for (dim_t outer = start; outer < end; ++outer) {
                const dim_t base = outer * outer_stride;
                compute_softmax_xf16_rvv<T, dt>(src_p + base, dst_p + base,
                        rsp.axis_size, rsp.is_logsoftmax,
                        is_softmax_inf_as_zero, tmp);
            }
        });
        return;
    }

    parallel(nthr, [&](int ithr, int nthr) {
        auto *tmp = reinterpret_cast<T *>(scratch)
                + static_cast<size_t>(ithr)
                        * static_cast<size_t>(rsp.axis_size);
        float *reduction_tmp = reduction
                + static_cast<size_t>(ithr)
                        * static_cast<size_t>(rsp.axis_size);

        const dim_t work_amount = rsp.outer_size * rsp.inner_size;
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        const dim_t stride_bytes = rsp.inner_size * sizeof(T);

        for (dim_t idx = start; idx < end; ++idx) {
            const dim_t outer = idx / rsp.inner_size;
            const dim_t i = idx % rsp.inner_size;
            const dim_t base = outer * outer_stride + i;

            jit_rvv_softmax_xf16_strided_kernel_t::call_params_t gp {
                    src_p + base, tmp, rsp.axis_size, stride_bytes};
            (*gather_kernel)(&gp);

            compute_softmax_xf16_rvv<T, dt>(tmp, tmp, rsp.axis_size,
                    rsp.is_logsoftmax, is_softmax_inf_as_zero, reduction_tmp);

            jit_rvv_softmax_xf16_strided_kernel_t::call_params_t sp {
                    tmp, dst_p + base, rsp.axis_size, stride_bytes};
            (*scatter_kernel)(&sp);
        }
    });
}
#endif

} // namespace

status_t rvv_softmax_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const auto &rsp = pd()->rsp_;
    const bool is_softmax_inf_as_zero
            = pd()->alg_kind() == alg_kind::softmax_accurate_inf_as_zero;

    switch (rsp.data_type) {
        case data_type::f32: {
            const float *src_f32 = static_cast<const float *>(src);
            float *dst_f32 = static_cast<float *>(dst);

            const dim_t outer_stride = pd()->axis_size(true) * rsp.inner_size;
            const int nthr = pd()->nthr_;

#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
            if (pd()->use_inner_jit_) {
                src_f32 += pd()->src_md()->offset0;
                dst_f32 += pd()->dst_md()->offset0;
                const dim_t tile = pd()->inner_tile_;
                const dim_t blocks = utils::div_up(rsp.inner_size, tile);
                const dim_t work = rsp.outer_size * blocks;
                auto scratch = ctx.get_scratchpad_grantor().template get<float>(
                        memory_tracking::names::key_softmax_reduction);
                auto interim = (pd()->pack_axis_
                                       || (pd()->use_axis_jit_
                                               && !rsp.is_logsoftmax))
                        ? ctx.get_scratchpad_grantor().template get<char>(
                                  memory_tracking::names::
                                          key_softmax_interim_store)
                        : nullptr;
                const int active_nthr
                        = static_cast<int>(nstl::min<dim_t>(nthr, work));
                parallel(active_nthr, [&](int ithr, int nthr) {
                    // Keep the reserved per-thread spacing when a small tile
                    // is selected, avoiding shared scratch cache lines.
                    float *maximum = scratch + ithr * 2 * pd_t::inner_tile_size;
                    float *sum = maximum + tile;
                    dim_t begin, end;
                    balance211(work, nthr, ithr, begin, end);
                    for (dim_t job = begin; job < end; ++job) {
                        const dim_t outer = job / blocks;
                        const dim_t inner = (job % blocks) * tile;
                        const dim_t count
                                = nstl::min(tile, rsp.inner_size - inner);
                        const dim_t base = outer * outer_stride + inner;
                        jit_rvv_softmax_inner_kernel_t::call_params_t p {
                                src_f32 + base, dst_f32 + base, rsp.axis_size,
                                rsp.inner_size
                                        * static_cast<dim_t>(sizeof(float)),
                                count, maximum, sum};
                        float *tmp = interim
                                ? reinterpret_cast<float *>(interim)
                                        + static_cast<size_t>(
                                                  pd()->buffered_axis_ ? job
                                                                       : ithr)
                                                * (pd()->buffered_axis_
                                                                ? pd()->buffered_axis_stride_
                                                                : pd()->axis_size(
                                                                          true))
                                : nullptr;
                        if (pd()->pack_axis_) {
                            for (dim_t a = 0; a < rsp.axis_size; ++a)
                                tmp[a] = src_f32[base + a * rsp.inner_size];
                            p.src = tmp;
                            p.dst = tmp;
                            p.stride_bytes = sizeof(float);
                        } else if (tmp) {
                            // Keep intermediate exponentials private to the
                            // thread. Adjacent columns may share cache lines
                            // in the destination, which is written only once.
                            p.dst = tmp;
                        }
                        (*inner_reduce_kernel_)(&p);
                        if (rsp.is_logsoftmax) {
                            for (dim_t c = 0; c < count; ++c)
                                sum[c] = logf(sum[c]);
                        } else {
                            p.src = p.dst;
                            if (!pd()->pack_axis_) p.dst = dst_f32 + base;
                        }
                        (*inner_apply_kernel_)(&p);
                        if (pd()->pack_axis_ && !pd()->buffered_axis_) {
                            for (dim_t a = 0; a < rsp.axis_size; ++a)
                                dst_f32[base + a * rsp.inner_size] = tmp[a];
                        }
                    }
                });
                if (pd()->buffered_axis_) {
                    // The completed parallel region separates all source reads
                    // from destination writes, including in-place execution.
                    // Assign rows, not columns, so threads write disjoint ranges.
                    const auto *values
                            = reinterpret_cast<const float *>(interim);
                    const size_t buffer_stride = pd()->buffered_axis_stride_;
                    parallel(active_nthr, [&](int ithr, int nthr) {
                        for_nd(ithr, nthr, rsp.outer_size, rsp.axis_size,
                                [&](dim_t outer, dim_t axis) {
                            for (dim_t inner = 0; inner < rsp.inner_size;
                                    ++inner) {
                                const dim_t column
                                        = outer * rsp.inner_size + inner;
                                dst_f32[outer * outer_stride
                                        + axis * rsp.inner_size + inner]
                                        = values[column * buffer_stride + axis];
                            }
                        });
                    });
                }
                break;
            }
#endif

            // Use the vectorized exp kernel only when it pays off: the axis
            // must fill at least one LMUL=4 e32 vector (16 elements at VLEN=128;
            // on larger VLEN the kernel still runs a correct partial vector),
            // and a strided axis must have a moderate stride so the reduction is
            // not dominated by the gather/scatter memory traffic.
            constexpr dim_t exp_jit_min_len = 16;
            constexpr dim_t exp_jit_max_inner = 8192;
            const bool jit_exp = affine_kernel_.get()
                    && rsp.axis_size >= exp_jit_min_len
                    && rsp.inner_size <= exp_jit_max_inner;

            // Stage 1 max reduction. Measurements show a stable RVV benefit
            // from 15 elements on VLEN=256, while the tested VLEN=128 system
            // regresses even for long reductions. Query VLEN once per
            // execution, not once per row in the hot loop. The cutoff is not
            // VLMAX: a reduction need not fill an LMUL=4 group to be
            // profitable.
            const bool jit_reduce_max
                    = get_platform_vlen() >= 256 && rsp.axis_size >= 15;

            if (rsp.inner_size == 1) {
                parallel_nd(rsp.outer_size, [&](dim_t outer) {
                    const dim_t base = outer * outer_stride;
                    compute_softmax_f32_rvv(src_f32 + base, dst_f32 + base,
                            rsp.axis_size, rsp.is_logsoftmax,
                            is_softmax_inf_as_zero, affine_kernel_.get(),
                            jit_exp, jit_reduce_max);
                });
            } else {
                auto scratch = ctx.get_scratchpad_grantor().template get<char>(
                        memory_tracking::names::key_softmax_interim_store);

                parallel(nthr, [&](int ithr, int nthr) {
                    float *tmp = reinterpret_cast<float *>(scratch)
                            + static_cast<size_t>(ithr)
                                    * static_cast<size_t>(rsp.axis_size);

                    const dim_t work_amount = rsp.outer_size * rsp.inner_size;
                    dim_t start {0}, end {0};
                    balance211(work_amount, nthr, ithr, start, end);
                    for (dim_t idx = start; idx < end; ++idx) {
                        const dim_t outer = idx / rsp.inner_size;
                        const dim_t i = idx % rsp.inner_size;
                        const dim_t base = outer * outer_stride + i;

                        // gather -> tmp
                        for (dim_t a = 0; a < rsp.axis_size; ++a)
                            tmp[a] = src_f32[base + a * rsp.inner_size];

                        // contiguous kernel (in-place)
                        compute_softmax_f32_rvv(tmp, tmp, rsp.axis_size,
                                rsp.is_logsoftmax, is_softmax_inf_as_zero,
                                affine_kernel_.get(), jit_exp, jit_reduce_max);

                        // write back
                        for (dim_t a = 0; a < rsp.axis_size; ++a)
                            dst_f32[base + a * rsp.inner_size] = tmp[a];
                    }
                });
            }
        } break;
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
        case data_type::f16:
        case data_type::bf16: {
            const dim_t outer_stride = pd()->axis_size(true) * rsp.inner_size;
            const int nthr = pd()->nthr_;
            auto reduction = ctx.get_scratchpad_grantor().template get<float>(
                    memory_tracking::names::key_softmax_reduction);
            char *scratch = rsp.inner_size > 1
                    ? ctx.get_scratchpad_grantor().template get<char>(
                              memory_tracking::names::key_softmax_interim_store)
                    : nullptr;

            if (rsp.data_type == data_type::bf16)
                execute_xf16<dnnl::impl::bfloat16_t, data_type::bf16>(src, dst,
                        rsp, outer_stride, nthr, is_softmax_inf_as_zero,
                        reduction, scratch, gather_kernel_, scatter_kernel_);
            else
                execute_xf16<dnnl::impl::float16_t, data_type::f16>(src, dst,
                        rsp, outer_stride, nthr, is_softmax_inf_as_zero,
                        reduction, scratch, gather_kernel_, scatter_kernel_);
        } break;
#endif
        default: return status::unimplemented;
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
