/******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 Intel Corporation
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

#ifndef CPU_RV64_RVV_SOFTMAX_HPP
#define CPU_RV64_RVV_SOFTMAX_HPP

#include <limits>
#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "cpu/cpu_softmax_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_rvv_softmax_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_softmax_conf_t {
    data_type_t data_type;
    dim_t inner_size;
    dim_t axis_size;
    dim_t outer_size;
    bool is_logsoftmax;
};

struct rvv_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;
        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa_, ""), rvv_softmax_fwd_t);

        rvv_softmax_conf_t rsp_;
        bool use_jit_ = false;
        bool use_inner_jit_ = false;
        bool use_axis_jit_ = false;
        bool pack_axis_ = false;
        bool buffered_axis_ = false;
        size_t buffered_axis_stride_ = 0;
        static constexpr dim_t inner_tile_size = 64;
        dim_t inner_tile_ = inner_tile_size;
        int inner_lmul_ = 4;
        // f32 uses isa v, f16 uses Zvfh (vfwcvt/vfncvt) and bf16 uses
        // Zvfbfmin (vfwcvtbf16/vfncvtbf16, tracked by the zvfbfwma bit);
        // drives the impl name.
        cpu_isa_t isa_ = v;

        status_t init(const engine_t *engine) {
            UNUSED(engine);

            VDISPATCH_SOFTMAX(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            rsp_.is_logsoftmax = is_logsoftmax();
            rsp_.data_type = src_md()->data_type;
            rsp_.inner_size = src_d.blocking_desc().strides[axis()];
            rsp_.axis_size = axis_size();
            rsp_.outer_size
                    = src_d.nelems(true) / (rsp_.inner_size * axis_size(true));

            const bool is_f16 = src_md()->data_type == data_type::f16;
            const bool is_bf16 = src_md()->data_type == data_type::bf16;
            const bool is_xf16 = is_f16 || is_bf16;
            VDISPATCH_SOFTMAX(utils::one_of(src_md()->data_type, data_type::f32,
                                      data_type::f16, data_type::bf16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_UNSUPPORTED_DT);
            // f16 runs the Zvfh widening/narrowing path, bf16 the Zvfbfmin one;
            // f32 stays on V. Set the ISA that drives the impl name (jit:rvv vs
            // jit:rvv_zvfh vs jit:rvv_zvfbfwma) before any mayiuse() rejection,
            // so a declined xf16 PD is not mislabeled jit:rvv.
            isa_ = is_f16 ? zvfh : is_bf16 ? zvfbfwma : v;
            // bf16 logsoftmax is left to the reference. The sum of exponentials
            // comes from the vectorized polynomial exp, so log(sum) lands within
            // about one f32 ulp of the reference's expf()-accumulated value.
            // logsoftmax then subtracts two similar magnitudes (src - max -
            // log_sum), which amplifies that difference; on bf16's 8-bit
            // significand it flips the rounding whenever the exact result sits
            // on a tie, and benchdnn compares xf16 softmax exactly against the
            // f32 reference under the default (strict) accumulation mode. The
            // same shapes pass with --attr-acc-mode=relaxed, and plain softmax
            // is unaffected because scaling by 1/sum preserves relative error
            // instead of amplifying it.
            VDISPATCH_SOFTMAX(!(is_bf16 && rsp_.is_logsoftmax),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "bf16 logsoftmax cannot meet the strict accumulation "
                    "contract with a vectorized exp");
            VDISPATCH_SOFTMAX(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
            if (is_xf16) {
                VDISPATCH_SOFTMAX(mayiuse(isa_), VERBOSE_UNSUPPORTED_ISA);
#if !(defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1)
                VDISPATCH_SOFTMAX(false, VERBOSE_UNSUPPORTED_ISA);
#endif
            }
            VDISPATCH_SOFTMAX(
                    platform::has_data_type_support(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    check_layouts(src_d, dst_d), VERBOSE_UNSUPPORTED_TAG);

#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
            use_jit_ = rsp_.data_type == data_type::f32;
            use_inner_jit_
                    = use_jit_ && rsp_.inner_size > 1 && src_d.is_dense();
#else
            use_jit_ = false;
#endif
            init_scratchpad();

            return status::success;
        }

        bool check_layouts(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &dst_d) const {
            if (!src_d.is_plain() || !dst_d.is_plain()) return false;
            if (!src_d.is_dense(true) || !dst_d.is_dense(true)) return false;
            if (!(src_d == dst_d)) return false;
            return true;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            const bool is_xf16 = utils::one_of(
                    rsp_.data_type, data_type::f16, data_type::bf16);
            const dim_t work_amount = rsp_.inner_size > 1
                    ? rsp_.outer_size * rsp_.inner_size
                    : rsp_.outer_size;
            nthr_ = rsp_.inner_size > 1 || is_xf16
                    ? static_cast<int>(nstl::min<dim_t>(dnnl_get_max_threads(),
                              nstl::max<dim_t>(work_amount, 1)))
                    : 1;
            if (use_inner_jit_) {
                // Give low-outer-count problems more independent jobs without
                // fragmenting every vector into tiny, cache-line-sharing tiles.
                const dim_t chunks
                        = utils::div_up<dim_t>(nthr_, rsp_.outer_size);
                const dim_t target = utils::div_up(rsp_.inner_size, chunks);
                inner_tile_ = nstl::min(inner_tile_size,
                        utils::rnd_up(nstl::max<dim_t>(target, 16), dim_t(16)));
                const dim_t lanes = nstl::max<dim_t>(
                        get_platform_vlen() / (8 * sizeof(float)), 1);
                const dim_t count = nstl::min(inner_tile_, rsp_.inner_size);
                inner_lmul_ = count <= lanes ? 1 : count <= 2 * lanes ? 2 : 4;
                // On short vectors, a long axis amortizes packing better than
                // iterating the axis with only a few independent inner lanes.
                const bool short_vector_columns = lanes <= 4
                        && rsp_.inner_size > lanes
                        && rsp_.inner_size < 2 * lanes && rsp_.axis_size >= 512;
                // Long reductions can fill axis vectors when a single-thread
                // inner-vector kernel leaves much of its register group idle.
                const bool sparse_serial_inner = nthr_ == 1 && lanes > 4
                        && count * 4 < 3 * lanes * inner_lmul_
                        // Large column strides lose cache-line reuse even
                        // when the inner-vector register group is sparse.
                        && static_cast<size_t>(rsp_.inner_size)
                                <= platform::get_cache_line_size()
                                        / sizeof(float)
                        && rsp_.axis_size >= 4 * lanes;
                // Narrow inner dimensions leave most lanes idle. Process
                // independent columns with an axis-vectorized kernel instead.
                // A wide vector with fewer than half its lanes occupied,
                // or too few independent rows to fill the thread team,
                // can instead process packed columns in parallel.
                pack_axis_
                        = (lanes > 8
                                  && (sparse_serial_inner
                                          || rsp_.inner_size <= lanes / 2
                                          || (rsp_.outer_size < nthr_
                                                  && rsp_.inner_size <= lanes)))
                        || (short_vector_columns && nthr_ == 1);
                // Axis-strided passes revisit the same cache lines for each
                // column. Beyond the per-core L1 estimate, retain inner-vector
                // reuse when enough outer rows exist, or buffer column jobs
                // when extra parallelism is needed.
                const size_t l1_elements
                        = platform::get_per_core_cache_size(1) / sizeof(float);
                const bool long_strided_axis = !pack_axis_
                        // Very narrow columns cannot amortize whole-tensor
                        // packing; reserve it for near-full inner vectors.
                        && nthr_ > 1 && rsp_.inner_size * 4 >= 3 * lanes
                        && rsp_.inner_size <= nstl::min<dim_t>(lanes, 8)
                        && static_cast<size_t>(rsp_.axis_size) > l1_elements
                                        / static_cast<size_t>(rsp_.inner_size);
                // Expose column jobs when both outer parallelism and m2 lane
                // utilization are low. Keep every column's result until all
                // input reads finish; then write contiguous destination rows.
                buffered_axis_ = (lanes == 8 && rsp_.outer_size <= nthr_ / 4
                                         && rsp_.inner_size > lanes
                                         && rsp_.inner_size <= 3 * lanes / 2
                                         && rsp_.axis_size >= 128)
                        || (short_vector_columns && nthr_ > 1
                                && rsp_.outer_size <= nthr_ / 4)
                        || (long_strided_axis
                                && (rsp_.inner_size < lanes
                                        || rsp_.outer_size
                                                < utils::div_up(nthr_, 2)));
                if (buffered_axis_) {
                    const size_t line
                            = platform::get_cache_line_size() / sizeof(float);
                    const size_t columns = static_cast<size_t>(work_amount);
                    const size_t max_stride = std::numeric_limits<size_t>::max()
                            / sizeof(float) / columns;
                    const size_t axis = static_cast<size_t>(axis_size(true));
                    // Odd cache-line strides distribute columns across cache
                    // sets when the final pass reads contiguous output rows.
                    buffered_axis_ = max_stride >= 3 * line
                            && axis <= max_stride - 3 * line;
                    if (buffered_axis_)
                        buffered_axis_stride_
                                = utils::rnd_up(axis, 2 * line) + line;
                }
                pack_axis_ = pack_axis_ || buffered_axis_;
                use_axis_jit_ = pack_axis_ || sparse_serial_inner
                        || (!long_strided_axis
                                && rsp_.inner_size
                                        <= nstl::min<dim_t>(lanes, 8));
                if (use_axis_jit_) {
                    inner_tile_ = 1;
                    inner_lmul_ = 4;
                }
            }
            const size_t dt_size = types::data_type_size(rsp_.data_type);
            if (rsp_.inner_size > 1
                    && (!use_inner_jit_ || pack_axis_
                            || (use_axis_jit_ && !rsp_.is_logsoftmax))) {
                scratchpad.template book<char>(
                        memory_tracking::names::key_softmax_interim_store,
                        (buffered_axis_ ? buffered_axis_stride_
                                        : static_cast<size_t>(axis_size(true)))
                                * dt_size
                                * static_cast<size_t>(buffered_axis_
                                                ? rsp_.outer_size
                                                        * rsp_.inner_size
                                                : nthr_));
            }
            if (is_xf16) {
                scratchpad.template book<float>(
                        memory_tracking::names::key_softmax_reduction,
                        static_cast<size_t>(axis_size(true))
                                * static_cast<size_t>(nthr_));
            }
            if (use_inner_jit_) {
                scratchpad.template book<float>(
                        memory_tracking::names::key_softmax_reduction,
                        2 * inner_tile_size * static_cast<size_t>(nthr_));
            }
        }

        int nthr_ = 0;
    };

    rvv_softmax_fwd_t(const pd_t *apd);
    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_rvv_softmax_affine_kernel_t> affine_kernel_;
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const jit_rvv_softmax_xf16_strided_kernel_t *gather_kernel_ = nullptr;
    const jit_rvv_softmax_xf16_strided_kernel_t *scatter_kernel_ = nullptr;
    const jit_rvv_softmax_inner_kernel_t *inner_reduce_kernel_ = nullptr;
    const jit_rvv_softmax_inner_kernel_t *inner_apply_kernel_ = nullptr;
#endif
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_SOFTMAX_HPP
