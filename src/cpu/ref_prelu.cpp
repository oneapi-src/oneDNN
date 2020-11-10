/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/ref_prelu.hpp"
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::math;
using namespace memory_tracking;

static dim_t offset(const memory_desc_wrapper &mem, dim_t n, dim_t c, dim_t d,
        dim_t h, dim_t w) {
    const int ndims = mem.ndims();
    switch (ndims) {
        case 3: return mem.off(n, c, w);
        case 4: return mem.off(n, c, h, w);
        case 5: return mem.off(n, c, d, h, w);
        default: assert(!"Unsupported ndims count");
    }
    return -1;
}

static dim_t weights_offset(broadcasting_strategy_t bcast_type,
        const memory_desc_wrapper mem, dim_t n, dim_t c, dim_t d, dim_t h,
        dim_t w) {
    switch (bcast_type) {
        case broadcasting_strategy_t::no_broadcast:
            return offset(mem, n, c, d, h, w);
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial:
            return offset(mem, 0, c, 0, 0, 0);
        case broadcasting_strategy_t::scalar: return offset(mem, 0, 0, 0, 0, 0);
        default: assert(!"unsupported broadcast type");
    }
    return -1;
}

template <data_type_t d_type>
void ref_prelu_fwd_t<d_type>::execute_forward(const exec_ctx_t &ctx) const {
    using data_t = typename prec_traits<d_type>::type;

    if (pd()->has_zero_dim_memory()) return;

    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto bcast_type = dnnl::impl::get_rhs_arg_broadcasting_strategy(
            *weights_d.md_, data_d);

    const dim_t N = pd()->N();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();

    const dim_t work_amount = data_d.nelems();
    parallel(0, [&](std::size_t ithr, std::size_t nthr) {
        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dim_t n {0}, c {0}, d {0}, h {0}, w {0};

        balance211(work_amount, nthr, ithr, start, end);
        utils::nd_iterator_init(start, n, N, c, C, d, D, h, H, w, W);

        for (dim_t iwork = start; iwork < end; ++iwork) {
            const auto data_off = offset(data_d, n, c, d, h, w);
            const auto weight_off
                    = weights_offset(bcast_type, weights_d, n, c, d, h, w);
            float res = relu_fwd(src[data_off], weights[weight_off]);
            dst[data_off] = cpu::saturate_and_round<data_t>(res);
            utils::nd_iterator_step(n, N, c, C, d, D, h, H, w, W);
        }
    });
}

static float reduce(float *mem, dim_t size) {
    bool tail = size % 2;
    const auto reduce_iteration = [&](float *mem) {
        const auto div_res = std::div(size, (dim_t)2);
        tail = div_res.rem;
        size = div_res.quot;
        if (!tail && !size) {
            mem[0] = 0;
            return;
        }
        dim_t i {0}, off {0};
        if (tail) {
            if (size) mem[0] += mem[1 + off] + mem[2 + off];
            ++off;
            ++i;
        }
        for (; i < size; i++) {
            mem[i] = mem[2 * i + off] + mem[(2 * i + 1) + off];
        }
    };
    while (size > 1) {
        reduce_iteration(mem);
    }
    return mem[0];
}

void set_reduction_buffers(
        const dim_t work_amount, dim_t &group_size, dim_t &buf_size) {
    float sqrt = std::sqrt(work_amount);
    group_size = std::ceil(sqrt);
    buf_size = std::floor(sqrt);
    if (group_size * buf_size < work_amount) group_size++;
}

dim_t get_scalar_scratchpad_offset(const std::size_t ithr,
        const std::size_t nthr, const dim_t work_amount) {
    dim_t offset {0}, group_size, buf_size;
    for (std::size_t i = 0; i < ithr; i++) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, i, start, end);
        const dim_t workload = end - start;
        set_reduction_buffers(workload, group_size, buf_size);
        offset += buf_size;
        offset += group_size;
    }
    return offset;
}

template <typename data_t>
static float ker(const data_t *src, const data_t *weights,
        const data_t *diff_dst, data_t *diff_src,
        broadcasting_strategy_t bcast_type, dim_t data_off, dim_t weight_off,
        dim_t diff_data_off) {

    const float diff_src_res = relu_bwd_use_dst(
            diff_dst[diff_data_off], src[data_off], weights[weight_off]);
    diff_src[diff_data_off] = cpu::saturate_and_round<data_t>(diff_src_res);
    const float diff_weight_res
            = src[data_off] > 0 ? 0 : (diff_dst[diff_data_off] * src[data_off]);
    return diff_weight_res;
}

template <data_type_t d_type>
void ref_prelu_bwd_t<d_type>::calculate_scalar(const data_t *src,
        const data_t *weights, data_t *diff_weights, const data_t *diff_dst,
        data_t *diff_src, float *scratchpad_buf) const {

    const auto bcast_type = broadcasting_strategy_t::scalar;

    const memory_desc_wrapper data_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_data_d(pd()->diff_dst_md(0));

    const dim_t N = pd()->N();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();

    const int nscr = nstl::min(dnnl_get_max_threads(), (int)data_d.nelems());
    std::vector<float> buf_nthr_partial_results(dnnl_get_max_threads());

    const dim_t work_amount = data_d.nelems();
    parallel(0, [&](std::size_t ithr, std::size_t nthr) {
        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dim_t n {0}, c {0}, d {0}, h {0}, w {0};

        balance211(work_amount, nthr, ithr, start, end);
        const dim_t workload = end - start;

        utils::nd_iterator_init(start, n, N, c, C, d, D, h, H, w, W);

        dim_t group_size, buf_size;
        set_reduction_buffers(workload, group_size, buf_size);

        const dim_t scratchpad_offset
                = get_scalar_scratchpad_offset(ithr, nthr, work_amount);
        auto *buf = &scratchpad_buf[scratchpad_offset];
        auto *buf_lvl2 = &scratchpad_buf[scratchpad_offset + buf_size];

        dim_t off {0}, group {0}, data_size {buf_size};
        for (dim_t iwork = start; iwork < end; ++iwork) {
            const auto data_off = offset(data_d, n, c, d, h, w);
            const auto weight_off
                    = weights_offset(bcast_type, weights_d, n, c, d, h, w);
            const auto diff_data_off = offset(diff_data_d, n, c, d, h, w);
            buf[off] = ker(src, weights, diff_dst, diff_src, bcast_type,
                    data_off, weight_off, diff_data_off);
            if (++off == data_size) {
                buf_lvl2[group++] = reduce(buf, off);
                off = 0;
                data_size = ((group + 1) * buf_size <= workload)
                        ? buf_size
                        : workload - (group * buf_size);
            }
            utils::nd_iterator_step(n, N, c, C, d, D, h, H, w, W);
        }
        buf_nthr_partial_results[ithr] = reduce(buf_lvl2, group_size);
    });
    diff_weights[0] = cpu::saturate_and_round<data_t>(
            reduce(&buf_nthr_partial_results[0], nscr));
}

template <data_type_t d_type>
void ref_prelu_bwd_t<d_type>::calculate_no_broadcast(const data_t *src,
        const data_t *weights, data_t *diff_weights, const data_t *diff_dst,
        data_t *diff_src, float *scratchpad_buf) const {

    const auto bcast_type = broadcasting_strategy_t::no_broadcast;

    const dim_t N = pd()->N();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();

    const memory_desc_wrapper data_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_data_d(pd()->diff_dst_md(0));

    const dim_t work_amount = data_d.nelems();
    parallel(0, [&](std::size_t ithr, std::size_t nthr) {
        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dim_t n {0}, c {0}, d {0}, h {0}, w {0};

        balance211(work_amount, nthr, ithr, start, end);
        utils::nd_iterator_init(start, n, N, c, C, d, D, h, H, w, W);

        for (dim_t iwork = start; iwork < end; ++iwork) {
            const auto diff_weight_off
                    = weights_offset(bcast_type, weights_d, n, c, d, h, w);
            const auto data_off = offset(data_d, n, c, d, h, w);
            const auto weight_off
                    = weights_offset(bcast_type, weights_d, n, c, d, h, w);
            const auto diff_data_off = offset(diff_data_d, n, c, d, h, w);
            auto res = ker(src, weights, diff_dst, diff_src, bcast_type,
                    data_off, weight_off, diff_data_off);
            diff_weights[diff_weight_off]
                    = cpu::saturate_and_round<data_t>(res);
            utils::nd_iterator_step(n, N, c, C, d, D, h, H, w, W);
        }
    });
}

template <data_type_t d_type>
void ref_prelu_bwd_t<d_type>::calculate_per_oc(const data_t *src,
        const data_t *weights, data_t *diff_weights, const data_t *diff_dst,
        data_t *diff_src, float *scratchpad_buf) const {

    const auto bcast_type = broadcasting_strategy_t::per_oc;

    const dim_t N = pd()->N();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();

    const memory_desc_wrapper data_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_data_d(pd()->diff_dst_md(0));

    const dim_t work_amount = C;
    parallel(0, [&](std::size_t ithr, std::size_t nthr) {
        if ((dim_t)ithr >= work_amount) return;
        dim_t start {0}, end {0}, c {0};
        balance211(work_amount, nthr, ithr, start, end);
        utils::nd_iterator_init(start, c, C);
        const dim_t workload = N * D * H * W;
        dim_t group_size, buf_size;
        set_reduction_buffers(workload, group_size, buf_size);
        dim_t scratchpad_offset = (buf_size + group_size) * ithr;
        auto *buf = &scratchpad_buf[scratchpad_offset];
        auto *buf_lvl2 = &scratchpad_buf[scratchpad_offset + buf_size];
        dim_t off {0}, group {0}, data_size {buf_size};
        for (dim_t iwork = start; iwork < end; ++iwork) {
            for_(dim_t n = 0; n < N; ++n)
            for_(dim_t d = 0; d < D; ++d)
            for_(dim_t h = 0; h < H; ++h)
            for (dim_t w = 0; w < W; ++w) {
                const auto data_off = offset(data_d, n, c, d, h, w);
                const auto weight_off
                        = weights_offset(bcast_type, weights_d, n, c, d, h, w);
                const auto diff_data_off = offset(diff_data_d, n, c, d, h, w);
                buf[off] = ker(src, weights, diff_dst, diff_src, bcast_type,
                        data_off, weight_off, diff_data_off);
                if (++off == data_size) {
                    buf_lvl2[group++] = reduce(buf, off);
                    off = 0;
                    data_size = ((group + 1) * buf_size <= workload)
                            ? buf_size
                            : workload - (group * buf_size);
                }
            }
            group = 0;
            off = 0;
            data_size = buf_size;
            diff_weights[c] = cpu::saturate_and_round<data_t>(
                    reduce(buf_lvl2, group_size));
            utils::nd_iterator_step(c, C);
        }
    });
}

template <data_type_t d_type>
void ref_prelu_bwd_t<d_type>::execute_backward(const exec_ctx_t &ctx) const {

    if (pd()->has_zero_dim_memory()) return;

    const auto scratchpad = ctx.get_scratchpad_grantor();
    const auto scratchpad_buf
            = scratchpad.template get<float>(names::key_prelu_reduction);

    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    const auto diff_weights
            = CTX_OUT_MEM(const data_t *, DNNL_ARG_DIFF_WEIGHTS);
    const auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    const auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper data_d(pd()->src_md(0));
    const auto bcast_type = dnnl::impl::get_rhs_arg_broadcasting_strategy(
            *weights_d.md_, data_d);

    switch (bcast_type) {
        case broadcasting_strategy_t::scalar:
            calculate_scalar(src, weights, diff_weights, diff_dst, diff_src,
                    scratchpad_buf);
            break;
        case broadcasting_strategy_t::no_broadcast:
            calculate_no_broadcast(src, weights, diff_weights, diff_dst,
                    diff_src, scratchpad_buf);
            break;
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial:
            calculate_per_oc(src, weights, diff_weights, diff_dst, diff_src,
                    scratchpad_buf);
            break;
        default: assert(!"unsupported broadcast type");
    }
}

template void ref_prelu_fwd_t<data_type::f32>::execute_forward(
        const exec_ctx_t &ctx) const;
template void ref_prelu_bwd_t<data_type::f32>::execute_backward(
        const exec_ctx_t &ctx) const;
template void ref_prelu_fwd_t<data_type::bf16>::execute_forward(
        const exec_ctx_t &ctx) const;
template void ref_prelu_bwd_t<data_type::bf16>::execute_backward(
        const exec_ctx_t &ctx) const;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
