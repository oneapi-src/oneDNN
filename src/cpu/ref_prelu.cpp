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

#include <array>
#include <cassert>
#include <cmath>

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
using namespace data_type;

static float load(data_type_t src_dtype, const byte *base, dim_t offset) {
    switch (src_dtype) {
        case f32: return reinterpret_cast<const float *>(base)[offset];
        case bf16:
            return static_cast<float>(
                    reinterpret_cast<const bfloat16_t *>(base)[offset]);
        default: assert(!"Unsupported data type");
    }
    return -1;
}

static void store(data_type_t dst_dtype, float val, byte *base, dim_t offset) {
    switch (dst_dtype) {
        case f32:
            *reinterpret_cast<float *>(base + sizeof(float) * offset) = val;
            break;
        case bf16:
            *reinterpret_cast<bfloat16_t *>(base + sizeof(bfloat16_t) * offset)
                    = cpu::saturate_and_round<bfloat16_t>(val);
            break;
        default: assert(!"Unsupported data type");
    }
}

static dim_t offset(const memory_desc_wrapper &mem, dims_t dims) {
    const int ndims = mem.ndims();
    switch (ndims) {
        case 1: return mem.off(dims[0]);
        case 2: return mem.off(dims[0], dims[1]);
        case 3: return mem.off(dims[0], dims[1], dims[2]);
        case 4: return mem.off(dims[0], dims[1], dims[2], dims[3]);
        case 5: return mem.off(dims[0], dims[1], dims[2], dims[3], dims[4]);
        default: assert(!"Unsupported ndims count");
    }
    return -1;
}

static dim_t weights_offset(
        const int mask, const memory_desc_wrapper &mem, dims_t &dims) {
    dims_t wei_dims;
    std::copy(dims, dims + max_supported_ndims, wei_dims);
    utils::apply_mask_on_dims(wei_dims, mem.ndims(), mask);
    return offset(mem, wei_dims);
}

void ref_prelu_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return;

    const auto src = CTX_IN_MEM(const byte *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const byte *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(byte *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const int mask = utils::get_dims_mask(
            data_d.dims(), weights_d.dims(), data_d.ndims());
    const dim_t work_amount = data_d.nelems();

    parallel(0, [&](std::size_t ithr, std::size_t nthr) {
        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dims_t dims_d, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            off[i] = 0;
            dims_d[i] = (data_d.dims()[i] != 0) ? data_d.dims()[i] : 1;
        }

        balance211(work_amount, nthr, ithr, start, end);
        utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);

        for (dim_t iwork = start; iwork < end; ++iwork) {
            const auto data_off = offset(data_d, off);
            const auto weight_off = weights_offset(mask, weights_d, off);
            const float src_val = load(data_d.data_type(), src, data_off);
            const float weights_val
                    = load(weights_d.data_type(), weights, weight_off);

            const float res = relu_fwd(src_val, weights_val);

            store(data_d.data_type(), res, dst, data_off);
            utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);
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

float ref_prelu_bwd_t::ker(const byte *src, const byte *weights,
        const byte *diff_dst, byte *diff_src, dim_t data_off, dim_t weight_off,
        dim_t diff_data_off) const {

    const auto dtype = pd()->src_md(0)->data_type;
    const auto wtype = pd()->weights_md(0)->data_type;
    const float src_val = load(dtype, src, data_off);
    const float diff_dst_val = load(dtype, diff_dst, diff_data_off);
    const float weights_val = load(wtype, weights, weight_off);

    const float diff_src_res
            = relu_bwd_use_dst(diff_dst_val, src_val, weights_val);
    const float diff_weight_res = src_val > 0 ? 0 : (diff_dst_val * src_val);

    store(dtype, diff_src_res, diff_src, data_off);

    return diff_weight_res;
}

void ref_prelu_bwd_t::calculate_scalar(const byte *src, const byte *weights,
        byte *diff_weights, const byte *diff_dst, byte *diff_src,
        float *scratchpad_buf) const {

    const memory_desc_wrapper data_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const dim_t work_amount = data_d.nelems();
    const int thread_count
            = nstl::min((dim_t)dnnl_get_max_threads(), work_amount);

    std::vector<float> buf_nthr_partial_results(dnnl_get_max_threads());

    parallel(0, [&](std::size_t ithr, std::size_t nthr) {
        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dims_t dims_d, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            off[i] = 0;
            dims_d[i] = (data_d.dims()[i] != 0) ? data_d.dims()[i] : 1;
        }

        balance211(work_amount, nthr, ithr, start, end);
        const dim_t workload = end - start;

        utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);

        dim_t group_size, buf_size;
        set_reduction_buffers(workload, group_size, buf_size);

        const dim_t scratchpad_offset
                = get_scalar_scratchpad_offset(ithr, nthr, work_amount);
        auto *buf = &scratchpad_buf[scratchpad_offset];
        auto *group_buf = &scratchpad_buf[scratchpad_offset + buf_size];

        dim_t offset_buf {0}, group_off {0}, data_size {buf_size};
        for (dim_t iwork = start; iwork < end; ++iwork) {
            const auto data_off = offset(data_d, off);
            const auto weight_off = 0;
            const auto diff_data_off = data_off;
            buf[offset_buf] = ker(src, weights, diff_dst, diff_src, data_off,
                    weight_off, diff_data_off);
            if (++offset_buf == data_size) {
                group_buf[group_off++] = reduce(buf, offset_buf);
                offset_buf = 0;
                data_size = ((group_off + 1) * buf_size <= workload)
                        ? buf_size
                        : workload - (group_off * buf_size);
            }
            utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);
        }
        buf_nthr_partial_results[ithr] = reduce(group_buf, group_size);
    });
    store(weights_d.data_type(),
            reduce(&buf_nthr_partial_results[0], thread_count), diff_weights,
            0);
}

void ref_prelu_bwd_t::calculate_no_broadcast(const byte *src,
        const byte *weights, byte *diff_weights, const byte *diff_dst,
        byte *diff_src, float *scratchpad_buf) const {

    const memory_desc_wrapper data_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const dim_t work_amount = data_d.nelems();
    const int mask = utils::get_dims_mask(
            data_d.dims(), weights_d.dims(), data_d.ndims());

    parallel(0, [&](std::size_t ithr, std::size_t nthr) {
        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dims_t dims_d, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            off[i] = 0;
            dims_d[i] = (data_d.dims()[i] != 0) ? data_d.dims()[i] : 1;
        }

        balance211(work_amount, nthr, ithr, start, end);
        utils::nd_iterator_init(start, off[0], dims_d[0], off[1], dims_d[1],
                off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);

        for (dim_t iwork = start; iwork < end; ++iwork) {
            const auto data_off = offset(data_d, off);
            const auto weight_off = weights_offset(mask, weights_d, off);
            const auto diff_data_off = data_off;
            const auto diff_weight_off = weight_off;
            const auto res = ker(src, weights, diff_dst, diff_src, data_off,
                    weight_off, diff_data_off);

            store(weights_d.data_type(), res, diff_weights, diff_weight_off);
            utils::nd_iterator_step(off[0], dims_d[0], off[1], dims_d[1],
                    off[2], dims_d[2], off[3], dims_d[3], off[4], dims_d[4]);
        }
    });
}

void ref_prelu_bwd_t::calculate_shared_axes(const byte *src,
        const byte *weights, byte *diff_weights, const byte *diff_dst,
        byte *diff_src, float *scratchpad_buf) const {

    const memory_desc_wrapper data_d(pd()->src_md(0));
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    dims_t dims_d, dims_w;
    for (int i = 0; i < max_supported_ndims; i++) {
        dims_d[i] = (data_d.dims()[i] != 0) ? data_d.dims()[i] : 1;
        dims_w[i] = (weights_d.dims()[i] != 0) ? weights_d.dims()[i] : 1;
    }

    const dim_t work_amount = weights_d.nelems();
    const int mask = utils::get_dims_mask(
            data_d.dims(), weights_d.dims(), data_d.ndims());

    parallel(0, [&](std::size_t ithr, std::size_t nthr) {
        if ((dim_t)ithr >= work_amount) return;

        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        dim_t group_size, buf_size;
        const dim_t workload = data_d.nelems() / weights_d.nelems();
        set_reduction_buffers(workload, group_size, buf_size);
        dim_t scratchpad_offset = (buf_size + group_size) * ithr;
        auto *buf = &scratchpad_buf[scratchpad_offset];
        auto *group_buf = &scratchpad_buf[scratchpad_offset + buf_size];

        dims_t off_w, off_d, dims_start, dims_end;
        utils::nd_iterator_init(start, off_w[0], dims_w[0], off_w[1], dims_w[1],
                off_w[2], dims_w[2], off_w[3], dims_w[3], off_w[4], dims_w[4]);

        for (dim_t iwork = start; iwork < end; ++iwork) {
            for (int i = 0; i < max_supported_ndims; i++) {
                dims_start[i] = (dims_d[i] == dims_w[i]) ? off_w[i] : 0;
                dims_end[i]
                        = (dims_d[i] == dims_w[i]) ? off_w[i] + 1 : dims_d[i];
            }
            dim_t buf_off {0}, group_off {0}, data_size {buf_size};
            for_(off_d[0] = dims_start[0]; off_d[0] < dims_end[0]; ++off_d[0])
            for_(off_d[1] = dims_start[1]; off_d[1] < dims_end[1]; ++off_d[1])
            for_(off_d[2] = dims_start[2]; off_d[2] < dims_end[2]; ++off_d[2])
            for_(off_d[3] = dims_start[3]; off_d[3] < dims_end[3]; ++off_d[3])
            for (off_d[4] = dims_start[4]; off_d[4] < dims_end[4]; ++off_d[4]) {
                const auto data_off = offset(data_d, off_d);
                const auto weight_off = weights_offset(mask, weights_d, off_d);
                const auto diff_data_off = data_off;
                const auto diff_weight = ker(src, weights, diff_dst, diff_src,
                        data_off, weight_off, diff_data_off);
                buf[buf_off] = diff_weight;
                if (++buf_off == data_size) {
                    group_buf[group_off++] = reduce(buf, buf_off);
                    buf_off = 0;
                    data_size = ((group_off + 1) * buf_size <= workload)
                            ? buf_size
                            : workload - (group_off * buf_size);
                }
            }
            const auto diff_weight_off = offset(weights_d, off_w);
            store(weights_d.data_type(), reduce(group_buf, group_size),
                    diff_weights, diff_weight_off);
            utils::nd_iterator_step(off_w[0], dims_w[0], off_w[1], dims_w[1],
                    off_w[2], dims_w[2], off_w[3], dims_w[3], off_w[4],
                    dims_w[4]);
        }
    });
}

void ref_prelu_bwd_t::execute_backward(const exec_ctx_t &ctx) const {

    if (pd()->has_zero_dim_memory()) return;

    const auto scratchpad = ctx.get_scratchpad_grantor();
    auto scratchpad_buf = scratchpad.template get<float>(
            memory_tracking::names::key_prelu_reduction);

    const auto src = CTX_IN_MEM(const byte *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const byte *, DNNL_ARG_WEIGHTS);
    auto diff_weights = CTX_OUT_MEM(const byte *, DNNL_ARG_DIFF_WEIGHTS);
    const auto diff_dst = CTX_IN_MEM(const byte *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(byte *, DNNL_ARG_DIFF_SRC);

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
        case broadcasting_strategy_t::shared_axes:
            calculate_shared_axes(src, weights, diff_weights, diff_dst,
                    diff_src, scratchpad_buf);
            break;
        default: assert(!"unsupported broadcast type");
    }
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
