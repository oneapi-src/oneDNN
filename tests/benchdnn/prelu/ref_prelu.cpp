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

#include <algorithm>
#include "prelu/prelu.hpp"
#include "tests/test_thread.hpp"

namespace prelu {

enum class broadcast_strategy {
    unsupported = 0,
    no_broadcast,
    scalar, // scalar
    per_oc, // Channel-wise
};

static broadcast_strategy get_prelu_weights_type(
        const dnnl_memory_desc_t &data_desc,
        const dnnl_memory_desc_t &weights_desc) {

    const auto ndims = data_desc.ndims; //same as weights

    bool all_ones = true;
    bool all_same = true;

    for (int i = 0; i < ndims; ++i) {
        if (weights_desc.dims[i] != data_desc.dims[i]) all_same = false;
        if (weights_desc.dims[i] != 1) all_ones = false;
    }
    if (all_ones) return broadcast_strategy::scalar;
    if (all_same) return broadcast_strategy::no_broadcast;

    bool ch_wise = true;

    if (weights_desc.dims[1] != data_desc.dims[1]) ch_wise = false;

    for (int i = 2; i < ndims; ++i) {
        if (weights_desc.dims[i] != 1) ch_wise = false;
    }
    if (ch_wise) return broadcast_strategy::per_oc;

    return broadcast_strategy::unsupported;
}

float reduce(std::vector<float> &mem, int64_t size) {
    bool tail = size % 2;
    const auto reduce_iteration = [&](std::vector<float> &mem) {
        const auto div_res = std::div(size, (int64_t)2);
        tail = div_res.rem;
        size = div_res.quot;
        if (!tail && !size) {
            mem[0] = 0;
            return;
        }
        int64_t i {0}, off {0};
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

static void set_reduction_buffers(
        const int64_t work_amount, int64_t &group_size, int64_t &buf_size) {
    float sqrt = std::sqrt(work_amount);
    group_size = std::ceil(sqrt);
    buf_size = std::floor(sqrt);
    if (group_size * buf_size < work_amount) group_size++;
}

static int64_t inline offset(int64_t n, int64_t C, int64_t c, int64_t D,
        int64_t d, int64_t H, int64_t h, int64_t W, int64_t w) {
    return ((((n * C + c) * D + d) * H + h) * W + w);
}

static int64_t weights_offset(broadcast_strategy bcast_type, int64_t n,
        int64_t C, int64_t c, int64_t D, int64_t d, int64_t H, int64_t h,
        int64_t W, int64_t w) {
    switch (bcast_type) {
        case broadcast_strategy::no_broadcast:
            return offset(n, C, c, D, d, H, h, W, w);
        case broadcast_strategy::per_oc: return c;
        case broadcast_strategy::scalar: return (int64_t)0;
        default: assert(!"unsupported broadcast type");
    }
    return (int64_t)-1;
}

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &dst) {
    const auto ndims = prb->ndims;
    const auto bcast_type = get_prelu_weights_type(src.md_, weights.md_);

    auto &src_dims = prb->dims[0];
    const int64_t N = src_dims[0];
    const int64_t C = src_dims[1];
    const int64_t D = ndims >= 5 ? src_dims[ndims - 3] : 1;
    const int64_t H = ndims >= 4 ? src_dims[ndims - 2] : 1;
    const int64_t W = ndims >= 3 ? src_dims[ndims - 1] : 1;

    dnnl::impl::parallel_nd(N, C, D, H, W,
            [&](int64_t n, int64_t c, int64_t d, int64_t h, int64_t w) {
                const auto data_offset = offset(n, C, c, D, d, H, h, W, w);
                const auto wei_offset
                        = weights_offset(bcast_type, n, C, c, D, d, H, h, W, w);
                float res = src.get_elem(data_offset) > 0
                        ? src.get_elem(data_offset)
                        : src.get_elem(data_offset)
                                * weights.get_elem(wei_offset);

                dst.set_elem(data_offset, res);
            });
}

static float ker(const dnn_mem_t &src, const dnn_mem_t &weights,
        dnn_mem_t &diff_src, const dnn_mem_t &diff_dst, int64_t data_off,
        int64_t weight_off, int64_t diff_data_off) {
    float diff_src_res = src.get_elem(data_off) > 0
            ? diff_dst.get_elem(diff_data_off)
            : diff_dst.get_elem(diff_data_off) * weights.get_elem(weight_off);
    diff_src.set_elem(diff_data_off, diff_src_res);
    float diff_weight_res = src.get_elem(data_off) > 0
            ? 0
            : (diff_dst.get_elem(diff_data_off) * src.get_elem(data_off));
    return diff_weight_res;
}

static void calculate_scalar(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_weights) {

    const auto bcast_type = broadcast_strategy::scalar;

    const auto ndims = prb->ndims;
    const auto &src_dims = prb->dims[0];
    const int64_t N = src_dims[0];
    const int64_t C = src_dims[1];
    const int64_t D = ndims >= 5 ? src_dims[ndims - 3] : 1;
    const int64_t H = ndims >= 4 ? src_dims[ndims - 2] : 1;
    const int64_t W = ndims >= 3 ? src_dims[ndims - 1] : 1;

    const int64_t work_amount = src.nelems();
    const int nscr = std::min((int64_t)dnnl_get_max_threads(), src.nelems());
    std::vector<float> buf_lvl3(nscr);
    dnnl::impl::parallel(0, [&](size_t ithr, size_t nthr) {
        if ((int64_t)ithr >= work_amount) return;

        int64_t start {0}, end {0};
        int64_t n {0}, c {0}, d {0}, h {0}, w {0};

        dnnl::impl::balance211(work_amount, nthr, ithr, start, end);
        dnnl::impl::utils::nd_iterator_init(
                start, n, N, c, C, d, D, h, H, w, W);

        const int64_t workload = end - start;
        int64_t group_size {0}, buf_size {0};

        set_reduction_buffers(workload, group_size, buf_size);

        std::vector<float> buf(buf_size);
        std::vector<float> buf_lvl2(group_size);

        int64_t off {0}, group {0}, data_size {buf_size};
        for (int64_t iwork = start; iwork < end; ++iwork) {
            const auto data_off = offset(n, C, c, D, d, H, h, W, w);
            const auto weight_off
                    = weights_offset(bcast_type, n, C, c, D, d, H, h, W, w);
            const auto diff_data_off = offset(n, C, c, D, d, H, h, W, w);
            buf[off] = ker(src, weights, diff_src, diff_dst, data_off,
                    weight_off, diff_data_off);
            if (++off == data_size) {
                buf_lvl2[group++] = reduce(buf, off);
                off = 0;
                data_size = ((group + 1) * buf_size <= workload)
                        ? buf_size
                        : workload - (group * buf_size);
            }
            dnnl::impl::utils::nd_iterator_step(n, N, c, C, d, D, h, H, w, W);
        }
        buf_lvl3[ithr] = reduce(buf_lvl2, buf_lvl2.size());
    });
    diff_weights.set_elem(0, reduce(buf_lvl3, nscr));
}

static void calculate_no_broadcast(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_weights) {

    const auto bcast_type = broadcast_strategy::no_broadcast;

    const auto ndims = prb->ndims;
    const auto &src_dims = prb->dims[0];
    const int64_t N = src_dims[0];
    const int64_t C = src_dims[1];
    const int64_t D = ndims >= 5 ? src_dims[ndims - 3] : 1;
    const int64_t H = ndims >= 4 ? src_dims[ndims - 2] : 1;
    const int64_t W = ndims >= 3 ? src_dims[ndims - 1] : 1;

    const int64_t work_amount = src.nelems();
    dnnl::impl::parallel(0, [&](size_t ithr, size_t nthr) {
        if ((int64_t)ithr >= work_amount) return;

        int64_t start {0}, end {0};
        int64_t n {0}, c {0}, d {0}, h {0}, w {0};

        dnnl::impl::balance211(work_amount, nthr, ithr, start, end);
        dnnl::impl::utils::nd_iterator_init(
                start, n, N, c, C, d, D, h, H, w, W);

        for (int64_t iwork = start; iwork < end; ++iwork) {
            const auto diff_weight_off
                    = weights_offset(bcast_type, n, C, c, D, d, H, h, W, w);
            const auto data_off = offset(n, C, c, D, d, H, h, W, w);
            const auto weight_off
                    = weights_offset(bcast_type, n, C, c, D, d, H, h, W, w);
            const auto diff_data_off = offset(n, C, c, D, d, H, h, W, w);
            auto res = ker(src, weights, diff_src, diff_dst, data_off,
                    weight_off, diff_data_off);
            diff_weights.set_elem(diff_weight_off, res);
            dnnl::impl::utils::nd_iterator_step(n, N, c, C, d, D, h, H, w, W);
        }
    });
}

static void calculate_per_oc(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_weights) {

    const auto bcast_type = broadcast_strategy::per_oc;

    const auto ndims = prb->ndims;
    const auto &src_dims = prb->dims[0];
    const int64_t N = src_dims[0];
    const int64_t C = src_dims[1];
    const int64_t D = ndims >= 5 ? src_dims[ndims - 3] : 1;
    const int64_t H = ndims >= 4 ? src_dims[ndims - 2] : 1;
    const int64_t W = ndims >= 3 ? src_dims[ndims - 1] : 1;

    const int64_t work_amount = C;
    dnnl::impl::parallel(0, [&](size_t ithr, size_t nthr) {
        if ((int64_t)ithr >= work_amount) return;

        int64_t start {0}, end {0};
        int64_t c {0};

        dnnl::impl::balance211(work_amount, nthr, ithr, start, end);
        dnnl::impl::utils::nd_iterator_init(start, c, C);

        const int64_t workload = N * D * H * W;
        int64_t group_size {0}, buf_size {0};

        set_reduction_buffers(workload, group_size, buf_size);
        std::vector<float> buf(buf_size);
        std::vector<float> buf_lvl2(group_size);

        int64_t off {0}, group {0}, data_size {buf_size};
        for (int64_t iwork = start; iwork < end; ++iwork) {
            for_(int n = 0; n < N; ++n)
            for_(int d = 0; d < D; ++d)
            for_(int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                const auto data_off = offset(n, C, c, D, d, H, h, W, w);
                const auto weight_off
                        = weights_offset(bcast_type, n, C, c, D, d, H, h, W, w);
                const auto diff_data_off = offset(n, C, c, D, d, H, h, W, w);
                buf[off] = ker(src, weights, diff_src, diff_dst, data_off,
                        weight_off, diff_data_off);
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
            diff_weights.set_elem(c, reduce(buf_lvl2, buf_lvl2.size()));
            dnnl::impl::utils::nd_iterator_step(c, C);
        }
    });
}

void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_weights) {

    const auto bcast_type = get_prelu_weights_type(src.md_, weights.md_);
    switch (bcast_type) {
        case broadcast_strategy::scalar:
            calculate_scalar(
                    prb, src, weights, diff_src, diff_dst, diff_weights);
            return;
        case broadcast_strategy::no_broadcast:
            calculate_no_broadcast(
                    prb, src, weights, diff_src, diff_dst, diff_weights);
            return;
        case broadcast_strategy::per_oc:
            calculate_per_oc(
                    prb, src, weights, diff_src, diff_dst, diff_weights);
            return;
        default: assert(!"unsupported broadcast type");
    }
}

} // namespace prelu
