/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include "common/float16.hpp"
#include "common/mkldnn_traits.hpp"
#include "common/type_helpers.hpp"

#include "ocl/jit_gen9_gemm.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <data_type_t c_type>
struct jit_gen9_gemm_driver_params {};

template <>
struct jit_gen9_gemm_driver_params<data_type::f32> {
    static constexpr auto block_m = 512 * 16;
    static constexpr auto block_n = 64 * 32;
    static constexpr auto block_k = 1024;
};

template <>
struct jit_gen9_gemm_driver_params<data_type::f16> {
    static constexpr auto block_m = 512 * 16;
    static constexpr auto block_n = 64 * 32;
    static constexpr auto block_k = 2048;
};

template <data_type_t a_type, data_type_t b_type, data_type_t c_type>
status_t jit_gen9_gemm_t<a_type, b_type, c_type>::launch_beta(stream_t *s,
        int64_t m, int64_t n, c_t alpha, const memory_storage_t &a,
        int64_t offset_a, int64_t lda) const {
    assert(beta_kernel_);

    beta_kernel_.set_arg(0, m);
    beta_kernel_.set_arg(1, n);
    beta_kernel_.set_arg(2, alpha);
    beta_kernel_.set_arg(3, a);
    beta_kernel_.set_arg(4, offset_a);
    beta_kernel_.set_arg(5, lda);

    size_t gws[3] = { 1, size_t(n), 1 };
    size_t lws[3] = { 1, 1, 1 };
    auto nd_range = cl_nd_range_t(gws, lws);

    auto &executor = *(utils::downcast<cl_stream_t *>(s)->cl_executor());

    return executor.parallel_for(nd_range, beta_kernel_);
}

template <data_type_t a_type, data_type_t b_type, data_type_t c_type>
status_t jit_gen9_gemm_t<a_type, b_type, c_type>::launch_copy(stream_t *s,
        int64_t x, int64_t y, const memory_storage_t &a, int64_t offset_a,
        int64_t lda, c_t alpha, const memory_storage_t &b, int64_t offset_b,
        bool outer, bool trans) const {
    auto &kernel = copy_kernel_[outer][trans];

    assert(kernel);
    kernel.set_arg(0, x);
    kernel.set_arg(1, y);
    kernel.set_arg(2, a);
    kernel.set_arg(3, offset_a);
    kernel.set_arg(4, lda);
    kernel.set_arg(5, alpha);
    kernel.set_arg(6, b);
    kernel.set_arg(7, offset_b);

    auto unroll = outer ? jit_gen9_gemm_kernel_params<c_type>::unroll_n
                        : jit_gen9_gemm_kernel_params<c_type>::unroll_m;

    size_t gws[3] = { size_t(x), size_t((y + unroll - 1) / unroll), 1 };
    size_t lws[3] = { 1, 1, 1 };

    auto nd_range = cl_nd_range_t(gws, lws);

    auto &executor = *(utils::downcast<cl_stream_t *>(s)->cl_executor());

    return executor.parallel_for(nd_range, kernel);
}

template <data_type_t a_type, data_type_t b_type, data_type_t c_type>
status_t jit_gen9_gemm_t<a_type, b_type, c_type>::launch_compute(stream_t *s,
        int64_t m, int64_t n, int64_t k, const memory_storage_t &base,
        int32_t offset_a, int32_t offset_b, const memory_storage_t &c,
        int64_t offset_c, int64_t ldc, bool beta0) const {
    auto &kernel = compute_kernel_[beta0];

    assert(kernel);
    kernel.set_arg(0, m);
    kernel.set_arg(1, n);
    kernel.set_arg(2, k);
    kernel.set_arg(3, base);
    kernel.set_arg(4, offset_a);
    kernel.set_arg(5, offset_b);
    kernel.set_arg(6, c);
    kernel.set_arg(7, offset_c);
    kernel.set_arg(8, ldc);

    auto unroll_m = jit_gen9_gemm_kernel_params<c_type>::unroll_m;
    auto unroll_n = jit_gen9_gemm_kernel_params<c_type>::unroll_n;

    int nthreads_x = (m + unroll_m - 1) / unroll_m;
    int nthreads_y = (n + unroll_n - 1) / unroll_n;

    int lws_y = 8;
    while (nthreads_y % lws_y)
        lws_y--;

    size_t gws[3] = { size_t(nthreads_x * 8), size_t(nthreads_y), 1 };
    size_t lws[3] = { 8, size_t(lws_y), 1 };

    if (c_type == data_type::f16)
        lws[1] = 1;

    auto nd_range = cl_nd_range_t(gws, lws);

    auto &executor = *(utils::downcast<cl_stream_t *>(s)->cl_executor());

    return executor.parallel_for(nd_range, kernel);
}

template <data_type_t a_type, data_type_t b_type, data_type_t c_type>
status_t jit_gen9_gemm_t<a_type, b_type, c_type>::execute(
        const exec_ctx_t &ctx) const {

    using a_t = typename prec_traits<a_type>::type;
    using b_t = typename prec_traits<b_type>::type;
    using c_t = typename prec_traits<c_type>::type;

    status_t status;
    constexpr int64_t align = 0x1000;
    auto block_m = jit_gen9_gemm_driver_params<c_type>::block_m;
    auto block_n = jit_gen9_gemm_driver_params<c_type>::block_n;
    auto block_k = jit_gen9_gemm_driver_params<c_type>::block_k;

    auto m = pd()->desc()->m;
    auto n = pd()->desc()->n;
    auto k = pd()->desc()->k;

    bool transa = (pd()->desc()->transa == mkldnn_trans);
    bool transb = (pd()->desc()->transb == mkldnn_trans);

    auto lda = pd()->desc()->lda;
    auto ldb = pd()->desc()->ldb;
    auto ldc = pd()->desc()->ldc;

    auto alpha = pd()->desc()->alpha;
    auto beta = pd()->desc()->beta;

    c_t alpha_native, beta_native, one_native;
    alpha_native = alpha;
    beta_native = beta;
    one_native = 1.0f;

    auto &a = CTX_IN_STORAGE(MKLDNN_ARG_SRC_0);
    auto &b = CTX_IN_STORAGE(MKLDNN_ARG_SRC_1);
    auto &c = CTX_OUT_STORAGE(MKLDNN_ARG_DST);

    size_t off_a0 = a.get_offset() / sizeof(a_t) + pd()->dyn_offset_a;
    size_t off_b0 = b.get_offset() / sizeof(b_t) + pd()->dyn_offset_b;
    size_t off_c0 = c.get_offset() / sizeof(c_t) + pd()->dyn_offset_c;

    if (beta != 0. && beta != 1.) {
        status = launch_beta(ctx.stream(), m, n, beta_native, c, off_c0, ldc);
        if (status)
            return status;
    }

    int64_t off_b_packed = 0;
    int64_t off_a_packed
            = ((off_b_packed + block_n * block_k) + align - 1) & -align;

    for (int64_t Bk = 0; Bk < k; Bk += block_k) {
        int64_t size_k = k - Bk;
        if (size_k > block_k)
            size_k = block_k;

        for (int64_t Bm = 0; Bm < m; Bm += block_m) {
            int64_t size_m = m - Bm;
            if (size_m > block_m)
                size_m = block_m;

            auto off_a_src
                    = off_a0 + (!transa ? (Bm + Bk * lda) : (Bk + Bm * lda));
            status = launch_copy(ctx.stream(), size_k, size_m, a, off_a_src,
                    lda, alpha_native, *temp_buf_, off_a_packed, false, !transa);
            if (status)
                return status;

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n)
                    size_n = block_n;

                if ((Bn == 0) || (n > block_n)) {
                    auto off_b_src = off_b0
                            + (!transb ? (Bk + Bn * ldb) : (Bn + Bk * lda));

                    status = launch_copy(ctx.stream(), size_k, size_n, b,
                            off_b_src, ldb, one_native, *temp_buf_, off_b_packed,
                            true, transb);
                    if (status)
                        return status;
                }

                auto off_c = off_c0 + Bm + Bn * ldc;

                bool beta0 = (beta == 0) && (Bk == 0);
                status = launch_compute(ctx.stream(), size_m, size_n, size_k,
                        *temp_buf_, off_a_packed, off_b_packed, c, off_c, ldc, beta0);
                if (status)
                    return status;
            }
        }
    }

    return status::success;
}

using namespace data_type;

template struct jit_gen9_gemm_t<f16>;
template struct jit_gen9_gemm_t<f32>;

} // namespace ocl
} // namespace impl
} // namespace mkldnn

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
