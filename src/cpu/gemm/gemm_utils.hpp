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

#ifndef GEMM_UTILS_HPP
#define GEMM_UTILS_HPP

#include "dnnl_thread.hpp"
#include "dnnl_traits.hpp"
#include "gemm_pack_storage.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace gemm_utils {

template <typename T>
static inline dim_t get_ld_padd(const dnnl::impl::dim_t x) {
    return x != 1 ? utils::rnd_up(x, 2048 / sizeof(T)) + (64 / sizeof(T)) : 1;
}

template <typename mat_t, typename acc_t>
void prep_gemm_pack(bool do_a, int is_trans, dnnl::impl::dim_t nrows,
        dnnl::impl::dim_t ncols, gemm_pack_storage_t *pack_dst) {

    auto ld = !is_trans ? get_ld_padd<mat_t>(nrows) : get_ld_padd<mat_t>(ncols);
    auto td = !is_trans ? ncols : nrows;

    // TODO Do we need to use only one thread?
    pack_dst->which() = do_a ? matrix_id::a : matrix_id::b;
    pack_dst->setup(1);
    pack_dst->threading().copy = copy_type::no_copy;
    pack_dst->threading().nthrs_m = 1;
    pack_dst->threading().nthrs_n = 1;
    pack_dst->threading().nthrs_k = 1;
    pack_dst->set_nocopy(0, is_trans, ld, td);
    pack_dst->finalize<mat_t, acc_t>();
}

template <typename T>
dnnl_status_t pack_no_copy(const T *src, dnnl::impl::dim_t ld_src,
        dnnl::impl::dim_t nrows, dnnl::impl::dim_t ncols, int trans_src,
        float alpha, gemm_pack_storage_t *dst_pack) {

    auto dst = dst_pack->matrix<T>(0);
    int trans_dst;
    dim_t nrows_dst, ncols_dst;
    dnnl::impl::dim_t ld_dst, td_dst;

    constexpr bool is_f32 = data_traits<T>::data_type == data_type::f32;

    if (!dst_pack->get_nocopy(0, trans_dst, ld_dst, td_dst))
        return dnnl_invalid_arguments;

    if (!trans_dst) {
        nrows_dst = nrows;
        ncols_dst = ncols;
    } else {
        nrows_dst = ncols;
        ncols_dst = nrows;
    }

    if (trans_src == trans_dst) {
        parallel_nd(ncols_dst, [=](dim_t j) {
            auto src_col = src + j * ld_src;
            auto dst_col = dst + j * ld_dst;

            PRAGMA_OMP_SIMD()
            for (dim_t i = 0; i < nrows_dst; i++)
                if (is_f32)
                    dst_col[i] = alpha * src_col[i];
                else
                    dst_col[i] = src_col[i];
        });
    } else {
        // Naive code for now.
        parallel_nd(ncols_dst, [=](dim_t j) {
            auto src_col = src + j;
            auto dst_col = dst + j * ld_dst;

            PRAGMA_OMP_SIMD()
            for (dim_t i = 0; i < nrows_dst; i++)
                if (is_f32)
                    dst_col[i] = alpha * src_col[i * ld_src];
                else
                    dst_col[i] = src_col[i * ld_src];
        });
    }

    return dnnl_success;
}

} // namespace gemm_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // GEMM_UTILS_HPP
