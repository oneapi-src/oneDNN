/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "gemm_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

// TODO: move BLAS wrappers to a separate header?
#ifdef USE_MKL
#include "mkl_cblas.h"
typedef MKL_INT cblas_int;
#endif

#ifdef USE_CBLAS
namespace {

template <mkldnn::impl::precision_t prec>
using data_t = typename prec_trait<prec>::type;

template <mkldnn::impl::precision_t prec>
inline void cblas_gemm(CBLAS_LAYOUT layout,
        CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        cblas_int M, cblas_int N, cblas_int K,
        data_t<prec> alpha, const data_t<prec> *A, cblas_int lda,
        const data_t<prec> *B, cblas_int ldb,
        data_t<prec> beta, data_t<prec> *C, cblas_int ldc);

template <>
inline void cblas_gemm<mkldnn::impl::precision::f32>(CBLAS_LAYOUT layout,
        CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        cblas_int M, cblas_int N, cblas_int K,
        float alpha, const float *A, cblas_int lda,
        const float *B, cblas_int ldb,
        float beta, float *C, cblas_int ldc) {
    cblas_sgemm(layout, transa, transb,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <mkldnn::impl::precision_t prec>
inline void cblas_axpy(cblas_int N,
        data_t<prec> alpha, const data_t<prec> *X, cblas_int incx,
        data_t<prec> *Y, cblas_int incy);

template <>
inline void cblas_axpy<mkldnn::impl::precision::f32>(cblas_int N,
        float alpha, const float *X, cblas_int incx,
        float *Y, cblas_int incy) {
    cblas_saxpy(N, alpha, X, incx, Y, incy);
}

}
#endif

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::precision;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;

template <impl::precision_t prec>
status_t gemm_inner_product<prec>::execute_forward() {
#ifdef USE_CBLAS
    auto obtain_ptr = [this](uint32_t idx) {
        const size_t oi = this->input()[idx].output_index;
        return reinterpret_cast<const data_t *>(
                this->input()[idx].primitive->output()[oi]->memory_const());
    };

    const data_t *src = obtain_ptr(0);
    const data_t *weights = obtain_ptr(1);
    const data_t *bias = this->_with_bias ? obtain_ptr(2) : nullptr;
    data_t *dst = reinterpret_cast<data_t *>(this->output()[0]->memory());

    const memory_desc_wrapper src_d(this->_ippd.src_primitive_desc.memory_desc),
            dst_d(this->_ippd.dst_primitive_desc.memory_desc);

    // TODO: consistency checks
    // XXX: need to switch to signed ints everywhere
    const cblas_int M = src_d.dims()[0];
    const cblas_int K = array_product(&src_d.dims()[1], src_d.ndims() - 1);
    const cblas_int N = dst_d.dims()[1];

    cblas_gemm<prec>(CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, K, 1.0, src, K, weights, K, 0.0, dst, N);
    if (bias)
#pragma omp parallel for schedule(static)
        for (cblas_int mb = 0; mb < M; mb++)
            cblas_axpy<prec>(N, 1.0, bias, 1, dst + dst_d.blk_off(mb), 1);

    return success;
#else
    return unimplemented;
#endif
}

template <impl::precision_t prec>
status_t gemm_inner_product<prec>::set_default_parameters(
        inner_product_desc_t &ip_d) {
    if (ip_d.src_desc.format == any) {
        if (ip_d.src_desc.tensor_desc.ndims == 4)
            CHECK(types::set_default_format<prec>(ip_d.src_desc, nChw8c));
    }
    if (ip_d.weights_desc.format == any) {
        if (ip_d.weights_desc.tensor_desc.ndims == 4)
            CHECK(types::set_default_format<prec>(ip_d.weights_desc, oIhw8i));
    }

    return inner_product<gemm_inner_product<prec>>::template
        set_default_parameters<void>(ip_d);
}

template <impl::precision_t prec>
status_t gemm_inner_product<prec>::constraint(const inner_product_desc_t &ip_d)
{
#ifdef USE_CBLAS
    bool args_ok = one_of(ip_d.prop_kind, prop_kind::forward_training,
            prop_kind::forward_scoring);
    if (!args_ok) return unimplemented;

    const bool dense_src = memory_desc_wrapper(ip_d.src_desc).is_dense();
    const bool dense_weights = memory_desc_wrapper(ip_d.weights_desc).is_dense();
    const bool dense_bias = memory_desc_wrapper(ip_d.bias_desc).is_dense();
    const bool dense_dst = memory_desc_wrapper(ip_d.dst_desc).is_dense();

    // TODO: need a proper check for blocked formats: orders and numbers of
    // spatial and channel dimensions in src and weights must agree one with
    // another.

    if (!dense_src || !dense_weights || !dense_bias || !dense_dst)
        return unimplemented;

    return success;
#else
    return unimplemented;
#endif
}

template <impl::precision_t prec>
const primitive_impl gemm_inner_product<prec>::implementation = {
    gemm_inner_product<prec>::create
};

template class gemm_inner_product<f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
