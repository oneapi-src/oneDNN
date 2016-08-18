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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "gemm_inner_product.hpp"
#include "type_helpers.hpp"

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
status_t gemm_inner_product<prec>::execute_forward()
{
#ifdef USE_CBLAS
    auto obtain_ptr = [this](uint32_t idx) {
        const size_t oi = this->input()[idx].output_index;
        return reinterpret_cast<const data_t *>(
                this->input()[idx].primitive->output()[oi]->memory_const());
    };

    const data_t *src = obtain_ptr(0);
    const data_t *weights = obtain_ptr(1);
    const data_t *bias = _with_bias ? obtain_ptr(2) : nullptr;
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
#pragma omp parallel for
        for (cblas_int mb = 0; mb < M; mb++)
            cblas_axpy<prec>(N, 1.0, bias, 1, dst + dst_d.blk_off(mb), 1);

    return success;
#else
    return unimplemented;
#endif
}

template <impl::precision_t prec>
status_t gemm_inner_product<prec>::execute_backward_data()
{
    return unimplemented;
}

template <impl::precision_t prec>
status_t gemm_inner_product<prec>::execute_backward_weights()
{
    return unimplemented;
}

template <impl::precision_t prec>
status_t gemm_inner_product<prec>::execute_backward_bias()
{
    return unimplemented;
}

template <impl::precision_t prec>
status_t set_default_format(memory_desc_t &memory_desc,
        memory_format_t memory_format)
{
    return mkldnn_memory_desc_init(&memory_desc,
            &memory_desc.tensor_desc, prec, memory_format);
}

template <impl::precision_t prec>
status_t gemm_inner_product<prec>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkldnn::impl::engine &engine)
{
#ifdef USE_CBLAS
    if (op_desc._kind != primitive_kind::inner_product)
        return invalid_arguments;

    auto ip_d = op_desc.inner_product;
    if (ip_d.prop_kind != forward)
        return unimplemented;

    /* memory descriptors check and fill-in */
    if (ip_d.src_desc.format == any) {
        if (ip_d.src_desc.tensor_desc.ndims == 4)
            CHECK(set_default_format<prec>(ip_d.src_desc, nChw8c));
        else if (ip_d.src_desc.tensor_desc.ndims == 2)
            CHECK(set_default_format<prec>(ip_d.src_desc, nc));
        else
            return unimplemented;
    }

    if (ip_d.weights_desc.format == any) {
        if (ip_d.weights_desc.tensor_desc.ndims == 4)
            CHECK(set_default_format<prec>(ip_d.weights_desc, oihw));
        else if (ip_d.src_desc.tensor_desc.ndims == 2)
            CHECK(set_default_format<prec>(ip_d.weights_desc, oi));
            return unimplemented;
    }

    const bool with_bias = !memory_desc_wrapper(ip_d.bias_desc).is_zero();
    if (with_bias && ip_d.bias_desc.format == any)
        CHECK(set_default_format<prec>(ip_d.bias_desc, x));
    if (ip_d.dst_desc.format == any)
        CHECK(set_default_format<prec>(ip_d.dst_desc, nc));

    const bool dense_src = memory_desc_wrapper(ip_d.src_desc).is_dense();
    const bool dense_weights = memory_desc_wrapper(ip_d.weights_desc).is_dense();
    const bool dense_bias = memory_desc_wrapper(ip_d.bias_desc).is_dense();
    const bool dense_dst = memory_desc_wrapper(ip_d.dst_desc).is_dense();

    // TODO: need a proper check for blocked formats: orders and numbers of
    // spatial and channel dimensions in src and weights must agree one with
    // another.

    if (!dense_src || !dense_weights || !dense_bias || !dense_dst)
        return unimplemented;

    /* memory primitive descriptors check */
    memory_primitive_desc_t src_pd, weights_pd, bias_pd, dst_pd;
    CHECK(mkldnn_memory_primitive_desc_init(&src_pd, &ip_d.src_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(
            &weights_pd, &ip_d.weights_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(
            &bias_pd, &ip_d.bias_desc, &engine));
    CHECK(mkldnn_memory_primitive_desc_init(&dst_pd, &ip_d.dst_desc, &engine));

    /* final stage */
    inner_product_primitive_desc_t ippd;
    ippd.base.primitive_kind = inner_product;
    ippd.base.engine = &engine;
    ippd.base.implementation = reinterpret_cast<const void*>(&implementation);
    ippd.inner_product_desc = ip_d;
    ippd.src_primitive_desc = src_pd;
    ippd.weights_primitive_desc = weights_pd;
    ippd.bias_primitive_desc = bias_pd;
    ippd.dst_primitive_desc = dst_pd;

    // if (!inner_product_primitive_desc_is_ok(ippd)) return invalid_arguments; // ???

    primitive_desc->inner_product = ippd;

    return success;
#else
    return unimplemented;
#endif
}

namespace {
template <impl::precision_t prec>
status_t create(primitive **aprimitive,
        const primitive_desc_t *primitive_desc, const primitive_at_t inputs[],
        const primitive *outputs[])
{
    assert(primitive_desc->base.primitive_kind == inner_product);

    auto &ippd = primitive_desc->inner_product;
    // TODO: some checks here.

    *aprimitive = new gemm_inner_product<prec>(ippd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}
}

template <impl::precision_t prec>
const primitive_impl gemm_inner_product<prec>::implementation = {
    create<prec>
};

template class gemm_inner_product<f32>;
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
