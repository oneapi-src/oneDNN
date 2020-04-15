/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef COMMON_GEMM_TYPES_HPP
#define COMMON_GEMM_TYPES_HPP

#include "dnnl_types.h"

namespace dnnl {
namespace impl {

enum transpose_t { dnnl_notrans, dnnl_trans };

namespace transpose {
const transpose_t notrans = dnnl_notrans;
const transpose_t trans = dnnl_trans;
} // namespace transpose

enum offsetc_t { dnnl_fixed, dnnl_column, dnnl_row };

namespace offsetc {
const offsetc_t fixed = dnnl_fixed;
const offsetc_t column = dnnl_column;
const offsetc_t row = dnnl_row;
} // namespace offsetc

/** A descriptor for a matrix multiplication (gemm) operation */
struct dnnl_gemm_desc_t {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #dnnl_gemm. */
    dnnl_primitive_kind_t primitive_kind = dnnl_gemm;
    /** Flag for transposing matrix A. */
    transpose_t transa = dnnl_notrans;
    /** Flag for transposing matrix B. */
    transpose_t transb = dnnl_notrans;
    /** Number of C matrices. */
    dnnl_dim_t batch = 0;
    /** Number of rows of C. */
    dnnl_dim_t m = 0;
    /** Number of columns of C. */
    dnnl_dim_t n = 0;
    /** Size of inner dimension shared between A and B. */
    dnnl_dim_t k = 0;
    /** Stride between 2 matrices A in a batch. */
    dnnl_dim_t stride_a = 0;
    /** Stride between 2 matrices B in a batch. */
    dnnl_dim_t stride_b = 0;
    /** Stride between 2 matrices C in a batch. */
    dnnl_dim_t stride_c = 0;
    /** Leading dimension of A. */
    dnnl_dim_t lda = 0;
    /** Leading dimension of B. */
    dnnl_dim_t ldb = 0;
    /** Leading dimension of C. */
    dnnl_dim_t ldc = 0;

    /** Describes size of bias matrix. */
    dnnl_dim_t bias_mask = 0;
    /** Type of matrix A. */
    dnnl_data_type_t a_type = dnnl_data_type_undef;
    /** Type of matrix B. */
    dnnl_data_type_t b_type = dnnl_data_type_undef;
    /** Type of matrix C. */
    dnnl_data_type_t c_type = dnnl_data_type_undef;
    /** Type for accumulating A*B. */
    dnnl_data_type_t acc_type = dnnl_data_type_undef;
    /** Type of bias. */
    dnnl_data_type_t bias_type = dnnl_data_type_undef;
};

} // namespace impl
} // namespace dnnl

#endif // COMMON_GEMM_TYPES_HPP
