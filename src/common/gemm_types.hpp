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

#ifndef GEMM_TYPES_HPP
#define GEMM_TYPES_HPP

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
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #dnnl_gemm. */
    dnnl_primitive_kind_t primitive_kind;
    /** Flag for transposing matrix A. */
    transpose_t transa;
    /** Flag for transposing matrix B. */
    transpose_t transb;
    /** Number of rows of C. */
    dnnl_dim_t m;
    /** Number of columns of C. */
    dnnl_dim_t n;
    /** Size of inner dimension shared between A and B. */
    dnnl_dim_t k;
    /** Leading dimension of A. */
    dnnl_dim_t lda;
    /** Leading dimension of B. */
    dnnl_dim_t ldb;
    /** Leading dimension of C. */
    dnnl_dim_t ldc;

    /** Scaling factor for A * B. */
    float alpha;
    /** Scaling factor for C. */
    float beta;
    /** Type of matrix A. */
    dnnl_data_type_t a_type;
    /** Type of matrix B. */
    dnnl_data_type_t b_type;
    /** Type of matrix C. */
    dnnl_data_type_t c_type;
    /** Type for accumulating A*B. */
    dnnl_data_type_t acc_type;
} dnnl_gemm_desc_t;

} // namespace impl
} // namespace dnnl

#endif // GEMM_TYPES_HPP
