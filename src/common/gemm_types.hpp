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

#include "mkldnn_types.h"

namespace mkldnn {
namespace impl {

enum transpose_t {
    mkldnn_notrans,
    mkldnn_trans
};

namespace transpose {
    const transpose_t notrans = mkldnn_notrans;
    const transpose_t trans = mkldnn_trans;
}

/** A descriptor for a matrix multiplication (gemm) operation */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_gemm. */
    mkldnn_primitive_kind_t primitive_kind;
    /** Flag for transposing matrix A. */
    transpose_t transa;
    /** Flag for transposing matrix B. */
    transpose_t transb;
    /** Number of rows of C. */
    mkldnn_dim_t m;
    /** Number of columns of C. */
    mkldnn_dim_t n;
    /** Size of inner dimension shared between A and B. */
    mkldnn_dim_t k;
    /** Leading dimension of A. */
    mkldnn_dim_t lda;
    /** Leading dimension of B. */
    mkldnn_dim_t ldb;
    /** Leading dimension of C. */
    mkldnn_dim_t ldc;
    /** Scaling factor for A*B. */
    float alpha;
    /** Scaling factor for C. */
    float beta;
    /** Type of matrix A. */
    mkldnn_data_type_t a_type;
    /** Type of matrix B. */
    mkldnn_data_type_t b_type;
    /** Type of matrix C. */
    mkldnn_data_type_t c_type;
} mkldnn_gemm_desc_t;

} // namespace impl
} // namespace mkldnn

#endif // GEMM_TYPES_HPP
