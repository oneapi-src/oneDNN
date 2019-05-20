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

#include "ocl/ocl_types.h"

#define IN_OFF(x0, x1, x2, x3, x4, x5) ( \
    ((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 + \
    ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 + \
    ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 + \
    ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 + \
    ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 + \
    ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)

DATA_T relu_fwd(DATA_T s, DATA_T alpha) {
    return s > 0 ? s : s * alpha;
}
DATA_T relu_bwd(DATA_T dd, DATA_T s, DATA_T alpha) {
    return s > 0 ? dd : dd * alpha;
}

DATA_T linear_fwd(DATA_T s, DATA_T alpha, DATA_T beta) {
    return alpha * s + beta;
}
DATA_T linear_bwd(DATA_T dd, DATA_T alpha) {
    return dd * alpha;
}

DATA_T bounded_relu_fwd(DATA_T s, DATA_T alpha) {
    s = s > 0 ? s : 0;
    return s > alpha ? alpha : s;
}
DATA_T bounded_relu_bwd(DATA_T dd, DATA_T s, DATA_T alpha) {
    return dd * (0 < s && s < alpha ? 1 : 0);
}

DATA_T soft_relu_fwd(DATA_T s) {
    return s < log(DATA_MAX) ? log1p(exp(s)) : s;
}
DATA_T soft_relu_bwd(DATA_T dd, DATA_T s) {
    return dd / (1 + exp(-s));
}

DATA_T logistic_fwd(DATA_T s) {
    return 1 / (1 + exp(-s));
}
DATA_T logistic_bwd(DATA_T dd, DATA_T s) {
    DATA_T v = logistic_fwd(s);
    return dd * v * (1 - v);
}

DATA_T square_fwd(DATA_T s){
    return s*s;
}
DATA_T square_bwd(DATA_T dd, DATA_T s){
    return dd * 2*s;
}

DATA_T sqrt_fwd(DATA_T s){
    return s > 0 ? (sqrt(s)) : 0;
}
DATA_T sqrt_bwd(DATA_T dd, DATA_T s){
    return s > 0 ?  dd / (2 * sqrt(s)) : 0;
}

DATA_T abs_fwd(DATA_T s){
    return s > 0 ? s : -s;
}
DATA_T abs_bwd(DATA_T dd, DATA_T s){
    return s > 0 ? dd : s < 0 ? -dd : 0;
}

DATA_T tanh_fwd(DATA_T s){
    return tanh(s);
}
DATA_T tanh_bwd(DATA_T dd, DATA_T s){
    DATA_T e = tanh_fwd(s);
    return dd * (1 - e) * (1 + e);
}

DATA_T elu_fwd(DATA_T s, DATA_T alpha){
    return s > 0 ? s :  alpha * expm1(s);
}
DATA_T elu_bwd(DATA_T dd, DATA_T s, DATA_T alpha){
    return  dd * (s > 0 ? 1 : alpha * exp(s));
}

__kernel void ref_eltwise_fwd(
        __global DATA_T *src, __global DATA_T *dst, float alpha, float beta) {
    const int i = get_global_id(0);

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    d0 = i % SRC_D0;
    #if NDIMS > 1
    d1 = (i / SRC_D0) % SRC_D1;
    #endif
    #if NDIMS > 2
    d2 = (i / (SRC_D0 * SRC_D1)) % SRC_D2;
    #endif
    #if NDIMS > 3
    d3 = (i / (SRC_D0 * SRC_D1 * SRC_D2)) % SRC_D3;
    #endif
    #if NDIMS > 4
    d4 = (i / (SRC_D0 * SRC_D1 * SRC_D2 * SRC_D3)) % SRC_D4;
    #endif
    #if NDIMS > 5
    d5 = (i / (SRC_D0 * SRC_D1 * SRC_D2 * SRC_D3 * SRC_D4)) % SRC_D5;
    #endif

    const size_t off = IN_OFF(d0,d1,d2,d3,d4,d5);

    DATA_T alpha_ = CONVERT_DATA_T(alpha);
    DATA_T beta_ = CONVERT_DATA_T(beta);

    switch (ALG_KIND) {
    case RELU: dst[off] = relu_fwd(src[off], alpha_); break;
    case LINEAR: dst[off] = linear_fwd(src[off], alpha_, beta_); break;
    case BOUNDED_RELU: dst[off] = bounded_relu_fwd(src[off], alpha_); break;
    case SOFT_RELU: dst[off] = soft_relu_fwd(src[off]); break;
    case LOGISTIC: dst[off] = logistic_fwd(src[off]); break;
    case TANH: dst[off] = tanh_fwd(src[off]); break;
    case ELU: dst[off] = elu_fwd(src[off], alpha_); break;
    case SQUARE: dst[off] = square_fwd(src[off]); break;
    case SQRT: dst[off] = sqrt_fwd(src[off]); break;
    case ABS: dst[off] = abs_fwd(src[off]); break;
    default: return;
    }
}

__kernel void ref_eltwise_bwd(__global DATA_T *src, __global DATA_T *diff_src,
        __global DATA_T *diff_dst, float alpha) {
    const int i = get_global_id(0);

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    d0 = i % SRC_D0;
    #if NDIMS > 1
    d1 = (i / SRC_D0) % SRC_D1;
    #endif
    #if NDIMS > 2
    d2 = (i / (SRC_D0 * SRC_D1)) % SRC_D2;
    #endif
    #if NDIMS > 3
    d3 = (i / (SRC_D0 * SRC_D1 * SRC_D2)) % SRC_D3;
    #endif
    #if NDIMS > 4
    d4 = (i / (SRC_D0 * SRC_D1 * SRC_D2 * SRC_D3)) % SRC_D4;
    #endif
    #if NDIMS > 5
    d5 = (i / (SRC_D0 * SRC_D1 * SRC_D2 * SRC_D3 * SRC_D4)) % SRC_D5;
    #endif

    DATA_T alpha_ = CONVERT_DATA_T(alpha);

    const size_t off = IN_OFF(d0,d1,d2,d3,d4,d5);

    switch (ALG_KIND) {
    case RELU: diff_src[off] = relu_bwd(diff_dst[off], src[off], alpha_); break;
    case LINEAR: diff_src[off] = linear_bwd(diff_dst[off], alpha_); break;
    case BOUNDED_RELU: diff_src[off] = bounded_relu_bwd(diff_dst[off], src[off], alpha_); break;
    case SOFT_RELU: diff_src[off] = soft_relu_bwd(diff_dst[off], src[off]); break;
    case LOGISTIC: diff_src[off] = logistic_bwd(diff_dst[off], src[off]); break;
    case TANH: diff_src[off] = tanh_bwd(diff_dst[off], src[off]); break;
    case ELU: diff_src[off] = elu_bwd(diff_dst[off], src[off], alpha_); break;
    case SQUARE: diff_src[off] = square_bwd(diff_dst[off], src[off]); break;
    case SQRT: diff_src[off] = sqrt_bwd(diff_dst[off], src[off]); break;
    case ABS: diff_src[off] = abs_bwd(diff_dst[off], src[off]); break;
    default: return;
    }
}
