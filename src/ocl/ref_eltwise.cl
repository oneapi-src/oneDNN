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

__kernel void ref_eltwise_fwd(
        __global DATA_T *src, __global DATA_T *dst, float alpha, float beta) {
    const int i = get_global_id(0);

    DATA_T alpha_ = CONVERT_DATA_T(alpha);
    DATA_T beta_ = CONVERT_DATA_T(beta);

    switch (ALG_KIND) {
    case RELU: dst[i] = relu_fwd(src[i], alpha_); break;
    case LINEAR: dst[i] = linear_fwd(src[i], alpha_, beta_); break;
    case BOUNDED_RELU: dst[i] = bounded_relu_fwd(src[i], alpha_); break;
    case SOFT_RELU: dst[i] = soft_relu_fwd(src[i]); break;
    case LOGISTIC: dst[i] = logistic_fwd(src[i]); break;
    default: return;
    }
}

__kernel void ref_eltwise_bwd(__global DATA_T *src, __global DATA_T *diff_src,
        __global DATA_T *diff_dst, float alpha) {
    const int i = get_global_id(0);

    DATA_T alpha_ = CONVERT_DATA_T(alpha);

    switch (ALG_KIND) {
    case RELU: diff_src[i] = relu_bwd(diff_dst[i], src[i], alpha_); break;
    case LINEAR: diff_src[i] = linear_bwd(diff_dst[i], alpha_); break;
    case BOUNDED_RELU:
        diff_src[i] = bounded_relu_bwd(diff_dst[i], src[i], alpha_);
        break;
    case SOFT_RELU: diff_src[i] = soft_relu_bwd(diff_dst[i], src[i]); break;
    case LOGISTIC: diff_src[i] = logistic_bwd(diff_dst[i], src[i]); break;
    default: return;
    }
}
