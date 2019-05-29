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

#ifndef OCL_POST_OPS_H
#define OCL_POST_OPS_H

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

DATA_T exp_fwd(DATA_T s) {
    return exp(s);
}
DATA_T exp_bwd(DATA_T dd, DATA_T s) {
    return dd * exp_fwd(s);
}

DATA_T fwd_eltwise(DATA_T x, DATA_T alpha_, DATA_T beta_) {
#ifdef ALG_KIND
    switch (ALG_KIND) {
    case RELU: return relu_fwd(x, alpha_); break;
    case LINEAR: return linear_fwd(x, alpha_, beta_); break;
    case BOUNDED_RELU: return bounded_relu_fwd(x, alpha_); break;
    case SOFT_RELU: return soft_relu_fwd(x); break;
    case LOGISTIC: return logistic_fwd(x); break;
    case TANH: return tanh_fwd(x); break;
    case ELU: return elu_fwd(x, alpha_); break;
    case SQUARE: return square_fwd(x); break;
    case SQRT: return sqrt_fwd(x); break;
    case ABS: return abs_fwd(x); break;
    case EXP: return exp_fwd(x); break;
    default: return x; break;
    }
#else
    return x;
#endif
}

DATA_T bwd_eltwise(DATA_T x, DATA_T y, DATA_T alpha_) {
#ifdef ALG_KIND
    switch (ALG_KIND) {
    case RELU: return relu_bwd(x, y, alpha_); break;
    case LINEAR: return linear_bwd(x, alpha_); break;
    case BOUNDED_RELU: return bounded_relu_bwd(x, y, alpha_); break;
    case SOFT_RELU: return soft_relu_bwd(x, y); break;
    case LOGISTIC: return logistic_bwd(x, y); break;
    case TANH: return tanh_bwd(x, y); break;
    case ELU: return elu_bwd(x, y, alpha_); break;
    case SQUARE: return square_bwd(x, y); break;
    case SQRT: return sqrt_bwd(x, y); break;
    case ABS: return abs_bwd(x, y); break;
    case EXP: return exp_bwd(x, y); break;
    default: return x; break;
    }
#else
    return x;
#endif
}
#endif
