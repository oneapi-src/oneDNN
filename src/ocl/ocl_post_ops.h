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

#if DT_F16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifndef POST_OP_DATA_T
#if DT_F16 == 1
#define POST_OP_DATA_T half
#else
#define POST_OP_DATA_T float
#endif
#endif

#ifndef DATA_MAX
#if DT_F16 == 1
#define DATA_MAX HALF_MAX
#elif DT_S8 == 1
#define DATA_MAX CHAR_MAX
#elif DT_U8 == 1
#define DATA_MAX UCHAR_MAX
#else
#define DATA_MAX FLT_MAX
#endif
#endif

POST_OP_DATA_T relu_fwd(POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return s > 0 ? s : s * alpha;
}
POST_OP_DATA_T relu_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return s > 0 ? dd : dd * alpha;
}

POST_OP_DATA_T linear_fwd(
        POST_OP_DATA_T s, POST_OP_DATA_T alpha, POST_OP_DATA_T beta) {
    return alpha * s + beta;
}
POST_OP_DATA_T linear_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T alpha) {
    return dd * alpha;
}

POST_OP_DATA_T bounded_relu_fwd(POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    s = s > 0 ? s : 0;
    return s > alpha ? alpha : s;
}
POST_OP_DATA_T bounded_relu_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return dd * (0 < s && s < alpha ? 1 : 0);
}

POST_OP_DATA_T soft_relu_fwd(POST_OP_DATA_T s) {
    return s < log((POST_OP_DATA_T)DATA_MAX) ? log1p(exp(s)) : s;
}
POST_OP_DATA_T soft_relu_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return dd / (1 + exp(-s));
}

POST_OP_DATA_T logistic_fwd(POST_OP_DATA_T s) {
    return 1 / (1 + exp(-s));
}
POST_OP_DATA_T logistic_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    POST_OP_DATA_T v = logistic_fwd(s);
    return dd * v * (1 - v);
}

POST_OP_DATA_T square_fwd(POST_OP_DATA_T s) {
    return s * s;
}
POST_OP_DATA_T square_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return dd * 2 * s;
}

POST_OP_DATA_T sqrt_fwd(POST_OP_DATA_T s) {
    return sqrt(s);
}
POST_OP_DATA_T sqrt_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return dd / (2 * sqrt(s));
}

POST_OP_DATA_T abs_fwd(POST_OP_DATA_T s) {
    return s > 0 ? s : -s;
}
POST_OP_DATA_T abs_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return s > 0 ? dd : s < 0 ? -dd : 0;
}

POST_OP_DATA_T tanh_fwd(POST_OP_DATA_T s) {
    return tanh(s);
}
POST_OP_DATA_T tanh_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    POST_OP_DATA_T e = tanh_fwd(s);
    return dd * (1 - e) * (1 + e);
}

POST_OP_DATA_T elu_fwd(POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return s > 0 ? s : alpha * expm1(s);
}
POST_OP_DATA_T elu_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return dd * (s > 0 ? 1 : alpha * exp(s));
}

POST_OP_DATA_T exp_fwd(POST_OP_DATA_T s) {
    return exp(s);
}
POST_OP_DATA_T exp_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return dd * exp_fwd(s);
}

POST_OP_DATA_T gelu_fwd(POST_OP_DATA_T s) {
    const float a = 0.797884f;
    const float b = 0.044715f;
    const float g = a * s * (1.0f + b * s * s);
    return (0.5f * s * (1.0f + tanh_fwd(g)));
}
POST_OP_DATA_T gelu_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    const float a = 0.797884f;
    const float b = 0.044715f;
    const float g = a * s * (1.0f + b * s * s);
    const float dg = a * (1.0f + 3.0f * b * s * s);
    return dd
            * (0.5f * (1.0f + tanh_fwd(g))
                    * (1.0f + s * (1.0f - tanh_fwd(g)) * dg));
}

POST_OP_DATA_T swish_fwd(POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return s / (1.0f + exp(-alpha * s));
}
POST_OP_DATA_T swish_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    POST_OP_DATA_T v = logistic_fwd(alpha * s);
    return dd * (v + s * alpha * v * (1.0f - v));
}

POST_OP_DATA_T log_fwd(POST_OP_DATA_T s) {
    return log(s);
}
POST_OP_DATA_T log_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return dd / s;
}

POST_OP_DATA_T clip_fwd(
        POST_OP_DATA_T s, POST_OP_DATA_T alpha, POST_OP_DATA_T beta) {
    s = s > alpha ? s : alpha;
    return s > beta ? beta : s;
}
POST_OP_DATA_T clip_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s,
        POST_OP_DATA_T alpha, POST_OP_DATA_T beta) {
    return dd * (alpha < s && s <= beta ? 1 : 0);
}

POST_OP_DATA_T pow_fwd(
        POST_OP_DATA_T s, POST_OP_DATA_T alpha, POST_OP_DATA_T beta) {
    return alpha * pow(s, beta);
}
POST_OP_DATA_T pow_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s,
        POST_OP_DATA_T alpha, POST_OP_DATA_T beta) {
    if (beta == 0) return 0;

    float v = pow_fwd(s, alpha * beta, beta - 1);
    return dd * v;
}

POST_OP_DATA_T fwd_eltwise(
        POST_OP_DATA_T x, POST_OP_DATA_T alpha_, POST_OP_DATA_T beta_) {
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
        case GELU: return gelu_fwd(x); break;
        case SWISH: return swish_fwd(x, alpha_); break;
        case LOG: return log_fwd(x); break;
        case CLIP: return clip_fwd(x, alpha_, beta_); break;
        case POW: return pow_fwd(x, alpha_, beta_); break;
        default: return x; break;
    }
#else
    return x;
#endif
}

POST_OP_DATA_T bwd_eltwise(POST_OP_DATA_T x, POST_OP_DATA_T y,
        POST_OP_DATA_T alpha_, POST_OP_DATA_T beta_) {
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
        case GELU: return gelu_bwd(x, y); break;
        case SWISH: return swish_bwd(x, y, alpha_); break;
        case LOG: return log_bwd(x, y); break;
        case CLIP: return clip_bwd(x, y, alpha_, beta_); break;
        case POW: return pow_bwd(x, y, alpha_, beta_); break;
        default: return x; break;
    }
#else
    return x;
#endif
}
#endif
