/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GPU_OCL_OCL_ELTWISE_H
#define GPU_OCL_OCL_ELTWISE_H

#if WITH_ELTWISE
#include "gpu/ocl/ocl_types.h"

#if DT_F16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
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

POST_OP_DATA_T relu_fwd(POST_OP_DATA_T s, float alpha) {
    return s > 0 ? s : ((alpha == 0) ? 0 : s * alpha);
}
POST_OP_DATA_T relu_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha) {
    return s > 0 ? dd : dd * alpha;
}
POST_OP_DATA_T relu_bwd_use_dst(
        POST_OP_DATA_T dd, POST_OP_DATA_T d, float alpha) {
    return d > 0 ? dd : dd * alpha;
}

POST_OP_DATA_T linear_fwd(POST_OP_DATA_T s, float alpha, float beta) {
    return alpha * s + beta;
}
POST_OP_DATA_T linear_bwd(POST_OP_DATA_T dd, float alpha) {
    return dd * alpha;
}

POST_OP_DATA_T soft_relu_fwd(POST_OP_DATA_T s, float alpha) {
    s = alpha * s;
    POST_OP_DATA_T v = (s < log((POST_OP_DATA_T)DATA_MAX) ? log1p(exp(s)) : s);
    return v / alpha;
}
POST_OP_DATA_T soft_relu_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha) {
    s = alpha * s;
    return dd / ((POST_OP_DATA_T)1 + exp(-s));
}

POST_OP_DATA_T logistic_fwd(POST_OP_DATA_T s) {
    return (AS_POST_OP_DATA_T(1.0)) / (AS_POST_OP_DATA_T(1.0) + exp(-s));
}
POST_OP_DATA_T logistic_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    POST_OP_DATA_T v = logistic_fwd(s);
    return dd * v * (1 - v);
}
POST_OP_DATA_T logistic_bwd_use_dst(POST_OP_DATA_T dd, POST_OP_DATA_T d) {
    return dd * d * (1 - d);
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
POST_OP_DATA_T sqrt_bwd_use_dst(POST_OP_DATA_T dd, POST_OP_DATA_T d) {
    return dd / (2 * d);
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
POST_OP_DATA_T tanh_bwd_use_dst(POST_OP_DATA_T dd, POST_OP_DATA_T d) {
    return dd * (1 - d) * (1 + d);
}

POST_OP_DATA_T mish_fwd(POST_OP_DATA_T s) {
    return s * tanh_fwd(soft_relu_fwd(s, 1.f));
}
POST_OP_DATA_T mish_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    const POST_OP_DATA_T tanh = tanh_fwd(soft_relu_fwd(s, (POST_OP_DATA_T)1.0));
    const POST_OP_DATA_T srelu_bwd
            = soft_relu_bwd((POST_OP_DATA_T)1.0, s, (POST_OP_DATA_T)1.0);
    const POST_OP_DATA_T derivative = tanh
            + s * srelu_bwd
                    * ((POST_OP_DATA_T)1 - pow(tanh, (POST_OP_DATA_T)2.0));
    return dd * derivative;
}

POST_OP_DATA_T elu_fwd(POST_OP_DATA_T s, float alpha) {
    return s > 0 ? s : alpha * expm1(s);
}
POST_OP_DATA_T elu_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha) {
    return dd * (s > 0 ? 1 : alpha * exp(s));
}
POST_OP_DATA_T elu_bwd_use_dst(
        POST_OP_DATA_T dd, POST_OP_DATA_T d, float alpha) {
    return dd * (d > 0 ? 1 : d + alpha);
}

POST_OP_DATA_T exp_fwd(POST_OP_DATA_T s) {
    return exp(s);
}

POST_OP_DATA_T exp_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return dd * exp_fwd(s);
}
POST_OP_DATA_T exp_bwd_use_dst(POST_OP_DATA_T dd, POST_OP_DATA_T d) {
    return dd * d;
}

POST_OP_DATA_T gelu_tanh_fwd(POST_OP_DATA_T s) {
    const POST_OP_DATA_T sqrt_2_over_pi
            = AS_POST_OP_DATA_T(0.79788458347320556640625);
    const POST_OP_DATA_T fitting_const = AS_POST_OP_DATA_T(0.044715);
    const POST_OP_DATA_T g = sqrt_2_over_pi * s
            * ((POST_OP_DATA_T)1.0 + fitting_const * s * s);
    return ((POST_OP_DATA_T)0.5 * s * ((POST_OP_DATA_T)1.0 + tanh_fwd(g)));
}
POST_OP_DATA_T gelu_tanh_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    const POST_OP_DATA_T sqrt_2_over_pi
            = AS_POST_OP_DATA_T(0.79788458347320556640625);
    const POST_OP_DATA_T fitting_const = AS_POST_OP_DATA_T(0.044715);
    const POST_OP_DATA_T g = sqrt_2_over_pi * s
            * ((POST_OP_DATA_T)1.0 + fitting_const * s * s);
    const POST_OP_DATA_T dg = sqrt_2_over_pi
            * ((POST_OP_DATA_T)1.0
                    + (POST_OP_DATA_T)3.0 * fitting_const * s * s);
    const POST_OP_DATA_T v = tanh_fwd(g);
    return dd * (POST_OP_DATA_T)0.5 * ((POST_OP_DATA_T)1.0 + v)
            * ((POST_OP_DATA_T)1.0 + s * ((POST_OP_DATA_T)1.0 - v) * dg);
}

POST_OP_DATA_T swish_fwd(POST_OP_DATA_T s, float alpha) {
    POST_OP_DATA_T w = -alpha * s;
    return s / ((POST_OP_DATA_T)1.0 + exp(w));
}
POST_OP_DATA_T swish_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha) {
    POST_OP_DATA_T v = logistic_fwd(alpha * s);
    return dd * (v + s * alpha * v * ((POST_OP_DATA_T)1.0 - v));
}

POST_OP_DATA_T log_fwd(POST_OP_DATA_T s) {
    return log(s);
}
POST_OP_DATA_T log_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return dd / s;
}

POST_OP_DATA_T clip_fwd(POST_OP_DATA_T s, float alpha, float beta) {
    s = s > alpha ? s : alpha;
    return s > beta ? beta : s;
}
POST_OP_DATA_T clip_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha, float beta) {
    return dd * (alpha < s && s <= beta ? 1 : 0);
}

POST_OP_DATA_T clip_v2_fwd(POST_OP_DATA_T s, float alpha, float beta) {
    s = s > alpha ? s : alpha;
    return s < beta ? s : beta;
}
POST_OP_DATA_T clip_v2_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha, float beta) {
    return dd * (alpha < s && s < beta ? 1 : 0);
}
POST_OP_DATA_T clip_v2_bwd_use_dst(
        POST_OP_DATA_T dd, POST_OP_DATA_T d, float alpha, float beta) {
    return dd * (alpha < d && d < beta ? 1 : 0);
}

POST_OP_DATA_T pow_fwd(POST_OP_DATA_T s, float alpha, float beta) {
    return alpha * pow(s, (POST_OP_DATA_T)(beta));
}
POST_OP_DATA_T pow_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha, float beta) {
    if (beta == 0) return 0;

    POST_OP_DATA_T v = pow_fwd(s, alpha * beta, beta - 1);
    return dd * v;
}

POST_OP_DATA_T gelu_erf_fwd(POST_OP_DATA_T s) {
    const POST_OP_DATA_T sqrt_2_over_2
            = AS_POST_OP_DATA_T(0.707106769084930419921875);
    POST_OP_DATA_T v = s * sqrt_2_over_2;
    return (POST_OP_DATA_T)0.5 * s * ((POST_OP_DATA_T)1.0 + erf(v));
}

POST_OP_DATA_T gelu_erf_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    const POST_OP_DATA_T two_over_sqrt_pi
            = AS_POST_OP_DATA_T(1.12837922573089599609375);
    const POST_OP_DATA_T sqrt_2_over_2
            = AS_POST_OP_DATA_T(0.707106769084930419921875);
    POST_OP_DATA_T v = s * sqrt_2_over_2;
    return dd * (POST_OP_DATA_T)0.5
            * ((POST_OP_DATA_T)1.0 + erf(v)
                    + v * two_over_sqrt_pi * exp(-v * v));
}

float round_fwd(POST_OP_DATA_T s) {
    return (float)rint((float)s);
}

POST_OP_DATA_T hardsigmoid_fwd(POST_OP_DATA_T s, float alpha, float beta) {
    POST_OP_DATA_T v = (POST_OP_DATA_T)alpha * s + (POST_OP_DATA_T)beta;
    return v <= AS_POST_OP_DATA_T(0.0)    ? AS_POST_OP_DATA_T(0.0)
            : v >= AS_POST_OP_DATA_T(1.0) ? AS_POST_OP_DATA_T(1.0)
                                          : v;
}
POST_OP_DATA_T hardsigmoid_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha, float beta) {
    POST_OP_DATA_T v = alpha * s + beta;
    return v <= 0.f ? 0.f : v >= 1.f ? 0.f : dd * alpha;
}

POST_OP_DATA_T hardswish_fwd(POST_OP_DATA_T s, float alpha, float beta) {
    return s * hardsigmoid_fwd(s, alpha, beta);
}
POST_OP_DATA_T hardswish_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, float alpha, float beta) {
    POST_OP_DATA_T v = alpha * s + beta;
    POST_OP_DATA_T w = 2.f * alpha * s + beta;
    return (v <= 0.f ? 0.f : v >= 1.f ? dd : dd * w);
}

float fwd_eltwise_common(int eltwise_alg, POST_OP_DATA_T x, float alpha_,
        float beta_, float scale_) {
    switch (eltwise_alg) {
        case RELU: return scale_ * relu_fwd(x, alpha_); break;
        case LINEAR: return scale_ * linear_fwd(x, alpha_, beta_); break;
        case SOFT_RELU: return scale_ * soft_relu_fwd(x, alpha_); break;
        case MISH: return scale_ * mish_fwd(x); break;
        case LOGISTIC: return scale_ * logistic_fwd(x); break;
        case TANH: return scale_ * tanh_fwd(x); break;
        case ELU: return scale_ * elu_fwd(x, alpha_); break;
        case SQUARE: return scale_ * square_fwd(x); break;
        case SQRT: return scale_ * sqrt_fwd(x); break;
        case ABS: return scale_ * abs_fwd(x); break;
        case EXP: return scale_ * exp_fwd(x); break;
        case GELU_TANH: return scale_ * gelu_tanh_fwd(x); break;
        case SWISH: return scale_ * swish_fwd(x, alpha_); break;
        case LOG: return scale_ * log_fwd(x); break;
        case CLIP: return scale_ * clip_fwd(x, alpha_, beta_); break;
        case CLIP_V2: return scale_ * clip_v2_fwd(x, alpha_, beta_); break;
        case POW: return scale_ * pow_fwd(x, alpha_, beta_); break;
        case GELU_ERF: return scale_ * gelu_erf_fwd(x); break;
        case ROUND: return scale_ * round_fwd(x); break;
        case HARDSWISH: return scale_ * hardswish_fwd(x, alpha_, beta_); break;
        case HARDSIGMOID:
            return scale_ * hardsigmoid_fwd(x, alpha_, beta_);
            break;
        case RELU_DST: return scale_ * relu_fwd(x, alpha_); break;
        case LOGISTIC_DST: return scale_ * logistic_fwd(x); break;
        case TANH_DST: return scale_ * tanh_fwd(x); break;
        case ELU_DST: return scale_ * elu_fwd(x, alpha_); break;
        case SQRT_DST: return scale_ * sqrt_fwd(x); break;
        case EXP_DST: return scale_ * exp_fwd(x); break;
        case CLIP_V2_DST: return scale_ * clip_v2_fwd(x, alpha_, beta_); break;
        default: return x; break;
    }
}

float fwd_eltwise(POST_OP_DATA_T x, float alpha_, float beta_, float scale_) {
#ifdef ELTWISE_ALG
    return fwd_eltwise_common(ELTWISE_ALG, x, alpha_, beta_, scale_);
#else
    return x;
#endif
}

float bwd_eltwise(
        POST_OP_DATA_T x, POST_OP_DATA_T y, float alpha_, float beta_) {
#ifdef ELTWISE_ALG
    switch (ELTWISE_ALG) {
        case RELU: return relu_bwd(x, y, alpha_); break;
        case LINEAR: return linear_bwd(x, alpha_); break;
        case SOFT_RELU: return soft_relu_bwd(x, y, alpha_); break;
        case MISH: return mish_bwd(x, y); break;
        case LOGISTIC: return logistic_bwd(x, y); break;
        case TANH: return tanh_bwd(x, y); break;
        case ELU: return elu_bwd(x, y, alpha_); break;
        case SQUARE: return square_bwd(x, y); break;
        case SQRT: return sqrt_bwd(x, y); break;
        case ABS: return abs_bwd(x, y); break;
        case EXP: return exp_bwd(x, y); break;
        case GELU_TANH: return gelu_tanh_bwd(x, y); break;
        case SWISH: return swish_bwd(x, y, alpha_); break;
        case LOG: return log_bwd(x, y); break;
        case CLIP: return clip_bwd(x, y, alpha_, beta_); break;
        case CLIP_V2: return clip_v2_bwd(x, y, alpha_, beta_); break;
        case POW: return pow_bwd(x, y, alpha_, beta_); break;
        case GELU_ERF: return gelu_erf_bwd(x, y); break;
        case HARDSWISH: return hardswish_bwd(x, y, alpha_, beta_); break;
        case HARDSIGMOID: return hardsigmoid_bwd(x, y, alpha_, beta_); break;
        case RELU_DST: return relu_bwd_use_dst(x, y, alpha_); break;
        case LOGISTIC_DST: return logistic_bwd_use_dst(x, y); break;
        case TANH_DST: return tanh_bwd_use_dst(x, y); break;
        case ELU_DST: return elu_bwd_use_dst(x, y, alpha_); break;
        case SQRT_DST: return sqrt_bwd_use_dst(x, y); break;
        case EXP_DST: return exp_bwd_use_dst(x, y); break;
        case CLIP_V2_DST:
            return clip_v2_bwd_use_dst(x, y, alpha_, beta_);
            break;

        default: return x; break;
    }
#else
    return x;
#endif
}

#endif // WITH_ELTWISE

#endif // GPU_OCL_OCL_ELTWISE_H
