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

#ifndef GPU_OCL_OCL_POST_OPS_H
#define GPU_OCL_OCL_POST_OPS_H

#if WITH_ELTWISE

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

#ifndef ELTWISE_ALPHA0
#define ELTWISE_ALPHA0 0
#endif

POST_OP_DATA_T relu_fwd(POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return s > 0 ? s : (ELTWISE_ALPHA0 ? 0 : s * alpha);
}
POST_OP_DATA_T relu_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return s > 0 ? dd : dd * alpha;
}
POST_OP_DATA_T relu_bwd_use_dst(
        POST_OP_DATA_T dd, POST_OP_DATA_T d, POST_OP_DATA_T alpha) {
    return d > 0 ? dd : dd * alpha;
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
    return dd * (0 < s && s <= alpha ? 1 : 0);
}

POST_OP_DATA_T soft_relu_fwd(POST_OP_DATA_T s) {
    return s < log((POST_OP_DATA_T)DATA_MAX) ? log1p(exp(s)) : s;
}
POST_OP_DATA_T soft_relu_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    return dd / (1 + exp(-s));
}

POST_OP_DATA_T logistic_fwd(POST_OP_DATA_T s) {
    return 1.0f / (1.0f + exp(-s));
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

POST_OP_DATA_T elu_fwd(POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return s > 0 ? s : alpha * expm1(s);
}
POST_OP_DATA_T elu_bwd(
        POST_OP_DATA_T dd, POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    return dd * (s > 0 ? 1 : alpha * exp(s));
}
POST_OP_DATA_T elu_bwd_use_dst(
        POST_OP_DATA_T dd, POST_OP_DATA_T d, POST_OP_DATA_T alpha) {
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
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    const float g = sqrt_2_over_pi * s * (1.f + fitting_const * s * s);
    return (0.5f * s * (1.f + tanh_fwd(g)));
}
POST_OP_DATA_T gelu_tanh_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    const float g = sqrt_2_over_pi * s * (1.f + fitting_const * s * s);
    const float dg = sqrt_2_over_pi * (1.f + 3.f * fitting_const * s * s);
    const float v = tanh_fwd(g);
    return dd * 0.5f * (1.f + v) * (1.f + s * (1.f - v) * dg);
}

POST_OP_DATA_T swish_fwd(POST_OP_DATA_T s, POST_OP_DATA_T alpha) {
    float w = -alpha * s;
    return s / (1.0f + exp(w));
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

POST_OP_DATA_T gelu_erf_fwd(POST_OP_DATA_T s) {
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = s * sqrt_2_over_2;
    return 0.5f * s * (1.f + erf(v));
}

POST_OP_DATA_T gelu_erf_bwd(POST_OP_DATA_T dd, POST_OP_DATA_T s) {
    const float two_over_sqrt_pi = 1.12837922573089599609375f;
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = s * sqrt_2_over_2;
    return dd * 0.5f * (1.f + erf(v) + v * two_over_sqrt_pi * exp(-v * v));
}

POST_OP_DATA_T fwd_eltwise(POST_OP_DATA_T x, POST_OP_DATA_T alpha_,
        POST_OP_DATA_T beta_, POST_OP_DATA_T scale_) {
#ifdef ELTWISE_ALG
    switch (ELTWISE_ALG) {
        case RELU: return scale_ * relu_fwd(x, alpha_); break;
        case LINEAR: return scale_ * linear_fwd(x, alpha_, beta_); break;
        case BOUNDED_RELU: return scale_ * bounded_relu_fwd(x, alpha_); break;
        case SOFT_RELU: return scale_ * soft_relu_fwd(x); break;
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
        case POW: return scale_ * pow_fwd(x, alpha_, beta_); break;
        case GELU_ERF: return scale_ * gelu_erf_fwd(x); break;

        case RELU_DST: return scale_ * relu_fwd(x, alpha_); break;
        case LOGISTIC_DST: return scale_ * logistic_fwd(x); break;
        case TANH_DST: return scale_ * tanh_fwd(x); break;
        case ELU_DST: return scale_ * elu_fwd(x, alpha_); break;
        case SQRT_DST: return scale_ * sqrt_fwd(x); break;
        case EXP_DST: return scale_ * exp_fwd(x); break;

        default: return x; break;
    }
#else
    return x;
#endif
}

POST_OP_DATA_T bwd_eltwise(POST_OP_DATA_T x, POST_OP_DATA_T y,
        POST_OP_DATA_T alpha_, POST_OP_DATA_T beta_) {
#ifdef ELTWISE_ALG
    switch (ELTWISE_ALG) {
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
        case GELU_TANH: return gelu_tanh_bwd(x, y); break;
        case SWISH: return swish_bwd(x, y, alpha_); break;
        case LOG: return log_bwd(x, y); break;
        case CLIP: return clip_bwd(x, y, alpha_, beta_); break;
        case POW: return pow_bwd(x, y, alpha_, beta_); break;
        case GELU_ERF: return gelu_erf_bwd(x, y); break;

        case RELU_DST: return relu_bwd_use_dst(x, y, alpha_); break;
        case LOGISTIC_DST: return logistic_bwd_use_dst(x, y); break;
        case TANH_DST: return tanh_bwd_use_dst(x, y); break;
        case ELU_DST: return elu_bwd_use_dst(x, y, alpha_); break;
        case SQRT_DST: return sqrt_bwd_use_dst(x, y); break;
        case EXP_DST: return exp_bwd_use_dst(x, y); break;

        default: return x; break;
    }
#else
    return x;
#endif
}

#endif // WITH_ELTWISE

#endif
