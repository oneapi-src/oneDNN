/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_SYCL_SYCL_MATH_UTILS_HPP
#define GPU_SYCL_SYCL_MATH_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {
namespace math {

// Regular math functions.

// The idea is to reuse functions from common math utils whenever it's possible,
// usually that's the case when the functions do not use functions from a math
// library. Those functions that use such math functions should be
// re-implemented for SYCL.

template <typename T>
using rem_ref = typename utils::remove_reference<T>::type;

template <typename T, typename U = rem_ref<T>>
inline U div_up(const T a, const U b) {
    return (a + b - 1) / b;
}

template <typename T, typename U = rem_ref<T>>
inline U exp_fwd(T s) {
    return (U)::sycl::exp(s);
}

template <typename T, typename U = rem_ref<T>>
inline U gelu_erf_fwd(T s) {
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = static_cast<float>(s) * sqrt_2_over_2;
    return (U)(0.5f * static_cast<float>(s) * (1.f + ::sycl::erf(v)));
}

template <typename T, typename U = rem_ref<T>>
inline U log_fwd(T s) {
    return (U)(::sycl::log(s));
}

template <typename T, typename U = rem_ref<T>>
inline U gelu_tanh_fwd(T s) {
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    const float v
            = ::sycl::tanh(sqrt_2_over_pi * s * (1 + fitting_const * s * s));
    return (0.5f * s * (1.f + v));
}

template <typename T, typename U = rem_ref<T>>
inline U soft_relu_fwd(T s) {
    float exp_overflow_bound = 88.72283172607421875;
    float in = (float)s;
    return in < exp_overflow_bound ? (U)(::sycl::log1p(::sycl::exp(in)))
                                   : (U)in;
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U soft_relu_fwd(T s, A alpha) {
    float exp_overflow_bound = 88.72283172607421875;
    float in = (float)s * (float)alpha;
    float v = in < exp_overflow_bound ? (U)(::sycl::log1p(::sycl::exp(in)))
                                      : (U)in;
    return (U)(v / alpha);
}

template <typename T, typename U = rem_ref<T>>
inline U logsigmoid_fwd(T s) {
    return static_cast<U>(-soft_relu_fwd<T, U>(-s));
}

template <typename T, typename U = rem_ref<T>>
inline U mish_fwd(T s) {
    float e = soft_relu_fwd<float>(s);
    float l = (float)(::sycl::tanh<float>((float)e));
    float val = s * l;
    return (U)(val);
}

template <typename T, typename U = rem_ref<T>>
inline U logistic_fwd(T s) {
    // Here we avoid division/inverse by infinity as some architectures have
    // non-standard behavior
    float exp_overflow_bound = 88.72283172607421875;
    float in = (float)-s;
    return in < exp_overflow_bound ? (U)(1.f / (1.f + ::sycl::exp(in))) : 0.f;
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U swish_fwd(T s, A alpha) {
    return (U)(s * logistic_fwd<float>(alpha * s));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U soft_relu_v2_fwd(T s, A alpha) {
    return soft_relu_fwd<float>(s * alpha) / alpha;
}

template <typename T, typename U = rem_ref<T>>
inline U sqrt_fwd(T s) {
    return (U)(::sycl::sqrt((float)(s)));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U elu_fwd(T s, A alpha) {
    return s > 0 ? s : (U)(alpha * (::sycl::expm1((float)s)));
}

template <typename T, typename U = rem_ref<T>>
inline U abs_bwd(T dd, T s) {
    return s > 0 ? (U)(dd) : (U)(s < 0 ? -dd : 0.f);
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U elu_bwd(T dd, T s, A alpha) {
    return (U)(dd * (s > 0 ? 1.f : alpha * ::sycl::exp((float)s)));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U bounded_relu_bwd(T dd, T s, A alpha) {
    return (U)(dd * (0 < s && s <= alpha ? 1.f : 0.f));
}

template <typename T, typename U = rem_ref<T>>
inline U soft_relu_bwd(T dd, T s) {
    return static_cast<U>(dd * logistic_fwd<float>(s));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U soft_relu_bwd(T dd, T s, A alpha) {
    float in = (float)s * (float)alpha;
    return static_cast<U>(dd * logistic_fwd<float>(in));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U soft_relu_v2_bwd(T dd, T s, A alpha) {
    return static_cast<U>(soft_relu_bwd<float>(dd, s * alpha));
}

template <typename T, typename U = rem_ref<T>>
inline U logistic_bwd(T dd, T s) {
    float v = logistic_fwd<float>(s);
    return static_cast<U>(dd * v * (1.f - v));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U swish_bwd(T dd, T s, A alpha) {
    float v = logistic_fwd<float>(alpha * s);
    return static_cast<U>(dd * (v + s * alpha * v * (1.f - v)));
}

template <typename T, typename U = rem_ref<T>>
inline U logsigmoid_bwd(T dd, T s) {
    return (U)(soft_relu_bwd((float)dd, (float)(-s)));
}

template <typename T, typename U = rem_ref<T>>
inline U gelu_erf_bwd(T dd, T s) {
    const float two_over_sqrt_pi = 1.12837922573089599609375f;
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = s * sqrt_2_over_2;
    return static_cast<U>(dd * 0.5f
            * (1.f + ::sycl::erf(v)
                    + v * two_over_sqrt_pi * ::sycl::exp(-v * v)));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U relu_bwd_use_dst(T dd, T d, A alpha) {
    return static_cast<U>(d > 0 ? dd : (dd * alpha));
}

template <typename T, typename U = rem_ref<T>>
inline U hardswish_bwd(T dd, T s) {
    return (U)((s < 3 && s > -3) ? (dd * ((2.f * s + 3.f) / 6.f))
                                 : (s >= 3 ? dd : 0.f));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U hardsigmoid_bwd(T dd, T s, A alpha, A beta) {
    float v = alpha * s + beta;
    return static_cast<U>(v <= 0 ? 0.f : (v >= 1 ? 0.f : (dd * alpha)));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U elu_bwd_use_dst(T dd, T d, A alpha) {
    return (U)(dd * (d > 0 ? 1.f : (d + alpha)));
}

template <typename T, typename U = rem_ref<T>>
inline U tanh_bwd_use_dst(T dd, T d) {
    const float e = d;
    return static_cast<U>(dd * ((1.f - e) * (1.f + e)));
}

template <typename T, typename U = rem_ref<T>>
inline U exp_bwd(T dd, T s) {
    return static_cast<U>(dd * (::sycl::exp((float)s)));
}

template <typename T, typename U = rem_ref<T>>
inline U tanh_fwd(T s) {
    const float e = ::sycl::tanh((float)s);
    return (U)e;
}
template <typename T, typename U = rem_ref<T>>
inline U tanh_bwd(T dd, T s) {
    const float e = tanh_fwd<float>((float)s);
    return static_cast<U>(dd * (1.f - e) * (1.f + e));
}

template <typename T, typename U = rem_ref<T>>
inline U gelu_tanh_bwd(T dd, T s) {
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    float g = s * sqrt_2_over_pi * (1.f + fitting_const * s * s);
    float dg = sqrt_2_over_pi * (1.f + 3.f * fitting_const * s * s);
    float v = tanh_fwd(g);
    return static_cast<U>(dd * 0.5f * (1.f + v) * (1.f + s * (1.f - v) * dg));
}

template <typename T, typename U = rem_ref<T>>
inline U mish_bwd(T dd, T s) {
    const float tanh = tanh_fwd(soft_relu_fwd((float)s));
    const float srelu_bwd = soft_relu_bwd(1.0f, (float)s);
    const float derivative
            = tanh + s * srelu_bwd * (1.0f - ::sycl::pow((float)tanh, 2.0f));
    return static_cast<U>(dd * derivative);
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U pow_fwd(T s, A alpha, A beta) {
    return static_cast<U>(
            (float)alpha * static_cast<U>(::sycl::pow((float)s, (float)beta)));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U pow_bwd(T dd, T s, A alpha, A beta) {
    if (beta == 0) return (U)0.f;

    float v = pow_fwd((float)s, (float)(alpha * beta), beta - 1.0f);
    return (U)(dd * v);
}

template <typename T, typename U = rem_ref<T>>
inline U log_bwd(T dd, T s) {
    float val = 1.f / s;
    return (U)(dd * val);
}

template <typename T, typename A, typename U = rem_ref<T>>
inline typename utils::enable_if<nstl::is_integral<U>::value, U>::type relu_fwd(
        T s, A alpha) {
    return s > 0 ? s : (U)::sycl::rint(static_cast<float>(s * alpha));
}

template <typename T, typename A, typename U = rem_ref<T>>
inline typename utils::enable_if<!nstl::is_integral<U>::value, U>::type
relu_fwd(T s, A alpha) {
    return impl::math::relu_fwd(s, alpha);
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U linear_fwd(T s, A alpha, A beta) {
    return impl::math::linear_fwd(s, alpha, beta);
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U clip_fwd(T s, A alpha, A beta) {
    return impl::math::clip_fwd(s, alpha, beta);
}

template <typename T, typename A, typename U = rem_ref<T>>
inline U clip_v2_fwd(T s, A alpha, A beta) {
    return impl::math::clip_v2_fwd(s, alpha, beta);
}

// Math functions that work with `sycl::vec`.
template <typename T, int width, bool max>
inline ::sycl::vec<T, width> min_max_vec_impl(
        ::sycl::vec<T, width> vec_a, ::sycl::vec<T, width> vec_b) {
    auto vec_res = [&]() {
        if constexpr (max) {
            return vec_a > vec_b;
        } else {
            return vec_a < vec_b;
        }
    }();

    auto max_mask_a = vec_res * -1;
    auto max_mask_b = !max_mask_a;

    auto max_vec_a = vec_a * max_mask_a.template convert<T>();
    auto max_vec_b = vec_b * max_mask_b.template convert<T>();

    return max_vec_a + max_vec_b;
}

template <typename T, int width>
inline ::sycl::vec<T, width> max_vec(
        ::sycl::vec<T, width> vec_a, ::sycl::vec<T, width> vec_b) {
    return min_max_vec_impl<T, width, true>(vec_a, vec_b);
}

template <typename T, int width>
inline ::sycl::vec<T, width> min_vec(
        ::sycl::vec<T, width> vec_a, ::sycl::vec<T, width> vec_b) {
    return min_max_vec_impl<T, width, false>(vec_a, vec_b);
}

template <int width>
inline ::sycl::vec<float, width> relu_fwd(
        ::sycl::vec<float, width> src_vec, float alpha) {
    constexpr ::sycl::vec<float, width> zero_vec(0.0f);

    if (alpha == 0.0f) {
        return max_vec(src_vec, zero_vec);
    } else {
        ::sycl::vec<float, width> alpha_vec(alpha);
        auto src_copy_vec = src_vec;
        // Mask to nullify elements that are greater than 0 in src_vec.
        auto src_vec_mask = (src_vec < zero_vec) * -1;
        // Scale all elements in `src_vec`.
        src_vec *= alpha_vec;
        // Nullify elements that are greater than 0.
        src_vec *= src_vec_mask.template convert<float>();
        // Combine scaled negative elements with the elements that are greater
        // than 0 in `src_copy_vec`.
        return src_vec + max_vec(src_copy_vec, zero_vec);
    }
}

template <int width>
inline ::sycl::vec<float, width> linear_fwd(
        ::sycl::vec<float, width> src_vec, float alpha, float beta) {
    ::sycl::vec<float, width> alpha_vec(alpha);
    ::sycl::vec<float, width> beta_vec(beta);
    return alpha_vec * src_vec + beta_vec;
}

} // namespace math
} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
