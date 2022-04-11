/*******************************************************************************
* Copyright 2022 Intel Corporation
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
        ::sycl::vec<float, 8> alpha_vec(alpha);
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
} // namespace math
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
