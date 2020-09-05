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
namespace sycl {
namespace math {

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

} // namespace math
} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
