/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_GEMM_UTILS_HPP
#define GPU_INTEL_JIT_GEMM_UTILS_HPP

#include <stdexcept>

#include "common/math_utils.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <typename T>
static inline constexpr bool equal(T t) {
    return true;
}
template <typename T1, typename T2>
static inline constexpr bool equal(T1 t1, T2 t2) {
    return (t1 == t2);
}
template <typename T1, typename T2, typename... To>
static inline constexpr bool equal(T1 t1, T2 t2, To... to) {
    return (t1 == t2) && equal(t2, to...);
}

template <typename T>
static inline constexpr T clamp(T val, T lo, T hi) {
    return std::min<T>(hi, std::max<T>(lo, val));
}

static inline int div_up(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

// Round value down to a multiple of factor.
static inline int align_down(int value, int factor) {
    return factor * (value / factor);
}

// Round value up to a multiple of factor.
static inline int align_up(int value, int factor) {
    return factor * div_up(value, factor);
}

using dnnl::impl::math::gcd;

template <typename T>
static inline T lcm(T x, T y) {
    if (x == 0 || y == 0) return 0;
    return dnnl::impl::math::lcm(x, y);
}

static inline int largest_pow2_divisor(int x) {
    return x & ~(x - 1);
}

class stub_exception : public std::runtime_error {
public:
    stub_exception()
        : std::runtime_error("Functionality not yet implemented") {}
};

[[noreturn]] static inline void stub() {
    throw stub_exception();
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* GPU_INTEL_JIT_GEMM_UTILS_HPP */
