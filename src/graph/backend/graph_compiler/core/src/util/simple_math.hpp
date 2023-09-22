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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_SIMPLE_MATH_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_SIMPLE_MATH_HPP
#include <cstdint>
#include <immintrin.h>
#include <stddef.h>
#include <util/compiler_macros.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {
static constexpr size_t divide_and_ceil(size_t x, size_t y) {
    return (x + y - 1) / y;
}

static constexpr size_t rnd_up(const size_t a, const size_t b) {
    return (divide_and_ceil(a, b) * b);
}

static constexpr size_t rnd_dn(const size_t a, const size_t b) {
    return (a / b) * b;
}

inline bool is_power_of_2(const uint64_t val) {
    return (val > 0) && ((val & (val - 1)) == 0);
}

// get leading zeros
inline int clz(const uint32_t val) {
#if SC_IS_MSVC()
    return _lzcnt_u32(val);
#else
    return __builtin_clz(val);
#endif
}

inline int clz(const uint64_t val) {
#if SC_IS_MSVC()
    return _lzcnt_u64(val);
#else
    return __builtin_clzll(val);
#endif
}

// get trailing zeros
inline int ctz(const uint32_t val) {
#if SC_IS_MSVC()
    return _tzcnt_u32(val);
#else
    return __builtin_ctz(val);
#endif
}

inline int ctz(const uint64_t val) {
#if SC_IS_MSVC()
    return _tzcnt_u64(val);
#else
    return __builtin_ctzll(val);
#endif
}

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
