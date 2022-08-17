/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef UTILS_UTILS_HPP
#define UTILS_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <type_traits>

#include "interface/c_types_map.hpp"

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

#ifndef MAYBE_UNUSED
#define MAYBE_UNUSED(x) UNUSED(x)
#endif

#ifndef assertm
#define assertm(exp, msg) assert(((void)(msg), (exp)))
#endif

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

#define CHECK(f) \
    do { \
        status_t _status_ = f; \
        if (_status_ != status::success) return _status_; \
    } while (0)

#ifndef NDEBUG
#define DEBUG_PRINT_ERROR(message) \
    do { \
        std::cerr << "ERROR: " << message << std::endl; \
    } while (0)
#else
#define DEBUG_PRINT_ERROR(message)
#endif

inline static size_t size_of(data_type_t dtype) {
    switch (dtype) {
        case data_type::f32:
        case data_type::s32: return 4U;
        case data_type::s8:
        case data_type::u8: return 1U;
        case data_type::f16:
        case data_type::bf16: return 2U;
        default: return 0;
    }
}

inline static size_t prod(const std::vector<dim_t> &shape) {
    if (shape.empty()) return 0;

    size_t p = (std::accumulate(
            shape.begin(), shape.end(), size_t(1), std::multiplies<dim_t>()));

    return p;
}

inline static size_t size_of(
        const std::vector<dim_t> &shape, data_type_t dtype) {
    return prod(shape) * size_of(dtype);
}

template <typename T, typename P>
constexpr bool one_of(T val, P item) {
    return (T)val == item;
}

template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return (T)val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) {
    return one_of(nullptr, ptrs...);
}

template <typename T>
inline bool any_le(const std::vector<T> &v, T i) {
    return std::any_of(v.begin(), v.end(), [i](T k) { return k <= i; });
}

template <typename T, typename R = T>
inline R array_product(const T *arr, size_t size) {
    if (size == 0) return 0;

    R prod = 1;
    for (size_t i = 0; i < size; ++i) {
        prod *= arr[i];
    }

    return prod;
}

template <typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

template <typename T, typename U>
inline void array_set(T *arr, const U &val, size_t size) {
    for (size_t i = 0; i < size; ++i)
        arr[i] = static_cast<T>(val);
}

// solve the Greatest Common Divisor
inline size_t gcd(size_t a, size_t b) {
    size_t temp = 0;
    while (b != 0) {
        temp = a;
        a = b;
        b = temp % b;
    }
    return a;
}

// solve the Least Common Multiple
inline size_t lcm(size_t a, size_t b) {
    return a * b / gcd(a, b);
}

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
template <typename T>
static size_t hash_combine(size_t seed, const T &v) {
    return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Returns a value of type T by reinterpretting the representation of the input
// value (part of C++20).
//
// Provides a safe implementation of type punning.
//
// Constraints:
// - U and T must have the same size
// - U and T must be trivially copyable
template <typename T, typename U>
inline T bit_cast(const U &u) {
    static_assert(sizeof(T) == sizeof(U), "Bit-casting must preserve size.");
    // Use std::is_pod as older GNU versions do not support
    // std::is_trivially_copyable.
    static_assert(std::is_pod<T>::value, "T must be trivially copyable.");
    static_assert(std::is_pod<U>::value, "U must be trivially copyable.");

    T t;
    std::memcpy(&t, &u, sizeof(U));
    return t;
}

inline int float2int(float x) {
    return bit_cast<int>(x);
}

int getenv(const char *name, char *buffer, int buffer_size);
int getenv_int(const char *name, int default_value);
int getenv_int_user(const char *name, int default_value);
int getenv_int_internal(const char *name, int default_value);
std::string getenv_string_user(const char *name);
bool check_verbose_string_user(const char *name, const char *expected);

inline std::string thread_id_to_str(std::thread::id id) {
    std::stringstream ss;
    ss << id;
    return ss.str();
}

inline int div_and_ceil(float x, float y) {
    return std::ceil(x / y);
}

inline int div_and_floor(float x, float y) {
    return std::floor(x / y);
}

#define DNNL_GRAPH_DISALLOW_COPY_AND_ASSIGN(T) \
    T(const T &) = delete; \
    T &operator=(const T &) = delete; // NOLINT

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
