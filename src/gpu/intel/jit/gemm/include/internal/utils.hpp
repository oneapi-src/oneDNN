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

#ifndef GEMMSTONE_GUARD_UTILS_HPP
#define GEMMSTONE_GUARD_UTILS_HPP
#include "common/cpp_compat.hpp"

#include <stdexcept>

#ifdef __cpp_lib_source_location
#include <source_location>
#endif

#include "common/math_utils.hpp"
#include "common/utils.hpp"

#include "namespace_start.hxx"

using dnnl::impl::utils::one_of;
static inline int ilog2(size_t x) {
    return math::ilog2q(x);
}

template <typename T> static inline constexpr bool equal(T t) { return true; }
template <typename T1, typename T2> static inline constexpr bool equal(T1 t1, T2 t2) { return (t1 == t2); }
template <typename T1, typename T2, typename... To> static inline constexpr bool equal(T1 t1, T2 t2, To... to) {
    return (t1 == t2) && equal(t2, to...);
}

template <typename T> static inline constexpr T clamp(T val, T lo, T hi)
{
    return std::min<T>(hi, std::max<T>(lo, val));
}

static inline int div_up(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

// Round value down to a multiple of factor.
static inline int align_down(int value, int factor)
{
    return factor * (value / factor);
}

// Round value up to a multiple of factor.
static inline int align_up(int value, int factor)
{
    return factor * div_up(value, factor);
}

using dnnl::impl::math::gcd;

template <typename T> static inline T lcm(T x, T y)
{
    if (x == 0 || y == 0) return 0;
    return dnnl::impl::math::lcm(x, y);
}

static inline int largest_pow2_divisor(int x)
{
    return x & ~(x - 1);
}


class stub_exception : public std::runtime_error
{
public:
    stub_exception(const char *msg = unimpl()) : std::runtime_error(msg) {}
    stub_exception(const char *msg, const char *file, size_t line)
        : std::runtime_error(std::string(msg) + " (at " + std::string(file) + ":" + std::to_string(line) + ")") {}
    stub_exception(const char *file, size_t line) : stub_exception(unimpl(), file, line) {}

protected:
    static const char *unimpl() { return "Functionality is unimplemented"; }
};

class hw_unsupported_exception : public std::runtime_error {
public:
    hw_unsupported_exception() : std::runtime_error("Unsupported in hardware") {}
};

#if defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
[[noreturn]] static inline void stub(
                                     std::source_location where = std::source_location::current()) {
    throw stub_exception(where.file_name(), where.line());
}

[[noreturn]] static inline void stub(const char * msg,
                                     std::source_location where = std::source_location::current()) {
    throw stub_exception(msg, where.file_name(), where.line());
}

#else
[[noreturn]] static inline void stub() {
    throw stub_exception();
}

[[noreturn]] static inline void stub(const char * msg) {
    throw stub_exception(msg);
}
#endif

[[noreturn]] static inline void hw_unsupported()
{
    throw hw_unsupported_exception();
}

static inline void noop() {}

#include "namespace_end.hxx"

#endif /* GEMMSTONE_GUARD_UTILS_HPP */
