/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_UINT128_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_UINT128_HPP

#include <cmath>
#include <cstdint>
#include <utility>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {

struct uint128_t {
public:
    // constructor
    uint128_t() = default;
    constexpr uint128_t(const uint64_t val) //
        : upper_64_(0), lower_64_(val) {}
    constexpr uint128_t(const uint64_t upper, const uint64_t lower) //
        : upper_64_(upper), lower_64_(lower) {}
#ifdef __SIZEOF_INT128__
    constexpr uint128_t(const __uint128_t val) //
        : upper_64_(uint64_t(val << 64)), lower_64_(uint64_t(val)) {}
#endif

    // cast
    inline operator uint64_t() const { return lower_64_; }
#ifdef __SIZEOF_INT128__
    inline operator __uint128_t() const {
        return (__uint128_t(upper_64_) << 64) | __uint128_t(lower_64_);
    }
#endif

    // helper
    inline int clz() const;
    inline uint64_t upper_64() const;
    inline uint64_t lower_64() const;
    inline static constexpr uint128_t one();
    inline static constexpr uint128_t zero();

    // compare
    bool operator==(const uint128_t &other) const;
    bool operator!=(const uint128_t &other) const;
    bool operator>(const uint128_t &other) const;
    bool operator<(const uint128_t &other) const;
    bool operator>=(const uint128_t &other) const;
    bool operator<=(const uint128_t &other) const;

    // bitwise or/and/xor
    uint128_t operator|(const uint128_t &rhs) const;
    uint128_t operator&(const uint128_t &rhs) const;
    uint128_t operator^(const uint128_t &rhs) const;
    uint128_t &operator|=(const uint128_t &rhs);
    uint128_t &operator&=(const uint128_t &rhs);
    uint128_t &operator^=(const uint128_t &rhs);

    // bitwise not
    uint128_t operator~() const;

    // logical shift
    uint128_t operator<<(const int bits) const;
    uint128_t operator>>(const int bits) const;
    uint128_t &operator<<=(const int bits);
    uint128_t &operator>>=(const int bits);

    // add/sub
    uint128_t operator+(const uint128_t &rhs) const;
    uint128_t operator-(const uint128_t &rhs) const;
    uint128_t &operator+=(const uint128_t &rhs);
    uint128_t &operator-=(const uint128_t &rhs);

    // mul
    uint128_t operator*(const uint128_t &rhs) const;
    uint128_t &operator*=(const uint128_t &rhs);

    // div/mod
    uint128_t operator/(const uint128_t &rhs) const;
    uint128_t operator%(const uint128_t &rhs) const;
    uint128_t &operator/=(const uint128_t &rhs);
    uint128_t &operator%=(const uint128_t &rhs);

private:
    // uint128 data storage
    uint64_t upper_64_;
    uint64_t lower_64_;

    // calculation implementation
    using qr_pair = std::pair<uint128_t, uint128_t>;
    uint128_t mul_impl(const uint128_t &lhs, const uint128_t &rhs) const;
    qr_pair div_mod_impl(const uint128_t &lhs, const uint128_t &rhs) const;
};

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
