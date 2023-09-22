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
#include <stdexcept>

#include "simple_math.hpp"
#include "uint128.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {

/*******************************************************************************
 ********************** Function definition of uint64_t ************************
 *******************************************************************************/

// count leading zero of 128 bit val
inline int uint128_t::clz() const {
    return (*this).upper_64_ == 0 //
            ? utils::clz((*this).lower_64_) + 64
            : utils::clz((*this).upper_64_);
}
// get upper/lower
inline uint64_t uint128_t::upper_64() const {
    return (*this).upper_64_;
}
inline uint64_t uint128_t::lower_64() const {
    return (*this).lower_64_;
}
// get constant
inline constexpr uint128_t uint128_t::one() {
    return uint128_t(UINT64_C(1));
}
inline constexpr uint128_t uint128_t::zero() {
    return uint128_t(UINT64_C(0));
}

// compare
bool uint128_t::operator==(const uint128_t &other) const {
    return (*this).upper_64_ == other.upper_64_
            && (*this).lower_64_ == other.lower_64_;
}
bool uint128_t::operator!=(const uint128_t &other) const {
    return (*this).upper_64_ != other.upper_64_
            || (*this).lower_64_ != other.lower_64_;
}
bool uint128_t::operator>(const uint128_t &other) const {
    return (*this).upper_64_ == other.upper_64_
            ? (*this).lower_64_ > other.lower_64_
            : (*this).upper_64_ > other.upper_64_;
}
bool uint128_t::operator<(const uint128_t &other) const {
    return ((*this).upper_64_ == other.upper_64_)
            ? (*this).lower_64_ < other.lower_64_
            : (*this).upper_64_ < other.upper_64_;
}
bool uint128_t::operator>=(const uint128_t &other) const {
    return (*this) == other || (*this) > other;
}
bool uint128_t::operator<=(const uint128_t &other) const {
    return (*this) == other || (*this) < other;
}

// bitwise or
uint128_t uint128_t::operator|(const uint128_t &rhs) const {
    return uint128_t((*this).upper_64_ | rhs.upper_64_,
            (*this).lower_64_ | rhs.lower_64_);
}
uint128_t &uint128_t::operator|=(const uint128_t &rhs) {
    *this = *this | rhs;
    return *this;
}
// bitwise and
uint128_t uint128_t::operator&(const uint128_t &rhs) const {
    return uint128_t((*this).upper_64_ & rhs.upper_64_,
            (*this).lower_64_ & rhs.lower_64_);
}
uint128_t &uint128_t::operator&=(const uint128_t &rhs) {
    *this = *this & rhs;
    return *this;
}
// bitwise xor
uint128_t uint128_t::operator^(const uint128_t &rhs) const {
    return uint128_t((*this).upper_64_ ^ rhs.upper_64_,
            (*this).lower_64_ ^ rhs.lower_64_);
}
uint128_t &uint128_t::operator^=(const uint128_t &rhs) {
    *this = *this ^ rhs;
    return *this;
}

// bitwise not
uint128_t uint128_t::operator~() const {
    return uint128_t(~(*this).upper_64_, ~(*this).lower_64_);
}

// add
uint128_t uint128_t::operator+(const uint128_t &rhs) const {
    uint64_t lower = (*this).lower_64_ + rhs.lower_64_;
    uint64_t upper = (*this).upper_64_ + rhs.upper_64_;
    uint64_t carry = uint64_t(lower < (*this).lower_64_);
    return uint128_t(upper + carry, lower);
}
uint128_t &uint128_t::operator+=(const uint128_t &rhs) {
    *this = *this + rhs;
    return *this;
}
// sub
uint128_t uint128_t::operator-(const uint128_t &rhs) const {
    uint64_t lower = (*this).lower_64_ - rhs.lower_64_;
    uint64_t upper = (*this).upper_64_ - rhs.upper_64_;
    uint64_t carry = uint64_t(lower > (*this).lower_64_);
    return uint128_t(upper - carry, lower);
}
uint128_t &uint128_t::operator-=(const uint128_t &rhs) {
    *this = *this - rhs;
    return *this;
}

// logical shift left
uint128_t uint128_t::operator<<(const int bits) const {
    if (bits < 0 || bits >= 128) {
        throw std::runtime_error("Undefined Behavior");
    }
    //
    uint64_t upper;
    uint64_t lower;
    if (bits == 0) {
        return *this;
    } else if (bits >= 64) {
        upper = (*this).lower_64_ << (bits - 64);
        lower = 0;
    } else {
        upper = ((*this).upper_64_ << bits)
                | ((*this).lower_64_ >> (64 - bits));
        lower = (*this).lower_64_ << bits;
    }
    return uint128_t(upper, lower);
}
uint128_t &uint128_t::operator<<=(int bits) {
    *this = *this << bits;
    return *this;
}
// logical shift right
uint128_t uint128_t::operator>>(const int bits) const {
    if (bits < 0 || bits >= 128) {
        throw std::runtime_error("Undefined Behavior");
    }
    //
    uint64_t upper;
    uint64_t lower;
    if (bits == 0) {
        return *this;
    } else if (bits >= 64) {
        upper = 0;
        lower = (*this).upper_64_ >> (bits - 64);
    } else {
        upper = (*this).upper_64_ >> bits;
        lower = ((*this).lower_64_ >> bits)
                | ((*this).upper_64_ << (64 - bits));
    }
    return uint128_t(upper, lower);
}
uint128_t &uint128_t::operator>>=(int bits) {
    *this = *this >> bits;
    return *this;
}

// mul
uint128_t uint128_t::operator*(const uint128_t &rhs) const {
    return mul_impl(*this, rhs);
}
uint128_t &uint128_t::operator*=(const uint128_t &rhs) {
    *this = *this * rhs;
    return *this;
}
// simple distributive algorithm to calculate mul
uint128_t uint128_t::mul_impl(
        const uint128_t &lhs, const uint128_t &rhs) const {
    // Split the 128 bit to 1 64-bit for high and 2 32-bit for low
    //
    // lhs = ah[63:0]*2^64 + al1[31:0]*2^32 + al2[31:0]
    // rhs = bh[63:0]*2^64 + bl1[31:0]*2^32 + bl2[31:0]
    //
    // lhs * rhs =
    //     ah * bh * 2^128       // truncated
    //   + ah * (bl1:bl2) * 2^64 // mul_upper_a * 2^64
    //   + bh * (al1:al2) * 2^64 // mul_upper_b * 2^64
    //   + al1 * bl1 * 2^64      // mul_upper_c * 2^64
    //   + al1 * bl2 * 2^32      // add_a
    //   + al2 * bl1 * 2^32      // add_b
    //   + al2 * bl2             // lower
    //
    uint64_t l_low_63_32 = lhs.lower_64_ >> 32;
    uint64_t l_low_31_00 = lhs.lower_64_ & 0xffffffff;
    uint64_t r_low_63_32 = rhs.lower_64_ >> 32;
    uint64_t r_low_31_00 = rhs.lower_64_ & 0xffffffff;
    //
    uint64_t mul_upper_a = lhs.upper_64_ * rhs.lower_64_;
    uint64_t mul_upper_b = lhs.lower_64_ * rhs.upper_64_;
    uint64_t mul_upper_c = l_low_63_32 * r_low_63_32;
    //
    uint64_t upper = mul_upper_a + mul_upper_b + mul_upper_c;
    uint64_t lower = l_low_31_00 * r_low_31_00;
    //
    uint128_t add_a = uint128_t(l_low_63_32 * r_low_31_00) << 32;
    uint128_t add_b = uint128_t(l_low_31_00 * r_low_63_32) << 32;
    return uint128_t(upper, lower) + add_a + add_b;
}

// div
uint128_t uint128_t::operator/(const uint128_t &rhs) const {
    return div_mod_impl(*this, rhs).first;
}
uint128_t &uint128_t::operator/=(const uint128_t &rhs) {
    *this = *this / rhs;
    return *this;
}
// mod
uint128_t uint128_t::operator%(const uint128_t &rhs) const {
    return div_mod_impl(*this, rhs).second;
}
uint128_t &uint128_t::operator%=(const uint128_t &rhs) {
    *this = *this % rhs;
    return *this;
}
// simple shift-subtract algorithm to calculate div/mod
uint128_t::qr_pair uint128_t::div_mod_impl(
        const uint128_t &lhs, const uint128_t &rhs) const {
    if (rhs == zero()) { throw std::runtime_error("Divide by Zero"); }
    if (lhs == rhs) { return {one(), zero()}; }
    if (lhs < rhs) { return {zero(), lhs}; }
    // Distance between most significant bits
    // 0 <= shift < 128
    int shift = rhs.clz() - lhs.clz();
    uint128_t divisor = rhs << shift;
    uint128_t quotient = zero();
    uint128_t dividend = lhs;
    for (int i = 0; i <= shift; ++i) {
        quotient = quotient << 1;
        if (dividend >= divisor) {
            dividend = dividend - divisor;
            quotient = quotient | one();
        }
        divisor = divisor >> 1;
    }
    // {quotient, remainder}
    return {quotient, dividend};
}

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
