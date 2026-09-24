/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef COMMON_NIBBLE_HPP
#define COMMON_NIBBLE_HPP

#include <cassert>
#include <cstdint>

namespace dnnl {
namespace impl {

namespace {
// This helper is mandatory to resolve -Wconversion hits due to a long-standing
// behavior in GCC and Clang's -Wconversion analysis engine:
// In C++, bitwise operations (like >>) on uint8_t are automatically promoted
// to int. Because the compiler sees the assignment of 32-bit int down to
// a 4-bit width variable inside a constructor initializer list, it flags it as
// a narrowing conversion, completely ignoring `& 0xf` mask.
constexpr uint8_t shift_bits(uint8_t v, uint8_t shift) {
    return v >> shift;
}
} // namespace

// An abstraction to manipulate with bits as bytes. `2` means there are two
// elements in it.
struct nibble2_t {
    // constructs a nibble pair from a pair of uint8_t values
    nibble2_t(uint8_t low_, uint8_t high_)
        : low(low_ & 0xf), high(high_ & 0xf) {}

    // constructs a nibble pairs from an uin8_t, taking its low and high part
    nibble2_t(uint8_t pack_)
        : low(shift_bits(pack_, 0) & 0xf), high(shift_bits(pack_, 4) & 0xf) {}

    // sets low (idx=0) or high (idx=1)  nibble.
    inline void set(uint8_t val, int idx) {
        switch (idx) {
            case 0: low = val & 0xf; return;
            case 1: high = val & 0xf; return;
            default: assert(!"Out of range index"); return;
        }
    }

    // returns low (idx = 0) or high (idx = 1) nibble in a uint8_t
    inline uint8_t get(int idx) const {
        switch (idx) {
            case 0: return low;
            case 1: return high;
            default: assert(!"out of range index"); return 0;
        }
    }

    // returns pair of nibbles as uint8_t
    inline uint8_t get() const { return static_cast<uint8_t>(high << 4 | low); }

    // Returns a size of a nibble object in bytes.
    static constexpr size_t size() { return 1; }

    // Returns the number of elements in this type of nibble.
    static constexpr int nelems() { return 2; }

private:
    uint8_t low : 4;
    uint8_t high : 4;
};
static_assert(sizeof(nibble2_t) == 1, "nibble2_t must be 1 byte");
static_assert(nibble2_t::size() == 1, "nibble2_t must be 1 byte");

// An abstraction to manipulate with bits as bytes. `4` means there are four
// elements in it.
struct nibble4_t {
    // constructs a nibble quartet from a quartet of uint8_t values
    nibble4_t(uint8_t e0, uint8_t e1, uint8_t e2, uint8_t e3)
        : e0_(e0 & 0x3), e1_(e1 & 0x3), e2_(e2 & 0x3), e3_(e3 & 0x3) {}

    // constructs a nibble quartet from an uin8_t
    nibble4_t(uint8_t pack)
        : e0_(shift_bits(pack, 0) & 0x3)
        , e1_(shift_bits(pack, 2) & 0x3)
        , e2_(shift_bits(pack, 4) & 0x3)
        , e3_(shift_bits(pack, 6) & 0x3) {}

    // sets an element @val in the nibble according to the @idx.
    inline void set(uint8_t val, int idx) {
        switch (idx) {
            case 0: e0_ = val & 0x3; return;
            case 1: e1_ = val & 0x3; return;
            case 2: e2_ = val & 0x3; return;
            case 3: e3_ = val & 0x3; return;
            default: assert(!"Out of range index"); return;
        }
    }

    // returns an element from the nibble according to the @idx.
    inline uint8_t get(int idx) const {
        switch (idx) {
            case 0: return e0_;
            case 1: return e1_;
            case 2: return e2_;
            case 3: return e3_;
            default: assert(!"out of range index"); return 0;
        }
    }

    // returns a quartet of nibbles as uint8_t
    inline uint8_t get() const {
        return static_cast<uint8_t>(e3_ << 6 | e2_ << 4 | e1_ << 2 | e0_);
    }

    // Returns a size of a nibble object in bytes.
    static constexpr size_t size() { return 1; }

    // Returns the number of elements in this type of nibble.
    static constexpr int nelems() { return 4; }

private:
    uint8_t e0_ : 2;
    uint8_t e1_ : 2;
    uint8_t e2_ : 2;
    uint8_t e3_ : 2;
};
static_assert(sizeof(nibble4_t) == 1, "nibble2_t must be 1 byte");
static_assert(nibble4_t::size() == 1, "nibble2_t must be 1 byte");

} // namespace impl
} // namespace dnnl

#endif
