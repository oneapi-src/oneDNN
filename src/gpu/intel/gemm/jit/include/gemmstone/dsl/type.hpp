/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_DSL_TYPE_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_DSL_TYPE_HPP

#include <cstdint>
#include <string>

#include "gemmstone/config.hpp"
#include "internal/utils.hpp"

namespace ngen {
enum class DataType : uint8_t;
}

GEMMSTONE_NAMESPACE_START
namespace dsl {

namespace type {
enum class attr_t : uint32_t { undef = 0, ptr = 1, mut = 2, simd = 4, slm = 8 };

constexpr attr_t operator&(attr_t a, attr_t b) {
    return static_cast<attr_t>(
            static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr attr_t operator|(attr_t a, attr_t b) {
    return static_cast<attr_t>(
            static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr attr_t operator~(attr_t a) {
    return static_cast<attr_t>(~static_cast<uint32_t>(a));
}
constexpr bool any(attr_t a) {
    return a != static_cast<attr_t>(0);
}

inline attr_t &operator|=(attr_t &a, attr_t b) {
    return a = a | b;
}
inline attr_t &operator&=(attr_t &a, attr_t b) {
    return a = a & b;
}
} // namespace type

class type_t : public stringify_t<type_t> {
public:
    friend struct type_internal_accessor_t;
    using attr_t = type::attr_t;

    static type_t undef() { return type_t(kind_t::undef); }

    static type_t _bool(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::_bool, elems, attr);
    }

    static type_t u2(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::u2, elems, attr);
    }
    static type_t u4(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::u4, elems, attr);
    }
    static type_t s4(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::s4, elems, attr);
    }
    static type_t u8(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::u8, elems, attr);
    }
    static type_t s8(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::s8, elems, attr);
    }
    static type_t u16(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::u16, elems, attr);
    }
    static type_t s16(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::s16, elems, attr);
    }
    static type_t u32(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::u32, elems, attr);
    }
    static type_t s32(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::s32, elems, attr);
    }
    static type_t u64(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::u64, elems, attr);
    }
    static type_t s64(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::s64, elems, attr);
    }

    // Returns unsigned integer type.
    static type_t u(int bits, int elems = 1, attr_t attr = attr_t::undef) {
        switch (bits) {
            case 2: return u2(elems, attr);
            case 4: return u4(elems, attr);
            case 8: return u8(elems, attr);
            case 16: return u16(elems, attr);
            case 32: return u32(elems, attr);
            case 64: return u64(elems, attr);
            default: stub();
        }
        return type_t::undef();
    }

    // Returns signed integer type.
    static type_t s(int bits, int elems = 1, attr_t attr = attr_t::undef) {
        switch (bits) {
            case 4: return s4(elems, attr);
            case 8: return s8(elems, attr);
            case 16: return s16(elems, attr);
            case 32: return s32(elems, attr);
            case 64: return s64(elems, attr);
            default: stub();
        }
        return type_t::undef();
    }

    static type_t f4_e2m1(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::f4_e2m1, elems, attr);
    }
    static type_t f8_e5m2(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::f8_e5m2, elems, attr);
    }
    static type_t f8_e4m3(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::f8_e4m3, elems, attr);
    }
    static type_t bf8(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::bf8, elems, attr);
    }
    static type_t hf8(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::hf8, elems, attr);
    }
    static type_t bf16(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::bf16, elems, attr);
    }
    static type_t f16(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::f16, elems, attr);
    }
    static type_t tf32(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::tf32, elems, attr);
    }
    static type_t f32(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::f32, elems, attr);
    }
    static type_t f64(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::f64, elems, attr);
    }
    static type_t byte(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::byte, elems, attr);
    }
    static type_t byte(attr_t attr) { return type_t(kind_t::byte, 1, attr); }
    static type_t dword(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::dword, elems, attr);
    }
    static type_t qword(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::qword, elems, attr);
    }
    static type_t oword(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::oword, elems, attr);
    }
    static type_t hword(int elems = 1, attr_t attr = attr_t::undef) {
        return type_t(kind_t::hword, elems, attr);
    }

    template <typename T>
    T max() const {
        switch (kind()) {
            case kind_t::u2:
            case kind_t::u4:
            case kind_t::s4:
            case kind_t::u8:
            case kind_t::s8:
            case kind_t::u16:
            case kind_t::s16:
            case kind_t::u32:
            case kind_t::s32:
            case kind_t::u64:
            case kind_t::s64: {
                int bits = base().bitsize();
                if (is_signed()) bits--;
                T ret = T(1) << (bits - 1);
                return ret + (ret - 1);
            }
            default: stub();
        }
        return 0;
    }

    template <typename T>
    T min() const {
        switch (kind()) {
            case kind_t::u2:
            case kind_t::u4:
            case kind_t::s4:
            case kind_t::u8:
            case kind_t::s8:
            case kind_t::u16:
            case kind_t::s16:
            case kind_t::u32:
            case kind_t::s32:
            case kind_t::u64:
            case kind_t::s64: {
                if (is_unsigned()) return 0;
                return -max<T>() - 1;
            }
            default: stub();
        }
        return 0;
    }

    type_t() : type_t(type_t::undef()) {}

    type_t(ngen::DataType type, uint32_t elems = 1,
            attr_t attr = attr_t::undef);

    int elems() const { return elems_; }

    attr_t attr() const { return attr_; }

    type_t operator[](int elems) const { return with_elems(elems); }

    bool operator==(const type_t &other) const {
        return (kind() == other.kind()) && (elems() == other.elems())
                && (attr() == other.attr());
    }

    bool operator!=(const type_t &other) const { return !operator==(other); }

    size_t get_hash() const;

    bool is_ptr() const { return any(attr() & attr_t::ptr); }

    bool is_slm() const { return any(attr() & attr_t::slm); }

    bool is_undef() const { return kind() == kind_t::undef; }

    bool is_bool() const { return kind() == kind_t::_bool; }

    bool is_fp() const {
        return is_fp4() || is_fp8() || is_bf16() || is_f16() || is_tf32()
                || is_f32() || is_f64();
    }

    bool is_f4_e2m1() const { return kind() == kind_t::f4_e2m1; }
    bool is_bf8() const { return kind() == kind_t::bf8; }
    bool is_hf8() const { return kind() == kind_t::hf8; }
    bool is_bf16() const { return kind() == kind_t::bf16; }
    bool is_f16() const { return kind() == kind_t::f16; }
    bool is_tf32() const { return kind() == kind_t::tf32; }
    bool is_f32() const { return kind() == kind_t::f32; }
    bool is_f64() const { return kind() == kind_t::f64; }

    bool is_fp4() const { return is_f4_e2m1(); }
    bool is_fp8() const { return is_bf8() || is_hf8(); }

    bool is_int() const {
        return is_x2() || is_x4() || is_x8() || is_x16() || is_x32()
                || is_x64();
    }

    bool is_u2() const { return kind() == kind_t::u2; }
    bool is_x2() const { return is_u2(); }

    bool is_s4() const { return kind() == kind_t::s4; }
    bool is_u4() const { return kind() == kind_t::u4; }
    bool is_x4() const { return is_s4() || is_u4(); }

    bool is_s8() const { return kind() == kind_t::s8; }
    bool is_u8() const { return kind() == kind_t::u8; }
    bool is_x8() const { return is_s8() || is_u8(); }

    bool is_s16() const { return kind() == kind_t::s16; }
    bool is_u16() const { return kind() == kind_t::u16; }
    bool is_x16() const { return is_s16() || is_u16(); }

    bool is_s32() const { return kind() == kind_t::s32; }
    bool is_u32() const { return kind() == kind_t::u32; }
    bool is_x32() const { return is_s32() || is_u32(); }

    bool is_s64() const { return kind() == kind_t::s64; }
    bool is_u64() const { return kind() == kind_t::u64; }
    bool is_x64() const { return is_s64() || is_u64(); }

    bool is_byte() const { return kind() == kind_t::byte; }
    bool is_dword() const { return kind() == kind_t::dword; }
    bool is_qword() const { return kind() == kind_t::qword; }
    bool is_oword() const { return kind() == kind_t::oword; }
    bool is_hword() const { return kind() == kind_t::hword; }

    bool is_signed(int elems = -1) const {
        if (elems != -1 && elems_ != elems) return false;
        return is_s4() || is_s8() || is_s16() || is_s32() || is_s64();
    }

    bool is_unsigned(int elems = -1) const {
        if (elems != -1 && elems_ != elems) return false;
        return is_u2() || is_u4() || is_u8() || is_u16() || is_u32()
                || is_u64();
    }

    bool is_scalar() const { return elems() == 1; }

    bool is_mutable() const { return any(attr() & attr_t::mut); }

    bool is_simd() const { return any(attr() & attr_t::simd); }

    type_t with_elems(int new_elems) const {
        type_t copy = *this;
        copy.elems_ = new_elems;
        copy.check();
        return copy;
    }

    type_t with_ptr() const {
        type_t copy = *this;
        copy.attr_ |= attr_t::ptr;
        copy.elems_ = 1;
        copy.check();
        return copy;
    }

    type_t with_attr(attr_t attr) const {
        type_t copy = *this;
        copy.attr_ = attr;
        if (copy.is_ptr()) copy.elems_ = 1;
        copy.check();
        return copy;
    }

    type_t with_simd() const {
        type_t copy = *this;
        copy.attr_ |= attr_t::simd;
        copy.check();
        return copy;
    }

    type_t with_slm() const {
        type_t copy = *this;
        copy.attr_ |= attr_t::slm;
        copy.check();
        return copy;
    }

    type_t scalar() const { return type_t(kind()); }
    type_t base() const { return type_t(kind()); }

    // Returns size in bytes.
    int size() const;

    // Returns size in bits.
    int bitsize() const {
        if (is_ptr()) return 64;
        // 8 elements occupy the same number of bytes that a single element
        // occupies in bits.
        constexpr int bits_per_byte = 8;
        return with_elems(bits_per_byte * elems()).size();
    }

    // Returns number of elements that fit in `size()` bytes.
    // The size in bytes of `n` packed elements is
    //     `div_up(n * size(), packing())`.
    int packing() const {
        constexpr int bits_per_byte = 8;
        return bits_per_byte * size() / bitsize();
    }

    std::string str() const;
    void parse(std::istream &in);

protected:
    enum class kind_t {
        undef,
        _bool,

        // Integer types.
        u2,
        u4,
        s4,
        u8,
        s8,
        u16,
        s16,
        u32,
        s32,
        u64,
        s64,

        // Floating point types.
        f4_e2m1,
        bf8,
        f8_e5m2 = bf8,
        hf8,
        f8_e4m3 = hf8,
        bf16,
        f16,
        tf32,
        f32,
        f64,

        // Message data types.
        byte,
        dword,
        qword,
        oword,
        hword
    };

    type_t(kind_t kind, uint32_t elems = 1, attr_t attr = attr_t::undef)
        : kind_(kind), elems_(elems), attr_(attr) {
        check();
    }

    kind_t kind() const { return kind_; }

    int mantissa_bits() const;

private:
    kind_t kind_ = kind_t::undef;
    int elems_ = 0;
    attr_t attr_ = attr_t::undef;

    void check() const {
        if (is_ptr())
            gemm_assert(
                    elems_ == 1, "Pointer type must have default elems value.");
        ;
    }
};

static type_t _bool = type_t::_bool();
static type_t s8 = type_t::s8();
static type_t u8 = type_t::u8();
static type_t s16 = type_t::s16();
static type_t u16 = type_t::u16();
static type_t s32 = type_t::s32();
static type_t u32 = type_t::u32();
static type_t s64 = type_t::s64();
static type_t u64 = type_t::u64();
static type_t f32 = type_t::f32();
static type_t f16 = type_t::f16();
static type_t bf16 = type_t::bf16();

} // namespace dsl
GEMMSTONE_NAMESPACE_END
#endif
