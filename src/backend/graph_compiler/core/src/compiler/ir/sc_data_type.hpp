/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_DATA_TYPE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_DATA_TYPE_HPP
#include <cassert>
#include <functional> // std::hash
#include <ostream>
#include <stdint.h> // uint64_t, uint32_t
#include <string>
#include <vector>
#include <util/def.hpp>
#include <util/string_utils.hpp>

namespace sc {

// The basic data types for scalars and pointers. The lower 8-bits represent
// the "base type". And the basic type is a pointer if the 9-th bit is 1.
// For example, `sc_data_etype::POINTER || sc_data_etype::F32` means the pointer
// type to float32. A sc_data_etype which equals to POINTER is a `void*` type.
enum class sc_data_etype : uint32_t {
    /// Undefined data type.
    UNDEF = 0,
    /// 16-bit/half-precision floating point.
    F16 = 1,
    /// non-standard 16-bit floating point with 7-bit mantissa.
    BF16 = 2,
    /// 16-bit unsigned integer.
    U16 = 3,
    /// 32-bit/single-precision floating point.
    F32 = 4,
    /// 32-bit signed integer.
    S32 = 5,
    /// 32-bit unsigned integer.
    U32 = 6,
    /// 8-bit signed integer.
    S8 = 7,
    /// 8-bit unsigned integer.
    U8 = 8,
    /// data type used for indexing.
    INDEX = 9,
    /// sc::generic_val type, a union type for all supported scalar types
    GENERIC = 10,
    /// boolean
    BOOLEAN = 11,
    /// void type
    VOID_T = 12,
    /// the max enum value + 1
    MAX_VALUE = 13,
    /// general pointer type, also used as a pointer bit mask
    POINTER = 0x100,
};

namespace etypes {

inline constexpr bool is_pointer(sc_data_etype type_code) {
    return static_cast<int>(type_code)
            & static_cast<int>(sc_data_etype::POINTER);
}

// given a "base type", returns the pointer type of the type. If a pointer type
// is given, will return POINTER (void*)
inline constexpr sc_data_etype get_pointerof(sc_data_etype type_code) {
    return is_pointer(type_code)
            ? sc_data_etype::POINTER
            : static_cast<sc_data_etype>(
                    static_cast<int>(sc_data_etype::POINTER)
                    | static_cast<int>(type_code));
}

// given a pointer type, returns the element type. for POINTER, returns POINTER
inline constexpr sc_data_etype get_pointer_element(sc_data_etype type_code) {
    return type_code == sc_data_etype::POINTER
            ? sc_data_etype::POINTER
            : static_cast<sc_data_etype>(0xFF & static_cast<int>(type_code));
}

} // namespace etypes
/// Data type specification. Can be scalar or vector
///
/// NOTE: The \c lanes_ field is not meaningful when \c type_code_ is
/// UNDEF or VOID_T.
struct SC_API sc_data_type_t {
    sc_data_etype type_code_;
    uint32_t lanes_;
    constexpr sc_data_type_t(sc_data_etype code, uint32_t lanes = 1)
        : type_code_(code), lanes_(lanes) {}
    constexpr sc_data_type_t() : type_code_(sc_data_etype::UNDEF), lanes_(0) {}

    sc_data_etype as_etype() const {
        assert(lanes_ == 1);
        return type_code_;
    }

    int32_t as_etype_int() const { return (int32_t)as_etype(); }

    sc_data_type_t get_pointer_element() const {
        assert(is_pointer());
        return sc_data_type_t(etypes::get_pointer_element(type_code_));
    }

    sc_data_type_t get_pointerof() const {
        return sc_data_type_t(etypes::get_pointerof(type_code_));
    }

    constexpr bool is_etype(const sc_data_etype &other) const {
        return type_code_ == other;
    }
    constexpr bool is_etype_pointer() const {
        return etypes::is_pointer(type_code_);
    }
    // returns true if the datatype is scalar pointer type
    constexpr bool is_pointer() const {
        return lanes_ == 1 && etypes::is_pointer(type_code_);
    }

    constexpr bool operator==(const sc_data_type_t &other) const {
        return type_code_ == other.type_code_ && lanes_ == other.lanes_;
    }

    constexpr bool operator!=(const sc_data_type_t &other) const {
        return !(*this == other);
    }
    // this converter enables switching on sc_data_type_t
    constexpr operator uint64_t() const {
        return (uint64_t(lanes_) << 16) | uint64_t(type_code_);
    }
    // Below are helper functions to construct a datatype with given SIMD lanes.
    // The result can be a constexpr, which can be further used as a constant,
    // for example, in the "case" label of switch-case of C++
    inline static constexpr sc_data_type_t undef(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::UNDEF, lanes);
    }
    inline static constexpr sc_data_type_t f16(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::F16, lanes);
    }
    inline static constexpr sc_data_type_t bf16(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::BF16, lanes);
    }
    inline static constexpr sc_data_type_t u16(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::U16, lanes);
    }
    inline static constexpr sc_data_type_t f32(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::F32, lanes);
    }
    inline static constexpr sc_data_type_t s32(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::S32, lanes);
    }
    inline static constexpr sc_data_type_t u32(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::U32, lanes);
    }
    inline static constexpr sc_data_type_t s8(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::S8, lanes);
    }
    inline static constexpr sc_data_type_t u8(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::U8, lanes);
    }
    inline static constexpr sc_data_type_t index(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::INDEX, lanes);
    }
    inline static constexpr sc_data_type_t boolean(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::BOOLEAN, lanes);
    }
    inline static constexpr sc_data_type_t generic(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::GENERIC, lanes);
    }
    // void* type
    inline static constexpr sc_data_type_t pointer(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::POINTER, lanes);
    }
    inline static constexpr sc_data_type_t pointerof(
            sc_data_etype etype, uint32_t lanes = 1) {
        return sc_data_type_t(etypes::get_pointerof(etype), lanes);
    }
    inline static constexpr sc_data_type_t void_t(uint32_t lanes = 1) {
        return sc_data_type_t(sc_data_etype::VOID_T, lanes);
    }
};

enum type_category { CATE_FLOAT, CATE_INT, CATE_UINT, CATE_OTHER };

extern type_category get_type_category(sc_data_type_t dtype);
extern type_category get_type_category_nothrow(sc_data_type_t dtype);
extern type_category get_etype_category_nothrow(sc_data_type_t dtype);

SC_INTERNAL_API std::ostream &operator<<(std::ostream &os, sc_data_etype etype);

SC_INTERNAL_API std::ostream &operator<<(
        std::ostream &os, sc_data_type_t dtype);

namespace utils {
template <>
std::string print_vector(const std::vector<sc::sc_data_type_t> &vec);
} // namespace utils

/**
 * Predefined scalar sc_data_type_t short-cuts which act like enums. Using them
 * have the same effect of using sc_data_type_t::XXX(1) (where XXX can be f16,
 * f32, etc.). Both members in datatypes and results of sc_data_type_t::XXX()
 * are constexpr, and can be used as normal constants. If you want to use
 * datatypes with lanes other than 1, simply use sc_data_type_t::XXX(n) instead.
 * It can still be a constexpr
 * */
namespace datatypes {
static constexpr sc_data_type_t undef = sc_data_type_t::undef();
static constexpr sc_data_type_t f16 = sc_data_type_t::f16();
static constexpr sc_data_type_t bf16 = sc_data_type_t::bf16();
static constexpr sc_data_type_t u16 = sc_data_type_t::u16();
static constexpr sc_data_type_t f32 = sc_data_type_t::f32();
static constexpr sc_data_type_t s32 = sc_data_type_t::s32();
static constexpr sc_data_type_t u32 = sc_data_type_t::u32();
static constexpr sc_data_type_t s8 = sc_data_type_t::s8();
static constexpr sc_data_type_t u8 = sc_data_type_t::u8();
static constexpr sc_data_type_t index = sc_data_type_t::index();
static constexpr sc_data_type_t boolean = sc_data_type_t::boolean();
static constexpr sc_data_type_t generic = sc_data_type_t::generic();
// void* type
static constexpr sc_data_type_t pointer = sc_data_type_t::pointer();
static constexpr sc_data_type_t void_t = sc_data_type_t::void_t();
} // namespace datatypes

template <typename>
struct sc_data_traits_t {};

template <>
struct sc_data_traits_t<float> {
    static constexpr sc_data_type_t type() { return datatypes::f32; }
};

template <>
struct sc_data_traits_t<int> {
    static constexpr sc_data_type_t type() { return datatypes::s32; }
};

template <>
struct sc_data_traits_t<uint32_t> {
    static constexpr sc_data_type_t type() { return datatypes::u32; }
};

template <>
struct sc_data_traits_t<int8_t> {
    static constexpr sc_data_type_t type() { return datatypes::s8; }
};

template <>
struct sc_data_traits_t<uint8_t> {
    static constexpr sc_data_type_t type() { return datatypes::u8; }
};

} // namespace sc

// definition of hash function of sc_data_type_t
namespace std {
template <>
struct hash<sc::sc_data_type_t> {
    std::size_t operator()(const sc::sc_data_type_t &k) const;
};
} // namespace std

#endif
