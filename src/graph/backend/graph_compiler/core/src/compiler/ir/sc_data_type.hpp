/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_DATA_TYPE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_DATA_TYPE_HPP
#include <cassert>
#include <functional> // std::hash
#include <ostream>
#include <string>
#include <vector>
#include <runtime/data_type.hpp>
#include <util/def.hpp>
#include <util/string_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
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
/// The rows_ field indicates dtype is a tile, where cols = lanes / rows
/// Currently tile dtype is used in AMX-based microkernel scenarios with limited
/// usage
///
/// NOTE: The \c lanes_ & rows_ field is not meaningful when \c type_code_ is
/// UNDEF or VOID_T.
struct SC_API sc_data_type_t {
    sc_data_etype type_code_;
    union {
        struct {
            uint16_t lanes_;
            uint16_t rows_;
        };
        uint32_t ldata_;
    };
    constexpr sc_data_type_t(sc_data_etype code, uint16_t lanes = 1)
        : type_code_(code), lanes_(lanes), rows_(0) {}
    constexpr sc_data_type_t(sc_data_etype code, uint16_t lanes, uint16_t rows)
        : type_code_(code), lanes_(lanes), rows_(rows) {}
    constexpr sc_data_type_t()
        : type_code_(sc_data_etype::UNDEF), lanes_(0), rows_(0) {}

    bool is_tile() const { return rows_ > 0; }

    sc_data_etype as_etype() const {
        assert(lanes_ == 1 && rows_ == 0);
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
        return lanes_ == 1 && rows_ == 0 && etypes::is_pointer(type_code_);
    }

    constexpr bool operator==(const sc_data_type_t &other) const {
        return type_code_ == other.type_code_ && lanes_ == other.lanes_;
    }

    constexpr bool operator!=(const sc_data_type_t &other) const {
        return !(*this == other);
    }
    // this converter enables switching on sc_data_type_t
    constexpr operator uint64_t() const {
        return (uint64_t(lanes_) << 32) | (uint64_t(rows_) << 16)
                | uint64_t(type_code_);
    }
    // Below are helper functions to construct a datatype with given SIMD lanes.
    // The result can be a constexpr, which can be further used as a constant,
    // for example, in the "case" label of switch-case of C++
    inline static constexpr sc_data_type_t undef(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::UNDEF, lanes, rows);
    }
    inline static constexpr sc_data_type_t f16(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::F16, lanes, rows);
    }
    inline static constexpr sc_data_type_t bf16(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::BF16, lanes, rows);
    }
    inline static constexpr sc_data_type_t u16(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::U16, lanes, rows);
    }
    inline static constexpr sc_data_type_t f32(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::F32, lanes, rows);
    }
    inline static constexpr sc_data_type_t s32(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::S32, lanes, rows);
    }
    inline static constexpr sc_data_type_t u32(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::U32, lanes, rows);
    }
    inline static constexpr sc_data_type_t s8(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::S8, lanes, rows);
    }
    inline static constexpr sc_data_type_t u8(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::U8, lanes, rows);
    }
    inline static constexpr sc_data_type_t index(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::INDEX, lanes, rows);
    }
    inline static constexpr sc_data_type_t boolean(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::BOOLEAN, lanes, rows);
    }
    inline static constexpr sc_data_type_t generic(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::GENERIC, lanes, rows);
    }
    // void* type
    inline static constexpr sc_data_type_t pointer(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::POINTER, lanes, rows);
    }
    inline static constexpr sc_data_type_t pointerof(
            sc_data_etype etype, uint32_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(etypes::get_pointerof(etype), lanes, rows);
    }
    inline static constexpr sc_data_type_t void_t(
            uint16_t lanes = 1, uint16_t rows = 0) {
        return sc_data_type_t(sc_data_etype::VOID_T, lanes, rows);
    }
};

enum type_category { CATE_FLOAT, CATE_INT, CATE_UINT, CATE_OTHER };

extern type_category get_type_category(sc_data_type_t dtype);
extern type_category get_type_category_nothrow(sc_data_type_t dtype);
extern type_category get_etype_category(sc_data_type_t dtype);
extern type_category get_etype_category_nothrow(sc_data_type_t dtype);

SC_INTERNAL_API std::ostream &operator<<(std::ostream &os, sc_data_etype etype);

SC_INTERNAL_API std::ostream &operator<<(
        std::ostream &os, sc_data_type_t dtype);

namespace utils {
template <>
std::string print_vector(const std::vector<sc_data_type_t> &vec);
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
// void* type. The opaque pointer type. Any pointers (including tensor/tensor
// ptr) can be auto-cast to a pointer value. But casting back is not allowed
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

struct bf16_t;
template <>
struct sc_data_traits_t<bf16_t> {
    static constexpr sc_data_type_t type() { return datatypes::bf16; }
};

struct fp16_t;
template <>
struct sc_data_traits_t<fp16_t> {
    static constexpr sc_data_type_t type() { return datatypes::f16; }
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

// definition of hash function of sc_data_type_t
namespace std {
template <>
struct hash<dnnl::impl::graph::gc::sc_data_type_t> {
    std::size_t operator()(
            const dnnl::impl::graph::gc::sc_data_type_t &k) const;
};
} // namespace std

#endif
