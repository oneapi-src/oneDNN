/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GEMMSTONE_GUARD_GENERATOR_TYPE_HPP
#define GEMMSTONE_GUARD_GENERATOR_TYPE_HPP

#include "config.hpp"

#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"

#include "internal/namespace_start.hxx"

// Enum-like class for data types.
class Type {
public:
    enum _Type : uint32_t {
        invalid = 0,
        f16  = 0x01000201,
        f32  = 0x01010402,
        f64  = 0x01020803,
        u4   = 0x21120100,
        s4   = 0x21130100,
        u8   = 0x01140100,
        s8   = 0x01150100,
        u16  = 0x01160201,
        s16  = 0x01170201,
        u32  = 0x01180402,
        s32  = 0x01190402,
        u64  = 0x011A0803,
        s64  = 0x011B0803,
        f4_e2m1 = 0x410B0100,
        f4_e3m0 = 0x410A0100,
        f8_e8m0 = 0x1080100,
        bf8  = 0x010E0100,
        hf8  = 0x010F0100,
        bf16 = 0x010C0201,
        tf32 = 0x010D0402,
    };

private:
    _Type val;

public:
    constexpr Type() : Type(f32) {}
    constexpr Type(_Type val_) : val(val_) {}
    constexpr operator _Type() const { return val; }

    constexpr Type real()             const { return *this; }
    constexpr bool isComplex()        const { return false; }
    constexpr int complexComponents() const { return 1; }
    constexpr int components()        const { return 1; }
    constexpr bool isInteger()        const { return uint32_t(val) & 0x100000; }
    constexpr bool isFP()             const { return !isInteger(); }
    constexpr bool isFP4()            const { return uint32_t(val) & 0x40000000; }
    constexpr bool isInt4()           const { return uint32_t(val) & 0x20000000; }
    constexpr bool is4Bit()           const { return isInt4() || isFP4(); }
    constexpr bool isInt8()           const { return (val == Type::u8)  || (val == Type::s8);  }
    constexpr bool isInt16()          const { return (val == Type::u16) || (val == Type::s16); }
    constexpr bool isF8()             const { return (val == Type::bf8) || (val == Type::hf8  || (val == Type::f8_e8m0)); }
    constexpr bool isF4()             const { return (val == Type::f4_e2m1 || val == Type::f4_e3m0) ;}
    constexpr bool isSigned()         const { return (uint32_t(val) & 0x110000) != 0x100000; }
    constexpr int bits()              const { return is4Bit() ? 4 : (paddedSize() * 8); }
    constexpr int paddedSize()        const { return (uint32_t(val) >> 8) & 0xFF; }
    int log2Size()                    const { subByteCheck(); return uint32_t(val) & 0xFF; }
    int size()                        const { subByteCheck(); return paddedSize(); }
    constexpr int perByte()           const { return is4Bit() ? 2 : 1; }
    void subByteCheck()               const { if (is4Bit()) stub(); }

    constexpr Type arithmetic() const {
        return (val == tf32) ? Type(f32) : real();
    }
    constexpr Type asUnsigned() const {
        return static_cast<_Type>(uint32_t(val) & ~(isInteger() ? 0x10000 : 0));
    }
    constexpr Type asSigned() const {
        return static_cast<_Type>(uint32_t(val) | (isInteger() ? 0x10000 : 0));
    }
    data_type_t get_dnnl_type() const {
        switch (val) {
            case Type::f64: return data_type::f64;
            case Type::f32: return data_type::f32;
            case Type::f16: return data_type::f16;
            case Type::hf8: return data_type::f8_e4m3;
            case Type::bf8: return data_type::f8_e5m2;
            case Type::s32: return data_type::s32;
            case Type::u8: return data_type::u8;
            case Type::s8: return data_type::s8;
            case Type::u4: return data_type::u4;
            case Type::s4: return data_type::s4;
            case Type::f4_e2m1: return data_type::f4_e2m1;
            case Type::f4_e3m0: return data_type::f4_e3m0;
            default: assert(!"Unsupported type"); return data_type::undef;
        }
    }
    constexpr Type baseType() const { return *this; }

    template <typename U> constexpr friend decltype(std::declval<U>()*1) operator*(U a, Type t) {
        return t.is4Bit() ?(a + 2 * (a >= 0) - 1) / 2 : a * int(1u << t.log2Size());
    }
    template <typename U> constexpr friend decltype(std::declval<U>()*1) operator*(Type t, U a) { return a * t; }
    template <typename U>           friend U operator*=(U &a, Type t) { a = a * t; return a; }
    template <typename U> constexpr friend decltype(std::declval<U>()/1) operator/(U a, Type t) {
        return t.is4Bit() ? a * 2 : a / int(1u << t.log2Size());
    }

    // Not a valid nGEN DataType; for gemmstone internal use only
    static  constexpr  ngen::DataType ngen_f4_e2m1() { return  static_cast<ngen::DataType>(0x5A);}
    static  constexpr  ngen::DataType ngen_f4_e3m0() { return  static_cast<ngen::DataType>(0x5B);}
    static  constexpr  ngen::DataType ngen_f8_e8m0() { return  static_cast<ngen::DataType>(0x69);}


    ngen::DataType ngen() const
    {
        using DT = ngen::DataType;
        auto none = DT::invalid;
        static const DT table[32] = {DT::hf,   DT::f,    DT::df,    none,
                                     none,     none,     none,      none,
                                     ngen_f8_e8m0(),     none,     ngen_f4_e3m0(),      ngen_f4_e2m1(),
                                     DT::bf,   DT::tf32, DT::bf8,   DT::hf8,
                                     none,     none,     DT::u4,    DT::s4,
                                     DT::ub,   DT::b,    DT::uw,    DT::w,
                                     DT::ud,   DT::d,    DT::uq,    DT::q,
                                     none,     none,     none,      none};
        return table[(uint32_t(val) >> 16) & 0x1F];
    }

    bool isSubsetOf(Type T) const
    {
        if (*this == T) return true;
        return (real().bits() < T.real().bits());
    }

    Type asSignedInt() const
    {
        switch (bits()) {
            case 8:  return Type::s8;
            case 16: return Type::s16;
            case 32: return Type::s32;
            default: return Type::invalid;
        }
    }
};

static_assert((-9 * Type(Type::s4) == -5) && (9 * Type(Type::s4) == 5), "Round away from zero is required non-integer type sizes");

inline char typeToChar(Type T)
{
    switch (T.baseType()) {
        case Type::bf8:   return 'Q';
        case Type::hf8:   return 'q';
        case Type::f16:   return 'H';
        case Type::f32:   return 'S';
        case Type::f64:   return 'D';
        case Type::u4:    return 'f';
        case Type::s4:    return 'F';
        case Type::u8:    return 'o';
        case Type::s8:    return 'O';
        case Type::u16:   return 'w';
        case Type::s16:   return 'W';
        case Type::u32:   return 'i';
        case Type::s32:   return 'I';
        case Type::u64:   return 'l';
        case Type::s64:   return 'L';
        case Type::bf16:  return 'B';
        case Type::tf32:  return 'T';
        case Type::f4_e2m1:   return 'E';
        case Type::f4_e3m0:   return 'e';
        case Type::f8_e8m0:   return 'X';
        default:          return '?';
    }
}

inline Type charToType(char c)
{
    switch (c) {
        case 'Q': return Type::bf8;
        case 'q': return Type::hf8;
        case 'H': return Type::f16;
        case 'S': return Type::f32;
        case 'D': return Type::f64;
        case 'f': return Type::u4;
        case 'F': return Type::s4;
        case 'o': return Type::u8;
        case 'O': return Type::s8;
        case 'w': return Type::u16;
        case 'W': return Type::s16;
        case 'i': return Type::u32;
        case 'I': return Type::s32;
        case 'B': return Type::bf16;
        case 'T': return Type::tf32;
        case 'E': return Type::f4_e2m1;
        case 'e': return Type::f4_e3m0;
        case 'X': return Type::f8_e8m0;
        default:  return Type::invalid;
    }
}

/****************************/
/* Old names -- deprecated. */
static inline char precisionChar(Type T) { return typeToChar(T); }
static inline Type charPrecision(char c) { return charToType(c); }
/****************************/

#include "internal/namespace_end.hxx"

#endif /* header guard */
