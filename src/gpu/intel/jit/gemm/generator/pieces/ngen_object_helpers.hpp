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


#ifndef GEMMSTONE_GUARD_NGEN_OBJECT_HELPERS_HPP
#define GEMMSTONE_GUARD_NGEN_OBJECT_HELPERS_HPP

#include "internal/ngen_includes.hpp"
#include "type.hpp"
#include "problem.hpp"

#include "internal/namespace_start.hxx"

// Move subregister to another pipe.
void movePipes(ngen::Subregister &s, bool sizeCanChange = true);

// Move subregister to integer pipe.
void moveToIntPipe(ngen::Subregister &s);

// Move register region to integer pipe.
void moveToIntPipe(int esize, ngen::RegData &s);

// Measure distance in elements between two subregisters.
int elementDiff(ngen::HW hw, const ngen::RegData &r1, const ngen::RegData &r2);

// Cast a value to an nGEN immediate of the given type.
template <typename U>
ngen::Immediate cast(Type T, U val)
{
    using ngen::half;
    switch (T) {
        case Type::f16:  return half(val);
        case Type::f32:  return float(val);
        case Type::f64:  return double(val);
        case Type::u8:   return  uint8_t(val);
        case Type::s8:   return   int8_t(val);
        case Type::u16:  return uint16_t(val);
        case Type::s16:  return  int16_t(val);
        case Type::u32:  return uint32_t(val);
        case Type::s32:  return  int32_t(val);
        case Type::u64:  return uint64_t(val);
        case Type::s64:  return  int64_t(val);
        case Type::bf16:
        case Type::tf32:
        default:         stub();
    }
}

static inline ngen::Immediate cast(Type T, Scalar val) {
    return cast(T, double(val));
}

// Comparison operators, for use with generic code.
/* check if these are still used */
constexpr bool operator==(const ngen::RegData &rd, int i)                    { return false; }
constexpr bool operator==(const ngen::RegData &rd, const ngen::Immediate &i) { return false; }
constexpr bool operator!=(const ngen::RegData &rd, int i)                    { return true; }
constexpr bool operator!=(const ngen::RegData &rd, const ngen::Immediate &i) { return true; }

/* Modify a cache policy to be L1-uncacheable */
ngen::CacheSettingsLSC makeL1Uncacheable(ngen::CacheSettingsLSC c);

#include "internal/namespace_end.hxx"

#endif /* header guard */
