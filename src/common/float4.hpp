/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef COMMON_FLOAT4_HPP
#define COMMON_FLOAT4_HPP

#include <cassert>
#include <cstdint>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"

namespace dnnl {
namespace impl {

struct float4_e2m1_t {
    uint8_t raw_bits_;
    float4_e2m1_t() = default;
    constexpr float4_e2m1_t(uint8_t r, bool = true) : raw_bits_(r) {}
    float4_e2m1_t(float f) { (*this) = f; }
    float4_e2m1_t(float16_t f) { (*this) = f; }
    float4_e2m1_t(bfloat16_t f) { (*this) = f; }

    float4_e2m1_t DNNL_API &operator=(float f);
    float4_e2m1_t DNNL_API &operator=(float16_t f);
    float4_e2m1_t DNNL_API &operator=(bfloat16_t f);

    DNNL_API operator float() const;
    DNNL_API operator float16_t() const;
    DNNL_API operator bfloat16_t() const;

    float4_e2m1_t &operator+=(const float a) {
        (*this) = float {*this} + a;
        return *this;
    }
};
static_assert(sizeof(float4_e2m1_t) == 1, "float4_e2m1_t must be 1 byte");

struct float4_e3m0_t {
    uint8_t raw_bits_;
    float4_e3m0_t() = default;
    constexpr float4_e3m0_t(uint8_t r, bool = true) : raw_bits_(r) {}
    float4_e3m0_t(float f) { (*this) = f; }
    float4_e3m0_t(float16_t f) { (*this) = f; }
    float4_e3m0_t(bfloat16_t f) { (*this) = f; }

    float4_e3m0_t DNNL_API &operator=(float f);
    float4_e3m0_t DNNL_API &operator=(float16_t f);
    float4_e3m0_t DNNL_API &operator=(bfloat16_t f);

    DNNL_API operator float() const;
    DNNL_API operator float16_t() const;
    DNNL_API operator bfloat16_t() const;

    float4_e3m0_t &operator+=(const float a) {
        (*this) = float {*this} + a;
        return *this;
    }
};
static_assert(sizeof(float4_e3m0_t) == 1, "float4_e3m0_t must be 1 byte");

} // namespace impl
} // namespace dnnl

#endif
