/*******************************************************************************
* Copyright 2016-2025 Intel Corporation
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

#ifndef COMMON_DNNL_TRAITS_HPP
#define COMMON_DNNL_TRAITS_HPP

#include "bfloat16.hpp"
#include "c_types_map.hpp"
#include "float16.hpp"
#include "float4.hpp"
#include "float8.hpp"
#include "int4.hpp"

#include <cstdint>

namespace dnnl {
namespace impl {

template <data_type_t>
struct prec_traits_t {}; /* ::type -> float */
template <typename>
struct data_traits_t {}; /* ::data_type -> f32 */
template <int>
struct typesize_traits_t {}; /* ::data_type_size -> f32 */
template <primitive_kind_t>
struct pkind_traits_t {}; /* ::desc_type, ::query_d */

template <>
struct prec_traits_t<data_type::f4_e3m0> {
    using type = float4_e3m0_t;
};
template <>
struct prec_traits_t<data_type::f4_e2m1> {
    using type = float4_e2m1_t;
};
template <>
struct prec_traits_t<data_type::e8m0> {
    using type = float8_e8m0_t;
};
template <>
struct prec_traits_t<data_type::f8_e5m2> {
    using type = float8_e5m2_t;
};
template <>
struct prec_traits_t<data_type::f8_e4m3> {
    using type = float8_e4m3_t;
};
template <>
struct prec_traits_t<data_type::f16> {
    using type = float16_t;
};
template <>
struct prec_traits_t<data_type::bf16> {
    using type = bfloat16_t;
};
template <>
struct prec_traits_t<data_type::f32> {
    using type = float;
};
template <>
struct prec_traits_t<data_type::f64> {
    using type = double;
};
template <>
struct prec_traits_t<data_type::s32> {
    using type = int32_t;
};
template <>
struct prec_traits_t<data_type::s8> {
    using type = int8_t;
};
template <>
struct prec_traits_t<data_type::u8> {
    using type = uint8_t;
};
template <>
struct prec_traits_t<data_type::s4> {
    using type = int4_t;
};
template <>
struct prec_traits_t<data_type::u4> {
    using type = uint4_t;
};
template <>
struct prec_traits_t<data_type::boolean> {
    using type = bool;
};

template <>
struct data_traits_t<float4_e3m0_t> {
    static constexpr data_type_t data_type = data_type::f4_e3m0;
};
template <>
struct data_traits_t<float4_e2m1_t> {
    static constexpr data_type_t data_type = data_type::f4_e2m1;
};
template <>
struct data_traits_t<float8_e8m0_t> {
    static constexpr data_type_t data_type = data_type::e8m0;
};
template <>
struct data_traits_t<float8_e5m2_t> {
    static constexpr data_type_t data_type = data_type::f8_e5m2;
};
template <>
struct data_traits_t<float8_e4m3_t> {
    static constexpr data_type_t data_type = data_type::f8_e4m3;
};
template <>
struct data_traits_t<float16_t> {
    static constexpr data_type_t data_type = data_type::f16;
};
template <>
struct data_traits_t<bfloat16_t> {
    static constexpr data_type_t data_type = data_type::bf16;
};
template <>
struct data_traits_t<float> {
    static constexpr data_type_t data_type = data_type::f32;
};
template <>
struct data_traits_t<int32_t> {
    static constexpr data_type_t data_type = data_type::s32;
};
template <>
struct data_traits_t<int8_t> {
    static constexpr data_type_t data_type = data_type::s8;
};
template <>
struct data_traits_t<uint8_t> {
    static constexpr data_type_t data_type = data_type::u8;
};
template <>
struct data_traits_t<int4_t> {
    static constexpr data_type_t data_type = data_type::s4;
};
template <>
struct data_traits_t<uint4_t> {
    static constexpr data_type_t data_type = data_type::u4;
};
template <>
struct data_traits_t<bool> {
    static constexpr data_type_t data_type = data_type::boolean;
};

template <>
struct typesize_traits_t<4> {
    using type = float;
};
template <>
struct typesize_traits_t<2> {
    using type = int16_t;
};
template <>
struct typesize_traits_t<1> {
    using type = uint8_t;
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
