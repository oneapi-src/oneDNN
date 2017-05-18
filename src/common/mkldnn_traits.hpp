/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef MKLDNN_TRAITS_HPP
#define MKLDNN_TRAITS_HPP

#include <assert.h>
#include <stdint.h>

#include "mkldnn.h"
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "z_magic.hpp"

namespace mkldnn {
namespace impl {

template <data_type_t> struct prec_trait {}; /* ::type -> float */
template <typename> struct data_trait {}; /* ::data_type -> f32 */
template <primitive_kind_t> struct pkind_trait {}; /* ::desc_type, ::query_d */

template <> struct prec_trait<data_type::f32> { typedef float type; };
template <> struct prec_trait<data_type::s32> { typedef int32_t type; };
template <> struct prec_trait<data_type::s16> { typedef int16_t type; };
template <> struct prec_trait<data_type::s8> { typedef int8_t type; };
template <> struct prec_trait<data_type::u8> { typedef uint8_t type; };

template <> struct data_trait<float>
{ static constexpr data_type_t data_type = data_type::f32; };
template <> struct data_trait<int>
{ static constexpr data_type_t data_type = data_type::s32; };
template <> struct data_trait<int16_t>
{ static constexpr data_type_t data_type = data_type::s16; };
template <> struct data_trait<int8_t>
{ static constexpr data_type_t data_type = data_type::s8; };
template <> struct data_trait<uint8_t>
{ static constexpr data_type_t data_type = data_type::u8; };

#define PKIND_TRAIT_INST(op) \
template <> struct pkind_trait<primitive_kind::op> { \
    typedef CONCAT2(op, _desc_t) desc_type; \
    static constexpr query_t query_d = query::CONCAT2(op, _d); \
}
PKIND_TRAIT_INST(memory);
PKIND_TRAIT_INST(convolution);
PKIND_TRAIT_INST(relu);
PKIND_TRAIT_INST(softmax);
PKIND_TRAIT_INST(pooling);
PKIND_TRAIT_INST(lrn);
PKIND_TRAIT_INST(batch_normalization);
PKIND_TRAIT_INST(inner_product);
PKIND_TRAIT_INST(convolution_relu);
#undef PKIND_TRAIT_INST

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
