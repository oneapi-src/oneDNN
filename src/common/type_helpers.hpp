/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef TYPE_HELPERS_HPP
#define TYPE_HELPERS_HPP

#include <assert.h>

#include "mkldnn.h"
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

template <precision_t> struct prec_trait {};
template <> struct prec_trait<precision::f32> { typedef float type; };
template <> struct prec_trait<precision::u32> { typedef uint32_t type; };

template <typename T> struct data_trait {};
template <> struct data_trait<float>
{ static constexpr precision_t prec = precision::f32; };
template <> struct data_trait<uint32_t>
{ static constexpr precision_t prec = precision::u32; };

namespace types {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::precision;
using namespace mkldnn::impl::memory_format;

inline size_t precision_size(precision_t prec) {
    switch (prec) {
    case f32: return sizeof(prec_trait<f32>::type);
    case u32: return sizeof(prec_trait<u32>::type);
    case precision::undef:
    default: assert(!"unknown precision");
    }
    return 0; // not supposed to be reachable
}

inline memory_format_t format_normalize(const memory_format_t fmt) {
    if (one_of(fmt, x, nc, nchw, nhwc, nChw8c, oi, oihw, OIhw8i8o, Ohwi8o,
                goihw, gOIhw8i8o)) return blocked;
    return fmt;
}

inline bool operator==(const tensor_desc_t &lhs, const tensor_desc_t &rhs) {
    return lhs.ndims == rhs.ndims
        && mkldnn::impl::array_cmp(lhs.dims, rhs.dims, lhs.ndims);
}

inline bool operator!=(const tensor_desc_t &lhs, const tensor_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

inline status_t tensor_is_ok(const tensor_desc_t &tensor) {
    return tensor.ndims <= TENSOR_MAX_DIMS ? success : invalid_arguments;
}

inline bool blocking_desc_is_equal(const blocking_desc_t &lhs,
        const blocking_desc_t &rhs, uint32_t ndims = TENSOR_MAX_DIMS) {
    using mkldnn::impl::array_cmp;
    return lhs.offset_padding == rhs.offset_padding
        && array_cmp(lhs.block_dims, rhs.block_dims, ndims)
        && array_cmp(lhs.strides[0], rhs.strides[0], ndims)
        && array_cmp(lhs.strides[1], rhs.strides[1], ndims)
        && array_cmp(lhs.padding_dims, rhs.padding_dims, ndims)
        && array_cmp(lhs.offset_padding_to_data, rhs.offset_padding_to_data,
                ndims);
}

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    if (lhs.tensor_desc != rhs.tensor_desc || lhs.format != rhs.format)
        return false;
    if (lhs.format == blocked)
        return blocking_desc_is_equal(lhs.layout_desc.blocking,
                rhs.layout_desc.blocking, lhs.tensor_desc.ndims);
    return true;
}

inline bool operator!=(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

inline bool operator==(const memory_primitive_desc_t &lhs,
        const memory_primitive_desc_t &rhs) {
    return lhs.base.engine == rhs.base.engine // XXX: is it true?
        && lhs.memory_desc == rhs.memory_desc;
}

template <impl::precision_t prec>
inline status_t set_default_format(memory_desc_t &md, memory_format_t fmt) {
    return mkldnn_memory_desc_init(&md, &md.tensor_desc, prec, fmt);
}

template <typename T>
inline T zero() { T zero = T(); return zero; }

inline status_t convolution_desc_is_ok(
        const convolution_desc_t &convolution_desc) {
    // XXX: fill-in
    return success;
}

inline status_t pooling_desc_is_ok(
    const pooling_desc_t &pooling_desc) {
    // XXX: fill-in
    return success;
}
inline status_t relu_desc_is_ok(
        const relu_desc_t &relu_desc) {
    // XXX: fill-in
    return success;
}

inline status_t lrn_desc_is_ok(
    const lrn_desc_t &lrn_desc) {
    // XXX: fill-in
    return success;
}

inline status_t inner_product_desc_is_ok(
        const inner_product_desc_t &inner_product_desc) {
    // XXX: fill-in
    return success;
}

}
}
}

#include "memory_desc_wrapper.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
