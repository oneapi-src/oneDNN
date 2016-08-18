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

#ifndef MKLDNN_TEST_COMMON_HPP
#define MKLDNN_TEST_COMMON_HPP

#include <numeric>
#include <vector>
#include <cmath>

#include "gtest/gtest.h"

#include "mkldnn.hpp"

template <typename data_t> struct data_traits { };
template <> struct data_traits<float> {
    using precision = mkldnn::memory::precision;
    static const precision prec = precision::f32;
};

template <typename T> inline void assert_eq(T a, T b);
template <> inline void assert_eq<float>(float a, float b) {
    ASSERT_FLOAT_EQ(a, b);
}

inline size_t map_index(const mkldnn::memory::desc &md, size_t index) {
    const uint32_t ndims = md.data.tensor_desc.ndims;
    const uint32_t *dims = md.data.tensor_desc.dims;
    const uint32_t *pdims = md.data.layout_desc.blocking.padding_dims;
    const uint32_t *optd = md.data.layout_desc.blocking.offset_padding_to_data;

    const uint32_t *strides_block = md.data.layout_desc.blocking.strides[0];
    const uint32_t *strides_within_block = md.data.layout_desc.blocking.strides[1];

    size_t ph_index = 0;

    for (uint32_t rd = 0; rd < ndims; ++rd) {
        uint32_t d = ndims - rd - 1;

        EXPECT_LE(dims[d], pdims[d]);

        uint32_t cur_dim = dims[d];
        uint32_t cur_block = md.data.layout_desc.blocking.block_dims[d];

        uint32_t cur_pos = optd[d] + (index % cur_dim);
        uint32_t cur_pos_block = cur_pos / cur_block;
        uint32_t cur_pos_within_block = cur_pos % cur_block;

        ph_index += cur_pos_block*strides_block[d];
        ph_index += cur_pos_within_block*strides_within_block[d];

        index /= cur_dim;
    }

    ph_index += md.data.layout_desc.blocking.offset_padding;

    return ph_index;
}

inline mkldnn::memory::desc create_md(mkldnn::tensor::dims dims,
        mkldnn::memory::precision prec, mkldnn::memory::format fmt) {
    using f = mkldnn::memory::format;
    uint32_t ndims = 0;

    switch (fmt) {
    case f::x:
        ndims = 1; break;
    case f::nc:
    case f::oi:
        ndims = 2; break;
    case f::nchw:
    case f::nhwc:
    case f::nChw8c:
    case f::oihw:
    case f::OIhw8i8o:
    case f::Ohwi8o:
        ndims = 4; break;
    case f::goihw:
    case f::gOIhw8i8o:
        ndims = 5; break;
    case f::format_undef:
        ndims = 0; break;
    case f::any:
        return mkldnn::memory::desc({dims}, prec, fmt);
    default: EXPECT_TRUE(false) << "test does not support format: " << int(fmt);
    }

    EXPECT_EQ(dims.size(), ndims) << "dims and format are inconsistent";

    return mkldnn::memory::desc({dims}, prec, fmt);
}

template <typename data_t>
static inline data_t set_value(size_t index, double sparsity)
{
    if (data_traits<data_t>::prec == mkldnn::memory::precision::f32) {
        const size_t group_size = (size_t)(1. / sparsity);
        const size_t group = index / group_size;
        const size_t in_group = index % group_size;
        return in_group == ((group % 1637) % group_size) ?
                1. + 2e-1 * sin((data_t)(index % 37)) :
                0;
    } else {
        return (data_t)0;
    }
}

template <typename data_t>
static void fill_data(const uint32_t size, data_t *data, double sparsity = 1.)
{
#pragma omp parallel for
    for (uint32_t n = 0; n < size; n++) {
        data[n] = set_value<data_t>(n, sparsity);
    }
}

template <typename data_t>
static void compare_data(mkldnn::memory& ref, mkldnn::memory& dst)
{
uint32_t num = ref.get_primitive_desc().get_number_of_elements();
data_t *ref_data = (data_t *)ref.get_data_handle();
data_t *dst_data = (data_t *)dst.get_data_handle();
#pragma omp parallel for
    for (uint32_t i = 0; i < num; ++i) {
        float _t = (std::abs(ref_data[i]) > 1e-4) ? (dst_data[i] - ref_data[i]) / ref_data[i] : (dst_data[i] - ref_data[i]);
        EXPECT_NEAR(_t, 0.0, 1e-4);
    }
}

struct test_convolution_descr_t {
    uint32_t mb;
    uint32_t ng;
    uint32_t ic, ih, iw;
    uint32_t oc, oh, ow;
    uint32_t kh, kw;
    int32_t padh, padw;
    uint32_t strh, strw;
};

#endif
