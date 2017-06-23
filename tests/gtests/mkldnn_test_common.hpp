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

#ifndef MKLDNN_TEST_COMMON_HPP
#define MKLDNN_TEST_COMMON_HPP

#include <numeric>
#include <vector>
#include <cmath>
#include <stdint.h>

#include "gtest/gtest.h"

#include "mkldnn.hpp"

template <typename data_t> struct data_traits { };
template <> struct data_traits<float> {
    static const auto data_type = mkldnn::memory::data_type::f32;
};
template <> struct data_traits<int16_t> {
    static const auto data_type = mkldnn::memory::data_type::s16;
};
template <> struct data_traits<int32_t> {
    static const auto data_type = mkldnn::memory::data_type::s32;
};

template <typename T> inline void assert_eq(T a, T b);
template <> inline void assert_eq<float>(float a, float b) {
    ASSERT_FLOAT_EQ(a, b);
}

inline size_t map_index(const mkldnn::memory::desc &md, size_t index) {
    const mkldnn::c_api::mkldnn_memory_format_t fwd_weights_g =
        static_cast<mkldnn::c_api::mkldnn_memory_format_t>(
            mkldnn::memory::format::gOIhw8i16o2i);
    const mkldnn::c_api::mkldnn_memory_format_t fwd_weights =
        static_cast<mkldnn::c_api::mkldnn_memory_format_t>(
            mkldnn::memory::format::OIhw8i16o2i);
    const mkldnn::c_api::mkldnn_memory_format_t bwd_weights_g =
        static_cast<mkldnn::c_api::mkldnn_memory_format_t>(
            mkldnn::memory::format::gOIhw8o16i2o);
    const mkldnn::c_api::mkldnn_memory_format_t bwd_weights =
        static_cast<mkldnn::c_api::mkldnn_memory_format_t>(
            mkldnn::memory::format::OIhw8o16i2o);

    const bool with_groups = (md.data.format == fwd_weights_g)
                          || (md.data.format == bwd_weights_g);

    const int ndims = md.data.ndims;
    const int *dims = md.data.dims;
    const int *pdims = md.data.layout_desc.blocking.padding_dims;
    const int *optd = md.data.layout_desc.blocking.offset_padding_to_data;

    auto *strides_block = md.data.layout_desc.blocking.strides[0];
    auto *strides_within_block = md.data.layout_desc.blocking.strides[1];

    size_t ph_index = 0;
    size_t oc_16 = 0, ic_2 = 0,
        oc_2 = 0, ic_16 = 0;

    for (int rd = 0; rd < ndims; ++rd) {
        int d = ndims - rd - 1;

        EXPECT_LE(dims[d], pdims[d]);

        int cur_dim = dims[d];
        EXPECT_GT(cur_dim, 0);
        int cur_block = md.data.layout_desc.blocking.block_dims[d];

        size_t pos_d = /*static_cast<ssize_t>*/(index % cur_dim);
        EXPECT_GE(optd[d], 0);
        size_t cur_pos = optd[d] + pos_d;

        size_t cur_pos_block = cur_pos / cur_block;
        size_t cur_pos_within_block = cur_pos % cur_block;

        if (d == (with_groups + 0)) { oc_16 = pos_d % 16; oc_2 = pos_d % 2; }
        if (d == (with_groups + 1)) { ic_2 = pos_d % 2; ic_16 = pos_d % 16; }

        ph_index += cur_pos_block*strides_block[d];
        ph_index += cur_pos_within_block*strides_within_block[d];

        index /= cur_dim;
    }
    if (md.data.format == fwd_weights_g || md.data.format == fwd_weights) {
        //ph_index += -16 * ic_2 + oc_16 + ic_2;
        ph_index += oc_16 + ic_2;
        EXPECT_GE(ph_index, 16*ic_2);
        ph_index -= 16*ic_2;
    } else
        if (md.data.format == bwd_weights_g || md.data.format == bwd_weights) {
            //ph_index += -16 * oc_2 + ic_16 + oc_2;
            ph_index += ic_16 + oc_2;
            EXPECT_GE(ph_index, 16 * oc_2);
            ph_index -= 16 * oc_2;
        }
    ph_index += md.data.layout_desc.blocking.offset_padding;

    return ph_index;
}

inline mkldnn::memory::desc create_md(mkldnn::memory::dims dims,
        mkldnn::memory::data_type data_type, mkldnn::memory::format fmt) {
    using f = mkldnn::memory::format;
    size_t ndims = 0;

    switch (fmt) {
    case f::x:
        ndims = 1; break;
    case f::nc:
    case f::oi:
        ndims = 2; break;
    case f::nchw:
    case f::nhwc:
    case f::chwn:
    case f::nChw8c:
    case f::nChw16c:
    case f::oihw:
    case f::OIhw8i8o:
    case f::OIhw16i16o:
    case f::OIhw8i16o2i:
    case f::OIhw8o16i2o:
    case f::OIhw8o8i:
    case f::OIhw16o16i:
    case f::Ohwi8o:
    case f::Ohwi16o:
        ndims = 4; break;
    case f::goihw:
    case f::gOIhw8i8o:
    case f::gOIhw16i16o:
    case f::gOIhw8i16o2i:
    case f::gOIhw8o16i2o:
    case f::gOIhw8o8i:
    case f::gOIhw16o16i:
        ndims = 5; break;
    case f::format_undef:
        ndims = 0; break;
    case f::any:
        return mkldnn::memory::desc(dims, data_type, fmt);
    default: EXPECT_TRUE(false) << "test does not support format: " << int(fmt);
    }

    EXPECT_EQ(dims.size(), ndims) << "dims and format are inconsistent";

    return mkldnn::memory::desc(dims, data_type, fmt);
}

template <typename data_t>
static inline data_t set_value(size_t index, data_t mean, data_t deviation,
        double sparsity)
{
    if (data_traits<data_t>::data_type == mkldnn::memory::data_type::f32) {
        const size_t group_size = (size_t)(1. / sparsity);
        const size_t group = index / group_size;
        const size_t in_group = index % group_size;
        const bool fill = in_group == ((group % 1637) % group_size);
        return fill ? static_cast<data_t>(mean + deviation * sinf(float(index % 37)))
            : data_t{0};
    } else if (data_traits<data_t>::data_type == mkldnn::memory::data_type::s32
        || data_traits<data_t>::data_type == mkldnn::memory::data_type::s16) {
        return data_t(rand()%11);
    } else {
        return data_t(0);
    }
}

template <typename data_t>
static void fill_data(const size_t size, data_t *data, data_t mean,
        data_t deviation, double sparsity = 1.)
{
#   pragma omp parallel for schedule(static)
    for (size_t n = 0; n < size; n++) {
        data[n] = set_value<data_t>(n, mean, deviation, sparsity);
    }
}

template <typename data_t>
static void fill_data(const size_t size, data_t *data, double sparsity = 1.,
        bool init_negs = false)
{
#   pragma omp parallel for schedule(static)
    for (size_t n = 0; n < size; n++) {
        data[n] = set_value<data_t>(n, data_t(1), data_t(2e-1), sparsity);

        if (init_negs && n%4 == 0U)
            data[n] = static_cast<data_t>(-data[n]); // weird for unsigned types!
    }
}

template <typename data_t>
static void compare_data(mkldnn::memory& ref, mkldnn::memory& dst)
{
    using data_type = mkldnn::memory::data_type;

    ASSERT_TRUE(data_traits<data_t>::data_type == data_type::f32 ||
            data_traits<data_t>::data_type == data_type::s32);

    // Only true for dense format
    /* Note: size_t incompatible with MSVC++ */
    ptrdiff_t num = ref.get_primitive_desc().get_size() / sizeof(data_t);
    data_t *ref_data = (data_t *)ref.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();
#   pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < num; ++i) {
        data_t ref = ref_data[i];
        data_t got = dst_data[i];
        if (data_traits<data_t>::data_type == data_type::f32) {
            data_t diff = got - ref;
            data_t e = std::abs(ref) > 1e-4 ? diff / ref : diff;
            EXPECT_NEAR(e, 0.0, 1e-4) << "Index: " << i << " Total: " << num;
        } else if (data_traits<data_t>::data_type == data_type::s32) {
            EXPECT_EQ(ref, got) << "Index: " << i << " Total: " << num;
        }
    }
}

struct test_convolution_sizes_t {
    test_convolution_sizes_t(
        int mb,
        int ng,
        int ic, int ih, int iw,
        int oc, int oh, int ow,
        int kh, int kw,
        int padh, int padw,
        int strh, int strw,
        int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        ic(ic), ih(ih), iw(iw),
        oc(oc), oh(oh), ow(ow),
        kh(kh), kw(kw),
        padh(padh), padw(padw),
        strh(strh), strw(strw),
        dilh(dilh), dilw(dilw) {}
    int mb;
    int ng;
    int ic, ih, iw;
    int oc, oh, ow;
    int kh, kw;
    int padh, padw;
    int strh, strw;
    int dilh, dilw;
};

struct test_convolution_formats_t {
    mkldnn::memory::format src_format;
    mkldnn::memory::format weights_format;
    mkldnn::memory::format bias_format;
    mkldnn::memory::format dst_format;
};

struct test_convolution_params_t {
    const mkldnn::engine::kind engine_kind;
    mkldnn::algorithm aalgorithm;
    test_convolution_formats_t formats;
    test_convolution_sizes_t sizes;
};

#endif
