#ifndef _MKL_DNN_TEST_COMMON_HPP
#define _MKL_DNN_TEST_COMMON_HPP

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "mkl_dnn.hpp"

template <typename data_t> struct data_traits { };
template <> struct data_traits<float> {
    using precision = mkl_dnn::memory::precision;
    static const precision prec = precision::f32;
};

template <typename T> inline void assert_eq(T a, T b);
template <> inline void assert_eq<float>(float a, float b) {
    ASSERT_FLOAT_EQ(a, b);
}

inline size_t map_index(mkl_dnn::memory::desc md, size_t index) {
    const uint32_t ndims = md.data.tensor_desc.ndims_batch +
        md.data.tensor_desc.ndims_channels + md.data.tensor_desc.ndims_spatial;
    const uint32_t *dims = md.data.tensor_desc.dims;
    const uint32_t *pdims = md.data.blocking_desc.padding_dims;
    const uint32_t *optd = md.data.blocking_desc.offset_padding_to_data;

    const uint32_t *strides_block = md.data.blocking_desc.strides[0];
    const uint32_t *strides_within_block = md.data.blocking_desc.strides[1];

    size_t ph_index = 0;

    for (uint32_t rd = 0; rd < ndims; ++rd) {
        uint32_t d = ndims - rd - 1;

        EXPECT_LE(dims[d], pdims[d]);

        uint32_t cur_dim = dims[d];
        uint32_t cur_block = md.data.blocking_desc.block_dims[d];

        uint32_t cur_pos = optd[d] + (index % cur_dim);
        uint32_t cur_pos_block = cur_pos / cur_block;
        uint32_t cur_pos_within_block = cur_pos % cur_block;

        ph_index += cur_pos_block*strides_block[d];
        ph_index += cur_pos_within_block*strides_within_block[d];

        index /= cur_dim;
    }

    ph_index += md.data.blocking_desc.offset_padding;

    return ph_index;
}

inline mkl_dnn::memory::desc create_md(mkl_dnn::tensor::dims dims,
        mkl_dnn::memory::precision prec, mkl_dnn::memory::format fmt) {
    using f = mkl_dnn::memory::format;
    std::vector<uint32_t> dspec;
    switch (fmt) {
    case f::x: dspec.insert(dspec.end(), {0, 0, 1}); break;
    case f::nc: dspec.insert(dspec.end(), {1, 1, 0}); break;
    case f::oi: dspec.insert(dspec.end(), {0, 2, 0}); break;
    case f::nchw:
    case f::nhwc:
    case f::nChw8c:
                dspec.insert(dspec.end(), {1, 1, 2}); break;
    case f::oihw:
    case f::OIhw8i8o:
                dspec.insert(dspec.end(), {0, 2, 2}); break;
    case f::goihw:
    case f::gOIhw8i8o:
                dspec.insert(dspec.end(), {1, 2, 2}); break;
    default: EXPECT_TRUE(false) << "test does not support format: " << int(fmt);
    }

    const size_t ndims = std::accumulate(dspec.begin(), dspec.end(), size_t(0));
    EXPECT_EQ(dims.size(), ndims) << "dims and format are inconsistent";

    return mkl_dnn::memory::desc({dspec, dims}, prec, fmt);
}

#endif
