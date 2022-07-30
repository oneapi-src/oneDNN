/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef COMMON_OPDESC_HPP
#define COMMON_OPDESC_HPP

#include "c_types_map.hpp"

namespace dnnl {
namespace impl {

struct reorder_desc_t {
    primitive_kind_t primitive_kind;
    const memory_desc_t *src_md;
    const memory_desc_t *dst_md;
    engine_kind_t src_engine_kind;
    engine_kind_t dst_engine_kind;
    bool is_cross_engine;
};

struct concat_desc_t {
    primitive_kind_t primitive_kind;
    const memory_desc_t *dst_md;
    dim_t n;
    dim_t concat_dimension;
    const memory_desc_t *src_mds;
};

struct sum_desc_t {
    primitive_kind_t primitive_kind;
    const memory_desc_t *dst_md;
    dim_t n;
    const float *scales;
    const memory_desc_t *src_mds;
};

struct zero_pad_desc_t {
    primitive_kind_t primitive_kind;
};

/* C op_desc_t, which eventually are just (void*) */
using c_op_desc_t = dnnl_op_desc_t;
using const_c_op_desc_t = const_dnnl_op_desc_t;

struct op_desc_t {
    union {
        primitive_kind_t kind;
        convolution_desc_t convolution;
        deconvolution_desc_t deconvolution;
        shuffle_desc_t shuffle;
        pooling_desc_t pooling;
        prelu_desc_t prelu;
        eltwise_desc_t eltwise;
        softmax_desc_t softmax;
        lrn_desc_t lrn;
        batch_normalization_desc_t batch_normalization;
        layer_normalization_desc_t layer_normalization;
        inner_product_desc_t inner_product;
        rnn_desc_t rnn;
        gemm_desc_t gemm;
        concat_desc_t concat;
        reorder_desc_t reorder;
        sum_desc_t sum;
        binary_desc_t binary;
        matmul_desc_t matmul;
        resampling_desc_t resampling;
        zero_pad_desc_t zero_pad;
        reduction_desc_t reduction;
    };

#define DECL_CTOR_AND_CONVERTERS(c_type) \
    op_desc_t(const c_type &) = delete; \
    static op_desc_t *convert_from_c(c_type *_) { \
        return reinterpret_cast<op_desc_t *>(_); \
    } \
    static const op_desc_t *convert_from_c(const c_type *_) { \
        return reinterpret_cast<const op_desc_t *>(_); \
    }

    DECL_CTOR_AND_CONVERTERS(convolution_desc_t);
    DECL_CTOR_AND_CONVERTERS(shuffle_desc_t);
    DECL_CTOR_AND_CONVERTERS(pooling_desc_t);
    DECL_CTOR_AND_CONVERTERS(prelu_desc_t);
    DECL_CTOR_AND_CONVERTERS(eltwise_desc_t);
    DECL_CTOR_AND_CONVERTERS(softmax_desc_t);
    DECL_CTOR_AND_CONVERTERS(lrn_desc_t);
    DECL_CTOR_AND_CONVERTERS(batch_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(layer_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(inner_product_desc_t);
    DECL_CTOR_AND_CONVERTERS(rnn_desc_t);
    DECL_CTOR_AND_CONVERTERS(gemm_desc_t);
    DECL_CTOR_AND_CONVERTERS(concat_desc_t);
    DECL_CTOR_AND_CONVERTERS(reorder_desc_t);
    DECL_CTOR_AND_CONVERTERS(sum_desc_t);
    DECL_CTOR_AND_CONVERTERS(binary_desc_t);
    DECL_CTOR_AND_CONVERTERS(matmul_desc_t);
    DECL_CTOR_AND_CONVERTERS(resampling_desc_t);
    DECL_CTOR_AND_CONVERTERS(zero_pad_desc_t);
    DECL_CTOR_AND_CONVERTERS(reduction_desc_t);

    // concat_desc_t and sum_desc_t have data members which have non-trivial
    // special member functions hence the default destructor is implicitly
    // deleted by the compiler which causes a warning on Windows so we should
    // delete the destructor explicitly.
    ~op_desc_t() = delete;

#undef DECL_CTOR_AND_CONVERTERS
};

} // namespace impl
} // namespace dnnl

#endif
