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

struct inner_product_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_inner_product.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: forward_training,
    // forward_inference, backward_data,
    // backward_weights, and backward_bias.
    prop_kind_t prop_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Weights memory descriptor.
    memory_desc_t weights_desc;
    // Weights gradient memory descriptor.
    memory_desc_t diff_weights_desc;
    // Bias memory descriptor.
    memory_desc_t bias_desc;
    // Bias gradient memory descriptor.
    memory_desc_t diff_bias_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // The accumulator data type.
    data_type_t accum_data_type;
};

struct convolution_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_convolution.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward_data,
    // #dnnl_backward_weights, and #dnnl_backward_bias.
    prop_kind_t prop_kind;
    // The kind of the convolution algorithm. Possible values:
    // #dnnl_convolution_direct.
    alg_kind_t alg_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Weights memory descriptor.
    memory_desc_t weights_desc;
    // Weights gradient memory descriptor.
    memory_desc_t diff_weights_desc;
    // Bias memory descriptor.
    memory_desc_t bias_desc;
    // Bias gradient memory descriptor.
    memory_desc_t diff_bias_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // Convolution strides in each spatial dimension.
    dims_t strides;
    // Convolution dilates in each spatial dimension.
    dims_t dilates;
    // Padding in each spatial dimension. padding[0] is a padding in the
    // beginning (@p padding_l), padding[1] is a padding in the end (@p
    // padding_r).
    dims_t padding[2];
    // The accumulator data type. Initialized automatically.
    data_type_t accum_data_type;
};

// A descriptor of a deconvolution operation.
using deconvolution_desc_t = convolution_desc_t;

// A descriptor of a shuffle operation.
struct shuffle_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_shuffle.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // Source and destination memory descriptor,
    // and source and destination gradient memory descriptor.
    memory_desc_t data_desc;
    // Axis for shuffling.
    int axis;
    // Number of groups.
    dim_t group_size;
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
