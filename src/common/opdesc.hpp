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

// A descriptor of resampling operation.
struct resampling_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_resampling.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward_data,
    prop_kind_t prop_kind;
    // The kind of the resampling algorithm. Possible values:
    // #dnnl_resampling_nearest, #dnnl_resampling_linear.
    alg_kind_t alg_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // Resampling factor in each spatial dimension.
    float factors[DNNL_MAX_NDIMS];
};

// A descriptor of a matrix multiplication operation.
//
// 2D case:
//     dst[m, n] = src[m, k] * weights[k, n] + bias[m, n]
//
// 3D case:
//     dst[mb, m, n] = src[mb, m, k] * weights[mb, k, n] + bias[mb, m, n]
struct matmul_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_matmul.
    primitive_kind_t primitive_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Weights memory descriptor.
    memory_desc_t weights_desc;
    // Bias memory descriptor.
    memory_desc_t bias_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // The accumulator data type. Initialized automatically.
    data_type_t accum_data_type;
};

// A descriptor of a element-wise operation.
struct eltwise_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_eltwise.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // The kind of eltwise algorithm. Possible values: #dnnl_eltwise_relu,
    // #dnnl_eltwise_tanh, #dnnl_eltwise_elu, #dnnl_eltwise_square,
    // #dnnl_eltwise_abs, #dnnl_eltwise_sqrt, #dnnl_eltwise_linear,
    // #dnnl_eltwise_bounded_relu, #dnnl_eltwise_soft_relu,
    // #dnnl_eltwise_soft_relu_v2, #dnnl_eltwise_logistic, #dnnl_eltwise_exp,
    // #dnnl_eltwise_gelu_tanh, #dnnl_eltwise_swish, #dnnl_eltwise_log,
    // #dnnl_eltwise_clip, #dnnl_eltwise_clip_v2, #dnnl_eltwise_pow,
    // #dnnl_eltwise_gelu_erf, #dnnl_eltwise_round, #dnnl_eltwise_logsigmoid,
    // #dnnl_eltwise_mish, #dnnl_eltwise_hardswish, #dnnl_eltwise_hardsigmoid.
    // Possible values for passing destination memory on backward:
    // #dnnl_eltwise_relu_use_dst_for_bwd, #dnnl_eltwise_tanh_use_dst_for_bwd,
    // #dnnl_eltwise_elu_use_dst_for_bwd, #dnnl_eltwise_sqrt_use_dst_for_bwd,
    // #dnnl_eltwise_logistic_use_dst_for_bwd,
    // #dnnl_eltwise_exp_use_dst_for_bwd,
    // #dnnl_eltwise_clip_v2_use_dst_for_bwd.
    alg_kind_t alg_kind;
    // Source and destination memory descriptor.
    memory_desc_t data_desc;
    // Source and destination gradient memory descriptor.
    memory_desc_t diff_data_desc;
    // Algorithm specific parameter.
    // Accordance table:
    //  - #dnnl_eltwise_relu: @p alpha -- negative slope, @p beta ignored
    //  - #dnnl_eltwise_tanh: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_elu: @p alpha -- negative slope, @p beta ignored
    //  - #dnnl_eltwise_square: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_abs: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_sqrt: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_linear: @p alpha -- scale, @p beta -- shift
    //  - #dnnl_eltwise_bounded_relu: @p alpha -- upper bound, @p beta ignored
    //  - #dnnl_eltwise_soft_relu: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_soft_relu_v2: @p alpha -- soft_relu_v2 arg scaling, @p beta ignored
    //  - #dnnl_eltwise_logistic: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_exp: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_gelu_tanh: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_swish: @p alpha -- sigmoid arg scaling, @p beta ignored
    //  - #dnnl_eltwise_log: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_clip: @p alpha -- lower bound, @p beta -- upper bound
    //  - #dnnl_eltwise_clip_v2: @p alpha -- lower bound, @p beta -- upper bound
    //  - #dnnl_eltwise_pow: @p alpha -- scale, @p beta -- exponent
    //  - #dnnl_eltwise_gelu_erf: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_round: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_logsigmoid: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_mish: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_hardswish: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_hardsigmoid: @p alpha -- scale, @p beta -- shift
    float alpha, beta;
};

// A descriptor of a Batch Normalization operation.
struct batch_normalization_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_batch_normalization.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // Source and destination memory descriptor.
    memory_desc_t data_desc;
    // Source and destination gradient memory descriptor.
    memory_desc_t diff_data_desc;
    // Scale and shift data and gradient memory descriptors.
    //
    // Scaleshift memory descriptor uses 2D #dnnl_nc format[2,Channels]. 1-st
    // dimension contains gamma parameter, 2-nd dimension contains beta
    // parameter.
    memory_desc_t data_scaleshift_desc;
    memory_desc_t diff_data_scaleshift_desc;
    // Statistics memory descriptor.
    //
    // Statistics (mean or variance) descriptor use 1D #dnnl_x format[Channels].
    memory_desc_t stat_desc;
    // Batch normalization epsilon parameter.
    float batch_norm_epsilon;
    unsigned flags;
};

// A descriptor of a Layer Normalization operation.
struct layer_normalization_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_layer_normalization.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Scale and shift data and gradient memory descriptors.
    //
    // Scaleshift memory descriptor uses 2D #dnnl_ab
    // format[2, normalized_dim] where 1-st dimension contains gamma parameter,
    // 2-nd dimension contains beta parameter. Normalized_dim is equal to the
    // last logical dimension of the data tensor across which normalization is
    // performed.
    memory_desc_t data_scaleshift_desc;
    memory_desc_t diff_data_scaleshift_desc;
    // Mean and variance data memory descriptors.
    //
    // Statistics (mean and variance) memory descriptor is the k-dimensional tensor
    // where k is equal to data_tensor_ndims - 1 and may have any plain
    // (stride[last_dim] == 1) user-provided format.
    memory_desc_t stat_desc;
    // Layer normalization epsilon parameter.
    float layer_norm_epsilon;
    unsigned flags;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
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
