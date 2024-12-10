/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"
#include "common/utils.hpp"

#include <vector>
#include <type_traits>

namespace dnnl {
namespace impl {

#define DECLARE_COMMON_OP_DESC_CLONE(op_desc_kind_t) \
    std::unique_ptr<op_desc_t> clone() const override { \
        return utils::make_unique<op_desc_kind_t>(*this); \
    }

// A base class for all descriptors that allows to dispatch between them through
// a dedicated `kind` field.
struct op_desc_t {
    virtual ~op_desc_t() = default;

    virtual std::unique_ptr<op_desc_t> clone() const = 0;

    // Converters to a inherited type.
    template <typename T>
    static const T *to_desc(const op_desc_t *op_desc) {
        static_assert(!std::is_pointer<T>::value,
                "T is not expected to be a pointer type.");
        return utils::downcast<const T *>(op_desc);
    }
    template <typename T>
    static T *to_desc(op_desc_t *op_desc) {
        static_assert(!std::is_pointer<T>::value,
                "T is not expected to be a pointer type.");
        return utils::downcast<T *>(op_desc);
    }

    // The kind of primitive. Used for self-identifying the primitive desc.
    primitive_kind_t primitive_kind;

protected:
    op_desc_t() : primitive_kind(primitive_kind::undefined) {}
    op_desc_t(primitive_kind_t pk) : primitive_kind(pk) {}
    op_desc_t(const op_desc_t &) = default;
    op_desc_t &operator=(const op_desc_t &) = default;
    op_desc_t(op_desc_t &&) = default;
    op_desc_t &operator=(op_desc_t &&) = default;
};

// A descriptor of a reorder operation.
struct reorder_desc_t : public op_desc_t {
    reorder_desc_t() = default;
    reorder_desc_t(primitive_kind_t primitive_kind, const memory_desc_t *src_md,
            const memory_desc_t *dst_md, engine_kind_t src_engine_kind,
            engine_kind_t dst_engine_kind, bool is_cross_engine)
        : op_desc_t(primitive_kind)
        , src_md(src_md)
        , dst_md(dst_md)
        , src_engine_kind(src_engine_kind)
        , dst_engine_kind(dst_engine_kind)
        , is_cross_engine(is_cross_engine) {}

    DECLARE_COMMON_OP_DESC_CLONE(reorder_desc_t);

    const memory_desc_t *src_md {};
    const memory_desc_t *dst_md {};
    engine_kind_t src_engine_kind {};
    engine_kind_t dst_engine_kind {};
    bool is_cross_engine {};
};

// A descriptor of a concat operation.
struct concat_desc_t : public op_desc_t {
    concat_desc_t() = default;
    concat_desc_t(primitive_kind_t primitive_kind, const memory_desc_t *dst_md,
            dim_t n, dim_t concat_dimension,
            const memory_desc_t *const *src_mds)
        : op_desc_t(primitive_kind)
        , dst_md(dst_md)
        , n(n)
        , concat_dimension(concat_dimension) {
        for (dim_t i = 0; i < n; i++)
            this->src_mds.push_back(src_mds[i]);
    }

    DECLARE_COMMON_OP_DESC_CLONE(concat_desc_t);

    const memory_desc_t *dst_md {};
    dim_t n {};
    dim_t concat_dimension {};
    std::vector<const memory_desc_t *> src_mds {};
};

// A descriptor of a sum operation.
struct sum_desc_t : public op_desc_t {
    sum_desc_t() = default;
    sum_desc_t(primitive_kind_t primitive_kind, const memory_desc_t *dst_md,
            dim_t n, const float *scales, const memory_desc_t *const *src_mds)
        : op_desc_t(primitive_kind), dst_md(dst_md), n(n), scales(scales) {
        for (dim_t i = 0; i < n; i++)
            this->src_mds.push_back(src_mds[i]);
    }

    DECLARE_COMMON_OP_DESC_CLONE(sum_desc_t);

    const memory_desc_t *dst_md {};
    dim_t n {};
    const float *scales {};
    std::vector<const memory_desc_t *> src_mds {};
};

// A descriptor of a zero padding operation.
struct zero_pad_desc_t : public op_desc_t {
    zero_pad_desc_t() : op_desc_t(primitive_kind::zero_pad) {}

    DECLARE_COMMON_OP_DESC_CLONE(zero_pad_desc_t);
};

// A descriptor of a inner product operation.
struct inner_product_desc_t : public op_desc_t {
    inner_product_desc_t() : op_desc_t(primitive_kind::inner_product) {}

    DECLARE_COMMON_OP_DESC_CLONE(inner_product_desc_t);

    // The kind of propagation. Possible values: forward_training,
    // forward_inference, backward_data,
    // backward_weights, and backward_bias.
    prop_kind_t prop_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Weights memory descriptor.
    memory_desc_t weights_desc {};
    // Weights gradient memory descriptor.
    memory_desc_t diff_weights_desc {};
    // Bias memory descriptor.
    memory_desc_t bias_desc {};
    // Bias gradient memory descriptor.
    memory_desc_t diff_bias_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
    // The accumulator data type.
    data_type_t accum_data_type {};
};

// A descriptor of a convolution operation.
struct convolution_desc_t : public op_desc_t {
    convolution_desc_t() : op_desc_t(primitive_kind::convolution) {}

    DECLARE_COMMON_OP_DESC_CLONE(convolution_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward_data,
    // #dnnl_backward_weights, and #dnnl_backward_bias.
    prop_kind_t prop_kind {};
    // The kind of the convolution algorithm. Possible values:
    // #dnnl_convolution_direct.
    alg_kind_t alg_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Weights memory descriptor.
    memory_desc_t weights_desc {};
    // Weights gradient memory descriptor.
    memory_desc_t diff_weights_desc {};
    // Bias memory descriptor.
    memory_desc_t bias_desc {};
    // Bias gradient memory descriptor.
    memory_desc_t diff_bias_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
    // Convolution strides in each spatial dimension.
    dims_t strides {};
    // Convolution dilates in each spatial dimension.
    dims_t dilates {};
    // Padding in each spatial dimension. padding[0] is a padding in the
    // beginning (@p padding_l), padding[1] is a padding in the end (@p
    // padding_r).
    dims_t padding[2] {};
    // The accumulator data type. Initialized automatically.
    data_type_t accum_data_type {};
    // For internal use only. To mark conv is used for deconv.
    bool use_inversion {};
};

// A descriptor of a deconvolution operation.
using deconvolution_desc_t = convolution_desc_t;

// A descriptor of a shuffle operation.
struct shuffle_desc_t : public op_desc_t {
    shuffle_desc_t() : op_desc_t(primitive_kind::shuffle) {}

    DECLARE_COMMON_OP_DESC_CLONE(shuffle_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, and #dnnl_backward_data.
    prop_kind_t prop_kind {};
    // Source or source gradient memory descriptor.
    memory_desc_t src_desc {};
    // Destination or destination gradient memory descriptor.
    memory_desc_t dst_desc {};
    // Axis for shuffling.
    int axis {};
    // Number of groups.
    dim_t group_size {};
};

// A descriptor of resampling operation.
struct resampling_desc_t : public op_desc_t {
    resampling_desc_t() : op_desc_t(primitive_kind::resampling) {}

    DECLARE_COMMON_OP_DESC_CLONE(resampling_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward_data,
    prop_kind_t prop_kind {};
    // The kind of the resampling algorithm. Possible values:
    // #dnnl_resampling_nearest, #dnnl_resampling_linear.
    alg_kind_t alg_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
    // Resampling factor in each spatial dimension.
    float factors[DNNL_MAX_NDIMS] {};
};

// A descriptor of a matrix multiplication operation.
//
// 2D case:
//     dst[m, n] = src[m, k] * weights[k, n] + bias[m, n]
//
// 3D case:
//     dst[mb, m, n] = src[mb, m, k] * weights[mb, k, n] + bias[mb, m, n]
struct matmul_desc_t : public op_desc_t {
    matmul_desc_t() : op_desc_t(primitive_kind::matmul) {}

    DECLARE_COMMON_OP_DESC_CLONE(matmul_desc_t);

    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Weights memory descriptor.
    memory_desc_t weights_desc {};
    // Bias memory descriptor.
    memory_desc_t bias_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // The accumulator data type. Initialized automatically.
    data_type_t accum_data_type {};
};

// A descriptor of a element-wise operation.
struct eltwise_desc_t : public op_desc_t {
    eltwise_desc_t() : op_desc_t(primitive_kind::eltwise) {}

    DECLARE_COMMON_OP_DESC_CLONE(eltwise_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind {};
    // The kind of eltwise algorithm. Possible values: #dnnl_eltwise_relu,
    // #dnnl_eltwise_tanh, #dnnl_eltwise_elu, #dnnl_eltwise_square,
    // #dnnl_eltwise_abs, #dnnl_eltwise_sqrt, #dnnl_eltwise_linear,
    // #dnnl_eltwise_soft_relu, #dnnl_eltwise_logistic, #dnnl_eltwise_exp,
    // #dnnl_eltwise_gelu_tanh, #dnnl_eltwise_swish, #dnnl_eltwise_log,
    // #dnnl_eltwise_clip, #dnnl_eltwise_clip_v2, #dnnl_eltwise_pow,
    // #dnnl_eltwise_gelu_erf, #dnnl_eltwise_round,
    // #dnnl_eltwise_mish, #dnnl_eltwise_hardswish, #dnnl_eltwise_hardsigmoid.
    // Possible values for passing destination memory on backward:
    // #dnnl_eltwise_relu_use_dst_for_bwd, #dnnl_eltwise_tanh_use_dst_for_bwd,
    // #dnnl_eltwise_elu_use_dst_for_bwd, #dnnl_eltwise_sqrt_use_dst_for_bwd,
    // #dnnl_eltwise_logistic_use_dst_for_bwd,
    // #dnnl_eltwise_exp_use_dst_for_bwd,
    // #dnnl_eltwise_clip_v2_use_dst_for_bwd.
    alg_kind_t alg_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
    // Algorithm specific parameter.
    // Accordance table:
    //  - #dnnl_eltwise_relu: @p alpha -- negative slope, @p beta ignored
    //  - #dnnl_eltwise_tanh: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_elu: @p alpha -- negative slope, @p beta ignored
    //  - #dnnl_eltwise_square: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_abs: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_sqrt: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_linear: @p alpha -- scale, @p beta -- shift
    //  - #dnnl_eltwise_soft_relu: @p alpha -- soft_relu arg scaling, @p beta ignored
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
    //  - #dnnl_eltwise_mish: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_hardswish: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_hardsigmoid: @p alpha -- scale, @p beta -- shift
    float alpha {};
    float beta {};
};

// A descriptor of a Batch Normalization operation.
struct batch_normalization_desc_t : public op_desc_t {
    batch_normalization_desc_t()
        : op_desc_t(primitive_kind::batch_normalization) {}

    DECLARE_COMMON_OP_DESC_CLONE(batch_normalization_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
    // Scale and/or shift data and gradient memory descriptor.
    // Scaleshift memory descriptor uses 1D #dnnl_x format[Channels].
    memory_desc_t scaleshift_desc {};
    memory_desc_t diff_scaleshift_desc {};
    // Statistics memory descriptor.
    //
    // Statistics (mean or variance) descriptor use 1D #dnnl_x format[Channels].
    memory_desc_t stat_desc {};
    // Batch normalization epsilon parameter.
    float batch_norm_epsilon {};
    unsigned flags {};
};

// A descriptor of a Group Normalization operation.
struct group_normalization_desc_t : public op_desc_t {
    group_normalization_desc_t()
        : op_desc_t(primitive_kind::group_normalization) {}

    DECLARE_COMMON_OP_DESC_CLONE(group_normalization_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Scale and/or shift data and gradient memory descriptor.
    // Scaleshift memory descriptor uses 1D #dnnl_x format[Channels].
    memory_desc_t scaleshift_desc {};
    memory_desc_t diff_scaleshift_desc {};
    // Mean and variance data memory descriptors.
    // Statistics (mean and variance) memory descriptor uses 2D #dnnl_ab
    // format[Batch, groups].
    memory_desc_t stat_desc {};
    // Group normalization groups parameter.
    dim_t groups {};
    // Group normalization epsilon parameter.
    float group_norm_epsilon {};
    unsigned flags {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
};

// A descriptor of a Layer Normalization operation.
struct layer_normalization_desc_t : public op_desc_t {
    layer_normalization_desc_t()
        : op_desc_t(primitive_kind::layer_normalization) {}

    DECLARE_COMMON_OP_DESC_CLONE(layer_normalization_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Scale and shift data and gradient memory descriptors.
    // Scaleshift memory descriptor uses 1D #dnnl_x format[normalized_dim].
    // Normalized_dim is equal to the last logical dimension of the source
    // tensor across which normalization is performed.
    memory_desc_t data_scaleshift_desc {};
    memory_desc_t diff_data_scaleshift_desc {};
    // Mean and variance data memory descriptors.
    //
    // Statistics (mean and variance) memory descriptor is the k-dimensional tensor
    // where k is equal to data_tensor_ndims - 1 and may have any plain
    // (stride[last_dim] == 1) user-provided format.
    memory_desc_t stat_desc {};
    // Layer normalization epsilon parameter.
    float layer_norm_epsilon {};
    unsigned flags {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
};

// A descriptor of a Local Response Normalization (LRN) operation.
struct lrn_desc_t : public op_desc_t {
    lrn_desc_t() : op_desc_t(primitive_kind::lrn) {}

    DECLARE_COMMON_OP_DESC_CLONE(lrn_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind {};
    // LRN algorithm. Possible values: #dnnl_lrn_within_channel and
    // #dnnl_lrn_across_channels.
    alg_kind_t alg_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
    // The number of channels to sum over (for cross-channel LRN) or the side
    // length of the square region to sum over (for within-channel LRN).
    dim_t local_size {};
    // LRN alpha parameter.
    float lrn_alpha {};
    // LRN beta parameter.
    float lrn_beta {};
    // LRN k parameter.
    float lrn_k {};
};

// A descriptor of reduction operation.
struct reduction_desc_t : public op_desc_t {
    reduction_desc_t() : op_desc_t(primitive_kind::reduction) {}

    DECLARE_COMMON_OP_DESC_CLONE(reduction_desc_t);

    // The kind of reduction algorithm. Possible values:
    // #dnnl_reduction_max, #dnnl_reduction_min, #dnnl_reduction_sum,
    // #dnnl_reduction_mul, #dnnl_reduction_mean, #dnnl_reduction_norm_lp_max,
    // #dnnl_reduction_norm_lp_sum, #dnnl_reduction_norm_lp_power_p_max,
    // #dnnl_reduction_norm_lp_power_p_sum.
    alg_kind_t alg_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Algorithm specific parameters.
    // Accordance table:
    // #dnnl_reduction_max: @p p and @p eps are ignored
    // #dnnl_reduction_min: @p p and @p eps are ignored
    // #dnnl_reduction_norm_lp_max: @p p -- power, @p eps -- epsilon
    // #dnnl_reduction_norm_lp_sum: @p p -- power, @p eps -- epsilon
    // #dnnl_reduction_norm_lp_power_p_max: @p p -- power, @p eps -- epsilon
    // #dnnl_reduction_norm_lp_power_p_sum: @p p -- power, @p eps -- epsilon
    // #dnnl_reduction_sum: @p p and @p eps are ignored
    // #dnnl_reduction_mul: @p p and @p eps are ignored
    // #dnnl_reduction_mean: @p p and @p eps are ignored
    float p {};
    float eps {};
};

/// A descriptor of a Softmax operation.
struct softmax_desc_t : public op_desc_t {
    softmax_desc_t() : op_desc_t(primitive_kind::softmax) {}

    DECLARE_COMMON_OP_DESC_CLONE(softmax_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, and #dnnl_backward_data.
    prop_kind_t prop_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // The axis along which to perform the softmax.
    int softmax_axis {};
    // Softmax algorithm. Possible values: #dnnl_softmax_accurate and
    // #dnnl_softmax_log.
    alg_kind_t alg_kind {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
};

// A descriptor of a binary operation.
struct binary_desc_t : public op_desc_t {
    binary_desc_t() : op_desc_t(primitive_kind::binary) {}

    DECLARE_COMMON_OP_DESC_CLONE(binary_desc_t);

    // The kind of the binary algorithm. Possible values:
    // #dnnl_binary_add, #dnnl_binary_mul, #dnnl_binary_max, #dnnl_binary_min,
    // #dnnl_binary_div, #dnnl_binary_sub, #dnnl_binary_ge, #dnnl_binary_gt,
    // #dnnl_binary_le, #dnnl_binary_lt, #dnnl_binary_eq, #dnnl_binary_ne,
    // and #dnnl_binary_select
    alg_kind_t alg_kind {};
    // Source memory descriptors.
    memory_desc_t src_desc[3] {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
};

/// A descriptor of a PReLU operation.
struct prelu_desc_t : public op_desc_t {
    prelu_desc_t() : op_desc_t(primitive_kind::prelu) {}

    DECLARE_COMMON_OP_DESC_CLONE(prelu_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward
    prop_kind_t prop_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Learnable parameter alpha memory descriptor.
    // Alpha describes negative slope.
    memory_desc_t weights_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Learnable parameter alpha gradient memory descriptor.
    memory_desc_t diff_weights_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
};

// A descriptor of a pooling operation.
struct pooling_desc_t : public op_desc_t {
    pooling_desc_t() : op_desc_t(primitive_kind::pooling) {}

    DECLARE_COMMON_OP_DESC_CLONE(pooling_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind {};
    // The kind of pooling algorithm.
    // Possible values: #dnnl_pooling_max,
    // #dnnl_pooling_avg_include_padding, and
    // #dnnl_pooling_avg_exclude_padding.
    alg_kind_t alg_kind {};
    // Source memory descriptor.
    memory_desc_t src_desc {};
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc {};
    // Destination memory descriptor.
    memory_desc_t dst_desc {};
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc {};
    // Pooling kernel strides for spatial dimensions.
    dims_t strides {};
    // Pooling kernel spatial dimensions.
    dims_t kernel {};
    // Padding in each spatial dimension. padding[0] is a padding in the
    // beginning (@p padding_l), padding[1] is a padding in the end (@p
    // padding_r).
    dims_t padding[2] {};
    // The accumulator data type. Initialized automatically.
    data_type_t accum_data_type {};
    // Pooling dilations for spatial dimensions.
    dims_t dilation {};
};

// A descriptor for an RNN operation.
struct rnn_desc_t : public op_desc_t {
    rnn_desc_t() : op_desc_t(primitive_kind::rnn) {}

    DECLARE_COMMON_OP_DESC_CLONE(rnn_desc_t);

    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, and #dnnl_backward.
    prop_kind_t prop_kind {};
    // RNN cell kind. Must be one of #dnnl_vanilla_rnn,
    // #dnnl_vanilla_lstm, #dnnl_vanilla_gru, or #dnnl_lbr_gru.
    alg_kind_t cell_kind {};
    // The direction of RNN primitive execution.
    rnn_direction_t direction {};
    // Source layer memory descriptor.
    memory_desc_t src_layer_desc {};
    // Source iteration memory descriptor for hidden state.
    memory_desc_t src_iter_desc {};
    // Source iteration memory descriptor for cell state.
    memory_desc_t src_iter_c_desc {};
    // Weights layer memory descriptor.
    memory_desc_t weights_layer_desc {};
    // Weights iteration memory descriptor.
    memory_desc_t weights_iter_desc {};
    // Bias memory descriptor.
    memory_desc_t bias_desc {};
    // Destination layer memory descriptor.
    memory_desc_t dst_layer_desc {};
    // Destination iter memory descriptor for hidden state.
    memory_desc_t dst_iter_desc {};
    // Destination iter memory descriptor for cell state.
    memory_desc_t dst_iter_c_desc {};
    // Weights peephole memory descriptor.
    // This memory descriptor is equal to zero memory descriptor in case of
    // non-peephole LSTMs and other non-LSTM RNNs.
    memory_desc_t weights_peephole_desc {};
    // Weights projection memory descriptor.
    // This memory descriptor is equal to zero memory descriptor in case of
    // non-projection LSTMs and other non-LSTM RNNs.
    memory_desc_t weights_projection_desc {};

    // Source gradient layer memory descriptor.
    memory_desc_t diff_src_layer_desc {};
    // Source gradient iter memory descriptor for hidden state.
    memory_desc_t diff_src_iter_desc {};
    // Source gradient iter memory descriptor for cell state.
    memory_desc_t diff_src_iter_c_desc {};
    // Weights gradient layer memory descriptor.
    memory_desc_t diff_weights_layer_desc {};
    // Weights gradient iter memory descriptor.
    memory_desc_t diff_weights_iter_desc {};
    // Bias gradient memory descriptor.
    memory_desc_t diff_bias_desc {};
    // Destination gradient layer memory descriptor.
    memory_desc_t diff_dst_layer_desc {};
    // Destination gradient iteration memory descriptor for hidden state.
    memory_desc_t diff_dst_iter_desc {};
    // Destination gradient iteration memory descriptor for cell state.
    memory_desc_t diff_dst_iter_c_desc {};
    // Weights gradient peephole memory descriptor.
    // This memory descriptor is equal to zero memory descriptor in case of
    // non-peephole LSTMs and other non-LSTM RNNs.
    memory_desc_t diff_weights_peephole_desc {};
    // Weights gradient projection memory descriptor.
    // This memory descriptor is equal to zero memory descriptor in case of
    // non-projection LSTMs and other non-LSTM RNNs.
    memory_desc_t diff_weights_projection_desc {};

    // RNN cell flags
    unsigned flags {};
    // Activation function used for vanilla_rnn cell kind.
    // Must be either #dnnl_eltwise_relu or #dnnl_eltwise_tanh.
    alg_kind_t activation_kind {};
    float alpha {};
    float beta {};
};

#undef DECLARE_COMMON_OP_DESC_CLONE

} // namespace impl
} // namespace dnnl

#endif
