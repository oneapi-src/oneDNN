/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/// @file
/// C API definitions

#ifndef ONEAPI_DNNL_DNNL_GRAPH_TYPES_H
#define ONEAPI_DNNL_DNNL_GRAPH_TYPES_H

#if defined _WIN32 || defined __CYGWIN__
#define DNNL_GRAPH_HELPER_DLL_IMPORT __declspec(dllimport)
#define DNNL_GRAPH_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define DNNL_GRAPH_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define DNNL_GRAPH_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#else
#define DNNL_GRAPH_HELPER_DLL_IMPORT
#define DNNL_GRAPH_HELPER_DLL_EXPORT
#endif
#endif

#ifdef DNNL_GRAPH_DLL
#ifdef DNNL_GRAPH_DLL_EXPORTS
#define DNNL_GRAPH_API DNNL_GRAPH_HELPER_DLL_EXPORT
#else
#define DNNL_GRAPH_API DNNL_GRAPH_HELPER_DLL_IMPORT
#endif
#else
#define DNNL_GRAPH_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// @cond DO_NOT_DOCUMENT_THIS
#include <stddef.h>
#include <stdint.h>
/// @endcond

/// @addtogroup dnnl_graph_api
/// @{

/// @addtogroup dnnl_graph_api_logical_tensor
/// @{

/// Maximum dimensions
#define DNNL_GRAPH_MAX_NDIMS 12
/// An integer to indicate unknown number of dimensions
#define DNNL_GRAPH_UNKNOWN_NDIMS -1
/// An integer to indicate unknown dimensions
#define DNNL_GRAPH_UNKNOWN_DIM -1

/// A type to describe tensor dimension.
typedef int64_t dnnl_graph_dim_t;

/// A type to describe tensor dimensions.
typedef dnnl_graph_dim_t dnnl_graph_dims_t[DNNL_GRAPH_MAX_NDIMS];

/// Data type specifications
typedef enum {
    /// undefined data type for initialization
    dnnl_graph_data_type_undef = 0,
    /// 16-bit/half-precision floating point.
    dnnl_graph_f16 = 1,
    /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
    dnnl_graph_bf16 = 2,
    /// 32-bit/single-precision floating point.
    dnnl_graph_f32 = 3,
    /// 32-bit signed integer.
    dnnl_graph_s32 = 4,
    /// 8-bit signed integer.
    dnnl_graph_s8 = 5,
    /// 8-bit unsigned integer.
    dnnl_graph_u8 = 6,
    /// boolean data type. The tensor element will be interpreted with C++ bool
    /// type. Note that the size of C++ bool type is language implementation
    /// implementation defined.
    dnnl_graph_boolean = 8,
} dnnl_graph_data_type_t;

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_partition
/// @{

/// Policy specifications for partitioning
typedef enum {
    /// Best optimization
    dnnl_graph_partition_policy_max = 0,
    /// Have fusion
    dnnl_graph_partition_policy_fusion = 1,
    /// No optimization
    dnnl_graph_partition_policy_debug = 2,
} dnnl_graph_partition_policy_t;

typedef enum {
    dnnl_graph_partition_kind_undef = 0,
    dnnl_graph_partition_kind_convolution_post_ops = 1,
    dnnl_graph_partition_kind_convtranspose_post_ops = 2,
    dnnl_graph_partition_kind_interpolate_post_ops = 3,
    dnnl_graph_partition_kind_matmul_post_ops = 4,
    dnnl_graph_partition_kind_reduction_post_ops = 5,
    dnnl_graph_partition_kind_unary_post_ops = 6,
    dnnl_graph_partition_kind_binary_post_ops = 7,
    dnnl_graph_partition_kind_pooling_post_ops = 8,
    dnnl_graph_partition_kind_batch_norm_post_ops = 9,
    dnnl_graph_partition_kind_misc_post_ops = 10,
    dnnl_graph_partition_kind_quantized_convolution_post_ops = 11,
    dnnl_graph_partition_kind_quantized_convtranspose_post_ops = 12,
    dnnl_graph_partition_kind_quantized_matmul_post_ops = 13,
    dnnl_graph_partition_kind_quantized_unary_post_ops = 14,
    dnnl_graph_partition_kind_quantized_pooling_post_ops = 15,
    dnnl_graph_partition_kind_misc_quantized_post_ops = 16,
    dnnl_graph_partition_kind_convolution_backprop_post_ops = 17,
    dnnl_graph_partition_kind_mha = 18,
    dnnl_graph_partition_kind_mlp = 19,
    dnnl_graph_partition_kind_quantized_mha = 20,
    dnnl_graph_partition_kind_quantized_mlp = 21,
    dnnl_graph_partition_kind_residual_conv_blocks = 22,
    dnnl_graph_partition_kind_quantized_residual_conv_blocks = 23,
} dnnl_graph_partition_kind_t;

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_utils
/// @{

/// Status values returned by the library functions.
typedef enum {
    /// The operation was successful
    dnnl_graph_success = 0,
    /// The operation failed due to an out-of-memory condition
    dnnl_graph_out_of_memory = 1,
    /// The operation failed because of incorrect function arguments
    dnnl_graph_invalid_arguments = 2,
    /// The operation failed because requested functionality is not implemented
    dnnl_graph_unimplemented = 3,
    /// Primitive iterator passed over last primitive descriptor
    dnnl_graph_iterator_ends = 4,
    /// Primitive or engine failed on execution
    dnnl_graph_runtime_error = 5,
    /// Queried element is not required for given primitive
    dnnl_graph_not_required = 6,
    /// The graph is not legitimate
    dnnl_graph_invalid_graph = 7,
    /// The operation is not legitimate according to op schema
    dnnl_graph_invalid_graph_op = 8,
    /// The shape cannot be inferred or compiled
    dnnl_graph_invalid_shape = 9,
    /// The data type cannot be inferred or compiled
    dnnl_graph_invalid_data_type = 10,
} dnnl_graph_status_t;

/// @} dnnl_graph_api_utils

/// @addtogroup dnnl_graph_api_engine
/// @{

/// Kind for engine
typedef enum {
    /// An unspecified engine
    dnnl_graph_any_engine = 0,
    /// CPU engine
    dnnl_graph_cpu = 1,
    /// GPU engine
    dnnl_graph_gpu = 2,
} dnnl_graph_engine_kind_t;

/// @} dnnl_graph_api_engine

/// @addtogroup dnnl_graph_api_graph
/// @{

/// floating-point math mode
typedef enum {
    /// Default behavior, no downconversions allowed
    dnnl_graph_fpmath_mode_strict = 0,
    /// Implicit f32->bf16 or f32->tf32 conversions allowed
    dnnl_graph_fpmath_mode_bf16 = 1,
    /// Implicit f32->f16 or f32->tf32 conversions allowed
    dnnl_graph_fpmath_mode_f16 = 2,
    /// Implicit f32->f16 or f32->bf16 or f32->tf32 conversions allowed
    dnnl_graph_fpmath_mode_any = 3,
    /// Implicit f32->tf32 conversions allowed
    dnnl_graph_fpmath_mode_tf32 = 4,
} dnnl_graph_fpmath_mode_t;

/// @}

/// @addtogroup dnnl_graph_api_op
/// @{

/// Kinds of operations
typedef enum {
    dnnl_graph_op_abs,
    dnnl_graph_op_abs_backprop,
    dnnl_graph_op_add,
    dnnl_graph_op_avg_pool,
    dnnl_graph_op_avg_pool_backprop,
    dnnl_graph_op_batch_norm_backprop,
    dnnl_graph_op_batch_norm_forward_training,
    dnnl_graph_op_batch_norm_inference,
    dnnl_graph_op_bias_add,
    dnnl_graph_op_bias_add_backprop,
    dnnl_graph_op_clamp,
    dnnl_graph_op_clamp_backprop,
    dnnl_graph_op_concat,
    dnnl_graph_op_convolution,
    dnnl_graph_op_convolution_backprop_data,
    dnnl_graph_op_convolution_backprop_filters,
    dnnl_graph_op_conv_transpose,
    dnnl_graph_op_conv_transpose_backprop_data,
    dnnl_graph_op_conv_transpose_backprop_filters,
    dnnl_graph_op_dequantize,
    dnnl_graph_op_divide,
    dnnl_graph_op_dynamic_dequantize,
    dnnl_graph_op_dynamic_quantize,
    dnnl_graph_op_dynamic_reshape,
    dnnl_graph_op_dynamic_transpose,
    dnnl_graph_op_elu,
    dnnl_graph_op_elu_backprop,
    dnnl_graph_op_end,
    dnnl_graph_op_erf,
    dnnl_graph_op_exp,
    dnnl_graph_op_gelu,
    dnnl_graph_op_gelu_backprop,
    dnnl_graph_op_hard_swish,
    dnnl_graph_op_hard_swish_backprop,
    dnnl_graph_op_index,
    dnnl_graph_op_interpolate,
    dnnl_graph_op_interpolate_backprop,
    dnnl_graph_op_layer_norm,
    dnnl_graph_op_layer_norm_backprop,
    dnnl_graph_op_leaky_relu,
    dnnl_graph_op_log,
    dnnl_graph_op_log_softmax,
    dnnl_graph_op_log_softmax_backprop,
    dnnl_graph_op_logical_and,
    dnnl_graph_op_logical_not,
    dnnl_graph_op_logical_or,
    dnnl_graph_op_logical_xor,
    dnnl_graph_op_matmul,
    dnnl_graph_op_maximum,
    dnnl_graph_op_max_pool,
    dnnl_graph_op_max_pool_backprop,
    dnnl_graph_op_minimum,
    dnnl_graph_op_mish,
    dnnl_graph_op_mish_backprop,
    dnnl_graph_op_multiply,
    dnnl_graph_op_negative,
    dnnl_graph_op_pow,
    dnnl_graph_op_pow_backprop,
    dnnl_graph_op_pow_backprop_exponent,
    dnnl_graph_op_prelu,
    dnnl_graph_op_prelu_backprop,
    dnnl_graph_op_quantize,
    dnnl_graph_op_reciprocal,
    dnnl_graph_op_reduce_l1,
    dnnl_graph_op_reduce_l2,
    dnnl_graph_op_reduce_max,
    dnnl_graph_op_reduce_mean,
    dnnl_graph_op_reduce_min,
    dnnl_graph_op_reduce_prod,
    dnnl_graph_op_reduce_sum,
    dnnl_graph_op_relu,
    dnnl_graph_op_relu_backprop,
    dnnl_graph_op_reorder,
    dnnl_graph_op_round,
    dnnl_graph_op_select,
    dnnl_graph_op_sigmoid,
    dnnl_graph_op_sigmoid_backprop,
    dnnl_graph_op_sign,
    dnnl_graph_op_softmax,
    dnnl_graph_op_softmax_backprop,
    dnnl_graph_op_softplus,
    dnnl_graph_op_softplus_backprop,
    dnnl_graph_op_sqrt,
    dnnl_graph_op_sqrt_backprop,
    dnnl_graph_op_square,
    dnnl_graph_op_squared_difference,
    dnnl_graph_op_static_reshape,
    dnnl_graph_op_static_transpose,
    dnnl_graph_op_subtract,
    dnnl_graph_op_tanh,
    dnnl_graph_op_tanh_backprop,
    dnnl_graph_op_type_cast,
    dnnl_graph_op_wildcard,
    dnnl_graph_op_last_symbol,
} dnnl_graph_op_kind_t;

/// Attributes of operations
typedef enum {
    dnnl_graph_op_attr_undef = 0,

    // float32 attributes
    dnnl_graph_op_attr_alpha = 1,
    dnnl_graph_op_attr_beta,
    dnnl_graph_op_attr_epsilon,
    dnnl_graph_op_attr_max,
    dnnl_graph_op_attr_min,
    dnnl_graph_op_attr_momentum,

    // float32 vector attributes
    dnnl_graph_op_attr_scales,

    // int64_t attributes
    dnnl_graph_op_attr_axis = 0x20,
    dnnl_graph_op_attr_begin_norm_axis,
    dnnl_graph_op_attr_groups,

    // int64_t vector attributes
    dnnl_graph_op_attr_axes,
    dnnl_graph_op_attr_dilations,
    dnnl_graph_op_attr_filter_shape,
    dnnl_graph_op_attr_input_shape,
    dnnl_graph_op_attr_kernel,
    dnnl_graph_op_attr_order,
    dnnl_graph_op_attr_output_padding,
    dnnl_graph_op_attr_output_shape,
    dnnl_graph_op_attr_pads_begin,
    dnnl_graph_op_attr_pads_end,
    dnnl_graph_op_attr_shape,
    dnnl_graph_op_attr_sizes,
    dnnl_graph_op_attr_strides,
    dnnl_graph_op_attr_zps,

    // bool attributes
    dnnl_graph_op_attr_exclude_pad = 0x40,
    dnnl_graph_op_attr_keep_dims,
    dnnl_graph_op_attr_keep_stats,
    dnnl_graph_op_attr_per_channel_broadcast,
    dnnl_graph_op_attr_special_zero,
    dnnl_graph_op_attr_transpose_a,
    dnnl_graph_op_attr_transpose_b,
    dnnl_graph_op_attr_use_affine,
    dnnl_graph_op_attr_use_dst,

    // string attributes
    dnnl_graph_op_attr_auto_broadcast = 0x60,
    dnnl_graph_op_attr_auto_pad,
    dnnl_graph_op_attr_coordinate_transformation_mode,
    dnnl_graph_op_attr_data_format,
    dnnl_graph_op_attr_filter_format,
    dnnl_graph_op_attr_mode,
    dnnl_graph_op_attr_qtype,
    dnnl_graph_op_attr_rounding_type,
} dnnl_graph_op_attr_t;

/// @} dnnl_graph_api_op

/// @addtogroup dnnl_graph_api_allocator
/// @{

/// Allocation call-back function interface for CPU
typedef void *(*dnnl_graph_host_allocate_f)(size_t size, size_t alignment);

/// Deallocation call-back function interface for CPU
typedef void (*dnnl_graph_host_deallocate_f)(void *);

/// Allocation call-back function interface for SYCL device
typedef void *(*dnnl_graph_sycl_allocate_f)(
        size_t size, size_t alignment, const void *dev, const void *context);

/// brief Deallocation call-back function interface for SYCL device
typedef void (*dnnl_graph_sycl_deallocate_f)(
        void *buf, const void *dev, const void *context, void *event);

/// @} dnnl_graph_api_allocator

/// @addtogroup dnnl_graph_api_logical_tensor
/// @{

/// Layout type specification
typedef enum {
    /// undefined layout type
    dnnl_graph_layout_type_undef = 0,
    /// any means that oneDNN graph implementation needs to decide the
    /// layout for the compiled partition.
    dnnl_graph_layout_type_any = 1,
    /// strided means that the layout is determined by the strides field.
    dnnl_graph_layout_type_strided = 2,
    /// opaque means that the layout is a target-specific layout decided by
    /// oneDNN graph implementation.
    dnnl_graph_layout_type_opaque = 3,
} dnnl_graph_layout_type_t;

/// Logical tensor property
typedef enum {
    /// undefined tensor property
    dnnl_graph_tensor_property_undef = 0,
    /// variable means the tensor can be changed among iterations
    dnnl_graph_tensor_property_variable = 1,
    /// constant means the tensor will keep unchanged among iterations
    dnnl_graph_tensor_property_constant = 2,
} dnnl_graph_tensor_property_t;

/// @brief logical tensor definition
typedef struct {
    /// Unique id of each logical tensor. Provided by framework.
    size_t id;

    /// Number of dimension. Default -1 means not initialized.
    int32_t ndims;

    /// Size of each dimension. -1 means the size is unknown on the axis.
    dnnl_graph_dims_t dims;

    /// Data type of the tensor elements.
    dnnl_graph_data_type_t data_type;

    /// Tensor property: undef, variable, or constant.
    dnnl_graph_tensor_property_t property;

    /// Layout type of the tensor: any, strided, or opaque.
    dnnl_graph_layout_type_t layout_type;
    union {
        /// Valid when layout_type is `dnnl_graph_strided`.
        /// -1 means the stride is unknown on the axis.
        dnnl_graph_dims_t strides;

        /// Valid when layout_type is `dnnl_graph_opaque`.
        /// `layout_id` is generated and managed by backend.
        size_t layout_id;
    } layout;
} dnnl_graph_logical_tensor_t;

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_compiled_partition
/// @{

/// @brief In-place pair definition
typedef struct {
    /// The id of input tensor
    size_t input_id;

    /// The id of output tensor
    size_t output_id;
} dnnl_graph_inplace_pair_t;

/// @} dnnl_graph_api_compiled_partition

/// @cond DO_NOT_DOCUMENT_THIS

/// @brief An allocator handle
struct dnnl_graph_allocator;
typedef struct dnnl_graph_allocator *dnnl_graph_allocator_t;
typedef const struct dnnl_graph_allocator *const_dnnl_graph_allocator_t;

/// @brief A tensor handle
struct dnnl_graph_tensor;
typedef struct dnnl_graph_tensor *dnnl_graph_tensor_t;
typedef const struct dnnl_graph_tensor *const_dnnl_graph_tensor_t;

/// @brief A op handle
struct dnnl_graph_op;
typedef struct dnnl_graph_op *dnnl_graph_op_t;
typedef const struct dnnl_graph_op *const_dnnl_graph_op_t;

/// @brief A partition handle
struct dnnl_graph_partition;
typedef struct dnnl_graph_partition *dnnl_graph_partition_t;
typedef const struct dnnl_graph_partition *const_dnnl_graph_partition_t;

/// @brief A compiled partition handle
struct dnnl_graph_compiled_partition;
typedef struct dnnl_graph_compiled_partition *dnnl_graph_compiled_partition_t;
typedef const struct dnnl_graph_compiled_partition
        *const_dnnl_graph_compiled_partition_t;

/// @brief A graph handle
struct dnnl_graph_graph;
typedef struct dnnl_graph_graph *dnnl_graph_graph_t;
typedef const struct dnnl_graph_graph *const_dnnl_graph_graph_t;

/// @brief An engine handle
struct dnnl_graph_engine;
typedef struct dnnl_graph_engine *dnnl_graph_engine_t;
typedef const struct dnnl_graph_engine *const_dnnl_graph_engine_t;

/// @brief A stream handle
struct dnnl_graph_stream;
typedef struct dnnl_graph_stream *dnnl_graph_stream_t;
typedef const struct dnnl_graph_stream *const_dnnl_graph_stream_t;

/// @endcond

/// @addtogroup dnnl_graph_api_service
/// @{

/// Structure containing version information as per [Semantic
/// Versioning](https://semver.org)
typedef struct {
    int major; ///< Major version
    int minor; ///< Minor version
    int patch; ///< Patch version
    const char *hash; ///< Git hash of the sources (may be absent)
} dnnl_graph_version_t;

/// No runtime (disabled)
#define DNNL_GRAPH_RUNTIME_NONE 0u

/// Sequential runtime (CPU only)
#define DNNL_GRAPH_RUNTIME_SEQ 1u

/// OpenMP runtime (CPU only)
#define DNNL_GRAPH_RUNTIME_OMP 2u

/// TBB runtime (CPU only)
#define DNNL_GRAPH_RUNTIME_TBB 4u

/// Threadpool runtime (CPU only)
#define DNNL_GRAPH_RUNTIME_THREADPOOL 8u

/// SYCL runtime
#define DNNL_GRAPH_RUNTIME_SYCL 512u

/// DPC++ runtime
#define DNNL_GRAPH_RUNTIME_DPCPP DNNL_GRAPH_RUNTIME_SYCL

/// @} dnnl_graph_api_service

/// @} dnnl_graph_api

#ifdef __cplusplus
}
#endif
#endif
