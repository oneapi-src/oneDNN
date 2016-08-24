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

#ifndef MKLDNN_TYPES_H
#define MKLDNN_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stddef.h>
#include <stdint.h>
#endif

/** @addtogroup c_api C API
 *  @{
 *
 *  @addtogroup c_api_types Types
 *  @{
 *
 *  @addtogroup c_api_types_generic Generic
 *  @{ */

/** Status values returned by Intel(R) MKL-DNN functions. */
typedef enum {
    /** The operation was successful */
    mkldnn_success = 0,
    /** The operation failed due to an out-of-memory condition */
    mkldnn_out_of_memory = 1,
    /** The operation failed and should be retried */
    mkldnn_try_again = 2,
    /** The operation failed because of incorrect function arguments  */
    mkldnn_invalid_arguments = 3,
    /** The operation failed because a primitive was not ready for execution */
    mkldnn_not_ready = 4,
    /** The operation failed because requested functionality is not implemented
     */
    mkldnn_unimplemented = 5,
} mkldnn_status_t;

/* TODO: rename to mkldnn_data_type_t */
/** Data type specification */
typedef enum {
    /** Undefined precision, used for empty memory descriptors. */
    mkldnn_precision_undef = 0,
    /** 32-bit/single-precision floating point. */
    mkldnn_f32 = 1,
    /** 32-bit unsigned integer. */
    mkldnn_u32 = 2,
} mkldnn_precision_t;

/** Memory format specification.
 *
 * Intel(R) MKL-DNN uses the following notation for memory format names: 
 *  - @c 'n' denotes the mini-batch dimension
 *  - @c 'c' denotes a channels dimension
 *  - When there are multiple channel dimensions (for example, in convolution
 *    weights tensor), @c 'i' and @c 'o' denote dimensions of input and output channels
 *  - @c 'h' and @c 'w' denote spatial width and height
 *  - Upper-case letters indicate that the data is laid out in blocks
 *    for a particular dimension. In such cases, the format name contains both upper-
 *    and lower-case letters for that dimension with lower-case letter preceded
 *    by the block size. For example: @c 'mkldnn_nChw8c' describes a format where
 *    the outermost dimension is mini-batch, followed by the channel block number,
 *    followed by the spatial height and width, and finally followed by 8-element
 *    channel blocks.
 * 
 * Note that channel designations can be different. For example:
 * both the @c 'mkldnn_nc' and @c 'mkldnn_io' formats can be used to describe a 2D tensor.
 */
typedef enum {
    /** Undefined memory format, used for empty memory descriptors. */
    mkldnn_format_undef = 0,
    /** Unspecified format. The primitive selects a format
     * automatically. */
    mkldnn_any,
    /** A tensor in a generic format described by the stride and blocking values
     * in each dimension. See #mkldnn_blocking_desc_t for more information. */
    mkldnn_blocked,
    /** 1D data tensor. */
    mkldnn_x,
    /** 2D data tensor. */
    mkldnn_nc,
    /** 4D data tensor in the @c nchw format typically used in Caffe. */
    mkldnn_nchw,
    /** 4D data tensor in the @c nhwc format typically used in TensorFlow. */
    mkldnn_nhwc,
    /** 4D data tensor in the @c nchw format with channels data laid out in memory in
	* 8-element blocks. */
    mkldnn_nChw8c,
    /** 2D weights tensor in the format (input channels, output channels). */
    mkldnn_oi,
    /** 4D weights tensor in the format (input channels, output channels, width,
     * height). */
    mkldnn_oihw,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 8-element blocks. */
    mkldnn_OIhw8i8o,
    /** 4D weights tensor in the format (output channels, width, height,
     * input channels) with output channels data laid out in memory
     * in 8-element blocks. */
    mkldnn_Ohwi8o,
    /** 5D weights tensor in the @c oihw format with extra outer dimension for
     * groups. */
    mkldnn_goihw,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 8-element blocks. */
    mkldnn_gOIhw8i8o,
    /** 4D weights tensor in the oihw format with input channels data laid out in memory in
	* 8-element blocks. */
    mkldnn_oIhw8i = mkldnn_nChw8c,
} mkldnn_memory_format_t;

/** Kinds of padding. Define how to interpret the data in padding regions. */
typedef enum {
    /** The data in padding regions is zero. */
    mkldnn_padding_zero,
} mkldnn_padding_kind_t;

/** Kinds of propagation. */
typedef enum {
    /* TODO: suggest renames */
    /** Forward data propagation (training mode). In this mode primitives
     * perform computations necessary for subsequent backward propagation. */
    mkldnn_forward_training = 1,
    /** Forward data propagation (scoring mode). In this mode primitives only
     * perform computations that are necessary for scoring and omit
     * computations that are only necessary for backward propagation. */
    mkldnn_forward_scoring = 2,
   /** Forward data propagation (alias for @c mkldnn_forward_training) */
    mkldnn_forward = mkldnn_forward_training,
    /** Backward data propagation */
    mkldnn_backward_data = 3,
    /** Backward weights propagation */
    mkldnn_backward_weights = 4,
    /** Backward bias propagation */
    mkldnn_backward_bias = 5,
} mkldnn_prop_kind_t;

/** Kinds of algorithms. */
typedef enum {
    /** Direct convolution */
    mkldnn_convolution_direct = 1,
    /** Max pooling */
    mkldnn_pooling_max = 101,
    /** Local (LRN) response normalization across multiple channels */
    mkldnn_lrn_across_channels = 201,
    /** LRN within a single channel */
    mkldnn_lrn_within_channel = 202,
} mkldnn_alg_kind_t;

/** @} */

/** @addtogroup c_api_types_memory Memory description
 *  @{ */

/** Maximum number of dimensions a tensor can have. Only restricts
 * the amount of space used for the tensor description. Individual computational
 * primitives may support only tensors of certain dimensions. */
#define TENSOR_MAX_DIMS 12U

/** A type to describe tensor dimensions. */
typedef uint32_t mkldnn_dims_t[TENSOR_MAX_DIMS];
/** A type to describe offsets within a tensor. */
typedef int32_t mkldnn_nd_offset_t[TENSOR_MAX_DIMS];

/** A tensor descriptor. The description is not tied to any memory format, but
 * enables describing tensor dimensions as belonging to mini-batch,
 * channel/feature map, or spatial kind. Intel(R) MKL-DNN uses this type when a
 * mathematical description of data is required. */
typedef struct {
    /** Number dimensions */
    uint32_t ndims;
    /** Tensor dimensions in the following order: mini-batch, channel, spatial.
     * For example: <code>{N, C, H, W}</code>. */
    mkldnn_dims_t dims;
} mkldnn_tensor_desc_t;

/** Generic description of blocked data layout for most memory formats. */
typedef struct {
    /** Block size for each of the dimensions. */
    mkldnn_dims_t block_dims;
    /** strides[0]: stride between the first elements of adjacent blocks.
     * @n strides[1]: strides between elements in the same block. */
    mkldnn_dims_t strides[2];
    /** Size of the data including padding in each dimension. */
    mkldnn_dims_t padding_dims;
    /** Per-dimension offset from the padding to actual data, the top-level
     * tensor with offsets applied must lie within the padding area. */
    mkldnn_dims_t offset_padding_to_data;
    /** Offset from memory origin to the current block, non-zero only in
     * a description of a memory sub-block. */
    size_t offset_padding;
} mkldnn_blocking_desc_t;

/** Memory descriptor. The description is based on a tensor description plus
 * information about elements precision and memory layout. Additionally,
 * contains format-specific descriptions of the data layout. */
typedef struct {
    /** A descriptor of the underlying tensor. */
    mkldnn_tensor_desc_t tensor_desc;
    /** Data type of the tensor elements. */
    mkldnn_precision_t precision;
    /** Memory format. */
    mkldnn_memory_format_t format;
    union {
        /** Description of the data layout for memory formats that use blocking. */
        mkldnn_blocking_desc_t blocking;
        /* ... other descriptions possible */
    } layout_desc;
} mkldnn_memory_desc_t;

/** @} */

/** @addtogroup c_api_types_op_descs Operation descriptors
 *  @{*/

/** A descriptor of a convolution operation. */
typedef struct {
    /** The kind of propagation. Possible values: #mkldnn_forward,
     * #mkldnn_backward_data, #mkldnn_backward_weights,
     * and #mkldnn_backward_bias. */
    mkldnn_prop_kind_t prop_kind;
    /** The kind of the convolution algorithm. Possible values:
     * #mkldnn_convolution_direct. */
    mkldnn_alg_kind_t alg_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Weights memory descriptor. */
    mkldnn_memory_desc_t weights_desc;
    /** (Optional) bias memory descriptor. */
    mkldnn_memory_desc_t bias_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Convolution strides in each spatial dimension. */
    mkldnn_dims_t strides;
    /** Padding in each spatial dimension (only symmetric padding is currently
     * supported). */
    mkldnn_nd_offset_t padding;
    /** The kind of padding to use. */
    mkldnn_padding_kind_t padding_kind;
} mkldnn_convolution_desc_t;

/** A descriptor of a pooling operation. */
typedef struct {
    /** The kind of propagation. Possible values: #mkldnn_forward
     * and #mkldnn_backward_data. */
    mkldnn_prop_kind_t prop_kind;
    /** The kind of pooling algorithm. Possible values:
     * #mkldnn_pooling_max. */
    mkldnn_alg_kind_t alg_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Pooling kernel strides for spatial dimensions. */
    mkldnn_dims_t strides;
    /** Pooling kernel spatial dimensions. */
    mkldnn_dims_t kernel;
    /** Padding in each spatial dimension (only symmetric padding is currently
     * supported). */
    mkldnn_nd_offset_t padding;
    /** The kind of padding to use. */
    mkldnn_padding_kind_t padding_kind;
} mkldnn_pooling_desc_t;

/** A descriptor of a rectifier linear unit (ReLU) operation. */
typedef struct {
    /** The kind of propagation. Possible values: #mkldnn_forward
     * and #mkldnn_backward_data. */
    mkldnn_prop_kind_t prop_kind;
    /** Scaling factor for negative values. Stored as double-precision, but
     * interpreted in a way specific to the data type in each implementation. */
    double negative_slope;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
} mkldnn_relu_desc_t;

/** A descriptor of a Local Response Normalization (LRN) operation. */
typedef struct {
    /** The kind of propagation. Possible values: #mkldnn_forward
     * and #mkldnn_backward_data. */
    mkldnn_prop_kind_t prop_kind;
    /** LRN algorithm: #mkldnn_lrn_within_channel
     * or #mkldnn_lrn_across_channels. */
    mkldnn_alg_kind_t alg_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** LRN alpha parameter. */
    double alpha;
    /** LRN beta parameter. */
    double beta;
    /** The number of channels to sum over (for cross-channel LRN) or the side
     * length of the square region to sum over (for within-channel LRN). */
    uint32_t local_size;
} mkldnn_lrn_desc_t;

/** A descriptor of an inner product operation. */
typedef struct {
    /** The kind of propagation. Possible values: #mkldnn_forward,
     * #mkldnn_backward_data, and #mkldnn_backward_weights. */
    mkldnn_prop_kind_t prop_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Weights memory descriptor. */
    mkldnn_memory_desc_t weights_desc;
    /** Bias memory descriptor. */
    mkldnn_memory_desc_t bias_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
} mkldnn_inner_product_desc_t;

/** A descriptor of a convolution followed by relu operation. */
typedef struct {
    /** A descriptor of a convolution operation. */
    mkldnn_convolution_desc_t convolution_desc;
    /** Scaling factor for negative values, stored as double-precision but
     * interpreted in a way specific to the data type in each implementation */
    double negative_slope;
} mkldnn_convolution_relu_desc_t;

/** @} */

/** @addtogroup c_api_engine_types Engine
 * @{ */

/** @brief Kinds of engines. */
typedef enum {
    /** An unspecified engine. */
    mkldnn_any_engine,
    /** CPU engine. */
    mkldnn_cpu,
    /** CPU engine in a lazy execution mode. */
    mkldnn_cpu_lazy,
} mkldnn_engine_kind_t;

/** @struct mkldnn_engine
 * @brief An opaque structure to describe an engine. */
struct mkldnn_engine;
/** @brief An engine handle. */
typedef struct mkldnn_engine *mkldnn_engine_t;
/** @brief A constant engine handle. */
typedef const struct mkldnn_engine *const_mkldnn_engine_t;

/** @} */

/** @addtogroup c_api_primitive_descs Primitive descriptors
 * @{ */

/** Kinds of primitives. Used to implement a way to extend the library with new
 * primitives without changing the ABI. */
typedef enum {
    /** Undefined primitive (XXX: why do we have it?). */
    mkldnn_undefined_primitive,
    /** A memory primitive. */
    mkldnn_memory,
    /** A reorder primitive.*/
    mkldnn_reorder,
    /** A convolution primitive. */
    mkldnn_convolution,
    /** A pooling primitive. */
    mkldnn_pooling,
    /** A ReLU primitive. */
    mkldnn_relu,
    /** An LRN primitive. */
    mkldnn_lrn,
    /** An inner product primitive. */
    mkldnn_inner_product,
    /** A convolution primitive merged with relu */
    mkldnn_convolution_relu,
} mkldnn_primitive_kind_t;

/** Basic primitive descriptor. It is the first member of all
 * primitive-specific descriptors, which makes it possible to use its data to
 * correctly interpret a #mkldnn_primitive_desc_t pointer. */
typedef struct {
    /** Kind of the primitive being described. */
    mkldnn_primitive_kind_t primitive_kind; // TODO: rename to just kind
    /** Engine used by the primitive. */
    const_mkldnn_engine_t engine;
    /** Pointer to a library-internal structure. */
    const void *implementation;
} mkldnn_primitive_base_desc_t;

/** Descriptor of a memory primitive. */
typedef struct {
    /** Basic primitive descriptor. */
    mkldnn_primitive_base_desc_t base;
    /** Descriptor of the underlying memory.  */
    mkldnn_memory_desc_t memory_desc;
} mkldnn_memory_primitive_desc_t;

/** Descriptor of a reorder primitive. */
typedef struct {
    /** Basic primitive descriptor. */
    mkldnn_primitive_base_desc_t base;
    /** Descriptor of the input memory primitive. */
    mkldnn_memory_primitive_desc_t input;
    /** Descriptor of the output memory primitive. */
    mkldnn_memory_primitive_desc_t output;
} mkldnn_reorder_primitive_desc_t;

/** Descriptor of convolution primitive. */
typedef struct {
    /** Basic primitive descriptor. */
    mkldnn_primitive_base_desc_t base;
    /** Descriptor of the underlying convolution operation. */
    mkldnn_convolution_desc_t convolution_desc;
    /** Descriptor of the source memory primitive. */
    mkldnn_memory_primitive_desc_t src_primitive_desc;
    /** Descriptor of the weights memory primitive. */
    mkldnn_memory_primitive_desc_t weights_primitive_desc;
    /** Descriptor of the bias memory primitive. */
    mkldnn_memory_primitive_desc_t bias_primitive_desc;
    /** Descriptor of the destination memory primitive. */
    mkldnn_memory_primitive_desc_t dst_primitive_desc;
} mkldnn_convolution_primitive_desc_t;

typedef struct {
    /** Basic primitive descriptor. */
    mkldnn_primitive_base_desc_t base;
    /** Descriptor of the underlying pooling operation. */
    mkldnn_pooling_desc_t pooling_desc;
    /** Descriptor of the source memory primitive. */
    mkldnn_memory_primitive_desc_t src_primitive_desc;
    /** Descriptor of the indices memory primitive. */
    mkldnn_memory_primitive_desc_t indices_primitive_desc;
    /** Descriptor of the destination memory primitive. */
    mkldnn_memory_primitive_desc_t dst_primitive_desc;
} mkldnn_pooling_primitive_desc_t;

typedef struct {
    /** Basic primitive descriptor. */
    mkldnn_primitive_base_desc_t base;
    mkldnn_relu_desc_t relu_desc;
    /** Descriptor of the source memory primitive. */
    mkldnn_memory_primitive_desc_t src_primitive_desc;
    /** Descriptor of the destination memory primitive. */
    mkldnn_memory_primitive_desc_t dst_primitive_desc;
} mkldnn_relu_primitive_desc_t;

typedef struct {
    /** Basic primitive descriptor. */
    mkldnn_primitive_base_desc_t base;
    mkldnn_lrn_desc_t lrn_desc;
    /** Descriptor of the source memory primitive. */
    mkldnn_memory_primitive_desc_t src_primitive_desc;
    /** Descriptor of the scratch/workspace memory primitive. */
    mkldnn_memory_primitive_desc_t scratch_primitive_desc;
    /** Descriptor of the destination memory primitive. */
    mkldnn_memory_primitive_desc_t dst_primitive_desc;
} mkldnn_lrn_primitive_desc_t;

typedef struct {
    /** Basic primitive descriptor. */
    mkldnn_primitive_base_desc_t base;
    mkldnn_inner_product_desc_t inner_product_desc;
    /** Descriptor of the source memory primitive. */
    mkldnn_memory_primitive_desc_t src_primitive_desc;
    /** Descriptor of the weights memory primitive. */
    mkldnn_memory_primitive_desc_t weights_primitive_desc;
    /** Descriptor of the bias memory primitive. */
    mkldnn_memory_primitive_desc_t bias_primitive_desc;
    /** Descriptor of the destination memory primitive. */
    mkldnn_memory_primitive_desc_t dst_primitive_desc;
} mkldnn_inner_product_primitive_desc_t;

/** Descriptor of a convolution followed by relu primitive */
typedef struct {
    /** Basic primitive descriptor */
    mkldnn_primitive_base_desc_t base;
    /** Descriptor of the underlying convolution followed by relu operation */
    mkldnn_convolution_relu_desc_t convolution_relu_desc;
    /** Descriptor of the source memory primitive */
    mkldnn_memory_primitive_desc_t src_primitive_desc;
    /** Descriptor of the weights memory primitive */
    mkldnn_memory_primitive_desc_t weights_primitive_desc;
    /** Descriptor of the bias memory primitive */
    mkldnn_memory_primitive_desc_t bias_primitive_desc;
    /** Descriptor of the destination memory primitive */
    mkldnn_memory_primitive_desc_t dst_primitive_desc;
} mkldnn_convolution_relu_primitive_desc_t;

/** A pointer to any of the primitive descriptors. */
typedef void *mkldnn_primitive_desc_t;
/** A pointer to any of the primitive descriptors (constant variant). */
typedef const void *const_mkldnn_primitive_desc_t;

/** @} */

/** @addtogroup c_api_types_primitive Primitive
 * @{ */

/** @struct mkldnn_primitive
 * An opaque structure to describe a primitive. */
struct mkldnn_primitive;
/** A primitive handle. */
typedef struct mkldnn_primitive *mkldnn_primitive_t;
/** A constant primitive handle. */
typedef const struct mkldnn_primitive *const_mkldnn_primitive_t;

/** A wrapper structure to specify a particular output of a primitive. */
typedef struct {
    /** Primitive to specify the output for. */
    const_mkldnn_primitive_t primitive;
    /** Desired output index. */
    size_t output_index;
} mkldnn_primitive_at_t;

/** @} */

/** @addtogroup c_api_types_stream Execution stream
 * @{ */

/** @struct mkldnn_stream
 * An opaque structure to describe an execution stream. */
struct mkldnn_stream;
/** An execution stream handle. */
typedef struct mkldnn_stream *mkldnn_stream_t;
/** A constant execution stream handle. */
typedef const struct mkldnn_stream *const_mkldnn_stream_t;

/** @} */
/** @} */
/** @} */

#ifdef __cplusplus
}
#endif


#endif
