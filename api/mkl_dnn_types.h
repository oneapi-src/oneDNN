#ifndef MKL_DNN_TYPES_H
#define MKL_DNN_TYPES_H

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

/** Status values returned by MKL-DNN functions */
typedef enum {
    /** An operation was successful */
    mkl_dnn_success = 0,
    /** An operation failed due to an out-of-memory condition */
    mkl_dnn_out_of_memory = 1,
    /** An operation failed and should be retried */
    mkl_dnn_try_again = 2,
    /** An operation failed because function arguments were incorrect */
    mkl_dnn_invalid_arguments = 3,
    /** An operation failed because a primitive was not ready for execution */
    mkl_dnn_not_ready = 4,
    /** An operation failed because requested functionality is not implemented
     */
    mkl_dnn_unimplemented = 5,
} mkl_dnn_status_t;

/* TODO: rename to mkl_dnn_data_type_t */
/** Data type specification */
typedef enum {
    /** undefined precision; used for empty memory descriptors */
    mkl_dnn_precision_undef = 0,
    /** 32-bit/single-precision floating point */
    mkl_dnn_f32 = 1,
    /** 32-bit unsigned integer */
    mkl_dnn_u32 = 2,
} mkl_dnn_precision_t;

/** Memory format specification.
 *
 * MKL-DNN uses the following notation for memory format names. Letter 'n' is
 * used to denote the minibatch dimension; letter 'c', is used for channels
 * dimension; when there are multiple channel dimensions (e.g. in convolution
 * weights tensor), letters 'i' and 'o' are used to denote input and output
 * channels dimensions; 'h' and 'w' are used for spatial width and height.
 * MKL-DNN uses upper-case letters to indicate that data is laid out in blocks
 * for a particular dimension. In this case, a format name contains both upper-
 * and lower-case letter for such a dimension with lower-case letter preceded
 * by the block size. For example, 'mkl_dnn_nChw8c' describes a format where
 * the outermost dimension is minibatch, followed by channel block number,
 * followed by spatial height and width, and then followed by 8-element channel
 * blocks. Note that channel designations are a convention. Thus both the
 * 'mkl_dnn_nc' and 'mkl_dnn_io' formats can be used to describe any 2D tensor.
 */
typedef enum {
    /** Undefined memory format; used for empty memory descriptors */
    mkl_dnn_format_undef = 0,
    /** Unspecified format; use this to let a primitive to select a format
     * automatically */
    mkl_dnn_any,
    /** A tensor in a generic format described by stride and blocking values in
     * each dimension. See #mkl_dnn_blocking_desc_t for more information. */
    mkl_dnn_blocked,
    /** 1D data tensor */
    mkl_dnn_x,
    /** 2D data tensor */
    mkl_dnn_nc,
    /** 4D data tensor in the nchw format typically used in Caffe */
    mkl_dnn_nchw,
    /** 4D data tensor in the nhwc format typically used in TensorFlow */
    mkl_dnn_nhwc,
    /** 4D data tensor in the nchw format with channels blocked by 8 */
    mkl_dnn_nChw8c,
    /** 2D weights tensor in the (input channels, output channels) format */
    mkl_dnn_oi,
    /** 4D weights tensor in the (input channels, output channels, width,
     * height) format */
    mkl_dnn_oihw,
    /** 4D weights tensor in the oihw format with both input and output
     * channels blocked by 8 */
    mkl_dnn_OIhw8i8o,
    /** 5D weights tensor in the oihw format with extra outer dimension for
     * groups */
    mkl_dnn_goihw,
    /** 5D weights tensor in the blocked version of goihw format with both
     * input and output channels blocked by 8 */
    mkl_dnn_gOIhw8i8o,
} mkl_dnn_memory_format_t;

/** Padding kind -- how to interpret the data in the padding regions */
typedef enum {
    /** Assume that the data in padding regions is zero */
    mkl_dnn_padding_zero,
} mkl_dnn_padding_kind_t;

/** Propagation kind */
typedef enum {
    /** Forward data propagation (training mode). In this mode primitives
     * perform computations necessary for subsequent backward propagation. */
    mkl_dnn_forward = 1,
#if 0
    /* TODO: suggest renames */
    mkl_dnn_forward_training = 1,
    /** Forward data propagation (scoring mode). In this mode primitives only
     * perform computations that are necessary for scoring and omit
     * computations that are only necessary for backward propagation. */
    mkl_dnn_forward_scoring = 1,
#endif
    /** Backward data propagation */
    mkl_dnn_backward_data = 2,
    /** Backward weights propagation */
    mkl_dnn_backward_weights = 3,
    /** Backward bias propagation */
    mkl_dnn_backward_bias = 4,
} mkl_dnn_prop_kind_t;

/** Algorithm kind */
typedef enum {
    /** Direct convolution */
    mkl_dnn_convolution_direct = 1,
    /** Max pooling */
    mkl_dnn_pooling_max = 101,
    /** Local response normalization across multiple channels */
    mkl_dnn_lrn_across_channels = 201,
    /** LRN within a single channel */
    mkl_dnn_lrn_within_channel = 202,
} mkl_dnn_alg_kind_t;

/** @} */

/** @addtogroup c_api_types_memory Memory description
 *  @{ */

/** Max number of dimensions a tensor can have. Note that this only restricts
 * the amount of space used for tensor description. Individual computational
 * primitives may support only tensors of certain dimensions. */
#define TENSOR_MAX_DIMS 12U

/** A type describing tensor dimensions. */
typedef uint32_t mkl_dnn_dims_t[TENSOR_MAX_DIMS];
/** A type describing offsets within a tensor. */
typedef int32_t mkl_dnn_nd_offset_t[TENSOR_MAX_DIMS];

/** A tensor descriptor. The description is not tied to any memory format, but
 * allows describing tensor dimensions as belonging to minibatch,
 * channel/feature map, and spatial kind. MKL-DNN uses this type when a
 * mathematical description of data is required. */
typedef struct {
    /** Number of minibatch dimensions */
    uint32_t ndims_batch;
    /** Number of channel dimensions */
    uint32_t ndims_channels;
    /** Number of spatial dimensions */
    uint32_t ndims_spatial;
    /** Tensor dimensions in the following order: minibatch, channel, spatial.
     * For example: {N, C, H, W}. */
    mkl_dnn_dims_t dims;
} mkl_dnn_tensor_desc_t;

/** Generic blocked data layout description for most of the memory formats. */
typedef struct {
    /** Block size for each of the dimensions */
    mkl_dnn_dims_t block_dims;
    /** strides[0]: stride between the first elements of adjacent blocks;
     * strides[1]: strides between elements in the same block */
    mkl_dnn_dims_t strides[2];
    /** Size of the data including padding in each dimension */
    mkl_dnn_dims_t padding_dims;
    /** Per-dimension offset from the padding to actual data; the top-level
     * tensor with offsets applied must lie within the padding area */
    mkl_dnn_dims_t offset_padding_to_data;
    /** Offset from memory origin to the current block; non-zero only when
     * describing a memory sub-block */
    size_t offset_padding;
} mkl_dnn_blocking_desc_t;

/** Memory descriptor. The description is based on a tensor description to plus
 * information about elements precision and memory layout. Additionally,
 * contains format-specific data layout descriptions. */
typedef struct {
    /** A descriptor of the underlying tensor */
    mkl_dnn_tensor_desc_t tensor_desc;
    /** Data type of the tensor elements */
    mkl_dnn_precision_t precision;
    /** Memory format */
    mkl_dnn_memory_format_t format;
    union {
        /** Data layout description for memory formats that use blocking */
        mkl_dnn_blocking_desc_t blocking;
        /* ... other descriptions possible */
    } layout_desc;
} mkl_dnn_memory_desc_t;

/** @} */

/** @addtogroup c_api_types_op_descs Operations descriptors
 *  @{*/

/** A descriptor of a convolution operation. */
typedef struct {
    /** Propagation kind; valid values are: #mkl_dnn_forward,
     * #mkl_dnn_backward_data, #mkl_dnn_backward_weights,
     * #mkl_dnn_backward_bias */
    mkl_dnn_prop_kind_t prop_kind;
    /** Convolution algorithm kind; valid values are:
     * #mkl_dnn_convolution_direct */
    mkl_dnn_alg_kind_t alg_kind;
    /** Source memory descriptor */
    mkl_dnn_memory_desc_t src_desc;
    /** Weights memory descriptor */
    mkl_dnn_memory_desc_t weights_desc;
    /** (Optional) bias memory descriptor */
    mkl_dnn_memory_desc_t bias_desc;
    /** Destination memory descriptor */
    mkl_dnn_memory_desc_t dst_desc;
    /** Convolution strides in each spatial dimension */
    mkl_dnn_dims_t strides;
    /** Padding in each spatial dimension (only symmetric padding is currently
     * supported) */
    mkl_dnn_nd_offset_t padding;
    /** What kind of padding to use */
    mkl_dnn_padding_kind_t padding_kind;
} mkl_dnn_convolution_desc_t;

/** A descriptor of a pooling operation */
typedef struct {
    /** Propagation kind; valid values are: #mkl_dnn_forward,
     * #mkl_dnn_backward_data */
    mkl_dnn_prop_kind_t prop_kind;
    /** Pooling algorithm kind; valid values are:
     * #mkl_dnn_pooling_max, #mkl_dnn_pooling_min, #mkl_dnn_pooling_avg */
    mkl_dnn_alg_kind_t alg_kind;
    /** Source memory descriptor */
    mkl_dnn_memory_desc_t src_desc;
    /** Destination memory descriptor */
    mkl_dnn_memory_desc_t dst_desc;
    /** Pooling kernel strides for spatial dimensions*/
    mkl_dnn_dims_t strides;
    /** Pooling kenrel spatial dimensions */
    mkl_dnn_dims_t kernel;
    /** Padding in each spatial dimension (only symmetric padding is currently
     * supported) */
    mkl_dnn_nd_offset_t padding;
    /** What kind of padding to use */
    mkl_dnn_padding_kind_t padding_kind;
} mkl_dnn_pooling_desc_t;

/** A descriptor of a rectifier linear unit (ReLU) operation */
typedef struct {
    /** Propagation kind; valid values are: #mkl_dnn_forward,
     * #mkl_dnn_backward_data */
    mkl_dnn_prop_kind_t prop_kind;
    /** Scaling factor for negative values; stored as double-precision but
     * interpreted in data type-specific way in each of the implementations */
    double negative_slope;
    /** Source memory descriptor */
    mkl_dnn_memory_desc_t src_desc;
    /** Destination memory descriptor */
    mkl_dnn_memory_desc_t dst_desc;
} mkl_dnn_relu_desc_t;

/** A descriptor of a Local Response Normalization (LRN) operation */
typedef struct {
    /** Propagation kind; valid values are: #mkl_dnn_forward,
     * #mkl_dnn_backward_data */
    mkl_dnn_prop_kind_t prop_kind;
    /** LRN algorithm: #mkl_dnn_lrn_within_channel,
     * #mkl_dnn_lrn_across_channels */
    mkl_dnn_alg_kind_t alg_kind;
    /** Destination memory descriptor */
    mkl_dnn_memory_desc_t src_desc;
    /** Destination memory descriptor */
    mkl_dnn_memory_desc_t dst_desc;
    /** LRN alpha parameter */
    double alpha;
    /** LRN beta parameter */
    double beta;
    /** The number of channels to sum over (for cross channel LRN) or the side
     * length of the square region to sum over (for within channel LRN) */
    uint32_t local_size;
} mkl_dnn_lrn_desc_t;

/** A descriptor of a inner product operation */
typedef struct {
    /** Propagation kind; valid values are: #mkl_dnn_forward,
     * #mkl_dnn_backward_data, #mkl_dnn_backward_weights */
    mkl_dnn_prop_kind_t prop_kind;
    /** Source memory descriptor */
    mkl_dnn_memory_desc_t src_desc;
    /** Weights memory descriptor */
    mkl_dnn_memory_desc_t weights_desc;
    /** Bias memory descriptor */
    mkl_dnn_memory_desc_t bias_desc;
    /** Destination memory descriptor */
    mkl_dnn_memory_desc_t dst_desc;
} mkl_dnn_inner_product_desc_t;

/** @} */

/** @addtogroup c_api_engine_types Engine
 * @{ */

/** @brief Engine kinds. */
typedef enum {
    /** An unspecified engine */
    mkl_dnn_any_engine,
    /** CPU engine */
    mkl_dnn_cpu,
    /** CPU engine in a lazy execution mode */
    mkl_dnn_cpu_lazy,
} mkl_dnn_engine_kind_t;

/** @struct mkl_dnn_engine
 * @brief An opaque structure describing an engine. */
struct mkl_dnn_engine;
/** @brief An engine handle. */
typedef struct mkl_dnn_engine *mkl_dnn_engine_t;
/** @brief A constant engine handle. */
typedef const struct mkl_dnn_engine *const_mkl_dnn_engine_t;

/** @} */

/** @addtogroup c_api_primitive_descs Primitive descriptors
 * @{ */

/** A primitive kind. Used to implement a way to extend the library with new
 * primitives without changing the ABI. */
typedef enum {
    /** Undefined primitive (XXX: why do we have it?) */
    mkl_dnn_undefined_primitive,
    /** A memory primitive */
    mkl_dnn_memory,
    /** A reorder primitive */
    mkl_dnn_reorder,
    /** A convolution primitive */
    mkl_dnn_convolution,
    /** A pooling primitive */
    mkl_dnn_pooling,
    /** A ReLU primitive */
    mkl_dnn_relu,
    /** An LRN primitive */
    mkl_dnn_lrn,
    /** An inner product primitive */
    mkl_dnn_inner_product,
} mkl_dnn_primitive_kind_t;

/** Basic primitive descriptor. It is the first member of all
 * primitive-specific descriptor which make it possible to use its data to
 * correctly interpret a #mkl_dnn_primitive_desc_t pointer. */
typedef struct {
    /** Kind of the primitive being described */
    mkl_dnn_primitive_kind_t primitive_kind; // TODO: rename to just kind
    /** Engine used by the primitive */
    const_mkl_dnn_engine_t engine;
    /** Pointer to a library-internal structure */
    const void *implementation;
} mkl_dnn_primitive_base_desc_t;

/** Memory primitive descriptor */
typedef struct {
    /** Basic primitive descriptor */
    mkl_dnn_primitive_base_desc_t base;
    /** Underlying memory descriptor */
    mkl_dnn_memory_desc_t memory_desc;
} mkl_dnn_memory_primitive_desc_t;

/** Reorder primitive descriptor */
typedef struct {
    /** Basic primitive descriptor */
    mkl_dnn_primitive_base_desc_t base;
    /** Input memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t input;
    /** Ouput memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t output;
} mkl_dnn_reorder_primitive_desc_t;

/** Convolution primitive descriptor */
typedef struct {
    /** Basic primitive descriptor */
    mkl_dnn_primitive_base_desc_t base;
    /** Underlying convolution operation descriptor */
    mkl_dnn_convolution_desc_t convolution_desc;
    /** Source memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    /** Weights memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t weights_primitive_desc;
    /** Bias memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t bias_primitive_desc;
    /** Destination memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_convolution_primitive_desc_t;

typedef struct {
    /** Basic primitive descriptor */
    mkl_dnn_primitive_base_desc_t base;
    /** Underlying pooling operation descriptor */
    mkl_dnn_pooling_desc_t pooling_desc;
    /** Source memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    /** Indices memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t indices_primitive_desc;
    /** Destination memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_pooling_primitive_desc_t;

typedef struct {
    /** Basic primitive descriptor */
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_relu_desc_t relu_desc;
    /** Source memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    /** Destination memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_relu_primitive_desc_t;

typedef struct {
    /** Basic primitive descriptor */
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_lrn_desc_t lrn_desc;
    /** Source memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    /** Scratch/workspace memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t scratch_primitive_desc;
    /** Destination memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_lrn_primitive_desc_t;

typedef struct {
    /** Basic primitive descriptor */
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_inner_product_desc_t inner_product_desc;
    /** Source memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    /** Weights memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t weights_primitive_desc;
    /** Bias memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t bias_primitive_desc;
    /** Destination memory primitive descriptor */
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_inner_product_primitive_desc_t;

/** A pointer to any of the primitive descriptors. */
typedef void *mkl_dnn_primitive_desc_t;
/** A pointer to any of the primitive descriptors (constant variant) */
typedef const void *const_mkl_dnn_primitive_desc_t;

/** @} */

/** @addtogroup c_api_types_primitive Primitive
 * @{ */

/** @struct mkl_dnn_primitive
 * An opaque structure describing a primitive. */
struct mkl_dnn_primitive;
/** A primitive handle */
typedef struct mkl_dnn_primitive *mkl_dnn_primitive_t;
/** A constant primitive handle */
typedef const struct mkl_dnn_primitive *const_mkl_dnn_primitive_t;

/** A wrapper structure to address a specific output of a primitive. */
typedef struct {
    /** Primitive to address */
    const_mkl_dnn_primitive_t primitive;
    /** Desired output index */
    size_t output_index;
} mkl_dnn_primitive_at_t;

/** @} */

/** @addtogroup c_api_types_stream Execution stream
 * @{ */

/** @struct mkl_dnn_stream
 * An opaque structure describing an execution stream. */
struct mkl_dnn_stream;
/** An execution stream handle */
typedef struct mkl_dnn_stream *mkl_dnn_stream_t;
/** A constant execution stream handle */
typedef const struct mkl_dnn_stream *const_mkl_dnn_stream_t;

/** @} */
/** @} */
/** @} */

#ifdef __cplusplus
}
#endif


#endif
