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
    /** an operation was successful */
    mkl_dnn_success = 0,
    /** an operation failed due to an out-of-memory condition */
    mkl_dnn_out_of_memory = 1,
    /** an operation failed and should be retried */
    mkl_dnn_try_again = 2,
    /** an operation failed because function arguments were incorrect */
    mkl_dnn_invalid_arguments = 3,
    /** an operation failed because a primitive was not ready for execution */
    mkl_dnn_not_ready = 4,
    /** an operation failed because requested functionality is not implemented */
    mkl_dnn_unimplemented = 5,
} mkl_dnn_status_t;

/* TODO: rename to mkl_dnn_data_type_t */
/** Data type specification */
typedef enum {
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
 * MKL-DNN uses upper-case letters to indicate that data is layed out in blocks
 * for a particular dimension. In this case, a format name contains both upper-
 * and lower-case letter for such a dimension with lower-case letter preceeded
 * by the block size. For example, 'mkl_dnn_nChw8c' describes a format where
 * the outermost dimension is minibatch, followed by channel block number,
 * followed by spatial height and width, and then followed by 8-element channel
 * blocks. Note that channel designations are a convention. Thus both the
 * 'mkl_dnn_nc' and 'mkl_dnn_oo' formats can be used to describe any 2D tensor.
 * */
typedef enum {
    /** unspecified format; use this to let a primitive to select a format
     * automatically */
    mkl_dnn_any,
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
    /** A tensor in a generic format described by stride and blocking values in
     * each dimension. See #mkl_dnn_blocking_desc_t for more information. */
    mkl_dnn_blocked,
} mkl_dnn_memory_format_t;

/** Padding kind -- how to interpret the data in the padding regions */
typedef enum {
    /** assume that the data in padding regions is zero */
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
    /** LRN across multiple channels */
    mkl_dnn_lrn_across_channels = 201,
    /** LRN within a single channel */
    mkl_dnn_lrn_within_channel = 202,
} mkl_dnn_alg_kind_t;

/** @} */

/** @addtogroup c_api_types_memory Memory description
 *  @{*/

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
    /** block size for each of the dimensions */
    mkl_dnn_dims_t block_dims;
    /** strides[0]: stride between the first elements of adjacent blocks;
     * strides[1]: strides between elements in the same block */
    mkl_dnn_dims_t strides[2];
    /** size of the data including padding in each dimension */
    mkl_dnn_dims_t padding_dims;
    /** per-dimension offset from the padding to actual data; the top-level
     * tensor with offsets applied must lie within the padding area */
    mkl_dnn_dims_t offset_padding_to_data;
    /** offset from memory origin to the current block; non-zero only when
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
        mkl_dnn_blocking_desc_t blocking_desc;
        /* ... other descriptions possible */
    };
} mkl_dnn_memory_desc_t;

/* @} */

/** A descriptor of a convolution operation. */
typedef struct {
    /** propagation kind; valid values are: #mkl_dnn_forward,
     * #mkl_dnn_backward_data, #mkl_dnn_backward_weights,
     * #mkl_dnn_backward_bias */
    mkl_dnn_prop_kind_t prop_kind;
    /** convolution algorithm kind; valid values are:
     * #mkl_dnn_convolution_primitive_desc_t */
    mkl_dnn_alg_kind_t alg_kind;
    mkl_dnn_memory_desc_t src_desc;
    mkl_dnn_memory_desc_t weights_desc;
    mkl_dnn_memory_desc_t bias_desc;
    mkl_dnn_memory_desc_t dst_desc;
    mkl_dnn_dims_t strides;
    mkl_dnn_nd_offset_t padding;
    mkl_dnn_padding_kind_t padding_kind;
} mkl_dnn_convolution_desc_t;

typedef struct {
    mkl_dnn_prop_kind_t prop_kind;
    mkl_dnn_alg_kind_t alg_kind;
    mkl_dnn_memory_desc_t src_desc;
    mkl_dnn_memory_desc_t indices_desc;
    mkl_dnn_memory_desc_t dst_desc;
    mkl_dnn_dims_t strides;
    mkl_dnn_dims_t kernel;
    mkl_dnn_nd_offset_t padding;
    mkl_dnn_padding_kind_t padding_kind;
} mkl_dnn_pooling_desc_t;

typedef struct {
    mkl_dnn_prop_kind_t prop_kind;
    double negative_slope; // XXX: real precision obtained from the memory
    mkl_dnn_memory_desc_t src_desc;
    mkl_dnn_memory_desc_t dst_desc;
} mkl_dnn_relu_desc_t;

typedef struct {
    mkl_dnn_prop_kind_t prop_kind;
    mkl_dnn_alg_kind_t alg_kind;
    mkl_dnn_memory_desc_t src_desc;
    mkl_dnn_memory_desc_t scratch_desc;
    mkl_dnn_memory_desc_t dst_desc;
    double alpha;
    double beta;
    uint32_t local_size;
} mkl_dnn_lrn_desc_t;

typedef struct {
    mkl_dnn_prop_kind_t prop_kind;
    mkl_dnn_memory_desc_t src_desc;
    mkl_dnn_memory_desc_t weights_desc;
    mkl_dnn_memory_desc_t dst_desc;
} mkl_dnn_inner_product_desc_t;

/** engine section */

typedef enum {
    mkl_dnn_any_engine,
    mkl_dnn_cpu,
    mkl_dnn_cpu_lazy,
} mkl_dnn_engine_kind_t;

struct mkl_dnn_engine;
typedef struct mkl_dnn_engine *mkl_dnn_engine_t;
typedef const struct mkl_dnn_engine *const_mkl_dnn_engine_t;

/** primitive descriptor section */

typedef enum {
    mkl_dnn_undefined_primitive,
    mkl_dnn_memory,
    mkl_dnn_reorder,
    mkl_dnn_convolution,
    mkl_dnn_pooling,
    mkl_dnn_relu,
    mkl_dnn_lrn,
    mkl_dnn_inner_product,
} mkl_dnn_primitive_kind_t;

typedef struct {
    mkl_dnn_primitive_kind_t primitive_kind; // TODO: rename to just kind
    const_mkl_dnn_engine_t engine;
    const void *implementation;
} mkl_dnn_primitive_base_desc_t;

typedef struct {
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_memory_desc_t memory_desc;
} mkl_dnn_memory_primitive_desc_t;

typedef struct {
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_memory_primitive_desc_t input;
    mkl_dnn_memory_primitive_desc_t output;
} mkl_dnn_reorder_primitive_desc_t;

typedef struct {
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_convolution_desc_t convolution_desc;
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    mkl_dnn_memory_primitive_desc_t weights_primitive_desc;
    mkl_dnn_memory_primitive_desc_t bias_primitive_desc;
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_convolution_primitive_desc_t;

typedef struct {
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_pooling_desc_t pooling_desc;
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    mkl_dnn_memory_primitive_desc_t indices_primitive_desc;
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_pooling_primitive_desc_t;

typedef struct {
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_relu_desc_t relu_desc;
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_relu_primitive_desc_t;

typedef struct {
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_lrn_desc_t lrn_desc;
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    mkl_dnn_memory_primitive_desc_t scratch_primitive_desc;
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_lrn_primitive_desc_t;

typedef struct {
    mkl_dnn_primitive_base_desc_t base;
    mkl_dnn_inner_product_desc_t inner_product_desc;
    mkl_dnn_memory_primitive_desc_t src_primitive_desc;
    mkl_dnn_memory_primitive_desc_t weights_primitive_desc;
    mkl_dnn_memory_primitive_desc_t dst_primitive_desc;
} mkl_dnn_inner_product_primitive_desc_t;

/** primitive section */

typedef void *mkl_dnn_primitive_desc_t;
typedef const void *const_mkl_dnn_primitive_desc_t;

struct mkl_dnn_primitive;
typedef struct mkl_dnn_primitive *mkl_dnn_primitive_t;
typedef const struct mkl_dnn_primitive *const_mkl_dnn_primitive_t;

typedef struct {
    const_mkl_dnn_primitive_t primitive;
    size_t output_index;
} mkl_dnn_primitive_at_t;

/** stream section */

struct mkl_dnn_stream;
typedef struct mkl_dnn_stream *mkl_dnn_stream_t;
typedef const struct mkl_dnn_stream *const_mkl_dnn_stream_t;

#ifdef __cplusplus
}
#endif

#endif
