#ifndef MKL_DNN_TYPES_H
#define MKL_DNN_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/** generic types section */

typedef enum {
    mkl_dnn_success = 0,
    mkl_dnn_invalid = 1,
    mkl_dnn_out_of_memory = 2,
    mkl_dnn_try_again = 3,
    mkl_dnn_invalid_arguments = 4,
    mkl_dnn_not_ready = 5,
    mkl_dnn_unimplemented = 6,
} mkl_dnn_status_t;

typedef enum {
    mkl_dnn_f32 = 1,
} mkl_dnn_precision_t;

typedef enum {
    mkl_dnn_any,
    /* data formats */
    mkl_dnn_n,
    mkl_dnn_nc,
    mkl_dnn_nchw,
    mkl_dnn_nhwc,
    mkl_dnn_nChw8c,
    /* weights formats */
    mkl_dnn_oi,
    mkl_dnn_oihw,
    mkl_dnn_OIhw8i8o,
    /* weights formats w/ groups */
    mkl_dnn_goihw,
    mkl_dnn_gOIhw8i8o,
    /* generic format */
    mkl_dnn_blocked,
} mkl_dnn_memory_format_t;

typedef enum {
    mkl_dnn_padding_zero,
} mkl_dnn_padding_kind_t;

typedef enum {
    mkl_dnn_forward = 1,
    mkl_dnn_backward_data = 2,
    mkl_dnn_backward_weights = 3,
    mkl_dnn_backward_bias = 4,
} mkl_dnn_prop_kind_t;

typedef enum {
    mkl_dnn_convolution_direct = 1,
    mkl_dnn_pooling_max = 101,
    mkl_dnn_lrn_across_channels = 201,
    mkl_dnn_lrn_within_channel = 202,
} mkl_dnn_alg_kind_t;

/** memory descriptor section */

#define TENSOR_MAX_DIMS 12U

typedef uint32_t mkl_dnn_dims_t[TENSOR_MAX_DIMS];
typedef int32_t mkl_dnn_nd_offset_t[TENSOR_MAX_DIMS];

typedef struct {
    uint32_t ndims_batch;
    uint32_t ndims_channels;
    uint32_t ndims_spatial;
    mkl_dnn_dims_t dims; // { N, C1, ..., Cn, Xi, ..., Xm }
} mkl_dnn_tensor_desc_t;

typedef struct {
    size_t offset_padding;
    size_t offset_padding_to_data;
    mkl_dnn_dims_t padding_dims;
    mkl_dnn_dims_t block_dims;
    mkl_dnn_dims_t strides[2];
    /** strides[0] -- block-to-block strides,
     *  strides[1] -- strides within blocks */
} mkl_dnn_blocking_desc_t;

typedef struct {
    mkl_dnn_tensor_desc_t tensor_desc;
    mkl_dnn_precision_t precision;
    mkl_dnn_memory_format_t format;
    union {
        mkl_dnn_blocking_desc_t blocking_desc;
        // other descriptions possible
    };
} mkl_dnn_memory_desc_t;

/** descriptor sections */

typedef struct {
    mkl_dnn_prop_kind_t prop_kind;
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
