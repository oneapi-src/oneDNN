#ifndef MKL_DNN_TYPES_H
#define MKL_DNN_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/** generic types section */

typedef enum status {
    success = 0,
    invalid = 1,
    out_of_memory = 2,
    try_again = 3,
    invalid_arguments = 4,
    not_ready = 5, /* XXX: should be same as try_again? */
    unimplemented = 6,
} status_t;

typedef enum prop_kind {
    forward = 1,
    backward_data = 2,
    backward_weights = 3,
    backward_bias = 4,
} prop_kind_t;

typedef enum alg_kind {
    convolution_direct = 1,
} alg_kind_t;

/** memory descriptor section */

#define TENSOR_MAX_DIMS 12U

// FIXME: asap!
typedef uint32_t nd_offset_t[TENSOR_MAX_DIMS];
typedef uint32_t nd_pos_t[TENSOR_MAX_DIMS];
typedef uint32_t dims_t[TENSOR_MAX_DIMS];

typedef struct tensor_desc {
    uint32_t ndims_batch;
    uint32_t ndims_channels;
    uint32_t ndims_spatial;
    dims_t dims; // { N, C1, ..., Cn, Xi, ..., Xm }
} tensor_desc_t;

typedef enum memory_format {
    /* XXX: do we really want to keep precision in format??? */
    memory_format_any = 0,
    memory_format_automatic = memory_format_any,
    memory_format_n_f32,
    memory_format_nchw_f32,
    memory_format_oihw_f32 = memory_format_nchw_f32,
    memory_format_nhwc_f32,
    memory_format_nChw8_f32,
    memory_format_IOhw88_f32,
    memory_format_nChw16_f32,
    memory_format_IOhw1616_f32,
    memory_format_blocked_f32,
} memory_format_t;

typedef enum padding_kind {
    padding_kind_zero = 0,
} padding_kind_t;

typedef struct blocking_desc {
    size_t offset_padding;
    size_t offset_padding_to_data;
    dims_t padding_dims;
    dims_t block_dims;
    dims_t strides[2];
    /** strides[0] -- block-to-block strides,
     *  strides[1] -- strides within blocks */
} blocking_desc_t;

typedef struct memory_desc {
    tensor_desc_t tensor_desc;
    memory_format_t format; // includes information about type size
    union {
        blocking_desc_t blocking_desc;
        // other descriptions possible
    };
} memory_desc_t;

/** descriptor sections */

typedef struct convolution_desc {
    prop_kind_t prop_kind;
    alg_kind_t alg_kind;
    memory_desc_t input_desc;
    memory_desc_t weights_desc;
    memory_desc_t bias_desc;
    memory_desc_t output_desc;
    nd_offset_t padding;
    nd_pos_t strides;
    padding_kind_t padding_kind;
} convolution_desc_t;

/** engine section */

/* XXX: probably engine_kind_any should be (-1), and engine_kind_cpu should
 * be 0, so that engine_kind cpu is the value by default for C++/C... where? */
typedef enum engine_kind {
    engine_kind_any = 0,
    engine_kind_automatic = engine_kind_any,
    engine_kind_cpu = 1,
    engine_kind_cpu_lazy = 2,
    engine_kind_last = 3,
} engine_kind_t;

typedef void *engine_t;
typedef const void *const_engine_t;

/** primitive descriptor section */

typedef enum primitive_kind {
    primitive_kind_undef = 0,
    primitive_kind_memory = 1,
    primitive_kind_reorder = 2,
    primitive_kind_convolution = 3,
} primitive_kind_t;

typedef struct primitive_base_desc {
    primitive_kind_t primitive_kind;
    const_engine_t engine;
    const void *implementation;
} primitive_base_desc_t;

typedef struct memory_primitive_desc {
    primitive_base_desc_t base;
    memory_desc_t memory_desc;
} memory_primitive_desc_t;

typedef struct reorder_primitive_desc {
    primitive_base_desc_t base;
    memory_primitive_desc_t input;
    memory_primitive_desc_t output;
} reorder_primitive_desc_t;

typedef struct convolution_primitive_desc {
    primitive_base_desc_t base;
    convolution_desc_t convolution_desc;
    memory_primitive_desc_t input_primitive_desc;
    memory_primitive_desc_t weights_primitive_desc;
    memory_primitive_desc_t bias_primitive_desc;
    memory_primitive_desc_t output_primitive_desc;
} convolution_primitive_desc_t;

/** primitive section */

typedef void *primitive_desc_t;
typedef const void *const_primitive_desc_t;
typedef void *primitive_t;
typedef const void *const_primitive_t;

static const const_primitive_t primitive_self = ((const_primitive_t)-1);

/** stream section */

typedef void *stream_t;
typedef const void *const_stream_t;

#ifdef __cplusplus
}
#endif

#endif
