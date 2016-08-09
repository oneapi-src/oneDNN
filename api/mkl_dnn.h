#ifndef MKL_DNN_H
#define MKL_DNN_H

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include "mkl_dnn_types.h"
#endif

/** @addtogroup c_api C API
 *  @{ */

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup c_api_primitive Primitive operations
 * @{ */

/* XXX: is this even usable by user? */
/** Creates a @p primitive using a @p primitive descriptor and array of
 * @p inputs and @p outputs. */
mkl_dnn_status_t mkl_dnn_primitive_create(mkl_dnn_primitive_t *primitive,
        const_mkl_dnn_primitive_desc_t primitive_desc,
        const mkl_dnn_primitive_at_t *inputs,
        const_mkl_dnn_primitive_t *outputs);

/** Retrieves @p primitive_desc of a @p primitive. */
mkl_dnn_status_t mkl_dnn_primitive_get_primitive_desc(
        const_mkl_dnn_primitive_t primitive,
        mkl_dnn_primitive_desc_t *primitive_desc);

/** For a @p primitive returns @p input at @p index position. */
mkl_dnn_status_t mkl_dnn_primitive_get_input_at(
        const_mkl_dnn_primitive_t primitive, size_t index,
        const mkl_dnn_primitive_at_t *input);

/** For a @p primitive returns @p output at @p index position. */
mkl_dnn_status_t mkl_dnn_primitive_get_output(
        const_mkl_dnn_primitive_t primitive, size_t index,
        const_mkl_dnn_primitive_t *output);

// XXX: do we need this?
/** For a @p primitive returns constant primitive @p output at @p
 * index position */
mkl_dnn_status_t mkl_dnn_primitive_get_const_output(
        const_mkl_dnn_primitive_t primitive, size_t index,
        const_mkl_dnn_primitive_t *output);

/** Deletes @p primitive. Also deallocates internally allocated memory if
 * primitive represents memory, previously created via
 * memory_create(&primitive, ..., NULL). */
mkl_dnn_status_t mkl_dnn_primitive_destroy(mkl_dnn_primitive_t primitive);

/** Creates a dnn_primitive_at structure out of a @p primitive and a @p
 * output_index. Note that this function only fills in the data structure
 * without checking whether parameters are correct. The actual error checking
 * is done when the resulting dnn_primitive_at structure is passed to a
 * primitive creation function. */
mkl_dnn_primitive_at_t mkl_dnn_primitive_at(const_mkl_dnn_primitive_t primitive,
        size_t output_index);

/** @addtogroup c_api_memory Memory
 * @{ */

/** Initializes a @p tensor_desc using @p ndims_batch, @p
 * ndims_channels, @p ndims_spatial, and array @p dims. */
mkl_dnn_status_t mkl_dnn_tensor_desc_init(mkl_dnn_tensor_desc_t *tensor_desc,
        uint32_t ndims_batch, uint32_t ndims_channels, uint32_t ndims_spatial,
        const mkl_dnn_dims_t dims);

/** Initializes a @p memory_desc memory descriptor using @p tensor,
 * data @p precision and @p format. @p format is allowed to be
 * memory_format_any, which means do not specify any specific data layout. */
mkl_dnn_status_t mkl_dnn_memory_desc_init(mkl_dnn_memory_desc_t *memory_desc,
        const mkl_dnn_tensor_desc_t *tensor, mkl_dnn_precision_t precision,
        mkl_dnn_memory_format_t format);

/** Returns the number of elements in @p memory_desc memory descriptor. If @p
 * with_padding is zero the function returns the number of elements not
 * including the padding, otherwise returns the number of elements including
 * the padding. If @p memory_desc describes null tensor, or format is not fully
 * specified (e.g. memory_desc->format == any), or @p memory_desc is
 * incosistent the function returns 0. */
size_t mkl_dnn_memory_desc_get_number_of_elements(
        const mkl_dnn_memory_desc_t *memory_desc, int with_padding);

/** Returns the size (in bytes) that required for memory described by @p
 * memory_desc. If @p memory_desc describes null tensor, or format is not fully
 * specified (e.g. memory_desc->format == any), or @p memory_desc is
 * incosistent the function returns 0. */
size_t mkl_dnn_memory_desc_get_size(const mkl_dnn_memory_desc_t *memory_desc);

/** Initializes a @p memory_primtive_desc memory primitive descriptor using
 * @p memory_desc and @p engine. Note that @p memory_desc cannot be
 * uncertain, i.e. initialized with memory_format_any. */
mkl_dnn_status_t mkl_dnn_memory_primitive_desc_init(
        mkl_dnn_memory_primitive_desc_t *memory_primitive_desc,
        const mkl_dnn_memory_desc_t *memory_desc,
        const_mkl_dnn_engine_t engine);

/** Compares two memory primitive descriptors. \return 1 if the memory
 * primitives descriptors are the same, \return 0 otherwise. Use this function
 * in order to identify whether reorder between two memory primitives is
 * required. */
int mkl_dnn_memory_primitive_desc_equal(
        const mkl_dnn_memory_primitive_desc_t *lhs,
        const mkl_dnn_memory_primitive_desc_t *rhs);

/* XXX: consider mkl_dnn_memory_create_by_ptr (user set ptr or NULL) and
 * mkl_dnn_memory_create_by_memory (required for split/concat). */

/** Creates a @p memory with a @p memory_primitive_desc and @p
 * data_ptr. If @p data_ptr is NULL @p memory would be initialized with
 * internally allocated memory. primitive_destroy() will takes care on
 * allocated memory then. */
mkl_dnn_status_t mkl_dnn_memory_create(mkl_dnn_primitive_t *memory,
        const mkl_dnn_memory_primitive_desc_t *memory_primitive_desc,
        void *data_ptr);

/** Retrieves the @p memory_primitive_desc associated with a @p memory
 * primitive. */
mkl_dnn_status_t mkl_dnn_memory_get_primitive_desc(
        const_mkl_dnn_primitive_t memory,
        mkl_dnn_memory_primitive_desc_t *memory_primitive_desc);

/** For a primitive @p memory returns data @p handle. For cpu engine
 * data handle is a pointer to the actual data. */
mkl_dnn_status_t mkl_dnn_memory_get_data_handle(
        const_mkl_dnn_primitive_t memory, void **handle);

/* @} */

/** @addtogroup c_api_reorder Reorder
 * @{ */

/** Initializes a @p reorder_primitive_desc using @p input and @p
 * output memory primitive descriptors. */
mkl_dnn_status_t mkl_dnn_reorder_primitive_desc_init(
        mkl_dnn_reorder_primitive_desc_t *reorder_primitive_desc,
        const mkl_dnn_memory_primitive_desc_t *input,
        const mkl_dnn_memory_primitive_desc_t *output);

/** Creates a @p reorder primitive using descriptor @p reorder_primitive_desc,
 * input primitive_ats @p input, and output primitive @p output. */
mkl_dnn_status_t mkl_dnn_reorder_create(mkl_dnn_primitive_t *reorder,
        const mkl_dnn_reorder_primitive_desc_t *reorder_primitive_desc,
        const mkl_dnn_primitive_at_t input, const_mkl_dnn_primitive_t output);

/* @} */

/** @addtogroup c_api_convolution Convolution
 * @{ */

/** Initializes a @p convolution_desc using @p prop_kind, @p
 * alg_kind, memory descriptors, @p strides, @p padding and @p
 * padding_kind. Please note that memory descriptors may be initialized with
 * format_kind_any format_kind. In this case convolution_primitive_desc_init()
 * will choose the best convolution possible in terms of performance. */
mkl_dnn_status_t mkl_dnn_convolution_desc_init(
        mkl_dnn_convolution_desc_t *convolution_desc,
        mkl_dnn_prop_kind_t prop_kind, mkl_dnn_alg_kind_t alg_kind,
        const mkl_dnn_memory_desc_t *src_desc,
        const mkl_dnn_memory_desc_t *weights_desc,
        const mkl_dnn_memory_desc_t *bias_desc,
        const mkl_dnn_memory_desc_t *dst_desc,
        const mkl_dnn_dims_t strides, const mkl_dnn_nd_offset_t padding,
        mkl_dnn_padding_kind_t padding_kind);

/** Initializes a @p convolution_primitve_desc with @p convolution_desc
 * and @p engine. */
mkl_dnn_status_t mkl_dnn_convolution_primitive_desc_init(
        mkl_dnn_convolution_primitive_desc_t *convolution_primitive_desc,
        const mkl_dnn_convolution_desc_t *convolution_desc,
        const_mkl_dnn_engine_t engine);

/** Creates a @p convolution primitive using descriptor @p
 * convolution_primitive_desc, input primitive_ats @p src, @p weights,
 * @p bias and output primitive @p dst. */
mkl_dnn_status_t mkl_dnn_convolution_create(mkl_dnn_primitive_t *convolution,
        const mkl_dnn_convolution_primitive_desc_t *convolution_primitive_desc,
        const mkl_dnn_primitive_at_t src, const mkl_dnn_primitive_at_t weights,
        const mkl_dnn_primitive_at_t bias, const_mkl_dnn_primitive_t dst);

/* TODO: add convolution_forward_primitive_desc_init for given
 * mkl_dnn_memory_primitive_desc_t input, weights, ... so that user has more
 * flexibility. */

/* @} */

/** @addtogroup c_api_relu ReLU
 * @{ */

/** Initialized a @p relu_desc using @p negative_slope, and memory
 * descriptors @p src_desc and @p dst_desc. */
mkl_dnn_status_t mkl_dnn_relu_desc_init(
        mkl_dnn_relu_desc_t *relu_desc, mkl_dnn_prop_kind_t prop_kind,
        double negative_slope, const mkl_dnn_memory_desc_t *src_desc,
        const mkl_dnn_memory_desc_t *dst_desc);

/** Initializes a @p relu_primitive_desc using @p relu_desc and @p
 * engine. */
mkl_dnn_status_t mkl_dnn_relu_primitive_desc_init(
        mkl_dnn_relu_primitive_desc_t *relu_primitive_desc,
        const mkl_dnn_relu_desc_t *relu_desc, const_mkl_dnn_engine_t engine);

/** Creates a @p relu primitive using descrptor @p relu_primitive_desc, input
 * @p src and output @p dst. */
mkl_dnn_status_t mkl_dnn_relu_create(mkl_dnn_primitive_t *relu,
        const mkl_dnn_relu_primitive_desc_t *relu_primitive_desc,
        const mkl_dnn_primitive_at_t src, const_mkl_dnn_primitive_t dst);

/* @} */

/** @addtogroup c_api_pooling Pooling
 * @{ */

/** Initializes a @p pooling_desc using @p prop_kind, @p alg_kind,
 * memory descriptors, @p strides, @p padding and @p padding_kind.
 * Please note that memory descriptors may be initialized with format_kind_any
 * format_kind. In this case pooling_primitive_desc_init() will choose the best
 * pooling possible in terms of performance. */
mkl_dnn_status_t mkl_dnn_pooling_desc_init(
        mkl_dnn_pooling_desc_t *pooling_desc,
        mkl_dnn_prop_kind_t prop_kind, mkl_dnn_alg_kind_t alg_kind,
        const mkl_dnn_memory_desc_t *src_desc,
        const mkl_dnn_memory_desc_t *dst_desc,
        const mkl_dnn_dims_t strides, const mkl_dnn_dims_t kernel,
        const mkl_dnn_nd_offset_t padding,
        mkl_dnn_padding_kind_t padding_kind);

/** Initializes a @p pooling_primitve_desc using @p pooling_desc and
 * @p engine. */
mkl_dnn_status_t mkl_dnn_pooling_primitive_desc_init(
        mkl_dnn_pooling_primitive_desc_t *pooling_primitive_desc,
        const mkl_dnn_pooling_desc_t *pooling_desc,
        const_mkl_dnn_engine_t engine);

/** Creates a @p pooling primitive using descriptor @p
 * pooling_primitive_desc, input primitive_ats @p src, @p indices,
 * and output primitive @p dst. */
mkl_dnn_status_t mkl_dnn_pooling_create(mkl_dnn_primitive_t *pooling,
        const mkl_dnn_pooling_primitive_desc_t *pooling_primitive_desc,
        const mkl_dnn_primitive_at_t src, const mkl_dnn_primitive_at_t indices,
        const_mkl_dnn_primitive_t dst);

/* @} */

/** @addtogroup c_api_lrn LRN operations
 * @{ */

/** Initializes a @p lrn_desc using @p prop_kind, @p alg_kind, memory
 * descriptors, @p strides, @p padding and @p padding_kind.  Please note that
 * memory descriptors may be initialized with format_kind_any format_kind. In
 * this case lrn_primitive_desc_init() will choose the best implementation
 * possible in terms of performance. */
mkl_dnn_status_t mkl_dnn_lrn_desc_init(mkl_dnn_lrn_desc_t *lrn_desc,
    mkl_dnn_prop_kind_t prop_kind, mkl_dnn_alg_kind_t alg_kind,
    const mkl_dnn_memory_desc_t *src_desc,
    const mkl_dnn_memory_desc_t *dst_desc, double alpha, double beta,
    uint32_t local_size);

/** Initializes a @p lrn_primitve_desc using @p lrn_desc and @p engine. */
mkl_dnn_status_t mkl_dnn_lrn_primitive_desc_init(
    mkl_dnn_lrn_primitive_desc_t *lrn_primitive_desc,
    const mkl_dnn_lrn_desc_t *lrn_desc,
    const_mkl_dnn_engine_t engine);

/** Creates a @p lrn primitive using descriptor @p lrn_primitive_desc, input
 * primitive_at @p src and output primitive @p dst. */
mkl_dnn_status_t mkl_dnn_lrn_create(mkl_dnn_primitive_t *lrn,
    const mkl_dnn_lrn_primitive_desc_t *lrn_primitive_desc,
    const mkl_dnn_primitive_at_t src, const mkl_dnn_primitive_at_t scratch,
    const_mkl_dnn_primitive_t dst);

/* @} */

/** @addtogroup c_api_inner_product Inner product
 * Aka fully connected
 * @{ */

/** Initializes a @p inner_product_desc using @p prop_kind and memory
 * descriptors. Please note that memory descriptors may be initialized with
 * format_kind_any format_kind. In this case inner_product_primitive_desc_init()
 * will choose the best inner product possible in terms of performance. */
mkl_dnn_status_t mkl_dnn_inner_product_desc_init(
        mkl_dnn_inner_product_desc_t *inner_product_desc,
        mkl_dnn_prop_kind_t prop_kind,
        const mkl_dnn_memory_desc_t *src_desc,
        const mkl_dnn_memory_desc_t *weights_desc,
        const mkl_dnn_memory_desc_t *dst_desc);

/** Initializes a @p inner_product_primitve_desc with @p
 * inner_product_desc and @p engine. */
mkl_dnn_status_t mkl_dnn_inner_product_primitive_desc_init(
        mkl_dnn_inner_product_primitive_desc_t *inner_product_primitive_desc,
        const mkl_dnn_inner_product_desc_t *inner_product_desc,
        const_mkl_dnn_engine_t engine);

/** Creates a @p inner product primitive using descriptor @p
 * inner_product_primitive_desc, input primitive_ats @p src, @p weights and
 * output primitive @p dst. */
mkl_dnn_status_t mkl_dnn_inner_product_create(mkl_dnn_primitive_t *inner_product,
        const mkl_dnn_inner_product_primitive_desc_t *inner_product_primitive_desc,
        const mkl_dnn_primitive_at_t src, const mkl_dnn_primitive_at_t weights,
        const_mkl_dnn_primitive_t dst);

/* @} */

/* @} */

/** @addtogroup c_api_engine Engine operations
 * @{ */

/** Returns the number of engines of a particular @p kind. */
size_t mkl_dnn_engine_get_count(mkl_dnn_engine_kind_t kind);

/** Creates an @p engine of particular @p kind and @p index. */
mkl_dnn_status_t mkl_dnn_engine_create(mkl_dnn_engine_t *engine,
        mkl_dnn_engine_kind_t kind, size_t index);

/** Returns the kind of an @p engine. */
mkl_dnn_status_t mkl_dnn_engine_get_kind(mkl_dnn_engine_t engine,
        mkl_dnn_engine_kind_t *kind);

/** Returns whether an @p engine is lazy. */
mkl_dnn_status_t mkl_dnn_engine_get_is_lazy(mkl_dnn_engine_t engine, int is_lazy);

/** Destroys an @p engine. */
mkl_dnn_status_t mkl_dnn_engine_destroy(mkl_dnn_engine_t engine);

/* @} */

/** @addtogroup c_api_stream Execution stream operations
 * @{ */

/** Creates an execution @p stream. */
mkl_dnn_status_t mkl_dnn_stream_create(mkl_dnn_stream_t *stream);

/** Submits @p n @p primitives to execution @p stream.  All or none
 * of the primitives must be lazy. In case of an error, may return offending
 * @p error_primitive if it is not NULL. */
mkl_dnn_status_t mkl_dnn_stream_submit(mkl_dnn_stream_t stream, size_t n,
        mkl_dnn_primitive_t primitives[],
        mkl_dnn_primitive_t *error_primitive);

/** Waits for all primitives in the execution @p stream to finish. Returns
 * immediately with a status if @p block is 0. In case of error, may return
 * offending @p error_primitive if it is not NULL. */
mkl_dnn_status_t mkl_dnn_stream_wait(mkl_dnn_stream_t stream, int block,
        mkl_dnn_primitive_t *error_primitive);

/** Destroys an execution @p stream. */
mkl_dnn_status_t mkl_dnn_stream_destroy(mkl_dnn_stream_t stream);

/* @} */

/* @} */

#ifdef __cplusplus
}
#endif

#endif
