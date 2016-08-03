#ifndef MKL_DNN_H
#define MKL_DNN_H

#include "mkl_dnn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Initializes a \param tensor_desc using \param ndims_batch, \param
 * ndims_channels, \param ndims_spatial, and array \param dims */
mkl_dnn_status_t mkl_dnn_tensor_desc_init(mkl_dnn_tensor_desc_t *tensor_desc,
        uint32_t ndims_batch, uint32_t ndims_channels, uint32_t ndims_spatial,
        const mkl_dnn_dims_t dims);

/** Initializes a \param memory_desc memory descriptor using \param tensor,
 * data \param precision and \param format. \param format is allowed to be
 * memory_format_any, which means do not specify any specific data layout */
mkl_dnn_status_t mkl_dnn_memory_desc_init(mkl_dnn_memory_desc_t *memory_desc,
        const mkl_dnn_tensor_desc_t *tensor, mkl_dnn_precision_t precision,
        mkl_dnn_memory_format_t format);

/** Initializes a \param memory_primtive_desc memory primitive descriptor using
 * \param memory_desc and \param engine. Note that \param memory_desc cannot be
 * uncertain, i.e. initialized with memory_format_any */
mkl_dnn_status_t mkl_dnn_memory_primitive_desc_init(
        mkl_dnn_memory_primitive_desc_t *memory_primitive_desc,
        const mkl_dnn_memory_desc_t *memory_desc,
        const_mkl_dnn_engine_t engine);

/** Compares two memory primitive descriptors. \return 1 if the memory
 * primitives descriptors are the same, \return 0 otherwise. Use this function
 * in order to identify whether reorder between two memory primitives is
 * required */
int mkl_dnn_memory_primitive_desc_equal(
        const mkl_dnn_memory_primitive_desc_t *lhs,
        const mkl_dnn_memory_primitive_desc_t *rhs);

/* XXX: consider mkl_dnn_memory_create_by_ptr (user set ptr or NULL) and
 * mkl_dnn_memory_create_by_memory (required for split/concat) */

/** Creates a \param memory with a \param memory_primitive_desc and \param
 * data_ptr. If \param data_ptr is NULL \param memory would be initialized with
 * internally allocated memory. primitive_destroy() will takes care on
 * allocated memory then */
mkl_dnn_status_t mkl_dnn_memory_create(mkl_dnn_primitive_t *memory,
        const mkl_dnn_memory_primitive_desc_t *memory_primitive_desc,
        void *data_ptr);

/** Retrieves the \param memory_primitive_desc associated with a \param memory
 * primitive */
mkl_dnn_status_t mkl_dnn_memory_get_primitive_desc(
        const_mkl_dnn_primitive_t memory,
        mkl_dnn_memory_primitive_desc_t *memory_primitive_desc);

/** Initializes a \param reorder_primitive_desc using \param input and \param
 * output memory primitive descriptors */
mkl_dnn_status_t mkl_dnn_reorder_primitive_desc_init(
        mkl_dnn_reorder_primitive_desc_t *reorder_primitive_desc,
        const mkl_dnn_memory_primitive_desc_t *input,
        const mkl_dnn_memory_primitive_desc_t *output);

/** Initializes a \param convolution_desc using \param prop_kind, \param
 * alg_kind, memory descriptors, \param strides, \param padding and \param
 * padding_kind. Please note that memory descriptors may be initialized with
 * format_kind_any format_kind. In this case convolution_primitive_desc_init()
 * will choose the best convolution possible in terms of performance */
mkl_dnn_status_t mkl_dnn_convolution_desc_init(
        mkl_dnn_convolution_desc_t *convolution_desc,
        mkl_dnn_prop_kind_t prop_kind, mkl_dnn_alg_kind_t alg_kind,
        const mkl_dnn_memory_desc_t *src_desc,
        const mkl_dnn_memory_desc_t *weights_desc,
        const mkl_dnn_memory_desc_t *bias_desc,
        const mkl_dnn_memory_desc_t *dst_desc,
        const mkl_dnn_dims_t strides, const mkl_dnn_nd_offset_t padding,
        mkl_dnn_padding_kind_t padding_kind);

/** Initializes a \param pooling_desc using \param prop_kind, \param alg_kind,
 * memory descriptors, \param strides, \param padding and \param padding_kind.
 * Please note that memory descriptors may be initialized with format_kind_any
 * format_kind. In this case pooling_primitive_desc_init() will choose the best
 * pooling possible in terms of performance */
mkl_dnn_status_t mkl_dnn_pooling_desc_init(
        mkl_dnn_pooling_desc_t *pooling_desc,
        mkl_dnn_prop_kind_t prop_kind, mkl_dnn_alg_kind_t alg_kind,
        const mkl_dnn_memory_desc_t *src_desc,
        const mkl_dnn_memory_desc_t *indices_desc,
        const mkl_dnn_memory_desc_t *dst_desc,
        const mkl_dnn_dims_t strides, const mkl_dnn_dims_t kernel,
        const mkl_dnn_nd_offset_t padding,
        mkl_dnn_padding_kind_t padding_kind);

/** Initializes a \param convolution_primitve_desc with \param convolution_desc
 * and \param engine */
mkl_dnn_status_t mkl_dnn_convolution_primitive_desc_init(
        mkl_dnn_convolution_primitive_desc_t *convolution_primitive_desc,
        const mkl_dnn_convolution_desc_t *convolution_desc,
        const_mkl_dnn_engine_t engine);

/* TODO: add convolution_forward_primitive_desc_init for given
 * mkl_dnn_memory_primitive_desc_t input, weights, ... so that user has more
 * flexibility */

/* XXX: think on this: either add "forward" in function name or put all inputs
 * and outputs in-to arrays, otherwise it is unclear how to create bwd filt */

/** Initializes a \param pooling_primitve_desc using \param pooling_desc and
 * \param engine */
mkl_dnn_status_t mkl_dnn_pooling_primitive_desc_init(
        mkl_dnn_pooling_primitive_desc_t *pooling_primitive_desc,
        const mkl_dnn_pooling_desc_t *pooling_desc,
        const_mkl_dnn_engine_t engine);

/** Creates a \param reorder primitive using descriptor \param
 * reorder_primitive_desc, input primitive_ats \param input, and output
 * primitive \param output */
mkl_dnn_status_t mkl_dnn_reorder_create(mkl_dnn_primitive_t *reorder,
        const mkl_dnn_reorder_primitive_desc_t *reorder_primitive_desc,
        const mkl_dnn_primitive_at_t input, mkl_dnn_primitive_t output);

/** Creates a \param convolution primitive using descriptor \param
 * convolution_primitive_desc, input primitive_ats \param src, \param weights,
 * \param bias and output primitive \param dst */
mkl_dnn_status_t mkl_dnn_convolution_create(mkl_dnn_primitive_t *convolution,
        const mkl_dnn_convolution_primitive_desc_t *convolution_primitive_desc,
        const mkl_dnn_primitive_at_t src, const mkl_dnn_primitive_at_t weights,
        const mkl_dnn_primitive_at_t bias, mkl_dnn_primitive_t dst);

/** Creates a \param pooling primitive using descriptor \param
 * pooling_primitive_desc, input primitive_ats \param src, \param indices,
 * and output primitive \param dst */
mkl_dnn_status_t mkl_dnn_pooling_create(mkl_dnn_primitive_t *pooling,
        const mkl_dnn_pooling_primitive_desc_t *pooling_primitive_desc,
        const mkl_dnn_primitive_at_t src, const mkl_dnn_primitive_at_t indices,
        mkl_dnn_primitive_t dst);

/* XXX: is this even usable by user? */
/** Creates a \param primitive using a \param primitive descriptor and array of
 * \param inputs and \param outputs */
mkl_dnn_status_t mkl_dnn_primitive_create(mkl_dnn_primitive_t *primitive,
        const_mkl_dnn_primitive_desc_t primitive_desc,
        const mkl_dnn_primitive_at_t *inputs, mkl_dnn_primitive_t *outputs);

/** Retrieves \param primitive_desc of a \param primitive */
mkl_dnn_status_t mkl_dnn_primitive_get_primitive_desc(
        const_mkl_dnn_primitive_t primitive,
        mkl_dnn_primitive_desc_t *primitive_desc);

/** For a \param primitive returns \param input at \param index position */
mkl_dnn_status_t mkl_dnn_primitive_get_input_at(
        const_mkl_dnn_primitive_t primitive, size_t index,
        const mkl_dnn_primitive_at_t *input);

/** For a \param primitive returns \param output at \param index position */
mkl_dnn_status_t mkl_dnn_primitive_get_output(
        const_mkl_dnn_primitive_t primitive, size_t index,
        mkl_dnn_primitive_t *output);

// XXX: do we need this?
/** For a \param primitive returns constant primitive \param output at \param
 * index position */
mkl_dnn_status_t mkl_dnn_primitive_get_const_output(
        const_mkl_dnn_primitive_t primitive, size_t index,
        const_mkl_dnn_primitive_t *output);

/** Deletes \param primitive. Also deallocates internally allocated memory if
 * primitive represents memory, previously created via
 * memory_create(&primitive, ..., NULL) */
mkl_dnn_status_t mkl_dnn_primitive_destroy(mkl_dnn_primitive_t primitive);

/** Creates a dnn_primitive_at structure out of a \param primitive and a \param
 * output_index. Note that this function only fills in the data structure
 * without checking whether parameters are correct. The actual error checking
 * is done when the resulting dnn_primitive_at structure is passed to a
 * primitive creation function. */
mkl_dnn_primitive_at_t mkl_dnn_primitive_at(const_mkl_dnn_primitive_t primitive,
        size_t output_index);

/** Returns the number of engines of a particular \param kind */
size_t mkl_dnn_engine_get_count(mkl_dnn_engine_kind_t kind);

/** Creates an \param engine of particular \param kind and \param index */
mkl_dnn_status_t mkl_dnn_engine_create(mkl_dnn_engine_t *engine,
        mkl_dnn_engine_kind_t kind, size_t index);

/** Returns the kind of an \param engine */
mkl_dnn_status_t mkl_dnn_engine_get_kind(mkl_dnn_engine_t engine,
        mkl_dnn_engine_kind_t *kind);

/** Returns whether an \param engine is lazy */
mkl_dnn_status_t mkl_dnn_engine_get_is_lazy(mkl_dnn_engine_t engine, int is_lazy);

/** Destroys an \param engine */
mkl_dnn_status_t mkl_dnn_engine_destroy(mkl_dnn_engine_t engine);

/** Creates an execution \param stream */
mkl_dnn_status_t mkl_dnn_stream_create(mkl_dnn_stream_t *stream);

/** Submits \param n \param primitives to execution \param stream.  All or none
 * of the primitives must be lazy. In case of an error, may return offending
 * \param error_primitive if it is not NULL. */
mkl_dnn_status_t mkl_dnn_stream_submit(mkl_dnn_stream_t stream, size_t n,
        mkl_dnn_primitive_t primitives[],
        mkl_dnn_primitive_t *error_primitive);

/** Waits for all primitives in the execution \param stream to finish. Returns
 * immediately with a status if \param block is 0. In case of error, may return
 * offending \param error_primitive if it is not NULL. */
mkl_dnn_status_t mkl_dnn_stream_wait(mkl_dnn_stream_t stream, int block,
        mkl_dnn_primitive_t *error_primitive);

/** Destroys an execution \param stream */
mkl_dnn_status_t mkl_dnn_stream_destroy(mkl_dnn_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif
