#ifndef MKL_DNN_H
#define MKL_DNN_H

#include "mkl_dnn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Initializes a \param tensor_desc by given \param ndims_batch, \param
 *  ndims_channels, \param ndims_spatial, and array \param dims */
status_t tensor_desc_init(tensor_desc_t *tensor_desc, uint32_t ndims_batch,
        uint32_t ndims_channels, uint32_t ndims_spatial, const dims_t dims);

/** Initializes a \param memory_desc memory descriptor by given \param tensor
 *  and \param format. \param format is allowed to be memory_format_any */
status_t memory_desc_init(memory_desc_t *memory_desc,
        const tensor_desc_t *tensor, memory_format_t format);

/** Initializes a \param memory_primtive_desc memory primitive descriptor by
 *  given \param memory_desc and \param engine. Note that \param memory_desc
 *  cannot be uncertain, i.e. initialized with memory_format_any */
status_t memory_primitive_desc_init(
        memory_primitive_desc_t *memory_primitive_desc,
        const memory_desc_t *memory_desc, const_engine_t engine);

/** Compares two memory primitive descriptors. \return 1 if the memory
 *  primitives descriptors are the same, \return 0 otherwise. Use this function
 *  in order to identify whether reorder between two memory primitives is
 *  required */
int memory_primitive_desc_equal(const memory_primitive_desc_t *lhs,
        const memory_primitive_desc_t *rhs);

/** Creates a \param memory by given \param memory_primitive_desc and \param
 *  data_ptr. If \param data_ptr is NULL \param memory would be initialized
 *  with internally allocated memory. primitive_destroy() will takes care on
 *  allocated memory then */
status_t memory_create(primitive_t *memory,
        const memory_primitive_desc_t *memory_primitive_desc,
        const void *data_ptr);

/** For given \param memory primitive fills in coresponding \param
 *  memory_primitive_desc */
status_t memory_get_primitive_desc(const_primitive_t memory,
        memory_primitive_desc_t *memory_primitive_desc);

/** Initializes a \param reorder_primitive_desc by given \param input and \param
 *  output memory primitive descriptors */
status_t reorder_primitive_desc_init(
        reorder_primitive_desc_t *reorder_primitive_desc,
        const memory_primitive_desc_t *input,
        const memory_primitive_desc_t *output);

/** Initializes a \param convolution_desc by given \param prop_kind, \param
 *  alg_kind, memory descriptors, \param strides, \param padding and \param
 *  padding_kind. Please note that memory descriptors may be initialized with
 *  format_kind_any format_kind. In this case convolution_primitive_desc_init()
 *  will choose the best convolution possible in terms of performance */
status_t convolution_desc_init(convolution_desc_t *convolution_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *input_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *output_desc,
        const dims_t strides, const nd_pos_t padding,
        padding_kind_t padding_kind);

/** Initializes a \param convolution_primitve_desc by given \param
 *  convolution_desc and \param engine */
status_t convolution_primitive_desc_init(
        convolution_primitive_desc_t *convolution_primitive_desc,
        const convolution_desc_t *convolution_desc, const_engine_t engine);

/* TODO: add convolution_forward_primitive_desc_init for given
 * memory_primitive_desc_t input, weights, ... so that user has more
 * flexibility */

/** Creates a \param primitive by given \param primitive descriptor and array
 *  of \param inputs and \param outputs */
status_t primitive_create(primitive_t *primitive,
        const_primitive_desc_t primitive_desc, const_primitive_t *inputs,
        const_primitive_t *outputs);

/** Deletes \param primitive. Also deallocates internally allocated memory if
 *  primitive represents memory, previously created via
 *  memory_create(&primitive, ..., NULL) */
status_t primitive_destroy(primitive_t primitive);

/** Returns the number of engines of a particular \param kind */
size_t engine_get_count(engine_kind_t kind);

/** Creates an \param engine of particular \param kind and \param index */
status_t engine_create(engine_t *engine, engine_kind_t kind, size_t index);

/** Destroys an \param engine */
status_t engine_destroy(engine_t engine);

/** Creates an execution \param stream */
status_t stream_create(stream_t *stream);

/** Submits \param n \param primitives to execution \param stream.  All or none
 * of the primitives must be lazy. In case of an error, may return offending
 * \param error_primitive if it is not NULL. */
status_t stream_submit(stream_t stream, size_t n, primitive_t primitives[],
        primitive_t *error_primitive);

/** Waits for all primitives in the execution \param stream to finish. Returns
 * immediately with a status if \param block is 0. In case of error, may return
 * offending \param error_primitive if it is not NULL. */
status_t stream_wait(stream_t stream, int block, primitive_t *error_primitive);

/** Destroys an execution \param stream */
status_t stream_destroy(stream_t stream);

/** Destroys a \param primitive */
status_t primitive_destroy(primitive_t primitive);


#ifdef __cplusplus
}
#endif

#endif
