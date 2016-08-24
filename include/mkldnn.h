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

#ifndef MKLDNN_H
#define MKLDNN_H

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// All symbols shall be internal unless marked as MKLDNN_API

#if defined _WIN32 || defined __CYGWIN__
#   define MKLDNN_HELPER_DLL_IMPORT __declspec(dllimport)
#   define MKLDNN_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#   if __GNUC__ >= 4
#       define MKLDNN_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
#       define MKLDNN_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
#   else
#       define MKLDNN_HELPER_DLL_IMPORT
#       define MKLDNN_HELPER_DLL_EXPORT
#   endif
#endif

#ifdef MKLDNN_DLL
#   ifdef MKLDNN_DLL_EXPORTS
#       define MKLDNN_API MKLDNN_HELPER_DLL_EXPORT
#   else
#       define MKLDNN_API MKLDNN_HELPER_DLL_IMPORT
#   endif
#else
#   define MKLDNN_API
#endif // MKLDNN_DLL

#include "mkldnn_types.h"
#endif

/** @addtogroup c_api C API
 *  @{ */

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup c_api_primitive Primitive operations
 * @{ */

/* XXX: is this even usable by user? */
/** Creates a @p primitive using a @p primitive_desc descriptor and arrays of @p
 * inputs and @p outputs. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_create(mkldnn_primitive_t *primitive,
        const_mkldnn_primitive_desc_t primitive_desc,
        const mkldnn_primitive_at_t *inputs,
        const_mkldnn_primitive_t *outputs);

/** Retrieves the @p primitive_desc descriptor of a @p primitive. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_get_primitive_desc(
        const_mkldnn_primitive_t primitive,
        mkldnn_primitive_desc_t *primitive_desc);

/** For a @p primitive, returns @p input at the @p index position. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_get_input_at(
        const_mkldnn_primitive_t primitive, size_t index,
        mkldnn_primitive_at_t *input);

/** For a @p primitive, returns @p output at the @p index position. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_get_output(
        const_mkldnn_primitive_t primitive, size_t index,
        const_mkldnn_primitive_t *output);

/** Deletes a @p primitive. Also deallocates internally allocated memory if the
 * primitive represents memory previously created by
 * ::mkldnn_memory_create (@p &primitive, ..., @c NULL). */
mkldnn_status_t MKLDNN_API mkldnn_primitive_destroy(mkldnn_primitive_t primitive);

/** Creates an #mkldnn_primitive_at_t structure from a @p primitive and @p
 * output_index. This function only fills in the data structure
 * and does not check whether parameters are correct. The actual error checking
 * is done when the resulting #mkldnn_primitive_at structure is passed to a
 * primitive creation function. */
mkldnn_primitive_at_t MKLDNN_API mkldnn_primitive_at(const_mkldnn_primitive_t primitive,
        size_t output_index);

/** @addtogroup c_api_memory Memory
 * A primitive to describe data.
 * @{ */

/** Initializes a @p tensor_desc using @p ndims and array @p dims. */
mkldnn_status_t MKLDNN_API mkldnn_tensor_desc_init(mkldnn_tensor_desc_t *tensor_desc,
        uint32_t ndims, const mkldnn_dims_t dims);

/** Initializes a @p memory_desc memory descriptor using @p tensor, data @p
 * precision, and data @p format. @p format can only be #mkldnn_any, which
 * means that specific data layouts are not permitted. */
mkldnn_status_t MKLDNN_API mkldnn_memory_desc_init(mkldnn_memory_desc_t *memory_desc,
        const mkldnn_tensor_desc_t *tensor, mkldnn_precision_t precision,
        mkldnn_memory_format_t format);

/** Returns the number of elements in @p memory_desc memory descriptor:
 *  - If @p with_padding is zero, the number returned does not
 *    include the padding.
 *  - If @p with_padding is non-zero, the number returned
 *    includes the padding.
 * 
 * If @p memory_desc describes a @c NULL tensor, format is not fully
 * specified (for example: @c memory_desc->format == @c any), or
 * @p memory_desc is inconsistent, the function returns zero. */
size_t MKLDNN_API mkldnn_memory_desc_get_number_of_elements(
        const mkldnn_memory_desc_t *memory_desc, int with_padding);

/** Returns the size (in bytes) that is required for the memory described by @p
 * memory_desc. If @p memory_desc describes a @c NULL tensor, format is not fully
 * specified (for example: @c memory_desc->format == @c any), or @p memory_desc is
 * inconsistent, the function returns zero. */
size_t MKLDNN_API mkldnn_memory_desc_get_size(const mkldnn_memory_desc_t *memory_desc);

/** Initializes a @p memory_primitive_desc memory primitive descriptor using @p
 * memory_desc and @p engine. @p memory_desc cannot be uncertain,
 * that is, initialized with #mkldnn_any. */
mkldnn_status_t MKLDNN_API mkldnn_memory_primitive_desc_init(
        mkldnn_memory_primitive_desc_t *memory_primitive_desc,
        const mkldnn_memory_desc_t *memory_desc,
        const_mkldnn_engine_t engine);

/** Compares two descriptors of memory primitives.
 * @return 1 if the descriptors are the same.
 * @return 0 if the descriptors are different.
 *
 * Use this function to identify whether a reorder
 * is required for the memory primitives. */
int MKLDNN_API mkldnn_memory_primitive_desc_equal(
        const mkldnn_memory_primitive_desc_t *lhs,
        const mkldnn_memory_primitive_desc_t *rhs);

/* XXX: consider mkldnn_memory_create_by_ptr (user set ptr or NULL) and
 * mkldnn_memory_create_by_memory (required for split/concat). */

/** Creates a @p memory primitive with a @p memory_primitive_desc and @p data_ptr.
 * If @p data_ptr is @c NULL, @p memory is initialized with internally allocated
 * memory. mkldnn_primitive_destroy() function takes care of the allocated memory. */
mkldnn_status_t MKLDNN_API mkldnn_memory_create(mkldnn_primitive_t *memory,
        const mkldnn_memory_primitive_desc_t *memory_primitive_desc,
        void *data_ptr);

/** Retrieves the @p memory_primitive_desc descriptor associated with a @p memory
 * primitive. */
mkldnn_status_t MKLDNN_API mkldnn_memory_get_primitive_desc(
        const_mkldnn_primitive_t memory,
        mkldnn_memory_primitive_desc_t *memory_primitive_desc);

/** For a @p memory primitive, returns the data @p handle. For the CPU engine,
 * the data handle is a pointer to the actual data. */
mkldnn_status_t MKLDNN_API mkldnn_memory_get_data_handle(
        const_mkldnn_primitive_t memory, void **handle);

/** @} */

/** @addtogroup c_api_reorder Reorder
 * A primitive to copy data between memory formats.
 * @{ */

/** Initializes a @p reorder_primitive_desc using descriptors of @p input and
 * @p output memory primitives. */
mkldnn_status_t MKLDNN_API mkldnn_reorder_primitive_desc_init(
        mkldnn_reorder_primitive_desc_t *reorder_primitive_desc,
        const mkldnn_memory_primitive_desc_t *input,
        const mkldnn_memory_primitive_desc_t *output);

/** Creates a @p reorder primitive using a @p reorder_primitive_desc descriptor,
 * input parameter @p input of type #mkldnn_primitive_at_t, and output primitive
 * @p output. */
mkldnn_status_t MKLDNN_API mkldnn_reorder_create(mkldnn_primitive_t *reorder,
        const mkldnn_reorder_primitive_desc_t *reorder_primitive_desc,
        const mkldnn_primitive_at_t input, const_mkldnn_primitive_t output);

/** Retrieves the @p reorder_primitive_desc descriptor associated with a @p reorder
 * primitive. */
mkldnn_status_t MKLDNN_API mkldnn_reorder_get_primitive_desc(
        const_mkldnn_primitive_t reorder,
        mkldnn_reorder_primitive_desc_t *reorder_primitive_desc);

/** @} */

/** @addtogroup c_api_convolution Convolution
 * A primitive to compute convolution using different algorithms.
 * @{ */

/** Initializes a @p convolution_desc descriptor using @p prop_kind, @p alg_kind,
 * memory descriptors, @p strides, @p padding, and @p padding_kind.
 * If memory descriptors are initialized with #mkldnn_any value of
 * @p format_kind, mkldnn_convolution_primitive_desc_init() chooses the best possible
 * convolution in terms of performance. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_desc_init(
        mkldnn_convolution_desc_t *convolution_desc,
        mkldnn_prop_kind_t prop_kind, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc,
        const mkldnn_dims_t strides, const mkldnn_nd_offset_t padding,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a @p convolution_primitive_desc with @p convolution_desc and @p
 * engine. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_primitive_desc_init(
        mkldnn_convolution_primitive_desc_t *convolution_primitive_desc,
        const mkldnn_convolution_desc_t *convolution_desc,
        const_mkldnn_engine_t engine);

/** Creates a @p convolution primitive using a @p convolution_primitive_desc
 * descriptor, input parameters @p src, @p weights, and @p bias of type
 * #mkldnn_primitive_at_t, and output primitive @p dst. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_create(mkldnn_primitive_t *convolution,
        const mkldnn_convolution_primitive_desc_t *convolution_primitive_desc,
        const mkldnn_primitive_at_t src, const mkldnn_primitive_at_t weights,
        const mkldnn_primitive_at_t bias, const_mkldnn_primitive_t dst);

/** Retrieves the @p convolution_primitive_desc descriptor associated with a @p
 * convolution primitive. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_get_primitive_desc(
        const_mkldnn_primitive_t convolution,
        mkldnn_convolution_primitive_desc_t *convolution_primitive_desc);

/* TODO: add convolution_forward_primitive_desc_init for given
 * mkldnn_memory_primitive_desc_t input, weights, ... so that user has more
 * flexibility. */

/** @} */

/** @addtogroup c_api_relu ReLU
 * A primitive to compute a parametric rectifier linear unit (ReLU).
 * @{ */

/** Initializes a @p relu_desc using @p negative_slope and memory descriptors
 * @p src_desc and @p dst_desc. */
mkldnn_status_t MKLDNN_API mkldnn_relu_desc_init(mkldnn_relu_desc_t *relu_desc,
        mkldnn_prop_kind_t prop_kind, double negative_slope,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *dst_desc);

/** Initializes a @p relu_primitive_desc using a @p relu_desc and @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_relu_primitive_desc_init(
        mkldnn_relu_primitive_desc_t *relu_primitive_desc,
        const mkldnn_relu_desc_t *relu_desc, const_mkldnn_engine_t engine);

/** Creates a @p relu primitive using a @p relu_primitive_desc descriptor, input
 * @p src, and output @p dst. */
mkldnn_status_t MKLDNN_API mkldnn_relu_create(mkldnn_primitive_t *relu,
        const mkldnn_relu_primitive_desc_t *relu_primitive_desc,
        const mkldnn_primitive_at_t src, const_mkldnn_primitive_t dst);

/** Retrieves the @p reorder_primitive_desc descriptor associated with a @p reorder
 * primitive. */
mkldnn_status_t MKLDNN_API mkldnn_relu_get_primitive_desc(
        const_mkldnn_primitive_t relu,
        mkldnn_relu_primitive_desc_t *relu_primitive_desc);

/** @} */

/** @addtogroup c_api_pooling Pooling
 * A primitive to perform max, min, or average pooling.
 * @{ */

/** Initializes a @p pooling_desc using @p prop_kind, @p alg_kind, memory
 * descriptors, @p strides, @p padding, and @p padding_kind.
 * If memory descriptors are initialized with #mkldnn_any value of
 * @p format_kind, mkldnn_pooling_primitive_desc_init() chooses the best possible
 * pooling in terms of performance. */
mkldnn_status_t MKLDNN_API mkldnn_pooling_desc_init(
        mkldnn_pooling_desc_t *pooling_desc,
        mkldnn_prop_kind_t prop_kind, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *dst_desc,
        const mkldnn_dims_t strides, const mkldnn_dims_t kernel,
        const mkldnn_nd_offset_t padding,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a @p pooling_primitive_desc using @p pooling_desc and @p engine.
 */
mkldnn_status_t MKLDNN_API mkldnn_pooling_primitive_desc_init(
        mkldnn_pooling_primitive_desc_t *pooling_primitive_desc,
        const mkldnn_pooling_desc_t *pooling_desc,
        const_mkldnn_engine_t engine);

/** Creates a @p pooling primitive using @p pooling_primitive_desc descriptor,
 * input parameters @p src and @p indices of type #mkldnn_primitive_at_t, and output
 * primitive @p dst. */
mkldnn_status_t MKLDNN_API mkldnn_pooling_create(mkldnn_primitive_t *pooling,
        const mkldnn_pooling_primitive_desc_t *pooling_primitive_desc,
        const mkldnn_primitive_at_t src, const mkldnn_primitive_at_t indices,
        const_mkldnn_primitive_t dst);

/** Retrieves the @p pooling_primitive_desc descriptor associated with a @p pooling
 * primitive. */
mkldnn_status_t MKLDNN_API mkldnn_pooling_get_primitive_desc(
        const_mkldnn_primitive_t pooling,
        mkldnn_pooling_primitive_desc_t *pooling_primitive_desc);

/** @} */

/** @addtogroup c_api_lrn LRN
 * A primitive to perform local response normalization (LRN) across or within
 * channels.
 * @{ */

/** Initializes an @p lrn_desc using @p prop_kind, @p alg_kind, memory
 * descriptors, @p strides, @p padding, and @p padding_kind.
 * If memory descriptors are initialized with #mkldnn_any value of
 * @p format_kind, mkldnn_lrn_primitive_desc_init() chooses the best possible
 * implementation in terms of performance. */
mkldnn_status_t MKLDNN_API mkldnn_lrn_desc_init(mkldnn_lrn_desc_t *lrn_desc,
        mkldnn_prop_kind_t prop_kind, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *dst_desc, double alpha, double beta,
        uint32_t local_size);

/** Initializes an @p lrn_primitive_desc using @p lrn_desc and @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_lrn_primitive_desc_init(
        mkldnn_lrn_primitive_desc_t *lrn_primitive_desc,
        const mkldnn_lrn_desc_t *lrn_desc, const_mkldnn_engine_t engine);

/** Creates an @p lrn primitive using an @p lrn_primitive_desc descriptor, input
 * parameter @p src of type #mkldnn_primitive_at_t, and output primitive @p dst. */
mkldnn_status_t MKLDNN_API mkldnn_lrn_create(mkldnn_primitive_t *lrn,
        const mkldnn_lrn_primitive_desc_t *lrn_primitive_desc,
        const mkldnn_primitive_at_t src, const mkldnn_primitive_at_t scratch,
        const_mkldnn_primitive_t dst);

/** Retrieves the @p lrn_primitive_desc descriptor associated with an @p lrn primitive. */
mkldnn_status_t MKLDNN_API mkldnn_lrn_get_primitive_desc(const_mkldnn_primitive_t lrn,
        mkldnn_lrn_primitive_desc_t *lrn_primitive_desc);

/** @} */

/** @addtogroup c_api_inner_product Inner product
 * A primitive to compute an inner product. 
 * Inner product layer is also known as fully connected layer.
 * @{ */

/** Initializes an @p inner_product_desc using @p prop_kind and memory
 * descriptors. If memory descriptors are initialized with #mkldnn_any value of
 * @p format_kind, mkldnn_inner_product_primitive_desc_init() chooses the best possible inner
 * product in terms of performance. */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_desc_init(
        mkldnn_inner_product_desc_t *inner_product_desc,
        mkldnn_prop_kind_t prop_kind, const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc);

/** Initializes an @p inner_product_primitive_desc with @p inner_product_desc and
 * @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_primitive_desc_init(
        mkldnn_inner_product_primitive_desc_t *inner_product_primitive_desc,
        const mkldnn_inner_product_desc_t *inner_product_desc,
        const_mkldnn_engine_t engine);

/** Creates an @p inner_product primitive using a @p inner_product_primitive_desc
 * descriptor, input parameters @p src, @p weights, and @p bias of type #mkldnn_primitive_at_t,
 * and output primitive @p dst. */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_create(
        mkldnn_primitive_t *inner_product,
        const mkldnn_inner_product_primitive_desc_t *inner_product_primitive_desc,
        const mkldnn_primitive_at_t src, const mkldnn_primitive_at_t weights,
        const mkldnn_primitive_at_t bias, const_mkldnn_primitive_t dst);

/** Retrieves the @p inner_product_primitive_desc descriptor associated with an @p
 * inner_product primitive. */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_get_primitive_desc(
        const_mkldnn_primitive_t inner_product,
        mkldnn_inner_product_primitive_desc_t *inner_product_primitive_desc);

/** @} */

/** @addtogroup c_api_convolution_relu Convolution followed by ReLU
 * A primitive to compute convolution followed by relu using different
 * algorithms.
 * @{ */

/** Initializes a @p convolution_relu_desc descriptor using @p convolution_desc
 * descriptor and @p negative slope */
mkldnn_status_t MKLDNN_API mkldnn_convolution_relu_desc_init(
        mkldnn_convolution_relu_desc_t *convolution_relu_desc,
        const mkldnn_convolution_desc_t *convolution_desc,
        double negative_slope);

/** Initializes a @p convolution_primitive_desc with @p convolution_desc and @p
 * engine. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_relu_primitive_desc_init(
        mkldnn_convolution_relu_primitive_desc_t *conv_relu_primitive_desc,
        const mkldnn_convolution_relu_desc_t *convolution_relu_desc,
        const_mkldnn_engine_t engine);

/** Creates a @p convolution primitive using a @p convolution_primitive_desc
 * descriptor, input primitive_at-s @p src, @p weights, @p bias,
 * and output primitive @p dst. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_relu_create(
        mkldnn_primitive_t *convolution_relu,
        const mkldnn_convolution_relu_primitive_desc_t *conv_relu_primitive_desc,
        const mkldnn_primitive_at_t src, const mkldnn_primitive_at_t weights,
        const mkldnn_primitive_at_t bias, const_mkldnn_primitive_t dst);

/** Retrieves the @p convolution_primitive_desc descriptor associated with a @p
 * convolution primitive. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_relu_get_primitive_desc(
        const_mkldnn_primitive_t convolution_relu,
        mkldnn_convolution_relu_primitive_desc_t *conv_relu_primitive_desc);

/** @} */

/** @} */

/** @addtogroup c_api_engine Engine operations
 * @{ */

/** Returns the number of engines of a particular @p kind. */
size_t MKLDNN_API mkldnn_engine_get_count(mkldnn_engine_kind_t kind);

/** Creates an @p engine of particular @p kind and @p index. */
mkldnn_status_t MKLDNN_API mkldnn_engine_create(mkldnn_engine_t *engine,
        mkldnn_engine_kind_t kind, size_t index);

/** Returns the kind of an @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_engine_get_kind(mkldnn_engine_t engine,
        mkldnn_engine_kind_t *kind);

/** Checks whether an @p engine is lazy. */
mkldnn_status_t MKLDNN_API mkldnn_engine_get_is_lazy(mkldnn_engine_t engine,
        int *is_lazy);

/** Destroys an @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_engine_destroy(mkldnn_engine_t engine);

/** @} */

/** @addtogroup c_api_stream Execution stream operations
 * @{ */

/** Creates an execution @p stream. */
mkldnn_status_t MKLDNN_API mkldnn_stream_create(mkldnn_stream_t *stream);

/** Submits @p primitives to an execution @p stream. The number of primitives is @p n.
 * All or none of the primitives can be lazy. In case of an error, returns
 * the offending @p error_primitive if it is not @c NULL. */
mkldnn_status_t MKLDNN_API mkldnn_stream_submit(mkldnn_stream_t stream, size_t n,
        mkldnn_primitive_t primitives[],
        mkldnn_primitive_t *error_primitive);

/** Waits for all primitives in the execution @p stream to finish. Returns
 * immediately if @p block is zero. In case of an error, returns
 * the offending @p error_primitive if it is not @c NULL. */
mkldnn_status_t MKLDNN_API mkldnn_stream_wait(mkldnn_stream_t stream, int block,
        mkldnn_primitive_t *error_primitive);

/** Destroys an execution @p stream. */
mkldnn_status_t MKLDNN_API mkldnn_stream_destroy(mkldnn_stream_t stream);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif
