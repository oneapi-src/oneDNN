#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "mkl_dnn.h"

#include "engine.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"

mkl_dnn_status_t mkl_dnn_tensor_desc_init(mkl_dnn_tensor_desc_t *tensor_desc,
		uint32_t ndims_batch, uint32_t ndims_channels, uint32_t ndims_spatial,
		const mkl_dnn_dims_t dims) {
    if (mkl_dnn::impl::any_null(tensor_desc, dims))
		return mkl_dnn_invalid_arguments;
    mkl_dnn_tensor_desc_t td = {
        .ndims_batch = ndims_batch,
        .ndims_channels = ndims_channels,
        .ndims_spatial = ndims_spatial};
    const uint32_t ndims = mkl_dnn::impl::types::ndims(td);
    if (ndims > TENSOR_MAX_DIMS)
		return mkl_dnn_invalid_arguments;
    mkl_dnn::impl::array_copy(td.dims, dims, ndims);
    *tensor_desc = td;
    return mkl_dnn_success;
}

mkl_dnn_status_t mkl_dnn_memory_desc_init(mkl_dnn_memory_desc_t *memory_desc,
        const mkl_dnn_tensor_desc_t *tensor, mkl_dnn_memory_format_t format) {
	if (mkl_dnn::impl::any_null(memory_desc, tensor))
		return mkl_dnn_invalid_arguments;
    mkl_dnn_memory_desc_t md = {
        .tensor_desc = *tensor,
        .format = format };

    mkl_dnn_status_t status = mkl_dnn_success;
    switch (format) {
	case mkl_dnn_any_f32:
        break;
    /* semidefined blocked format */
    case mkl_dnn_n_f32:
    case mkl_dnn_nchw_f32:
    case mkl_dnn_nhwc_f32:
        status = mkl_dnn::impl::types::compute_blocking(md);
        break;
    /* no enough information */
    case mkl_dnn_blocked_f32:
    default:
        return mkl_dnn_invalid_arguments;
    }

    if (status == mkl_dnn_success)
        *memory_desc = md;
    return status;
}

mkl_dnn_status_t mkl_dnn_memory_primitive_desc_init(
        mkl_dnn_memory_primitive_desc_t *memory_primitive_desc,
        const mkl_dnn_memory_desc_t *memory_desc,
        const_mkl_dnn_engine_t engine) {
    if (mkl_dnn::impl::any_null(memory_primitive_desc, memory_desc, engine))
        return mkl_dnn_invalid_arguments;
    if (memory_desc->format == mkl_dnn_any_f32 || !engine->is_ok())
        return mkl_dnn_invalid_arguments;

    for (auto init = engine->get_memory_inits(); *init; ++init) {
        using mkl_dnn::impl::const_op_desc_t;
        mkl_dnn_status_t status = (*init)(
                reinterpret_cast<mkl_dnn_primitive_desc_t*>(memory_primitive_desc),
                static_cast<const_op_desc_t>(memory_desc), *engine);
        if (status == mkl_dnn_success)
            return mkl_dnn_success;
    }

    return mkl_dnn_unimplemented;
}

int mkl_dnn_memory_primitive_desc_equal(const mkl_dnn_memory_primitive_desc_t *lhs,
        const mkl_dnn_memory_primitive_desc_t *rhs) {
    return mkl_dnn::impl::types::operator==(*lhs, *rhs);
}

mkl_dnn_status_t mkl_dnn_memory_create(mkl_dnn_primitive_t *memory,
        const mkl_dnn_memory_primitive_desc_t *memory_primitive_desc,
        const void *data_ptr) {
    // XXX: is this ok?
    mkl_dnn_primitive_at_t inputs[] = {
        { static_cast<const_mkl_dnn_primitive_t>(data_ptr), 0 } };
    const_mkl_dnn_primitive_t outputs[] = {
        static_cast<const_mkl_dnn_primitive_t>(data_ptr) };
    return mkl_dnn_primitive_create(memory,
            memory_primitive_desc, inputs, outputs);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0
