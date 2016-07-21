#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "mkl_dnn.h"

#include "engine.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"

status_t tensor_desc_init(tensor_desc_t *tensor_desc, uint32_t ndims_batch,
        uint32_t ndims_channels, uint32_t ndims_spatial, const dims_t dims) {
    if (mkl_dnn::impl::any_null(tensor_desc, dims)) return invalid_arguments;
    tensor_desc_t td = {
        .ndims_batch = ndims_batch,
        .ndims_channels = ndims_channels,
        .ndims_spatial = ndims_spatial};
    const uint32_t ndims = mkl_dnn::impl::types::ndims(td);
    if (ndims > TENSOR_MAX_DIMS) return invalid_arguments;
    mkl_dnn::impl::array_copy(td.dims, dims, ndims);
    *tensor_desc = td;
    return success;
}

status_t memory_desc_init(memory_desc_t *memory_desc,
        const tensor_desc_t *tensor, memory_format_t format) {
    if (mkl_dnn::impl::any_null(memory_desc, tensor)) return invalid_arguments;
    memory_desc_t md = {
        .tensor_desc = *tensor,
        .format = format };

    status_t status = success;
    switch (format) {
    case memory_format_any:
        break;
    /* semidefined blocked format */
    case memory_format_n_f32:
    case memory_format_nchw_f32:
    case memory_format_nhwc_f32:
        status = mkl_dnn::impl::types::compute_blocking(md);
        break;
    /* no enough information */
    case memory_format_blocked_f32:
    default:
        return invalid_arguments;
    }

    if (status == success) *memory_desc = md;
    return status;
}

status_t memory_primitive_desc_init(
        memory_primitive_desc_t *memory_primitive_desc,
        const memory_desc_t *memory_desc, const_dnn_engine_t engine) {
    if (mkl_dnn::impl::any_null(memory_primitive_desc, memory_desc, engine))
        return invalid_arguments;
    if (memory_desc->format == memory_format_any || !engine->is_ok())
        return invalid_arguments;

    for (auto init = engine->get_memory_inits(); *init; ++init) {
        using mkl_dnn::impl::const_op_desc_t;
        status_t status = (*init)(
                reinterpret_cast<primitive_desc_t*>(memory_primitive_desc),
                static_cast<const_op_desc_t>(memory_desc), *engine);
        if (status == success) return success;
    }

    return unimplemented;
}

int memory_primitive_desc_equal(const memory_primitive_desc_t *lhs,
        const memory_primitive_desc_t *rhs) {
    return mkl_dnn::impl::types::operator==(*lhs, *rhs);
}

status_t memory_create(dnn_primitive_t *memory,
        const memory_primitive_desc_t *memory_primitive_desc,
        const void *data_ptr) {
    // XXX: is this ok?
    dnn_primitive_at_t inputs[] = {
        { static_cast<const_dnn_primitive_t>(data_ptr), 0 } };
    const_dnn_primitive_t outputs[] = {
        static_cast<const_dnn_primitive_t>(data_ptr) };
    return primitive_create(memory, memory_primitive_desc, inputs, outputs);
}
