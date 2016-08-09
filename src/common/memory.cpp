#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "mkl_dnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::memory_format;
using namespace mkl_dnn::impl::precision;

status_t mkl_dnn_tensor_desc_init(tensor_desc_t *tensor_desc,
        uint32_t ndims_batch, uint32_t ndims_channels, uint32_t ndims_spatial,
        const dims_t dims) {
    if (any_null(tensor_desc, dims))
        return invalid_arguments;
    tensor_desc_t td = {
        .ndims_batch = ndims_batch,
        .ndims_channels = ndims_channels,
        .ndims_spatial = ndims_spatial};
    const uint32_t ndims = types::ndims(td);
    if (ndims > TENSOR_MAX_DIMS)
        return invalid_arguments;
    array_copy(td.dims, dims, ndims);
    *tensor_desc = td;
    return success;
}

status_t mkl_dnn_memory_desc_init(memory_desc_t *memory_desc,
        const tensor_desc_t *tensor, precision_t precision,
        memory_format_t format) {
    bool args_ok = !any_null(memory_desc, tensor)
        && one_of(precision, f32, u32);
    if (!args_ok)
        return invalid_arguments;

    memory_desc_t md = {
        .tensor_desc = *tensor,
        .precision = precision,
        .format = format };

    status_t status = success;
    switch (format) {
    case any:
        break;
    /* semidefined blocked format */
    case x:
    case nc:
    case nchw:
    case nhwc:
    case nChw8c:
    case oi:
    case oihw:
    case OIhw8i8o:
    case goihw:
    case gOIhw8i8o:
        status = memory_desc_wrapper::compute_blocking(md);
        break;
    /* no enough information */
    case blocked:
    default:
        return invalid_arguments;
    }

    if (status == success)
        *memory_desc = md;
    return status;
}

size_t mkl_dnn_memory_desc_get_number_of_elements(
        const memory_desc_t *memory_desc, int with_padding) {
    return memory_desc_wrapper(*memory_desc).nelems(with_padding);
}

size_t mkl_dnn_memory_desc_get_size(const memory_desc_t *memory_desc) {
    return memory_desc_wrapper(*memory_desc).size();
}

status_t mkl_dnn_memory_primitive_desc_init(
        memory_primitive_desc_t *memory_primitive_desc,
        const memory_desc_t *memory_desc, const engine *engine) {
    if (any_null(memory_primitive_desc, memory_desc, engine))
        return invalid_arguments;
    if (memory_desc->format == any || !engine->is_ok())
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(memory_primitive_desc),
            *memory_desc, *engine);
}

int mkl_dnn_memory_primitive_desc_equal(const memory_primitive_desc_t *lhs,
        const memory_primitive_desc_t *rhs) {
    return types::operator==(*lhs, *rhs);
}

status_t mkl_dnn_memory_create(primitive **memory,
        const memory_primitive_desc_t *memory_primitive_desc,
        void *data_ptr) {
    // XXX: is this ok?
    primitive_at_t inputs[] = { {static_cast<const primitive*>(data_ptr), 0} };
    const primitive *outputs[] = {static_cast<const primitive*>(data_ptr)};
    return mkl_dnn_primitive_create(memory, memory_primitive_desc, inputs,
            outputs);
}

status_t mkl_dnn_memory_get_primitive_desc(const primitive *memory,
        memory_primitive_desc_t *memory_primitive_desc) {
    if (any_null(memory, memory_primitive_desc)
            || memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    *memory_primitive_desc = memory->primitive_desc().memory;
    return success;
}

status_t mkl_dnn_memory_get_data_handle(const primitive *memory, void **handle)
{
    if (memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    *handle = static_cast<void*>(memory->memory());
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
