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

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::precision;

status_t mkldnn_tensor_desc_init(tensor_desc_t *tensor_desc, uint32_t ndims,
        const dims_t dims) {
    if (any_null(tensor_desc)) return invalid_arguments;
    tensor_desc_t td;
    td.ndims= ndims;
    if (ndims == 0) {
        array_set(td.dims, 0, TENSOR_MAX_DIMS);
    } else {
        if (any_null(dims) || ndims > TENSOR_MAX_DIMS)
            return invalid_arguments;
        array_copy(td.dims, dims, ndims);
    }
    *tensor_desc = td;
    return success;
}

status_t mkldnn_memory_desc_init(memory_desc_t *memory_desc,
        const tensor_desc_t *tensor, precision_t precision,
        memory_format_t format) {
    /* quick return for memory desc == 0 */
    if (tensor == nullptr || tensor->ndims == 0) {
        *memory_desc = types::zero<memory_desc_t>();
        return success;
    }

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc) && one_of(precision, f32, u32);
    if (!args_ok)
        return invalid_arguments;

    memory_desc_t md;
    md.tensor_desc = *tensor;
    md.precision = precision;
    md.format = format;

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
    case Ohwi8o:
    case goihw:
    case gOIhw8i8o:
        status = memory_desc_wrapper::compute_blocking(md);
        break;
    /* no enough information */
    case memory_format::undef:
    case blocked:
    default:
        return invalid_arguments;
    }

    if (status == success)
        *memory_desc = md;
    return status;
}

size_t mkldnn_memory_desc_get_number_of_elements(
        const memory_desc_t *memory_desc, int with_padding) {
    return memory_desc
        ? memory_desc_wrapper(*memory_desc).nelems(with_padding)
        : 0;
}

size_t mkldnn_memory_desc_get_size(const memory_desc_t *memory_desc) {
    return memory_desc ? memory_desc_wrapper(*memory_desc).size() : 0;
}

status_t mkldnn_memory_primitive_desc_init(
        memory_primitive_desc_t *memory_primitive_desc,
        const memory_desc_t *memory_desc, const engine *engine) {
    /* quick return for memory primitive descriptor == 0 */
    if (memory_desc == nullptr || memory_desc_wrapper(*memory_desc).is_zero()) {
        *memory_primitive_desc = types::zero<memory_primitive_desc_t>();
        memory_primitive_desc->base.primitive_kind = primitive_kind::memory;
        return success;
    }

    if (any_null(memory_primitive_desc, memory_desc, engine))
        return invalid_arguments;
    if (memory_desc->format == any || !engine->is_ok())
        return invalid_arguments;

    return primitive_desc_init(
            primitive_desc_t::convert_from_c(memory_primitive_desc),
            *memory_desc, *engine);
}

int mkldnn_memory_primitive_desc_equal(const memory_primitive_desc_t *lhs,
        const memory_primitive_desc_t *rhs) {
    if (any_null(lhs, rhs)) return 0;
    return types::operator==(*lhs, *rhs);
}

status_t mkldnn_memory_create(primitive **memory,
        const memory_primitive_desc_t *memory_primitive_desc,
        void *data_ptr) {
    // XXX: is this ok?
    primitive_at_t inputs[] = { {static_cast<const primitive*>(data_ptr), 0} };
    const primitive *outputs[] = {static_cast<const primitive*>(data_ptr)};
    return mkldnn_primitive_create(memory, memory_primitive_desc, inputs,
            outputs);
}

status_t mkldnn_memory_get_primitive_desc(const primitive *memory,
        memory_primitive_desc_t *memory_primitive_desc) {
    if (any_null(memory_primitive_desc))
        return invalid_arguments;

    if (memory == nullptr) {
        *memory_primitive_desc = types::zero<memory_primitive_desc_t>();
        return success;
    }

    if (memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    *memory_primitive_desc = memory->primitive_desc().memory;
    return success;
}

status_t mkldnn_memory_get_data_handle(const primitive *memory, void **handle)
{
    if (any_null(handle))
        return invalid_arguments;

    if (memory == nullptr) {
        *handle = nullptr;
        return success;
    }

    if (memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    *handle = static_cast<void*>(memory->memory());
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
