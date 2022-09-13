/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl.hpp"

#include "common/c_types_map.hpp"

using namespace dnnl::impl;

status_t dnnl_memory_desc_query(
        const memory_desc_t *md, query_t what, void *result) {
    const bool is_blocked = md->format_kind == format_kind::blocked;

    switch (what) {
        case query::ndims_s32: *(int32_t *)result = md->ndims; break;
        case query::dims: *(const dims_t **)result = &md->dims; break;
        case query::data_type: *(data_type_t *)result = md->data_type; break;
        case query::submemory_offset_s64: *(dim_t *)result = md->offset0; break;
        case query::padded_dims:
            *(const dims_t **)result = &md->padded_dims;
            break;
        case query::padded_offsets:
            *(const dims_t **)result = &md->padded_offsets;
            break;
        case query::format_kind:
            *(format_kind_t *)result = md->format_kind;
            break;
        case query::strides:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.strides;
            break;
        case query::inner_nblks_s32:
            if (!is_blocked) return status::invalid_arguments;
            *(int32_t *)result = md->format_desc.blocking.inner_nblks;
            break;
        case query::inner_blks:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.inner_blks;
            break;
        case query::inner_idxs:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.inner_idxs;
            break;
        default: return status::unimplemented;
    }
    return status::success;
}
