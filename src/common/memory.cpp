/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include "memory_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::data_type;

namespace {
bool memory_desc_sanity_check(int ndims,const dims_t dims,
        data_type_t data_type, memory_format_t format) {
    if (ndims == 0) return true;

    bool ok = true
        && dims != nullptr
        && 0 < ndims && ndims <= TENSOR_MAX_DIMS
        && one_of(data_type, f32, s32, s16, s8, u8)
        && format != memory_format::undef;
    if (!ok) return false;
    for (int d = 0; d < ndims; ++d)
        if (dims[d] < 0) return false;

    return true;
}

bool memory_desc_sanity_check(const memory_desc_t *md) {
    if (md == nullptr) return false;
    return memory_desc_sanity_check(md->ndims, md->dims, md->data_type,
            md->format);
}
}

status_t mkldnn_memory_desc_init(memory_desc_t *memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, memory_format_t format) {
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0 || format == memory_format::undef) {
        *memory_desc = types::zero_md();
        return success;
    }

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
        && memory_desc_sanity_check(ndims, dims, data_type, format);
    if (!args_ok) return invalid_arguments;

    memory_desc_t md;
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.primitive_kind = primitive_kind::memory_primitive_kind;
    md.data_type = data_type;
    md.format = format;

    status_t status = success;
    if (one_of(format, memory_format::undef, blocked, wino_fmt, rnn_packed)) {
        status = invalid_arguments;
    } else if (format == any) {
        // nop
    } else if (types::format_normalize(format) == blocked) {
        status = memory_desc_wrapper::compute_blocking(md);
    } else {
        assert(!"unreachable");
        status = invalid_arguments;
    }

    if (status == success)
        *memory_desc = md;

    return status;
}

status_t mkldnn_memory_desc_init_submemory(memory_desc_t *md,
        const memory_desc_t *parent_md, const dims_t dims,
        const dims_t offsets) {
    if (any_null(md, parent_md) || !memory_desc_sanity_check(parent_md))
        return invalid_arguments;

    const memory_desc_t &src_d = *parent_md;
    const auto &src_d_blk = src_d.layout_desc.blocking;
    const int ndims = src_d.ndims;

    for (int d = 0; d < ndims; ++d) {
        if (dims[d] < 0 || offsets[d] < 0
                || (offsets[d] + dims[d] > src_d.dims[d]))
            return invalid_arguments;
    }

    if (one_of(src_d.format, memory_format::undef, blocked, wino_fmt))
        return unimplemented;

    memory_desc_t dst_d = *parent_md;
    auto &dst_d_blk = dst_d.layout_desc.blocking;

    /* TODO: put this into memory_desc_wrapper */
    for (int d = 0; d < ndims; ++d) {
        /* very limited functionality for now */
        const bool ok = true
            && offsets[d] % src_d_blk.block_dims[d] == 0 /* [r1] */
            && src_d_blk.offset_padding_to_data[d] == 0
            && (false
                    || dims[d] % src_d_blk.block_dims[d] == 0
                    || dims[d] < src_d_blk.block_dims[d]);
        if (!ok)
            return unimplemented;

        const bool is_right_border = offsets[d] + dims[d] == src_d.dims[d];

        dst_d.dims[d] = dims[d];
        dst_d_blk.padding_dims[d] = is_right_border
            ? src_d_blk.padding_dims[d] - offsets[d] : dst_d.dims[d];
        dst_d_blk.offset_padding_to_data[d] =
            src_d_blk.offset_padding_to_data[d];
        dst_d_blk.offset_padding += /* [r1] */
            offsets[d] / src_d_blk.block_dims[d] * dst_d_blk.strides[0][d];
    }

    *md = dst_d;

    return success;
}

status_t mkldnn_memory_primitive_desc_create(primitive_desc_t **memory_pd,
        const memory_desc_t *memory_desc, engine_t *engine) {
    bool args_ok = !any_null(memory_pd, memory_desc, engine)
        && memory_desc_sanity_check(memory_desc)
        && memory_desc_wrapper(*memory_desc).is_defined();
    if (!args_ok) return invalid_arguments;
    return engine->memory_primitive_desc_create(
            (memory_pd_t**)memory_pd, memory_desc);
}

int mkldnn_memory_primitive_desc_equal(const primitive_desc_t *lhs,
        const primitive_desc_t *rhs) {
    bool args_ok = !any_null(lhs, rhs)
        && lhs->engine() == rhs->engine()
        && one_of(lhs->kind(), primitive_kind::memory_primitive_kind)
        && one_of(rhs->kind(), primitive_kind::memory_primitive_kind);
    if (!args_ok) return 0;
    auto l = (const memory_pd_t *)lhs;
    auto r = (const memory_pd_t *)rhs;

    return l->is_equal(r);
}

size_t mkldnn_memory_primitive_desc_get_size(const primitive_desc_t *memory_pd)
{
    bool args_ok = !any_null(memory_pd)
        && memory_pd->kind() == primitive_kind::memory_primitive_kind;
    if (!args_ok) return 0;
    return ((memory_pd_t*)memory_pd)->get_size();
}

status_t mkldnn_memory_create(memory_t **memory,
        const primitive_desc_t *memory_pd, void *handle) {
    if (any_null(memory, memory_pd)) return invalid_arguments;
    return memory_pd->create_memory(memory);
}

status_t mkldnn_memory_get_primitive_desc(const memory_t *memory,
        const primitive_desc_t **memory_pd) {
    if (any_null(memory, memory_pd)) return invalid_arguments;
    *memory_pd = memory->pd();
    return success;
}

status_t mkldnn_memory_get_data_handle(const memory_t *memory,
        void **handle) {
    if (any_null(handle))
        return invalid_arguments;
    if (memory == nullptr) {
        *handle = nullptr;
        return success;
    }
    return memory->get_data_handle(handle);
}

status_t mkldnn_memory_set_data_handle(memory_t *memory, void *handle) {
    if (any_null(memory)) return invalid_arguments;
    return memory->set_data_handle(handle);
}

status_t mkldnn_memory_destroy(memory_t *memory) {
    delete memory;
    return success;
}

status_t mkldnn_concat_primitive_desc_create(primitive_desc_t **concat_pd,
        const memory_desc_t *output_d, int n, int concat_dim,
        const primitive_desc_t **input_pds, const primitive_attr_t *attr) {
    bool args_ok = !any_null(concat_pd, input_pds) && n > 0;
    if (!args_ok) return invalid_arguments;
    for (int i = 0; i < n; ++i) {
        if (input_pds[i] == nullptr ||
                input_pds[i]->kind() != primitive_kind::memory_primitive_kind)
            return invalid_arguments;
    }

    const primitive_attr_t dummy_attr;
    if (attr == NULL)
        attr = &dummy_attr;

    auto i_mpds = (const memory_pd_t **)input_pds;
    engine_t *engine = i_mpds[0]->engine();
    const int ndims = i_mpds[0]->desc()->ndims;
    const dims_t &dims = i_mpds[0]->desc()->dims;
    const data_type_t dt = i_mpds[0]->desc()->data_type;

    int concat_dim_sz = dims[concat_dim];
    for (int i = 1; i < n; ++i) {
        if (i_mpds[i]->engine() != engine) return invalid_arguments;
        if (i_mpds[i]->desc()->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (d == concat_dim) continue;
            if (i_mpds[i]->desc()->dims[d] != dims[d])
                return invalid_arguments;
        }
        if (i_mpds[i]->desc()->data_type != dt) return invalid_arguments;
        concat_dim_sz += i_mpds[i]->desc()->dims[concat_dim];
    }

    memory_desc_t dummy_output_d;
    if (output_d) {
        if (output_d->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (output_d->dims[d] !=
                    (d == concat_dim ? concat_dim_sz : dims[d]))
                return invalid_arguments;
        }
    } else {
        dummy_output_d = *i_mpds[0]->desc();
        dummy_output_d.dims[concat_dim] = concat_dim_sz;
        dummy_output_d.format = memory_format::any;
        output_d = &dummy_output_d;
    }

    auto c_pd = reinterpret_cast<concat_pd_t **>(concat_pd);

    for (auto c = engine->get_concat_implementation_list(); *c; ++c) {
        if ((*c)(c_pd, output_d, n, concat_dim, i_mpds, attr) == success) {
            (*c_pd)->init_info();
            return success;
        }
    }
    return unimplemented;
}

status_t mkldnn_sum_primitive_desc_create(primitive_desc_t **sum_pd,
        const memory_desc_t *output_d, int n, const float *scales,
        const primitive_desc_t **input_pds, const primitive_attr_t *attr) {
    bool args_ok = !any_null(sum_pd, input_pds, scales) && n > 0;
    if (!args_ok) return invalid_arguments;
    for (int i = 0; i < n; ++i) {
        if (input_pds[i] == nullptr ||
                input_pds[i]->kind() != primitive_kind::memory_primitive_kind)
            return invalid_arguments;
    }

    const primitive_attr_t dummy_attr;
    if (attr == NULL)
        attr = &dummy_attr;

    auto i_mpds = (const memory_pd_t **)input_pds;
    engine_t *engine = i_mpds[0]->engine();
    const int ndims = i_mpds[0]->desc()->ndims;
    const dims_t &dims = i_mpds[0]->desc()->dims;
    const data_type_t dt = i_mpds[0]->desc()->data_type;

    for (int i = 1; i < n; ++i) {
        if (i_mpds[i]->engine() != engine) return invalid_arguments;
        if (i_mpds[i]->desc()->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (i_mpds[i]->desc()->dims[d] != dims[d])
                return invalid_arguments;
        }
        if (i_mpds[i]->desc()->data_type != dt) return invalid_arguments;
    }

    memory_desc_t dummy_output_d;
    if (output_d) {
        if (output_d->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (output_d->dims[d] != dims[d])
                return invalid_arguments;
        }
    } else {
        dummy_output_d = *i_mpds[0]->desc();
        dummy_output_d.format = memory_format::any;
        output_d = &dummy_output_d;
    }

    auto s_pd = reinterpret_cast<sum_pd_t **>(sum_pd);

    for (auto s = engine->get_sum_implementation_list(); *s; ++s) {
        if ((*s)(s_pd, output_d, n, scales, i_mpds, attr) == success) {
            (*s_pd)->init_info();
            return success;
        }
    }
    return unimplemented;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
